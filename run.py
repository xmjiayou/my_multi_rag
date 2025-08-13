# -*- coding: utf-8 -*-
# rag_from_page_chunks_stable.py
# 使用说明（一步步）
# 运行新分块器（页内二次切分）

# bash
# 复制
# 编辑
# python chunker.py --in ./all_pdf_page_chunks.json --out ./all_pdf_page_subchunks.json --target-tokens 480 --overlap 60
# # 然后把主脚本里使用的 chunk 路径指向这个新文件（或直接改名覆盖原文件）
# 替换路由
# 把主脚本里的
# from router import route_query
# 改为：
# from router_enhanced import route_query, extract_filters, apply_filters
# 并在视觉重排后加：

# python
# 复制
# 编辑
# candidates = apply_filters(candidates[:10], extract_filters(question))
# 正常跑主脚本（之前的流程不变）

# 评估

# bash
# 复制
# 编辑
# python eval_report.py \
#   --gt ./datas/多模态RAG图文问答挑战赛训练集.json \
#   --pred-jsonl ./rag_results_raw.jsonl
# 输出 eval_report.md + eval_details.csv
import json
import os
import time
import threading
import socket
import httpx
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
import concurrent.futures
import random
import pandas as pd

from dotenv import load_dotenv
from openai import OpenAI, APITimeoutError

# 你现有的
from get_text_embedding import get_text_embedding
from extract_json_array import extract_json_array

# 新增模块（同目录）
from hybrid_search import HybridSearcher
from vision_rerank import VisionReranker
from router_enhanced import route_query, extract_filters, apply_filters
from chart_extract import extract_tables_from_page_markdown, chart_image_to_csv

# 统一加载项目根目录的 .env
load_dotenv()

# ========= 可配置的输出文件 =========
RESULTS_JSONL = "./rag_results_raw.jsonl"        # 逐条写入，支持续跑
RAW_SNAPSHOT_JSON = "./rag_top1_pred_raw.json"   # 快照（从 JSONL 汇总生成）
FINAL_PRED_JSON = "./rag_top1_pred.json"         # 最终提交结果

# ========= 并发 / 限流 / 超时 =========
OPENAI_RPM_LIMIT = int(os.getenv("OPENAI_RPM_LIMIT", "60"))
REQUEST_TIMEOUT = float(os.getenv("OPENAI_REQUEST_TIMEOUT", "75"))
MAX_WORKERS = int(os.getenv("RAG_MAX_WORKERS", "3"))
FAST_MODE = os.getenv("FAST_MODE", "0") == "1"

# ========= 写文件锁（多线程安全）=========
write_lock = threading.Lock()


class PageChunkLoader:
    def __init__(self, json_path: str):
        self.json_path = json_path

    def load_chunks(self) -> List[Dict[str, Any]]:
        with open(self.json_path, 'r', encoding='utf-8') as f:
            return json.load(f)


class PageFullLoader:
    """加载页级文件，提供 (file_name, page) -> full_text 映射，用于回溯整页"""
    def __init__(self, page_json_path: str):
        self.page_json_path = page_json_path

    def load_full_map(self) -> Dict[Tuple[str, int], str]:
        mp: Dict[Tuple[str, int], str] = {}
        if not os.path.exists(self.page_json_path):
            return mp
        with open(self.page_json_path, 'r', encoding='utf-8') as f:
            pages = json.load(f)
        for p in pages:
            try:
                fn = p["metadata"]["file_name"]
                pg = int(p["metadata"]["page"])
                mp[(fn, pg)] = p.get("content", "") or ""
            except Exception:
                continue
        return mp


class EmbeddingModel:
    def __init__(self, batch_size: int = 64):
        self.api_key = os.getenv('LOCAL_API_KEY')
        self.base_url = os.getenv('LOCAL_BASE_URL')
        self.embedding_model = os.getenv('LOCAL_EMBEDDING_MODEL')
        self.batch_size = batch_size
        if not self.api_key or not self.base_url:
            raise ValueError('请在 .env 中配置 LOCAL_API_KEY 和 LOCAL_BASE_URL')

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return get_text_embedding(
            texts,
            api_key=self.api_key,
            base_url=self.base_url,
            embedding_model=self.embedding_model,
            batch_size=self.batch_size
        )

    def embed_text(self, text: str) -> List[float]:
        return self.embed_texts([text])[0]


class SimpleVectorStore:
    def __init__(self):
        self.embeddings = []
        self.chunks = []  # 这里可以是“页内子分片”或“整页”，由输入文件决定

    def add_chunks(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]):
        self.chunks.extend(chunks)
        self.embeddings.extend(embeddings)

    def search(self, query_embedding: List[float], top_k: int = 3) -> List[Dict[str, Any]]:
        from numpy.linalg import norm
        import numpy as np
        if not self.embeddings:
            return []
        emb_matrix = np.array(self.embeddings)
        query_emb = np.array(query_embedding)
        sims = emb_matrix @ query_emb / (norm(emb_matrix, axis=1) * norm(query_emb) + 1e-8)
        idxs = sims.argsort()[::-1][:top_k]
        return [self.chunks[i] for i in idxs]


# ======== 限流 / 重试 / 裁剪工具 ========

class RequestRateLimiter:
    """按请求数/分钟限速；不能精准对齐 TPM，但能显著缓解 429。"""
    def __init__(self, rpm: int):
        self.capacity = max(1, rpm)
        self.timestamps = []
        self.lock = threading.Lock()

    def acquire(self):
        while True:
            with self.lock:
                now = time.time()
                self.timestamps = [t for t in self.timestamps if now - t < 60.0]
                if len(self.timestamps) < self.capacity:
                    self.timestamps.append(now)
                    return
                wait = 60.0 - (now - self.timestamps[0])
            time.sleep(max(0.05, wait))

rate_limiter = RequestRateLimiter(OPENAI_RPM_LIMIT)


def with_retry(callable_fn, *args, base_backoff=1.0, max_backoff=20.0, max_tries=6, **kwargs):
    """
    对 429/TPM、网络抖动、读超时等进行指数退避 + 抖动。
    """
    attempt = 0
    backoff = base_backoff
    while True:
        attempt += 1
        try:
            return callable_fn(*args, **kwargs)
        except (APITimeoutError, httpx.ReadTimeout, httpx.ConnectTimeout, socket.timeout):
            if attempt >= max_tries:
                raise
            sleep_s = min(max_backoff, backoff) + random.uniform(0, 0.4)
            time.sleep(sleep_s)
            backoff = min(max_backoff, backoff * 1.8)
        except Exception as e:
            msg = str(e)
            if ("429" in msg or "RateLimit" in msg or "TPM" in msg) and attempt < max_tries:
                sleep_s = min(max_backoff, backoff) + random.uniform(0, 0.5)
                time.sleep(sleep_s)
                backoff = min(max_backoff, backoff * 2)
                continue
            raise


def truncate_text(s: str, max_chars: int = 1200) -> str:
    s = (s or "").strip()
    return s if len(s) <= max_chars else s[:max_chars] + "…"


class SimpleRAG:
    def __init__(self, chunk_json_path: str, page_json_path: str = None, batch_size: int = 32):
        self.loader = PageChunkLoader(chunk_json_path)
        self.embedding_model = EmbeddingModel(batch_size=batch_size)
        self.vector_store = SimpleVectorStore()
        self.hybrid = None
        self.vis_reranker = None
        self.page_full_map: Dict[Tuple[str, int], str] = {}
        self.page_json_path = page_json_path

    def setup(self):
        print("加载 chunk（子分片或整页）...")
        chunks = self.loader.load_chunks()
        print(f"共加载 {len(chunks)} 条")
        print("生成嵌入...")
        embeddings = self.embedding_model.embed_texts([c['content'] for c in chunks])
        print("存储向量...")
        self.vector_store.add_chunks(chunks, embeddings)
        print("RAG 向量库构建完成！")

        # 页级内容映射（用于回溯整页上下文）
        if not self.page_json_path:
            self.page_json_path = os.path.join(os.path.dirname(__file__), "all_pdf_page_chunks.json")
        self.page_full_map = PageFullLoader(self.page_json_path).load_full_map()
        if self.page_full_map:
            print(f"已载入页级内容：{len(self.page_full_map)} 页")
        else:
            print("警告：未找到 all_pdf_page_chunks.json，整页回溯将退化为子分片内容")

        # === Hybrid 接入 ===
        def text_search_fn(q, top_k=50):
            q_emb = self.embedding_model.embed_text(q)
            return self.vector_store.search(q_emb, top_k)

        self.hybrid = HybridSearcher(self.vector_store.chunks, text_search_fn, vision_index_dir="./vision_index")

        # 视觉重排（CLIP），失败则降级为直通
        try:
            self.vis_reranker = VisionReranker(index_dir="./vision_index", device="cuda")
        except Exception as e:
            print(f"视觉重排初始化失败，降级直通：{e}")
            class NoOpRerank:
                def rerank(self, q, cands, top_k=10): return cands[:top_k]
            self.vis_reranker = NoOpRerank()

    def query(self, question: str, top_k: int = 3) -> Dict[str, Any]:
        q_emb = self.embedding_model.embed_text(question)
        results = self.vector_store.search(q_emb, top_k)
        return {"question": question, "chunks": results}

    def generate_answer(self, question: str, top_k: int = 3) -> Dict[str, Any]:
        """
        检索 + 大模型生成式回答，返回结构化结果
        """
        qwen_api_key = os.getenv('LOCAL_API_KEY')
        qwen_base_url = os.getenv('LOCAL_BASE_URL')
        qwen_model = os.getenv('LOCAL_TEXT_MODEL')
        if not qwen_api_key or not qwen_base_url or not qwen_model:
            raise ValueError('请在 .env 中配置 LOCAL_API_KEY、LOCAL_BASE_URL、LOCAL_TEXT_MODEL')

        # 快速模式：进一步压缩
        if FAST_MODE:
            MAX_TOTAL = 1500
            def trunc_fast(x):
                x = (x or "").strip()
                return x[:500] + ("…" if len(x) > 500 else "")
            truncate_fn = trunc_fast
            gen_max_tokens = 256
            temperature = 0.0
        else:
            MAX_TOTAL = 3500
            truncate_fn = truncate_text
            gen_max_tokens = 512
            temperature = 0.1

        # ===== Router：判题型 =====
        qtype = route_query(question)

        # ===== Hybrid 召回 =====
        cand = self.hybrid.search_candidates(question, top_k_text=50, top_k_bm25=100, final_k=80)
        # 初选（文本相关性 Top-20）→ 视觉重排 → 合并
        prim = cand[:20]
        prim = self.vis_reranker.rerank(question, prim, top_k=10)
        others = [c for c in cand if c["id"] not in {x["id"] for x in prim}]
        candidates = prim + others
        # 结合公司/年份/季度进行加权重排（你上传思路）
        candidates = apply_filters(candidates[:10], extract_filters(question))

        # ===== 组装上下文（按题型做“表格/图表增强”） =====
        parts = []
        for c in candidates:
            fn = c['file_name']; pg = int(c['page'])
            header = f"[文件名]{fn} [页码]{pg}"
            # 回溯整页：优先用页级全文，其次回退子分片
            page_text = self.page_full_map.get((fn, pg), c.get("content","") or "")

            ctx = page_text
            if qtype in ("table", "chart"):
                # 1) 表格抽取（HTML <table> → DataFrame → CSV摘录）
                try:
                    dfs = extract_tables_from_page_markdown(page_text)
                except Exception:
                    dfs = []
                if dfs:
                    for df in dfs[:2]:
                        df = df.fillna("").astype(str)
                        pri = df[df.apply(lambda row: row.astype(str).str.contains(
                            r"(合计|总计|同比|环比|增长|率|%|亿元|万元|收入|利润)", regex=True).any(), axis=1)]
                        sub = (pd.concat([pri.head(6), df.head(4)])
                               if not pri.empty else df.head(8)).drop_duplicates().head(8)
                        ctx = f"{ctx}\n[表格摘录]\n{sub.to_csv(index=False)}"
                        break
                # 2) 图表结构化（DePlot）：若有页图，转 CSV
                if c.get("image_path"):
                    try:
                        csv_text = chart_image_to_csv(c["image_path"])
                    except Exception:
                        csv_text = None
                    if csv_text:
                        ctx = f"{ctx}\n[图表CSV]\n{csv_text}"

            parts.append(f"{header}\n{truncate_fn(ctx)}")

        # 限制总长度
        total = 0
        final_parts = []
        for p in parts:
            if total + len(p) > MAX_TOTAL:
                break
            final_parts.append(p)
            total += len(p)
        context = "\n\n".join(final_parts)

        # ===== Prompt 与 LLM 调用（严格 JSON 输出） =====
        prompt = (
            f"你是一名专业的金融分析助手，请严格依据下方提供的证据作答，不得使用外部知识。\n"
            f"请用 JSON 输出，字段固定为：{{\"answer\": \"简洁回答\", \"filename\": \"来源文件名\", \"page\": \"来源页码\"}}\n"
            f"若证据不足以确定，请输出：{{\"answer\": \"根据提供信息无法确定\", \"filename\": \"最相关的来源文件名\", \"page\": \"页码\"}}\n"
            f"证据（可能包含页面文本、表格摘录、图表CSV）：\n{context}\n\n"
            f"问题：{question}\n"
            f"只输出 JSON，不要多余文字。"
        )

        client = OpenAI(
            api_key=qwen_api_key,
            base_url=qwen_base_url,
            timeout=REQUEST_TIMEOUT,
            max_retries=0
        )

        def _do_call():
            rate_limiter.acquire()
            return client.chat.completions.create(
                model=qwen_model,
                messages=[
                    {"role": "system", "content": "你是一名专业的金融分析助手。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=gen_max_tokens,
                timeout=REQUEST_TIMEOUT
            )

        completion = with_retry(_do_call, base_backoff=1.0, max_backoff=20.0)
        raw = completion.choices[0].message.content.strip()
        json_str = extract_json_array(raw, mode='objects')

        if json_str:
            try:
                arr = json.loads(json_str)
                if isinstance(arr, list) and arr:
                    j = arr[0]
                    answer = j.get('answer', '')
                    filename = j.get('filename', '')
                    page = j.get('page', '')
                else:
                    answer = raw
                    filename = candidates[0]['file_name'] if candidates else ''
                    page = candidates[0]['page'] if candidates else ''
            except Exception:
                answer = raw
                filename = candidates[0]['file_name'] if candidates else ''
                page = candidates[0]['page'] if candidates else ''
        else:
            answer = raw
            filename = candidates[0]['file_name'] if candidates else ''
            page = candidates[0]['page'] if candidates else ''

        return {
            "question": question,
            "answer": answer,
            "filename": filename,
            "page": page,
            "retrieval_chunks": candidates
        }


# ========= 续跑：读写工具 =========

def load_done_map(jsonl_path: str) -> Dict[int, Dict[str, Any]]:
    done = {}
    if not os.path.exists(jsonl_path):
        return done
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                idx = obj.get("idx")
                if isinstance(idx, int):
                    done[idx] = obj  # 后写覆盖先写
            except Exception:
                continue
    return done


def append_jsonl_threadsafe(jsonl_path: str, obj: Dict[str, Any]):
    data = json.dumps(obj, ensure_ascii=False)
    with write_lock:
        with open(jsonl_path, 'a', encoding='utf-8') as f:
            f.write(data + "\n")
            f.flush()
            os.fsync(f.fileno())


def aggregate_results_from_jsonl(jsonl_path: str) -> Tuple[Dict[int, Dict[str, Any]], list]:
    idx2result = {}
    raw_list = []
    if not os.path.exists(jsonl_path):
        return idx2result, raw_list
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                idx = obj.get("idx")
                status = obj.get("status", "ok")
                result = obj.get("result")
                if isinstance(idx, int) and result is not None:
                    raw_list.append([idx, result])
                    if status == "ok":
                        idx2result[idx] = result
            except Exception:
                continue
    return idx2result, raw_list


if __name__ == '__main__':
    base_dir = os.path.dirname(__file__)

    # 优先使用“子分片”文件；没有就用整页文件
    sub_path = os.path.join(base_dir, "all_pdf_page_subchunks.json")
    page_path = os.path.join(base_dir, "all_pdf_page_chunks.json")
    if os.path.exists(sub_path):
        chunk_json_path = sub_path
        print(f"使用子分片文件：{chunk_json_path}")
    elif os.path.exists(page_path):
        chunk_json_path = page_path
        print(f"使用整页文件：{chunk_json_path}")
    else:
        raise FileNotFoundError("未找到 all_pdf_page_subchunks.json 或 all_pdf_page_chunks.json")

    # 构建向量库（并加载页级回溯用的 all_pdf_page_chunks.json）
    rag = SimpleRAG(chunk_json_path, page_json_path=page_path if os.path.exists(page_path) else None)
    rag.setup()

    # 读取测试集
    test_path = os.path.join(base_dir, "datas", "test.json")
    if not os.path.exists(test_path):
        print("datas/test.json 不存在")
        raise SystemExit(0)

    with open(test_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    all_indices = list(range(len(test_data)))

    # 续跑：读取已完成项
    done_map = load_done_map(RESULTS_JSONL)
    done_indices = set(done_map.keys())

    # 过滤出待处理的索引
    pending_indices = [i for i in all_indices if i not in done_indices]

    if not pending_indices:
        print("没有待处理的样本，直接汇总现有结果…")
    else:
        print(f"总样本: {len(all_indices)}，已完成: {len(done_indices)}，待处理: {len(pending_indices)}")

    def process_one(idx: int):
        item = test_data[idx]
        question = item['question']
        tqdm.write(f"[{len(done_indices) + pending_indices.index(idx) + 1}/{len(all_indices)}] 处理: {question[:30]}...")

        try:
            result = rag.generate_answer(question, top_k=3)
            rec = {"idx": idx, "status": "ok", "result": result}
            append_jsonl_threadsafe(RESULTS_JSONL, rec)
            return idx, result, None
        except Exception as e:
            rec = {
                "idx": idx,
                "status": "error",
                "error": str(e),
                "result": {
                    "question": item.get("question", ""),
                    "answer": "",
                    "filename": "",
                    "page": "",
                    "retrieval_chunks": []
                }
            }
            append_jsonl_threadsafe(RESULTS_JSONL, rec)
            return idx, rec["result"], e

    # 并发跑剩余的
    if pending_indices:
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            for _ in tqdm(executor.map(process_one, pending_indices),
                          total=len(pending_indices),
                          desc='并发批量生成（可续跑）'):
                pass

    # ===== 汇总阶段：从 JSONL 聚合出两个文件 =====
    idx2result, raw_list = aggregate_results_from_jsonl(RESULTS_JSONL)

    # 兼容原先 raw 快照
    with open(RAW_SNAPSHOT_JSON, 'w', encoding='utf-8') as f:
        json.dump(raw_list, f, ensure_ascii=False, indent=2)
    print(f'已输出原始未过滤结果到: {RAW_SNAPSHOT_JSON}')

    # 生成最终结构化结果（按 test_data 顺序，未答补默认内容）
    filtered_results = []
    for idx, item in enumerate(test_data):
        if idx in idx2result:
            r = idx2result[idx]
            filtered_results.append({k: v for k, v in r.items() if k != 'retrieval_chunks'})
        else:
            filtered_results.append({
                "question": item.get("question", ""),
                "answer": "",
                "filename": "",
                "page": "",
            })

    with open(FINAL_PRED_JSON, 'w', encoding='utf-8') as f:
        json.dump(filtered_results, f, ensure_ascii=False, indent=2)
    print(f'已输出结构化检索+大模型生成结果到: {FINAL_PRED_JSON}')
