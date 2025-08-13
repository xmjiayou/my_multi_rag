# rag_from_page_chunks_stable.py
# 变化点：
# 1) JSONL 写入 meta: type_bucket / pre_vis_top10 / post_vis_top10
# 2) 召回不足触发 HyDE（环境变量 HYDE_ENABLE=1）
# 3) 多跳题可选邻页扩展（环境变量 GRAPH_ENABLE=1）

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

from get_text_embedding import get_text_embedding
from extract_json_array import extract_json_array

from hybrid_search import HybridSearcher
from vision_rerank import VisionReranker
from router_enhanced import route_query, extract_filters, apply_filters, maybe_hyde_rewrite
from chart_extract import extract_tables_from_page_markdown, chart_image_to_csv
from mini_graph import neighbor_expand  # 可选使用

load_dotenv()

RESULTS_JSONL = "./rag_results_raw.jsonl"
RAW_SNAPSHOT_JSON = "./rag_top1_pred_raw.json"
FINAL_PRED_JSON = "./rag_top1_pred.json"

OPENAI_RPM_LIMIT = int(os.getenv("OPENAI_RPM_LIMIT", "60"))
REQUEST_TIMEOUT = float(os.getenv("OPENAI_REQUEST_TIMEOUT", "75"))
MAX_WORKERS = int(os.getenv("RAG_MAX_WORKERS", "3"))
FAST_MODE = os.getenv("FAST_MODE", "0") == "1"

HYDE_TRIGGER_K = int(os.getenv("HYDE_TRIGGER_K", "20"))
GRAPH_ENABLE = os.getenv("GRAPH_ENABLE", "0") == "1"

write_lock = threading.Lock()


class PageChunkLoader:
    def __init__(self, json_path: str):
        self.json_path = json_path

    def load_chunks(self) -> List[Dict[str, Any]]:
        with open(self.json_path, 'r', encoding='utf-8') as f:
            return json.load(f)


class PageFullLoader:
    def __init__(self, page_json_path: str):
        self.page_json_path = page_json_path

    def load_full_map(self) -> Dict[Tuple[str, int], str]:
        mp = {}
        if not self.page_json_path or not os.path.exists(self.page_json_path):
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
        self.chunks = []

    def add_chunks(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]):
        self.chunks.extend(chunks)
        self.embeddings.extend(embeddings)

    def search(self, query_embedding: List[float], top_k: int = 3) -> List[Dict[str, Any]]:
        from numpy.linalg import norm
        import numpy as np
        if not self.embeddings:
            return []
        M = np.array(self.embeddings)
        q = np.array(query_embedding)
        sims = M @ q / (norm(M, axis=1) * norm(q) + 1e-8)
        idxs = sims.argsort()[::-1][:top_k]
        return [self.chunks[i] for i in idxs]


class RequestRateLimiter:
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
    attempt = 0
    backoff = base_backoff
    while True:
        attempt += 1
        try:
            return callable_fn(*args, **kwargs)
        except (APITimeoutError, httpx.ReadTimeout, httpx.ConnectTimeout, socket.timeout):
            if attempt >= max_tries:
                raise
            sleep_s = min(max_backoff, backoff)
            import random as rd
            time.sleep(sleep_s + rd.uniform(0, 0.4))
            backoff = min(max_backoff, backoff * 1.8)
        except Exception as e:
            if ("429" in str(e) or "RateLimit" in str(e) or "TPM" in str(e)) and attempt < max_tries:
                import random as rd
                time.sleep(min(max_backoff, backoff) + rd.uniform(0, 0.5))
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
        self.page_full_map = {}
        self.page_json_path = page_json_path

    def setup(self):
        print("加载 chunk（子分片或整页）...")
        chunks = self.loader.load_chunks()
        print(f"共加载 {len(chunks)} 条")
        print("生成嵌入...")
        emb = self.embedding_model.embed_texts([c['content'] for c in chunks])
        print("存储向量...")
        self.vector_store.add_chunks(chunks, emb)
        print("RAG 向量库构建完成！")

        if not self.page_json_path:
            self.page_json_path = os.path.join(os.path.dirname(__file__), "all_pdf_page_chunks.json")
        self.page_full_map = PageFullLoader(self.page_json_path).load_full_map()

        if self.page_full_map:
            print(f"已载入页级内容：{len(self.page_full_map)} 页")
        else:
            print("警告：未找到 all_pdf_page_chunks.json，整页回溯将退化为子分片内容")

        def text_search_fn(q, top_k=50):
            q_emb = self.embedding_model.embed_text(q)
            return self.vector_store.search(q_emb, top_k)

        self.hybrid = HybridSearcher(self.vector_store.chunks, text_search_fn, vision_index_dir="./vision_index")

        try:
            self.vis_reranker = VisionReranker(index_dir="./vision_index", device=os.getenv("VISION_DEVICE", "cuda"))
        except Exception as e:
            print(f"视觉重排初始化失败，降级直通：{e}")

            class NoOp:
                def rerank(self, q, c, top_k=10):
                    return c[:top_k]

            self.vis_reranker = NoOp()

    def generate_answer(self, question: str, top_k: int = 3) -> Dict[str, Any]:
        qwen_api_key = os.getenv('LOCAL_API_KEY')
        qwen_base_url = os.getenv('LOCAL_BASE_URL')
        # 路由式：优先从 GEN_MODEL_TEXT / GEN_MODEL_VLM 读；否则退回 LOCAL_TEXT_MODEL
        default_model = os.getenv('LOCAL_TEXT_MODEL')
        gen_model_text = os.getenv('GEN_MODEL_TEXT', default_model)
        gen_model_vlm = os.getenv('GEN_MODEL_VLM', default_model)
        if not (qwen_api_key and qwen_base_url and (gen_model_text or gen_model_vlm)):
            raise ValueError('请在 .env 中配置 LOCAL_API_KEY、LOCAL_BASE_URL、以及 GEN_MODEL_TEXT/GEN_MODEL_VLM 或 LOCAL_TEXT_MODEL')

        if FAST_MODE:
            MAX_TOTAL = 1500
            gen_max_tokens = 256
            temperature = 0.0

            def trunc(x):
                x = (x or "").strip()
                return x[:500] + ("…" if len(x) > 500 else "")
        else:
            MAX_TOTAL = 3500
            gen_max_tokens = 512
            temperature = 0.1

            def trunc(x):
                return truncate_text(x)

        qtype = route_query(question)

        # ===== 召回 =====
        cand = self.hybrid.search_candidates(question, top_k_text=50, top_k_bm25=100, final_k=80)

        # ---- HyDE 改写（仅当召回不足）----
        if os.getenv("HYDE_ENABLE", "0") == "1" and len(cand) < HYDE_TRIGGER_K:
            q2 = maybe_hyde_rewrite(question)
            if q2 and q2 != question:
                cand2 = self.hybrid.search_candidates(q2, top_k_text=50, top_k_bm25=100, final_k=80)
                seen = set()
                merged = []
                for c in (cand + cand2):
                    key = (c["file_name"], int(c["page"]), c["id"])
                    if key in seen:
                        continue
                    merged.append(c)
                    seen.add(key)
                cand = merged

        # 初选→视觉重排→合并
        prim = cand[:20]
        prim = self.vis_reranker.rerank(question, prim, top_k=10)
        others = [c for c in cand if c["id"] not in {x["id"] for x in prim}]
        candidates = prim + others

        # 记录重排前后（用于 JSONL 指标）
        pre_vis = [(c["file_name"], int(c["page"])) for c in cand[:10]]
        post_vis = [(c["file_name"], int(c["page"])) for c in candidates[:10]]
        type_bucket = qtype

        # 公司/年份/季度加权重排
        candidates = apply_filters(candidates[:10], extract_filters(question))

        # 多跳题：可选邻页扩展
        if GRAPH_ENABLE and qtype == "multihop":
            candidates = neighbor_expand(candidates, self.page_full_map, hops=1, limit=15)

        # ===== 组装上下文（表格/图表增强）=====
        parts = []
        for c in candidates:
            fn = c['file_name']
            pg = int(c['page'])
            header = f"[文件名]{fn} [页码]{pg}"
            page_text = self.page_full_map.get((fn, pg), c.get("content", "") or "")
            ctx = page_text

            if qtype in ("table", "chart"):
                try:
                    dfs = extract_tables_from_page_markdown(page_text)
                except Exception:
                    dfs = []
                if dfs:
                    for df in dfs[:2]:
                        df = df.fillna("").astype(str)
                        pri = df[df.apply(
                            lambda row: row.astype(str).str.contains(
                                r"(合计|总计|同比|环比|增长|率|%|亿元|万元|收入|利润)",
                                regex=True
                            ).any(),
                            axis=1
                        )]
                        sub = (pd.concat([pri.head(6), df.head(4)]) if not pri.empty else df.head(8)).drop_duplicates().head(8)
                        ctx = f"{ctx}\n[表格摘录]\n{sub.to_csv(index=False)}"
                        break
                if c.get("image_path"):
                    try:
                        csv_text = chart_image_to_csv(c["image_path"])
                    except Exception:
                        csv_text = None
                    if csv_text:
                        ctx = f"{ctx}\n[图表CSV]\n{csv_text}"

            parts.append(f"{header}\n{trunc(ctx)}")

        total = 0
        final_parts = []
        for p in parts:
            if total + len(p) > MAX_TOTAL:
                break
            final_parts.append(p)
            total += len(p)
        context = "\n\n".join(final_parts)

        # 路由选择模型：图表/表格题优先用 VLM，其余用文本模型
        qwen_model = gen_model_vlm if qtype in ("chart", "table") and gen_model_vlm else gen_model_text

        # prompt = (
        #     "你是一名专业的金融分析助手，请根据以下检索到的内容回答用户问题.\n"
        #     "以 JSON 输出：{\"answer\":\"简洁回答\",\"filename\":\"来源文件名\",\"page\":\"来源页码\"}\n"
        #     "证据：\n" + context + "\n\n"
        #     "问题：" + question + "\n只输出 JSON。"
        # )
        prompt = (
            f"你是一名专业的金融分析助手，请根据以下检索到的内容回答用户问题。\n"
            f"请严格按照如下JSON格式输出：\n"
            f'{{"answer": "你的简洁回答", "filename": "来源文件名", "page": "来源页码"}}'"\n"
            f"检索内容：\n{context}\n\n问题：{question}\n"
            f"请确保输出内容为合法JSON字符串，不要输出多余内容。"
        )

        client = OpenAI(api_key=qwen_api_key, base_url=qwen_base_url, timeout=REQUEST_TIMEOUT, max_retries=0)

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
            "retrieval_chunks": candidates,
            "pre_vis_top10": pre_vis,
            "post_vis_top10": post_vis,
            "type_bucket": type_bucket
        }


# ===== 续跑工具 =====
def load_done_map(jsonl_path: str) -> Dict[int, Dict[str, Any]]:
    done = {}
    if not os.path.exists(jsonl_path):
        return done
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                obj = json.loads(ln)
                idx = obj.get("idx")
                if isinstance(idx, int):
                    done[idx] = obj
            except Exception:
                continue
    return done


def append_jsonl_threadsafe(path: str, obj: Dict[str, Any]):
    data = json.dumps(obj, ensure_ascii=False)
    with write_lock:
        with open(path, 'a', encoding='utf-8') as f:
            f.write(data + "\n")
            f.flush()
            os.fsync(f.fileno())


def aggregate_results_from_jsonl(path: str):
    idx2result = {}
    raw_list = []
    if not os.path.exists(path):
        return idx2result, raw_list
    with open(path, 'r', encoding='utf-8') as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                obj = json.loads(ln)
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

    rag = SimpleRAG(chunk_json_path, page_json_path=page_path if os.path.exists(page_path) else None)
    rag.setup()

    test_path = os.path.join(base_dir, "datas", "test.json")
    if not os.path.exists(test_path):
        print("datas/test.json 不存在")
        raise SystemExit(0)
    with open(test_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    all_idx = list(range(len(test_data)))
    done_map = load_done_map(RESULTS_JSONL)
    done_idx = set(done_map.keys())
    pending = [i for i in all_idx if i not in done_idx]

    if not pending:
        print("没有待处理样本，直接汇总…")
    else:
        print(f"总样本: {len(all_idx)}，已完成: {len(done_idx)}，待处理: {len(pending)}")

    def process_one(idx: int):
        item = test_data[idx]
        question = item['question']
        tqdm.write(f"[{len(done_idx) + pending.index(idx) + 1}/{len(all_idx)}] 处理: {question[:28]}...")
        try:
            result = rag.generate_answer(question, top_k=3)
            rec = {
                "idx": idx,
                "status": "ok",
                "result": result,
                "meta": {
                    "type_bucket": result.get("type_bucket"),
                    "pre_vis_top10": result.get("pre_vis_top10", []),
                    "post_vis_top10": result.get("post_vis_top10", [])
                }
            }
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

    if pending:
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            for _ in tqdm(ex.map(process_one, pending), total=len(pending), desc="并发批量生成（可续跑）"):
                pass

    idx2result, raw_list = aggregate_results_from_jsonl(RESULTS_JSONL)

    with open(RAW_SNAPSHOT_JSON, 'w', encoding='utf-8') as f:
        json.dump(raw_list, f, ensure_ascii=False, indent=2)
    print(f'已输出原始未过滤结果到: {RAW_SNAPSHOT_JSON}')

    filtered = []
    for idx, item in enumerate(test_data):
        if idx in idx2result:
            r = idx2result[idx]
            filtered.append({k: v for k, v in r.items() if k != "retrieval_chunks"})
        else:
            filtered.append({
                "question": item.get("question", ""),
                "answer": "",
                "filename": "",
                "page": ""
            })

    with open(FINAL_PRED_JSON, 'w', encoding='utf-8') as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)
    print(f'已输出结构化检索+大模型生成结果到: {FINAL_PRED_JSON}')
