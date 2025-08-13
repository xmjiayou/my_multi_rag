# -*- coding: utf-8 -*-
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

from dotenv import load_dotenv
from openai import OpenAI, APITimeoutError

from get_text_embedding import get_text_embedding

# 统一加载项目根目录的 .env
load_dotenv()

# ========= 可配置的输出文件 =========
RESULTS_JSONL = "./rag_results_raw.jsonl"   # 逐条写入，支持续跑
RAW_SNAPSHOT_JSON = "./rag_top1_pred_raw.json"  # 可选：一次性快照（从 JSONL 汇总生成）
FINAL_PRED_JSON = "./rag_top1_pred.json"    # 最终结构化结果

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
        emb_matrix = np.array(self.embeddings)
        query_emb = np.array(query_embedding)
        sims = emb_matrix @ query_emb / (norm(emb_matrix, axis=1) * norm(query_emb) + 1e-8)
        idxs = sims.argsort()[::-1][:top_k]
        return [self.chunks[i] for i in idxs]


# ======== 限流 / 重试 / 裁剪工具（缓解 429 与超时） ========

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
    其他异常直接抛出。
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
    def __init__(self, chunk_json_path: str, model_path: str = None, batch_size: int = 32):
        self.loader = PageChunkLoader(chunk_json_path)
        self.embedding_model = EmbeddingModel(batch_size=batch_size)
        self.vector_store = SimpleVectorStore()

    def setup(self):
        print("加载所有页 chunk...")
        chunks = self.loader.load_chunks()
        print(f"共加载 {len(chunks)} 个 chunk")
        print("生成嵌入...")
        embeddings = self.embedding_model.embed_texts([c['content'] for c in chunks])
        print("存储向量...")
        self.vector_store.add_chunks(chunks, embeddings)
        print("RAG 向量库构建完成！")

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
            top_k = 1
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
            temperature = 0.2

        q_emb = self.embedding_model.embed_text(question)
        chunks = self.vector_store.search(q_emb, top_k)

        # 拼接检索内容时裁剪每段，并限制总长度
        parts = []
        total_chars = 0
        for c in chunks:
            seg = f"[文件名]{c['metadata']['file_name']} [页码]{c['metadata']['page']}\n{truncate_fn(c['content'])}"
            if total_chars + len(seg) > MAX_TOTAL:
                break
            parts.append(seg)
            total_chars += len(seg)
        context = "\n".join(parts)

        prompt = (
            f"你是一名专业的金融分析助手，请根据以下检索到的内容回答用户问题。\n"
            f"请严格按照如下JSON格式输出：\n"
            f'{{"answer": "你的简洁回答", "filename": "来源文件名", "page": "来源页码"}}'"\n"
            f"检索内容：\n{context}\n\n问题：{question}\n"
            f"请确保输出内容为合法JSON字符串，不要输出多余内容。"
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

        import json as pyjson
        from extract_json_array import extract_json_array
        raw = completion.choices[0].message.content.strip()
        json_str = extract_json_array(raw, mode='objects')
        if json_str:
            try:
                arr = pyjson.loads(json_str)
                if isinstance(arr, list) and arr:
                    j = arr[0]
                    answer = j.get('answer', '')
                    filename = j.get('filename', '')
                    page = j.get('page', '')
                else:
                    answer = raw
                    filename = chunks[0]['metadata']['file_name'] if chunks else ''
                    page = chunks[0]['metadata']['page'] if chunks else ''
            except Exception:
                answer = raw
                filename = chunks[0]['metadata']['file_name'] if chunks else ''
                page = chunks[0]['metadata']['page'] if chunks else ''
        else:
            answer = raw
            filename = chunks[0]['metadata']['file_name'] if chunks else ''
            page = chunks[0]['metadata']['page'] if chunks else ''

        return {
            "question": question,
            "answer": answer,
            "filename": filename,
            "page": page,
            "retrieval_chunks": chunks
        }


# ========= 续跑：读写工具 =========

def load_done_map(jsonl_path: str) -> Dict[int, Dict[str, Any]]:
    """
    读取已完成的 JSONL，返回 idx -> {result/status/...}
    若文件不存在，则返回空字典。
    """
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
                    done[idx] = obj  # 若重复，后写覆盖先写（以最新为准）
            except Exception:
                continue
    return done


def append_jsonl_threadsafe(jsonl_path: str, obj: Dict[str, Any]):
    """
    线程安全地将一条记录追加到 JSONL。
    """
    data = json.dumps(obj, ensure_ascii=False)
    with write_lock:
        with open(jsonl_path, 'a', encoding='utf-8') as f:
            f.write(data + "\n")
            f.flush()
            os.fsync(f.fileno())


def aggregate_results_from_jsonl(jsonl_path: str) -> Tuple[Dict[int, Dict[str, Any]], list]:
    """
    读取 JSONL，返回：
    - idx -> result 映射（仅 status == 'ok' 的）
    - 原始列表（(idx, result)）用于兼容旧的 raw 快照
    """
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
    # 数据路径
    chunk_json_path = "./all_pdf_page_chunks.json"
    test_path = "./datas/test.json"

    # 构建向量库
    rag = SimpleRAG(chunk_json_path)
    rag.setup()

    # 读取测试集
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
            # 出错也记录一条，便于续跑排障
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

    # 兼容原先 raw 快照（[(idx, result), ...]）
    with open(RAW_SNAPSHOT_JSON, 'w', encoding='utf-8') as f:
        json.dump(raw_list, f, ensure_ascii=False, indent=2)
    print(f'已输出原始未过滤结果到: {RAW_SNAPSHOT_JSON}')

    # 生成最终结构化结果（按 test_data 顺序，未答补默认内容）
    filtered_results = []
    for idx, item in enumerate(test_data):
        if idx in idx2result:
            # 去除 retrieval_chunks（与原逻辑保持一致）
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
