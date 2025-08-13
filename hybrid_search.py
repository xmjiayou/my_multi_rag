# -*- coding: utf-8 -*-
"""
HybridSearcher:
- 文本语义召回：依赖外部传入的 text_search_fn
- 稀疏BM25召回：对 chunks['content'] 建本地BM25
- 视觉索引接入：加载 vision_index/meta.json，给候选补上 image_path（用于后续视觉重排）
- RRF 融合 & 去重
"""
import os
import json
from typing import List, Dict, Any, Tuple
from collections import defaultdict

from rank_bm25 import BM25Okapi


def _load_meta(index_dir: str) -> Dict[str, Any]:
    meta_path = os.path.join(index_dir, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    # 兼容老的 meta.jsonl（如果有人曾这样写过）
    meta_jsonl = os.path.join(index_dir, "meta.jsonl")
    if os.path.exists(meta_jsonl):
        try:
            lines = [x for x in open(meta_jsonl, encoding="utf-8") if x.strip()]
            return json.loads(lines[-1])
        except Exception:
            pass
    return {}


class HybridSearcher:
    def __init__(self, chunks: List[Dict[str, Any]], text_search_fn, vision_index_dir: str = "./vision_index"):
        self.chunks = chunks
        self.text_search_fn = text_search_fn

        # Build BM25 over chunk texts
        self._bm25_corpus = []
        for c in chunks:
            content = (c.get("content") or "").strip()
            self._bm25_corpus.append(content.split())
        self.bm25 = BM25Okapi(self._bm25_corpus)

        # Load vision meta & lookup to attach image_path
        self.vision_meta = _load_meta(vision_index_dir)
        self.lookup = {}
        if self.vision_meta:
            lp = os.path.join(vision_index_dir, self.vision_meta.get("lookup_path", "lookup.json"))
            if os.path.exists(lp):
                try:
                    self.lookup = json.load(open(lp, "r", encoding="utf-8"))
                except Exception:
                    self.lookup = {}

    def _attach_image_path(self, item: Dict[str, Any]) -> Dict[str, Any]:
        fn = item.get("metadata", {}).get("file_name") or item.get("file_name")
        pg = item.get("metadata", {}).get("page") if "metadata" in item else item.get("page")
        if fn is None or pg is None:
            return item
        key = f"{fn}::{int(pg)}"
        rel = self.lookup.get(key)
        if rel:
            item = dict(item)
            item["file_name"] = fn
            item["page"] = int(pg)
            item["image_path"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vision_index", rel)
        else:
            # 保底也补齐 file_name/page 字段，方便后续使用
            item = dict(item)
            item["file_name"] = fn
            item["page"] = int(pg)
        # id 字段保底
        if "id" not in item:
            item["id"] = f"{fn}:::{int(pg)}"
        return item

    @staticmethod
    def _rrf_merge(lists: List[List[Dict[str, Any]]], k: int = 60, k_param: int = 60) -> List[Dict[str, Any]]:
        """
        Reciprocal Rank Fusion: 给多个有序列表融合排序
        """
        scores = defaultdict(float)
        seen_obj = {}

        def _key(obj):
            fn = obj.get("file_name") or obj.get("metadata", {}).get("file_name")
            pg = obj.get("page") if "page" in obj else obj.get("metadata", {}).get("page")
            return (fn, int(pg) if pg is not None else -1, obj.get("id") or "")

        for lst in lists:
            for rank, obj in enumerate(lst, start=1):
                key = _key(obj)
                seen_obj[key] = obj
                scores[key] += 1.0 / (k_param + rank)

        ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        out = []
        for key, _ in ordered[:k]:
            out.append(seen_obj[key])
        return out

    def _bm25_search(self, query: str, top_k: int = 100) -> List[Dict[str, Any]]:
        toks = query.split()
        scores = self.bm25.get_scores(toks)
        idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [self.chunks[i] for i in idxs]

    def search_candidates(self, query: str, top_k_text: int = 60, top_k_bm25: int = 100, final_k: int = 80) -> List[Dict[str, Any]]:
        # 语义召回（向量）+ 稀疏召回（BM25）
        sem = self.text_search_fn(query, top_k=top_k_text) or []
        bm = self._bm25_search(query, top_k=top_k_bm25) or []

        merged = self._rrf_merge([sem, bm], k=max(final_k, 1))
        # 给每个候选补上 file_name/page/id 和 image_path（若有）
        enriched = [self._attach_image_path(x) for x in merged]
        # 去重（以 file_name/page/id）
        seen = set()
        uniq = []
        for c in enriched:
            key = (c.get("file_name"), int(c.get("page", -1)), c.get("id"))
            if key in seen:
                continue
            uniq.append(c)
            seen.add(key)
        return uniq
