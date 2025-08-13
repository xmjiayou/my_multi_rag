# mini_graph.py
# 目的：多跳/对比类问题，为候选页追加“同文档相邻页”作为辅助证据。 新增：轻量“邻页扩展”通道
from typing import List, Dict, Any, Tuple

def _dedup(seq, keyfn):
    seen = set()
    out = []
    for x in seq:
        k = keyfn(x)
        if k in seen: continue
        seen.add(k); out.append(x)
    return out

def neighbor_expand(candidates: List[Dict[str, Any]],
                    page_full_map: Dict[Tuple[str,int], str],
                    hops: int = 1,
                    limit: int = 15) -> List[Dict[str, Any]]:
    if not candidates or not page_full_map:
        return candidates
    base = candidates[:max(5, min(8, len(candidates)))]  # 只对前几名扩展
    added = []
    for c in base:
        fn = c.get("file_name",""); pg = int(c.get("page", -1))
        for d in range(1, hops+1):
            for npg in (pg-d, pg+d):
                key = (fn, npg)
                if key in page_full_map:
                    nc = {
                        "id": f"{fn}_page_{npg}_nbr",
                        "file_name": fn,
                        "page": npg,
                        "content": page_full_map[key],
                        "image_path": c.get("image_path", None),
                        "_nbr": True
                    }
                    added.append(nc)
    merged = _dedup(base + added + candidates[len(base):], keyfn=lambda x: (x.get("file_name",""), int(x.get("page",-1))))
    return merged[:limit]
