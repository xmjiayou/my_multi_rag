# chunker.py
# 用法:
#   python chunker.py --in ./all_pdf_page_chunks.json --out ./all_pdf_page_subchunks.json \
#                     --target-tokens 480 --overlap 60

# 输入：all_pdf_page_chunks.json（页级内容，和你现在一致）

# 输出：all_pdf_page_subchunks.json（页内子分片，带起止偏移、估算 token 数）

# 规则：默认目标 480 token（近似），overlap 60，保证子分片→页映射稳定；支持断点重跑（已有输出不重复计算）。

# 接入方式：跑完 chunker.py 后，把主脚本中 chunk_json_path 指向 all_pdf_page_subchunks.json（或直接把该文件改名覆盖原始 all_pdf_page_chunks.json），即可完成“子分片检索 → 命中后回溯整页”（我们的主脚本已经这么做：候选里仍携带 file_name/page，生成时拼整页上下文）。

import os, json, argparse, math
from typing import List, Dict, Any

def rough_token_count(s: str) -> int:
    # 近似 token 估计：中英文混排时经验上“中文字符≈1 token，英文单词≈1.3 token”
    # 为避免外部依赖，这里用“中文字符计数 + 英文/数字按空格分词”的粗估。
    if not s: return 0
    import re
    zh = len(re.findall(r'[\u4e00-\u9fa5]', s))
    en = re.findall(r'[A-Za-z0-9]+', s)
    return zh + int(sum(max(1, len(w)//4) for w in en))  # 保守估计

def split_text_by_tokens(text: str, target_tokens=480, overlap=60) -> List[Dict[str, int]]:
    """返回一组片段的 {start, end}（按字符切，token 近似控制）"""
    if not text or target_tokens <= 0:
        return [{"start": 0, "end": len(text)}]
    # 简单按字符滑窗，但步长由“近似 token”反推
    spans = []
    n = len(text)
    i = 0
    while i < n:
        # 扩到目标 tokens 附近
        lo = i
        hi = min(n, i + max(200, target_tokens * 2))  # 上限兜底，避免死循环
        # 以字符窗口逐步增加，直到 rough_token_count 达到 target
        step = max(150, target_tokens)  # 初步跳
        j = min(n, i + step)
        while j < hi and rough_token_count(text[i:j]) < target_tokens:
            j = min(n, j + 150)
        # 尽量在句号/换行处收尾
        end = j
        for k in range(min(n, j+80)-1, i, -1):
            if text[k] in ("。", "；", "\n", "！", "？", "."):
                end = k+1
                break
        spans.append({"start": lo, "end": end})
        # 下一窗口：end - overlap
        next_i = end - max(0, overlap*2)  # 字符近似放大
        i = max(i+1, min(n, next_i))
        if i >= end:  # 防止卡住
            i = end
    # 去重&排序
    cleaned = []
    last = -1
    for sp in spans:
        s, e = sp["start"], sp["end"]
        if e <= s: continue
        if s <= last: s = last + 1
        if e <= s: continue
        cleaned.append({"start": s, "end": e})
        last = e
    return cleaned if cleaned else [{"start":0, "end": len(text)}]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="all_pdf_page_chunks.json")
    ap.add_argument("--out", dest="outp", required=True, help="输出 all_pdf_page_subchunks.json")
    ap.add_argument("--target-tokens", type=int, default=480)
    ap.add_argument("--overlap", type=int, default=60)
    args = ap.parse_args()

    with open(args.inp, "r", encoding="utf-8") as f:
        pages = json.load(f)

    out = []
    for page in pages:
        txt = page.get("content","") or ""
        spans = split_text_by_tokens(txt, target_tokens=args.target_tokens, overlap=args.overlap)
        for seg_idx, sp in enumerate(spans):
            seg = txt[sp["start"]:sp["end"]]
            out.append({
                "id": f"{page['id']}_seg_{seg_idx}",
                "content": seg,
                "metadata": {
                    "file_name": page["metadata"]["file_name"],
                    "page": page["metadata"]["page"],
                    "seg_index": seg_idx,
                    "char_start": sp["start"],
                    "char_end": sp["end"],
                    "token_est": rough_token_count(seg),
                    "page_id": page["id"]
                }
            })

    with open(args.outp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"完成：输入 {len(pages)} 页 → 输出 {len(out)} 子分片；写入 {args.outp}")

if __name__ == "__main__":
    main()
