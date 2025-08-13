# eval_report.py
# 用法:
#   python eval_report.py --gt ./datas/多模态RAG图文问答挑战赛训练集.json \
#                         --pred-jsonl ./rag_results_raw.jsonl \
#                         [--recompute]
# 3) 评估脚本 eval_report.py
# 输入：

# --gt：地面真值（如 ./datas/多模态RAG图文问答挑战赛训练集.json，需含 question/answer/filename/page）

# --pred-jsonl：你运行时的 rag_results_raw.jsonl

# 可选 --recompute：若想评估“视觉重排带来的收益”，脚本会调用 HybridSearcher + VisionReranker 重算一遍重排前后的 hit@k（会稍慢）

# 输出：

# eval_report.md：总体指标、各题型 bucket 指标

# eval_details.csv：逐题详情（匹配/命中@k、预测来源/真值来源……）
import os, json, argparse, pandas as pd
from typing import Dict, Any, List, Tuple
from router_enhanced import route_query
from collections import defaultdict

def load_gt(gt_path: str) -> List[Dict[str, Any]]:
    with open(gt_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # 规范化
    out = []
    for d in data:
        out.append({
            "question": d.get("question",""),
            "answer": d.get("answer",""),
            "filename": d.get("filename",""),
            "page": int(d.get("page", -1) if d.get("page") is not None else -1)
        })
    return out

def load_pred_jsonl(pred_path: str) -> Dict[int, Dict[str, Any]]:
    ret = {}
    if not os.path.exists(pred_path):
        return ret
    with open(pred_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                obj = json.loads(line)
                idx = obj.get("idx")
                if isinstance(idx, int):
                    ret[idx] = obj
            except Exception:
                continue
    return ret

def norm_filename(s: str) -> str:
    s = (s or "").strip().lower()
    return s.replace(".pdf","").replace(" ","")

def eval_basic(gt_list: List[Dict[str, Any]], pred_map: Dict[int, Dict[str, Any]]) -> Tuple[pd.DataFrame, Dict[str,float]]:
    rows = []
    for idx, gt in enumerate(gt_list):
        pred_obj = pred_map.get(idx, {})
        res = pred_obj.get("result", {})
        pred_fn = res.get("filename","")
        pred_pg = res.get("page","")
        try:
            pred_pg = int(pred_pg)
        except Exception:
            pred_pg = -1

        # 命中标志
        fn_ok = int(norm_filename(pred_fn) == norm_filename(gt["filename"]))
        pg_ok = int(pred_pg == gt["page"])

        # 检索 hit@k（基于已保存的 retrieval_chunks）
        hits = res.get("retrieval_chunks", [])
        ids = [(norm_filename(x.get("file_name","")), int(x.get("page",-999))) for x in hits]
        def hitk(k, fn, pg):
            subset = ids[:k]
            return int(any(a==norm_filename(fn) and b==pg for a,b in subset))
        hit1 = hitk(1, gt["filename"], gt["page"])
        hit3 = hitk(3, gt["filename"], gt["page"])
        hit5 = hitk(5, gt["filename"], gt["page"])
        hit10= hitk(10, gt["filename"], gt["page"])

        q = gt["question"]
        bucket = route_query(q)

        rows.append({
            "idx": idx,
            "bucket": bucket,
            "gt_filename": gt["filename"],
            "gt_page": gt["page"],
            "pred_filename": pred_fn,
            "pred_page": pred_pg,
            "filename_ok": fn_ok,
            "page_ok": pg_ok,
            "both_ok": int(fn_ok and pg_ok),
            "retrieval_hit@1": hit1,
            "retrieval_hit@3": hit3,
            "retrieval_hit@5": hit5,
            "retrieval_hit@10": hit10,
        })
    df = pd.DataFrame(rows)
    report = {
        "filename_acc": df["filename_ok"].mean(),
        "page_acc": df["page_ok"].mean(),
        "both_acc": df["both_ok"].mean(),
        "hit@1": df["retrieval_hit@1"].mean(),
        "hit@3": df["retrieval_hit@3"].mean(),
        "hit@5": df["retrieval_hit@5"].mean(),
        "hit@10": df["retrieval_hit@10"].mean(),
    }
    return df, report

def per_bucket_report(df: pd.DataFrame) -> pd.DataFrame:
    agg = df.groupby("bucket").agg({
        "filename_ok":"mean",
        "page_ok":"mean",
        "both_ok":"mean",
        "retrieval_hit@1":"mean",
        "retrieval_hit@3":"mean",
        "retrieval_hit@5":"mean",
        "retrieval_hit@10":"mean",
    }).reset_index()
    return agg

def save_markdown(report: Dict[str,float], bucket_df: pd.DataFrame, out_md="eval_report.md"):
    lines = []
    lines.append("# RAG 评估报告\n")
    lines.append("## 总览\n")
    lines.append(f"- 文件名准确率：{report['filename_acc']:.3f}")
    lines.append(f"- 页码准确率：{report['page_acc']:.3f}")
    lines.append(f"- 两者同时正确：{report['both_acc']:.3f}")
    lines.append(f"- 检索 Hit@1/3/5/10：{report['hit@1']:.3f} / {report['hit@3']:.3f} / {report['hit@5']:.3f} / {report['hit@10']:.3f}\n")
    lines.append("## 各题型分桶\n")
    lines.append(bucket_df.to_markdown(index=False))
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"已写入 {out_md}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True, help="含 question/answer/filename/page 的标注集 JSON")
    ap.add_argument("--pred-jsonl", required=True, help="rag_results_raw.jsonl")
    ap.add_argument("--details-csv", default="eval_details.csv")
    ap.add_argument("--report-md", default="eval_report.md")
    # 可选：--recompute 可扩展为重算重排收益（此处留钩子，默认关闭以更快）
    ap.add_argument("--recompute", action="store_true")
    args = ap.parse_args()

    gt = load_gt(args.gt)
    pred_map = load_pred_jsonl(args.pred_jsonl)

    df, rpt = eval_basic(gt, pred_map)
    df.to_csv(args.details_csv, index=False, encoding="utf-8-sig")
    print(f"已写入 {args.details_csv}")

    bucket_df = per_bucket_report(df)
    save_markdown(rpt, bucket_df, out_md=args.report_md)

if __name__ == "__main__":
    main()
