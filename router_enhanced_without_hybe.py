# router_enhanced.py
# 维持原有 route_query(q)，额外提供：

# extract_filters(q)：从问题里抓公司/年份/季度（轻量规则 + 正则），形成过滤线索；

# apply_filters(candidates, filters)：对候选打加权分（同公司/同年/同季度/同页邻近）并重排；

# maybe_hyde_rewrite(q)：保留 HyDE 改写占位（默认关闭，避免额外 LLM 调用）。

# 你主脚本里只需在视觉重排后加一行 candidates = apply_filters(candidates, extract_filters(question)) 即可。

# 主脚本小改（两行）：把 from router import route_query 改为
# from router_enhanced import route_query, extract_filters, apply_filters，然后在视觉重排结束后增加一行重排：

# python
# 复制
# 编辑
# # （主脚本 generate_answer 内）
# prim = self.vis_reranker.rerank(question, prim, top_k=10)
# others = [c for c in cand if c["id"] not in {x["id"] for x in prim}]
# candidates = prim + others
# candidates = apply_filters(candidates[:10], extract_filters(question))  # ★ 新增
import re
from typing import List, Dict, Any, Tuple

# ---- 路由：维持与旧接口兼容 ----
CHART_HINT = re.compile(r"(柱状|折线|饼图|图表|曲线|走势|同比|环比|增速)", re.I)
TABLE_HINT = re.compile(r"(表格|数据|明细|金额|收入|利润|同比|环比|占比|增长率|单位[:：])", re.I)
MULTI_HOP_HINT = re.compile(r"(分别|对比|对照|以及|同时|两者|三者)", re.I)

def route_query(q: str) -> str:
    qs = q.strip()
    if CHART_HINT.search(qs):
        return "chart"
    if TABLE_HINT.search(qs):
        return "table"
    if MULTI_HOP_HINT.search(qs):
        return "multihop"
    return "text"

# ---- 轻量过滤线索提取 ----
COMPANY_PAT = re.compile(r"([\u4e00-\u9fa5A-Za-z0-9（）()·\-]{2,20})(股份有限公司|集团|公司|有限责任公司)")
YEAR_PAT = re.compile(r"(20\d{2})年")
QUARTER_PAT = re.compile(r"(第[一二三四1234]季度|Q[1-4]|一季报|二季报|三季报|四季报)", re.I)

def extract_filters(q: str) -> Dict[str, Any]:
    companies = [m.group(0) for m in COMPANY_PAT.finditer(q)]
    years = [int(y) for y in YEAR_PAT.findall(q)]
    quarter_raw = QUARTER_PAT.findall(q)
    def norm_quarter(x: str) -> str:
        m = x.upper()
        if "一" in x or "1" in m: return "Q1"
        if "二" in x or "2" in m: return "Q2"
        if "三" in x or "3" in m: return "Q3"
        if "四" in x or "4" in m: return "Q4"
        if "Q1" in m: return "Q1"
        if "Q2" in m: return "Q2"
        if "Q3" in m: return "Q3"
        if "Q4" in m: return "Q4"
        return ""
    quarters = [norm_quarter(x) for x in quarter_raw if norm_quarter(x)]
    return {"companies": companies, "years": years, "quarters": quarters}

# ---- 候选重排（加权） ----
def apply_filters(cands: List[Dict[str, Any]], filters: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not cands or not filters:
        return cands
    comps = filters.get("companies", [])
    years = set(filters.get("years", []))
    quarters = set(filters.get("quarters", []))

    scored = []
    for c in cands:
        s = 0.0
        name = c.get("file_name","")
        content = c.get("content","") or ""
        # 公司名：文件名中出现加 0.6；正文中出现加 0.3
        if any(x.replace("（","(").replace("）",")") in name for x in comps): s += 0.6
        if any(x in content for x in comps): s += 0.3
        # 年份：文件名或正文出现（简单查找）加 0.2
        if any(str(y) in name for y in years) or any(str(y) in content for y in years): s += 0.2
        # 季度：正文/文件名出现 Q1..Q4 加 0.2
        if any(q in name.upper() for q in quarters) or any(q in content.upper() for q in quarters): s += 0.2
        # 页邻近：若候选有 page 字段，优先小页码差（当前无对标页时可忽略，此处占位）
        c2 = dict(c)
        c2["_filter_score"] = s
        scored.append(c2)
    scored.sort(key=lambda x: x.get("_filter_score",0.0), reverse=True)
    return scored

# ---- 可选：HyDE 改写占位（默认不用，避免开销） ----
def maybe_hyde_rewrite(q: str) -> str:
    # 你可以接你们的本地小模型，或走已有 LLM，生成“更具体”的子查询
    # 这里返回 None 表示不改写
    return None
