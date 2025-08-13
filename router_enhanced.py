# router_enhanced.py
import re
from typing import List, Dict, Any, Optional
import os
from openai import OpenAI

# ---- 路由：与旧接口兼容 ----
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
        if any(x.replace("（","(").replace("）",")") in name for x in comps): s += 0.6
        if any(x in content for x in comps): s += 0.3
        if any(str(y) in name for y in years) or any(str(y) in content for y in years): s += 0.2
        if any(q in name.upper() for q in quarters) or any(q in content.upper() for q in quarters): s += 0.2
        c2 = dict(c)
        c2["_filter_score"] = s
        scored.append(c2)
    scored.sort(key=lambda x: x.get("_filter_score",0.0), reverse=True)
    return scored

# ---- HyDE 改写（按需触发；需 HYDE_ENABLE=1）----
def maybe_hyde_rewrite(q: str) -> Optional[str]:
    if os.getenv("HYDE_ENABLE", "0") != "1":
        return None
    api_key = os.getenv("LOCAL_API_KEY")
    base_url = os.getenv("LOCAL_BASE_URL")
    model = os.getenv("HYDE_TEXT_MODEL") or os.getenv("LOCAL_TEXT_MODEL")
    if not (api_key and base_url and model):
        return None
    prompt = (
        "将下列财报/研报查询改写为更具体、易检索的一句短查询。"
        "保留公司名/年份/季度/指标等关键信息，控制在30字以内：\n"
        f"原始问题：{q}\n"
        "改写："
    )
    try:
        client = OpenAI(api_key=api_key, base_url=base_url, timeout=20)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "你是精简查询生成器。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=64,
        )
        newq = (resp.choices[0].message.content or "").strip().replace("改写：","").strip()
        if newq and len(newq) >= 4 and newq != q:
            return newq
    except Exception:
        return None
    return None
