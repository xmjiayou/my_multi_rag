# chart_extract.py
# chart_extract.py（带 DePlot 推理，默认开启）
# 如果你暂时不想装 DePlot 依赖，把 USE_DEPLOT = False 就会自动跳过图表转 CSV（保留表格抽取）。
import os
import re
import traceback
from typing import List, Optional

import pandas as pd
from bs4 import BeautifulSoup
from PIL import Image

# ==== 开关 ====
USE_DEPLOT = True  # 若环境未就绪可改 False
DEPLOT_MODEL_ID = os.getenv("DEPLOT_MODEL", "google/deplot")
DEPLOT_DEVICE = os.getenv("DEPLOT_DEVICE", "cuda")
DEPLOT_MAX_NEW_TOKENS = int(os.getenv("DEPLOT_MAX_NEW_TOKENS", "512"))

# 全局单例
_DEPLOT_MODEL = None
_DEPLOT_PROCESSOR = None

def _load_deplot():
    """懒加载 DePlot 模型/处理器"""
    global _DEPLOT_MODEL, _DEPLOT_PROCESSOR
    if _DEPLOT_MODEL is not None and _DEPLOT_PROCESSOR is not None:
        return _DEPLOT_MODEL, _DEPLOT_PROCESSOR
    from transformers import VisionEncoderDecoderModel, AutoProcessor
    _DEPLOT_MODEL = VisionEncoderDecoderModel.from_pretrained(DEPLOT_MODEL_ID).to(DEPLOT_DEVICE)
    _DEPLOT_MODEL.eval()
    _DEPLOT_PROCESSOR = AutoProcessor.from_pretrained(DEPLOT_MODEL_ID)
    return _DEPLOT_MODEL, _DEPLOT_PROCESSOR


def _normalize_table_text(txt: str) -> str:
    """尝试把模型输出的表格样式转成 CSV 友好格式"""
    s = (txt or "").strip()

    # 常见的 " | col1 | col2 | " Markdown 表
    if "|" in s and "\n" in s:
        lines = []
        for line in s.splitlines():
            line = line.strip()
            if not line:
                continue
            # 去 Markdown 表头的对齐行: |---|---|
            if re.match(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?\s*$", line):
                continue
            # 去掉首尾 '|'
            if line.startswith("|"):
                line = line[1:]
            if line.endswith("|"):
                line = line[:-1]
            cols = [c.strip() for c in line.split("|")]
            lines.append(",".join(cols))
        return "\n".join(lines)

    # 逗号/制表符分隔
    if "," in s or "\t" in s:
        return s

    # 空格分隔的“等宽表”，粗略转 CSV
    lines = []
    for line in s.splitlines():
        line = re.sub(r"\s{2,}", ",", line.strip())
        if line:
            lines.append(line)
    return "\n".join(lines)


def chart_image_to_csv(image_path: str) -> Optional[str]:
    """图表图片 -> CSV 文本（DePlot 推理）。失败返回 None。"""
    if not USE_DEPLOT:
        return None
    try:
        if not os.path.exists(image_path):
            return None
        model, processor = _load_deplot()
        image = Image.open(image_path).convert("RGB")

        # 不同权重的 prompt 会有差异，可以按需调整
        prompt = "Generate the underlying data table of the figure in CSV format."
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)

        import torch
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=DEPLOT_MAX_NEW_TOKENS,
                num_beams=3,
                early_stopping=True
            )
        out = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        out = out.strip()
        if not out:
            return None
        csv_text = _normalize_table_text(out)
        # 简单有效性检查：至少两行两列
        rows = [r for r in csv_text.splitlines() if r.strip()]
        if len(rows) >= 2 and any("," in r for r in rows):
            return csv_text
        return None
    except Exception:
        # 打印一次错误即可，别影响主流程
        traceback.print_exc()
        return None


def extract_tables_from_page_markdown(page_md: str) -> List[pd.DataFrame]:
    """从 MinerU 生成的页面 Markdown 里抓出 <table>...</table> 并转成 DataFrame 列表"""
    if not page_md:
        return []
    soup = BeautifulSoup(page_md, "html.parser")
    tables = soup.find_all("table")
    dfs = []
    for t in tables:
        html = str(t)
        try:
            df_list = pd.read_html(html)
            if df_list:
                dfs.append(df_list[0])
        except Exception:
            continue
    return dfs
