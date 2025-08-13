# -*- coding: utf-8 -*-
# vision_index_build.py
# 功能：递归扫描 PDF -> 渲染页图 -> 计算 CLIP 图像向量 -> 产出 ./vision_index/
# 依赖：PyMuPDF(fitz), pillow, numpy, torch, open_clip_torch, tqdm
# 构建器（一次性把 PDF 渲染成页图，并预计算图像向量，产出 ./vision_index/ 索引）。
# 你需要的是下面这个建库脚本。跑完后主脚本会自动检测 ./vision_index/，视觉重排就能用；没这个目录则会自动降级为“无视觉重排”。

# 安装依赖
# # 你项目里已有 fitz / torch 就跳过对应行
# pip install pymupdf pillow numpy tqdm
# pip install open_clip_torch  # CLIP 实现
# # 若本机未装 torch，请选与你 CUDA 匹配的版本（示例为 cu121）
# pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
# 你的 GPU 24GB：ViT-L-14 + batch=32 一般稳，OOM 就把 --batch 改小（如 16/8），或者用 --model ViT-B-32。

# 一键构建视觉索引
# # 在项目根目录执行
# python vision_index_build.py \
#   --pdf-root ./datas/财报数据库 \
#   --outdir ./vision_index \
#   --dpi 170 \
#   --model ViT-L-14 \
#   --pretrained openai \
#   --device cuda \
#   --batch 32
# 跑完会得到：

# vision_index/
# ├── images/
# │   ├── 文档Astem/0.png
# │   ├── 文档Astem/1.png
# │   └── ...
# ├── meta.json          # 每行一个：{file_name, page, image_path, doc_stem}
# ├── lookup.json         # "文件名::页码" -> image_path
# ├── clip_features.npy   # (N, D) 归一化图像向量
# ├── id_map.json         # 索引 -> {file_name, page, image_path}
# └── stats.json
# 和主脚本/重排模块的衔接
# 你的 rag_from_page_chunks_stable.py 会在 setup() 里用：

# HybridSearcher(..., vision_index_dir="./vision_index")

# VisionReranker(index_dir="./vision_index", device="cuda")

# 这两个模块会优先读取 meta.json / lookup.json / clip_features.npy：

# 有则直接用（快）；

# 没有也能跑（会临时找图并逐条算向量，慢）。




#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a page-level vision index for PDF documents:
- Render each PDF page to PNG
- (Optional) Compute CLIP image features
- Write: images/, lookup.json, id_map.json, clip_features.npy (optional), meta.json

Usage (online weights):
  python vision_index_build.py --pdf-root ./datas/财报数据库 --outdir ./vision_index \
    --dpi 170 --model ViT-L-14 --pretrained openai --device cuda --batch 32

Usage (offline weights):
  python vision_index_build.py ... --pretrained-path ./models/open_clip/ViT-L-14-openai/open_clip_pytorch_model.bin

Skip embeddings (no internet, quick):
  python vision_index_build.py ... --no-embed
"""
import os
import json
import math
import argparse
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

import fitz  # PyMuPDF
import torch
import open_clip


def render_pdf_to_images(pdf_path: Path, out_dir: Path, dpi: int = 170) -> List[Tuple[int, str]]:
    """
    Render PDF pages to PNG files.

    Returns list of (page_index, image_path).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf_path)
    results = []
    for i in range(doc.page_count):
        out_png = out_dir / f"{i}.png"
        if out_png.exists():
            results.append((i, str(out_png)))
            continue
        page = doc.load_page(i)
        mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        pix.save(out_png)
        results.append((i, str(out_png)))
    doc.close()
    return results


def load_open_clip(model: str, pretrained: str, device: str, pretrained_path: str = None):
    """
    Create OpenCLIP model and preprocess. If pretrained_path is provided, load from local path.
    """
    if pretrained_path and os.path.exists(pretrained_path):
        model, _, preprocess = open_clip.create_model_and_transforms(
            model,
            pretrained=pretrained_path,
            device=device
        )
    else:
        model, _, preprocess = open_clip.create_model_and_transforms(
            model,
            pretrained=pretrained,
            device=device
        )
    model.eval()
    return model, preprocess


def compute_clip_features(image_paths: List[str], model, preprocess, device: str, batch: int = 32) -> np.ndarray:
    feats = []
    with torch.no_grad():
        for i in tqdm(range(0, len(image_paths), batch), desc="计算CLIP特征"):
            batch_paths = image_paths[i:i + batch]
            imgs = []
            for p in batch_paths:
                img = Image.open(p).convert("RGB")
                imgs.append(preprocess(img))
            if not imgs:
                continue
            x = torch.stack(imgs).to(device)
            f = model.encode_image(x)
            f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f.cpu().numpy())
    if not feats:
        return np.zeros((0, 512), dtype=np.float32)
    return np.concatenate(feats, axis=0).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf-root", type=str, required=True, help="Root directory of PDFs (recursively)")
    ap.add_argument("--outdir", type=str, default="./vision_index")
    ap.add_argument("--dpi", type=int, default=170)
    ap.add_argument("--model", type=str, default="ViT-L-14")
    ap.add_argument("--pretrained", type=str, default="openai")
    ap.add_argument("--pretrained-path", type=str, default=None, help="Local .bin/.pt for offline loading")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--no-embed", action="store_true", help="Skip image embedding")
    args = ap.parse_args()

    pdf_root = Path(args.pdf_root)
    outdir = Path(args.outdir)
    images_root = outdir / "images"
    outdir.mkdir(parents=True, exist_ok=True)
    images_root.mkdir(parents=True, exist_ok=True)

    # 1) Render all PDFs
    pdf_files = list(pdf_root.rglob("*.pdf"))
    if not pdf_files:
        print(f"[vision] 未在 {pdf_root} 发现PDF")
        return

    lookup = {}      # "file.pdf::page" -> relative image path (under outdir)
    id_map = {}      # "file.pdf::page" -> integer index (order in features)
    image_paths = [] # absolute image paths ordered by id_map

    idx = 0
    for pdf in tqdm(pdf_files, desc="渲染PDF为页图"):
        stem = pdf.stem
        subdir = images_root / stem
        rendered = render_pdf_to_images(pdf, subdir, dpi=args.dpi)
        for page_idx, png_path in rendered:
            key = f"{pdf.name}::{page_idx}"
            rel_path = os.path.relpath(png_path, outdir)
            lookup[key] = rel_path
            id_map[key] = idx
            image_paths.append(png_path)
            idx += 1

    # 2) Save lookup & id_map
    with open(outdir / "lookup.json", "w", encoding="utf-8") as f:
        json.dump(lookup, f, ensure_ascii=False, indent=2)
    with open(outdir / "id_map.json", "w", encoding="utf-8") as f:
        json.dump(id_map, f, ensure_ascii=False, indent=2)

    # 3) Compute image features (optional)
    features_path = ""
    if not args.no_embed:
        try:
            model, preprocess = load_open_clip(
                model=args.model,
                pretrained=args.pretrained,
                pretrained_path=args.pretrained_path,
                device=args.device
            )
            feats = compute_clip_features(image_paths, model, preprocess, device=args.device, batch=args.batch)
            np.save(outdir / "clip_features.npy", feats)
            features_path = "clip_features.npy"
            print(f"[vision] 已保存特征到: {outdir/'clip_features.npy'}  形状: {feats.shape}")
        except Exception as e:
            print(f"[vision] 计算特征失败，跳过：{e}")
            features_path = ""

    # 4) meta.json（统一供检索/重排使用）
    meta = {
        "images_root": "images",
        "lookup_path": "lookup.json",
        "id_map_path": "id_map.json",
        "features_path": features_path,
        "model": args.model,
        "pretrained": args.pretrained if not args.pretrained_path else args.pretrained_path,
        "dpi": args.dpi,
        "no_embed": bool(args.no_embed),
    }
    with open(outdir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[vision] meta 写入: {outdir/'meta.json'}")


if __name__ == "__main__":
    main()
