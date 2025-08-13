# vision_index.py 渲染 PDF → 页图 → CLIP 向量索引
import argparse, os, json, pathlib, numpy as np, torch
import fitz  # PyMuPDF
from PIL import Image
import open_clip
from tqdm import tqdm

def render_pdf_pages(pdf_path: str, out_dir: str, dpi=170):
    os.makedirs(out_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    paths = []
    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=dpi)
        img_path = os.path.join(out_dir, f"page_{i}.png")
        pix.save(img_path)
        paths.append((i, img_path))
    doc.close()
    return paths

def build_clip(device="cuda", model_name="ViT-L-14", pretrained="openai"):
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()
    return model, preprocess, tokenizer

def encode_images(model, preprocess, image_paths, device="cuda", batch_size=16):
    feats = []
    with torch.no_grad():
        for i in range(0, len(image_paths), batch_size):
            batch = image_paths[i:i+batch_size]
            imgs = [preprocess(Image.open(p).convert("RGB")).unsqueeze(0) for p in batch]
            imgs = torch.cat(imgs, dim=0).to(device)
            f = model.encode_image(imgs)
            f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f.cpu().numpy())
    return np.concatenate(feats, axis=0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf-root", type=str, required=True, help="根目录（递归包含PDF）")
    ap.add_argument("--out-dir", type=str, default="./vision_index")
    ap.add_argument("--dpi", type=int, default=170)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    pdf_root = pathlib.Path(args.pdf_root)
    out_dir = pathlib.Path(args.out_dir)
    img_root = out_dir / "page_images"
    img_root.mkdir(parents=True, exist_ok=True)

    model, preprocess, _ = build_clip(device=args.device)

    meta = []
    img_paths_all = []

    pdf_files = list(pdf_root.rglob("*.pdf"))
    if not pdf_files:
        print(f"未找到PDF: {pdf_root}")
        return

    for pdf in tqdm(pdf_files, desc="渲染PDF为页图"):
        sub = img_root / pdf.stem
        pages = render_pdf_pages(str(pdf), str(sub), dpi=args.dpi)
        for page_idx, img_path in pages:
            meta.append({
                "id": f"{pdf.stem}_page_{page_idx}",
                "file_name": pdf.name,
                "page": page_idx,
                "image_path": img_path
            })
            img_paths_all.append(img_path)

    feats = encode_images(model, preprocess, img_paths_all, device=args.device)
    np.save(out_dir / "feats.npy", feats)
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"完成：{len(meta)} 页，特征 -> {out_dir/'feats.npy'}；元数据 -> {out_dir/'meta.json'}")

if __name__ == "__main__":
    main()
