# vision_rerank.py 用 CLIP 对候选页做视觉重排
# -*- coding: utf-8 -*-
# vision_rerank.py
# 作用：用 CLIP 文本-图像相似度对候选页进行视觉重排
# 依赖：numpy, torch, open_clip_torch, pillow
# 索引目录结构：由 vision_index_build.py 生成
#   vision_index/
#     ├── images/<doc_stem>/<page>.png
#     ├── clip_features.npy        # (N, D) 归一化图像向量
#     ├── id_map.json              # [{"idx":i,"file_name":"..","page":0,"image_path":".."}, ...]
#     └── lookup.json              # "文件名.pdf::页码" -> image_path

# 重写后的 vision_rerank.py，支持使用你通过 vision_index_build.py 预先构建的索引（clip_features.npy / id_map.json / lookup.json）。如果某个候选没有预计算向量，会自动回退到按需读图并编码；实在找不到图则保留原顺序。接口与主脚本兼容：VisionReranker(index_dir="./vision_index", device="cuda").rerank(query, candidates, top_k=10)。

# 说明要点

# 若 ./vision_index/ 存在 clip_features.npy + id_map.json，会直接用预计算向量，速度快；

# 候选里没带 image_path 时，会用 lookup.json 自动补齐；

# 索引缺失或模型不可用时，自动降级为直通（不重排）；

# 打分字段为 "_vis_score"，你也可以在评估时读取它看排序质量。

# -*- coding: utf-8 -*-
# """
# VisionReranker（JSON/JSONL 兼容版）
# - 优先读取 vision_index/meta.json（找不到则回退 meta.jsonl）
# - 依据 meta 中的 features_path / id_map_path / lookup_path 寻址；若无 meta 则用默认文件名
# - 所有 JSON 支持 JSON / JSONL，两种结构（dict 或 list[dict]）皆可解析
# - 有 clip 特征则用 open_clip 做文本->图像相似度重排；无特征或模型不可用则安全直通
# - 输出候选里补 "_vis_score" 便于评估
# """

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import torch
from PIL import Image

try:
    import open_clip
except Exception:
    open_clip = None


# ---------------- Helpers ----------------

def _device_of(pref: str) -> torch.device:
    if pref.startswith("cuda") and torch.cuda.is_available():
        return torch.device(pref)
    return torch.device("cpu")


def _safe_load_json_or_jsonl(path: Path):
    """健壮加载：支持 JSON / JSONL。JSONL 返回合并或最后一条（视文件结构）"""
    if not path or not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    text_stripped = text.strip()
    if not text_stripped:
        return None
    # 先整体解析
    try:
        return json.loads(text_stripped)
    except json.JSONDecodeError:
        pass
    # 再按 JSONL 逐行解析
    objs = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            objs.append(obj)
        except Exception:
            continue
    if not objs:
        return None
    # 如果每行像是累积日志，取最后一条；如果每行都是条目，返回列表
    if isinstance(objs[-1], dict) and set(objs[-1].keys()) & {
        "images_root", "lookup_path", "id_map_path", "features_path"
    }:
        return objs[-1]
    return objs


def _normalize_lookup(obj, index_dir: Path) -> Dict[str, str]:
    """
    统一为: {"file.pdf::page": "/abs/path/to/image.png"}
    支持:
      - dict: {"f::p": "rel/path.png"}
      - list[dict]: [{"key": "f::p", "path": "rel/path.png"}, ...]
    """
    out = {}
    if obj is None:
        return out

    if isinstance(obj, dict):
        for k, v in obj.items():
            if not isinstance(k, str):
                continue
            if isinstance(v, str):
                p = (index_dir / v).resolve() if not os.path.isabs(v) else Path(v)
                out[k] = str(p)
            else:
                # 若值是对象，尝试取 path 字段
                if isinstance(v, dict) and "path" in v and isinstance(v["path"], str):
                    pv = v["path"]
                    p = (index_dir / pv).resolve() if not os.path.isabs(pv) else Path(pv)
                    out[k] = str(p)
        return out

    if isinstance(obj, list):
        for it in obj:
            if not isinstance(it, dict):
                continue
            key = it.get("key")
            path = it.get("path") or it.get("image_path")
            if isinstance(key, str) and isinstance(path, str):
                p = (index_dir / path).resolve() if not os.path.isabs(path) else Path(path)
                out[key] = str(p)
        return out

    return out


def _normalize_id_map(obj) -> Dict[str, int]:
    """
    统一为: {"file.pdf::page": idx}
    支持:
      - dict: {"f::p": 123, ...}
      - list[dict]: [{"file_name":"xxx.pdf","page":3,"idx":123}, ...]
      - list[list/tuple]: [["f::p", 123], ...] 兼容极端情况
    """
    out: Dict[str, int] = {}
    if obj is None:
        return out

    if isinstance(obj, dict):
        for k, v in obj.items():
            try:
                out[str(k)] = int(v)
            except Exception:
                continue
        return out

    if isinstance(obj, list):
        for it in obj:
            if isinstance(it, dict):
                fn = it.get("file_name")
                pg = it.get("page")
                idx = it.get("idx")
                if fn is not None and pg is not None and idx is not None:
                    try:
                        key = f"{str(fn)}::{int(pg)}"
                        out[key] = int(idx)
                    except Exception:
                        continue
            elif isinstance(it, (list, tuple)) and len(it) == 2:
                k, v = it
                try:
                    out[str(k)] = int(v)
                except Exception:
                    continue
        return out

    return out


def _page_key_from_candidate(c: Dict[str, Any]) -> Optional[str]:
    fn = c.get("file_name") or (c.get("metadata") or {}).get("file_name")
    pg = c.get("page")
    if pg is None and "metadata" in c:
        pg = (c["metadata"] or {}).get("page")
    try:
        if fn is not None and pg is not None:
            return f"{str(fn)}::{int(pg)}"
    except Exception:
        return None
    return None


# ---------------- Reranker ----------------

class VisionReranker:
    def __init__(
        self,
        index_dir: str = "./vision_index",
        device: str = "cuda",
        model_name: str = "ViT-L-14",
        pretrained: str = "openai",
        batch_size: int = 16,
    ):
        """
        index_dir: 视觉索引目录（由 vision_index_build.py 生成）
        device: "cuda" 或 "cpu"
        model_name/pretrained: open_clip 模型配置（若 meta 指定了 model/pretrained，会覆盖）
        batch_size: 视觉回退时的按需批量编码大小（当前逐张也可）
        """
        self.index_dir = Path(index_dir).resolve()
        self.device = _device_of(device)
        self.model_name = model_name
        self.pretrained = pretrained
        self.batch_size = batch_size

        self._ok = False
        self.model = None
        self.preprocess = None

        # 预载索引（若存在）
        self._features = None  # np.ndarray (N, D) or None
        self._key2idx: Dict[str, int] = {}         # "file::page" -> idx
        self._lookup: Dict[str, str] = {}          # "file::page" -> abs image path

        # 缓存按需编码的图像向量，避免重复算
        self._imgfeat_cache: Dict[str, np.ndarray] = {}

        # 先尝试加载 meta.json / meta.jsonl
        meta = self._load_meta(self.index_dir)

        # 解析路径
        features_path = meta.get("features_path") if isinstance(meta, dict) else None
        id_map_path = meta.get("id_map_path") if isinstance(meta, dict) else None
        lookup_path = meta.get("lookup_path") if isinstance(meta, dict) else None

        # 允许 meta 覆盖模型配置
        if isinstance(meta, dict):
            if meta.get("model"):
                self.model_name = meta["model"]
            if meta.get("pretrained"):
                self.pretrained = meta["pretrained"]

        # 默认文件名回退
        feat_file = (self.index_dir / (features_path or "clip_features.npy")).resolve()
        idmap_file = (self.index_dir / (id_map_path or "id_map.json")).resolve()
        lookup_file = (self.index_dir / (lookup_path or "lookup.json")).resolve()

        # 加载 id_map / lookup
        try:
            idmap_obj = _safe_load_json_or_jsonl(idmap_file)
            self._key2idx = _normalize_id_map(idmap_obj)
        except Exception as e:
            print(f"[VisionReranker] 读取 id_map 失败: {e}")

        try:
            lookup_obj = _safe_load_json_or_jsonl(lookup_file)
            self._lookup = _normalize_lookup(lookup_obj, self.index_dir)
        except Exception as e:
            print(f"[VisionReranker] 读取 lookup 失败: {e}")

        # 加载特征（可选）
        if feat_file.exists():
            try:
                self._features = np.load(str(feat_file), mmap_mode="r")
            except Exception as e:
                print(f"[VisionReranker] 读取特征失败: {e}")
                self._features = None
        else:
            self._features = None  # 没有就直通

        # 初始化 open_clip（只有需要用到时才重要）
        self._init_model()

        if self.model is None and self._features is None:
            # 没模型也没特征，完全直通
            print("[VisionReranker] 无模型 & 无特征，直通不重排。")
            self._ok = False
        else:
            self._ok = True

    # ---------- meta 读取 ----------
    @staticmethod
    def _load_meta(index_dir: Path) -> dict:
        meta_json = index_dir / "meta.json"
        meta_jsonl = index_dir / "meta.jsonl"
        meta = {}
        if meta_json.exists():
            try:
                obj = _safe_load_json_or_jsonl(meta_json)
                if isinstance(obj, dict):
                    meta = obj
            except Exception:
                pass
        elif meta_jsonl.exists():
            try:
                obj = _safe_load_json_or_jsonl(meta_jsonl)
                if isinstance(obj, dict):
                    meta = obj
            except Exception:
                pass
        return meta

    # ---------- 公共方法 ----------
    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int = 10) -> List[Dict[str, Any]]:
        """
        输入：
          query: 文本查询
          candidates: 候选列表（需要包含 file_name, page，可选 image_path）
          top_k: 返回前 k 个重排结果
        输出：
          按 '_vis_score' 由高到低排序后的候选（副本），并尽可能补充 image_path
        """
        if not candidates:
            return candidates

        # 如果模型/特征均不可用，直接返回前 top_k
        if not self._ok:
            return candidates[:max(1, top_k)]

        # 1) 文本编码（有特征或需要按需编码时才用）
        q_vec = None
        if self.model is not None or self._features is not None:
            try:
                q_vec = self._encode_text(query)  # np.ndarray (D,)
            except Exception as e:
                print(f"[VisionReranker] 文本编码失败：{e}，降级直通")
                return candidates[:max(1, top_k)]

        # 2) 为候选获取图像向量（优先用预计算；否则按需编码），并计算相似度
        rescored: List[Dict[str, Any]] = []
        for c in candidates:
            fn = str(c.get("file_name", "")).strip()
            try:
                pg = int(c.get("page", -1))
            except Exception:
                pg = -1

            feat = None
            # 先试预计算向量
            if self._features is not None and self._key2idx:
                feat = self._get_precomputed_feat(fn, pg)

            # 如果没有预计算，尝试找到 image_path 并按需编码
            img_path = c.get("image_path")
            if feat is None:
                key = f"{fn}::{pg}"
                if not img_path and key in self._lookup:
                    img_path = self._lookup[key]
                if img_path and os.path.exists(img_path) and self.model is not None:
                    feat = self._encode_image_path_cached(img_path)

            # 计算分数
            score = float(np.dot(q_vec, feat)) if (q_vec is not None and feat is not None) else -1e9

            c_new = dict(c)
            if img_path and "image_path" not in c_new:
                c_new["image_path"] = img_path
            c_new["_vis_score"] = score
            rescored.append(c_new)

        # 3) 按分数排序
        rescored.sort(key=lambda x: x.get("_vis_score", -1e9), reverse=True)
        return rescored[:max(1, top_k)]

    # ---------- 模型与编码 ----------
    def _init_model(self):
        if open_clip is None:
            return
        # 如果只想用预计算特征，可以不加载模型；但为了按需编码兜底，这里尝试加载
        try:
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name=self.model_name,
                pretrained=self.pretrained,
                device=self.device,
            )
            self.model.eval()
        except Exception as e:
            print(f"[VisionReranker] 创建 open_clip 模型失败：{e}")
            self.model = None
            self.preprocess = None

    @torch.no_grad()
    def _encode_text(self, text: str) -> np.ndarray:
        # 没模型时不该调用（上游做了判断）
        assert self.model is not None and self.preprocess is not None
        tok = open_clip.get_tokenizer(self.model_name)
        tokens = tok([text])
        tokens = tokens.to(self.device)
        feat = self.model.encode_text(tokens)
        feat = feat.float()
        feat = feat / (feat.norm(dim=-1, keepdim=True) + 1e-12)
        return feat[0].detach().cpu().numpy()

    @torch.no_grad()
    def _encode_image_path_cached(self, path: str) -> Optional[np.ndarray]:
        if path in self._imgfeat_cache:
            return self._imgfeat_cache[path]
        if self.model is None or self.preprocess is None:
            return None
        try:
            im = Image.open(path).convert("RGB")
        except Exception:
            return None
        im = self.preprocess(im).unsqueeze(0).to(self.device)
        feat = self.model.encode_image(im)
        feat = feat.float()
        feat = feat / (feat.norm(dim=-1, keepdim=True) + 1e-12)
        vec = feat[0].detach().cpu().numpy().astype(np.float32)
        self._imgfeat_cache[path] = vec
        return vec

    def _get_precomputed_feat(self, file_name: str, page: int) -> Optional[np.ndarray]:
        if self._features is None or not self._key2idx:
            return None
        key = f"{file_name}::{page}"
        idx = self._key2idx.get(key, -1)
        if idx < 0:
            return None
        try:
            vec = np.array(self._features[idx], dtype=np.float32)
            n = float(np.linalg.norm(vec) + 1e-12)
            return (vec / n).astype(np.float32)
        except Exception:
            return None

