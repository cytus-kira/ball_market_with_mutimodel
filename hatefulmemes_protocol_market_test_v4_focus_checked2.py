#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hateful Memes: direct training on raw multimodal features vs optimized
training on a log-guided best granular-ball configuration with fast defaults.

This version bakes in the strongest region observed in the uploaded logs as the
fast default: pca=64, purity=0.92, min_ball_size=8,
repr_mode=purity_gated_hybrid, adaptive_accept=False. It also keeps the
accuracy-oriented model selection and dev-threshold tuning introduced in v5.

Typical usage
-------------
python hatefulmemes_ball_vs_direct_v6_best.py
"""

from __future__ import annotations

import argparse
import itertools
import hashlib
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple, Any

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def l2_normalize_np(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(denom, eps)


def json_dump(obj: Dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def parse_int_csv(text: str) -> List[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def parse_float_csv(text: str) -> List[float]:
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def parse_str_csv(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def softmax_np(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / np.maximum(ex.sum(), 1e-12)


def sharpen_probs(p: np.ndarray, power: float) -> np.ndarray:
    if power <= 1.0:
        return p
    q = np.power(np.clip(p, 1e-8, 1.0), power)
    q = q / np.maximum(q.sum(), 1e-8)
    return q.astype(np.float32)


# -----------------------------------------------------------------------------
# Dataset discovery and loading
# -----------------------------------------------------------------------------

TRAIN_CANDIDATES = ["train.jsonl", "train_seen.jsonl", "train.csv"]
DEV_CANDIDATES = ["dev_seen.jsonl", "dev.jsonl", "val.jsonl", "dev_seen.csv", "dev.csv", "val.csv"]
TEST_CANDIDATES = ["test_seen.jsonl", "test.jsonl", "test_seen.csv", "test.csv"]
IMAGE_DIR_CANDIDATES = ["img", "imgs", "images", "data", "."]


def find_first_existing(root: Path, candidates: Sequence[str]) -> Optional[Path]:
    for name in candidates:
        p = root / name
        if p.exists():
            return p
    return None


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported metadata format: {path}")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    cols = set(df.columns)
    if "sentence" in cols and "text" not in cols:
        rename_map["sentence"] = "text"
    if "image" in cols and "img" not in cols:
        rename_map["image"] = "img"
    if "gold_label" in cols and "label" not in cols:
        rename_map["gold_label"] = "label"
    if rename_map:
        df = df.rename(columns=rename_map)
    for req in ["text", "img"]:
        if req not in df.columns:
            raise ValueError(f"Required column '{req}' not found. Columns={list(df.columns)}")
    return df


def discover_splits(data_root: Path) -> Dict[str, Optional[Path]]:
    train_path = find_first_existing(data_root, TRAIN_CANDIDATES)
    dev_path = find_first_existing(data_root, DEV_CANDIDATES)
    test_path = find_first_existing(data_root, TEST_CANDIDATES)
    if train_path is None:
        raise FileNotFoundError(f"Could not find a train split in {data_root}. Tried: {TRAIN_CANDIDATES}")
    if dev_path is None:
        raise FileNotFoundError(f"Could not find a dev split in {data_root}. Tried: {DEV_CANDIDATES}")
    return {"train": train_path, "dev": dev_path, "test": test_path}


def resolve_image_path(data_root: Path, rel_path: str) -> Path:
    rel_path = str(rel_path)
    direct = data_root / rel_path
    if direct.exists():
        return direct
    for d in IMAGE_DIR_CANDIDATES:
        cand = data_root / d / rel_path
        if cand.exists():
            return cand
    rel_path2 = rel_path.lstrip("./")
    direct2 = data_root / rel_path2
    if direct2.exists():
        return direct2
    raise FileNotFoundError(f"Image not found for relative path: {rel_path}")


# -----------------------------------------------------------------------------
# CLIP feature extraction
# -----------------------------------------------------------------------------

class HatefulMemesFeatureDataset(Dataset):
    def __init__(self, df: pd.DataFrame, data_root: Path):
        self.df = df.reset_index(drop=True)
        self.data_root = data_root

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = resolve_image_path(self.data_root, row["img"])
        image = Image.open(img_path).convert("RGB")
        text = str(row["text"])
        label = row["label"] if "label" in row and not pd.isna(row["label"]) else -1
        return image, text, int(label)


class CLIPFeatureExtractor:
    def __init__(self, model_name: str, device: str):
        try:
            from transformers import AutoProcessor, CLIPModel
        except Exception as exc:
            raise RuntimeError(
                "This script requires transformers with CLIP support. Please install transformers."
            ) from exc

        self.device = torch.device(device)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode_batch(self, images: Sequence[Image.Image], texts: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
        inputs = self.processor(text=list(texts), images=list(images), return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        image_features = self.model.get_image_features(pixel_values=inputs["pixel_values"])
        text_features = self.model.get_text_features(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        return (
            image_features.detach().cpu().numpy().astype(np.float32),
            text_features.detach().cpu().numpy().astype(np.float32),
        )


def collate_images_texts(batch):
    images, texts, labels = zip(*batch)
    return list(images), list(texts), torch.tensor(labels, dtype=torch.long)


def make_fused_features(img_feats: np.ndarray, txt_feats: np.ndarray, fusion: str) -> np.ndarray:
    if fusion == "concat":
        fused = np.concatenate([img_feats, txt_feats], axis=1)
    elif fusion == "concat_mul_absdiff":
        fused = np.concatenate([img_feats, txt_feats, img_feats * txt_feats, np.abs(img_feats - txt_feats)], axis=1)
    else:
        raise ValueError(f"Unknown fusion mode: {fusion}")
    return fused.astype(np.float32)


def extract_or_load_features(
    split_name: str,
    df: pd.DataFrame,
    data_root: Path,
    cache_dir: Path,
    model_name: str,
    fusion: str,
    batch_size: int,
    num_workers: int,
    device: str,
) -> Dict[str, np.ndarray]:
    ensure_dir(cache_dir)
    safe_model_name = model_name.replace("/", "__")
    cache_path = cache_dir / f"{split_name}_{safe_model_name}_{fusion}.npz"

    if cache_path.exists():
        pack = np.load(cache_path, allow_pickle=True)
        return {"X": pack["X"], "y": pack["y"], "img": pack["img"], "txt": pack["txt"]}

    dataset = HatefulMemesFeatureDataset(df, data_root)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_images_texts,
        pin_memory=(device.startswith("cuda")),
    )

    extractor = CLIPFeatureExtractor(model_name=model_name, device=device)
    all_img, all_txt, all_y = [], [], []
    start = time.time()
    for images, texts, labels in loader:
        img_f, txt_f = extractor.encode_batch(images, texts)
        all_img.append(img_f)
        all_txt.append(txt_f)
        all_y.append(labels.numpy())

    img = np.concatenate(all_img, axis=0)
    txt = np.concatenate(all_txt, axis=0)
    y = np.concatenate(all_y, axis=0).astype(np.int64)
    X = make_fused_features(img, txt, fusion)
    X = l2_normalize_np(X)
    np.savez_compressed(cache_path, X=X, y=y, img=img, txt=txt)
    elapsed = time.time() - start
    print(f"[feature-cache] wrote {cache_path} in {elapsed:.1f}s")
    return {"X": X, "y": y, "img": img, "txt": txt}


# -----------------------------------------------------------------------------
# Granular-ball generation
# -----------------------------------------------------------------------------

@dataclass
class Ball:
    indices: np.ndarray
    center: np.ndarray     # center in partition space
    radius: float          # radius in partition space
    purity: float
    majority_label: int
    label_hist: np.ndarray


def compute_ball(X: np.ndarray, y: np.ndarray, indices: np.ndarray, num_classes: int) -> Ball:
    xs = X[indices]
    ys = y[indices]
    center = xs.mean(axis=0)
    d = np.linalg.norm(xs - center[None, :], axis=1)
    radius = float(d.max()) if len(d) else 0.0
    hist = np.bincount(ys, minlength=num_classes).astype(np.float32)
    majority_label = int(hist.argmax())
    purity = float(hist[majority_label] / max(len(indices), 1))
    return Ball(
        indices=indices,
        center=center.astype(np.float32),
        radius=radius,
        purity=purity,
        majority_label=majority_label,
        label_hist=hist,
    )


def representative_seed(points: np.ndarray, point_indices: np.ndarray) -> Tuple[np.ndarray, int]:
    center = points.mean(axis=0)
    dist = np.linalg.norm(points - center[None, :], axis=1)
    local_idx = int(np.argmin(dist))
    return points[local_idx], int(point_indices[local_idx])


def weighted_purity_sum(children: List[np.ndarray], y: np.ndarray, num_classes: int) -> float:
    tot = sum(len(c) for c in children)
    if tot <= 0:
        return 0.0
    s = 0.0
    for ids in children:
        hist = np.bincount(y[ids], minlength=num_classes)
        s += float(hist.max())
    return s / float(tot)


def split_ball_acceleration_style(
    X: np.ndarray,
    y: np.ndarray,
    indices: np.ndarray,
    num_classes: int,
) -> Optional[List[np.ndarray]]:
    ys = y[indices]
    unique_labels = np.unique(ys)
    k = len(unique_labels)
    if k <= 1:
        return None

    centers = []
    for lab in unique_labels:
        lab_ids = indices[ys == lab]
        lab_points = X[lab_ids]
        seed_vec, _ = representative_seed(lab_points, lab_ids)
        centers.append(seed_vec)

    centers = np.stack(centers, axis=0).astype(np.float32)
    sub_x = X[indices]
    dist = np.linalg.norm(sub_x[:, None, :] - centers[None, :, :], axis=2)
    assign = np.argmin(dist, axis=1)

    children = []
    for cid in range(len(centers)):
        child_ids = indices[assign == cid]
        if len(child_ids) == 0:
            return None
        children.append(child_ids)

    if len(children) <= 1:
        return None
    return children


def global_refine_once(X: np.ndarray, y: np.ndarray, balls: List[Ball], num_classes: int) -> List[Ball]:
    if len(balls) <= 1:
        return balls
    centers = np.stack([b.center for b in balls], axis=0)
    dist = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
    assign = np.argmin(dist, axis=1)
    out = []
    for i in range(len(balls)):
        ids = np.where(assign == i)[0]
        if len(ids) == 0:
            continue
        out.append(compute_ball(X, y, ids.astype(np.int64), num_classes=num_classes))
    return out


def build_granular_balls(
    X: np.ndarray,
    y: np.ndarray,
    purity_threshold: float,
    min_ball_size: int,
    max_depth: int,
    apply_global_refine: bool = True,
    use_adaptive_accept: bool = False,
    enforce_initial_purity_lower_bound: bool = False,
) -> List[Ball]:
    num_classes = int(y.max()) + 1
    queue: List[Tuple[np.ndarray, int]] = [(np.arange(len(X), dtype=np.int64), 0)]
    final_balls: List[Ball] = []
    initial_purity = compute_ball(X, y, np.arange(len(X), dtype=np.int64), num_classes=num_classes).purity

    while queue:
        indices, depth = queue.pop(0)
        ball = compute_ball(X, y, indices, num_classes=num_classes)

        should_attempt_split = (
            ball.purity < purity_threshold
            and len(indices) >= 2
            and depth < max_depth
            and np.count_nonzero(ball.label_hist) > 1
        )
        if not should_attempt_split:
            final_balls.append(ball)
            continue

        children = split_ball_acceleration_style(X=X, y=y, indices=indices, num_classes=num_classes)
        if children is None:
            final_balls.append(ball)
            continue

        if use_adaptive_accept:
            W = weighted_purity_sum(children, y=y, num_classes=num_classes)
            keep_split = (W > ball.purity + 1e-12)
            if enforce_initial_purity_lower_bound:
                child_purities = [compute_ball(X, y, ids, num_classes).purity for ids in children]
                keep_split = keep_split and all(p >= initial_purity - 1e-12 for p in child_purities)
            if not keep_split:
                final_balls.append(ball)
                continue

        for ids in children:
            if len(ids) < min_ball_size:
                final_balls.append(compute_ball(X, y, ids, num_classes=num_classes))
            else:
                queue.append((ids, depth + 1))

    if apply_global_refine:
        final_balls = global_refine_once(X, y, final_balls, num_classes=num_classes)
    return final_balls


def summarize_balls(balls: List[Ball], n_samples: int) -> Dict[str, float]:
    sizes = np.array([len(b.indices) for b in balls], dtype=np.float32)
    purities = np.array([b.purity for b in balls], dtype=np.float32)
    radii = np.array([b.radius for b in balls], dtype=np.float32)
    return {
        "num_balls": int(len(balls)),
        "compression_ratio": float(n_samples / max(len(balls), 1)),
        "avg_ball_size": float(sizes.mean()) if len(sizes) else 0.0,
        "median_ball_size": float(np.median(sizes)) if len(sizes) else 0.0,
        "max_ball_size": float(sizes.max()) if len(sizes) else 0.0,
        "avg_purity": float(purities.mean()) if len(purities) else 0.0,
        "median_purity": float(np.median(purities)) if len(purities) else 0.0,
        "min_purity": float(purities.min()) if len(purities) else 0.0,
        "max_purity": float(purities.max()) if len(purities) else 0.0,
        "avg_radius": float(radii.mean()) if len(radii) else 0.0,
        "median_radius": float(np.median(radii)) if len(radii) else 0.0,
    }


def ball_weight_value(ball: Ball, weight_mode: str) -> float:
    if weight_mode == "count":
        return float(len(ball.indices))
    if weight_mode == "count_purity":
        return float(len(ball.indices) * ball.purity)
    if weight_mode == "count_purity2":
        return float(len(ball.indices) * (ball.purity ** 2))
    if weight_mode == "uniform":
        return 1.0
    raise ValueError(f"Unknown weight_mode: {weight_mode}")


def balls_to_center_soft_training_set(
    balls: List[Ball],
    X_source: np.ndarray,
    y_full: np.ndarray,
    weight_mode: str,
    label_sharpen_power: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    Xb, Yb, wb = [], [], []
    num_classes = int(y_full.max()) + 1
    for b in balls:
        center = X_source[b.indices].mean(axis=0).astype(np.float32)
        y_soft = (b.label_hist / max(b.label_hist.sum(), 1.0)).astype(np.float32)
        y_soft = sharpen_probs(y_soft, label_sharpen_power)
        Xb.append(center)
        Yb.append(y_soft)
        wb.append(ball_weight_value(b, weight_mode))
    Xb = np.stack(Xb, axis=0).astype(np.float32)
    Yb = np.stack(Yb, axis=0).astype(np.float32)
    wb = np.array(wb, dtype=np.float32)
    return Xb, Yb, wb


def unique_preserve_order(items: List[int]) -> List[int]:
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def select_gbs_boundary_ids(
    ball: Ball,
    X_source: np.ndarray,
    y_full: np.ndarray,
    boundary_axes: int,
) -> List[int]:
    ids_all = ball.indices
    ids_majority = ids_all[y_full[ids_all] == ball.majority_label]
    base_ids = ids_majority if len(ids_majority) > 0 else ids_all
    xs = X_source[base_ids]
    if len(base_ids) <= 1:
        return list(map(int, base_ids.tolist()))

    if len(base_ids) <= 2 * X_source.shape[1]:
        return list(map(int, base_ids.tolist()))

    var = xs.var(axis=0)
    topk = min(boundary_axes, xs.shape[1])
    axis_ids = np.argpartition(var, -topk)[-topk:]

    chosen = []
    for ax in axis_ids.tolist():
        chosen.append(int(base_ids[int(np.argmax(xs[:, ax]))]))
        chosen.append(int(base_ids[int(np.argmin(xs[:, ax]))]))
    return unique_preserve_order(chosen)


def balls_to_gbs_training_set(
    balls: List[Ball],
    X_source: np.ndarray,
    y_full: np.ndarray,
    weight_mode: str,
    boundary_axes: int,
    include_center: bool,
    center_weight_frac: float,
    label_sharpen_power: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    Xr, Yr, Wr = [], [], []
    num_classes = int(y_full.max()) + 1

    for b in balls:
        total_w = ball_weight_value(b, weight_mode)
        boundary_ids = select_gbs_boundary_ids(b, X_source=X_source, y_full=y_full, boundary_axes=boundary_axes)
        n_boundary = len(boundary_ids)
        center_vec = X_source[b.indices].mean(axis=0).astype(np.float32)
        center_soft = (b.label_hist / max(b.label_hist.sum(), 1.0)).astype(np.float32)
        center_soft = sharpen_probs(center_soft, label_sharpen_power)

        if include_center:
            if n_boundary > 0:
                w_center = total_w * center_weight_frac
                w_each = total_w * (1.0 - center_weight_frac) / max(n_boundary, 1)
            else:
                w_center = total_w
                w_each = 0.0
            Xr.append(center_vec)
            Yr.append(center_soft)
            Wr.append(w_center)
        else:
            w_each = total_w / max(n_boundary, 1)

        onehot = np.zeros(num_classes, dtype=np.float32)
        onehot[b.majority_label] = 1.0
        for sid in boundary_ids:
            Xr.append(X_source[sid].astype(np.float32))
            Yr.append(onehot)
            Wr.append(w_each)

    Xr = np.stack(Xr, axis=0).astype(np.float32)
    Yr = np.stack(Yr, axis=0).astype(np.float32)
    Wr = np.array(Wr, dtype=np.float32)
    return Xr, Yr, Wr


# -----------------------------------------------------------------------------
# Classifier and training
# -----------------------------------------------------------------------------

class HardLabelDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, w: Optional[np.ndarray] = None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.w = torch.tensor(w if w is not None else np.ones(len(X), dtype=np.float32), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx], self.w[idx]


class SoftLabelDataset(Dataset):
    def __init__(self, X: np.ndarray, y_soft: np.ndarray, w: Optional[np.ndarray] = None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y_soft, dtype=torch.float32)
        self.w = torch.tensor(w if w is not None else np.ones(len(X), dtype=np.float32), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx], self.w[idx]


class MLPClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 512, dropout: float = 0.2, num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@torch.no_grad()
def predict_model(model: nn.Module, X: np.ndarray, device: str, batch_size: int = 256) -> Dict[str, np.ndarray]:
    model.eval()
    device_t = torch.device(device)
    out_logits = []
    for start in range(0, len(X), batch_size):
        xb = torch.tensor(X[start:start + batch_size], dtype=torch.float32, device=device_t)
        logits = model(xb)
        out_logits.append(logits.detach().cpu())
    logits = torch.cat(out_logits, dim=0).numpy()
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
    pred = probs.argmax(axis=1)
    return {"logits": logits, "probs": probs, "pred": pred}


def compute_metrics(y_true: np.ndarray, probs: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro")),
    }
    if len(np.unique(y_true)) == 2:
        try:
            metrics["auroc"] = float(roc_auc_score(y_true, probs[:, 1]))
        except Exception:
            metrics["auroc"] = float("nan")
    else:
        metrics["auroc"] = float("nan")
    return metrics


def probs_to_pred(probs: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    if probs.ndim == 2 and probs.shape[1] == 2:
        return (probs[:, 1] >= threshold).astype(np.int64)
    return probs.argmax(axis=1).astype(np.int64)


def select_score(metrics: Dict[str, float], metric_name: str) -> float:
    if metric_name == "accuracy":
        return float(metrics["accuracy"])
    if metric_name == "macro_f1":
        return float(metrics["macro_f1"])
    if metric_name == "auroc":
        v = float(metrics.get("auroc", float("nan")))
        return -1.0 if math.isnan(v) else v
    if metric_name == "acc_f1_mean":
        return 0.5 * float(metrics["accuracy"]) + 0.5 * float(metrics["macro_f1"])
    raise ValueError(f"Unknown metric_name: {metric_name}")


def find_best_binary_threshold(y_true: np.ndarray, probs: np.ndarray, metric_name: str) -> Tuple[float, Dict[str, float]]:
    if not (probs.ndim == 2 and probs.shape[1] == 2):
        pred = probs.argmax(axis=1)
        metrics = compute_metrics(y_true, probs, pred)
        return 0.5, metrics

    candidate_thresholds = np.unique(np.concatenate([
        np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], dtype=np.float32),
        probs[:, 1].astype(np.float32),
    ]))
    best_t = 0.5
    best_metrics = None
    best_score = -1.0
    for t in candidate_thresholds.tolist():
        pred = (probs[:, 1] >= float(t)).astype(np.int64)
        metrics = compute_metrics(y_true, probs, pred)
        score = select_score(metrics, metric_name=metric_name)
        if score > best_score + 1e-12:
            best_score = score
            best_t = float(t)
            best_metrics = metrics
    assert best_metrics is not None
    return best_t, best_metrics


def compute_soft_class_weights(y_soft_train: np.ndarray, weights: Optional[np.ndarray]) -> np.ndarray:
    w = np.ones(len(y_soft_train), dtype=np.float32) if weights is None else weights.astype(np.float32)
    class_mass = (y_soft_train * w[:, None]).sum(axis=0)
    class_mass = np.maximum(class_mass, 1e-6)
    total = float(class_mass.sum())
    num_classes = len(class_mass)
    class_weight = total / (num_classes * class_mass)
    class_weight = class_weight / np.mean(class_weight)
    return class_weight.astype(np.float32)


def train_classifier_hard(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_dev: np.ndarray,
    y_dev: np.ndarray,
    train_weights: Optional[np.ndarray],
    device: str,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    hidden_dim: int,
    dropout: float,
    patience: int,
    class_balance: bool,
    selection_metric: str,
    tune_decision_threshold: bool,
) -> Tuple[nn.Module, Dict[str, float], float, float]:
    device_t = torch.device(device)
    num_classes = int(np.max(y_train)) + 1
    model = MLPClassifier(in_dim=X_train.shape[1], hidden_dim=hidden_dim, dropout=dropout, num_classes=num_classes).to(device_t)
    dataset = HardLabelDataset(X_train, y_train, train_weights)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    ce_weight = None
    if class_balance:
        hist = np.bincount(y_train, minlength=num_classes).astype(np.float32)
        hist = np.maximum(hist, 1e-6)
        ce_weight = (hist.sum() / (num_classes * hist)).astype(np.float32)
        ce_weight = ce_weight / np.mean(ce_weight)
        ce_weight = torch.tensor(ce_weight, dtype=torch.float32, device=device_t)

    best_state = None
    best_metrics = None
    best_threshold = 0.5
    best_score = -1.0
    bad_epochs = 0
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, total_weight = 0.0, 0.0
        for xb, yb, wb in loader:
            xb, yb, wb = xb.to(device_t), yb.to(device_t), wb.to(device_t)
            logits = model(xb)
            per_sample = F.cross_entropy(logits, yb, weight=ce_weight, reduction="none")
            loss = (per_sample * wb).sum() / torch.clamp(wb.sum(), min=1e-6)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running_loss += float((per_sample * wb).sum().item())
            total_weight += float(wb.sum().item())

        dev_pred = predict_model(model, X_dev, device=device)
        if tune_decision_threshold and dev_pred["probs"].shape[1] == 2:
            dev_threshold, dev_metrics = find_best_binary_threshold(y_dev, dev_pred["probs"], metric_name=selection_metric)
        else:
            dev_threshold = 0.5
            dev_metrics = compute_metrics(y_dev, dev_pred["probs"], dev_pred["pred"])
        score = select_score(dev_metrics, metric_name=selection_metric)
        epoch_loss = running_loss / max(total_weight, 1e-6)
        print(f"[train-direct] epoch={epoch:03d} loss={epoch_loss:.4f} dev_acc={dev_metrics['accuracy']:.4f} dev_f1={dev_metrics['macro_f1']:.4f} dev_auroc={dev_metrics['auroc']:.4f}")

        if score > best_score:
            best_score = score
            best_metrics = dev_metrics
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_threshold = float(dev_threshold)
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    assert best_state is not None and best_metrics is not None
    model.load_state_dict(best_state)
    return model, best_metrics, time.time() - t0, best_threshold


def train_classifier_soft(
    X_train: np.ndarray,
    y_soft_train: np.ndarray,
    X_dev: np.ndarray,
    y_dev: np.ndarray,
    train_weights: Optional[np.ndarray],
    device: str,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    hidden_dim: int,
    dropout: float,
    patience: int,
    class_balance: bool,
    log_prefix: str,
    selection_metric: str,
    tune_decision_threshold: bool,
) -> Tuple[nn.Module, Dict[str, float], float, float]:
    device_t = torch.device(device)
    num_classes = y_soft_train.shape[1]
    model = MLPClassifier(in_dim=X_train.shape[1], hidden_dim=hidden_dim, dropout=dropout, num_classes=num_classes).to(device_t)
    dataset = SoftLabelDataset(X_train, y_soft_train, train_weights)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    class_weight = None
    if class_balance:
        class_weight = torch.tensor(
            compute_soft_class_weights(y_soft_train, train_weights),
            dtype=torch.float32,
            device=device_t,
        )

    best_state = None
    best_metrics = None
    best_threshold = 0.5
    best_score = -1.0
    bad_epochs = 0
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, total_weight = 0.0, 0.0
        for xb, yb_soft, wb in loader:
            xb, yb_soft, wb = xb.to(device_t), yb_soft.to(device_t), wb.to(device_t)
            logits = model(xb)
            log_probs = F.log_softmax(logits, dim=1)
            per_sample = -(yb_soft * log_probs).sum(dim=1)
            if class_weight is not None:
                sample_cw = (yb_soft * class_weight[None, :]).sum(dim=1)
                per_sample = per_sample * sample_cw
            loss = (per_sample * wb).sum() / torch.clamp(wb.sum(), min=1e-6)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running_loss += float((per_sample * wb).sum().item())
            total_weight += float(wb.sum().item())

        dev_pred = predict_model(model, X_dev, device=device)
        if tune_decision_threshold and dev_pred["probs"].shape[1] == 2:
            dev_threshold, dev_metrics = find_best_binary_threshold(y_dev, dev_pred["probs"], metric_name=selection_metric)
        else:
            dev_threshold = 0.5
            dev_metrics = compute_metrics(y_dev, dev_pred["probs"], dev_pred["pred"])
        score = select_score(dev_metrics, metric_name=selection_metric)
        epoch_loss = running_loss / max(total_weight, 1e-6)
        print(f"[{log_prefix}] epoch={epoch:03d} loss={epoch_loss:.4f} dev_acc={dev_metrics['accuracy']:.4f} dev_f1={dev_metrics['macro_f1']:.4f} dev_auroc={dev_metrics['auroc']:.4f}")

        if score > best_score:
            best_score = score
            best_metrics = dev_metrics
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_threshold = float(dev_threshold)
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    assert best_state is not None and best_metrics is not None
    model.load_state_dict(best_state)
    return model, best_metrics, time.time() - t0, best_threshold


# -----------------------------------------------------------------------------
# Ball branch evaluation / search
# -----------------------------------------------------------------------------


def balls_to_purity_gated_training_set(
    balls: List[Ball],
    X_source: np.ndarray,
    y_full: np.ndarray,
    weight_mode: str,
    boundary_axes: int,
    center_weight_frac: float,
    label_sharpen_power: float,
    low_purity_threshold: float,
    low_purity_raw_keep: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    Xr, Yr, Wr = [], [], []
    num_classes = int(y_full.max()) + 1

    for b in balls:
        total_w = ball_weight_value(b, weight_mode)
        ids_all = b.indices.astype(np.int64)
        if b.purity >= low_purity_threshold and len(ids_all) > low_purity_raw_keep:
            boundary_ids = select_gbs_boundary_ids(b, X_source=X_source, y_full=y_full, boundary_axes=boundary_axes)
            n_boundary = len(boundary_ids)
            center_vec = X_source[ids_all].mean(axis=0).astype(np.float32)
            center_soft = (b.label_hist / max(b.label_hist.sum(), 1.0)).astype(np.float32)
            center_soft = sharpen_probs(center_soft, label_sharpen_power)
            w_center = total_w * center_weight_frac
            w_each = total_w * (1.0 - center_weight_frac) / max(n_boundary, 1)
            Xr.append(center_vec)
            Yr.append(center_soft)
            Wr.append(w_center)
            for sid in boundary_ids:
                onehot = np.zeros(num_classes, dtype=np.float32)
                onehot[int(y_full[sid])] = 1.0
                Xr.append(X_source[sid].astype(np.float32))
                Yr.append(onehot)
                Wr.append(w_each)
        else:
            ids_list = ids_all.tolist()
            if len(ids_list) > low_purity_raw_keep:
                by_label = {}
                for sid in ids_list:
                    by_label.setdefault(int(y_full[sid]), []).append(int(sid))
                chosen = []
                per_label_quota = max(1, low_purity_raw_keep // max(len(by_label), 1))
                for _, sid_list in sorted(by_label.items()):
                    xs = X_source[np.array(sid_list, dtype=np.int64)]
                    center = xs.mean(axis=0)
                    dist = np.linalg.norm(xs - center[None, :], axis=1)
                    order = np.argsort(-dist)
                    chosen.extend([sid_list[int(i)] for i in order[:per_label_quota]])
                seen = set(chosen)
                for sid in ids_list:
                    if len(chosen) >= low_purity_raw_keep:
                        break
                    if sid not in seen:
                        chosen.append(sid)
                        seen.add(sid)
                ids_list = chosen[:low_purity_raw_keep]
            w_each = total_w / max(len(ids_list), 1)
            for sid in ids_list:
                onehot = np.zeros(num_classes, dtype=np.float32)
                onehot[int(y_full[sid])] = 1.0
                Xr.append(X_source[int(sid)].astype(np.float32))
                Yr.append(onehot)
                Wr.append(w_each)

    return np.stack(Xr, axis=0).astype(np.float32), np.stack(Yr, axis=0).astype(np.float32), np.array(Wr, dtype=np.float32)

def fit_pca_cache(X_train: np.ndarray, X_dev: np.ndarray, X_test: Optional[np.ndarray], dims: List[int], seed: int):
    cache: Dict[int, Dict[str, Optional[np.ndarray]]] = {}
    for dim in sorted(set(dims)):
        if dim <= 0 or dim >= X_train.shape[1]:
            cache[dim] = {"train": X_train, "dev": X_dev, "test": X_test, "pca": None}
        else:
            pca = PCA(n_components=dim, random_state=seed)
            cache[dim] = {
                "train": pca.fit_transform(X_train).astype(np.float32),
                "dev": pca.transform(X_dev).astype(np.float32),
                "test": pca.transform(X_test).astype(np.float32) if X_test is not None else None,
                "pca": pca,
            }
    return cache


def build_training_set_from_balls(
    balls: List[Ball],
    X_source: np.ndarray,
    y_full: np.ndarray,
    repr_mode: str,
    weight_mode: str,
    boundary_axes: int,
    center_weight_frac: float,
    label_sharpen_power: float,
    low_purity_threshold: float,
    low_purity_raw_keep: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if repr_mode == "center_soft":
        return balls_to_center_soft_training_set(
            balls=balls,
            X_source=X_source,
            y_full=y_full,
            weight_mode=weight_mode,
            label_sharpen_power=label_sharpen_power,
        )
    if repr_mode == "gbs_only":
        return balls_to_gbs_training_set(
            balls=balls,
            X_source=X_source,
            y_full=y_full,
            weight_mode=weight_mode,
            boundary_axes=boundary_axes,
            include_center=False,
            center_weight_frac=center_weight_frac,
            label_sharpen_power=label_sharpen_power,
        )
    if repr_mode == "hybrid_gbs":
        return balls_to_gbs_training_set(
            balls=balls,
            X_source=X_source,
            y_full=y_full,
            weight_mode=weight_mode,
            boundary_axes=boundary_axes,
            include_center=True,
            center_weight_frac=center_weight_frac,
            label_sharpen_power=label_sharpen_power,
        )
    if repr_mode == "purity_gated_hybrid":
        return balls_to_purity_gated_training_set(
            balls=balls,
            X_source=X_source,
            y_full=y_full,
            weight_mode=weight_mode,
            boundary_axes=boundary_axes,
            center_weight_frac=center_weight_frac,
            label_sharpen_power=label_sharpen_power,
            low_purity_threshold=low_purity_threshold,
            low_purity_raw_keep=low_purity_raw_keep,
        )
    raise ValueError(f"Unknown repr_mode: {repr_mode}")


def evaluate_ball_config(
    cfg: Dict,
    X_train_orig: np.ndarray,
    y_train: np.ndarray,
    X_dev_orig: np.ndarray,
    y_dev: np.ndarray,
    X_test_orig: Optional[np.ndarray],
    y_test: Optional[np.ndarray],
    pca_cache: Dict[int, Dict[str, Optional[np.ndarray]]],
    device: str,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    hidden_dim: int,
    dropout: float,
    patience: int,
    class_balance: bool,
    selection_metric: str,
    tune_decision_threshold: bool,
) -> Dict:
    dim = int(cfg["pca_dim"])
    part_pack = pca_cache[dim]
    X_train_part = part_pack["train"]

    t0 = time.time()
    balls = build_granular_balls(
        X=X_train_part,
        y=y_train,
        purity_threshold=float(cfg["purity_threshold"]),
        min_ball_size=int(cfg["min_ball_size"]),
        max_depth=int(cfg["max_ball_depth"]),
        apply_global_refine=True,
        use_adaptive_accept=bool(cfg["use_adaptive_accept"]),
        enforce_initial_purity_lower_bound=bool(cfg["enforce_initial_purity_lower_bound"]),
    )
    ball_build_time = time.time() - t0
    ball_stats = summarize_balls(balls, n_samples=len(X_train_part))

    if cfg["repr_feature_space"] == "orig":
        X_train_repr = X_train_orig
        X_dev_eval = X_dev_orig
        X_test_eval = X_test_orig
    elif cfg["repr_feature_space"] == "ball":
        X_train_repr = X_train_part
        X_dev_eval = part_pack["dev"]
        X_test_eval = part_pack["test"]
    else:
        raise ValueError(f"Unknown repr_feature_space: {cfg['repr_feature_space']}")

    Xb, Yb_soft, wb = build_training_set_from_balls(
        balls=balls,
        X_source=X_train_repr,
        y_full=y_train,
        repr_mode=str(cfg["repr_mode"]),
        weight_mode=str(cfg["weight_mode"]),
        boundary_axes=int(cfg["boundary_axes"]),
        center_weight_frac=float(cfg["center_weight_frac"]),
        label_sharpen_power=float(cfg["label_sharpen_power"]),
        low_purity_threshold=float(cfg["low_purity_threshold"]),
        low_purity_raw_keep=int(cfg["low_purity_raw_keep"]),
    )

    log_prefix = f"ball-{cfg['repr_mode']}"
    model, dev_metrics, train_time, best_threshold = train_classifier_soft(
        X_train=Xb,
        y_soft_train=Yb_soft,
        X_dev=X_dev_eval,
        y_dev=y_dev,
        train_weights=wb,
        device=device,
        epochs=epochs,
        batch_size=min(batch_size, max(8, len(Xb))),
        lr=lr,
        weight_decay=weight_decay,
        hidden_dim=hidden_dim,
        dropout=dropout,
        patience=patience,
        class_balance=class_balance,
        log_prefix=log_prefix,
        selection_metric=selection_metric,
        tune_decision_threshold=tune_decision_threshold,
    )

    test_metrics = None
    if X_test_eval is not None and y_test is not None and np.all(y_test >= 0):
        pred = predict_model(model, X_test_eval, device=device)
        test_pred = probs_to_pred(pred["probs"], threshold=best_threshold)
        test_metrics = compute_metrics(y_test, pred["probs"], test_pred)

    return {
        "config": cfg,
        "ball_build_time_sec": float(ball_build_time),
        "ball_train_time_sec": float(train_time),
        "ball_stats": ball_stats,
        "train_items": int(len(Xb)),
        "dev": dev_metrics,
        "test": test_metrics,
        "decision_threshold": float(best_threshold),
        "model": model,
    }


# -----------------------------------------------------------------------------
# Main experiment
# -----------------------------------------------------------------------------

def resolve_data_root(user_value: Optional[str]) -> Path:
    """Resolve a runnable default data root for local Hateful Memes experiments."""
    candidates: List[Path] = []
    if user_value:
        candidates.append(Path(user_value))

    env_root = os.environ.get("HATEFUL_MEMES_ROOT", "").strip()
    if env_root:
        candidates.append(Path(env_root))

    candidates.extend([
        Path("../data/ori_data/hateful_memes"),
        Path("./data/ori_data/hateful_memes"),
        Path("../data/hateful_memes"),
        Path("./data/hateful_memes"),
        Path("../hateful_memes"),
        Path("./hateful_memes"),
    ])

    seen = set()
    for cand in candidates:
        cand = cand.expanduser()
        key = str(cand.resolve()) if cand.exists() else str(cand)
        if key in seen:
            continue
        seen.add(key)
        if cand.exists():
            try:
                _ = discover_splits(cand)
                return cand
            except Exception:
                pass

    tried = [str(c) for c in candidates]
    raise FileNotFoundError(
        "Could not locate the local Hateful Memes data root automatically. "
        f"Tried: {tried}. You can pass --data_root explicitly or set HATEFUL_MEMES_ROOT."
    )



# -----------------------------------------------------------------------------
# Protocol-adapted market simulation for the paper
# -----------------------------------------------------------------------------

try:
    from scipy.stats import kendalltau as scipy_kendalltau, spearmanr as scipy_spearmanr
except Exception:
    scipy_kendalltau = None
    scipy_spearmanr = None


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_numpy_arrays(*arrays: np.ndarray) -> str:
    h = hashlib.sha256()
    for arr in arrays:
        arr = np.ascontiguousarray(arr)
        h.update(str(arr.shape).encode("utf-8"))
        h.update(str(arr.dtype).encode("utf-8"))
        h.update(arr.tobytes())
    return h.hexdigest()


def safe_float(x: Any) -> float:
    try:
        xf = float(x)
    except Exception:
        return float("nan")
    if math.isnan(xf) or math.isinf(xf):
        return float("nan")
    return xf


def _rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    vals, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    for i, c in enumerate(counts):
        if c > 1:
            ids = np.where(inv == i)[0]
            ranks[ids] = ranks[ids].mean()
    return ranks


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    if scipy_spearmanr is not None:
        try:
            return float(scipy_spearmanr(x, y).correlation)
        except Exception:
            pass
    rx = _rankdata(x)
    ry = _rankdata(y)
    if np.std(rx) < 1e-12 or np.std(ry) < 1e-12:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def kendall_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    if scipy_kendalltau is not None:
        try:
            return float(scipy_kendalltau(x, y).correlation)
        except Exception:
            pass
    # O(n^2) fallback, fine for small seller counts
    n = len(x)
    concordant = 0
    discordant = 0
    ties_x = 0
    ties_y = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = np.sign(x[i] - x[j])
            dy = np.sign(y[i] - y[j])
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                ties_x += 1
            elif dy == 0:
                ties_y += 1
            elif dx == dy:
                concordant += 1
            else:
                discordant += 1
    denom = math.sqrt((concordant + discordant + ties_x) * (concordant + discordant + ties_y))
    if denom <= 1e-12:
        return float("nan")
    return float((concordant - discordant) / denom)


def exact_budget_select(score_by_id: Dict[str, float], cost_by_id: Dict[str, float], budget: float, max_k: int) -> List[str]:
    sellers = list(score_by_id.keys())
    best_subset: List[str] = []
    best_score = -1e18
    max_take = min(max_k, len(sellers))
    for k in range(1, max_take + 1):
        for combo in itertools.combinations(sellers, k):
            cost = sum(cost_by_id[s] for s in combo)
            if cost - budget > 1e-9:
                continue
            score = sum(score_by_id[s] for s in combo)
            if score > best_score:
                best_score = score
                best_subset = list(combo)
    return best_subset


def random_budget_select(seller_ids: List[str], cost_by_id: Dict[str, float], budget: float, max_k: int, rng: np.random.RandomState) -> List[str]:
    order = seller_ids.copy()
    rng.shuffle(order)
    picked = []
    spent = 0.0
    for sid in order:
        if len(picked) >= max_k:
            break
        c = float(cost_by_id[sid])
        if spent + c <= budget + 1e-9:
            picked.append(sid)
            spent += c
    return picked


def distribute_types(num_sellers: int, type_cycle: List[str], rng: np.random.RandomState) -> List[str]:
    types = [type_cycle[i % len(type_cycle)] for i in range(num_sellers)]
    rng.shuffle(types)
    return types


def stratified_pick_indices(y: np.ndarray, n_pick: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    idx_all = np.arange(len(y))
    if n_pick >= len(y):
        return idx_all, np.array([], dtype=np.int64)
    train_idx, rest_idx = train_test_split(
        idx_all,
        train_size=n_pick,
        random_state=seed,
        shuffle=True,
        stratify=y,
    )
    return np.array(train_idx, dtype=np.int64), np.array(rest_idx, dtype=np.int64)


def permute_without_fixed_points(n: int, rng: np.random.RandomState) -> np.ndarray:
    if n <= 1:
        return np.arange(n)
    perm = np.arange(n)
    for _ in range(20):
        perm = rng.permutation(n)
        if not np.any(perm == np.arange(n)):
            return perm
    perm = np.roll(np.arange(n), 1)
    return perm


def make_random_unit_vectors(n: int, d: int, rng: np.random.RandomState) -> np.ndarray:
    z = rng.normal(size=(n, d)).astype(np.float32)
    return l2_normalize_np(z)


def degrade_text_features(
    txt_feats: np.ndarray,
    ratio: float,
    mode: str,
    rng: np.random.RandomState,
    template_vec: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, int]]:
    out = txt_feats.copy().astype(np.float32)
    n, d = out.shape
    n_corrupt = int(round(max(0.0, min(1.0, ratio)) * n))
    if n_corrupt <= 0:
        return out, {"clean": n}

    ids = rng.choice(n, size=n_corrupt, replace=False)
    stats = Counter()
    stats["clean"] = int(n - n_corrupt)

    if mode == "mismatch":
        perm = permute_without_fixed_points(len(ids), rng)
        out[ids] = txt_feats[ids[perm]]
        stats["mismatch"] = int(len(ids))
    elif mode == "template":
        if template_vec is None:
            template_vec = txt_feats.mean(axis=0)
        out[ids] = np.repeat(template_vec[None, :].astype(np.float32), len(ids), axis=0)
        stats["template"] = int(len(ids))
    elif mode == "noise":
        out[ids] = make_random_unit_vectors(len(ids), d, rng)
        stats["noise"] = int(len(ids))
    elif mode == "mixed":
        ids = ids.tolist()
        rng.shuffle(ids)
        n1 = len(ids) // 2
        n2 = len(ids) // 4
        mismatch_ids = np.array(ids[:n1], dtype=np.int64)
        template_ids = np.array(ids[n1:n1 + n2], dtype=np.int64)
        noise_ids = np.array(ids[n1 + n2:], dtype=np.int64)
        if len(mismatch_ids) > 0:
            perm = permute_without_fixed_points(len(mismatch_ids), rng)
            out[mismatch_ids] = txt_feats[mismatch_ids[perm]]
            stats["mismatch"] = int(len(mismatch_ids))
        if len(template_ids) > 0:
            if template_vec is None:
                template_vec = txt_feats.mean(axis=0)
            out[template_ids] = np.repeat(template_vec[None, :].astype(np.float32), len(template_ids), axis=0)
            stats["template"] = int(len(template_ids))
        if len(noise_ids) > 0:
            out[noise_ids] = make_random_unit_vectors(len(noise_ids), d, rng)
            stats["noise"] = int(len(noise_ids))
    else:
        raise ValueError(f"Unknown degradation mode: {mode}")

    out = l2_normalize_np(out)
    return out.astype(np.float32), dict(stats)


def corrupt_seller_pool(
    txt_feats: np.ndarray,
    y: np.ndarray,
    quality_type: str,
    rng: np.random.RandomState,
    template_vec: np.ndarray,
    num_classes: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    txt = txt_feats.copy().astype(np.float32)
    labels = y.copy().astype(np.int64)
    meta: Dict[str, Any] = {"quality_type": quality_type}

    if quality_type == "good":
        meta["corruption"] = {"clean": int(len(txt))}
        return txt, labels, meta

    if quality_type == "mixed":
        txt, cstats = degrade_text_features(txt, ratio=0.40, mode="mixed", rng=rng, template_vec=template_vec)
        meta["corruption"] = cstats
        return txt, labels, meta

    if quality_type == "irrelevant":
        txt, cstats = degrade_text_features(txt, ratio=1.00, mode="noise", rng=rng, template_vec=template_vec)
        meta["corruption"] = cstats
        return txt, labels, meta

    if quality_type == "noisy":
        txt, cstats = degrade_text_features(txt, ratio=0.70, mode="mixed", rng=rng, template_vec=template_vec)
        flip_ratio = 0.25
        n_flip = int(round(len(labels) * flip_ratio))
        if n_flip > 0:
            ids = rng.choice(len(labels), size=n_flip, replace=False)
            for idx in ids:
                new_lab = int(rng.randint(0, num_classes - 1))
                if new_lab >= labels[idx]:
                    new_lab += 1
                labels[idx] = new_lab
        cstats["label_flip"] = int(n_flip)
        meta["corruption"] = cstats
        return txt, labels, meta

    raise ValueError(f"Unknown seller quality type: {quality_type}")


def build_buyer_anchor_interface(
    buyer_img_feats: np.ndarray,
    buyer_txt_weak: np.ndarray,
    buyer_y: np.ndarray,
    pca_dim: int,
    purity_threshold: float,
    min_ball_size: int,
    max_ball_depth: int,
    use_adaptive_accept: bool,
    enforce_initial_purity_lower_bound: bool,
    seed: int,
    anchor_weight_mode: str = "size_weakness",
    weak_weight_lambda: float = 1.0,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    t0 = time.time()
    if pca_dim > 0 and pca_dim < buyer_img_feats.shape[1]:
        pca = PCA(n_components=pca_dim, random_state=seed)
        part_X = pca.fit_transform(buyer_img_feats).astype(np.float32)
    else:
        part_X = buyer_img_feats.astype(np.float32)

    balls = build_granular_balls(
        X=part_X,
        y=buyer_y,
        purity_threshold=purity_threshold,
        min_ball_size=min_ball_size,
        max_depth=max_ball_depth,
        apply_global_refine=True,
        use_adaptive_accept=use_adaptive_accept,
        enforce_initial_purity_lower_bound=enforce_initial_purity_lower_bound,
    )
    elapsed = time.time() - t0

    anchors: List[Dict[str, Any]] = []
    total = float(len(buyer_img_feats))
    raw_weights = []
    raw_weaknesses = []
    for aid, b in enumerate(balls):
        ids = b.indices.astype(np.int64)
        img_local = buyer_img_feats[ids]
        txt_local = buyer_txt_weak[ids]
        center = img_local.mean(axis=0).astype(np.float32)
        dist = np.linalg.norm(img_local - center[None, :], axis=1)
        radius = float(dist.mean()) if len(dist) else 0.0
        buyer_txt_center = txt_local.mean(axis=0).astype(np.float32)
        txt_dist = np.linalg.norm(txt_local - buyer_txt_center[None, :], axis=1)
        buyer_txt_radius = float(txt_dist.mean()) if len(txt_dist) else 0.0
        weakness = float(np.mean(1.0 - np.sum(img_local * txt_local, axis=1))) if len(ids) else 0.0
        size_weight = float(len(ids) / max(total, 1.0))
        anchors.append({
            "anchor_id": int(aid),
            "indices": ids,
            "center": center,
            "radius": radius,
            "weight": size_weight,
            "majority_label": int(b.majority_label),
            "purity": float(b.purity),
            "size": int(len(ids)),
            "buyer_txt_center": buyer_txt_center,
            "buyer_txt_radius": buyer_txt_radius,
            "weakness": weakness,
        })
        raw_weights.append(size_weight)
        raw_weaknesses.append(weakness)

    raw_weights = np.asarray(raw_weights, dtype=np.float32) if raw_weights else np.zeros((0,), dtype=np.float32)
    raw_weaknesses = np.asarray(raw_weaknesses, dtype=np.float32) if raw_weaknesses else np.zeros((0,), dtype=np.float32)
    if len(anchors) > 0:
        if anchor_weight_mode == "size_only":
            final_w = raw_weights.copy()
        else:
            weak_norm = raw_weaknesses / max(float(raw_weaknesses.mean()), 1e-6)
            weak_term = np.maximum(0.25, 1.0 + weak_weight_lambda * (weak_norm - 1.0))
            final_w = raw_weights * weak_term
        final_w = final_w / max(float(final_w.sum()), 1e-8)
        for a, w in zip(anchors, final_w.tolist()):
            a["weight"] = float(w)

    sizes = [a["size"] for a in anchors]
    radii = [a["radius"] for a in anchors]
    purities = [a["purity"] for a in anchors]
    weaknesses = [a["weakness"] for a in anchors]
    stats = {
        "num_anchors": int(len(anchors)),
        "build_time_sec": float(elapsed),
        "avg_anchor_size": float(np.mean(sizes)) if sizes else 0.0,
        "median_anchor_size": float(np.median(sizes)) if sizes else 0.0,
        "avg_anchor_radius": float(np.mean(radii)) if radii else 0.0,
        "avg_anchor_purity": float(np.mean(purities)) if purities else 0.0,
        "avg_anchor_weakness": float(np.mean(weaknesses)) if weaknesses else 0.0,
        "partition_pca_dim": int(pca_dim),
        "purity_threshold": float(purity_threshold),
        "min_ball_size": int(min_ball_size),
        "anchor_weight_mode": anchor_weight_mode,
        "weak_weight_lambda": float(weak_weight_lambda),
    }
    return anchors, stats


def compute_local_match_scores(
    txt_feats: np.ndarray,
    anchors: List[Dict[str, Any]],
    alpha1: float,
    alpha2: float,
    eps: float,
    radius_power: float = 1.0,
) -> np.ndarray:
    centers = np.stack([a["center"] for a in anchors], axis=0).astype(np.float32)
    radii = np.array([max(a["radius"], 1e-6) for a in anchors], dtype=np.float32)
    dot = txt_feats @ centers.T
    sq = (
        np.sum(txt_feats ** 2, axis=1, keepdims=True)
        + np.sum(centers ** 2, axis=1)[None, :]
        - 2.0 * dot
    )
    denom = np.power(radii[None, :] + eps, radius_power)
    compat = alpha1 * dot - alpha2 * sq / denom
    return compat.astype(np.float32)


def seller_local_match_and_package(
    seller_id: str,
    raw_ids: np.ndarray,
    txt_feats: np.ndarray,
    anchors: List[Dict[str, Any]],
    alpha1: float,
    alpha2: float,
    eps: float,
    topk: int = 3,
    match_margin: float = 0.15,
    match_temperature: float = 0.2,
    radius_power: float = 1.0,
    min_effective_count: float = 2.0,
) -> Tuple[Dict[str, Any], float]:
    t0 = time.time()
    compat = compute_local_match_scores(
        txt_feats, anchors, alpha1=alpha1, alpha2=alpha2, eps=eps, radius_power=radius_power
    )
    n_txt, n_anchor = compat.shape
    topk = max(1, min(int(topk), n_anchor))
    parts = np.argpartition(-compat, kth=topk - 1, axis=1)[:, :topk]

    aggs: Dict[int, Dict[str, Any]] = {}
    for i in range(n_txt):
        idxs = parts[i]
        vals = compat[i, idxs]
        order = np.argsort(-vals)
        idxs = idxs[order]
        vals = vals[order]
        keep = vals >= (vals[0] - match_margin)
        idxs = idxs[keep]
        vals = vals[keep]
        logits = (vals - vals.max()) / max(match_temperature, 1e-6)
        ws = np.exp(logits)
        ws = ws / max(float(ws.sum()), 1e-8)
        for aid, w in zip(idxs.tolist(), ws.tolist()):
            rec = aggs.setdefault(int(aid), {
                "sum_w": 0.0,
                "sum_feat": np.zeros((txt_feats.shape[1],), dtype=np.float64),
                "members": [],
                "member_w": [],
            })
            rec["sum_w"] += float(w)
            rec["sum_feat"] += float(w) * txt_feats[i].astype(np.float64)
            rec["members"].append(int(i))
            rec["member_w"].append(float(w))

    units = []
    for aid in sorted(aggs.keys()):
        rec = aggs[aid]
        eff_n = float(rec["sum_w"])
        if eff_n < min_effective_count:
            continue
        member_ids = np.array(rec["members"], dtype=np.int64)
        member_w = np.array(rec["member_w"], dtype=np.float32)
        local_txt = txt_feats[member_ids]
        mu = (rec["sum_feat"] / max(eff_n, 1e-8)).astype(np.float32)
        dist = np.linalg.norm(local_txt - mu[None, :], axis=1)
        rho = float(np.sum(member_w * dist) / max(float(member_w.sum()), 1e-8)) if len(local_txt) else 0.0
        units.append({
            "anchor_id": int(aid),
            "mu_txt": mu,
            "rho_txt": rho,
            "n": float(eff_n),
            "raw_count": int(len(np.unique(member_ids))),
            "raw_ids": raw_ids[np.unique(member_ids)].astype(np.int64),
        })

    if len(units) == 0:
        global_mu = txt_feats.mean(axis=0).astype(np.float32)
        global_rho = float(np.linalg.norm(txt_feats - global_mu[None, :], axis=1).mean()) if len(txt_feats) else 0.0
        best_anchor = int(np.argmax(compat.mean(axis=0))) if n_anchor > 0 else 0
        units = [{
            "anchor_id": best_anchor,
            "mu_txt": global_mu,
            "rho_txt": global_rho,
            "n": float(len(txt_feats)),
            "raw_count": int(len(txt_feats)),
            "raw_ids": raw_ids.astype(np.int64),
        }]

    package_digest_parts = [
        seller_id.encode("utf-8"),
        np.array(raw_ids, dtype=np.int64).tobytes(),
    ]
    for u in units:
        package_digest_parts.append(np.array([u["anchor_id"], int(round(u["n"] * 1000)), u["raw_count"]], dtype=np.int64).tobytes())
        package_digest_parts.append(np.asarray(u["mu_txt"], dtype=np.float32).tobytes())
        package_digest_parts.append(np.asarray([u["rho_txt"]], dtype=np.float32).tobytes())
    package_commitment = sha256_hex(b"".join(package_digest_parts))
    elapsed = time.time() - t0
    package = {
        "seller_id": seller_id,
        "units": units,
        "package_commitment": package_commitment,
        "num_units": int(len(units)),
        "total_count": int(len(raw_ids)),
        "effective_total_count": float(sum(float(u["n"]) for u in units)),
    }
    return package, float(elapsed)


def package_to_ciphertext_sim(package: Dict[str, Any], session_id: str) -> Dict[str, Any]:
    ct_hash = sha256_hex(f"{session_id}|{package['seller_id']}|{package['package_commitment']}".encode("utf-8"))
    return {
        "ciphertext_hash": ct_hash,
        "seller_id": package["seller_id"],
        "package_commitment": package["package_commitment"],
    }


def score_packaged_seller(
    anchors: List[Dict[str, Any]],
    package: Dict[str, Any],
    alpha: float,
    beta: float,
    gamma: float,
    delta: float,
    count_g: str,
    coverage_bonus: float,
    normalize_mode: str,
    use_buyer_delta: bool,
    positive_margin: float = 0.02,
    bad_penalty: float = 0.35,
    top_anchor_frac: float = 0.35,
    purity_power: float = 1.0,
    count_cap: float = 1.0,
    focus_penalty: float = 1.5,
    peak_power: float = 1.5,
    negative_top_frac: float = 0.25,
) -> Tuple[float, List[Dict[str, Any]]]:
    anchor_map = {a["anchor_id"]: a for a in anchors}
    contribs = []
    positive_terms: List[float] = []
    negative_terms: List[float] = []
    active_weight = 0.0
    positive_coverage = 0.0

    for u in package["units"]:
        a = anchor_map[u["anchor_id"]]
        ca = a["center"]
        ra = max(float(a["radius"]), 1e-6)
        mu = u["mu_txt"]
        rho = float(u["rho_txt"])
        n_eff = float(u["n"])
        raw_count = float(u.get("raw_count", max(1, int(round(n_eff)))))
        dot = float(np.dot(ca, mu))
        sqdist = float(np.sum((mu - ca) ** 2))
        seller_core = alpha * dot - beta * sqdist - gamma * ((rho - ra) ** 2)

        if use_buyer_delta:
            mu_b = a["buyer_txt_center"]
            rho_b = float(a["buyer_txt_radius"])
            dot_b = float(np.dot(ca, mu_b))
            sqdist_b = float(np.sum((mu_b - ca) ** 2))
            buyer_core = alpha * dot_b - beta * sqdist_b - gamma * ((rho_b - ra) ** 2)
            improve = seller_core - buyer_core
        else:
            improve = seller_core

        demand_cap = max(1.0, float(a.get("size", 1)))
        capped_count = min(raw_count, count_cap * demand_cap)
        density = capped_count / demand_cap
        if count_g == "log1p":
            gval = math.log1p(capped_count) / max(math.log1p(count_cap * demand_cap), 1e-8)
        elif count_g == "sqrt":
            gval = math.sqrt(capped_count) / max(math.sqrt(count_cap * demand_cap), 1e-8)
        else:
            gval = density

        purity_fac = max(float(a.get("purity", 1.0)), 1e-4) ** purity_power
        base_weight = float(a["weight"]) * purity_fac
        active_weight += base_weight

        focus_ratio = rho / ra
        focus_fac = math.exp(-focus_penalty * max(0.0, focus_ratio - 1.0))
        gap = max(0.0, improve - positive_margin)
        loss = max(0.0, positive_margin - improve)
        count_term = delta * gval * 0.25

        if gap > 0.0:
            ua = ((gap ** peak_power) + count_term) * purity_fac * focus_fac
            weighted = float(a["weight"]) * ua
            positive_terms.append(weighted)
            positive_coverage += base_weight * focus_fac
            sign = "positive"
        else:
            ua = -bad_penalty * loss * purity_fac * max(1.0, 1.0 / max(focus_fac, 1e-4))
            weighted = float(a["weight"]) * ua
            negative_terms.append(weighted)
            sign = "negative"

        contribs.append({
            "anchor_id": int(u["anchor_id"]),
            "u_anchor": float(ua),
            "weighted_u_anchor": float(weighted),
            "n_eff": float(n_eff),
            "raw_count": int(raw_count),
            "improve_over_buyer": float(improve),
            "focus_ratio": float(focus_ratio),
            "focus_fac": float(focus_fac),
            "sign": sign,
            "anchor_purity": float(a.get("purity", 1.0)),
        })

    positive_terms.sort(reverse=True)
    if positive_terms:
        if top_anchor_frac <= 1.0:
            keep_k = max(1, int(math.ceil(len(positive_terms) * max(top_anchor_frac, 1e-6))))
        else:
            keep_k = max(1, min(len(positive_terms), int(round(top_anchor_frac))))
        pos_sum = float(sum(positive_terms[:keep_k])) / float(keep_k)
    else:
        pos_sum = 0.0

    negative_terms.sort()
    if negative_terms:
        if negative_top_frac <= 1.0:
            neg_k = max(1, int(math.ceil(len(negative_terms) * max(negative_top_frac, 1e-6))))
        else:
            neg_k = max(1, min(len(negative_terms), int(round(negative_top_frac))))
        neg_sum = float(sum(negative_terms[:neg_k])) / float(neg_k)
    else:
        neg_sum = 0.0

    if normalize_mode == "active_weight":
        denom = max(active_weight, 1e-8)
    elif normalize_mode == "num_units":
        denom = max(float(len(package["units"])), 1e-8)
    else:
        denom = 1.0

    total = (pos_sum + neg_sum) / denom
    total += coverage_bonus * positive_coverage / max(active_weight, 1e-8)
    return float(total), contribs


def score_raw_seller_no_packaging(
    anchors: List[Dict[str, Any]],
    txt_feats: np.ndarray,
    alpha1: float,
    alpha2: float,
    eps: float,
    radius_power: float = 1.0,
) -> float:
    compat = compute_local_match_scores(txt_feats, anchors, alpha1=alpha1, alpha2=alpha2, eps=eps, radius_power=radius_power)
    best_anchor = np.argmax(compat, axis=1)
    weights = np.array([anchors[int(i)]["weight"] for i in best_anchor], dtype=np.float32)
    scores = compat[np.arange(len(txt_feats)), best_anchor] * weights
    return float(scores.mean()) if len(scores) else float("-inf")


def score_seller_global(
    anchors: List[Dict[str, Any]],
    seller_txt_feats: np.ndarray,
    alpha: float,
    beta: float,
    gamma: float,
    delta: float,
    count_g: str,
    coverage_bonus: float = 0.0,
    normalize_mode: str = "none",
    use_buyer_delta: bool = False,
    buyer_txt_feats_weak: Optional[np.ndarray] = None,
    positive_margin: float = 0.02,
    bad_penalty: float = 0.35,
    focus_penalty: float = 1.5,
    peak_power: float = 1.5,
) -> float:
    if seller_txt_feats is None or len(seller_txt_feats) == 0:
        return -1e9
    seller_mean = seller_txt_feats.mean(axis=0)
    seller_center = l2_normalize_np(seller_mean[None, :])[0]
    seller_radius = float(np.mean(np.linalg.norm(seller_txt_feats - seller_mean[None, :], axis=1)))

    if use_buyer_delta and buyer_txt_feats_weak is not None and len(buyer_txt_feats_weak) > 0:
        buyer_mean = buyer_txt_feats_weak.mean(axis=0)
        buyer_center = l2_normalize_np(buyer_mean[None, :])[0]
        buyer_radius = float(np.mean(np.linalg.norm(buyer_txt_feats_weak - buyer_mean[None, :], axis=1)))
    else:
        buyer_center = None
        buyer_radius = 0.0

    pos_terms=[]
    neg_terms=[]
    active_weight=0.0
    coverage=0.0
    for a in anchors:
        ca = a["center"]
        ra = max(float(a["radius"]), 1e-6)
        dot = float(np.dot(ca, seller_center))
        sqdist = float(np.sum((seller_center - ca) ** 2))
        seller_core = alpha * dot - beta * sqdist - gamma * ((seller_radius - ra) ** 2)
        if buyer_center is not None:
            dot_b = float(np.dot(ca, buyer_center))
            sqdist_b = float(np.sum((buyer_center - ca) ** 2))
            buyer_core = alpha * dot_b - beta * sqdist_b - gamma * ((buyer_radius - ra) ** 2)
            improve = seller_core - buyer_core
        else:
            improve = seller_core

        purity_fac = max(float(a.get("purity", 1.0)), 1e-4)
        w = float(a["weight"]) * purity_fac
        active_weight += w
        focus_ratio = seller_radius / ra
        focus_fac = math.exp(-focus_penalty * max(0.0, focus_ratio - 1.0))
        gap = max(0.0, improve - positive_margin)
        loss = max(0.0, positive_margin - improve)
        if gap > 0.0:
            pos_terms.append(w * ((gap ** peak_power) * focus_fac))
            coverage += w * focus_fac
        else:
            neg_terms.append(-w * bad_penalty * loss * max(1.0, 1.0 / max(focus_fac, 1e-4)))

    pos_terms.sort(reverse=True)
    neg_terms.sort()
    pos_k = max(1, int(math.ceil(len(pos_terms) * 0.2))) if pos_terms else 0
    neg_k = max(1, int(math.ceil(len(neg_terms) * 0.25))) if neg_terms else 0
    pos_sum = float(sum(pos_terms[:pos_k])) / float(pos_k) if pos_terms else 0.0
    neg_sum = float(sum(neg_terms[:neg_k])) / float(neg_k) if neg_terms else 0.0
    if normalize_mode == "active_weight":
        denom=max(active_weight,1e-8)
    elif normalize_mode == "num_units":
        denom=max(float(len(anchors)),1e-8)
    else:
        denom=1.0
    total=(pos_sum+neg_sum)/denom
    total += coverage_bonus * coverage / max(active_weight,1e-8)
    return float(total)


def summarize_screen_alignment(
    sellers: List[Dict[str, Any]],
    realized: Dict[str, Dict[str, Any]],
    score_dict: Dict[str, float],
) -> Dict[str, Any]:
    ids = [s["seller_id"] for s in sellers]
    score_vec = np.array([safe_float(score_dict[sid]) for sid in ids], dtype=np.float64)
    gain_vec = np.array([safe_float(realized[sid]["gain_test_macro_f1"]) for sid in ids], dtype=np.float64)
    top_score_sid = ids[int(np.nanargmax(score_vec))]
    top_gain_sid = ids[int(np.nanargmax(gain_vec))]
    return {
        "spearman": spearman_corr(score_vec, gain_vec),
        "kendall": kendall_corr(score_vec, gain_vec),
        "top1_selected_seller": top_score_sid,
        "oracle_best_seller": top_gain_sid,
        "hit_at_1": float(top_score_sid == top_gain_sid),
        "selected_gain_test_macro_f1": float(realized[top_score_sid]["gain_test_macro_f1"]),
        "best_gain_test_macro_f1": float(realized[top_gain_sid]["gain_test_macro_f1"]),
    }


def maybe_train_selected_subset(
    selected_ids: List[str],
    sellers_by_id: Dict[str, Dict[str, Any]],
    buyer_X_weak: np.ndarray,
    buyer_y: np.ndarray,
    X_dev_clean: np.ndarray,
    y_dev: np.ndarray,
    X_test_clean: np.ndarray,
    y_test: np.ndarray,
    device: str,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    hidden_dim: int,
    dropout: float,
    patience: int,
    class_balance: bool,
    selection_metric: str,
    tune_decision_threshold: bool,
    log_prefix: str,
) -> Dict[str, Any]:
    if len(selected_ids) == 1:
        s = sellers_by_id[selected_ids[0]]
        train_X = np.concatenate([buyer_X_weak, s["X_delivered"]], axis=0).astype(np.float32)
        train_y = np.concatenate([buyer_y, s["y_delivered"]], axis=0).astype(np.int64)
    else:
        Xs = [buyer_X_weak]
        Ys = [buyer_y]
        for sid in selected_ids:
            s = sellers_by_id[sid]
            Xs.append(s["X_delivered"])
            Ys.append(s["y_delivered"])
        train_X = np.concatenate(Xs, axis=0).astype(np.float32)
        train_y = np.concatenate(Ys, axis=0).astype(np.int64)

    return evaluate_train_set(
        X_train=train_X,
        y_train=train_y,
        X_dev=X_dev_clean,
        y_dev=y_dev,
        X_test=X_test_clean,
        y_test=y_test,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        hidden_dim=hidden_dim,
        dropout=dropout,
        patience=patience,
        class_balance=class_balance,
        selection_metric=selection_metric,
        tune_decision_threshold=tune_decision_threshold,
        log_prefix=log_prefix,
    )



def evaluate_train_set(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_dev: np.ndarray,
    y_dev: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    device: str,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    hidden_dim: int,
    dropout: float,
    patience: int,
    class_balance: bool,
    selection_metric: str,
    tune_decision_threshold: bool,
    log_prefix: str,
) -> Dict[str, Any]:
    model, dev_metrics, train_time, best_threshold = train_classifier_hard(
        X_train=X_train,
        y_train=y_train,
        X_dev=X_dev,
        y_dev=y_dev,
        train_weights=None,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        hidden_dim=hidden_dim,
        dropout=dropout,
        patience=patience,
        class_balance=class_balance,
        selection_metric=selection_metric,
        tune_decision_threshold=tune_decision_threshold,
    )
    pred = predict_model(model, X_test, device=device)
    y_pred = probs_to_pred(pred["probs"], threshold=best_threshold)
    test_metrics = compute_metrics(y_test, pred["probs"], y_pred)
    return {
        "model": model,
        "dev": dev_metrics,
        "test": test_metrics,
        "train_time_sec": float(train_time),
        "decision_threshold": float(best_threshold),
    }


def build_market_from_train(
    train_feats: Dict[str, np.ndarray],
    buyer_train_size: int,
    num_sellers: int,
    seller_size: int,
    seller_types: List[str],
    buyer_degrade_ratio: float,
    buyer_degrade_mode: str,
    fusion: str,
    seed: int,
) -> Dict[str, Any]:
    rng = np.random.RandomState(seed)
    y_train = train_feats["y"]
    buyer_ids, rest_ids = stratified_pick_indices(y_train, n_pick=buyer_train_size, seed=seed)
    rng.shuffle(rest_ids)

    needed = num_sellers * seller_size
    if needed > len(rest_ids):
        raise ValueError(
            f"Not enough training rows for seller market: need {needed}, have {len(rest_ids)} after buyer split."
        )

    template_vec = train_feats["txt"].mean(axis=0).astype(np.float32)
    num_classes = int(y_train.max()) + 1

    buyer_img = train_feats["img"][buyer_ids].astype(np.float32)
    buyer_txt_clean = train_feats["txt"][buyer_ids].astype(np.float32)
    buyer_txt_weak, buyer_degrade_stats = degrade_text_features(
        buyer_txt_clean,
        ratio=buyer_degrade_ratio,
        mode=buyer_degrade_mode,
        rng=np.random.RandomState(seed + 17),
        template_vec=template_vec,
    )
    buyer_X_clean = l2_normalize_np(make_fused_features(buyer_img, buyer_txt_clean, fusion)).astype(np.float32)
    buyer_X_weak = l2_normalize_np(make_fused_features(buyer_img, buyer_txt_weak, fusion)).astype(np.float32)
    buyer_y = y_train[buyer_ids].astype(np.int64)

    qualities = distribute_types(num_sellers, seller_types, rng)
    sellers = []
    cursor = 0
    for sidx in range(num_sellers):
        ids = np.array(rest_ids[cursor:cursor + seller_size], dtype=np.int64)
        cursor += seller_size
        img = train_feats["img"][ids].astype(np.float32)
        txt_clean = train_feats["txt"][ids].astype(np.float32)
        y = y_train[ids].astype(np.int64)
        txt_delivered, y_delivered, meta = corrupt_seller_pool(
            txt_clean,
            y,
            quality_type=qualities[sidx],
            rng=np.random.RandomState(seed + 1000 + sidx),
            template_vec=template_vec,
            num_classes=num_classes,
        )
        X_delivered = l2_normalize_np(make_fused_features(img, txt_delivered, fusion)).astype(np.float32)
        raw_commitment = hash_numpy_arrays(img, txt_delivered, y_delivered.astype(np.int64))
        sellers.append({
            "seller_id": f"S{sidx}",
            "quality_type": qualities[sidx],
            "raw_ids": ids,
            "img": img,
            "txt_clean": txt_clean,
            "txt_delivered": txt_delivered,
            "X_delivered": X_delivered,
            "y_delivered": y_delivered.astype(np.int64),
            "cost": 1.0,
            "raw_commitment": raw_commitment,
            "seller_meta": meta,
        })

    return {
        "buyer_ids": buyer_ids,
        "buyer_X_weak": buyer_X_weak,
        "buyer_X_clean": buyer_X_clean,
        "buyer_img": buyer_img,
        "buyer_txt_clean": buyer_txt_clean,
        "buyer_txt_weak": buyer_txt_weak,
        "buyer_y": buyer_y,
        "buyer_degrade_stats": buyer_degrade_stats,
        "template_vec": template_vec,
        "sellers": sellers,
    }


def evaluate_individual_seller_utilities(
    sellers: List[Dict[str, Any]],
    buyer_X_weak: np.ndarray,
    buyer_y: np.ndarray,
    X_dev_clean: np.ndarray,
    y_dev: np.ndarray,
    X_test_clean: np.ndarray,
    y_test: np.ndarray,
    weak_baseline_macro_f1: float,
    device: str,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    hidden_dim: int,
    dropout: float,
    patience: int,
    class_balance: bool,
    selection_metric: str,
    tune_decision_threshold: bool,
) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for seller in sellers:
        sid = seller["seller_id"]
        print(f"\n================ Post-purchase validation for {sid} ({seller['quality_type']}) ================")
        train_X = np.concatenate([buyer_X_weak, seller["X_delivered"]], axis=0).astype(np.float32)
        train_y = np.concatenate([buyer_y, seller["y_delivered"]], axis=0).astype(np.int64)
        eval_pack = evaluate_train_set(
            X_train=train_X,
            y_train=train_y,
            X_dev=X_dev_clean,
            y_dev=y_dev,
            X_test=X_test_clean,
            y_test=y_test,
            device=device,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            hidden_dim=hidden_dim,
            dropout=dropout,
            patience=patience,
            class_balance=class_balance,
            selection_metric=selection_metric,
            tune_decision_threshold=tune_decision_threshold,
            log_prefix=f"postbuy-{sid}",
        )
        gain_f1 = safe_float(eval_pack["test"]["macro_f1"]) - float(weak_baseline_macro_f1)
        out[sid] = {
            "seller_id": sid,
            "quality_type": seller["quality_type"],
            "dev": eval_pack["dev"],
            "test": eval_pack["test"],
            "train_time_sec": float(eval_pack["train_time_sec"]),
            "decision_threshold": float(eval_pack["decision_threshold"]),
            "gain_test_macro_f1": float(gain_f1),
            "raw_commitment": seller["raw_commitment"],
        }
    return out


def summarize_screen_alignment(
    sellers: List[Dict[str, Any]],
    realized: Dict[str, Dict[str, Any]],
    score_dict: Dict[str, float],
) -> Dict[str, Any]:
    ids = [s["seller_id"] for s in sellers]
    score_vec = np.array([safe_float(score_dict[sid]) for sid in ids], dtype=np.float64)
    gain_vec = np.array([safe_float(realized[sid]["gain_test_macro_f1"]) for sid in ids], dtype=np.float64)
    top_score_sid = ids[int(np.nanargmax(score_vec))]
    top_gain_sid = ids[int(np.nanargmax(gain_vec))]
    return {
        "spearman": spearman_corr(score_vec, gain_vec),
        "kendall": kendall_corr(score_vec, gain_vec),
        "top1_selected_seller": top_score_sid,
        "oracle_best_seller": top_gain_sid,
        "hit_at_1": float(top_score_sid == top_gain_sid),
        "selected_gain_test_macro_f1": float(realized[top_score_sid]["gain_test_macro_f1"]),
        "best_gain_test_macro_f1": float(realized[top_gain_sid]["gain_test_macro_f1"]),
    }


def maybe_train_selected_subset(
    selected_ids: List[str],
    sellers_by_id: Dict[str, Dict[str, Any]],
    buyer_X_weak: np.ndarray,
    buyer_y: np.ndarray,
    X_dev_clean: np.ndarray,
    y_dev: np.ndarray,
    X_test_clean: np.ndarray,
    y_test: np.ndarray,
    device: str,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    hidden_dim: int,
    dropout: float,
    patience: int,
    class_balance: bool,
    selection_metric: str,
    tune_decision_threshold: bool,
    log_prefix: str,
) -> Dict[str, Any]:
    if len(selected_ids) == 1:
        s = sellers_by_id[selected_ids[0]]
        train_X = np.concatenate([buyer_X_weak, s["X_delivered"]], axis=0).astype(np.float32)
        train_y = np.concatenate([buyer_y, s["y_delivered"]], axis=0).astype(np.int64)
    else:
        Xs = [buyer_X_weak]
        Ys = [buyer_y]
        for sid in selected_ids:
            s = sellers_by_id[sid]
            Xs.append(s["X_delivered"])
            Ys.append(s["y_delivered"])
        train_X = np.concatenate(Xs, axis=0).astype(np.float32)
        train_y = np.concatenate(Ys, axis=0).astype(np.int64)

    return evaluate_train_set(
        X_train=train_X,
        y_train=train_y,
        X_dev=X_dev_clean,
        y_dev=y_dev,
        X_test=X_test_clean,
        y_test=y_test,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        hidden_dim=hidden_dim,
        dropout=dropout,
        patience=patience,
        class_balance=class_balance,
        selection_metric=selection_metric,
        tune_decision_threshold=tune_decision_threshold,
        log_prefix=log_prefix,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Protocol-adapted Hateful Memes market test: buyer-conditioned packaging, secure screening, authorized-output release, and post-purchase validation."
    )
    parser.add_argument("--data_root", type=str, default="../data/ori_data/hateful_memes",
                        help="Local Hateful Memes root. Defaults to ../data/ori_data/hateful_memes and falls back to auto-detect if unavailable.")
    parser.add_argument("--output_dir", type=str, default="./outputs/hm_protocol_market_test")
    parser.add_argument("--model_name", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--fusion", type=str, default="concat", choices=["concat", "concat_mul_absdiff"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--feature_batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)

    # Protocol / market
    parser.add_argument("--buyer_train_size", type=int, default=2500)
    parser.add_argument("--num_sellers", type=int, default=8)
    parser.add_argument("--seller_size", type=int, default=400)
    parser.add_argument("--seller_types", type=str, default="good,good,mixed,mixed,irrelevant,irrelevant,noisy,noisy")
    parser.add_argument("--buyer_degrade_ratio", type=float, default=0.70)
    parser.add_argument("--buyer_degrade_mode", type=str, default="mixed", choices=["mismatch", "template", "noise", "mixed"])
    parser.add_argument("--purchase_budget", type=float, default=1.0)
    parser.add_argument("--max_purchase_sellers", type=int, default=1)
    parser.add_argument("--n_decrypt_parties", type=int, default=3)
    parser.add_argument("--session_id", type=str, default="hm_market_session")

    # Buyer anchor balls
    parser.add_argument("--anchor_pca_dim", type=int, default=64)
    parser.add_argument("--anchor_purity_threshold", type=float, default=0.92)
    parser.add_argument("--anchor_min_ball_size", type=int, default=8)
    parser.add_argument("--anchor_max_ball_depth", type=int, default=20)
    parser.add_argument("--anchor_use_adaptive_accept", dest="anchor_use_adaptive_accept", action="store_true")
    parser.add_argument("--anchor_no_use_adaptive_accept", dest="anchor_use_adaptive_accept", action="store_false")
    parser.add_argument("--anchor_enforce_initial_purity_lower_bound", dest="anchor_enforce_initial_purity_lower_bound", action="store_true")
    parser.add_argument("--anchor_no_enforce_initial_purity_lower_bound", dest="anchor_enforce_initial_purity_lower_bound", action="store_false")

    # Screening score parameters
    parser.add_argument("--anchor_weight_mode", type=str, default="size_weakness", choices=["size_only", "size_weakness"])
    parser.add_argument("--weak_weight_lambda", type=float, default=1.0)
    parser.add_argument("--match_alpha1", type=float, default=1.0)
    parser.add_argument("--match_alpha2", type=float, default=0.25)
    parser.add_argument("--match_radius_power", type=float, default=1.0)
    parser.add_argument("--match_topk", type=int, default=3)
    parser.add_argument("--match_margin", type=float, default=0.15)
    parser.add_argument("--match_temperature", type=float, default=0.2)
    parser.add_argument("--unit_min_effective_count", type=float, default=2.0)
    parser.add_argument("--screen_alpha", type=float, default=1.0)
    parser.add_argument("--screen_beta", type=float, default=0.5)
    parser.add_argument("--screen_gamma", type=float, default=0.2)
    parser.add_argument("--screen_delta", type=float, default=0.01)
    parser.add_argument("--count_g", type=str, default="sqrt", choices=["log1p", "sqrt", "linear"])
    parser.add_argument("--coverage_bonus", type=float, default=0.0)
    parser.add_argument("--screen_normalize_mode", type=str, default="active_weight", choices=["none", "active_weight", "num_units"])
    parser.add_argument("--screen_use_buyer_delta", dest="screen_use_buyer_delta", action="store_true")
    parser.add_argument("--screen_no_use_buyer_delta", dest="screen_use_buyer_delta", action="store_false")
    parser.add_argument("--screen_positive_margin", type=float, default=0.02)
    parser.add_argument("--screen_bad_penalty", type=float, default=0.8)
    parser.add_argument("--screen_top_anchor_frac", type=float, default=0.2)
    parser.add_argument("--screen_purity_power", type=float, default=1.5)
    parser.add_argument("--screen_count_cap", type=float, default=1.0)
    parser.add_argument("--screen_focus_penalty", type=float, default=1.5)
    parser.add_argument("--screen_peak_power", type=float, default=1.5)
    parser.add_argument("--screen_negative_top_frac", type=float, default=0.25)
    parser.add_argument("--screen_eps", type=float, default=1e-6)

    # Downstream model
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--class_balance", dest="class_balance", action="store_true")
    parser.add_argument("--no_class_balance", dest="class_balance", action="store_false")
    parser.add_argument("--selection_metric", type=str, default="macro_f1", choices=["accuracy", "macro_f1", "auroc", "acc_f1_mean"])
    parser.add_argument("--tune_decision_threshold", dest="tune_decision_threshold", action="store_true")
    parser.add_argument("--no_tune_decision_threshold", dest="tune_decision_threshold", action="store_false")

    parser.set_defaults(
        class_balance=True,
        tune_decision_threshold=True,
        anchor_use_adaptive_accept=False,
        anchor_enforce_initial_purity_lower_bound=True,
        screen_use_buyer_delta=True,
    )
    return parser.parse_args()


def simulate_authorized_output_release(
    seller_id: str,
    score_value: float,
    aux: Dict[str, Any],
    ciphertext_hash: str,
    session_id: str,
    n_parties: int,
    rng: np.random.RandomState,
) -> Dict[str, Any]:
    shares = []
    running = 0.0
    for pid in range(n_parties - 1):
        share_val = float(rng.normal(loc=0.0, scale=max(abs(score_value), 1.0)))
        running += share_val
        proof = sha256_hex(f"{session_id}|score:{seller_id}|{ciphertext_hash}|P{pid}|{share_val:.12f}".encode("utf-8"))
        shares.append({"party_id": f"P{pid}", "share_value": share_val, "proof": proof})
    last = float(score_value - running)
    pid = n_parties - 1
    proof = sha256_hex(f"{session_id}|score:{seller_id}|{ciphertext_hash}|P{pid}|{last:.12f}".encode("utf-8"))
    shares.append({"party_id": f"P{pid}", "share_value": last, "proof": proof})

    verified = True
    for sh in shares:
        expected = sha256_hex(
            f"{session_id}|score:{seller_id}|{ciphertext_hash}|{sh['party_id']}|{sh['share_value']:.12f}".encode("utf-8")
        )
        if expected != sh["proof"]:
            verified = False
            break

    fused = float(sum(sh["share_value"] for sh in shares)) if verified else float("nan")
    return {
        "seller_id": seller_id,
        "authorized_output": {
            "score": fused,
            "aux": aux,
        },
        "ciphertext_hash": ciphertext_hash,
        "verified": bool(verified),
        "shares": shares,
    }




def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    cache_dir = output_dir / "feature_cache"
    ensure_dir(output_dir)
    ensure_dir(cache_dir)

    data_root = resolve_data_root(args.data_root)
    split_paths = discover_splits(data_root)
    print("[splits]", {k: str(v) if v is not None else None for k, v in split_paths.items()})

    dfs: Dict[str, pd.DataFrame] = {}
    for split, path in split_paths.items():
        if path is None:
            continue
        df = normalize_columns(read_table(path))
        dfs[split] = df
        print(f"[{split}] rows={len(df)} cols={list(df.columns)}")

    feats = {}
    for split in ["train", "dev", "test"]:
        if split not in dfs:
            continue
        feats[split] = extract_or_load_features(
            split_name=split,
            df=dfs[split],
            data_root=data_root,
            cache_dir=cache_dir,
            model_name=args.model_name,
            fusion=args.fusion,
            batch_size=args.feature_batch_size,
            num_workers=args.num_workers,
            device=args.device,
        )

    market = build_market_from_train(
        train_feats=feats["train"],
        buyer_train_size=args.buyer_train_size,
        num_sellers=args.num_sellers,
        seller_size=args.seller_size,
        seller_types=parse_str_csv(args.seller_types),
        buyer_degrade_ratio=args.buyer_degrade_ratio,
        buyer_degrade_mode=args.buyer_degrade_mode,
        fusion=args.fusion,
        seed=args.seed,
    )
    sellers = market["sellers"]
    sellers_by_id = {s["seller_id"]: s for s in sellers}

    X_dev_clean = feats["dev"]["X"].astype(np.float32)
    y_dev = feats["dev"]["y"].astype(np.int64)
    X_test_clean = feats["test"]["X"].astype(np.float32)
    y_test = feats["test"]["y"].astype(np.int64)

    print("\n================ Phase 0: Buyer-side public query interface ================")
    anchors, anchor_stats = build_buyer_anchor_interface(
        buyer_img_feats=market["buyer_img"],
        buyer_txt_weak=market["buyer_txt_weak"],
        buyer_y=market["buyer_y"],
        pca_dim=args.anchor_pca_dim,
        purity_threshold=args.anchor_purity_threshold,
        min_ball_size=args.anchor_min_ball_size,
        max_ball_depth=args.anchor_max_ball_depth,
        use_adaptive_accept=args.anchor_use_adaptive_accept,
        enforce_initial_purity_lower_bound=args.anchor_enforce_initial_purity_lower_bound,
        seed=args.seed,
        anchor_weight_mode=args.anchor_weight_mode,
        weak_weight_lambda=args.weak_weight_lambda,
    )
    print(json.dumps(anchor_stats, ensure_ascii=False, indent=2))

    print("\n================ Step 1/6: Buyer initial model on degraded local view ================")
    weak_baseline = evaluate_train_set(
        X_train=market["buyer_X_weak"],
        y_train=market["buyer_y"],
        X_dev=X_dev_clean,
        y_dev=y_dev,
        X_test=X_test_clean,
        y_test=y_test,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        patience=args.patience,
        class_balance=args.class_balance,
        selection_metric=args.selection_metric,
        tune_decision_threshold=args.tune_decision_threshold,
        log_prefix="buyer-weak",
    )

    # Offline seller-local packaging
    print("\n================ Phase 1 + 1.5: Seller local matching, package construction, commitment ================")
    package_records = []
    total_packaging_time = 0.0
    for seller in sellers:
        pkg, pack_time = seller_local_match_and_package(
            seller_id=seller["seller_id"],
            raw_ids=seller["raw_ids"],
            txt_feats=seller["txt_delivered"],
            anchors=anchors,
            alpha1=args.match_alpha1,
            alpha2=args.match_alpha2,
            eps=args.screen_eps,
            topk=args.match_topk,
            match_margin=args.match_margin,
            match_temperature=args.match_temperature,
            radius_power=args.match_radius_power,
            min_effective_count=args.unit_min_effective_count,
        )
        total_packaging_time += pack_time
        seller["package"] = pkg
        seller["ciphertext"] = package_to_ciphertext_sim(pkg, session_id=args.session_id)
        deliverable_binding_ok = (pkg["package_commitment"] is not None and seller["raw_commitment"] is not None)
        package_records.append({
            "seller_id": seller["seller_id"],
            "quality_type": seller["quality_type"],
            "num_units": int(pkg["num_units"]),
            "total_count": int(pkg["total_count"]),
            "package_commitment": pkg["package_commitment"],
            "ciphertext_hash": seller["ciphertext"]["ciphertext_hash"],
            "packaging_time_sec": float(pack_time),
            "binding_present": bool(deliverable_binding_ok),
        })
        print(f"[package] seller={seller['seller_id']} type={seller['quality_type']} units={pkg['num_units']} total_count={pkg['total_count']} time={pack_time:.4f}s")

    # Step 2: secure screening on encrypted package summaries and authorized-output release
    print("\n================ Step 2/6: Secure screening on encrypted package summaries ================")
    ours_scores: Dict[str, float] = {}
    ours_aux: Dict[str, Dict[str, Any]] = {}
    ours_release = []
    t_screen_ours = time.time()
    for seller in sellers:
        score, contribs = score_packaged_seller(
            anchors=anchors,
            package=seller["package"],
            alpha=args.screen_alpha,
            beta=args.screen_beta,
            gamma=args.screen_gamma,
            delta=args.screen_delta,
            count_g=args.count_g,
            coverage_bonus=args.coverage_bonus,
            normalize_mode=args.screen_normalize_mode,
            use_buyer_delta=args.screen_use_buyer_delta,
        )
        ours_scores[seller["seller_id"]] = float(score)
        aux = {
            "seller_id": seller["seller_id"],
            "cost": float(seller["cost"]),
            "num_active_anchors": int(seller["package"]["num_units"]),
        }
        ours_aux[seller["seller_id"]] = aux
        release = simulate_authorized_output_release(
            seller_id=seller["seller_id"],
            score_value=float(score),
            aux=aux,
            ciphertext_hash=seller["ciphertext"]["ciphertext_hash"],
            session_id=args.session_id,
            n_parties=args.n_decrypt_parties,
            rng=np.random.RandomState(args.seed + 5000 + int(seller["seller_id"][1:])),
        )
        release["anchor_contribs"] = contribs
        ours_release.append(release)
    ours_online_time = time.time() - t_screen_ours

    print("\n================ Baseline screening scores ================")
    # No-packaging raw-level screening
    raw_scores: Dict[str, float] = {}
    t_screen_raw = time.time()
    for seller in sellers:
        raw_scores[seller["seller_id"]] = score_raw_seller_no_packaging(
            anchors=anchors,
            txt_feats=seller["txt_delivered"],
            alpha1=args.match_alpha1,
            alpha2=args.match_alpha2,
            eps=args.screen_eps,
            radius_power=args.match_radius_power,
        )
    raw_online_time = time.time() - t_screen_raw

    # Seller-global screening
    global_scores: Dict[str, float] = {}
    t_screen_global = time.time()
    for seller in sellers:
        global_scores[seller["seller_id"]] = score_seller_global(
            anchors=anchors,
            seller_txt_feats=seller["txt_delivered"],
            alpha=args.screen_alpha,
            beta=args.screen_beta,
            gamma=args.screen_gamma,
            delta=args.screen_delta,
            count_g=args.count_g,
            coverage_bonus=args.coverage_bonus,
            normalize_mode=args.screen_normalize_mode,
            use_buyer_delta=args.screen_use_buyer_delta,
            buyer_txt_feats_weak=market["buyer_txt_weak"],
        )
    global_online_time = time.time() - t_screen_global

    print("\n================ Step 3/6 + 4/6 + 5/6: Purchase, update dataset, retrain, realized utility ================")
    realized = evaluate_individual_seller_utilities(
        sellers=sellers,
        buyer_X_weak=market["buyer_X_weak"],
        buyer_y=market["buyer_y"],
        X_dev_clean=X_dev_clean,
        y_dev=y_dev,
        X_test_clean=X_test_clean,
        y_test=y_test,
        weak_baseline_macro_f1=float(weak_baseline["test"]["macro_f1"]),
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        patience=args.patience,
        class_balance=args.class_balance,
        selection_metric=args.selection_metric,
        tune_decision_threshold=args.tune_decision_threshold,
    )

    # Ranking / selection
    cost_by_id = {s["seller_id"]: float(s["cost"]) for s in sellers}
    ours_selected = exact_budget_select(ours_scores, cost_by_id, budget=args.purchase_budget, max_k=args.max_purchase_sellers)
    raw_selected = exact_budget_select(raw_scores, cost_by_id, budget=args.purchase_budget, max_k=args.max_purchase_sellers)
    global_selected = exact_budget_select(global_scores, cost_by_id, budget=args.purchase_budget, max_k=args.max_purchase_sellers)
    rng_sel = np.random.RandomState(args.seed + 9999)
    random_selected = random_budget_select([s["seller_id"] for s in sellers], cost_by_id, budget=args.purchase_budget, max_k=args.max_purchase_sellers, rng=rng_sel)

    good_ids = [s["seller_id"] for s in sellers if s["quality_type"] == "good"]
    if good_ids:
        oracle_good_scores = {sid: realized[sid]["gain_test_macro_f1"] for sid in good_ids}
        oracle_good_costs = {sid: cost_by_id[sid] for sid in good_ids}
        oracle_selected = exact_budget_select(oracle_good_scores, oracle_good_costs, budget=args.purchase_budget, max_k=args.max_purchase_sellers)
    else:
        oracle_selected = []

    def method_eval(selected_ids: List[str], name: str, screen_time_sec: float) -> Dict[str, Any]:
        if not selected_ids:
            return {
                "selected_sellers": [],
                "online_selection_time_sec": float(screen_time_sec),
                "post_purchase": None,
                "gain_test_macro_f1": float("nan"),
            }
        if len(selected_ids) == 1:
            sid = selected_ids[0]
            return {
                "selected_sellers": selected_ids,
                "online_selection_time_sec": float(screen_time_sec),
                "post_purchase": realized[sid],
                "gain_test_macro_f1": float(realized[sid]["gain_test_macro_f1"]),
            }
        eval_pack = maybe_train_selected_subset(
            selected_ids=selected_ids,
            sellers_by_id=sellers_by_id,
            buyer_X_weak=market["buyer_X_weak"],
            buyer_y=market["buyer_y"],
            X_dev_clean=X_dev_clean,
            y_dev=y_dev,
            X_test_clean=X_test_clean,
            y_test=y_test,
            device=args.device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            patience=args.patience,
            class_balance=args.class_balance,
            selection_metric=args.selection_metric,
            tune_decision_threshold=args.tune_decision_threshold,
            log_prefix=f"method-{name}",
        )
        gain = float(eval_pack["test"]["macro_f1"] - weak_baseline["test"]["macro_f1"])
        return {
            "selected_sellers": selected_ids,
            "online_selection_time_sec": float(screen_time_sec),
            "post_purchase": {
                "dev": eval_pack["dev"],
                "test": eval_pack["test"],
                "train_time_sec": float(eval_pack["train_time_sec"]),
                "decision_threshold": float(eval_pack["decision_threshold"]),
            },
            "gain_test_macro_f1": gain,
        }

    methods = {
        "ours_packaged": method_eval(ours_selected, "ours", ours_online_time),
        "no_packaging_raw": method_eval(raw_selected, "raw", raw_online_time),
        "seller_global": method_eval(global_selected, "global", global_online_time),
        "random": method_eval(random_selected, "random", 0.0),
        "oracle_good": method_eval(oracle_selected, "oracle_good", 0.0),
    }

    alignment = {
        "ours_packaged": summarize_screen_alignment(sellers, realized, ours_scores),
        "no_packaging_raw": summarize_screen_alignment(sellers, realized, raw_scores),
        "seller_global": summarize_screen_alignment(sellers, realized, global_scores),
    }

    seller_rows = []
    for seller in sellers:
        sid = seller["seller_id"]
        seller_rows.append({
            "seller_id": sid,
            "quality_type": seller["quality_type"],
            "cost": float(seller["cost"]),
            "num_package_units": int(seller["package"]["num_units"]),
            "package_total_count": int(seller["package"]["total_count"]),
            "score_ours_packaged": float(ours_scores[sid]),
            "score_no_packaging_raw": float(raw_scores[sid]),
            "score_seller_global": float(global_scores[sid]),
            "realized_test_acc": float(realized[sid]["test"]["accuracy"]),
            "realized_test_macro_f1": float(realized[sid]["test"]["macro_f1"]),
            "realized_test_auroc": float(realized[sid]["test"]["auroc"]),
            "gain_test_macro_f1": float(realized[sid]["gain_test_macro_f1"]),
            "packaging_time_sec": next(r["packaging_time_sec"] for r in package_records if r["seller_id"] == sid),
        })
    seller_df = pd.DataFrame(seller_rows).sort_values("seller_id")
    seller_df.to_csv(output_dir / "seller_protocol_results.csv", index=False, encoding="utf-8-sig")

    summary = {
        "config": vars(args),
        "splits": {k: str(v) if v is not None else None for k, v in split_paths.items()},
        "buyer_protocol": {
            "buyer_train_size": int(len(market["buyer_y"])),
            "buyer_degrade_stats": market["buyer_degrade_stats"],
            "anchor_stats": anchor_stats,
        },
        "weak_baseline": {
            "dev": weak_baseline["dev"],
            "test": weak_baseline["test"],
            "train_time_sec": float(weak_baseline["train_time_sec"]),
            "decision_threshold": float(weak_baseline["decision_threshold"]),
        },
        "offline_packaging": {
            "total_packaging_time_sec": float(total_packaging_time),
            "mean_packaging_time_sec": float(np.mean([r["packaging_time_sec"] for r in package_records])) if package_records else 0.0,
            "package_records": package_records,
        },
        "authorized_release": ours_release,
        "screening_alignment": alignment,
        "methods": methods,
        "seller_protocol_results": seller_rows,
    }

    json_dump(summary, output_dir / "protocol_market_summary.json")

    print("\n================ Final protocol summary ================")
    print(json.dumps({
        "weak_baseline_test": weak_baseline["test"],
        "methods": {
            k: {
                "selected_sellers": v["selected_sellers"],
                "online_selection_time_sec": v["online_selection_time_sec"],
                "gain_test_macro_f1": v["gain_test_macro_f1"],
                "post_purchase_test": None if v["post_purchase"] is None else v["post_purchase"]["test"],
            }
            for k, v in methods.items()
        },
        "screening_alignment": alignment,
    }, ensure_ascii=False, indent=2))
    print(f"[saved] {output_dir / 'protocol_market_summary.json'}")
    print(f"[saved] {output_dir / 'seller_protocol_results.csv'}")


if __name__ == "__main__":
    main()
