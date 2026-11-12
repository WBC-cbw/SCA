#!/usr/bin/env python3
"""B-E3/B-E4 共享组件：熵权重、target_adapt 诊断标签、训练期监控统计。

严格沿用 A-E2 的 uda_common（UDANet / GRL / RandomMultilinearMap / 数据划分），
不复制第二份实现，保证与 DANN/CDAN 基线是同一套代码路径。

熵权重公式（docs/research/主方法.md）：
    H(g) = -sum_k g_k log g_k
    w(g) = 1 - H(g) / log(256)
只此一个版本；experiments/legacy/CDAN/dann.py 的指数权重不使用。
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[2]
A_E2_SCRIPTS = ROOT / "experiments/01_member_a_baselines/a_e2_uda_baselines/scripts"
if str(A_E2_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(A_E2_SCRIPTS))

from uda_common import E0, UDANet, accuracy, domain_accuracy, predict_probs  # noqa: E402

SBOX = np.asarray(E0.AES_Sbox, dtype=np.uint8)
LOG256 = math.log(256.0)
COLLAPSE_WARN = 0.30
COLLAPSE_HARD = 0.50


def entropy_nats(probs: np.ndarray) -> np.ndarray:
    p = np.clip(probs, 1e-12, 1.0)
    return -(p * np.log(p)).sum(1)


def entropy_weight(probs: np.ndarray) -> np.ndarray:
    """w = 1 - H/log(256)，取值 [0, 1]。"""
    return 1.0 - entropy_nats(probs) / LOG256


def load_target_adapt_labels(h5_path: str, n: int, byte: int = 2) -> np.ndarray:
    """target_adapt 前 n 条的标签，仅用于事后校准诊断（绝不进损失/模型选择）。"""
    with h5py.File(h5_path, "r") as f:
        md = f["Profiling_traces/metadata"][:n]
        key = np.asarray(f["Profiling_traces/metadata"]["key"])
    pt = np.asarray(md["plaintext"])
    kb = int(key[0, byte])
    return SBOX[np.bitwise_xor(pt[:, byte], kb)].astype(np.int64)


def monitor_stats(probs: np.ndarray) -> dict:
    """训练期监控：top1 类别占比、预测熵、熵权重分布。"""
    pred = probs.argmax(1)
    counts = np.bincount(pred, minlength=256)
    top = np.argsort(counts)[::-1][:5]
    w = entropy_weight(probs)
    ent = entropy_nats(probs)
    return {
        "n_monitor": int(len(pred)),
        "top1_share": float(counts.max() / len(pred)),
        "top1_class": int(top[0]),
        "top5_classes": [[int(c), int(counts[c])] for c in top],
        "unique_pred_classes": int((counts > 0).sum()),
        "pred_entropy_mean": float(ent.mean()),
        "pred_entropy_p50": float(np.median(ent)),
        "max_conf_mean": float(probs.max(1).mean()),
        "w_mean": float(w.mean()),
        "w_std": float(w.std()),
        "w_p10": float(np.percentile(w, 10)),
        "w_p50": float(np.percentile(w, 50)),
        "w_p90": float(np.percentile(w, 90)),
    }


def collapse_level(top1_share: float) -> str:
    if top1_share > COLLAPSE_HARD:
        return "COLLAPSE"
    if top1_share > COLLAPSE_WARN:
        return "WARN"
    return "OK"


def calibration_stats(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> dict:
    """与 A-E2 analyze_confidence.py 同口径：ECE 15 bin、confident-wrong、熵。"""
    conf = probs.max(1)
    pred = probs.argmax(1)
    wrong = pred != labels
    ent = entropy_nats(probs)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(labels)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if hi < 1 else (conf >= lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        e += abs((pred[m] == labels[m]).mean() - conf[m].mean()) * m.sum() / total
    counts = np.bincount(pred, minlength=256)
    return {
        "n": int(total),
        "accuracy": float((~wrong).mean()),
        "mean_max_softmax": float(conf.mean()),
        "mean_entropy_nats": float(ent.mean()),
        "ece_15": float(e),
        "frac_conf_ge_0.9": float((conf >= 0.9).mean()),
        "frac_confwrong_ge_0.9": float(((conf >= 0.9) & wrong).mean()),
        "frac_confwrong_ge_0.99": float(((conf >= 0.99) & wrong).mean()),
        "frac_conf_lt_0.1": float((conf < 0.1).mean()),
        "top1_share": float(counts.max() / total),
        "top1_class": int(counts.argmax()),
        "unique_pred_classes": int((counts > 0).sum()),
    }
