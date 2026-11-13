#!/usr/bin/env python3
"""B-E5 共享组件：源域温度校准 + 校准后的熵权重。

与 B-E3 的唯一方法差异是权重函数：
    full : w = 1 - H(softmax(g)) / log(256)
    calib: T = argmin_T NLL(softmax(g / T), y_source_val)
           w = 1 - H(softmax(g / T)) / log(256)
条件表示 T(f,g)、GRL、lambda_adv、主干、数据划分、训练预算均与 full 完全一致；
domain_logits 内部的条件输入仍用未校准的 softmax(g)，保证唯一变量是权重。
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

LOG256 = math.log(256.0)
SBOX = None  # 由调用方从 uda_common 取，避免重复定义


def softmax_np(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def nll_np(logits: np.ndarray, labels: np.ndarray, temperature: float) -> float:
    """数值稳定的 NLL：log_softmax(logits/T)。"""
    z = logits / float(temperature)
    z = z - z.max(1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(1, keepdims=True))
    return float(-logp[np.arange(len(labels)), labels].mean())


def accuracy_np(logits: np.ndarray, labels: np.ndarray) -> float:
    return float((logits.argmax(1) == labels).mean())


def ece_np(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """与 A-E2/B-E3 同口径的 15-bin ECE。"""
    conf = probs.max(1)
    pred = probs.argmax(1)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(labels)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if hi < 1 else (conf >= lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        e += abs((pred[m] == labels[m]).mean() - conf[m].mean()) * m.sum() / total
    return float(e)


def fit_temperature(logits: np.ndarray, labels: np.ndarray,
                    lo: float = 0.05, hi: float = 20.0, n_grid: int = 400,
                    n_refine: int = 3) -> dict:
    """在带标签的 source_val 上最小化 NLL，拟合标量温度。

    先对数网格粗搜，再在最优格附近做三轮局部加密（每轮 101 点），确定性可复现。
    返回 T 与拟合前后的 NLL/ECE/accuracy。
    """
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    grid = np.geomspace(lo, hi, n_grid)
    vals = np.array([nll_np(logits, labels, t) for t in grid])
    i = int(vals.argmin())
    best = float(grid[i])
    step = (math.log(hi) - math.log(lo)) / (n_grid - 1)
    for _ in range(n_refine):
        cand = np.geomspace(best * math.exp(-2 * step), best * math.exp(2 * step), 101)
        cvals = np.array([nll_np(logits, labels, t) for t in cand])
        j = int(cvals.argmin())
        best = float(cand[j])
        step = (math.log(cand[-1]) - math.log(cand[0])) / 100
    p1 = softmax_np(logits)
    p2 = softmax_np(logits / best)
    return {
        "temperature": best,
        "nll_before": nll_np(logits, labels, 1.0),
        "nll_after": nll_np(logits, labels, best),
        "ece_before": ece_np(p1, labels),
        "ece_after": ece_np(p2, labels),
        "accuracy": accuracy_np(logits, labels),
        "n_val": int(len(labels)),
        "grid": {"lo": lo, "hi": hi, "n_grid": n_grid, "refine_rounds": n_refine},
        "fit_objective": "mean NLL of softmax(logits/T) on source_val",
    }


def apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    """由已 softmax 的概率恢复 log-softmax 后重新按 T 归一化（等价于 softmax(logits/T)）。"""
    logp = np.log(np.clip(probs, 1e-12, 1.0)) / float(temperature)
    logp = logp - logp.max(1, keepdims=True)
    e = np.exp(logp)
    return e / e.sum(1, keepdims=True)


def entropy_np(probs: np.ndarray) -> np.ndarray:
    p = np.clip(probs, 1e-12, 1.0)
    return -(p * np.log(p)).sum(1)


def weight_np(probs: np.ndarray) -> np.ndarray:
    return 1.0 - entropy_np(probs) / LOG256


def weight_stats(probs: np.ndarray, prefix: str = "w") -> dict:
    w = weight_np(probs)
    return {
        f"{prefix}_mean": float(w.mean()),
        f"{prefix}_std": float(w.std()),
        f"{prefix}_p10": float(np.percentile(w, 10)),
        f"{prefix}_p50": float(np.percentile(w, 50)),
        f"{prefix}_p90": float(np.percentile(w, 90)),
        f"{prefix}_frac_gt_0.99": float((w > 0.99).mean()),
        f"{prefix}_frac_lt_0.5": float((w < 0.5).mean()),
    }


def monitor_stats_calib(probs_bare: np.ndarray, temperature: float) -> dict:
    """训练期监控：既记录未校准量，也记录校准后的权重分布。"""
    from cdan_sca_common import monitor_stats as _mon
    base = _mon(probs_bare)
    cal = apply_temperature(probs_bare, temperature)
    ent_c = entropy_np(cal)
    base.update({
        "temperature": float(temperature),
        "pred_entropy_calib_mean": float(ent_c.mean()),
        "pred_entropy_calib_p50": float(np.median(ent_c)),
        "max_conf_calib_mean": float(cal.max(1).mean()),
    })
    base.update(weight_stats(cal, "w"))            # w_* 覆盖为校准后的权重
    base.update(weight_stats(probs_bare, "w_bare"))
    base["pred_entropy_bare_mean"] = base["pred_entropy_mean"]
    return base
