#!/usr/bin/env python3
"""CDPA-MMD 训练动力学：CE loss、MMD（总/逐层）、source val acc、GE probe（500 条）。

每个偏移一条 3-seed 均值曲线，阴影为 min-max；实线=desync50，虚线=desync100。
输出 figures/a_e3_cdpa_dynamics.{png,pdf}（300 dpi，双栏宽 7.0 in）。
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)

HERE = Path(__file__).resolve().parents[1]
RES = HERE / "results"
FIG = HERE / "figures"
SEEDS = [42, 43, 44]
SHIFTS = ["desync50", "desync100"]
COLOR = {"desync50": "#1f77b4", "desync100": "#d62728"}
STYLE = {"desync50": "-", "desync100": "--"}
LAYER_COLOR = {"conv": "#4c78a8", "fc1": "#f58518", "fc2": "#54a24b"}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 8,
    "axes.linewidth": 0.8,
    "xtick.direction": "in", "ytick.direction": "in",
    "legend.frameon": False,
    "savefig.dpi": 300,
})


def load(shift, seed):
    p = RES / f"cdpa_{shift}_s{seed}" / "history.json"
    if not p.exists():
        return None
    h = json.loads(p.read_text(encoding="utf-8"))
    if len(h["epochs"]) < 75:   # 训练中的 run 不参与作图
        return None
    return h


def series(hist, key):
    return np.array([np.nan if x is None else x for x in hist.get(key, [])], dtype=float)


def main():
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.9), sharex=True)
    panels = [("cls_loss", "Training classification loss", True),
              ("val_acc", "Source validation accuracy", False)]

    for shift in SHIFTS:
        hists = [h for h in (load(shift, s) for s in SEEDS) if h is not None]
        if not hists:
            continue
        ep = np.array(hists[0]["epochs"], dtype=float)
        for ax, (key, label, logy) in zip(axes.flat[:2], panels):
            vals = np.vstack([series(h, key) for h in hists])
            mean, lo, hi = vals.mean(0), vals.min(0), vals.max(0)
            ax.fill_between(ep, lo, hi, color=COLOR[shift], alpha=0.13, lw=0)
            ax.plot(ep, mean, STYLE[shift], color=COLOR[shift], lw=1.2,
                    label=f"CDPA-MMD {shift}")
            ax.set_ylabel(label)
            if logy:
                ax.set_yscale("log")
        # MMD：总损失
        vals = np.vstack([series(h, "mmd_loss") for h in hists])
        axes[1, 0].fill_between(ep, vals.min(0), vals.max(0), color=COLOR[shift],
                                alpha=0.13, lw=0)
        axes[1, 0].plot(ep, vals.mean(0), STYLE[shift], color=COLOR[shift], lw=1.2,
                        label=f"CDPA-MMD {shift}")
        # GE probe：只画有值的 epoch（每 2 epoch 一次）
        xs, ys = [], []
        for i, e in enumerate(ep):
            v = [series(h, "ge_probe")[i] for h in hists]
            v = [x for x in v if not np.isnan(x)]
            if v:
                xs.append(e); ys.append(np.mean(v))
        axes[1, 1].plot(xs, ys, STYLE[shift], color=COLOR[shift], lw=1.2,
                        marker="o", ms=2.0, label=f"CDPA-MMD {shift}")

    # 第二行左图改为 MMD 总量；把 val_acc 放大到第一行右
    axes[0, 0].set_ylabel("Training classification loss")
    axes[0, 1].set_ylabel("Source validation accuracy")
    axes[1, 0].set_ylabel("MMD loss (conv + fc1 + fc2)")
    axes[1, 0].set_yscale("log")
    axes[1, 1].set_ylabel("GE probe rank (500 attack traces)")
    axes[1, 1].set_ylim(0, 256)
    for ax in axes.flat:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=5.8)
    fig.tight_layout()
    fig.savefig(FIG / "a_e3_cdpa_dynamics.png", dpi=300)
    fig.savefig(FIG / "a_e3_cdpa_dynamics.pdf")
    plt.close(fig)
    print("figures ->", FIG)


if __name__ == "__main__":
    main()
