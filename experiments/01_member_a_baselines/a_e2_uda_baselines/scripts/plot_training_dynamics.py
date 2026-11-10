#!/usr/bin/env python3
"""训练动力学图：DANN / CDAN 的 cls loss、val acc、批内域判别准确率、GE probe（500 条）。

每个方法 x 偏移一条均值曲线（3 seed），阴影为 min-max；虚线=desync100，实线=desync50。
输出 figures/a_e2_training_dynamics.{png,pdf}（300 dpi）。
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)

HERE = Path(__file__).resolve().parents[1]
RES = HERE / "results"
FIG = HERE / "figures"
SEEDS = [42, 43, 44]
SHIFTS = ["desync50", "desync100"]
METHODS = ["dann", "cdan"]
COLOR = {"dann": "#1f77b4", "cdan": "#d62728"}
STYLE = {"desync50": "-", "desync100": "--"}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 8,
    "axes.linewidth": 0.8,
    "xtick.direction": "in", "ytick.direction": "in",
    "legend.frameon": False,
    "savefig.dpi": 300,
})


def load(method, shift, seed):
    p = RES / f"{method}_{shift}_s{seed}" / "history.json"
    if not p.exists():
        return None
    h = json.loads(p.read_text(encoding="utf-8"))
    if len(h["epochs"]) < 75:   # 训练中的 run 不参与作图
        return None
    return h


def series(hist, key):
    v = hist[key]
    return np.array([np.nan if x is None else x for x in v], dtype=float)


def main():
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.9), sharex=True)
    panels = [("cls_loss", "Training classification loss", True),
              ("val_acc", "Source validation accuracy", False),
              ("dom_acc", "Batch domain accuracy (src/tgt mean)", False),
              ("ge_probe", "GE probe rank (500 attack traces)", False)]
    for ax, (key, ylabel, logy) in zip(axes.flat, panels):
        for method in METHODS:
            for shift in SHIFTS:
                hs = [load(method, shift, s) for s in SEEDS]
                hs = [h for h in hs if h is not None]
                if not hs:
                    continue
                n = min(len(h["epochs"]) for h in hs)
                ep = np.array(hs[0]["epochs"][:n])
                if key == "dom_acc":
                    arr = np.stack([(series(h, "d_acc_src")[:n] + series(h, "d_acc_tgt")[:n]) / 2
                                    for h in hs])
                else:
                    arr = np.stack([series(h, key)[:n] for h in hs])
                mean = np.nanmean(arr, axis=0)
                kw = dict(color=COLOR[method], ls=STYLE[shift], lw=1.2,
                          label=f"{method.upper()} {shift}")
                if key == "ge_probe":
                    kw.update(marker="o", ms=2.0, mew=0)
                ax.plot(ep, mean, **kw)
                if arr.shape[0] > 1:
                    ax.fill_between(ep, np.nanmin(arr, axis=0), np.nanmax(arr, axis=0),
                                    color=COLOR[method], alpha=0.12, lw=0)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.25, lw=0.4)
        if logy:
            ax.set_yscale("log")
        if key == "ge_probe":
            ax.set_ylim(0, 260)
        if key == "dom_acc":
            ax.axhline(0.5, color="0.4", lw=0.6, ls=":")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=7)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    FIG.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"a_e2_training_dynamics.{ext}", bbox_inches="tight")
    print("wrote figures/a_e2_training_dynamics.{png,pdf}")


if __name__ == "__main__":
    main()
