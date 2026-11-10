#!/usr/bin/env python3
"""B-E1 论文标准图：三种 backbone 的同域 GE 曲线对比。

样式沿用 experiments/00_common_e0/scripts/plot_ge_paper.py：
  serif、9pt、3.5in 单栏宽、300dpi PNG + PDF 矢量、Y 轴 0-256 全量程、
  线性 + 对数双版（rank=0 在对数轴映射到 0.5 触底）。
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parents[1]
RES = HERE / "results"
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.4,
    "grid.linewidth": 0.4,
})

LABEL = {"cnn": "Official CNN", "resnet1d": "ResNet1D", "mlp": "MLP"}
COLOR = {"cnn": "#1f77b4", "resnet1d": "#d62728", "mlp": "#2ca02c"}


def load():
    data = {}
    for f in sorted((RES / "eval").glob("*.json")):
        if f.name in ("eval_all.json",):
            continue
        r = json.loads(f.read_text(encoding="utf-8"))
        arch = r["run"].split("_s")[0]
        data.setdefault(arch, []).append(r)
    return data


def grids(curves):
    """把若干条 (traces, ge) 曲线对齐到公共 x 网格并取均值/min/max。"""
    xs = [0] + [p["traces"] for p in curves[0]["ge_curve"]]
    stack = []
    for c in curves:
        ys = [255] + [p["ge"] for p in c["ge_curve"]]
        stack.append(ys)
    arr = np.array(stack, dtype=float)
    return np.array(xs), arr.mean(0), arr.min(0), arr.max(0)


def main():
    data = load()
    order = [a for a in ["cnn", "resnet1d", "mlp"] if a in data]

    for scale, suffix in [("linear", ""), ("log", "_log")]:
        fig, ax = plt.subplots(figsize=(3.5, 2.9))
        for arch in order:
            xs, mean, lo, hi = grids(data[arch])
            if scale == "log":
                f = lambda v: np.where(v > 0, v, 0.5)
                ax.fill_between(xs, f(lo), f(hi), color=COLOR[arch], alpha=0.15, lw=0)
                ax.plot(xs, f(mean), color=COLOR[arch], label=LABEL.get(arch, arch))
            else:
                ax.fill_between(xs, lo, hi, color=COLOR[arch], alpha=0.15, lw=0)
                ax.plot(xs, mean, color=COLOR[arch], label=LABEL.get(arch, arch))
        if scale == "log":
            ax.set_yscale("log")
            ax.set_ylim(0.5, 256)
            ax.set_yticks([0.5, 1, 4, 16, 64, 256])
            ax.set_yticklabels(["0", "1", "4", "16", "64", "256"])
        else:
            ax.set_ylim(0, 256)
            ax.set_yticks([0, 64, 128, 192, 256])
        ax.set_xlim(0, 10000)
        ax.set_xlabel("Number of attack traces")
        ax.set_ylabel("Key rank")
        ax.grid(alpha=0.3)
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, frameon=False, ncol=3, fontsize=7,
                   loc="lower center", bbox_to_anchor=(0.5, -0.01))
        fig.tight_layout(rect=(0, 0.09, 1, 1))
        fig.savefig(FIG / f"b_e1_ge_curves{suffix}.png", dpi=300)
        fig.savefig(FIG / f"b_e1_ge_curves{suffix}.pdf")
        plt.close(fig)

    # NTGE 汇总条形图：按结构分组，未达 GE=0（10k 轨迹内）的 run 画成剖线柱
    from matplotlib.patches import Patch
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    seeds = ["42", "43", "44"]
    width = 0.26
    for i, arch in enumerate(order):
        runs = {r["run"].split("_s")[-1]: r for r in data[arch]}
        for j, s in enumerate(seeds):
            r = runs.get(s)
            if r is None:
                continue
            x = i + (j - 1) * width
            if r["ntge"]:
                ax.bar(x, r["ntge"], width=width, color=COLOR[arch],
                       alpha=1.0 - 0.25 * j, linewidth=0)
            else:
                ax.bar(x, 10000, width=width, facecolor="white",
                       edgecolor=COLOR[arch], hatch="////", linewidth=0.6)
    ax.axhline(10000, color="0.55", lw=0.5, ls=":")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([LABEL.get(a, a) for a in order], fontsize=7.5)
    ax.set_ylabel("NTGE (traces)")
    ax.set_yscale("log")
    ax.set_ylim(100, 30000)
    ax.grid(alpha=0.3, axis="y")
    handles = [Patch(facecolor="0.75", label="GE=0 reached"),
               Patch(facecolor="white", edgecolor="0.4", hatch="////",
                     label="no GE=0 in 10k")]
    ax.legend(handles=handles, frameon=False, fontsize=6.2, ncol=2,
              loc="upper center", bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout()
    fig.savefig(FIG / "b_e1_ntge.png", dpi=300)
    fig.savefig(FIG / "b_e1_ntge.pdf")
    plt.close(fig)
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
