#!/usr/bin/env python3
"""B-E2 论文标准图：每个 backbone 一张三偏移强度 GE 对比图 + NTGE 矩阵 + GE@10000 退化图。

样式沿用 experiments/00_common_e0/scripts/plot_ge_paper.py：
  serif、9pt、3.5in 单栏宽、300dpi PNG + PDF、Y 轴 0-256 全量程、
  线性 + 对数双版（rank=0 在对数轴映射到 0.5 触底）、seed 带为 min-max。
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parents[1]
EVAL = HERE / "results/eval"
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

ARCH_LABEL = {"cnn": "Official CNN", "resnet1d": "ResNet1D", "mlp": "MLP"}
ARCH_COLOR = {"cnn": "#1f77b4", "resnet1d": "#d62728", "mlp": "#2ca02c"}
SHIFTS = ["fixed", "desync50", "desync100"]
SHIFT_LABEL = {"fixed": "Same domain (fixed)", "desync50": "desync50", "desync100": "desync100"}
SHIFT_COLOR = {"fixed": "#1f77b4", "desync50": "#ff7f0e", "desync100": "#c0392b"}


def load():
    data = {}
    for shift in SHIFTS:
        for f in sorted((EVAL / shift).glob("*.json")):
            if f.name == "eval_all.json":
                continue
            r = json.loads(f.read_text(encoding="utf-8"))
            data.setdefault((r["arch"], shift), []).append(r)
    return data


def grids(curves):
    xs = [0] + [p["traces"] for p in curves[0]["ge_curve"]]
    stack = []
    for c in curves:
        ys = [255] + [p["ge"] for p in c["ge_curve"]]
        stack.append(ys)
    arr = np.array(stack, dtype=float)
    return np.array(xs), arr.mean(0), arr.min(0), arr.max(0)


def ge_curves(data):
    for arch in ["cnn", "resnet1d", "mlp"]:
        for scale, suffix in [("linear", ""), ("log", "_log")]:
            fig, ax = plt.subplots(figsize=(3.5, 2.9))
            for shift in SHIFTS:
                curves = sorted(data.get((arch, shift), []), key=lambda r: r["seed"])
                if not curves:
                    continue
                xs, mean, lo, hi = grids(curves)
                if scale == "log":
                    f = lambda v: np.where(v > 0, v, 0.5)
                    ax.fill_between(xs, f(lo), f(hi), color=SHIFT_COLOR[shift], alpha=0.15, lw=0)
                    ax.plot(xs, f(mean), color=SHIFT_COLOR[shift], label=SHIFT_LABEL[shift])
                else:
                    ax.fill_between(xs, lo, hi, color=SHIFT_COLOR[shift], alpha=0.15, lw=0)
                    ax.plot(xs, mean, color=SHIFT_COLOR[shift], label=SHIFT_LABEL[shift])
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
            fig.legend(handles, labels, frameon=False, ncol=3, fontsize=6.8,
                       loc="lower center", bbox_to_anchor=(0.5, -0.01))
            fig.tight_layout(rect=(0, 0.09, 1, 1))
            fig.savefig(FIG / f"b_e2_ge_{arch}{suffix}.png", dpi=300)
            fig.savefig(FIG / f"b_e2_ge_{arch}{suffix}.pdf")
            plt.close(fig)


def ntge_matrix():
    """结构 x 偏移强度 NTGE 矩阵：柱为 3 seed 截尾均值，点上叠放每个 seed 的真实值。

    标记语义：实心圆 = 该 seed 的 GE=0 一直保持到 10000 条（stable）；
              空心圆 = 只在部分轨迹处瞬时归零（transient）；
              空心倒三角 = 10000 条内从未归零（censored 到 10k 作图）。
    """
    summary = json.loads((RES / "summary.json").read_text(encoding="utf-8"))
    archs = ["cnn", "resnet1d", "mlp"]
    width = 0.6
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    for i, arch in enumerate(archs):
        for j, shift in enumerate(SHIFTS):
            m = summary["metrics"][arch].get(shift)
            if not m:
                continue
            x = i + (j - 1) * 0.27
            v = m["ntge_first_censored_mean"]
            ax.bar(x, v, width=width * 0.32, color=SHIFT_COLOR[shift], alpha=0.35, linewidth=0)
            first = m["ntge_first"]
            stab = m["ntge_stable"]
            for k, (f, s) in enumerate(zip(first, stab)):
                y = 10000 if f is None else f
                dx = (k - 1) * 0.055
                if s is not None:
                    ax.plot(x + dx, y, "o", ms=3.0, color=SHIFT_COLOR[shift],
                            markeredgecolor="white", markeredgewidth=0.4, zorder=3)
                elif f is not None:
                    ax.plot(x + dx, y, "o", ms=3.0, mfc="white",
                            mec=SHIFT_COLOR[shift], mew=0.7, zorder=3)
                else:
                    ax.plot(x + dx, y, "v", ms=3.0, mfc="white",
                            mec=SHIFT_COLOR[shift], mew=0.7, zorder=3)
    ax.axhline(10000, color="0.55", lw=0.5, ls=":")
    ax.set_xticks(range(len(archs)))
    ax.set_xticklabels([ARCH_LABEL[a] for a in archs], fontsize=7.2)
    ax.set_ylabel("NTGE (traces)")
    ax.set_yscale("log")
    ax.set_ylim(100, 30000)
    ax.grid(alpha=0.3, axis="y")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", ls="", ms=3, color="0.35",
                      label="stable GE=0 to 10k"),
               Line2D([], [], marker="o", ls="", ms=3, mfc="white", mec="0.35",
                      label="transient GE=0 only"),
               Line2D([], [], marker="v", ls="", ms=3, mfc="white", mec="0.35",
                      label="never GE=0 in 10k")]
    fig.legend(handles=handles, frameon=False, fontsize=5.9, ncol=3,
               loc="lower center", bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    fig.savefig(FIG / "b_e2_ntge_matrix.png", dpi=300)
    fig.savefig(FIG / "b_e2_ntge_matrix.pdf")
    plt.close(fig)


def ge10000_vs_shift():
    summary = json.loads((RES / "summary.json").read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    x = np.arange(len(SHIFTS))
    for arch in ["cnn", "resnet1d", "mlp"]:
        ys = [summary["metrics"][arch][s]["ge_at_10000_mean"] for s in SHIFTS]
        es = [summary["metrics"][arch][s]["ge_at_10000_std"] for s in SHIFTS]
        ax.errorbar(x, ys, yerr=es, color=ARCH_COLOR[arch], marker="o", ms=3.2,
                    lw=1.3, capsize=2.5, label=ARCH_LABEL[arch])
    ax.axhline(127.5, color="0.55", lw=0.6, ls=":", label="random-guess rank (127.5)")
    ax.set_xticks(x)
    ax.set_xticklabels(["fixed", "desync50", "desync100"])
    ax.set_xlim(-0.3, 2.3)
    ax.set_ylim(0, 256)
    ax.set_yticks([0, 64, 128, 192, 256])
    ax.set_ylabel("GE @ 10,000 traces")
    ax.grid(alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=2, fontsize=6.5,
               loc="lower center", bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(FIG / "b_e2_ge10000_vs_shift.png", dpi=300)
    fig.savefig(FIG / "b_e2_ge10000_vs_shift.pdf")
    plt.close(fig)


if __name__ == "__main__":
    data = load()
    ge_curves(data)
    ntge_matrix()
    ge10000_vs_shift()
    print("figures written to", FIG)