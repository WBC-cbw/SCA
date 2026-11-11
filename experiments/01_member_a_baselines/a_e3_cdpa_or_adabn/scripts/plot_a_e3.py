#!/usr/bin/env python3
"""A-E3 论文标准图：四方法 GE 对比（每偏移一张，线性+对数）+ NTGE 柱状 + GE@10k 柱状。

样式沿用 experiments/00_common_e0/scripts/plot_ge_paper.py、
experiments/02_member_b_method/b_e2_architecture_shift/scripts/plot_b_e2_ge.py 与
experiments/01_member_a_baselines/a_e2_uda_baselines/scripts/plot_uda_ge.py：
serif、9pt、3.5in 单栏宽、300dpi PNG+PDF、Y 轴 0-256 全量程、
对数版 rank=0 映射到 0.5 触底、seed 带为 min-max。
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

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

METHODS = ["source_only", "dann", "cdan", "cdpa_mmd"]
LABEL = {"source_only": "Source Only", "dann": "DANN", "cdan": "CDAN",
         "cdpa_mmd": "CDPA-MMD"}
COLOR = {"source_only": "#1f77b4", "dann": "#ff7f0e", "cdan": "#2ca02c",
         "cdpa_mmd": "#d62728"}
SHIFTS = ["desync50", "desync100"]
SHIFT_TITLE = {"desync50": "desync50", "desync100": "desync100"}


def load():
    data = {}
    for shift in SHIFTS:
        for f in sorted((EVAL / shift).glob("*.json")):
            r = json.loads(f.read_text(encoding="utf-8"))
            data.setdefault((r["method"], shift), []).append(r)
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
    for shift in SHIFTS:
        for scale, suffix in [("linear", ""), ("log", "_log")]:
            fig, ax = plt.subplots(figsize=(3.5, 2.9))
            for method in METHODS:
                curves = sorted(data.get((method, shift), []), key=lambda r: r["seed"])
                if not curves:
                    continue
                xs, mean, lo, hi = grids(curves)
                if scale == "log":
                    f = lambda v: np.where(v > 0, v, 0.5)
                    ax.fill_between(xs, f(lo), f(hi), color=COLOR[method], alpha=0.12, lw=0)
                    ax.plot(xs, f(mean), color=COLOR[method], label=LABEL[method])
                else:
                    ax.fill_between(xs, lo, hi, color=COLOR[method], alpha=0.12, lw=0)
                    ax.plot(xs, mean, color=COLOR[method], label=LABEL[method])
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
            ax.set_title(SHIFT_TITLE[shift], pad=3)
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles, labels, frameon=False, ncol=2, fontsize=6.8,
                       loc="lower center", bbox_to_anchor=(0.5, -0.02))
            fig.tight_layout(rect=(0, 0.10, 1, 1))
            fig.savefig(FIG / f"a_e3_ge_{shift}{suffix}.png", dpi=300)
            fig.savefig(FIG / f"a_e3_ge_{shift}{suffix}.pdf")
            plt.close(fig)


def ntge_bars():
    summary = json.loads((RES / "summary.json").read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(3.5, 3.15))
    width = 0.2
    x = np.arange(len(SHIFTS))
    for i, method in enumerate(METHODS):
        vals, errs, stable = [], [], []
        for shift in SHIFTS:
            m = summary["metrics"].get(method, {}).get(shift)
            if not m:
                vals.append(np.nan); errs.append(0); stable.append(""); continue
            vals.append(m["ntge_first_mean"])
            errs.append(m["ntge_first_std"])
            stable.append(f"{m['ntge_stable_reached']}/{m['n_runs']}")
        xs = x + (i - 1.5) * width
        ax.bar(xs, vals, width, yerr=errs, capsize=1.8,
               color=COLOR[method], alpha=0.80,
               edgecolor=COLOR[method], linewidth=0.7,
               error_kw={"elinewidth": 0.6, "ecolor": "0.25"},
               label=LABEL[method])
        for xi, v, lab in zip(xs, vals, stable):
            if lab and np.isfinite(v):
                ax.text(xi, v * 0.55, lab, rotation=90, ha="center", va="center",
                        fontsize=5.4, color="0.12", fontweight="bold")
    ax.set_yscale("log")
    ax.set_ylim(100, 40000)
    ax.axhline(10000, color="0.5", ls=":", lw=0.8, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(SHIFT_TITLE[s] for s in SHIFTS)
    ax.set_ylabel("NTGE (traces)")
    ax.set_xlim(-0.55, 1.55)
    ax.grid(alpha=0.3, axis="y")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=4, fontsize=6.5,
               loc="lower center", bbox_to_anchor=(0.5, 0.072))
    fig.text(0.5, 0.006,
             "in-bar label = seeds reaching stable GE=0 (of 3);  dotted = 10k censor;\n"
             "bar = mean first-GE=0 point, error bar = sd over 3 seeds",
             ha="center", va="bottom", fontsize=5.0, color="0.30", linespacing=1.25)
    fig.tight_layout(rect=(0, 0.155, 1, 1))
    fig.savefig(FIG / "a_e3_ntge.png", dpi=300)
    fig.savefig(FIG / "a_e3_ntge.pdf")
    plt.close(fig)


def ge10_bars():
    summary = json.loads((RES / "summary.json").read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    width = 0.2
    x = np.arange(len(SHIFTS))
    for i, method in enumerate(METHODS):
        vals = [summary["metrics"].get(method, {}).get(s, {}).get("ge_at_10000_mean", np.nan)
                for s in SHIFTS]
        errs = [summary["metrics"].get(method, {}).get(s, {}).get("ge_at_10000_std", 0.0)
                for s in SHIFTS]
        ax.bar(x + (i - 1.5) * width, vals, width, yerr=errs, capsize=1.8,
               color=COLOR[method], alpha=0.75, edgecolor=COLOR[method], linewidth=0.7,
               error_kw={"elinewidth": 0.6, "ecolor": "0.35"}, label=LABEL[method])
    ax.axhline(127.5, color="0.5", ls=":", lw=0.8, zorder=1)
    ax.text(1.55, 131, "random-guess rank (127.5)", fontsize=6, color="0.35",
            ha="right", va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels(SHIFT_TITLE[s] for s in SHIFTS)
    ax.set_ylabel("GE @ 10,000 traces")
    ax.set_ylim(0, 256)
    ax.set_yticks([0, 64, 128, 192, 256])
    ax.set_xlim(-0.55, 1.55)
    ax.grid(alpha=0.3, axis="y")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=4, fontsize=6.5,
               loc="lower center", bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    fig.savefig(FIG / "a_e3_ge10k.png", dpi=300)
    fig.savefig(FIG / "a_e3_ge10k.pdf")
    plt.close(fig)


if __name__ == "__main__":
    d = load()
    ge_curves(d)
    ntge_bars()
    ge10_bars()
    print("figures ->", FIG)
