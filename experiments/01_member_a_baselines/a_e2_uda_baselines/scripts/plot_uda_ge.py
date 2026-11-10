#!/usr/bin/env python3
"""A-E2 论文标准图：每个偏移强度一张三方法 GE 对比图 + NTGE 柱状 + GE@10k 柱状。

样式沿用 experiments/00_common_e0/scripts/plot_ge_paper.py 与
experiments/02_member_b_method/b_e2_architecture_shift/scripts/plot_b_e2_ge.py：
serif、9pt、3.5in 单栏、300dpi PNG+PDF、Y 轴 0-256 全量程、线性+对数双版
（rank=0 在对数轴映射到 0.5 触底）、seed 带为 min-max。
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

METHODS = ["source_only", "dann", "cdan"]
LABEL = {"source_only": "Source Only", "dann": "DANN", "cdan": "CDAN"}
COLOR = {"source_only": "#1f77b4", "dann": "#ff7f0e", "cdan": "#2ca02c"}
SHIFTS = ["desync50", "desync100"]


def load():
    data = {}
    for shift in SHIFTS:
        for f in sorted((EVAL / shift).glob("*.json")):
            if f.name == "eval_all.json":
                continue
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
                    ax.fill_between(xs, f(lo), f(hi), color=COLOR[method], alpha=0.13, lw=0)
                    ax.plot(xs, f(mean), color=COLOR[method], label=LABEL[method])
                else:
                    ax.fill_between(xs, lo, hi, color=COLOR[method], alpha=0.13, lw=0)
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
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles, labels, frameon=False, ncol=3, fontsize=6.8,
                       loc="lower center", bbox_to_anchor=(0.5, -0.01))
            fig.tight_layout(rect=(0, 0.09, 1, 1))
            fig.savefig(FIG / f"a_e2_ge_{shift}{suffix}.png", dpi=300)
            fig.savefig(FIG / f"a_e2_ge_{shift}{suffix}.pdf")
            plt.close(fig)


def ntge_bars():
    summary = json.loads((RES / "summary.json").read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    width = 0.26
    x = np.arange(len(SHIFTS))
    for i, method in enumerate(METHODS):
        vals, errs, hatches = [], [], []
        for shift in SHIFTS:
            m = summary["metrics"].get(method, {}).get(shift)
            if not m:
                vals.append(np.nan); errs.append(0); hatches.append(""); continue
            vals.append(m["ntge_first_mean"])
            errs.append(m["ntge_first_std"])
            hatches.append("" if m["ntge_first_reached"] == m["n_runs"] else "///")
        bars = ax.bar(x + (i - 1) * width, vals, width, yerr=errs, capsize=2,
                      color=COLOR[method], alpha=0.85 if method != "source_only" else 0.6,
                      edgecolor=COLOR[method], linewidth=0.6,
                      error_kw={"elinewidth": 0.6, "ecolor": "0.4"},
                      label=LABEL[method])
        for b, h in zip(bars, hatches):
            if h:
                b.set_hatch(h)
                b.set_facecolor("white")
    ax.axhline(10000, color="0.55", lw=0.5, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s} (N_adapt=45k)" for s in SHIFTS], fontsize=7.2)
    ax.set_ylabel("NTGE (traces, censored at 10k)")
    ax.set_yscale("log")
    ax.set_ylim(100, 30000)
    ax.grid(alpha=0.3, axis="y")
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Patch(facecolor="white", edgecolor="0.4", hatch="///"))
    labels.append("not all seeds reached GE=0")
    fig.legend(handles, labels, frameon=False, ncol=2, fontsize=6.4,
               loc="lower center", bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.savefig(FIG / "a_e2_ntge_bars.png", dpi=300)
    fig.savefig(FIG / "a_e2_ntge_bars.pdf")
    plt.close(fig)


def ge10000_bars():
    summary = json.loads((RES / "summary.json").read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    width = 0.26
    x = np.arange(len(SHIFTS))
    for i, method in enumerate(METHODS):
        ys, es = [], []
        for shift in SHIFTS:
            m = summary["metrics"].get(method, {}).get(shift)
            ys.append(np.nan if not m else m["ge_at_10000_mean"])
            es.append(0 if not m else m["ge_at_10000_std"])
        ax.bar(x + (i - 1) * width, ys, width, yerr=es, capsize=2,
               color=COLOR[method], alpha=0.85, edgecolor="white", linewidth=0.4,
               error_kw={"elinewidth": 0.6, "ecolor": "0.4"}, label=LABEL[method])
    ax.axhline(127.5, color="0.55", lw=0.6, ls=":", label="random-guess rank (127.5)")
    ax.set_xticks(x)
    ax.set_xticklabels(SHIFTS)
    ax.set_ylim(0, 256)
    ax.set_yticks([0, 64, 128, 192, 256])
    ax.set_ylabel("GE @ 10,000 traces")
    ax.grid(alpha=0.3, axis="y")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=2, fontsize=6.4,
               loc="lower center", bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    fig.savefig(FIG / "a_e2_ge10000_bars.png", dpi=300)
    fig.savefig(FIG / "a_e2_ge10000_bars.pdf")
    plt.close(fig)


if __name__ == "__main__":
    data = load()
    ge_curves(data)
    ntge_bars()
    ge10000_bars()
    print("figures written to", FIG)
