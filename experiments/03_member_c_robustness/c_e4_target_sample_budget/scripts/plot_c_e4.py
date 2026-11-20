#!/usr/bin/env python3
"""C-E4 绘图：预算—NTGE / 预算—N_total 曲线、各预算档 GE 对比、稳定率、坍塌与校准。

论文标准：3.5 英寸单栏宽、衬线字体、300 dpi PNG + PDF 矢量；
GE 图 Y 轴线性 0–256（rank=0 映射 0.5）与对数双版，沿用 E0/B-E3 样式。
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parents[1]
RES = Path(os.environ.get("C_E4_RESULTS", str(HERE / "results")))
FIG = Path(os.environ.get("C_E4_FIGURES", str(HERE / "figures")))
FIG.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9.5,
    "legend.fontsize": 6.6, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.linewidth": 0.7, "grid.linewidth": 0.4,
    "lines.linewidth": 0.9, "lines.markersize": 3.2,
    "savefig.dpi": 300, "figure.dpi": 150,
})

METHODS = ["source_only", "dann", "cdan_sca", "cdpa_mmd"]
LABELS = {"source_only": "Source Only", "dann": "DANN", "cdan_sca": "CDAN-SCA full",
          "cdpa_mmd": "CDPA-MMD"}
COLORS = {"source_only": "#1f77b4", "dann": "#ff7f0e", "cdan_sca": "#6a3d9a",
          "cdpa_mmd": "#d62728"}
MARKERS = {"source_only": "s", "dann": "^", "cdan_sca": "D", "cdpa_mmd": "o"}
BUDGETS = [50, 500, 5000, 45000]
CENSOR = 10000


def load():
    ev = json.loads((RES / "eval/eval_all.json").read_text(encoding="utf-8"))
    diag = json.loads((RES / "diagnostics.json").read_text(encoding="utf-8"))
    return ev, diag


def cell(ev, shift, method, budget):
    return [r for r in ev if r["shift"] == shift and r["method"] == method
            and r["budget"] == budget]


def mean_sd(vals):
    v = np.asarray([CENSOR if x is None else x for x in vals], dtype=float)
    return v.mean(), v.std(ddof=0)


def budget_curve(ev, shift, key, budget_key=None):
    xs, series = [], {}
    for m in METHODS:
        pts = []
        for b in BUDGETS:
            rows = cell(ev, shift, m, b)
            if not rows:
                continue
            if m == "source_only":
                vals = [r[key] for r in rows]
                if m not in series:
                    series[m] = {"x": 0, "mean": np.mean([CENSOR if v is None else v for v in vals]),
                                 "sd": 0.0, "censored": 0}
                continue
            vals = [r[key] for r in rows]
            cap = (CENSOR + b) if key.startswith("n_total") else CENSOR
            v = np.asarray([cap if x is None else x for x in vals], dtype=float)
            pts.append({"x": b, "mean": float(v.mean()), "sd": float(v.std(ddof=0)),
                        "censored": int(sum(x is None for x in vals)), "n": len(vals)})
        if pts:
            series[m] = pts
    return series


def draw_budget(series, ylabel, fname, ylim=None, hline_so=True, censor_line=None,
                note=None, yticks=None, yticklabels=None):
    fig, ax = plt.subplots(figsize=(3.5, 3.15))
    for m in METHODS:
        s = series.get(m)
        if not s:
            continue
        if m == "source_only":
            ax.axhline(s["mean"], color=COLORS[m], lw=0.9, ls=(0, (4, 2)),
                       label=f"{LABELS[m]} (no adaptation)")
            continue
        x = np.array([p["x"] for p in s])
        y = np.array([p["mean"] for p in s])
        e = np.array([p["sd"] for p in s])
        ax.errorbar(x, y, yerr=e, color=COLORS[m], marker=MARKERS[m], ms=3.4,
                    lw=0.9, capsize=2, label=LABELS[m])
        for p in s:
            if p["censored"]:
                ax.plot(p["x"], p["mean"], marker=MARKERS[m], ms=4.6, mfc="white",
                        mec=COLORS[m], mew=0.8, zorder=5)
    if censor_line is not None:
        ax.axhline(censor_line, color="0.45", lw=0.6, ls=(0, (1, 2)),
                   label="censor (10k)")
    ax.set_xscale("log")
    ax.set_xticks([50, 500, 5000, 45000])
    ax.set_xticklabels(["50", "500", "5000", "45k"])
    ax.set_xlabel("Target adaptation traces")
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    else:
        ax.set_yscale("log")
    if yticks:
        from matplotlib.ticker import NullFormatter, NullLocator, FixedLocator
        ax.yaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_yticks(yticks)
        if yticklabels:
            ax.set_yticklabels(yticklabels)
    ax.grid(alpha=0.3)
    ax.legend(loc="best", frameon=False, fontsize=6.2)
    if note:
        ax.text(0.02, 0.02, note, transform=ax.transAxes, fontsize=5.6,
                color="0.35", va="bottom", ha="left")
    fig.tight_layout(pad=0.3)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{fname}.{ext}", bbox_inches="tight")
    plt.close(fig)


def plot_ge_grid(ev):
    for shift in sorted({r["shift"] for r in ev}):
      for b in BUDGETS:
        if not cell(ev, shift, "cdpa_mmd", b):
            continue
        for logy in (False, True):
            fig, ax = plt.subplots(figsize=(3.5, 3.05))
            for m in METHODS:
                rows = cell(ev, shift, m, b)
                if not rows:
                    continue
                curves = np.array([[pt["ge"] for pt in r["ge_curve"]] for r in rows], dtype=float)
                xs = np.array([pt["traces"] for pt in rows[0]["ge_curve"]])
                mean = curves.mean(0)
                lo, hi = curves.min(0), curves.max(0)
                y = np.where(mean > 0, mean, 0.5) if logy else mean
                ax.plot(xs, y, color=COLORS[m], label=LABELS[m])
                if len(rows) > 1:
                    ax.fill_between(xs, np.where(lo > 0, lo, 0.5) if logy else lo,
                                    np.where(hi > 0, hi, 0.5) if logy else hi,
                                    color=COLORS[m], alpha=0.10, linewidth=0)
            ax.set_xlabel("Number of attack traces")
            ax.set_ylabel("Key rank")
            ax.set_xlim(0, 10000)
            if logy:
                ax.set_yscale("log")
                ax.set_ylim(0.5, 256)
                ax.set_yticks([0.5, 1, 4, 16, 64, 256])
                ax.set_yticklabels(["0", "1", "4", "16", "64", "256"])
            else:
                ax.set_ylim(0, 256)
                ax.set_yticks([0, 64, 128, 192, 256])
            ax.set_title(f"{shift}, target_adapt={('45k' if b == 45000 else b)}", pad=2)
            so = cell(ev, shift, "source_only", 0)
            if so:
                c0 = np.array([[pt["ge"] for pt in r0["ge_curve"]] for r0 in so], dtype=float)
                m0 = c0.mean(0)
                ax.plot(np.array([pt["traces"] for pt in so[0]["ge_curve"]]),
                        np.where(m0 > 0, m0, 0.5) if logy else m0,
                        color="0.35", lw=0.8, ls=(0, (3, 2)), label="Source Only")
            ax.grid(alpha=0.3)
            ax.legend(loc="upper right", frameon=False)
            fig.tight_layout(pad=0.3)
            suf = "_log" if logy else ""
            for ext in ("png", "pdf"):
                fig.savefig(FIG / f"c_e4_ge_{shift}_b{b}{suf}.{ext}", bbox_inches="tight")
            plt.close(fig)


def plot_stability(ev):
  for shift in sorted({r["shift"] for r in ev}):
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    for m in ["cdpa_mmd", "dann", "cdan_sca"]:
        rates = []
        for b in BUDGETS:
            rows = cell(ev, shift, m, b)
            rates.append(0 if not rows else sum(r["ntge_stable"] is not None for r in rows) / len(rows))
        ax.plot(BUDGETS, rates, marker=MARKERS[m], color=COLORS[m], label=LABELS[m])
    ax.set_xscale("log")
    ax.set_xticks(BUDGETS)
    ax.set_xticklabels(["50", "500", "5000", "45k"])
    ax.set_yticks([0, 0.5, 1 / 3, 2 / 3, 1])
    ax.set_yticklabels(["0", "0.5", "1/3", "2/3", "1"])
    ax.set_xlabel("Target adaptation traces")
    ax.set_ylabel("Stable GE=0 rate (3 seeds)")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.3)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout(pad=0.3)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"c_e4_stability_{shift}.{ext}", bbox_inches="tight")
    plt.close(fig)


def plot_collapse(diag, shift="desync50"):
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 4.3), sharex=True)
    ax = axes[0]
    for m in ["cdpa_mmd", "dann", "cdan_sca"]:
        ys, lo, hi = [], [], []
        for b in BUDGETS:
            rows = [d for d in diag if d["shift"] == shift and d["method"] == m
                    and d["budget"] == b and "max_top1" in d]
            if not rows:
                ys.append(np.nan); lo.append(np.nan); hi.append(np.nan); continue
            v = [d["max_top1"] for d in rows]
            ys.append(np.mean(v)); lo.append(np.min(v)); hi.append(np.max(v))
        ax.plot(BUDGETS, ys, marker=MARKERS[m], color=COLORS[m], label=LABELS[m])
        ax.fill_between(BUDGETS, lo, hi, color=COLORS[m], alpha=0.12, linewidth=0)
    ax.axhline(0.30, color="0.4", lw=0.6, ls=":")
    ax.axhline(0.50, color="0.3", lw=0.6, ls="--")
    ax.set_ylabel("Max target top-1 share")
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", frameon=False)

    ax = axes[1]
    for m in ["cdpa_mmd", "dann", "cdan_sca"]:
        ys = []
        for b in BUDGETS:
            rows = [d for d in diag if d["shift"] == shift and d["method"] == m
                    and d["budget"] == b and "first_collapse_epoch" in d]
            vals = [d["first_collapse_epoch"] for d in rows if d["first_collapse_epoch"] is not None]
            ys.append(np.mean(vals) if vals else np.nan)
        ax.plot(BUDGETS, ys, marker=MARKERS[m], color=COLORS[m], label=LABELS[m])
    ax.set_xscale("log")
    ax.set_xticks(BUDGETS)
    ax.set_xticklabels(["50", "500", "5000", "45k"])
    ax.set_xlabel("Target adaptation traces")
    ax.set_ylabel("First collapse epoch (mean)")
    ax.grid(alpha=0.3)
    fig.tight_layout(pad=0.3)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"c_e4_collapse_{shift}.{ext}", bbox_inches="tight")
    plt.close(fig)


def plot_calibration(ev, shift="desync50"):
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 4.0), sharex=True)
    for ax, key, lab in [(axes[0], "attack_ece_15", "Attack ECE (15 bins)"),
                         (axes[1], "attack_frac_confwrong_ge09", "Confident-wrong (p>=0.9)")]:
        for m in ["cdpa_mmd", "dann", "cdan_sca"]:
            ys = []
            for b in BUDGETS:
                rows = cell(ev, shift, m, b)
                ys.append(np.mean([r[key] for r in rows]) if rows else np.nan)
            ax.plot(BUDGETS, ys, marker=MARKERS[m], color=COLORS[m], label=LABELS[m])
        ax.set_ylabel(lab)
        ax.grid(alpha=0.3)
        ax.legend(loc="best", frameon=False)
    axes[1].set_xscale("log")
    axes[1].set_xticks(BUDGETS)
    axes[1].set_xticklabels(["50", "500", "5000", "45k"])
    axes[1].set_xlabel("Target adaptation traces")
    fig.tight_layout(pad=0.3)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"c_e4_calibration_{shift}.{ext}", bbox_inches="tight")
    plt.close(fig)


def main():
    ev, diag = load()
    shifts = sorted({r["shift"] for r in ev})
    for shift in shifts:
        s1 = budget_curve(ev, shift, "ntge_first")
        draw_budget(s1, "NTGE (traces)", f"c_e4_budget_ntge_{shift}",
                    censor_line=CENSOR, ylim=(1000, 12000),
                    yticks=[1000, 2000, 4000, 6000, 8000, 10000],
                    yticklabels=["1k", "2k", "4k", "6k", "8k", "10k"],
                    note="hollow marker: >=1 seed censored at 10k")
        s2 = budget_curve(ev, shift, "n_total_at_ge1")
        draw_budget(s2, "N_total @ GE=1 (traces)", f"c_e4_budget_cost_{shift}",
                    ylim=(9000, 70000),
                    yticks=[10000, 15000, 20000, 30000, 50000],
                    yticklabels=["10k", "15k", "20k", "30k", "50k"],
                    note="hollow marker: >=1 seed censored\n(cost is a lower bound)")
        plot_collapse(diag, shift)
        plot_calibration(ev, shift)
    plot_ge_grid(ev)
    plot_stability(ev)
    print("figures ->", FIG)


if __name__ == "__main__":
    main()
