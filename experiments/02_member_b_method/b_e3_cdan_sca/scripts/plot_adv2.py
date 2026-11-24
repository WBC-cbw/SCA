#!/usr/bin/env python3
"""adv2 补充实验图：GE 曲线（含 CDPA/ full / SO）、训练期坍塌监控、NTGE 柱。"""
from __future__ import annotations
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
    "font.family": "serif", "font.size": 9, "axes.titlesize": 9,
    "axes.labelsize": 9, "legend.fontsize": 7.0, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "axes.linewidth": 0.8, "lines.linewidth": 1.2,
    "grid.linewidth": 0.4,
})
SHIFTS = ["desync50", "desync100"]
SEEDS = [42, 43, 44]
COLOR = {"source_only": "#1f77b4", "cdpa_mmd": "#d62728",
         "cdan_sca_full": "#6a3d9a", "cdan_sca_adv2": "#17becf"}
LABEL = {"source_only": "Source Only", "cdpa_mmd": "CDPA-MMD",
         "cdan_sca_full": "CDAN-SCA full", "cdan_sca_adv2": "CDAN-SCA adv2"}
STYLE = {"source_only": "--", "cdpa_mmd": "-", "cdan_sca_full": "-", "cdan_sca_adv2": "-"}


def load_runs():
    data = {}
    for shift in SHIFTS:
        for f in sorted((RES / "eval" / shift).glob("*.json")):
            r = json.loads(f.read_text(encoding="utf-8"))
            if r["method"] in COLOR:
                data.setdefault((r["method"], shift), []).append(r)
    adv2_all = RES / "eval_adv2" / "eval_all_adv2.json"
    if adv2_all.exists():
        for r in json.loads(adv2_all.read_text(encoding="utf-8")):
            data.setdefault((r["method"], r["shift"]), []).append(r)
    for k in data:
        data[k].sort(key=lambda r: r["seed"])
    return data


def grids(curves):
    xs = [0] + [p["traces"] for p in curves[0]["ge_curve"]]
    ys = [[255] + [p["ge"] for p in c["ge_curve"]] for c in curves]
    a = np.array(ys, dtype=float)
    return np.array(xs), a.mean(0), a.min(0), a.max(0)


def ge_figs(data):
    for shift in SHIFTS:
        for scale, suffix in [("linear", ""), ("log", "_log")]:
            fig, ax = plt.subplots(figsize=(3.5, 3.05))
            for method in ["source_only", "cdpa_mmd", "cdan_sca_full", "cdan_sca_adv2"]:
                curves = data.get((method, shift), [])
                if not curves:
                    continue
                xs, mean, lo, hi = grids(curves)
                f = (lambda v: np.where(v > 0, v, 0.5)) if scale == "log" else (lambda v: v)
                if method != "source_only":
                    ax.fill_between(xs, f(lo), f(hi), color=COLOR[method], alpha=0.10, lw=0)
                ax.plot(xs, f(mean), color=COLOR[method], ls=STYLE[method], lw=1.4,
                        label=f"{LABEL[method]} (n={len(curves)})")
            if scale == "log":
                ax.set_yscale("log"); ax.set_ylim(0.5, 256)
                ax.set_yticks([0.5, 1, 4, 16, 64, 256])
                ax.set_yticklabels(["0", "1", "4", "16", "64", "256"])
            else:
                ax.set_ylim(0, 256); ax.set_yticks([0, 64, 128, 192, 256])
            ax.set_xlim(0, 10000)
            ax.set_xlabel("Number of attack traces")
            ax.set_ylabel("Key rank")
            ax.grid(alpha=0.3)
            ax.set_title(shift, pad=3)
            h, l = ax.get_legend_handles_labels()
            fig.legend(h, l, frameon=False, ncol=2, fontsize=6.4,
                       loc="lower center", bbox_to_anchor=(0.5, -0.01))
            fig.tight_layout(rect=(0, 0.17, 1, 1))
            fig.savefig(FIG / f"b_e3_adv2_ge_{shift}{suffix}.png", dpi=300)
            fig.savefig(FIG / f"b_e3_adv2_ge_{shift}{suffix}.pdf")
            plt.close(fig)


def monitor_figs():
    for shift in SHIFTS:
        series = {}
        for method, tag in [("cdan_sca_full", "full"), ("cdan_sca_adv2", "adv2")]:
            curves = []
            for seed in SEEDS:
                p = RES / f"cdan_sca_{tag}_{shift}_s{seed}" / "history.json"
                if p.exists():
                    h = json.loads(p.read_text(encoding="utf-8"))
                    if len(h.get("epochs", [])) < 70:   # 排除中断/重启的残段
                        continue
                    curves.append(([m["top1_share"] for m in h["monitor"]],
                                   [m["pred_entropy_mean"] for m in h["monitor"]]))
            if curves:
                series[method] = curves
        if "cdan_sca_adv2" not in series:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
        for method, curves in series.items():
            n = min(len(c[0]) for c in curves)
            top1 = np.array([c[0][:n] for c in curves])
            ent = np.array([c[1][:n] for c in curves])
            x = np.arange(1, n + 1)
            axes[0].plot(x, top1.mean(0), color=COLOR[method], lw=1.3,
                         label=f"{LABEL[method]} (n={len(curves)})")
            axes[0].fill_between(x, top1.min(0), top1.max(0), color=COLOR[method], alpha=0.12, lw=0)
            axes[1].plot(x, ent.mean(0), color=COLOR[method], lw=1.3,
                         label=f"{LABEL[method]} (n={len(curves)})")
            axes[1].fill_between(x, ent.min(0), ent.max(0), color=COLOR[method], alpha=0.12, lw=0)
        axes[0].axhline(0.30, color="#7f7f7f", ls=":", lw=0.9)
        axes[0].axhline(0.50, color="#d62728", ls=":", lw=0.9)
        axes[0].text(1, 0.31, "WARN 30%", fontsize=6.6, color="#555555", va="bottom")
        axes[0].text(1, 0.51, "COLLAPSE 50%", fontsize=6.6, color="#d62728", va="bottom")
        axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Target top-1 class share")
        axes[0].set_ylim(0, 0.95); axes[0].grid(alpha=0.3)
        axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Target prediction entropy (nats)")
        axes[1].grid(alpha=0.3)
        axes[0].set_title("(a) Collapse monitor", pad=3)
        axes[1].set_title("(b) Prediction entropy", pad=3)
        h, l = axes[0].get_legend_handles_labels()
        h2, l2 = axes[1].get_legend_handles_labels()
        fig.legend(h + h2, l + l2, frameon=False, ncol=2, fontsize=7.0,
                   loc="lower center", bbox_to_anchor=(0.5, -0.02))
        fig.tight_layout(rect=(0, 0.13, 1, 1))
        fig.savefig(FIG / f"b_e3_adv2_monitor_{shift}.png", dpi=300)
        fig.savefig(FIG / f"b_e3_adv2_monitor_{shift}.pdf")
        plt.close(fig)


def ntge_fig():
    s = json.loads((RES / "summary_adv2.json").read_text(encoding="utf-8"))["metrics"]
    methods = ["cdpa_mmd", "cdan_sca_full", "cdan_sca_adv2"]
    fig, ax = plt.subplots(figsize=(3.5, 3.05))
    x = np.arange(len(SHIFTS)); w = 0.26
    for i, m in enumerate(methods):
        vals, errs, lab = [], [], []
        for shift in SHIFTS:
            d = s.get(m, {}).get(shift)
            if not d:
                vals.append(np.nan); errs.append(0); lab.append(""); continue
            vals.append(d["ntge_first_mean"]); errs.append(d["ntge_first_std"])
            lab.append(f"{d['ntge_stable_reached']}/{d['n_runs']}")
        xs = x + (i - 1) * w
        bars = ax.bar(xs, vals, w, color=COLOR[m], edgecolor="black", linewidth=0.4,
                      yerr=errs, capsize=2.0, error_kw={"lw": 0.7}, label=LABEL[m])
        for b, v, l in zip(bars, vals, lab):
            if np.isnan(v):
                continue
            ax.annotate(l, (b.get_x() + b.get_width() / 2, 0.965),
                        xycoords=("data", "axes fraction"), ha="center", va="top",
                        fontsize=6.3)
    ax.axhline(10000, color="#7f7f7f", ls=":", lw=0.9)
    ax.annotate("censor 10k", (0.015, 0.815), xycoords="axes fraction",
                fontsize=6.2, color="#555555", va="bottom")
    ax.set_yscale("log"); ax.set_ylim(100, 30000)
    ax.set_xticks(x); ax.set_xticklabels(SHIFTS)
    ax.set_ylabel("NTGE (traces)")
    ax.grid(alpha=0.3, axis="y")
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, frameon=False, ncol=3, fontsize=6.4,
               loc="lower center", bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    fig.savefig(FIG / "b_e3_adv2_ntge.png", dpi=300)
    fig.savefig(FIG / "b_e3_adv2_ntge.pdf")
    plt.close(fig)


def main():
    data = load_runs()
    ge_figs(data)
    ntge_fig()
    monitor_figs()
    print("figures done")


if __name__ == "__main__":
    main()
