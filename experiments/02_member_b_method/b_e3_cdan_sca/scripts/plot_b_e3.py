#!/usr/bin/env python3
"""B-E3 论文标准图：六方法 GE（线性+对数）、NTGE、训练期 top1 占比/熵、
ECE 与 confident-wrong 对比（attack / adapt 两侧）。

样式沿用 E0/A-E2/A-E3：serif、9pt、3.5in 单栏（GE/NTGE）、300dpi PNG+PDF、
Y 轴 0-256 全量程、对数版 rank=0 映射 0.5、seed 带 min-max。
监控与校准图为双栏宽（7.0in），因为要并排放两组面板。
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parents[1]
EVAL = HERE / "results/eval"
RES = HERE / "results"
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.titlesize": 9,
    "axes.labelsize": 9, "legend.fontsize": 7.0, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "axes.linewidth": 0.8, "lines.linewidth": 1.2,
    "grid.linewidth": 0.4,
})

METHODS = ["source_only", "dann", "cdan", "cdpa_mmd", "cdan_sca_full", "cdan_sca_marg"]
LABEL = {"source_only": "Source Only", "dann": "DANN", "cdan": "CDAN",
         "cdpa_mmd": "CDPA-MMD", "cdan_sca_full": "CDAN-SCA full",
         "cdan_sca_marg": "Abl.1 (marginal+w)"}
COLOR = {"source_only": "#1f77b4", "dann": "#ff7f0e", "cdan": "#2ca02c",
         "cdpa_mmd": "#d62728", "cdan_sca_full": "#6a3d9a", "cdan_sca_marg": "#b15928"}
STYLE = {"cdan_sca_full": "-", "cdan_sca_marg": "--"}
SHIFTS = ["desync50", "desync100"]
N_EPOCHS = 75


def load_eval():
    data = {}
    for shift in SHIFTS:
        for f in sorted((EVAL / shift).glob("*.json")):
            r = json.loads(f.read_text(encoding="utf-8"))
            data.setdefault((r["method"], shift), []).append(r)
    return data


def grids(curves):
    xs = [0] + [p["traces"] for p in curves[0]["ge_curve"]]
    ys = [[255] + [p["ge"] for p in c["ge_curve"]] for c in curves]
    arr = np.array(ys, dtype=float)
    return np.array(xs), arr.mean(0), arr.min(0), arr.max(0)


def ge_curves(data):
    for shift in SHIFTS:
        for scale, suffix in [("linear", ""), ("log", "_log")]:
            fig, ax = plt.subplots(figsize=(3.5, 3.05))
            for method in METHODS:
                curves = sorted(data.get((method, shift), []), key=lambda r: r["seed"])
                if not curves:
                    continue
                xs, mean, lo, hi = grids(curves)
                f = (lambda v: np.where(v > 0, v, 0.5)) if scale == "log" else (lambda v: v)
                ax.fill_between(xs, f(lo), f(hi), color=COLOR[method], alpha=0.10, lw=0)
                ax.plot(xs, f(mean), color=COLOR[method], ls=STYLE.get(method, "-"),
                        lw=1.5 if method in ("cdan_sca_full", "cdan_sca_marg") else 1.2,
                        label=LABEL[method])
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
            fig.tight_layout(rect=(0, 0.155, 1, 1))
            fig.savefig(FIG / f"b_e3_ge_{shift}{suffix}.png", dpi=300)
            fig.savefig(FIG / f"b_e3_ge_{shift}{suffix}.pdf")
            plt.close(fig)


def ntge_bars():
    summary = json.loads((RES / "summary.json").read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(3.5, 3.2))
    width = 0.13
    x = np.arange(len(SHIFTS))
    for i, method in enumerate(METHODS):
        vals, errs, stable = [], [], []
        for shift in SHIFTS:
            m = summary["metrics"].get(method, {}).get(shift)
            if not m:
                vals.append(np.nan); errs.append(0); stable.append(""); continue
            vals.append(m["ntge_first_mean"]); errs.append(m["ntge_first_std"])
            stable.append(f"{m['ntge_stable_reached']}/{m['n_runs']}")
        xs = x + (i - 2.5) * width
        ax.bar(xs, vals, width, yerr=errs, capsize=1.5, color=COLOR[method], alpha=0.85,
               edgecolor=COLOR[method], linewidth=0.6,
               error_kw={"elinewidth": 0.55, "ecolor": "0.25"}, label=LABEL[method])
        for xi, v, lab in zip(xs, vals, stable):
            if lab and np.isfinite(v):
                ax.text(xi, v * 0.52, lab, rotation=90, ha="center", va="center",
                        fontsize=4.6, color="0.10", fontweight="bold")
    ax.set_yscale("log"); ax.set_ylim(100, 45000)
    ax.axhline(10000, color="0.5", ls=":", lw=0.8, zorder=1)
    ax.set_xticks(x); ax.set_xticklabels(SHIFTS)
    ax.set_ylabel("NTGE (traces)")
    ax.set_xlim(-0.55, 1.55)
    ax.grid(alpha=0.3, axis="y")
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, frameon=False, ncol=2, fontsize=6.2,
               loc="lower center", bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.18, 1, 1))
    fig.savefig(FIG / "b_e3_ntge.png", dpi=300)
    fig.savefig(FIG / "b_e3_ntge.pdf")
    plt.close(fig)


def monitor_curves():
    for shift in SHIFTS:
        curves = {}
        for tag, method in [("full", "cdan_sca_full"), ("marg", "cdan_sca_marg")]:
            runs = []
            for seed in (42, 43, 44):
                h = HERE / "results" / f"cdan_sca_{tag}_{shift}_s{seed}" / "history.json"
                if h.exists():
                    runs.append(json.loads(h.read_text(encoding="utf-8")))
            if runs:
                curves[method] = runs
        if not curves:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.55))
        for ax, key, title in [(axes[0], "top1_share", "Target top-1 class share"),
                               (axes[1], "pred_entropy_mean", "Target prediction entropy (nats)")]:
            for method, runs in curves.items():
                ep = np.array([e for e in runs[0]["epochs"]])
                vals = np.array([[m[key] for m in r["monitor"]] for r in runs])
                lo, hi, mean = vals.min(0), vals.max(0), vals.mean(0)
                ax.fill_between(ep, lo, hi, color=COLOR[method], alpha=0.12, lw=0)
                ax.plot(ep, mean, color=COLOR[method], ls=STYLE.get(method, "-"),
                        label=LABEL[method])
            if key == "top1_share":
                ax.axhline(0.30, color="0.45", ls="--", lw=0.7)
                ax.axhline(0.50, color="0.30", ls=":", lw=0.8)
                ax.text(2, 0.305, "WARN 30%", fontsize=5.5, color="0.35", va="bottom")
                ax.text(2, 0.505, "COLLAPSE 50%", fontsize=5.5, color="0.25", va="bottom")
                ax.set_ylim(0, max(0.6, ax.get_ylim()[1]))
            ax.set_xlim(0, N_EPOCHS)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(title)
            ax.grid(alpha=0.3)
        # CDAN / CDPA 末态参考点（A-E2 未记录逐 epoch 监控，只给 epoch 75 终点）
        for method in ("cdan",):
            vals_top1, vals_h = [], []
            for seed in (42, 43, 44):
                p = RES / "adapt_diag" / f"{method}_{shift}_s{seed}.npz"
                if not p.exists():
                    continue
                z = np.load(p)
                pr = z["probs"][:2000].astype(np.float64)
                cnt = np.bincount(pr.argmax(1), minlength=256)
                vals_top1.append(cnt.max() / len(pr))
                q = np.clip(pr, 1e-12, 1)
                vals_h.append(float(-(q * np.log(q)).sum(1).mean()))
            if vals_top1:
                axes[0].plot([N_EPOCHS], [np.mean(vals_top1)], marker="s", ms=4,
                             color=COLOR[method], ls="none",
                             label="CDAN final (A-E2, epoch 75 only)")
                axes[1].plot([N_EPOCHS], [np.mean(vals_h)], marker="s", ms=4,
                             color=COLOR[method], ls="none")
        axes[0].legend(frameon=False, fontsize=5.8, loc="upper left")
        fig.suptitle(f"{shift}: entropy weighting during training (full vs ablation 1; "
                     f"A-E2 CDAN has no per-epoch monitoring)", fontsize=7.5, y=1.02)
        fig.tight_layout()
        fig.savefig(FIG / f"b_e3_monitor_{shift}.png", dpi=300, bbox_inches="tight")
        fig.savefig(FIG / f"b_e3_monitor_{shift}.pdf", bbox_inches="tight")
        plt.close(fig)


def calib_figs():
    import csv
    rows = list(csv.DictReader((RES / "entropy_weight_analysis.csv").open(encoding="utf-8")))
    for side in ("attack", "adapt"):
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharex=True)
        width = 0.13
        x = np.arange(len(SHIFTS))
        for ax, key, title in [(axes[0], "ece_15", "ECE (15 bins)"),
                               (axes[1], "frac_confwrong_ge_0.9", "Confident-wrong\n(p\u22650.9)")]:
            for i, method in enumerate(METHODS):
                vals = []
                for shift in SHIFTS:
                    rs = [r for r in rows if r["side"] == side and r["shift"] == shift
                          and r["method"] == method]
                    vals.append(np.mean([float(r[key]) for r in rs]) if rs else np.nan)
                ax.bar(x + (i - 2.5) * width, vals, width, color=COLOR[method], alpha=0.85,
                       edgecolor=COLOR[method], linewidth=0.6, label=LABEL[method])
            ax.set_xticks(x); ax.set_xticklabels(SHIFTS)
            ax.set_ylabel(title, fontsize=8.5, linespacing=1.0); ax.grid(alpha=0.3, axis="y")
        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, frameon=False, ncol=3, fontsize=6.4,
                   loc="lower center", bbox_to_anchor=(0.5, -0.02))
        fig.tight_layout(rect=(0, 0.16, 1, 1), w_pad=1.6)
        fig.savefig(FIG / f"b_e3_calibration_{side}.png", dpi=300)
        fig.savefig(FIG / f"b_e3_calibration_{side}.pdf")
        plt.close(fig)


if __name__ == "__main__":
    d = load_eval()
    ge_curves(d)
    ntge_bars()
    monitor_curves()
    calib_figs()
    print("figures ->", FIG)
