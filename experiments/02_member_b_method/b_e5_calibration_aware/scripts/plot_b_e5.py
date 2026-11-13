#!/usr/bin/env python3
"""B-E5 论文标准图：GE（线性+对数）、NTGE、权重分布、训练监控、校准对比。

样式沿用 E0/A-E2/A-E3/B-E3：serif、9pt、3.5in 单栏（GE/NTGE）、300dpi PNG+PDF、
Y 轴 0-256 全量程、对数版 rank=0 映射 0.5、seed 带 min-max。
权重分布/监控/校准图用双栏宽（7.0in）。

主图只画 5 条线（Source Only / CDAN / CDPA-MMD / CDAN-SCA full / CDAN-SCA-calib），
避免 7 条线不可读；DANN 与消融 1 的数字在表与附录图中。
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[2]
B_E3 = ROOT / "experiments/02_member_b_method/b_e3_cdan_sca"
EVAL = HERE / "results/eval"
RES = HERE / "results"
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.titlesize": 9,
    "axes.labelsize": 9, "legend.fontsize": 6.6, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "axes.linewidth": 0.8, "lines.linewidth": 1.2,
    "grid.linewidth": 0.4,
})

SHIFTS = ["desync50", "desync100"]
MAIN = ["source_only", "cdan", "cdpa_mmd", "cdan_sca_full", "cdan_sca_calib"]
ALL = ["source_only", "dann", "cdan", "cdpa_mmd", "cdan_sca_full", "cdan_sca_marg",
       "cdan_sca_calib"]
LABEL = {"source_only": "Source Only", "dann": "DANN", "cdan": "CDAN",
         "cdpa_mmd": "CDPA-MMD", "cdan_sca_full": "CDAN-SCA full",
         "cdan_sca_marg": "Abl.1 (marginal+w)", "cdan_sca_calib": "CDAN-SCA-calib"}
COLOR = {"source_only": "#1f77b4", "dann": "#ff7f0e", "cdan": "#2ca02c",
         "cdpa_mmd": "#d62728", "cdan_sca_full": "#6a3d9a",
         "cdan_sca_marg": "#b15928", "cdan_sca_calib": "#17becf"}
STYLE = {"cdan_sca_full": "--", "cdan_sca_calib": "-"}
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
    for methods, suffix_name in [(MAIN, ""), (ALL, "_all")]:
        for shift in SHIFTS:
            ok = [m for m in methods if data.get((m, shift))]
            if not ok:
                continue
            for scale, suffix in [("linear", ""), ("log", "_log")]:
                fig, ax = plt.subplots(figsize=(3.5, 3.05))
                for method in ok:
                    curves = sorted(data[(method, shift)], key=lambda r: r["seed"])
                    xs, mean, lo, hi = grids(curves)
                    f = (lambda v: np.where(v > 0, v, 0.5)) if scale == "log" else (lambda v: v)
                    ax.fill_between(xs, f(lo), f(hi), color=COLOR[method], alpha=0.10, lw=0)
                    ax.plot(xs, f(mean), color=COLOR[method], ls=STYLE.get(method, "-"),
                            lw=1.6 if method in ("cdan_sca_full", "cdan_sca_calib") else 1.1,
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
                fig.legend(h, l, frameon=False, ncol=2 if len(ok) > 3 else 1,
                           fontsize=6.0 if len(ok) > 5 else 6.6,
                           loc="lower center", bbox_to_anchor=(0.5, -0.01))
                bottom = 0.20 if len(ok) > 3 else 0.13
                fig.tight_layout(rect=(0, bottom, 1, 1))
                fig.savefig(FIG / f"b_e5_ge_{shift}{suffix_name}{suffix}.png", dpi=300)
                fig.savefig(FIG / f"b_e5_ge_{shift}{suffix_name}{suffix}.pdf")
                plt.close(fig)


def ntge_bars():
    summary = json.loads((RES / "summary.json").read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(3.5, 3.35))
    width = 0.115
    x = np.arange(len(SHIFTS))
    for i, method in enumerate(ALL):
        vals, errs, stable = [], [], []
        for shift in SHIFTS:
            m = summary["metrics"].get(method, {}).get(shift)
            if not m:
                vals.append(np.nan); errs.append(0); stable.append(""); continue
            vals.append(m["ntge_first_mean"]); errs.append(m["ntge_first_std"])
            stable.append(f"{m['ntge_stable_reached']}/{m['n_runs']}")
        xs = x + (i - 3) * width
        ax.bar(xs, vals, width, yerr=errs, capsize=1.2, color=COLOR[method], alpha=0.88,
               edgecolor=COLOR[method], linewidth=0.5,
               error_kw={"elinewidth": 0.5, "ecolor": "0.25"}, label=LABEL[method])
        for xi, v, lab in zip(xs, vals, stable):
            if lab and np.isfinite(v):
                ax.text(xi, v * 0.55, lab, rotation=90, ha="center", va="center",
                        fontsize=4.2, color="0.10", fontweight="bold")
    ax.set_yscale("log"); ax.set_ylim(100, 60000)
    ax.axhline(10000, color="0.5", ls=":", lw=0.8, zorder=1)
    ax.set_xticks(x); ax.set_xticklabels(SHIFTS)
    ax.set_ylabel("NTGE (traces)")
    ax.set_xlim(-0.55, 1.55)
    ax.grid(alpha=0.3, axis="y")
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, frameon=False, ncol=3, fontsize=5.9,
               loc="lower center", bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.20, 1, 1))
    fig.savefig(FIG / "b_e5_ntge.png", dpi=300)
    fig.savefig(FIG / "b_e5_ntge.pdf")
    plt.close(fig)


def _weights(method, shift, side):
    """返回 (w_bare, w_cal)：calib 的校准权重由温度重算；full 只有裸熵权重。"""
    if method == "cdan_sca_full":
        base = B_E3 / "results"
        tag = "full"
    elif method == "cdan_sca_calib":
        base = RES
        tag = "calib"
    else:
        return None
    bare, cal = [], []
    for seed in (42, 43, 44):
        run = f"cdan_sca_{tag}_{shift}_s{seed}"
        rdir = base / run
        if side == "adapt":
            p = rdir / "diag_target_adapt.npz"
            if not p.exists():
                continue
            d = np.load(p)
            bare.append(np.asarray(d["w_bare"] if "w_bare" in d.files else d["w"], dtype=float))
            cal.append(np.asarray(d["w"] if "w_bare" in d.files else d["w"], dtype=float))
        else:
            p = rdir / "predictions_attack.npy"
            if not p.exists():
                continue
            probs = np.clip(np.load(p).astype(np.float64), 1e-12, 1)
            ent = -(probs * np.log(probs)).sum(1)
            bare.append(1.0 - ent / np.log(256))
            tp = HERE / "results" / run / "temperature.json"
            if tp.exists():
                T = float(json.loads(tp.read_text(encoding="utf-8"))["temperature"])
                q = np.power(probs, 1.0 / T)
                q = q / q.sum(1, keepdims=True)
                entc = -(q * np.log(np.clip(q, 1e-12, 1))).sum(1)
                cal.append(1.0 - entc / np.log(256))
    if not bare:
        return None
    return np.concatenate(bare), (np.concatenate(cal) if cal and method == "cdan_sca_calib" else None)


def weight_figs():
    for side in ("adapt", "attack"):
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
        bins = np.linspace(0, 1, 51)
        for ax, shift in zip(axes, SHIFTS):
            for method, ls in [("cdan_sca_full", "--"), ("cdan_sca_calib", "-")]:
                got = _weights(method, shift, side)
                if got is None:
                    continue
                wb, wc = got
                if method == "cdan_sca_full":
                    ax.hist(wb, bins=bins, density=True, histtype="step", lw=1.3,
                            ls=ls, color=COLOR[method],
                            label=f"{LABEL[method]} bare (mean {wb.mean():.2f})")
                else:
                    ax.hist(wb, bins=bins, density=True, histtype="step", lw=1.1,
                            ls=":", color=COLOR[method],
                            label=f"{LABEL[method]} bare (mean {wb.mean():.2f})")
                    if wc is not None:
                        ax.hist(wc, bins=bins, density=True, histtype="step", lw=1.4,
                                ls="-", color=COLOR[method],
                                label=f"{LABEL[method]} calibrated (mean {wc.mean():.2f})")
            ax.set_xlim(0, 1)
            ax.set_xlabel("Entropy weight w")
            ax.grid(alpha=0.3)
            ax.set_title(f"{shift} ({side})", pad=3)
            if shift == SHIFTS[0]:
                ax.set_ylabel("Probability density")
        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, frameon=False, ncol=3, fontsize=5.9,
                   loc="lower center", bbox_to_anchor=(0.5, -0.02))
        fig.tight_layout(rect=(0, 0.16, 1, 1))
        fig.savefig(FIG / f"b_e5_weights_{side}.png", dpi=300)
        fig.savefig(FIG / f"b_e5_weights_{side}.pdf")
        plt.close(fig)


def monitor_curves():
    for shift in SHIFTS:
        series = {}
        for tag, method in [("full", "cdan_sca_full"), ("calib", "cdan_sca_calib")]:
            base = B_E3 if tag == "full" else HERE
            runs = []
            for seed in (42, 43, 44):
                h = base / "results" / f"cdan_sca_{tag}_{shift}_s{seed}" / "history.json"
                if h.exists():
                    runs.append(json.loads(h.read_text(encoding="utf-8")))
            if runs:
                series[method] = runs
        if not series:
            continue
        keys = [("top1_share", "Target top-1 class share"),
                ("pred_entropy_mean", "Target entropy (nats)"),
                ("w_mean", "Entropy weight w")]
        fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.35))
        for ax, (key, title) in zip(axes, keys):
            for method, runs in series.items():
                ep = np.array(runs[0]["epochs"])
                minlen = min(len(r["monitor"]) for r in runs)
                vals = np.array([[m[key] for m in r["monitor"][:minlen]] for r in runs])
                lo, hi, mean = vals.min(0), vals.max(0), vals.mean(0)
                ax.fill_between(ep[:minlen], lo, hi, color=COLOR[method], alpha=0.12, lw=0)
                ax.plot(ep[:minlen], mean, color=COLOR[method], ls=STYLE.get(method, "-"),
                        label=LABEL[method])
            if key == "top1_share":
                ax.axhline(0.30, color="0.45", ls="--", lw=0.7)
                ax.axhline(0.50, color="0.30", ls=":", lw=0.8)
                ax.text(2, 0.305, "WARN 30%", fontsize=5.2, color="0.35", va="bottom")
                ax.text(2, 0.505, "COLLAPSE 50%", fontsize=5.2, color="0.25", va="bottom")
                ax.set_ylim(0, max(0.6, ax.get_ylim()[1]))
            if key == "w_mean":
                vals_bare = array_to_mean(series.get("cdan_sca_calib"), "w_bare_mean")
                if vals_bare is not None:
                    ax.plot(vals_bare[0], vals_bare[1], color=COLOR["cdan_sca_calib"],
                            ls=":", lw=1.0, label="calib w_bare (uncalibrated)")
                ax.set_ylim(0, 1.02)
            ax.set_xlim(0, N_EPOCHS)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(title)
            ax.grid(alpha=0.3)
        axes[0].legend(frameon=False, fontsize=5.6, loc="upper left")
        axes[2].legend(frameon=False, fontsize=5.2, loc="lower left")
        fig.suptitle(f"{shift}: full (bare entropy weight) vs calib (temperature-calibrated weight)",
                     fontsize=7.4, y=1.02)
        fig.tight_layout()
        fig.savefig(FIG / f"b_e5_monitor_{shift}.png", dpi=300, bbox_inches="tight")
        fig.savefig(FIG / f"b_e5_monitor_{shift}.pdf", bbox_inches="tight")
        plt.close(fig)


def array_to_mean(runs, key, alt_key=None):
    if not runs:
        return None
    ep = np.array(runs[0]["epochs"])
    minlen = min(len(r["monitor"]) for r in runs)
    vals = np.array([[m.get(key, m.get(alt_key, np.nan)) for m in r["monitor"][:minlen]]
                     for r in runs], dtype=float)
    return ep[:minlen], vals.mean(0)


def calib_figs():
    rows = list(csv.DictReader((RES / "calibration_summary.csv").open(encoding="utf-8")))
    methods = ["cdan", "cdpa_mmd", "cdan_sca_full", "cdan_sca_calib"]
    for side in ("attack", "adapt"):
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.55))
        width = 0.19
        x = np.arange(len(SHIFTS))
        for ax, key, title in [(axes[0], "ece_15_mean", "ECE (15 bins)"),
                               (axes[1], "frac_confwrong_ge_0.9_mean",
                                "Confident-wrong\n(p\u22650.9)")]:
            for i, method in enumerate(methods):
                vals, errs = [], []
                for shift in SHIFTS:
                    cal = "calibrated" if method == "cdan_sca_calib" else "bare"
                    rs = [r for r in rows if r["side"] == side and r["shift"] == shift
                          and r["method"] == method and r["calibration"] == cal]
                    vals.append(float(rs[0][key]) if rs else np.nan)
                    sd_key = key.replace("_mean", "_std")
                    errs.append(float(rs[0][sd_key]) if rs else 0.0)
                ax.bar(x + (i - 1.5) * width, vals, width, yerr=errs, capsize=1.2,
                       color=COLOR[method], alpha=0.88, edgecolor=COLOR[method], linewidth=0.5,
                       error_kw={"elinewidth": 0.5, "ecolor": "0.25"}, label=LABEL[method])
            ax.set_xticks(x); ax.set_xticklabels(SHIFTS)
            ax.set_ylabel(title, fontsize=8.5, linespacing=1.0)
            ax.grid(alpha=0.3, axis="y")
            ax.set_ylim(0, 1.0)
        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, frameon=False, ncol=4, fontsize=6.2,
                   loc="lower center", bbox_to_anchor=(0.5, -0.02))
        fig.tight_layout(rect=(0, 0.13, 1, 1), w_pad=1.4)
        fig.savefig(FIG / f"b_e5_calibration_{side}.png", dpi=300)
        fig.savefig(FIG / f"b_e5_calibration_{side}.pdf")
        plt.close(fig)


if __name__ == "__main__":
    d = load_eval()
    ge_curves(d)
    ntge_bars()
    weight_figs()
    monitor_curves()
    calib_figs()
    print("figures ->", FIG)
