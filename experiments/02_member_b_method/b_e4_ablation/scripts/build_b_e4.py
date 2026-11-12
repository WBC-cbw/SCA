#!/usr/bin/env python3
"""B-E4 三向消融：CDAN（无熵权重） vs 消融 1（marginal + 熵权重） vs full（条件 + 熵权重）。

数据来源：B-E3 的 results/eval 与 summary.json（不重训、不重算 GE，只做汇总与出图）。
输出：
  results/three_way.json / three_way.csv
  figures/b_e4_three_way_ge_<shift>{,_log}.{png,pdf}
  figures/b_e4_three_way_ntge.{png,pdf}
  figures/b_e4_monitor_<shift>.{png,pdf} 与 b_e4_calibration_{attack,adapt}（从 B-E3 复制）
"""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[2]
B3 = ROOT / "experiments/02_member_b_method/b_e3_cdan_sca"
EVAL = B3 / "results/eval"
FIG = HERE / "figures"
RES = HERE / "results"
FIG.mkdir(exist_ok=True)
RES.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "legend.fontsize": 7.0, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.linewidth": 0.8, "lines.linewidth": 1.3, "grid.linewidth": 0.4,
})

VARIANTS = ["cdan", "cdan_sca_marg", "cdan_sca_full"]
LABEL = {"cdan": "Abl.2: CDAN (no weight)", "cdan_sca_marg": "Abl.1: marginal + w",
         "cdan_sca_full": "Full: conditional + w"}
COLOR = {"cdan": "#2ca02c", "cdan_sca_marg": "#b15928", "cdan_sca_full": "#6a3d9a"}
STYLE = {"cdan": ":", "cdan_sca_marg": "--", "cdan_sca_full": "-"}
SHIFTS = ["desync50", "desync100"]


def load_eval():
    data = {}
    for shift in SHIFTS:
        for f in sorted((EVAL / shift).glob("*.json")):
            r = json.loads(f.read_text(encoding="utf-8"))
            data.setdefault((r["method"], shift), []).append(r)
    return data


def ge_fig(data):
    for shift in SHIFTS:
        for scale, suffix in [("linear", ""), ("log", "_log")]:
            fig, ax = plt.subplots(figsize=(3.5, 2.85))
            for m in VARIANTS:
                cs = sorted(data.get((m, shift), []), key=lambda r: r["seed"])
                if not cs:
                    continue
                xs = [0] + [p["traces"] for p in cs[0]["ge_curve"]]
                arr = np.array([[255] + [p["ge"] for p in c["ge_curve"]] for c in cs], dtype=float)
                f = (lambda v: np.where(v > 0, v, 0.5)) if scale == "log" else (lambda v: v)
                ax.fill_between(xs, f(arr.min(0)), f(arr.max(0)), color=COLOR[m], alpha=0.12, lw=0)
                ax.plot(xs, f(arr.mean(0)), color=COLOR[m], ls=STYLE[m], label=LABEL[m])
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
            fig.legend(h, l, frameon=False, ncol=1, fontsize=6.3,
                       loc="lower center", bbox_to_anchor=(0.5, -0.02))
            fig.tight_layout(rect=(0, 0.13, 1, 1))
            fig.savefig(FIG / f"b_e4_three_way_ge_{shift}{suffix}.png", dpi=300)
            fig.savefig(FIG / f"b_e4_three_way_ge_{shift}{suffix}.pdf")
            plt.close(fig)


def ntge_fig(summary):
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    width = 0.26
    x = np.arange(len(SHIFTS))
    for i, m in enumerate(VARIANTS):
        vals, errs, stable = [], [], []
        for shift in SHIFTS:
            d = summary["three_way"].get(shift, {}).get(m, {})
            vals.append(d.get("ntge_first_mean", np.nan))
            errs.append(d.get("ntge_first_std", 0))
            stable.append("%d/%d" % (d.get("ntge_stable_reached", 0), 3))
        xs = x + (i - 1) * width
        ax.bar(xs, vals, width, yerr=errs, capsize=1.8, color=COLOR[m], alpha=0.85,
               edgecolor=COLOR[m], linewidth=0.6,
               error_kw={"elinewidth": 0.6, "ecolor": "0.25"}, label=LABEL[m])
        for xi, v, lab in zip(xs, vals, stable):
            if np.isfinite(v):
                ax.text(xi, v * 0.5, lab, rotation=90, ha="center", va="center",
                        fontsize=5.2, color="0.1", fontweight="bold")
    ax.set_yscale("log"); ax.set_ylim(100, 40000)
    ax.axhline(10000, color="0.5", ls=":", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(SHIFTS)
    ax.set_ylabel("NTGE (traces)")
    ax.set_xlim(-0.55, 1.55)
    ax.grid(alpha=0.3, axis="y")
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, frameon=False, ncol=1, fontsize=6.2,
               loc="lower center", bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    fig.savefig(FIG / "b_e4_three_way_ntge.png", dpi=300)
    fig.savefig(FIG / "b_e4_three_way_ntge.pdf")
    plt.close(fig)


def main():
    data = load_eval()
    summary = json.loads((B3 / "results/summary.json").read_text(encoding="utf-8"))
    ge_fig(data)
    ntge_fig(summary)

    three = {s: {m: summary["metrics"].get(m, {}).get(s) for m in VARIANTS} for s in SHIFTS}
    out = {"three_way": three,
           "paired_full_vs_cdan": {s: summary["paired"].get(f"full_vs_cdan@{s}") for s in SHIFTS},
           "paired_full_vs_marg": {s: summary["paired"].get(f"full_vs_cdan_sca_marg@{s}") for s in SHIFTS},
           "source": "B-E3 results/eval + summary.json"}
    (RES / "three_way.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    with (RES / "three_way.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["shift", "variant", "ntge_mean", "ntge_std", "reached", "stable_reached",
                    "ge10k_mean", "ge10k_std", "sr10k", "n_total", "acc", "entropy",
                    "max_top1_share", "collapse_epochs"])
        for shift, d in three.items():
            for m, v in d.items():
                if not v:
                    continue
                w.writerow([shift, LABEL[m], v["ntge_first_mean"], v["ntge_first_std"],
                            v["ntge_first_reached"], v["ntge_stable_reached"],
                            v["ge_at_10000_mean"], v["ge_at_10000_std"], v["sr_at_10000_mean"],
                            v["n_total_at_ge1_mean"], v["attack_accuracy_mean"],
                            v["mean_entropy_mean"], v["max_top1_share_mean"],
                            v["collapse_epochs_mean"]])

    for name in ["b_e3_monitor_desync50", "b_e3_monitor_desync100",
                 "b_e3_calibration_attack", "b_e3_calibration_adapt"]:
        for ext in (".png", ".pdf"):
            src = B3 / "figures" / (name + ext)
            if src.exists():
                shutil.copyfile(src, FIG / ("b_e4_" + name.replace("b_e3_", "") + ext))
    print("B-E4 built ->", FIG, RES)


if __name__ == "__main__":
    main()
