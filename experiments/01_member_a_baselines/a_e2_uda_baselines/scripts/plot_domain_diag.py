#!/usr/bin/env python3
"""域判别诊断图：训练前（随机判别器）与训练后的 balanced accuracy。

x 轴 = 方法 x 偏移；每组两根柱（before/after），误差棒为 3 seed 的 min-max；
0.5 虚线为随机判别。输出 figures/a_e2_domain_diag.{png,pdf}（300 dpi）。
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parents[1]
EVAL = HERE / "results/eval/eval_all.json"
FIG = HERE / "figures"
GROUPS = [("dann", "desync50", "DANN\ndesync50"), ("dann", "desync100", "DANN\ndesync100"),
          ("cdan", "desync50", "CDAN\ndesync50"), ("cdan", "desync100", "CDAN\ndesync100")]
COLOR = {"before": "#bbbbbb", "after": "#1f77b4"}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 8,
    "axes.linewidth": 0.8, "xtick.direction": "in", "ytick.direction": "in",
    "legend.frameon": False, "savefig.dpi": 300,
})


def main():
    rows = json.loads(EVAL.read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(3.9, 2.9))
    x = np.arange(len(GROUPS))
    w = 0.34
    for j, (key, color) in enumerate((("before", COLOR["before"]), ("after", COLOR["after"]))):
        means, lo, hi = [], [], []
        for method, shift, _ in GROUPS:
            vals = [r[f"domain_balanced_{key}"] for r in rows
                    if r["method"] == method and r["shift"] == shift
                    and r.get(f"domain_balanced_{key}") is not None]
            if not vals:
                means.append(np.nan); lo.append(0.0); hi.append(0.0); continue
            m = float(np.mean(vals))
            means.append(m); lo.append(m - float(np.min(vals))); hi.append(float(np.max(vals)) - m)
        means = np.array(means)
        ax.bar(x + (j - 0.5) * w, means, w, color=color, edgecolor="black", linewidth=0.5,
               label=key.capitalize(), yerr=[lo, hi], capsize=2, error_kw={"lw": 0.8})
    ax.axhline(0.5, color="0.35", lw=0.7, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels([g[2] for g in GROUPS], fontsize=7)
    ax.set_ylabel("Domain balanced accuracy")
    ax.set_ylim(0.3, 1.0)
    ax.grid(axis="y", alpha=0.25, lw=0.4)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=2, fontsize=7)
    fig.tight_layout()
    FIG.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"a_e2_domain_diag.{ext}", bbox_inches="tight")
    print("wrote figures/a_e2_domain_diag.{png,pdf}")


if __name__ == "__main__":
    main()
