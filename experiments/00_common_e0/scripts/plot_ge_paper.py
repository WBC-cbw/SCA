#!/usr/bin/env python3
"""按论文标准重画 E0 GE 曲线：全量程 Y 轴 + 单栏尺寸 + PDF 矢量输出。

数据来源不变（results/eval/summary.json），仅改绘图样式。
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "experiments/00_common_e0"

# 论文单栏样式
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.4,
    "grid.linewidth": 0.4,
})

curve = json.loads((RUN / "results/eval/summary.json")
                   .read_text(encoding="utf-8"))["predictions_attack"]["ge_curve"]
xs = [0] + [p["traces"] for p in curve]          # 补原点：0 条轨迹时 rank=255
ys = [255] + [p["ge"] for p in curve]

for scale, suffix in [("linear", ""), ("log", "_log")]:
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    if scale == "log":
        # rank=0 在对数轴无法显示，按惯例映射到 0.5（触底即收敛）
        ax.plot(xs, [r if r > 0 else 0.5 for r in ys],
                color="#1f77b4", label="Source Only (official CNN)")
        ax.set_yscale("log")
        ax.set_ylim(0.5, 256)
        ax.set_yticks([1, 4, 16, 64, 256])
        ax.set_yticklabels(["1", "4", "16", "64", "256"])
    else:
        ax.plot(xs, ys, color="#1f77b4", label="Source Only (official CNN)")
        ax.set_ylim(0, 256)
    ax.set_xlim(0, 10000)
    ax.set_xlabel("Number of attack traces")
    ax.set_ylabel("Key rank")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(RUN / f"figures/e0_ge_curve{suffix}.png", dpi=300)
    fig.savefig(RUN / f"figures/e0_ge_curve{suffix}.pdf")
    plt.close(fig)

print("paper-style figures saved (linear + log, png + pdf)")
