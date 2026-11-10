#!/usr/bin/env python3
"""λ=0 归因对照：Source Only（75 ep）/ DANN（150 ep）/ DANN λ=0（150 ep，无对抗梯度）。

目的：把“多训 75 epoch + 目标域前向”与“对抗对齐”分离。
输出 results/controls_comparison.{json,md} 与 figures/a_e2_ge_controls.{png,pdf}。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
import evaluate_uda as EU  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

EA = EU.load_eval_module()
FIG = HERE / "figures"
B_E2 = EU.B_E2
SHIFTS = ["desync50", "desync100"]
SEED = 42
COLOR = {"source_only": "#1f77b4", "dann": "#ff7f0e", "lam0": "#9467bd"}
LABEL = {"source_only": "Source Only (75 ep)", "dann": "DANN (150 ep)",
         "lam0": "DANN λ=0 (150 ep, no adv.)"}


def preds_path(kind: str, shift: str) -> Path:
    if kind == "source_only":
        return B_E2 / "results/predictions" / shift / f"cnn_s{SEED}.npy"
    if kind == "dann":
        return HERE / "results" / f"dann_{shift}_s{SEED}" / "predictions_attack.npy"
    return HERE / "results" / f"dann_{shift}_s{SEED}_lam0" / "predictions_attack.npy"


def load_labels(shift: str):
    with h5py.File(EU.SHIFTS[shift], "r") as h:
        meta = np.asarray(h["Attack_traces/metadata"])
    labels = EU.SBOX[np.bitwise_xor(np.asarray(meta["plaintext"])[:, 2],
                                    np.uint8(int(meta["key"][0, 2])))].astype(np.int64)
    return meta, labels


def metrics(probs, meta, labels, adapted):
    curve = EA.ranks(probs, np.asarray(meta["plaintext"]), np.asarray(meta["key"]), 2, step=100)
    rec = EU.summary_from_curve(curve, len(probs), adapted=adapted)
    rec["attack_accuracy"] = round(float((probs.argmax(1) == labels).mean()), 6)
    rec["mean_max_softmax"] = round(float(probs.max(1).mean()), 6)
    p = np.clip(probs, 1e-12, 1.0)
    rec["mean_entropy_nats"] = round(float(-(p * np.log(p)).sum(1).mean()), 6)
    rec["curve"] = curve
    return rec


def main():
    rows = []
    curves = {}
    for shift in SHIFTS:
        meta, labels = load_labels(shift)
        for kind in ("source_only", "dann", "lam0"):
            path = preds_path(kind, shift)
            if not path.exists():
                print(f"[skip] {kind} {shift}: {path}")
                continue
            probs = np.load(path)
            rec = metrics(probs, meta, labels, adapted=(kind != "source_only"))
            rec.update({"kind": kind, "shift": shift, "seed": SEED, "predictions": str(path)})
            rows.append(rec)
            curves[(kind, shift)] = rec["curve"]
            print(f"{shift:9s} {kind:11s} GE@1k={rec['ge_at_1000']} GE@10k={rec['ge_at_10000']} "
                  f"NTGE={rec['ntge_first']} stable={rec['ntge_stable']} acc={rec['attack_accuracy']*100:.2f}% "
                  f"conf={rec['mean_max_softmax']:.3f} H={rec['mean_entropy_nats']:.2f}")

    out_dir = HERE / "results"
    (out_dir / "controls_comparison.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    # 对照表（markdown）
    md = ["### λ=0 归因对照（seed 42）", "",
          "| 偏移 | 条件 | 训练 epoch | GE@1k | GE@10k | NTGE(first) | NTGE(stable) | acc | conf | H |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for shift in SHIFTS:
        sub = {r["kind"]: r for r in rows if r["shift"] == shift}
        for kind, label, ep in (("source_only", "Source Only", 75), ("dann", "DANN", 150),
                                ("lam0", "DANN λ=0（无对抗）", 150)):
            if kind not in sub:
                continue
            r = sub[kind]
            md.append(f"| {shift} | {label} | {ep} | {r['ge_at_1000']} | {r['ge_at_10000']} | "
                      f"{r['ntge_first'] or '—'} | {r['ntge_stable'] or '—'} | "
                      f"{r['attack_accuracy']*100:.2f}% | {r['mean_max_softmax']:.3f} | "
                      f"{r['mean_entropy_nats']:.2f} |")
    md.append("")
    for shift in SHIFTS:
        sub = {r["kind"]: r for r in rows if r["shift"] == shift}
        if {"source_only", "dann", "lam0"} <= set(sub):
            md.append(f"- {shift}: ΔGE@10k (λ=0 − Source Only) = "
                      f"{sub['lam0']['ge_at_10000'] - sub['source_only']['ge_at_10000']:+d}; "
                      f"(DANN − Source Only) = {sub['dann']['ge_at_10000'] - sub['source_only']['ge_at_10000']:+d}; "
                      f"(DANN − λ=0) = {sub['dann']['ge_at_10000'] - sub['lam0']['ge_at_10000']:+d}")
    (out_dir / "controls_comparison.md").write_text("\n".join(md), encoding="utf-8")

    # 图：两偏移，三种条件（seed 42）；线性 + 对数双版，serif，Y 轴 0-256 全量程
    plt.rcParams.update({
        "font.family": "serif", "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
        "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.linewidth": 0.8, "lines.linewidth": 1.4, "grid.linewidth": 0.4,
    })
    for scale, suffix in (("linear", ""), ("log", "_log")):
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
        for ax, shift in zip(axes, SHIFTS):
            for kind in ("source_only", "dann", "lam0"):
                curve = curves.get((kind, shift))
                if not curve:
                    continue
                xs = [0] + [n for n, _ in curve]
                ys = [255] + [r for _, r in curve]
                ys = np.asarray(ys, dtype=float)
                if scale == "log":
                    ys = np.where(ys > 0, ys, 0.5)
                ax.plot(xs, ys, color=COLOR[kind], lw=1.2, label=LABEL[kind])
            ax.set_xlim(0, 10000)
            ax.set_xlabel("Number of attack traces")
            ax.set_title(shift)
            ax.grid(alpha=0.3)
            if scale == "log":
                ax.set_yscale("log")
                ax.set_ylim(0.45, 320)
                ax.set_yticks([0.5, 1, 4, 16, 64, 256])
                ax.set_yticklabels(["0", "1", "4", "16", "64", "256"])
            else:
                ax.set_ylim(0, 256)
                ax.set_yticks([0, 64, 128, 192, 256])
        axes[0].set_ylabel("Key rank")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, frameon=False, ncol=3, fontsize=7,
                   loc="lower center", bbox_to_anchor=(0.5, -0.02))
        fig.tight_layout(rect=(0, 0.11, 1, 1))
        fig.savefig(FIG / f"a_e2_ge_controls{suffix}.png", dpi=300)
        fig.savefig(FIG / f"a_e2_ge_controls{suffix}.pdf")
    print("wrote results/controls_comparison.{json,md} and figures/a_e2_ge_controls{,_log}.{png,pdf}")


if __name__ == "__main__":
    main()
