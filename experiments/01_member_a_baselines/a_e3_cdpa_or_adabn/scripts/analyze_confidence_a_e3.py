#!/usr/bin/env python3
"""A-E3 置信度/校准诊断：四方法 x 两偏移的 acc / ECE / confident-wrong / 熵。

复用 A-E2 的 analyze_confidence.py（ECE 15 bin、SBOX 标签、confident-wrong 定义），
只替换预测来源：source_only（B-E2）、dann/cdan（A-E2）、cdpa_mmd（本实验）。
标签仅用于事后诊断，不参与训练与模型选择。

输出 results/confidence_analysis.csv。
"""
from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[2]
A_E2 = REPO / "experiments/01_member_a_baselines/a_e2_uda_baselines"
B_E2 = REPO / "experiments/02_member_b_method/b_e2_architecture_shift"
EVAL = HERE / "results/eval/eval_all.json"
OUT = HERE / "results/confidence_analysis.csv"
SEEDS = [42, 43, 44]
SHIFTS = ["desync50", "desync100"]


def load_a_e2_module():
    spec = importlib.util.spec_from_file_location(
        "a_e2_conf", str(A_E2 / "scripts/analyze_confidence.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def pred_path(method, shift, seed):
    if method == "source_only":
        return B_E2 / "results/predictions" / shift / f"cnn_s{seed}.npy"
    run = f"cdpa_{shift}_s{seed}" if method == "cdpa_mmd" else f"{method}_{shift}_s{seed}"
    base = HERE if method == "cdpa_mmd" else A_E2
    return base / "results" / run / "predictions_attack.npy"


def main():
    CON = load_a_e2_module()
    labels = {s: CON.labels_for(s) for s in SHIFTS}
    rows = []
    for shift in SHIFTS:
        for method in ["source_only", "dann", "cdan", "cdpa_mmd"]:
            for seed in SEEDS:
                p = pred_path(method, shift, seed)
                if not p.exists():
                    print(f"[skip] {method} {shift} s{seed}: {p}")
                    continue
                probs = np.load(p).astype(np.float64)
                rec = {"shift": shift, "method": method, "seed": seed,
                       "run": f"{method}_{shift}_s{seed}"}
                rec.update(CON.analyze(probs, labels[shift]))
                rows.append(rec)
    if not rows:
        raise SystemExit("no predictions found")
    fields = list(rows[0].keys())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {OUT}")
    for shift in SHIFTS:
        print(f"\n=== {shift} ===")
        for method in ["source_only", "dann", "cdan", "cdpa_mmd"]:
            rs = [r for r in rows if r["shift"] == shift and r["method"] == method]
            if not rs:
                continue
            print(f"{method:11s} acc {np.mean([r['attack_accuracy'] for r in rs]):.4f} "
                  f"ECE {np.mean([r['ece_15'] for r in rs]):.4f} "
                  f"conf-wrong>=0.9 {np.mean([r['frac_confwrong_ge_0.9'] for r in rs]):.4f} "
                  f"conf>=0.9 {np.mean([r['frac_conf_ge_0.9'] for r in rs]):.4f} "
                  f"entropy {np.mean([r['mean_entropy_nats'] for r in rs]):.3f}")


if __name__ == "__main__":
    main()
