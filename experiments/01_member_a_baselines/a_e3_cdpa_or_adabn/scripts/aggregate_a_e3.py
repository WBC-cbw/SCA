#!/usr/bin/env python3
"""A-E3 汇总：四方法 x 两偏移的 NTGE / GE@10k / SR / 成本，3 seed 均值±std。

口径（与 A-E2 一致，便于并表）：
- NTGE 主口径 ntge_first（首次 GE=0）；ntge_stable（此后一直为 0）作稳健性对照。
- 未达成的 run 截尾记 10000（攻击轨迹上限），reached 计数区分。
- N_total@GE=1 = N_attack@GE=1 + 45000（有适应的三种方法）；Source Only 无适应轨迹。
- 成本只统计训练时间；适应轨迹作为攻击成本单列（N_total 口径）。
"""
from __future__ import annotations

import csv
import json
from statistics import mean, pstdev
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
EVAL = HERE / "results/eval/eval_all.json"
RES = HERE / "results"
METHODS = ["source_only", "dann", "cdan", "cdpa_mmd"]
LABELS = {"source_only": "Source Only", "dann": "DANN", "cdan": "CDAN",
          "cdpa_mmd": "CDPA-MMD"}
SHIFTS = ["desync50", "desync100"]
CENSOR = 10000.0


def summarize(vals):
    vals = np.asarray(vals, dtype=float)
    return float(vals.mean()), float(vals.std(ddof=1)) if len(vals) > 1 else 0.0


def main():
    rows = json.loads(EVAL.read_text(encoding="utf-8"))
    data = {}
    for r in rows:
        data.setdefault((r["method"], r["shift"]), []).append(r)
    for k in data:
        data[k].sort(key=lambda r: r["seed"])

    metrics = {}
    per_run = []
    for method in METHODS:
        metrics[method] = {}
        for shift in SHIFTS:
            rs = data.get((method, shift), [])
            if not rs:
                continue
            ntge = [CENSOR if r["ntge_first"] is None else float(r["ntge_first"]) for r in rs]
            ntge_st = [CENSOR if r["ntge_stable"] is None else float(r["ntge_stable"]) for r in rs]
            adapted = method != "source_only"
            n_adapt = 45000 if adapted else 0
            m = {
                "n_runs": len(rs),
                "method": method,
                "label": LABELS[method],
                "shift": shift,
                "ntge_first_values": ntge,
                "ntge_first_mean": summarize(ntge)[0],
                "ntge_first_std": summarize(ntge)[1],
                "ntge_first_reached": int(sum(r["ntge_first"] is not None for r in rs)),
                "ntge_stable_values": ntge_st,
                "ntge_stable_mean": summarize(ntge_st)[0],
                "ntge_stable_std": summarize(ntge_st)[1],
                "ntge_stable_reached": int(sum(r["ntge_stable"] is not None for r in rs)),
                "transient_runs": int(sum(bool(r["transient_only"]) for r in rs)),
                "ge_at_1000_mean": summarize([r["ge_at_1000"] for r in rs])[0],
                "ge_at_1000_std": summarize([r["ge_at_1000"] for r in rs])[1],
                "ge_at_10000_values": [r["ge_at_10000"] for r in rs],
                "ge_at_10000_mean": summarize([r["ge_at_10000"] for r in rs])[0],
                "ge_at_10000_std": summarize([r["ge_at_10000"] for r in rs])[1],
                "sr_at_1000_mean": summarize([r["sr_at_1000"] for r in rs])[0],
                "sr_at_10000_mean": summarize([r["sr_at_10000"] for r in rs])[0],
                "attack_accuracy_mean": summarize([r["attack_accuracy"] for r in rs])[0],
                "attack_accuracy_std": summarize([r["attack_accuracy"] for r in rs])[1],
                "mean_max_softmax_mean": summarize([r["mean_max_softmax"] for r in rs])[0],
                "mean_entropy_mean": summarize([r["mean_entropy_nats"] for r in rs])[0],
                "n_adapt": n_adapt,
                "n_total_at_ge1_mean": summarize([v + n_adapt for v in ntge])[0],
                "n_total_at_ge1_std": summarize([v + n_adapt for v in ntge])[1],
                "train_seconds_mean": summarize([r["train_seconds"] for r in rs
                                                 if r.get("train_seconds") is not None])[0]
                if any(r.get("train_seconds") is not None for r in rs) else None,
                "train_seconds_total": float(sum(r["train_seconds"] for r in rs
                                                 if r.get("train_seconds") is not None)),
            }
            metrics[method][shift] = m

            for r in rs:
                per_run.append({
                    "method": method, "label": LABELS[method], "shift": shift,
                    "seed": r["seed"], "run": r["run"],
                    "ntge_first": r["ntge_first"], "ntge_stable": r["ntge_stable"],
                    "ge_at_1000": r["ge_at_1000"], "ge_at_10000": r["ge_at_10000"],
                    "sr_at_1000": r["sr_at_1000"], "sr_at_10000": r["sr_at_10000"],
                    "attack_accuracy": r["attack_accuracy"],
                    "n_attack_at_ge1": r["n_attack_at_ge1"],
                    "n_total_at_ge1": r["n_total_at_ge1"],
                    "train_seconds": r.get("train_seconds"),
                    "lambda_mmd": r.get("lambda_mmd"),
                    "final_mmd_loss": r.get("final_mmd_loss"),
                })

    # 配对比较（同 seed，NTGE 截尾后）
    paired = {}
    for shift in SHIFTS:
        base = data.get(("source_only", shift), [])
        base_by_seed = {r["seed"]: r for r in base}
        for method in ["dann", "cdan", "cdpa_mmd"]:
            rs = data.get((method, shift), [])
            deltas_ntge, deltas_ge = [], []
            for r in rs:
                b = base_by_seed.get(r["seed"])
                if not b:
                    continue
                v = CENSOR if r["ntge_first"] is None else r["ntge_first"]
                bv = CENSOR if b["ntge_first"] is None else b["ntge_first"]
                deltas_ntge.append(v - bv)
                deltas_ge.append(r["ge_at_10000"] - b["ge_at_10000"])
            paired[f"{method}_vs_source_only@{shift}"] = {
                "delta_ntge_values": deltas_ntge,
                "delta_ntge_mean": summarize(deltas_ntge)[0] if deltas_ntge else None,
                "delta_ge10k_values": deltas_ge,
                "delta_ge10k_mean": summarize(deltas_ge)[0] if deltas_ge else None,
            }
    # CDPA 对通用 UDA（DANN/CDAN）逐 seed 胜负
    for shift in SHIFTS:
        for other in ["dann", "cdan"]:
            a = {r["seed"]: r for r in data.get(("cdpa_mmd", shift), [])}
            b = {r["seed"]: r for r in data.get((other, shift), [])}
            wins = ties = losses = 0
            for s in sorted(set(a) & set(b)):
                va = CENSOR if a[s]["ntge_first"] is None else a[s]["ntge_first"]
                vb = CENSOR if b[s]["ntge_first"] is None else b[s]["ntge_first"]
                if va < vb:
                    wins += 1
                elif va == vb:
                    ties += 1
                else:
                    losses += 1
            paired[f"cdpa_vs_{other}@{shift}"] = {"wins": wins, "ties": ties, "losses": losses}

    out = {"metrics": metrics, "paired": paired,
           "censor": CENSOR, "n_adapt": 45000}
    (RES / "summary.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    fields = ["method", "label", "shift", "seed", "run", "ntge_first", "ntge_stable",
              "ge_at_1000", "ge_at_10000", "sr_at_1000", "sr_at_10000",
              "attack_accuracy", "n_attack_at_ge1", "n_total_at_ge1",
              "train_seconds", "lambda_mmd", "final_mmd_loss"]
    with (RES / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in per_run:
            w.writerow({k: r.get(k) for k in fields})

    tfields = ["method", "label", "shift", "n_runs", "ntge_first_mean", "ntge_first_std",
               "ntge_first_reached", "ntge_stable_mean", "ntge_stable_std",
               "ntge_stable_reached", "transient_runs", "ge_at_1000_mean",
               "ge_at_10000_mean", "ge_at_10000_std", "sr_at_10000_mean",
               "attack_accuracy_mean", "n_adapt", "n_total_at_ge1_mean",
               "train_seconds_mean", "train_seconds_total"]
    with (RES / "table_methods.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(tfields)
        for shift in SHIFTS:
            for method in METHODS:
                m = metrics.get(method, {}).get(shift)
                if not m:
                    continue
                w.writerow([m.get(k) for k in tfields])

    for shift in SHIFTS:
        print(f"\n=== {shift} ===")
        for method in METHODS:
            m = metrics.get(method, {}).get(shift)
            if not m:
                continue
            print(f"{m['label']:12s} NTGE {m['ntge_first_mean']:8.1f}±{m['ntge_first_std']:6.1f} "
                  f"({m['ntge_first_reached']}/{m['n_runs']})  GE@10k {m['ge_at_10000_mean']:6.1f}±"
                  f"{m['ge_at_10000_std']:5.1f}  SR@10k {m['sr_at_10000_mean']:.2f}  "
                  f"acc {m['attack_accuracy_mean']:.4f}  N_total {m['n_total_at_ge1_mean']:.0f}")
    print("\npaired:", json.dumps(paired, ensure_ascii=False))


if __name__ == "__main__":
    main()
