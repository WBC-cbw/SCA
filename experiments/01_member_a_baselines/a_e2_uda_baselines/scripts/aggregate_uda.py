#!/usr/bin/env python3
"""A-E2 汇总：方法 x 偏移的 NTGE / GE@10k / SR / acc / 域判别诊断（3 seed 均值±std）。

口径
- NTGE 主口径 ntge_first（首次 GE=0）；ntge_stable（此后一直为 0）作稳健性对照。
- 未达成的 run 截尾记 10000（=攻击轨迹上限），并在 reached 计数里区分。
- N_total@GE=1 = N_attack@GE=1 + 45000（DANN/CDAN）；Source Only 无适应轨迹。
- paired_delta 用同 seed 的 Source Only 作配对基线。
"""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, pstdev

import numpy as np

HERE = Path(__file__).resolve().parents[1]
EVAL = HERE / "results/eval/eval_all.json"
RES = HERE / "results"
METHODS = ["source_only", "dann", "cdan"]
SHIFTS = ["desync50", "desync100"]
CENSOR = 10000.0


def load():
    rows = json.loads(EVAL.read_text(encoding="utf-8"))
    data = {}
    for r in rows:
        data.setdefault((r["method"], r["shift"]), []).append(r)
    for k in data:
        data[k].sort(key=lambda r: r["seed"])
    return data


def summarize(vals):
    vals = np.asarray(vals, dtype=float)
    return float(vals.mean()), float(vals.std(ddof=1)) if len(vals) > 1 else 0.0


def main():
    data = load()
    metrics = {}
    run_rows = []
    for method in METHODS:
        metrics[method] = {}
        for shift in SHIFTS:
            rows = data.get((method, shift), [])
            if not rows:
                continue
            ntge_first = [CENSOR if r["ntge_first"] is None else r["ntge_first"] for r in rows]
            ntge_stable = [CENSOR if r["ntge_stable"] is None else r["ntge_stable"] for r in rows]
            ge10k = [r["ge_at_10000"] for r in rows]
            ge1k = [r["ge_at_1000"] for r in rows]
            acc = [r["attack_accuracy"] for r in rows]
            conf = [r["mean_max_softmax"] for r in rows]
            ent = [r["mean_entropy_nats"] for r in rows]
            m = {
                "n_runs": len(rows),
                "ntge_first_values": ntge_first,
                "ntge_first_mean": summarize(ntge_first)[0],
                "ntge_first_std": summarize(ntge_first)[1],
                "ntge_first_reached": sum(r["ntge_first"] is not None for r in rows),
                "ntge_stable_values": ntge_stable,
                "ntge_stable_mean": summarize(ntge_stable)[0],
                "ntge_stable_std": summarize(ntge_stable)[1],
                "ntge_stable_reached": sum(r["ntge_stable"] is not None for r in rows),
                "transient_runs": sum(bool(r["transient_only"]) for r in rows),
                "ge_at_1000_mean": summarize(ge1k)[0],
                "ge_at_1000_std": summarize(ge1k)[1],
                "ge_at_10000_values": ge10k,
                "ge_at_10000_mean": summarize(ge10k)[0],
                "ge_at_10000_std": summarize(ge10k)[1],
                "sr_at_10000_mean": summarize([r["sr_at_10000"] for r in rows])[0],
                "attack_accuracy_mean": summarize(acc)[0],
                "attack_accuracy_std": summarize(acc)[1],
                "mean_max_softmax_mean": summarize(conf)[0],
                "mean_entropy_mean": summarize(ent)[0],
                "domain_balanced_before_mean": summarize(
                    [r["domain_balanced_before"] for r in rows if r["domain_balanced_before"] is not None])[0]
                if any(r["domain_balanced_before"] is not None for r in rows) else None,
                "domain_balanced_after_mean": summarize(
                    [r["domain_balanced_after"] for r in rows if r["domain_balanced_after"] is not None])[0]
                if any(r["domain_balanced_after"] is not None for r in rows) else None,
                "n_adapt": 0 if method == "source_only" else 45000,
            }
            m["n_total_at_ge1_mean"] = summarize(
                [CENSOR + m["n_adapt"] if r["ntge_first"] is None else r["ntge_first"] + m["n_adapt"]
                 for r in rows])[0]
            metrics[method][shift] = m

            for r in rows:
                run_rows.append({
                    "method": method, "shift": shift, "seed": r["seed"], "run": r["run"],
                    "ntge_first": r["ntge_first"], "ntge_stable": r["ntge_stable"],
                    "n_total_at_ge1": r["n_total_at_ge1"],
                    "ge_at_1000": r["ge_at_1000"], "ge_at_10000": r["ge_at_10000"],
                    "min_ge": r["min_ge"], "transient_only": r["transient_only"],
                    "sr_at_10000": r["sr_at_10000"],
                    "attack_accuracy": r["attack_accuracy"],
                    "mean_max_softmax": r["mean_max_softmax"],
                    "mean_entropy_nats": r["mean_entropy_nats"],
                    "domain_balanced_before": r["domain_balanced_before"],
                    "domain_balanced_after": r["domain_balanced_after"],
                    "train_seconds": r["train_seconds"],
                })

    # 同 seed 配对差（相对 Source Only）
    paired = {}
    for method in ["dann", "cdan"]:
        paired[method] = {}
        for shift in SHIFTS:
            base = {r["seed"]: r for r in data.get(("source_only", shift), [])}
            cur = {r["seed"]: r for r in data.get((method, shift), [])}
            seeds = sorted(set(base) & set(cur))
            if not seeds:
                continue
            d_ge = [cur[s]["ge_at_10000"] - base[s]["ge_at_10000"] for s in seeds]
            d_ntge = [(CENSOR if cur[s]["ntge_first"] is None else cur[s]["ntge_first"])
                      - (CENSOR if base[s]["ntge_first"] is None else base[s]["ntge_first"]) for s in seeds]
            paired[method][shift] = {
                "seeds": seeds,
                "delta_ge_at_10000": d_ge,
                "delta_ge_at_10000_mean": float(np.mean(d_ge)),
                "delta_ntge_first": d_ntge,
                "delta_ntge_first_mean": float(np.mean(d_ntge)),
                "improved_ge": int(sum(d < 0 for d in d_ge)),
                "worsened_ge": int(sum(d > 0 for d in d_ge)),
            }

    out = {
        "experiment": "A-E2 DANN/CDAN baselines",
        "protocol": {
            "source_train": "ASCAD fixed Profiling[0:45000]",
            "source_val": "ASCAD fixed Profiling[45000:50000]",
            "target_adapt": "desync Profiling[0:45000] (unlabeled)",
            "target_domain_val": "desync Profiling[45000:50000] (diagnostic only)",
            "target_attack": "desync Attack_traces[0:10000]",
            "target_byte": 2, "seeds": [42, 43, 44], "step": 100,
            "ntge_censoring": 10000, "n_adapt": 45000,
        },
        "metrics": metrics,
        "paired_delta_vs_source_only": paired,
    }
    (RES / "summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    fields = ["method", "shift", "seed", "run", "ntge_first", "ntge_stable", "n_total_at_ge1",
              "ge_at_1000", "ge_at_10000", "min_ge", "transient_only", "sr_at_10000",
              "attack_accuracy", "mean_max_softmax", "mean_entropy_nats",
              "domain_balanced_before", "domain_balanced_after", "train_seconds"]
    import csv
    with (RES / "metrics_a_e2.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(run_rows)

    print("method       shift      NTGE(first)     GE@10k      reached  transient  acc")
    for method in METHODS:
        for shift in SHIFTS:
            m = metrics.get(method, {}).get(shift)
            if not m:
                continue
            print(f"{method:12s} {shift:9s} {m['ntge_first_mean']:7.0f}±{m['ntge_first_std']:<6.0f} "
                  f"{m['ge_at_10000_mean']:6.1f}±{m['ge_at_10000_std']:<6.1f} "
                  f"{m['ntge_first_reached']}/{m['n_runs']}      {m['transient_runs']}        "
                  f"{m['attack_accuracy_mean']:.4f}")
    print("paired delta GE@10k:", {k: {s: round(v['delta_ge_at_10000_mean'], 1) for s, v in d.items()}
                                  for k, d in paired.items()})


if __name__ == "__main__":
    main()
