#!/usr/bin/env python3
"""B-E2 汇总：结构 x 偏移强度的 NTGE / GE / SR / accuracy 矩阵（3 seed 均值±标准差）。

NTGE 的两种口径（必须分开，否则会把“瞬时归零”误报为收敛）：
  - ntge_first  : 曲线上第一次出现 rank=0 的轨迹数（任务要求的 NTGE 定义）
  - ntge_stable : 第一次 rank=0 且此后直到 10000 条一直保持 0；跨域退化时该值为 None
未达到的 run 在均值统计中按 10000 截尾（censored），并单独给出达成的 run 数。

输出: results/metrics_b_e2.csv, results/summary.json
"""
import csv
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
EVAL = HERE / "results/eval"
RES = HERE / "results"

ARCHS = ["cnn", "resnet1d", "mlp"]
SHIFTS = ["fixed", "desync50", "desync100"]
CENSOR = 10000


def mean_std(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    if len(vals) == 1:
        return float(vals[0]), 0.0
    return float(statistics.mean(vals)), float(statistics.stdev(vals))


def censored(vals):
    return [CENSOR if v is None else v for v in vals]


def main():
    runs = []
    for shift in SHIFTS:
        for f in sorted((EVAL / shift).glob("*.json")):
            if f.name == "eval_all.json":
                continue
            runs.append(json.loads(f.read_text(encoding="utf-8")))

    rows = []
    for r in runs:
        curve = {p["traces"]: p["ge"] for p in r["ge_curve"]}
        xs = sorted(curve)
        zeros = [x for x in xs if curve[x] == 0]
        first = zeros[0] if zeros else None
        stable = None
        if first is not None:
            tail = [x for x in xs if x >= first]
            if all(curve[x] == 0 for x in tail):
                stable = first
        rows.append({
            "arch": r["arch"], "shift": r["shift"], "seed": r["seed"], "run": r["run"],
            "ntge_first": first, "ntge_stable": stable,
            "zero_points": len(zeros),
            "ge_min": min(curve.values()),
            "ge_at_1000": r["ge_at_1000"], "ge_at_5000": r["ge_at_5000"],
            "ge_at_10000": r["ge_at_10000"],
            "sr_at_1000": r["sr_at_1000"], "sr_at_10000": r["sr_at_10000"],
            "attack_accuracy": r["attack_accuracy"],
            "mean_max_softmax": r["mean_max_softmax"],
            "mean_entropy_nats": r["mean_entropy_nats"],
        })

    fields = ["arch", "shift", "seed", "run", "ntge_first", "ntge_stable", "zero_points",
              "ge_min", "ge_at_1000", "ge_at_5000", "ge_at_10000",
              "sr_at_1000", "sr_at_10000", "attack_accuracy",
              "mean_max_softmax", "mean_entropy_nats"]
    with (RES / "metrics_b_e2.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)

    grouped = {}
    for row in rows:
        grouped.setdefault((row["arch"], row["shift"]), []).append(row)

    summary = {"experiment": "B-E2", "seeds": [42, 43, 44],
               "target_byte": 2, "ntge_censor_traces": CENSOR,
               "metrics": {}, "matrix": {}}
    for arch in ARCHS:
        summary["metrics"][arch] = {}
        summary["matrix"][arch] = {}
        for shift in SHIFTS:
            g = sorted(grouped.get((arch, shift), []), key=lambda r: r["seed"])
            if not g:
                continue
            first = [r["ntge_first"] for r in g]
            stable = [r["ntge_stable"] for r in g]
            ge10 = [r["ge_at_10000"] for r in g]
            m_first, s_first = mean_std(censored(first))
            m_stab, s_stab = mean_std(censored(stable))
            m_ge10, s_ge10 = mean_std(ge10)
            m_acc, s_acc = mean_std([r["attack_accuracy"] for r in g])
            item = {
                "seeds": [r["seed"] for r in g],
                "ntge_first": first,
                "ntge_first_reached": sum(1 for v in first if v is not None),
                "ntge_first_censored_mean": round(m_first, 1),
                "ntge_first_censored_std": round(s_first, 1) if s_first is not None else None,
                "ntge_stable": stable,
                "ntge_stable_reached": sum(1 for v in stable if v is not None),
                "ntge_stable_censored_mean": round(m_stab, 1),
                "ntge_stable_censored_std": round(s_stab, 1) if s_stab is not None else None,
                "ge_at_10000": ge10,
                "ge_at_10000_mean": round(m_ge10, 1),
                "ge_at_10000_std": round(s_ge10, 2),
                "ge_at_1000_mean": round(mean_std([r["ge_at_1000"] for r in g])[0], 1),
                "ge_min_mean": round(mean_std([r["ge_min"] for r in g])[0], 1),
                "sr_at_10000_mean": round(mean_std([r["sr_at_10000"] for r in g])[0], 3),
                "zero_points_mean": round(mean_std([r["zero_points"] for r in g])[0], 1),
                "attack_accuracy_mean": round(m_acc, 6),
                "attack_accuracy_std": round(s_acc, 6) if s_acc is not None else None,
                "mean_max_softmax_mean": round(mean_std([r["mean_max_softmax"] for r in g])[0], 4),
                "mean_entropy_nats_mean": round(mean_std([r["mean_entropy_nats"] for r in g])[0], 4),
            }
            summary["metrics"][arch][shift] = item
            summary["matrix"][arch][shift] = {
                "ntge_stable_censored_mean": item["ntge_stable_censored_mean"],
                "ntge_stable_reached": item["ntge_stable_reached"],
                "ge_at_10000_mean": item["ge_at_10000_mean"],
                "attack_accuracy_mean": item["attack_accuracy_mean"],
            }

    # 同 seed 配对退化量：desync 相对 fixed 的 GE@10000 变化
    summary["paired_delta_ge10000"] = {}
    for arch in ARCHS:
        fixed = {r["seed"]: r["ge_at_10000"] for r in grouped.get((arch, "fixed"), [])}
        d = {}
        for shift in ["desync50", "desync100"]:
            vals = [r["ge_at_10000"] - fixed[r["seed"]]
                    for r in grouped.get((arch, shift), []) if r["seed"] in fixed]
            d[shift] = vals
        summary["paired_delta_ge10000"][arch] = d

    (RES / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["matrix"], indent=2))
    print("wrote", RES / "metrics_b_e2.csv", "and", RES / "summary.json")


if __name__ == "__main__":
    main()
