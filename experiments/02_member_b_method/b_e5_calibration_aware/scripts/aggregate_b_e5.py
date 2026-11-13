#!/usr/bin/env python3
"""B-E5 汇总：七方法 x 两偏移，3 seed 均值±std，含胜负判定与权重函数对照表。

口径与 A-E2/A-E3/B-E3 完全一致：
- NTGE 主口径 ntge_first（首次 GE=0），ntge_stable（此后一直 0）作稳健性对照；
- 未达成截尾 10000；N_total@GE=1 = N_attack + 45000（Source Only 无适应）；
- 胜负按任务书硬性判据（见 verdict）：
    desync50 胜 = NTGE <= 5700 且 3/3 seed 稳定 GE=0
    desync100 胜 = GE@10k <= 4.0 且 NTGE <= 3700
  若 NTGE 打平但 N_total 更高 -> 未胜出。
- 本实验唯一方法变量是权重函数，故额外输出 calib 的 T 与校准权重分位，
  以及 calib vs full / calib vs CDPA-MMD 的配对差。
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
RES = HERE / "results"
EVAL = RES / "eval/eval_all.json"
METHODS = ["source_only", "dann", "cdan", "cdpa_mmd", "cdan_sca_full", "cdan_sca_marg",
           "cdan_sca_calib"]
LABELS = {"source_only": "Source Only", "dann": "DANN", "cdan": "CDAN",
          "cdpa_mmd": "CDPA-MMD", "cdan_sca_full": "CDAN-SCA full",
          "cdan_sca_marg": "Abl.1 marginal+w", "cdan_sca_calib": "CDAN-SCA-calib"}
SHIFTS = ["desync50", "desync100"]
CENSOR = 10000.0
THRESH = {"desync50": {"ntge": 5700.0}, "desync100": {"ntge": 3700.0, "ge10k": 4.0}}


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

    metrics, per_run = {}, []
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
                "n_runs": len(rs), "method": method, "label": LABELS[method], "shift": shift,
                "ntge_first_values": ntge,
                "ntge_first_mean": summarize(ntge)[0], "ntge_first_std": summarize(ntge)[1],
                "ntge_first_reached": int(sum(r["ntge_first"] is not None for r in rs)),
                "ntge_stable_values": ntge_st,
                "ntge_stable_mean": summarize(ntge_st)[0], "ntge_stable_std": summarize(ntge_st)[1],
                "ntge_stable_reached": int(sum(r["ntge_stable"] is not None for r in rs)),
                "ge_at_1000_mean": summarize([r["ge_at_1000"] for r in rs])[0],
                "ge_at_10000_values": [r["ge_at_10000"] for r in rs],
                "ge_at_10000_mean": summarize([r["ge_at_10000"] for r in rs])[0],
                "ge_at_10000_std": summarize([r["ge_at_10000"] for r in rs])[1],
                "sr_at_1000_mean": summarize([r["sr_at_1000"] for r in rs])[0],
                "sr_at_10000_mean": summarize([r["sr_at_10000"] for r in rs])[0],
                "attack_accuracy_mean": summarize([r["attack_accuracy"] for r in rs])[0],
                "mean_entropy_mean": summarize([r["mean_entropy_nats"] for r in rs])[0],
                "n_adapt": n_adapt,
                "n_attack_at_ge1_mean": summarize(ntge)[0],
                "n_total_at_ge1_mean": summarize([v + n_adapt for v in ntge])[0],
                "n_total_at_ge1_std": summarize([v + n_adapt for v in ntge])[1],
                "train_seconds_mean": summarize([r["train_seconds"] for r in rs
                                                 if r.get("train_seconds") is not None])[0]
                if any(r.get("train_seconds") is not None for r in rs) else None,
                "max_top1_share_mean": summarize([r.get("max_top1_share") or 0.0 for r in rs])[0],
                "collapse_epochs_mean": summarize([r.get("collapse_epochs") or 0 for r in rs])[0],
                "temperature_mean": summarize([r["temperature"] for r in rs
                                               if r.get("temperature") is not None])[0]
                if any(r.get("temperature") is not None for r in rs) else None,
                "w_mean_calib_mean": summarize([r["w_mean_calib"] for r in rs
                                                if r.get("w_mean_calib") is not None])[0]
                if any(r.get("w_mean_calib") is not None for r in rs) else None,
                "w_p90_calib_mean": summarize([r["w_p90_calib"] for r in rs
                                               if r.get("w_p90_calib") is not None])[0]
                if any(r.get("w_p90_calib") is not None for r in rs) else None,
            }
            m["verdict"] = verdict_for(m, shift)
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
                    "mean_entropy_nats": r.get("mean_entropy_nats"),
                    "max_top1_share": r.get("max_top1_share"),
                    "collapse_epochs": r.get("collapse_epochs"),
                    "entropy_weighting": r.get("entropy_weighting"),
                    "conditional_input": r.get("conditional_input"),
                    "temperature": r.get("temperature"),
                    "w_mean_calib": r.get("w_mean_calib"),
                    "w_p90_calib": r.get("w_p90_calib"),
                })

    paired = {}
    def val(method, shift, seed):
        for r in data.get((method, shift), []):
            if r["seed"] == seed:
                return r
        return None
    def delta(a, b, key, shift):
        out = []
        for seed in (42, 43, 44):
            ra, rb = val(a, shift, seed), val(b, shift, seed)
            if not ra or not rb:
                continue
            if key == "ntge":
                va = CENSOR if ra["ntge_first"] is None else ra["ntge_first"]
                vb = CENSOR if rb["ntge_first"] is None else rb["ntge_first"]
            else:
                va, vb = ra[key], rb[key]
            out.append(va - vb)
        return out
    for shift in SHIFTS:
        for base in ["source_only", "dann", "cdan", "cdpa_mmd", "cdan_sca_marg", "cdan_sca_full"]:
            d = delta("cdan_sca_calib", base, "ntge", shift)
            paired[f"calib_vs_{base}@{shift}"] = {
                "delta_ntge_values": d,
                "delta_ntge_mean": summarize(d)[0] if d else None,
                "wins": sum(1 for x in d if x < 0), "ties": sum(1 for x in d if x == 0),
                "losses": sum(1 for x in d if x > 0),
            }
        for base in ["cdan", "cdpa_mmd"]:
            d = delta("cdan_sca_calib", base, "ge_at_10000", shift)
            paired[f"calib_vs_{base}_ge10k@{shift}"] = {
                "delta_values": d, "delta_mean": summarize(d)[0] if d else None}
        d = delta("cdan_sca_full", "cdpa_mmd", "ntge", shift)
        paired[f"full_vs_cdpa_mmd@{shift}"] = {
            "delta_ntge_values": d,
            "delta_ntge_mean": summarize(d)[0] if d else None,
            "wins": sum(1 for x in d if x < 0), "ties": sum(1 for x in d if x == 0),
            "losses": sum(1 for x in d if x > 0),
        }

    # 权重函数对照：CDAN（无权重） / full（裸熵权重） / calib（温度校准权重）
    weight_variants = {}
    for shift in SHIFTS:
        weight_variants[shift] = {}
        for m in ("cdan", "cdan_sca_full", "cdan_sca_calib"):
            if m not in metrics or shift not in metrics[m]:
                continue
            v = metrics[m][shift]
            weight_variants[shift][m] = {k: v[k] for k in
                                         ("ntge_first_mean", "ntge_first_std", "ntge_first_reached",
                                          "ntge_stable_reached", "ge_at_10000_mean", "ge_at_10000_std",
                                          "sr_at_10000_mean", "n_total_at_ge1_mean", "attack_accuracy_mean",
                                          "mean_entropy_mean", "max_top1_share_mean", "collapse_epochs_mean")}
            weight_variants[shift][m]["temperature_mean"] = v.get("temperature_mean")
            weight_variants[shift][m]["w_mean_calib_mean"] = v.get("w_mean_calib_mean")
            weight_variants[shift][m]["w_p90_calib_mean"] = v.get("w_p90_calib_mean")

    out = {"metrics": metrics, "paired": paired, "three_way": weight_variants,
           "censor": CENSOR, "n_adapt": 45000, "thresholds": THRESH}
    (RES / "summary.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    fields = ["method", "label", "shift", "seed", "run", "ntge_first", "ntge_stable",
              "ge_at_1000", "ge_at_10000", "sr_at_1000", "sr_at_10000",
              "attack_accuracy", "n_attack_at_ge1", "n_total_at_ge1", "train_seconds",
              "mean_entropy_nats", "max_top1_share", "collapse_epochs",
              "entropy_weighting", "conditional_input", "temperature", "w_mean_calib",
              "w_p90_calib"]
    with (RES / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in per_run:
            w.writerow({k: r.get(k) for k in fields})

    tfields = ["method", "label", "shift", "n_runs", "ntge_first_mean", "ntge_first_std",
               "ntge_first_reached", "ntge_stable_mean", "ntge_stable_std",
               "ntge_stable_reached", "ge_at_1000_mean", "ge_at_10000_mean", "ge_at_10000_std",
               "sr_at_10000_mean", "attack_accuracy_mean", "n_adapt",
               "n_total_at_ge1_mean", "train_seconds_mean", "max_top1_share_mean",
               "collapse_epochs_mean", "temperature_mean", "w_mean_calib_mean",
               "w_p90_calib_mean", "verdict"]
    with (RES / "table_methods.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=tfields)
        w.writeheader()
        for shift in SHIFTS:
            for method in METHODS:
                m = metrics.get(method, {}).get(shift)
                if not m:
                    continue
                w.writerow({k: m.get(k) for k in tfields})

    with (RES / "table_weight_variants.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["shift", "variant", "ntge_mean", "ntge_std", "ntge_reached",
                    "stable_reached", "ge10k_mean", "ge10k_std", "sr10k",
                    "n_total", "acc", "entropy", "max_top1", "collapse_epochs",
                    "temperature_mean", "w_mean_calib", "w_p90_calib"])
        for shift, d in weight_variants.items():
            for m, v in d.items():
                w.writerow([shift, LABELS[m], v["ntge_first_mean"], v["ntge_first_std"],
                            v["ntge_first_reached"], v["ntge_stable_reached"],
                            v["ge_at_10000_mean"], v["ge_at_10000_std"], v["sr_at_10000_mean"],
                            v["n_total_at_ge1_mean"], v["attack_accuracy_mean"],
                            v["mean_entropy_mean"], v["max_top1_share_mean"],
                            v["collapse_epochs_mean"], v.get("temperature_mean"),
                            v.get("w_mean_calib_mean"), v.get("w_p90_calib_mean")])

    for shift in SHIFTS:
        print(f"\n=== {shift} ===")
        for method in METHODS:
            m = metrics.get(method, {}).get(shift)
            if not m:
                continue
            print(f"{m['label']:20s} NTGE {m['ntge_first_mean']:8.1f}±{m['ntge_first_std']:6.1f} "
                  f"({m['ntge_first_reached']}/{m['n_runs']})  GE@10k {m['ge_at_10000_mean']:6.1f}±"
                  f"{m['ge_at_10000_std']:5.1f}  SR {m['sr_at_10000_mean']:.2f}  "
                  f"N_total {m['n_total_at_ge1_mean']:7.0f}  top1max {m['max_top1_share_mean']:.3f}  "
                  f"verdict={m['verdict']}")
    print("\nweight_variants:", json.dumps(weight_variants, ensure_ascii=False)[:1500])
    print("\npaired:", json.dumps({k: paired[k] for k in list(paired)[:8]}, ensure_ascii=False)[:1500])


def verdict_for(m, shift):
    """硬性判据。\n\n    d50: NTGE <= 5700 且 3/3 seed 稳定 GE=0。\n    d100: GE@10k <= 4.0 且 NTGE <= 3700；若 NTGE 恰为 3700（打平）且\n          N_total 高于对比基线（CDPA 48700）则记 tie，不算胜出。\n    """
    if m["method"] == "source_only":
        return "baseline"
    if shift == "desync50":
        if m["ntge_first_mean"] <= THRESH["desync50"]["ntge"] and \
                m["ntge_stable_reached"] == m["n_runs"]:
            return "WIN" if m["ntge_first_mean"] < THRESH["desync50"]["ntge"] else "WIN(tie-ntge)"
        return "not-win"
    if m["ge_at_10000_mean"] > THRESH["desync100"]["ge10k"]:
        return "not-win"
    if m["ntge_first_mean"] < THRESH["desync100"]["ntge"]:
        return "WIN"
    if m["ntge_first_mean"] == THRESH["desync100"]["ntge"]:
        return "TIE" if m["n_total_at_ge1_mean"] <= 48700.0 else "not-win(n_total)"
    return "not-win"


if __name__ == "__main__":
    main()
