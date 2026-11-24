#!/usr/bin/env python3
"""adv2 与 CDPA-MMD / CDAN-SCA full / Source Only 的同口径对比（3 seed 均值±std）。"""
from __future__ import annotations
import csv, json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parents[1]
EVAL = HERE / "results"
CENSOR = 10000.0
SHIFTS = ["desync50", "desync100"]
KEEP = ["source_only", "cdpa_mmd", "cdan_sca_full", "cdan_sca_adv2"]
LABEL = {"source_only": "Source Only", "cdpa_mmd": "CDPA-MMD",
         "cdan_sca_full": "CDAN-SCA full", "cdan_sca_adv2": "CDAN-SCA adv2"}
N_ADAPT = {"source_only": 0, "cdpa_mmd": 45000, "cdan_sca_full": 45000, "cdan_sca_adv2": 45000}
THRESH = {"desync50": {"ntge": 5700.0}, "desync100": {"ntge": 3700.0, "ge10k": 4.0}}


def ms(vals):
    a = np.asarray(vals, dtype=float)
    return float(a.mean()), (float(a.std(ddof=1)) if len(a) > 1 else 0.0)


def load_rows():
    base = json.loads((EVAL / "eval" / "eval_all.json").read_text(encoding="utf-8"))
    rows = [r for r in base if r["method"] in KEEP]
    adv = json.loads((EVAL / "eval_adv2" / "eval_all_adv2.json").read_text(encoding="utf-8"))
    rows += adv
    data = {}
    for r in rows:
        data.setdefault((r["method"], r["shift"]), []).append(r)
    for k in data:
        data[k].sort(key=lambda r: r["seed"])
    return data


def verdict(m, shift):
    t = THRESH[shift]
    ok = m["n_runs"] == 3 and m["ntge_stable_reached"] == 3 and m["ntge_first_mean"] <= t["ntge"]
    if "ge10k" in t:
        ok = ok and m["ge_at_10000_mean"] <= t["ge10k"]
    return "WIN" if ok else "not-win"


def main():
    data = load_rows()
    summary = {"metrics": {}, "paired": {}}
    table = []
    for method in KEEP:
        summary["metrics"][method] = {}
        for shift in SHIFTS:
            rs = data.get((method, shift), [])
            if not rs:
                continue
            n_adapt = N_ADAPT[method]
            ntge = [CENSOR if r["ntge_first"] is None else float(r["ntge_first"]) for r in rs]
            st = [CENSOR if r["ntge_stable"] is None else float(r["ntge_stable"]) for r in rs]
            m = {
                "method": method, "label": LABEL[method], "shift": shift, "n_runs": len(rs),
                "ntge_first_mean": ms(ntge)[0], "ntge_first_std": ms(ntge)[1],
                "ntge_first_reached": int(sum(r["ntge_first"] is not None for r in rs)),
                "ntge_stable_mean": ms(st)[0], "ntge_stable_std": ms(st)[1],
                "ntge_stable_reached": int(sum(r["ntge_stable"] is not None for r in rs)),
                "ge_at_1000_mean": ms([r["ge_at_1000"] for r in rs])[0],
                "ge_at_10000_mean": ms([r["ge_at_10000"] for r in rs])[0],
                "ge_at_10000_std": ms([r["ge_at_10000"] for r in rs])[1],
                "sr_at_1000_mean": ms([r["sr_at_1000"] for r in rs])[0],
                "sr_at_10000_mean": ms([r["sr_at_10000"] for r in rs])[0],
                "attack_accuracy_mean": ms([r["attack_accuracy"] for r in rs])[0],
                "mean_entropy_mean": ms([r["mean_entropy_nats"] for r in rs])[0],
                "n_adapt": n_adapt,
                "n_attack_at_ge1_mean": ms(ntge)[0],
                "n_total_at_ge1_mean": ms([v + n_adapt for v in ntge])[0],
                "n_total_at_ge1_std": ms([v + n_adapt for v in ntge])[1],
                "max_top1_share_mean": ms([r.get("max_top1_share") or 0.0 for r in rs])[0],
                "collapse_epochs_mean": ms([r.get("collapse_epochs") or 0 for r in rs])[0],
                "train_seconds_mean": ms([r["train_seconds"] for r in rs if r.get("train_seconds")])[0]
                if any(r.get("train_seconds") for r in rs) else None,
            }
            m["verdict"] = verdict(m, shift)
            summary["metrics"][method][shift] = m
            table.append(m)

    for shift in SHIFTS:
        summary["paired"][shift] = {}
        for other in ["cdpa_mmd", "cdan_sca_full"]:
            deltas = []
            for r in data.get(("cdan_sca_adv2", shift), []):
                o = next((x for x in data.get((other, shift), []) if x["seed"] == r["seed"]), None)
                if not o:
                    continue
                a = CENSOR if r["ntge_first"] is None else float(r["ntge_first"])
                b = CENSOR if o["ntge_first"] is None else float(o["ntge_first"])
                deltas.append({"seed": r["seed"], "adv2": a, "other": b, "delta": a - b})
            if deltas:
                summary["paired"][shift][other] = {
                    "deltas": deltas,
                    "mean_delta": float(np.mean([d["delta"] for d in deltas])),
                    "wins": int(sum(d["delta"] < 0 for d in deltas)),
                    "ties": int(sum(d["delta"] == 0 for d in deltas)),
                    "losses": int(sum(d["delta"] > 0 for d in deltas)),
                }

    (EVAL / "summary_adv2.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    cols = ["shift", "method", "label", "n_runs", "ntge_first_mean", "ntge_first_std",
            "ntge_first_reached", "ntge_stable_mean", "ntge_stable_reached",
            "ge_at_1000_mean", "ge_at_10000_mean", "ge_at_10000_std",
            "sr_at_1000_mean", "sr_at_10000_mean", "n_total_at_ge1_mean",
            "max_top1_share_mean", "collapse_epochs_mean", "attack_accuracy_mean",
            "mean_entropy_mean", "verdict"]
    with (EVAL / "table_adv2.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader(); w.writerows(table)

    lines = ["# CDAN-SCA adv2 vs 基线（3 seed 均值±std，NTGE 截尾 10000）", ""]
    for shift in SHIFTS:
        lines += [f"## {shift}", "",
                  "| 方法 | n | NTGE(first) | NTGE(stable) | 稳定率 | GE@1k | GE@10k | SR@1k | N_total | max top1 | collapse ep | 判定 |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for method in KEEP:
            m = summary["metrics"].get(method, {}).get(shift)
            if not m:
                continue
            lines.append("| {label} | {n} | {a:.0f}±{b:.0f} | {c:.0f} | {d}/{n} | {e:.2f} | {f:.2f}±{g:.2f} | {h:.2f} | {i:.0f} | {j:.3f} | {k:.1f} | {v} |".format(
                label=m["label"], n=m["n_runs"], a=m["ntge_first_mean"], b=m["ntge_first_std"],
                c=m["ntge_stable_mean"], d=m["ntge_stable_reached"], e=m["ge_at_1000_mean"],
                f=m["ge_at_10000_mean"], g=m["ge_at_10000_std"], h=m["sr_at_1000_mean"],
                i=m["n_total_at_ge1_mean"], j=m["max_top1_share_mean"],
                k=m["collapse_epochs_mean"], v=m["verdict"]))
        lines.append("")
    (EVAL / "tables_adv2.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
