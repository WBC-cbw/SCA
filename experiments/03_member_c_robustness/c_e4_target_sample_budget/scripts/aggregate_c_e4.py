#!/usr/bin/env python3
"""C-E4 聚合：method x budget 的 NTGE / 稳定率 / 总成本，以及少样本生态位判定。

口径：
- ntge_first / ntge_stable 未达到记 None；均值统计时截尾 10000。
- n_total_at_ge1 = ntge_first + budget（未达到截尾 10000 + budget）。
- stable_rate = 稳定 GE=0（曲线末尾连续零段）的 seed 数 / 3。
- 主判据是总成本 N_total：同 seed 配对比较，负值表示方法 A 更省。
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parents[1]
SEEDS = [42, 43, 44]
BUDGETS = [50, 500, 5000, 45000]
CENSOR = 10000
B_E3 = ROOT / "experiments/02_member_b_method/b_e3_cdan_sca"
A_E2 = ROOT / "experiments/01_member_a_baselines/a_e2_uda_baselines"
A_E3 = ROOT / "experiments/01_member_a_baselines/a_e3_cdpa_or_adabn"
FULL_BUDGET_DIRS = {"cdpa_mmd": A_E3, "dann": A_E2, "cdan_sca": B_E3}

sys.path.insert(0, str(B_E3 / "scripts"))
from cdan_sca_common import calibration_stats  # noqa: E402

# 结果目录命名：训练时 cdpa 的前缀是 "cdpa"，评估里的 method 名是 "cdpa_mmd"
RUN_DIR_ALIAS = {"cdpa_mmd": "cdpa"}


def run_dir_name(run, method):
    alias = RUN_DIR_ALIAS.get(method)
    if alias and run.startswith(method + "_"):
        return alias + run[len(method):]
    return run


def adapt_npz_diag(shift, method, seed):
    """全量档 run 没有 per-run entropy_diag.json 时，用 B-E3 adapt_diag 的 probs/labels 补校准指标。"""
    name = {"cdpa_mmd": "cdpa_mmd", "dann": "dann", "cdan_sca": "cdan", "source_only": "source_only"}.get(method)
    if not name:
        return {}
    f = B_E3 / "results/adapt_diag" / (name + "_" + shift + "_s" + str(seed) + ".npz")
    if not f.exists():
        return {}
    try:
        z = np.load(f)
        probs = np.asarray(z["probs"], dtype=float)
        labels = np.asarray(z["labels"]).astype(int)
        st = calibration_stats(probs, labels)
        pp = np.clip(probs, 1e-12, 1.0)
        ent = -(pp * np.log(pp)).sum(axis=1)
        w = 1.0 - ent / np.log(256.0)
        return {"diag_n": int(len(labels)), "diag_acc": st["accuracy"],
                "adapt_w_mean": float(w.mean()),
                "adapt_w_p10": float(np.percentile(w, 10)),
                "adapt_w_p90": float(np.percentile(w, 90)),
                "adapt_entropy_mean": float(ent.mean()),
                "adapt_ece": st["ece_15"],
                "adapt_confwrong09": st["frac_confwrong_ge_0.9"],
                "adapt_top1_share": st["top1_share"],
                "diag_source": "B-E3 adapt_diag npz"}
    except Exception:
        return {}


def read_json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def mean_std(vals):
    v = [x for x in vals if x is not None]
    if not v:
        return None, None
    return float(np.mean(v)), float(np.std(v, ddof=0))


def censor(v, cap):
    return cap if v is None else float(v)


def cell_stats(rows):
    first = [r["ntge_first"] for r in rows]
    stable = [r["ntge_stable"] for r in rows]
    tot_first = [r["n_total_at_ge1"] for r in rows]
    tot_stable = [r["n_total_at_stable_ge0"] for r in rows]
    budget = rows[0]["budget"]
    m1, s1 = mean_std([censor(x, CENSOR) for x in first])
    ms, ss = mean_std([censor(x, CENSOR) for x in stable])
    mt, st = mean_std([censor(x, CENSOR + budget) for x in tot_first])
    mts, sts = mean_std([censor(x, CENSOR + budget) for x in tot_stable])
    def mn(key):
        return float(np.mean([r[key] for r in rows]))
    return {
        "method": rows[0]["method"], "budget": budget, "n_seeds": len(rows),
        "ntge_first_mean": m1, "ntge_first_sd": s1,
        "ntge_stable_mean": ms, "ntge_stable_sd": ss,
        "ntge_first_reached": int(sum(x is not None for x in first)),
        "ntge_stable_reached": int(sum(x is not None for x in stable)),
        "stable_rate": float(sum(x is not None for x in stable) / len(rows)),
        "n_total_ge1_mean": mt, "n_total_ge1_sd": st,
        "n_total_stable_mean": mts, "n_total_stable_sd": sts,
        "ge_at_1000_mean": mn("ge_at_1000"), "ge_at_2000_mean": mn("ge_at_2000"),
        "ge_at_10000_mean": mn("ge_at_10000"),
        "attack_accuracy_mean": mn("attack_accuracy"),
        "attack_ece_mean": mn("attack_ece_15"),
        "attack_confwrong09_mean": mn("attack_frac_confwrong_ge09"),
        "attack_top1_share_mean": mn("attack_top1_share"),
        "sr_at_1000_mean": mn("sr_at_1000"), "sr_at_10000_mean": mn("sr_at_10000"),
        "train_seconds_mean": mn("train_seconds") if rows[0].get("train_seconds") else None,
        "max_top1_share_mean": mn("max_top1_share") if rows[0].get("max_top1_share") else None,
        "collapse_epochs_mean": mn("collapse_epochs") if rows[0].get("collapse_epochs") is not None else None,
    }


def load_be3_adapt_diag():
    """B-E3 的 entropy_weight_analysis.csv：全量档的 adapt 侧诊断（前 5000 条 target_adapt，3 seed 均值）。
    全量档 CDPA/DANN 在 A-E3/A-E2 没有逐 epoch monitor、也没有 diag npz，这里用它补齐校准列。"""
    path = B_E3 / "results/entropy_weight_analysis.csv"
    acc = {}
    if not path.exists():
        return acc
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["side"] != "adapt" or row["method"] == "cdan_sca_calib":
                continue
            key = (row["shift"], row["method"])
            acc.setdefault(key, []).append(row)
    out = {}
    for key, rows in acc.items():
        def avg(name):
            vals = [float(r[name]) for r in rows if r.get(name) not in (None, "")]
            return float(np.mean(vals)) if vals else None
        out[key] = {"diag_n": int(avg("n") or 0), "diag_acc": avg("accuracy"),
                    "adapt_ece": avg("ece_15"),
                    "adapt_confwrong09": avg("frac_confwrong_ge_0.9"),
                    "adapt_top1_share": avg("top1_share"),
                    "adapt_w_mean": avg("w_mean"), "adapt_w_p90": avg("w_p90"),
                    "adapt_entropy_mean": avg("mean_entropy_nats"),
                    "diag_source": "B-E3 entropy_weight_analysis.csv (adapt first 5k, 3-seed mean)"}
    return out


BE3_ADAPT = {}


def npz_w_stats(run_dir):
    """从 diag_target_adapt.npz 取 w=1-H/log256 与熵（诊断用；非 CDAN 方法不把 w 用于训练）。"""
    p = Path(run_dir) / "diag_target_adapt.npz"
    if not p.exists():
        return {}
    try:
        z = np.load(p)
        out = {}
        if "w" in z.files:
            w = np.asarray(z["w"], dtype=float)
            out.update({"adapt_w_mean": float(w.mean()),
                        "adapt_w_p10": float(np.percentile(w, 10)),
                        "adapt_w_p90": float(np.percentile(w, 90))})
        if "entropy" in z.files:
            out["adapt_entropy_mean"] = float(np.asarray(z["entropy"], dtype=float).mean())
        return out
    except Exception:
        return {}


def history_diag(r):
    if r["budget"] == 45000 and r["method"] in FULL_BUDGET_DIRS:
        d = FULL_BUDGET_DIRS[r["method"]]
    else:
        d = HERE
    run_dir = d / "results" / run_dir_name(r["run"], r["method"])
    hist = read_json(run_dir / "history.json")
    ediag = read_json(run_dir / "entropy_diag.json")
    out = {"shift": r["shift"], "method": r["method"], "budget": r["budget"],
           "run": r["run"], "seed": r["seed"]}
    if hist and hist.get("monitor"):
        top1 = [m["top1_share"] for m in hist["monitor"]]
        ent = [m["pred_entropy_mean"] for m in hist["monitor"]]
        flags = hist.get("collapse_flags") or []
        first_warn = next((e for e, f in zip(hist["epochs"], flags) if f in ("WARN", "COLLAPSE")), None)
        first_collapse = next((e for e, f in zip(hist["epochs"], flags) if f == "COLLAPSE"), None)
        out.update({"epochs": len(top1), "max_top1": float(max(top1)),
                    "final_top1": float(top1[-1]), "min_entropy": float(min(ent)),
                    "final_entropy": float(ent[-1]),
                    "first_warn_epoch": first_warn, "first_collapse_epoch": first_collapse,
                    "collapse_epochs": int(sum(f == "COLLAPSE" for f in flags)),
                    "warn_epochs": int(sum(f == "WARN" for f in flags))})
    if ediag:
        out.update({"diag_n": ediag.get("n"), "diag_acc": ediag.get("accuracy"),
                    "adapt_ece": ediag.get("ece_15"),
                    "adapt_confwrong09": ediag.get("frac_confwrong_ge_0.9"),
                    "adapt_top1_share": ediag.get("top1_share"),
                    "diag_source": "run"})
    if r["reuse"] == "C-E4":
        out.update(npz_w_stats(HERE / "results" / run_dir_name(r["run"], r["method"])))
    elif r["method"] in FULL_BUDGET_DIRS:
        base = FULL_BUDGET_DIRS[r["method"]]
        out.update(npz_w_stats(base / "results" / run_dir_name(r["run"], r["method"])))
    if "adapt_ece" not in out:
        out.update(adapt_npz_diag(r["shift"], r["method"], r["seed"]))
    if "adapt_ece" not in out:
        ref = BE3_ADAPT.get((r["shift"], r["method"]))
        if ref:
            out.update(ref)
    return out


def paired(rows, method_a, method_b, metric):
    idx = {(r["shift"], r["method"], r["budget"], r["seed"]): r for r in rows}
    out = {}
    for shift in sorted({r["shift"] for r in rows}):
        for b in (BUDGETS + [0] if (method_a == "source_only" or method_b == "source_only") else BUDGETS):
            ds = []
            for s in SEEDS:
                a = idx.get((shift, method_a, b, s))
                bb = idx.get((shift, method_b, b, s))
                if a is None or bb is None:
                    continue
                cap = CENSOR + (0 if metric.startswith("ntge") else b)
                va = cap if a[metric] is None else a[metric]
                vb = cap if bb[metric] is None else bb[metric]
                ds.append((s, va - vb))
            if len(ds) == len(SEEDS):
                d = [x[1] for x in ds]
                out[f"{shift}|b{b}"] = {
                    "n": len(ds), "mean_delta": float(np.mean(d)),
                    "wins_a": int(sum(x < 0 for x in d)), "ties": int(sum(x == 0 for x in d)),
                    "wins_b": int(sum(x > 0 for x in d)),
                    "per_seed": [{"seed": s, "delta": v} for s, v in ds],
                }
    return out


def main():
    ev_path = Path(os.environ.get("C_E4_EVAL", str(HERE / "results/eval/eval_all.json")))
    ev = json.loads(ev_path.read_text(encoding="utf-8"))
    out_dir = Path(os.environ.get("C_E4_OUT", str(HERE / "results")))
    out_dir.mkdir(parents=True, exist_ok=True)
    cells = {}
    skipped_incomplete = []
    for shift in sorted({r["shift"] for r in ev}):
        for method in ["source_only", "cdpa_mmd", "dann", "cdan_sca"]:
            for b in ([0] if method == "source_only" else BUDGETS):
                rows = [r for r in ev if r["shift"] == shift and r["method"] == method
                        and r["budget"] == b]
                if not rows:
                    continue
                if len(rows) != len(SEEDS):
                    skipped_incomplete.append(f"{shift}|{method}|b{b} ({len(rows)}/3)")
                    continue
                cells[f"{shift}|{method}|b{b}"] = cell_stats(rows)

    summary = {"experiment": "C-E4", "censor": CENSOR,
               "note": "NTGE 截尾 10000；N_total = NTGE + budget；stable = 曲线末尾连续 GE=0",
               "incomplete_cells_skipped": skipped_incomplete,
               "cells": cells}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # 宽表 1：method×budget NTGE（首次 / 稳定）与稳定率
    with (out_dir / "table_ntge.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["shift", "method", "budget", "n_seeds",
                    "ntge_first_mean", "ntge_first_sd", "ntge_stable_mean", "ntge_stable_sd",
                    "stable_rate", "first_reached", "stable_reached",
                    "ge_at_1000", "ge_at_2000", "ge_at_10000", "sr_at_10000", "attack_acc"])
        for shift in sorted({r["shift"] for r in ev}):
            for method in ["source_only", "cdpa_mmd", "dann", "cdan_sca"]:
                for b in ([0] if method == "source_only" else BUDGETS):
                    c = cells.get(f"{shift}|{method}|b{b}")
                    if not c:
                        continue
                    w.writerow([shift, method, b, c["n_seeds"],
                                round(c["ntge_first_mean"], 1), round(c["ntge_first_sd"], 1),
                                None if c["ntge_stable_mean"] is None else round(c["ntge_stable_mean"], 1),
                                None if c["ntge_stable_sd"] is None else round(c["ntge_stable_sd"], 1),
                                round(c["stable_rate"], 3), c["ntge_first_reached"], c["ntge_stable_reached"],
                                round(c["ge_at_1000_mean"], 1), round(c["ge_at_2000_mean"], 1),
                                round(c["ge_at_10000_mean"], 2), round(c["sr_at_10000_mean"], 3),
                                round(c["attack_accuracy_mean"], 5)])

    # 宽表 2：总成本 N_total（双口径）
    with (out_dir / "table_cost.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["shift", "method", "budget", "n_adapt_included",
                    "n_total_at_ge1_mean", "n_total_at_ge1_sd",
                    "n_total_stable_mean", "n_total_stable_sd", "stable_rate"])
        for shift in sorted({r["shift"] for r in ev}):
            for method in ["source_only", "cdpa_mmd", "dann", "cdan_sca"]:
                for b in ([0] if method == "source_only" else BUDGETS):
                    c = cells.get(f"{shift}|{method}|b{b}")
                    if not c:
                        continue
                    w.writerow([shift, method, b, int(b > 0),
                                round(c["n_total_ge1_mean"], 1), round(c["n_total_ge1_sd"], 1),
                                None if c["n_total_stable_mean"] is None else round(c["n_total_stable_mean"], 1),
                                None if c["n_total_stable_sd"] is None else round(c["n_total_stable_sd"], 1),
                                round(c["stable_rate"], 3)])

    # 逐 run 指标
    with (out_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(ev[0].keys()))
        w.writeheader()
        for r in ev:
            w.writerow({k: (json.dumps(v) if k == "ge_curve" else v) for k, v in r.items()})

    # 诊断（坍塌 / 校准）
    global BE3_ADAPT
    BE3_ADAPT = load_be3_adapt_diag()
    diags = [history_diag(r) for r in ev]
    (out_dir / "diagnostics.json").write_text(json.dumps(diags, indent=2), encoding="utf-8")
    with (out_dir / "diagnostics.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["shift", "method", "budget", "run", "seed", "epochs", "max_top1", "final_top1",
                  "min_entropy", "final_entropy", "first_warn_epoch", "first_collapse_epoch",
                  "collapse_epochs", "warn_epochs", "diag_n", "diag_acc", "adapt_ece",
                  "adapt_confwrong09", "adapt_top1_share",
                  "adapt_w_mean", "adapt_w_p10", "adapt_w_p90", "adapt_entropy_mean", "diag_source"]
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for d in diags:
            w.writerow(d)

    # 配对比较（同 seed 同预算）
    pairs = {}
    for a, b in [("cdpa_mmd", "dann"), ("cdpa_mmd", "cdan_sca"),
                 ("cdan_sca", "dann"), ("source_only", "cdan_sca")]:
        for metric in ["ntge_first", "n_total_at_ge1"]:
            pairs[f"{a}_vs_{b}|{metric}"] = paired(ev, a, b, metric)
    (out_dir / "paired.json").write_text(json.dumps(pairs, indent=2), encoding="utf-8")

    # 判定：每方法最小可恢复预算；少样本生态位
    verdicts = {}
    for shift in sorted({r["shift"] for r in ev}):
        v = {}
        for method in ["cdpa_mmd", "dann", "cdan_sca"]:
            seq = []
            for b in BUDGETS:
                c = cells.get(f"{shift}|{method}|b{b}")
                if c:
                    seq.append({"budget": b, "stable_rate": c["stable_rate"],
                                "ntge_first_mean": c["ntge_first_mean"],
                                "n_total_ge1_mean": c["n_total_ge1_mean"]})
            v[method] = seq
            first_any = next((x["budget"] for x in seq if x["stable_rate"] > 0), None)
            first_all = next((x["budget"] for x in seq if x["stable_rate"] >= 1.0), None)
            v[method + "_first_budget_any_stable"] = first_any
            v[method + "_first_budget_full_stability"] = first_all
        for b in [50, 500]:
            cs = cells.get(f"{shift}|cdan_sca|b{b}")
            cp = cells.get(f"{shift}|cdpa_mmd|b{b}")
            if cs and cp:
                v[f"niche_b{b}"] = {
                    "cdan_sca_stable_rate": cs["stable_rate"],
                    "cdan_sca_n_total": cs["n_total_ge1_mean"],
                    "cdpa_stable_rate": cp["stable_rate"],
                    "cdpa_n_total": cp["n_total_ge1_mean"],
                    "cdan_sca_3of3_stable_and_cheaper":
                        bool(cs["stable_rate"] >= 1.0 and cp["n_total_ge1_mean"] is not None
                             and cs["n_total_ge1_mean"] < cp["n_total_ge1_mean"]),
                }
        verdicts[shift] = v
    (out_dir / "verdicts.json").write_text(json.dumps(verdicts, indent=2), encoding="utf-8")

    print(json.dumps({"cells": len(cells), "rows": len(ev)}, indent=0))
    for shift in sorted(verdicts):
        print("==", shift)
        for method in ["cdpa_mmd", "dann", "cdan_sca"]:
            print(" ", method, verdicts[shift][method])


if __name__ == "__main__":
    main()
