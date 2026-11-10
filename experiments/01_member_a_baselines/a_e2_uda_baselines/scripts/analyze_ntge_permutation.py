#!/usr/bin/env python3
"""NTGE 对攻击轨迹顺序的敏感性（诊断，不改写主指标）。

对每个 run 的 10000 条攻击轨迹做 10 次随机置换，重新计算 NTGE(first)；
主结果的 NTGE 是固定顺序（0..9999）下的值。目的：判断 NTGE 的 seed 间差异
（如 7500 vs 9000）是否大于轨迹顺序带来的抽样波动。

输出 results/ntge_permutation.csv 与 results/ntge_permutation.json。
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
import evaluate_uda as EU  # noqa: E402

EA = EU.load_eval_module()
N_PERM = 10
STEP = 100


def main():
    rows = json.loads((HERE / "results/eval/eval_all.json").read_text(encoding="utf-8"))
    meta_cache = {}
    for shift in ("desync50", "desync100"):
        with h5py.File(EU.SHIFTS[shift], "r") as h:
            m = np.asarray(h["Attack_traces/metadata"])
        meta_cache[shift] = (np.asarray(m["plaintext"]), np.asarray(m["key"]))

    out = []
    for r in rows:
        probs = np.load(r["predictions"])
        pt, key = meta_cache[r["shift"]]
        ntges, stables = [], []
        for k in range(N_PERM):
            rng = np.random.default_rng(1000 + k)
            idx = rng.permutation(len(probs))
            curve = EA.ranks(probs[idx], pt[idx], key, 2, step=STEP)
            zeros = [n for n, v in curve if v == 0]
            ntges.append(zeros[0] if zeros else None)
            if curve[-1][1] == 0:
                last_nonzero = max((n for n, v in curve if v != 0), default=0)
                stables.append(min(last_nonzero + STEP, len(probs)))
            else:
                stables.append(None)
        reached = [v for v in ntges if v is not None]
        stable_vals = [v for v in stables if v is not None]
        rec = {
            "shift": r["shift"], "method": r["method"], "seed": r["seed"],
            "ge_at_10000_fixed_order": r["ge_at_10000"],
            "ntge_fixed_order": r["ntge_first"],
            "ntge_perm_mean": round(float(np.mean(reached)), 1) if reached else None,
            "ntge_perm_std": round(float(np.std(reached)), 1) if reached else None,
            "ntge_perm_min": min(reached) if reached else None,
            "ntge_perm_max": max(reached) if reached else None,
            "ntge_stable_perm_mean": round(float(np.mean(stable_vals)), 1) if stable_vals else None,
            "ntge_stable_perm_std": round(float(np.std(stable_vals)), 1) if stable_vals else None,
            "ntge_stable_perm_min": min(stable_vals) if stable_vals else None,
            "ntge_stable_perm_max": max(stable_vals) if stable_vals else None,
            "n_perm_stable": len(stable_vals),
            "n_perm_reached": len(reached),
            "n_perm": N_PERM,
        }
        out.append(rec)
        print(f"{r['shift']:9s} {r['method']:12s} s{r['seed']} fixed={r['ntge_first']} "
              f"perm={rec['ntge_perm_mean']}±{rec['ntge_perm_std']} "
              f"stable={rec['ntge_stable_perm_mean']}±{rec['ntge_stable_perm_std']} "
              f"[{rec['ntge_perm_min']},{rec['ntge_perm_max']}] reached={len(reached)}/{N_PERM}")

    with (HERE / "results/ntge_permutation.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    (HERE / "results/ntge_permutation.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote results/ntge_permutation.csv ({len(out)} runs x {N_PERM} permutations)")


if __name__ == "__main__":
    main()
