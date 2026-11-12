#!/usr/bin/env python3
"""B-E3 NTGE 轨迹顺序稳健性检验（参照 A-E2 ntge_permutation.py）。

对每个达到 GE=0 的 full / marginal run，随机置换 attack 轨迹顺序 10 次，
重算 NTGE（首次 GE=0，step=100），报告 mean ± std 与原始固定顺序值。
置换只改变轨迹顺序，不改变模型与预测。

输出 results/permutation.json。
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[2]
EVAL_SRC = ROOT / "experiments/cdan-sca-diagnosis/scripts/evaluate_ascad.py"
DATA = Path("D:/sca_project/ASCAD_data")
SHIFTS = {"desync50": DATA / "ASCAD_desync50.h5", "desync100": DATA / "ASCAD_desync100.h5"}
SEEDS = [42, 43, 44]
N_PERM = 10
OUT = HERE / "results/permutation.json"


def load_eval_module():
    spec = importlib.util.spec_from_file_location("eval_ascad", str(EVAL_SRC))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    EA = load_eval_module()
    out = {"n_perm": N_PERM, "step": 100, "runs": []}
    for shift, path in SHIFTS.items():
        with h5py.File(path, "r") as h:
            meta = np.asarray(h["Attack_traces/metadata"])
        pt = np.asarray(meta["plaintext"]); key = np.asarray(meta["key"])
        for tag, method in [("full", "cdan_sca_full"), ("marg", "cdan_sca_marg")]:
            for seed in SEEDS:
                run = f"cdan_sca_{tag}_{shift}_s{seed}"
                p = HERE / "results" / run / "predictions_attack.npy"
                if not p.exists():
                    continue
                probs = np.load(p)
                base_curve = EA.ranks(probs, pt, key, 2, 100)
                zeros = [n for n, r in base_curve if r == 0]
                if not zeros:
                    out["runs"].append({"run": run, "shift": shift, "method": method,
                                        "seed": seed, "ntge_original": None,
                                        "note": "no GE=0 in 10k, permutation skipped"})
                    continue
                ntges, curves = [], []
                rng = np.random.default_rng(seed)
                for i in range(N_PERM):
                    perm = rng.permutation(len(probs))
                    c = EA.ranks(probs[perm], pt[perm], key, 2, 100)
                    z = [n for n, r in c if r == 0]
                    ntges.append(z[0] if z else 10000)
                    curves.append([z[0] if z else None])
                rec = {"run": run, "shift": shift, "method": method, "seed": seed,
                       "ntge_original": zeros[0],
                       "perm_ntge": ntges,
                       "perm_mean": float(np.mean(ntges)), "perm_std": float(np.std(ntges, ddof=1)),
                       "perm_min": int(np.min(ntges)), "perm_max": int(np.max(ntges))}
                out["runs"].append(rec)
                print(f"{run}: original {zeros[0]}  perm {np.mean(ntges):.0f}±{np.std(ntges, ddof=1):.0f} "
                      f"[{min(ntges)}, {max(ntges)}]", flush=True)
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
