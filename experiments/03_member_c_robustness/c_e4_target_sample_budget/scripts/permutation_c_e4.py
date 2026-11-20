#!/usr/bin/env python3
"""C-E4 NTGE 轨迹顺序稳健性检验：对每个新训练且达到 GE=0 的 run 做 10 次轨迹置换。

置换只改变 attack 轨迹顺序，不改变模型与预测；NTGE 定义为首个 GE=0（step=100）。
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
N_PERM = 10


def load_eval_module():
    spec = importlib.util.spec_from_file_location("eval_ascad", str(EVAL_SRC))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    EA = load_eval_module()
    ev = json.loads((HERE / "results/eval/eval_all.json").read_text(encoding="utf-8"))
    out = {"n_perm": N_PERM, "step": 100, "note": "只对 C-E4 新训练 run 做置换；45000 全量档复用已有结果不重测", "runs": []}
    for r in ev:
        if r["reuse"] != "C-E4":
            continue
        shift, method, budget, seed = r["shift"], r["method"], r["budget"], r["seed"]
        if r["ntge_first"] is None:
            out["runs"].append({"run": r["run"], "shift": shift, "method": method,
                                "budget": budget, "seed": seed, "ntge_original": None,
                                "note": "no GE=0 in 10k, permutation skipped"})
            continue
        # 预测文件路径以 evaluate 记录的绝对路径为准（cdpa 的目录名是 cdpa_<shift>...，
        # 与 evaluate 标签 cdpa_mmd_... 不同，按 run 名拼路径会漏掉全部 CDPA run）。
        cands = []
        if r.get("predictions"):
            cands.append(Path(r["predictions"]))
        cands.append(HERE / "results" / r["run"] / "predictions_attack.npy")
        p = next((c for c in cands if c.exists()), None)
        if p is None:
            out["runs"].append({"run": r["run"], "shift": shift, "method": method,
                                "budget": budget, "seed": seed,
                                "ntge_original": r["ntge_first"],
                                "note": "predictions not found, permutation skipped"})
            print(f"[skip] {r['run']}: predictions not found")
            continue
        probs = np.load(p)
        with h5py.File(SHIFTS[shift], "r") as h:
            meta = np.asarray(h["Attack_traces/metadata"])
        pt = np.asarray(meta["plaintext"]); key = np.asarray(meta["key"])
        ntges = []
        rng = np.random.default_rng(seed + budget * 31)
        for _ in range(N_PERM):
            perm = rng.permutation(len(probs))
            c = EA.ranks(probs[perm], pt[perm], key, 2, 100)
            z = [n for n, g in c if g == 0]
            ntges.append(z[0] if z else 10000)
        rec = {"run": r["run"], "shift": shift, "method": method, "budget": budget,
               "seed": seed, "ntge_original": r["ntge_first"], "perm_ntge": ntges,
               "perm_mean": float(np.mean(ntges)), "perm_std": float(np.std(ntges, ddof=1)),
               "perm_min": int(np.min(ntges)), "perm_max": int(np.max(ntges))}
        out["runs"].append(rec)
        print(f"{r['run']}: original {r['ntge_first']}  perm {np.mean(ntges):.0f}"
              f"±{np.std(ntges, ddof=1):.0f} [{min(ntges)}, {max(ntges)}]", flush=True)
    (HERE / "results/permutation.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote results/permutation.json")


if __name__ == "__main__":
    main()
