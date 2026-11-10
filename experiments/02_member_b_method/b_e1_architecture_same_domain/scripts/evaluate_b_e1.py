#!/usr/bin/env python3
"""B-E1 评估：从 predictions_attack.npy 计算 GE 曲线 / NTGE / SR / GE@N。

GE 计算直接复用仓库标准实现
experiments/cdan-sca-diagnosis/scripts/evaluate_ascad.py::ranks，
保证与 E0 / 其它基线同一口径（step=100，target byte=2）。

输出:
  results/eval/<run>_ge.csv    每 100 条轨迹的 key rank
  results/eval/<run>.json      ntge / sr@1000 / sr@10000 / ge@10000
"""
import argparse
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parents[1]
EVAL_SRC = ROOT / "experiments/cdan-sca-diagnosis/scripts/evaluate_ascad.py"


def load_eval_module():
    spec = importlib.util.spec_from_file_location("eval_ascad", str(EVAL_SRC))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(HERE / "results"))
    ap.add_argument("--dataset", default="D:/sca_project/ASCAD_data/ASCAD.h5")
    ap.add_argument("--step", type=int, default=100)
    ap.add_argument("--byte", type=int, default=2)
    args = ap.parse_args()

    EA = load_eval_module()
    import h5py

    with h5py.File(args.dataset, "r") as h:
        meta = np.array(h["Attack_traces/metadata"])
        plaintext = np.asarray(meta["plaintext"])
        key = np.asarray(meta["key"])
    print("real key byte", args.byte, "=", int(key[0, args.byte]))

    res_root = Path(args.results)
    out = res_root / "eval"
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for run_dir in sorted(p for p in res_root.iterdir() if p.is_dir() and p.name != "eval"):
        pred = run_dir / "predictions_attack.npy"
        if not pred.exists():
            print(f"[skip] {run_dir.name}: no predictions")
            continue
        probs = np.load(pred)
        assert probs.shape[1] == 256, probs.shape
        curve = EA.ranks(probs, plaintext, key, args.byte, args.step)

        ntge = next((n for n, r in curve if r == 0), None)
        ge_at = {n: r for n, r in curve}

        def sr_at(n):
            return int(ge_at.get(n, 255) == 0)

        with (out / f"{run_dir.name}_ge.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["traces", "ge"])
            w.writerows(curve)

        rec = {
            "run": run_dir.name,
            "predictions": str(pred),
            "n_attack_traces": int(len(probs)),
            "step": args.step,
            "target_byte": args.byte,
            "ntge": ntge,
            "n_total_at_ge1": ntge,      # 无 adaptation 轨迹，总成本 = 攻击轨迹数
            "ge_at_1000": ge_at.get(1000),
            "ge_at_5000": ge_at.get(5000),
            "ge_at_10000": ge_at.get(10000),
            "sr_at_1000": sr_at(1000),
            "sr_at_10000": sr_at(10000),
            "ge_curve": [{"traces": n, "ge": r} for n, r in curve],
        }
        (out / f"{run_dir.name}.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
        rows.append(rec)
        print(f"{run_dir.name}: NTGE={ntge} GER@1000={rec['ge_at_1000']} "
              f"GER@10000={rec['ge_at_10000']} SR@1000={rec['sr_at_1000']}")

    (out / "eval_all.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"wrote {len(rows)} evaluations -> {out}")


if __name__ == "__main__":
    main()
