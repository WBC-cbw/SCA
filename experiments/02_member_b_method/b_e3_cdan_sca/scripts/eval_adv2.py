#!/usr/bin/env python3
"""评估 CDAN-SCA adv2 稳定化改造的 6 个 run，字段与 evaluate_b_e3.py 完全一致。"""
from __future__ import annotations
import csv, json
from pathlib import Path
import numpy as np, h5py

HERE = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(HERE / "scripts"))
import evaluate_b_e3 as E

SHIFTS = E.SHIFTS
SEEDS = [42, 43, 44]
METHOD = "cdan_sca_adv2"
OUT = HERE / "results" / "eval_adv2"


def run_stats(run_dir):
    h = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))
    top1 = [m["top1_share"] for m in h.get("monitor", [])]
    flags = h.get("collapse_flags", [])
    return {
        "max_top1_share": max(top1) if top1 else None,
        "collapse_epochs": int(sum(1 for f in flags if f == "COLLAPSE")),
        "train_seconds": round(float(sum(h["epoch_seconds"])), 1),
        "epochs": len(h["epochs"]),
    }


def main():
    EA = E.load_eval_module()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for shift, dataset in SHIFTS.items():
        with h5py.File(dataset, "r") as h:
            meta = np.asarray(h["Attack_traces/metadata"])
        pt = np.asarray(meta["plaintext"]); key = np.asarray(meta["key"])
        key_byte = int(key[0, E.BYTE])
        labels = E.SBOX[np.bitwise_xor(pt[:, E.BYTE], np.uint8(key_byte))].astype(np.int64)
        (OUT / shift).mkdir(parents=True, exist_ok=True)
        for seed in SEEDS:
            run = f"cdan_sca_adv2_{shift}_s{seed}"
            run_dir = HERE / "results" / run
            pred = run_dir / "predictions_attack.npy"
            if not pred.exists():
                print(f"[skip] {run}")
                continue
            probs = np.load(pred)
            curve = EA.ranks(probs, pt, key, E.BYTE, 100)
            rec = E.summary_from_curve(curve, adapted=True)
            with (OUT / shift / f"{run}_ge.csv").open("w", newline="") as f:
                w = csv.writer(f); w.writerow(["traces", "ge"]); w.writerows(curve)
            p = np.clip(probs, 1e-12, 1.0)
            rec.update({
                "method": METHOD, "run": run, "shift": shift, "seed": seed,
                "dataset": str(dataset), "predictions": str(pred),
                "n_attack_traces": int(len(probs)), "step": 100,
                "target_byte": E.BYTE, "real_key_byte": key_byte,
                "attack_accuracy": round(float((probs.argmax(1) == labels).mean()), 6),
                "mean_max_softmax": round(float(probs.max(1).mean()), 6),
                "mean_entropy_nats": round(float(-(p * np.log(p)).sum(1).mean()), 6),
                "ge_curve": [{"traces": int(t), "ge": int(g)} for t, g in curve],
                "entropy_weighting": "global (1 - H/ln256)",
                "conditional_input": "consistency-gated + detach cond.",
            })
            rec.update(run_stats(run_dir))
            rows.append(rec)
            print(f"[ok] {run}: ntge={rec['ntge_first']} ge10k={rec['ge_at_10000']} "
                  f"top1max={rec['max_top1_share']:.3f} collapse={rec['collapse_epochs']}")
    rows.sort(key=lambda r: (r["shift"], r["seed"]))
    (OUT / "eval_all_adv2.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"[done] {len(rows)} runs -> {OUT}")


if __name__ == "__main__":
    main()
