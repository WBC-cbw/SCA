#!/usr/bin/env python3
"""评估 λ=0 / λ=1 pilot 对照 run（文件名带 tag），输出 results/eval_controls/。

与 evaluate_uda.py 同口径（ranks step=100, byte=2），并从 evaluate_uda 复用 summary_from_curve。
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "scripts"))
import evaluate_uda as EU  # noqa: E402

RUN_RE = re.compile(r"^(dann|cdan)_(desync\d+)_s(\d+)_(lam[0-9.eE+-]+)$")


def main():
    EA = EU.load_eval_module()
    labels = {}
    for shift in ("desync50", "desync100"):
        with h5py.File(EU.SHIFTS[shift], "r") as h:
            meta = np.asarray(h["Attack_traces/metadata"])
        labels[shift] = EU.SBOX[np.bitwise_xor(np.asarray(meta["plaintext"])[:, 2],
                                               np.uint8(int(meta["key"][0, 2])))].astype(np.int64)
    out_dir = HERE / "results/eval_controls"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for d in sorted((HERE / "results").iterdir()):
        if not d.is_dir():
            continue
        m = RUN_RE.match(d.name)
        if not m or not (d / "predictions_attack.npy").exists():
            continue
        method, shift, seed, tag = m.group(1), m.group(2), int(m.group(3)), m.group(4)
        probs = np.load(d / "predictions_attack.npy")
        y = labels[shift]
        with h5py.File(EU.SHIFTS[shift], "r") as h:
            meta = np.asarray(h["Attack_traces/metadata"])
        curve = EA.ranks(probs, np.asarray(meta["plaintext"]), np.asarray(meta["key"]), 2, step=100)
        rec = EU.summary_from_curve(curve, len(y), adapted=True)
        rec.update({
            "run": d.name, "method": method, "shift": shift, "seed": seed, "tag": tag,
            "attack_accuracy": float((probs.argmax(1) == y).mean()),
            "mean_max_softmax": float(probs.max(1).mean()),
            "mean_entropy_nats": float((-(probs * np.log(np.clip(probs, 1e-12, 1))).sum(1)).mean()),
        })
        diag_path = d / "domain_diag.json"
        if diag_path.exists():
            dg = json.loads(diag_path.read_text(encoding="utf-8"))
            rec["domain_balanced_before"] = dg["before"].get("balanced_acc")
            rec["domain_balanced_after"] = dg["after"].get("balanced_acc")
        (out_dir / f"{d.name}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
        rows.append(rec)
        print(f"{d.name}: ntge_first={rec['ntge_first']} ge@10k={rec['ge_at_10000']} "
              f"acc={rec['attack_accuracy']*100:.2f}% conf={rec['mean_max_softmax']:.3f}")
    if rows:
        keys = ["run", "method", "shift", "seed", "tag", "ntge_first", "ntge_stable",
                "n_total_at_ge1", "ge_at_1000", "ge_at_10000", "min_ge", "transient_only",
                "attack_accuracy", "mean_max_softmax", "mean_entropy_nats",
                "domain_balanced_before", "domain_balanced_after"]
        with (out_dir / "controls.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader(); w.writerows(rows)
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
