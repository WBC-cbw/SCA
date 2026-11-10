#!/usr/bin/env python3
"""置信度校准诊断：对每个 run 的 target_attack 预测计算 acc / ECE / confident-wrong。

- ECE：15 个等宽置信区间，|acc(bin) - conf(bin)| 加权求和；
- confident-wrong：max softmax >= 阈值且 argmax 错误的比例；
- 标签仅用于事后诊断（不参与训练与模型选择）。

输出 results/confidence_analysis.csv（逐 run）与 stdout 表格。
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
DATA = Path("D:/sca_project/ASCAD_data")
SHIFTS = {"desync50": DATA / "ASCAD_desync50.h5",
          "desync100": DATA / "ASCAD_desync100.h5"}
B_E2 = HERE.parents[1] / "02_member_b_method/b_e2_architecture_shift"
RUN_RE = re.compile(r"^(dann|cdan)_(desync\d+)_s(\d+)(?:_(.+))?$")

SBOX = np.array([0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16],
dtype=np.uint8)


def labels_for(shift):
    with h5py.File(SHIFTS[shift], "r") as h:
        meta = np.asarray(h["Attack_traces/metadata"])
    pt = np.asarray(meta["plaintext"]); key = np.asarray(meta["key"])
    return SBOX[np.bitwise_xor(pt[:, 2], np.uint8(key[0, 2]))].astype(np.int64)


def ece(probs, y, n_bins=15):
    conf = probs.max(1)
    pred = probs.argmax(1)
    acc = (pred == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    total = len(y)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if hi < 1 else (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        e += abs(acc[m].mean() - conf[m].mean()) * m.sum() / total
    return float(e)


def analyze(probs, y):
    conf = probs.max(1); pred = probs.argmax(1); wrong = pred != y
    ent = -(probs * np.log(np.clip(probs, 1e-12, 1))).sum(1)
    return {
        "n": int(len(y)),
        "attack_accuracy": float((~wrong).mean()),
        "mean_max_softmax": float(conf.mean()),
        "mean_entropy_nats": float(ent.mean()),
        "ece_15": ece(probs, y),
        "frac_conf_ge_0.9": float((conf >= 0.9).mean()),
        "frac_confwrong_ge_0.9": float(((conf >= 0.9) & wrong).mean()),
        "frac_confwrong_ge_0.99": float(((conf >= 0.99) & wrong).mean()),
        "frac_conf_lt_0.1": float((conf < 0.1).mean()),
    }


def main():
    labels = {s: labels_for(s) for s in SHIFTS}
    rows = []
    # Source Only（B-E2 预测）
    for shift in SHIFTS:
        for seed in (42, 43, 44):
            p = B_E2 / "results/predictions" / shift / f"cnn_s{seed}.npy"
            if p.exists():
                rec = analyze(np.load(p), labels[shift])
                rec.update({"method": "source_only", "shift": shift, "seed": seed, "tag": ""})
                rows.append(rec)
    # DANN/CDAN（含 tag）
    for d in sorted((HERE / "results").iterdir()):
        if not d.is_dir():
            continue
        m = RUN_RE.match(d.name)
        if not m:
            continue
        pred = d / "predictions_attack.npy"
        if not pred.exists():
            continue
        method, shift, seed, tag = m.group(1), m.group(2), int(m.group(3)), m.group(4) or ""
        rec = analyze(np.load(pred), labels[shift])
        rec.update({"method": method, "shift": shift, "seed": seed, "tag": tag})
        rows.append(rec)

    rows.sort(key=lambda r: (r["shift"], r["method"], r["tag"], r["seed"]))
    out = HERE / "results/confidence_analysis.csv"
    cols = ["method", "shift", "seed", "tag", "n", "attack_accuracy", "mean_max_softmax",
            "mean_entropy_nats", "ece_15", "frac_conf_ge_0.9", "frac_confwrong_ge_0.9",
            "frac_confwrong_ge_0.99", "frac_conf_lt_0.1"]
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader(); w.writerows(rows)
    for r in rows:
        print(f"{r['shift']:9s} {r['method']:11s} {r['seed']} {r['tag']:5s} "
              f"acc={r['attack_accuracy']*100:.2f}% conf={r['mean_max_softmax']:.3f} "
              f"ECE={r['ece_15']:.3f} cw@0.9={r['frac_confwrong_ge_0.9']*100:.1f}%")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
