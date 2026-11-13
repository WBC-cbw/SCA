#!/usr/bin/env python3
"""B-E5 校准诊断：攻击侧与 target_adapt 侧的 ECE / confident-wrong / 熵 / 权重分布。

三问中的前两问在这里给数据：
1. 温度校准是否让熵权重重新生效：calib 的 w 均值/分位 vs full 的裸 w；
2. 校准是否减少 confident-wrong 与 ECE：attack 与 target_adapt 两侧都算；
3. 收益归因留给 aggregate（唯一变量是权重函数）+ 本表逐 run 的温度/权重事实。

数据来源（不重训）：
- attack 侧：各方法 results/.../predictions_attack.npy（10000 条），calib 额外用自身 T 校准；
- adapt 侧：B-E3 的 results/<run>/diag_target_adapt.npz（前 5000 条 labeled 诊断，标签仅事后用）
  与 B-E5 calib 的同名文件；CDPA-MMD/DANN/CDAN 的 adapt 指标直接复用 B-E3
  results/entropy_weight_analysis.csv（已按同一 15-bin ECE 实现算过）。
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[2]
import sys
sys.path.insert(0, str(HERE / "scripts"))
B_E3 = ROOT / "experiments/02_member_b_method/b_e3_cdan_sca"
B_E2 = ROOT / "experiments/02_member_b_method/b_e2_architecture_shift"
A_E2 = ROOT / "experiments/01_member_a_baselines/a_e2_uda_baselines"
A_E3 = ROOT / "experiments/01_member_a_baselines/a_e3_cdpa_or_adabn"

from calib_common import (LOG256, apply_temperature, accuracy_np, ece_np, entropy_np,
                          weight_np)

SEEDS = [42, 43, 44]
SHIFTS = ["desync50", "desync100"]
N_BINS = 15


def attack_path(method, shift, seed):
    if method == "source_only":
        return B_E2 / "results/predictions" / shift / f"cnn_s{seed}.npy"
    if method in ("dann", "cdan"):
        return A_E2 / "results" / f"{method}_{shift}_s{seed}" / "predictions_attack.npy"
    if method == "cdpa_mmd":
        return A_E3 / "results" / f"cdpa_{shift}_s{seed}" / "predictions_attack.npy"
    if method == "cdan_sca_full":
        return B_E3 / "results" / f"cdan_sca_full_{shift}_s{seed}" / "predictions_attack.npy"
    if method == "cdan_sca_marg":
        return B_E3 / "results" / f"cdan_sca_marg_{shift}_s{seed}" / "predictions_attack.npy"
    if method == "cdan_sca_calib":
        return HERE / "results" / f"cdan_sca_calib_{shift}_s{seed}" / "predictions_attack.npy"
    raise KeyError(method)


def labels_for_attack(shift):
    import h5py
    path = Path("D:/sca_project/ASCAD_data") / f"ASCAD_{shift}.h5"
    with h5py.File(path, "r") as h:
        meta = np.asarray(h["Attack_traces/metadata"])
    sbox = np.array([0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
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
    key_byte = int(np.asarray(meta["key"])[0, 2])
    return sbox[np.bitwise_xor(np.asarray(meta["plaintext"])[:, 2], np.uint8(key_byte))].astype(np.int64)


def metrics(probs, labels, temperature=None):
    p = probs if temperature is None else apply_temperature(probs, temperature)
    ent = entropy_np(p)
    w = weight_np(p)
    pred = p.argmax(1)
    conf = p.max(1)
    correct = pred == labels
    counts = np.bincount(pred, minlength=256)
    ws = {"mean": float(w.mean()), "p10": float(np.percentile(w, 10)),
          "p50": float(np.percentile(w, 50)), "p90": float(np.percentile(w, 90))}
    return {
        "n": int(len(p)),
        "accuracy": round(float(correct.mean()), 6),
        "ece_15": round(float(ece_np(p, labels, N_BINS)), 6),
        "frac_conf_ge_0.9": round(float((conf >= 0.9).mean()), 6),
        "frac_confwrong_ge_0.9": round(float(((~correct) & (conf >= 0.9)).mean()), 6),
        "frac_confwrong_ge_0.99": round(float(((~correct) & (conf >= 0.99)).mean()), 6),
        "frac_conf_lt_0.1": round(float((conf < 0.1).mean()), 6),
        "mean_entropy_nats": round(float(ent.mean()), 6),
        "mean_max_softmax": round(float(conf.mean()), 6),
        "w_mean": round(ws["mean"], 6),
        "w_p10": round(ws["p10"], 6),
        "w_p50": round(ws["p50"], 6),
        "w_p90": round(ws["p90"], 6),
        "top1_share": round(float(counts.max() / len(p)), 6),
        "top1_class": int(counts.argmax()),
        "unique_pred_classes": int((counts > 0).sum()),
    }


def main():
    out_rows = []
    temp_rows = []

    # 1) 已有六方法的 attack/adapt 指标：直接读 B-E3 的 CSV（同口径）
    b3_csv = B_E3 / "results/entropy_weight_analysis.csv"
    if b3_csv.exists():
        with b3_csv.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                row = {"method": r["method"], "label": r["label"], "shift": r["shift"],
                       "seed": int(r["seed"]), "side": r["side"], "calibration": "bare",
                       "temperature": None}
                for k in ("n", "accuracy", "ece_15", "frac_confwrong_ge_0.9", "frac_conf_ge_0.9",
                          "frac_confwrong_ge_0.99", "frac_conf_lt_0.1", "mean_entropy_nats",
                          "mean_max_softmax", "w_mean", "w_p10", "w_p50", "w_p90",
                          "top1_share", "top1_class", "unique_pred_classes"):
                    row[k] = float(r[k]) if "." in r[k] else int(r[k])
                out_rows.append(row)

    # 2) calib：attack 用 predictions + T；adapt 用 diag npz（bare 与 calibrated 都算）
    labels_cache = {s: labels_for_attack(s) for s in SHIFTS}
    for shift in SHIFTS:
        labels_attack = labels_cache[shift]
        for seed in SEEDS:
            run = f"cdan_sca_calib_{shift}_s{seed}"
            run_dir = HERE / "results" / run
            tpath = run_dir / "temperature.json"
            ppath = run_dir / "predictions_attack.npy"
            if not (tpath.exists() and ppath.exists()):
                print(f"[skip] {run} (missing temperature.json or predictions)")
                continue
            tj = json.loads(tpath.read_text(encoding="utf-8"))
            T = float(tj["temperature"])
            temp_rows.append({"run": run, "shift": shift, "seed": seed, "temperature": T,
                              "nll_before": tj["nll_before"], "nll_after": tj["nll_after"],
                              "ece_before": tj["ece_before"], "ece_after": tj["ece_after"],
                              "accuracy": tj["accuracy"], "n_val": tj["n_val"]})
            probs = np.load(ppath)
            for cal, TT in (("bare", None), ("calibrated", T)):
                row = {"method": "cdan_sca_calib", "label": "CDAN-SCA-calib",
                       "shift": shift, "seed": seed, "side": "attack", "calibration": cal,
                       "temperature": T if cal == "calibrated" else None}
                row.update(metrics(probs, labels_attack, TT))
                out_rows.append(row)
            npz = run_dir / "diag_target_adapt.npz"
            if npz.exists():
                d = np.load(npz)
                lab = d["labels"].astype(np.int64)
                for cal, key, TT in (("bare", "probs", None),
                                     ("calibrated", "probs_calibrated", T)):
                    p = d[key] if key in d.files else apply_temperature(d["probs"], T)
                    row = {"method": "cdan_sca_calib", "label": "CDAN-SCA-calib",
                           "shift": shift, "seed": seed, "side": "adapt", "calibration": cal,
                           "temperature": T if cal == "calibrated" else None}
                    row.update(metrics(p, lab, None))
                    out_rows.append(row)

    fields = ["method", "label", "shift", "seed", "side", "calibration", "n", "accuracy",
              "ece_15", "frac_confwrong_ge_0.9", "frac_conf_ge_0.9",
              "frac_confwrong_ge_0.99", "frac_conf_lt_0.1", "mean_entropy_nats",
              "mean_max_softmax", "w_mean", "w_p10", "w_p50", "w_p90", "top1_share",
              "top1_class", "unique_pred_classes", "temperature"]
    out_root = HERE / "results"
    with (out_root / "calibration_analysis.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in out_rows:
            w.writerow({k: r.get(k) for k in fields})

    # 3) seed 均值表（报告直接用）
    def key(r):
        return (r["method"], r["label"], r["shift"], r["side"], r["calibration"])
    groups = {}
    for r in out_rows:
        groups.setdefault(key(r), []).append(r)
    metric_cols = ["accuracy", "ece_15", "frac_confwrong_ge_0.9", "frac_conf_ge_0.9",
                   "frac_confwrong_ge_0.99", "mean_entropy_nats", "mean_max_softmax",
                   "w_mean", "w_p10", "w_p50", "w_p90", "top1_share"]
    summary = []
    for k, rs in groups.items():
        rec = {"method": k[0], "label": k[1], "shift": k[2], "side": k[3],
               "calibration": k[4], "n_runs": len(rs),
               "temperature": rs[0].get("temperature") if k[0] == "cdan_sca_calib" else None}
        for c in metric_cols:
            vals = np.array([r[c] for r in rs], dtype=float)
            rec[c + "_mean"] = float(vals.mean())
            rec[c + "_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        summary.append(rec)
    summary.sort(key=lambda r: (r["shift"], r["side"], r["method"], r["calibration"]))
    with (out_root / "calibration_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    with (out_root / "temperature_fits.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(temp_rows[0].keys()) if temp_rows else ["run"])
        w.writeheader()
        w.writerows(temp_rows)
    print(f"wrote {len(out_rows)} rows, {len(summary)} groups, {len(temp_rows)} temperature fits")


if __name__ == "__main__":
    main()
