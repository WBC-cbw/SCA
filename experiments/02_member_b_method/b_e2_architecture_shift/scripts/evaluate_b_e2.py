#!/usr/bin/env python3
"""B-E2 评估：对每个 shift x run 计算 GE 曲线 / NTGE / GE@N / SR，以及攻击集 accuracy 与置信度。

GE 计算复用仓库标准实现
experiments/cdan-sca-diagnosis/scripts/evaluate_ascad.py::ranks（step=100，target byte=2），
与 E0 / B-E1 同一口径；每个 shift 用各自数据集的 metadata（plaintext/key）现算标签，
不复用 fixed 的标签数组，避免“偏移数据集标签未重新校验”的错误。

输出:
  results/eval/<shift>/<run>_ge.csv       每 100 条轨迹的 key rank
  results/eval/<shift>/<run>.json         ntge / ge@1000 / ge@5000 / ge@10000 / sr / accuracy / 置信度
  results/eval/summary.csv                全部 shift x run 的一行式汇总
  results/eval/eval_all.json              同上，JSON 版（含 ge_curve）
"""
import argparse
import csv
import importlib.util
import json
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parents[1]
EVAL_SRC = ROOT / "experiments/cdan-sca-diagnosis/scripts/evaluate_ascad.py"
DATA = Path("D:/sca_project/ASCAD_data")
SHIFTS = {"fixed": DATA / "ASCAD.h5",
          "desync50": DATA / "ASCAD_desync50.h5",
          "desync100": DATA / "ASCAD_desync100.h5"}
BYTE = 2

SBOX = np.array([
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
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
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16,
], dtype=np.uint8)


def load_eval_module():
    spec = importlib.util.spec_from_file_location("eval_ascad", str(EVAL_SRC))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", default=str(HERE / "results/predictions"))
    ap.add_argument("--out", default=str(HERE / "results/eval"))
    ap.add_argument("--step", type=int, default=100)
    ap.add_argument("--byte", type=int, default=BYTE)
    args = ap.parse_args()

    EA = load_eval_module()
    pred_root = Path(args.predictions)
    out_root = Path(args.out)
    rows = []

    for shift, dataset in SHIFTS.items():
        with h5py.File(dataset, "r") as h:
            meta = np.asarray(h["Attack_traces/metadata"])
        plaintext = np.asarray(meta["plaintext"])
        key = np.asarray(meta["key"])
        key_byte = int(key[0, args.byte])
        labels = SBOX[np.bitwise_xor(plaintext[:, args.byte], np.uint8(key_byte))].astype(np.int64)

        sdir = pred_root / shift
        if not sdir.exists():
            print(f"[skip] {shift}: no predictions dir")
            continue
        out = out_root / shift
        out.mkdir(parents=True, exist_ok=True)

        for pred in sorted(sdir.glob("*.npy")):
            run = pred.stem
            probs = np.load(pred)
            assert probs.shape[1] == 256, probs.shape
            curve = EA.ranks(probs, plaintext, key, args.byte, args.step)
            ntge = next((n for n, r in curve if r == 0), None)
            ge_at = {n: r for n, r in curve}

            with (out / f"{run}_ge.csv").open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["traces", "ge"])
                w.writerows(curve)

            p = np.clip(probs, 1e-12, 1.0)
            pred_lab = probs.argmax(1)
            rec = {
                "run": run, "arch": run.split("_s")[0], "seed": int(run.split("_s")[-1]),
                "shift": shift, "dataset": str(dataset),
                "predictions": str(pred), "n_attack_traces": int(len(probs)),
                "step": args.step, "target_byte": args.byte, "real_key_byte": key_byte,
                "ntge": ntge,
                "n_total_at_ge1": ntge,   # 无 adaptation 轨迹：N_total@GE=1 == N_attack@GE=1
                "ge_at_1000": ge_at.get(1000),
                "ge_at_5000": ge_at.get(5000),
                "ge_at_10000": ge_at.get(10000),
                "sr_at_1000": int(ge_at.get(1000, 255) == 0),
                "sr_at_10000": int(ge_at.get(10000, 255) == 0),
                "attack_accuracy": round(float((pred_lab == labels).mean()), 6),
                "mean_max_softmax": round(float(probs.max(1).mean()), 6),
                "mean_entropy_nats": round(float(-(p * np.log(p)).sum(1).mean()), 6),
                "ge_curve": [{"traces": n, "ge": r} for n, r in curve],
            }
            (out / f"{run}.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
            rows.append(rec)
            print(f"{shift:9s} {run:14s} NTGE={str(ntge):5s} GE@10000={str(rec['ge_at_10000']):4s} "
                  f"SR@10000={rec['sr_at_10000']} acc={rec['attack_accuracy']:.4f}", flush=True)

    out_root.mkdir(parents=True, exist_ok=True)
    fields = ["shift", "arch", "run", "seed", "ntge", "ge_at_1000", "ge_at_5000", "ge_at_10000",
              "sr_at_1000", "sr_at_10000", "attack_accuracy", "mean_max_softmax", "mean_entropy_nats"]
    with (out_root / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fields})
    (out_root / "eval_all.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"wrote {len(rows)} evaluations -> {out_root}")


if __name__ == "__main__":
    main()
