#!/usr/bin/env python3
"""C-E4 评估：method x budget x seed，攻击侧 GE/NTGE/成本双口径。

网格（desync50 第一阶段，desync100 第二阶段）：
  source_only  budget=0    无适应，复用 B-E2 交叉域预测（desync attack）
  cdpa / dann / cdan_sca  budget in {50, 500, 5000}   本实验 results/<run>/predictions_attack.npy
                           budget=45000              复用 A-E3 / A-E2 / B-E3 的已有预测
攻击侧统一为 desync Attack[0:10000]，标签 = Sbox[pt[2]^key[2]]，与适应子集不重叠。
成本口径：N_attack@GE=1 = 首个 GE=0 的攻击轨迹数；N_total@GE=1 = N_attack + budget。
GE=0 未出现记 None，聚合时截尾 10000（NTGE）/ 10000+budget（N_total）。
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parents[1]
EVAL_SRC = ROOT / "experiments/cdan-sca-diagnosis/scripts/evaluate_ascad.py"
B_E2 = ROOT / "experiments/02_member_b_method/b_e2_architecture_shift"
A_E2 = ROOT / "experiments/01_member_a_baselines/a_e2_uda_baselines"
A_E3 = ROOT / "experiments/01_member_a_baselines/a_e3_cdpa_or_adabn"
B_E3 = ROOT / "experiments/02_member_b_method/b_e3_cdan_sca"
sys.path.insert(0, str(B_E3 / "scripts"))
from cdan_sca_common import calibration_stats  # noqa: E402

DATA = Path("D:/sca_project/ASCAD_data")
SHIFTS = {"desync50": DATA / "ASCAD_desync50.h5", "desync100": DATA / "ASCAD_desync100.h5"}
SEEDS = [42, 43, 44]
BUDGETS = [50, 500, 5000]
FULL_BUDGET = 45000
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


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def summary_from_curve(curve, budget):
    ge_at = {n: r for n, r in curve}
    zeros = [n for n, r in curve if r == 0]
    ntge_first = zeros[0] if zeros else None
    ntge_stable = None
    tail_start = None
    for n, r in reversed(curve):
        if r == 0:
            tail_start = n
        else:
            break
    if tail_start is not None and curve[-1][1] == 0:
        ntge_stable = tail_start
    return {
        "ntge_first": ntge_first,
        "ntge_stable": ntge_stable,
        "ge_at_1000": ge_at.get(1000),
        "ge_at_2000": ge_at.get(2000),
        "ge_at_5000": ge_at.get(5000),
        "ge_at_10000": ge_at.get(10000),
        "min_ge": int(min(r for _, r in curve)),
        "final_ge": int(curve[-1][1]),
        "n_zero_points": len(zeros),
        "transient_only": bool(zeros) and curve[-1][1] != 0,
        "sr_at_1000": int(ge_at.get(1000, 255) == 0),
        "sr_at_10000": int(ge_at.get(10000, 255) == 0),
        "n_adapt": int(budget),
        "n_attack_at_ge1": ntge_first,
        "n_total_at_ge1": (ntge_first + budget) if ntge_first is not None else None,
        "n_attack_at_stable_ge0": ntge_stable,
        "n_total_at_stable_ge0": (ntge_stable + budget) if ntge_stable is not None else None,
        "n_total_includes_adapt": budget > 0,
    }


def collect_items(shift):
    items = []
    for seed in SEEDS:
        items.append(dict(method="source_only", budget=0, seed=seed,
                          run=f"source_only_{shift}_s{seed}",
                          pred=B_E2 / "results/predictions" / shift / f"cnn_s{seed}.npy",
                          run_dir=None, reuse="B-E2"))
        for b in BUDGETS:
            for method, pred_fn, run_dir_fn, reuse in [
                ("cdpa_mmd", lambda s, b=b: HERE / f"results/cdpa_{shift}_b{b}_s{s}/predictions_attack.npy",
                 lambda s, b=b: HERE / f"results/cdpa_{shift}_b{b}_s{s}", "C-E4"),
                ("dann", lambda s, b=b: HERE / f"results/dann_{shift}_b{b}_s{s}/predictions_attack.npy",
                 lambda s, b=b: HERE / f"results/dann_{shift}_b{b}_s{s}", "C-E4"),
                ("cdan_sca", lambda s, b=b: HERE / f"results/cdan_sca_{shift}_b{b}_s{s}/predictions_attack.npy",
                 lambda s, b=b: HERE / f"results/cdan_sca_{shift}_b{b}_s{s}", "C-E4"),
            ]:
                items.append(dict(method=method, budget=b, seed=seed,
                                  run=f"{method}_{shift}_b{b}_s{seed}",
                                  pred=pred_fn(seed), run_dir=run_dir_fn(seed), reuse=reuse))
        # 全量档（45000）：复用已有 run
        items.append(dict(method="cdpa_mmd", budget=FULL_BUDGET, seed=seed,
                          run=f"cdpa_{shift}_s{seed}",
                          pred=A_E3 / f"results/cdpa_{shift}_s{seed}/predictions_attack.npy",
                          run_dir=A_E3 / f"results/cdpa_{shift}_s{seed}", reuse="A-E3"))
        items.append(dict(method="dann", budget=FULL_BUDGET, seed=seed,
                          run=f"dann_{shift}_s{seed}",
                          pred=A_E2 / f"results/dann_{shift}_s{seed}/predictions_attack.npy",
                          run_dir=A_E2 / f"results/dann_{shift}_s{seed}", reuse="A-E2"))
        items.append(dict(method="cdan_sca", budget=FULL_BUDGET, seed=seed,
                          run=f"cdan_sca_full_{shift}_s{seed}",
                          pred=B_E3 / f"results/cdan_sca_full_{shift}_s{seed}/predictions_attack.npy",
                          run_dir=B_E3 / f"results/cdan_sca_full_{shift}_s{seed}", reuse="B-E3"))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "results/eval"))
    ap.add_argument("--step", type=int, default=100)
    ap.add_argument("--byte", type=int, default=BYTE)
    args = ap.parse_args()

    EA = load_eval_module()
    out_root = Path(args.out)
    rows = []
    for shift, dataset in SHIFTS.items():
        with h5py.File(dataset, "r") as h:
            meta = np.asarray(h["Attack_traces/metadata"])
        plaintext = np.asarray(meta["plaintext"])
        key = np.asarray(meta["key"])
        key_byte = int(key[0, args.byte])
        labels = SBOX[np.bitwise_xor(plaintext[:, args.byte], np.uint8(key_byte))].astype(np.int64)
        out = out_root / shift
        out.mkdir(parents=True, exist_ok=True)

        for it in collect_items(shift):
            pred_path = it["pred"]
            if not pred_path.exists():
                print(f"[skip] {it['method']:12s} b{it['budget']:<5d} s{it['seed']}: no predictions")
                continue
            probs = np.load(pred_path)
            assert probs.shape == (10000, 256), (pred_path, probs.shape)
            curve = EA.ranks(probs, plaintext, key, args.byte, args.step)
            rec = summary_from_curve(curve, it["budget"])
            with (out / f"{it['run']}_ge.csv").open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["traces", "ge"])
                w.writerows(curve)
            rmeta = read_json(it["run_dir"] / "run_meta.json") if it["run_dir"] else None
            calib = calibration_stats(probs, labels)
            rec.update({
                "shift": shift, "method": it["method"], "budget": it["budget"],
                "run": it["run"], "seed": int(it["seed"]), "reuse": it["reuse"],
                "dataset": str(dataset), "predictions": str(pred_path),
                "n_attack_traces": int(len(probs)), "step": args.step,
                "target_byte": args.byte, "real_key_byte": key_byte,
                "attack_accuracy": round(float((probs.argmax(1) == labels).mean()), 6),
                "attack_ece_15": round(calib["ece_15"], 6),
                "attack_frac_confwrong_ge09": round(calib["frac_confwrong_ge_0.9"], 6),
                "attack_mean_max_softmax": round(calib["mean_max_softmax"], 6),
                "attack_mean_entropy_nats": round(calib["mean_entropy_nats"], 6),
                "attack_top1_share": round(calib["top1_share"], 6),
                "train_seconds": (rmeta or {}).get("train_seconds"),
                "sec_per_epoch_mean": (rmeta or {}).get("sec_per_epoch_mean"),
                "max_top1_share": (rmeta or {}).get("max_top1_share"),
                "collapse_epochs": (rmeta or {}).get("collapse_epochs"),
                "warn_epochs": (rmeta or {}).get("warn_epochs"),
                "subset_sha256": (rmeta or {}).get("subset_sha256"),
                "subset_indices_sha256": (rmeta or {}).get("subset_indices_sha256"),
                "ge_curve": [{"traces": n, "ge": r} for n, r in curve],
            })
            (out / f"{it['run']}.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
            rows.append(rec)
            print(f"{shift:9s} {it['method']:12s} b{it['budget']:<5d} s{it['seed']} "
                  f"NTGE={str(rec['ntge_first']):6s} stable={str(rec['ntge_stable']):6s} "
                  f"GE@10k={str(rec['ge_at_10000']):4s} N_total={str(rec['n_total_at_ge1']):6s} "
                  f"acc={rec['attack_accuracy']:.4f}", flush=True)

    out_root.mkdir(parents=True, exist_ok=True)
    fields = ["shift", "method", "budget", "run", "seed", "reuse", "ntge_first", "ntge_stable",
              "ge_at_1000", "ge_at_2000", "ge_at_5000", "ge_at_10000", "min_ge", "final_ge",
              "n_zero_points", "transient_only", "sr_at_1000", "sr_at_10000",
              "n_adapt", "n_attack_at_ge1", "n_total_at_ge1",
              "n_attack_at_stable_ge0", "n_total_at_stable_ge0", "n_total_includes_adapt",
              "attack_accuracy", "attack_ece_15", "attack_frac_confwrong_ge09",
              "attack_mean_max_softmax", "attack_mean_entropy_nats", "attack_top1_share",
              "train_seconds", "sec_per_epoch_mean", "max_top1_share", "collapse_epochs",
              "warn_epochs", "subset_sha256", "subset_indices_sha256"]
    with (out_root / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})
    (out_root / "eval_all.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"wrote {len(rows)} evaluations -> {out_root}")


if __name__ == "__main__":
    main()
