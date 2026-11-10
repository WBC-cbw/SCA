#!/usr/bin/env python3
"""B-E2 零适应推理：B-E1 的 9 个 checkpoint 直接作用于 fixed / desync50 / desync100 的 Attack_traces。

关键约束：
  - 不重新训练、不做任何域适应、不做输入归一化；模型与预处理和 B-E1 完全一致（原始 int8 幅度）。
  - 三结构类定义不复制，直接从 B-E1 的 train_b_e1.py 导入（其内部又复用 E0 的 OfficialCNN）。
  - fixed 偏移同时重跑一遍，作为 pipeline 自检：应与 B-E1 的 predictions_attack.npy 完全一致。

输出:
  results/predictions/<shift>/<run>.npy            256 类 softmax 概率（10000 x 256, float32）
  results/predictions/<shift>/predictions_meta.json 每个 run 的攻击集 accuracy / 置信度 / 耗时
"""
import argparse
import importlib.util
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parents[1]
B1 = ROOT / "experiments/02_member_b_method/b_e1_architecture_same_domain"
TRAIN_B1 = B1 / "scripts/train_b_e1.py"
MODELS = ROOT / "outputs/models/b_e1"

DATA = Path("D:/sca_project/ASCAD_data")
SHIFTS = {
    "fixed": DATA / "ASCAD.h5",
    "desync50": DATA / "ASCAD_desync50.h5",
    "desync100": DATA / "ASCAD_desync100.h5",
}
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


def load_b1_module():
    spec = importlib.util.spec_from_file_location("train_b_e1", str(TRAIN_B1))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def discover_runs():
    runs = []
    for d in sorted((B1 / "results").iterdir()):
        meta_file = d / "run_meta.json"
        if not d.is_dir() or not meta_file.exists():
            continue
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        ckpt = MODELS / d.name / "ckpt.pt"
        if not ckpt.exists():
            print(f"[skip] {d.name}: no checkpoint at {ckpt}")
            continue
        runs.append({"run": d.name, "arch": meta["arch"], "seed": int(meta["seed"]),
                     "ckpt": ckpt, "b1_predictions": d / "predictions_attack.npy"})
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shift", required=True, choices=sorted(SHIFTS))
    ap.add_argument("--batch", type=int, default=2000)
    ap.add_argument("--out", default=str(HERE / "results/predictions"))
    args = ap.parse_args()

    TB = load_b1_module()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = SHIFTS[args.shift]

    with h5py.File(dataset, "r") as h:
        X = np.asarray(h["Attack_traces/traces"][:], dtype=np.float32)[:, None, :]
        meta_a = np.asarray(h["Attack_traces/metadata"])
    pt = np.asarray(meta_a["plaintext"])
    key = np.asarray(meta_a["key"])
    labels = SBOX[np.bitwise_xor(pt[:, BYTE], np.uint8(int(key[0, BYTE])))].astype(np.int64)

    out_dir = Path(args.out) / args.shift
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    x_t = torch.from_numpy(X)

    for rec in discover_runs():
        model = TB.ARCHS[rec["arch"]]().to(device)
        model.load_state_dict(torch.load(str(rec["ckpt"]), map_location=device, weights_only=True))
        model.eval()

        t0 = time.time()
        probs = []
        with torch.no_grad():
            for s in range(0, len(X), args.batch):
                xb = x_t[s:s + args.batch].to(device)
                probs.append(torch.softmax(model(xb), dim=1).float().cpu().numpy())
        probs = np.concatenate(probs)
        secs = time.time() - t0

        assert probs.shape == (len(X), 256), probs.shape
        np.save(str(out_dir / f"{rec['run']}.npy"), probs)

        pred = probs.argmax(1)
        acc = float((pred == labels).mean())
        conf = float(probs.max(1).mean())
        ent = float(-(np.clip(probs, 1e-12, 1.0) * np.log(np.clip(probs, 1e-12, 1.0))).sum(1).mean())
        item = {"run": rec["run"], "arch": rec["arch"], "seed": rec["seed"],
                "shift": args.shift, "dataset": str(dataset),
                "n_traces": int(len(X)), "inference_seconds": round(secs, 2),
                "attack_accuracy": round(acc, 6),
                "mean_max_softmax": round(conf, 6),
                "mean_entropy_nats": round(ent, 6),
                "ckpt": str(rec["ckpt"].relative_to(ROOT))}

        if args.shift == "fixed" and rec["b1_predictions"].exists():
            ref = np.load(rec["b1_predictions"])
            item["b1_reference_max_abs_diff"] = float(np.abs(ref - probs).max())
            item["b1_reference_argmax_agreement"] = float((ref.argmax(1) == pred).mean())
        records.append(item)
        print(json.dumps(item), flush=True)

    summary = {"shift": args.shift, "dataset": str(dataset), "device": device,
               "torch": torch.__version__, "batch": args.batch, "runs": records}
    (out_dir / "predictions_meta.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {len(records)} prediction files -> {out_dir}")


if __name__ == "__main__":
    main()
