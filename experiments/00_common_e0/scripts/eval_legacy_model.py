#!/usr/bin/env python3
"""E0 流程验证（阶段 A）：用五月已训练的 Source Only 模型跑通评估闭环。

加载 /d/sca_project 五月训练的 3-block CNN 权重，对 ASCAD 固定密钥
Attack_traces（10000 条）做 256 类预测，计算 GE/key rank 曲线、NTGE、SR，
产出 predictions、GE 曲线图与指标摘要，验证「模型加载→预测→GE 计算→
绘图→归档」全流程。

用法: python experiments/00_common_e0/scripts/eval_legacy_model.py
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

ROOT = Path(__file__).resolve().parents[3]
LEGACY_FE = r"D:\sca_project\results\models\cdan_supervised\best_feature_extractor.h5"
LEGACY_CLF = r"D:\sca_project\results\models\cdan_supervised\best_classifier.h5"
DATASET = r"D:\sca_project\ASCAD_data\ASCAD.h5"
OUT = ROOT / "experiments/00_common_e0/results"

AES_Sbox = np.array([
    0x63,0x7C,0x77,0x7B,0xF2,0x6B,0x6F,0xC5,0x30,0x01,0x67,0x2B,0xFE,0xD7,0xAB,0x76,
    0xCA,0x82,0xC9,0x7D,0xFA,0x59,0x47,0xF0,0xAD,0xD4,0xA2,0xAF,0x9C,0xA4,0x72,0xC0,
    0xB7,0xFD,0x93,0x26,0x36,0x3F,0xF7,0xCC,0x34,0xA5,0xE5,0xF1,0x71,0xD8,0x31,0x15,
    0x04,0xC7,0x23,0xC3,0x18,0x96,0x05,0x9A,0x07,0x12,0x80,0xE2,0xEB,0x27,0xB2,0x75,
    0x09,0x83,0x2C,0x1A,0x1B,0x6E,0x5A,0xA0,0x52,0x3B,0xD6,0xB3,0x29,0xE3,0x2F,0x84,
    0x53,0xD1,0x00,0xED,0x20,0xFC,0xB1,0x5B,0x6A,0xCB,0xBE,0x39,0x4A,0x4C,0x58,0xCF,
    0xD0,0xEF,0xAA,0xFB,0x43,0x4D,0x33,0x85,0x45,0xF9,0x02,0x7F,0x50,0x3C,0x9F,0xA8,
    0x51,0xA3,0x40,0x8F,0x92,0x9D,0x38,0xF5,0xBC,0xB6,0xDA,0x21,0x10,0xFF,0xF3,0xD2,
    0xCD,0x0C,0x13,0xEC,0x5F,0x97,0x44,0x17,0xC4,0xA7,0x7E,0x3D,0x64,0x5D,0x19,0x73,
    0x60,0x81,0x4F,0xDC,0x22,0x2A,0x90,0x88,0x46,0xEE,0xB8,0x14,0xDE,0x5E,0x0B,0xDB,
    0xE0,0x32,0x3A,0x0A,0x49,0x06,0x24,0x5C,0xC2,0xD3,0xAC,0x62,0x91,0x95,0xE4,0x79,
    0xE7,0xC8,0x37,0x6D,0x8D,0xD5,0x4E,0xA9,0x6C,0x56,0xF4,0xEA,0x65,0x7A,0xAE,0x08,
    0xBA,0x78,0x25,0x2E,0x1C,0xA6,0xB4,0xC6,0xE8,0xDD,0x74,0x1F,0x4B,0xBD,0x8B,0x8A,
    0x70,0x3E,0xB5,0x66,0x48,0x03,0xF6,0x0E,0x61,0x35,0x57,0xB9,0x86,0xC1,0x1D,0x9E,
    0xE1,0xF8,0x98,0x11,0x69,0xD9,0x8E,0x94,0x9B,0x1E,0x87,0xE9,0xCE,0x55,0x28,0xDF,
    0x8C,0xA1,0x89,0x0D,0xBF,0xE6,0x42,0x68,0x41,0x99,0x2D,0x0F,0xB0,0x54,0xBB,0x16,
])


def build_may_feature_extractor(input_shape=(700, 1)):
    """与 /d/sca_project/models/feature_extractor.py 完全一致（五月版）。"""
    return models.Sequential([
        layers.Conv1D(64, 11, activation='relu', padding='same', input_shape=input_shape),
        layers.MaxPooling1D(2),
        layers.Conv1D(128, 11, activation='relu', padding='same'),
        layers.MaxPooling1D(2),
        layers.Conv1D(256, 11, activation='relu', padding='same'),
        layers.MaxPooling1D(2),
        layers.Flatten(),
        layers.Dense(4096, activation='relu'),
    ], name="feature_extractor")


def build_may_classifier(feature_dim=4096, num_classes=256):
    return models.Sequential([
        layers.Dense(1024, activation='relu', input_shape=(feature_dim,)),
        layers.Dense(num_classes, activation='softmax'),
    ], name="classifier")


def ge_curve(probs, plaintext, real_key, byte, step=100):
    scores = np.zeros(256, dtype=np.float64)
    curve = []
    for end in range(step, len(probs) + 1, step):
        p = np.clip(probs[end - step:end], 1e-36, 1.0)
        values = AES_Sbox[np.bitwise_xor(plaintext[end - step:end, byte, None],
                                         np.arange(256, dtype=np.uint8))]
        scores += np.log(p[np.arange(step)[:, None], values]).sum(axis=0)
        curve.append((end, int(np.sum(scores > scores[real_key]))))
    return curve


def main():
    t0 = time.time()
    fe = build_may_feature_extractor()
    clf = build_may_classifier()
    fe.load_weights(LEGACY_FE)
    clf.load_weights(LEGACY_CLF)
    print("[ok] legacy weights loaded", flush=True)

    with h5py.File(DATASET, "r") as f:
        Xa = f["Attack_traces/traces"][:].astype(np.float32).reshape(-1, 700, 1)
        meta = f["Attack_traces/metadata"][:]

    probs = clf.predict(fe.predict(Xa, batch_size=500, verbose=0), batch_size=500, verbose=0)
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(str(OUT / "predictions_attack_legacy_source_only.npy"), probs)

    byte = 2
    pt = np.asarray(meta["plaintext"])
    real_key = int(meta["key"][0][byte])
    curve = ge_curve(probs, pt, real_key, byte, step=100)
    ntge = next((n for n, r in curve if r == 0), None)
    acc = float(np.mean(np.argmax(probs, axis=1) == np.asarray(
        h5py.File(DATASET, "r")["Attack_traces/labels"][:])))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig_dir = ROOT / "experiments/00_common_e0/figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    xs = [n for n, _ in curve]; ys = [r for _, r in curve]
    plt.figure(figsize=(7, 4))
    plt.plot(xs, ys, color="#1f77b4")
    plt.xlabel("Number of attack traces"); plt.ylabel("Guessing Entropy (key rank)")
    plt.title("E0 pipeline check: legacy Source Only (May) on ASCAD fixed-key")
    plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(str(fig_dir / "e0_ge_curve_legacy.png"), dpi=160)

    summary = {
        "model": "legacy 3-block CNN (May 12 weights)",
        "dataset": "ASCAD fixed-key, Attack_traces (10000)",
        "target_byte": byte,
        "attack_accuracy": acc,
        "ntge_traces": ntge,
        "ge_at_10000": curve[-1][1],
        "sr_at_1000": int(dict(curve)[1000] == 0),
        "sr_at_10000": int(curve[-1][1] == 0),
        "elapsed_seconds": round(time.time() - t0, 1),
        "ge_curve": [{"traces": n, "ge": r} for n, r in curve],
    }
    (OUT / "e0_legacy_eval_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "ge_curve"},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
