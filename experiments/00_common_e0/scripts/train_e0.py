#!/usr/bin/env python3
"""E0: ASCAD fixed-key -> ASCAD fixed-key, Source Only 流程验证.

分段续训设计：每次进程训练 CONFIG["epochs_per_run"] 个 epoch，
权重与训练历史落盘，下次启动自动续训，直到达到目标 epoch 或
GE 探针（500 条攻击轨迹）连续 patience 次为 0。

用法:
    python experiments/00_common_e0/scripts/train_e0.py \
        --config experiments/00_common_e0/configs/e0_source_only_fixed.py
"""
import argparse
import importlib.util
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import h5py
import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src/framework"))
from models.feature_extractor import build_feature_extractor  # noqa: E402
from models.classifier import build_classifier  # noqa: E402

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


def ge_probe(model, x_attack, plaintexts, real_key_byte, byte, n_traces, step=100):
    """用前 n_traces 条攻击轨迹计算 GE（key rank）曲线，返回最后一个点的 rank。"""
    probs = model.predict(x_attack[:n_traces], batch_size=500, verbose=0)
    scores = np.zeros(256, dtype=np.float64)
    rank = 255
    for end in range(step, n_traces + 1, step):
        p = np.clip(probs[end - step:end], 1e-36, 1.0)
        values = AES_Sbox[np.bitwise_xor(plaintexts[end - step:end, byte, None],
                                         np.arange(256, dtype=np.uint8))]
        scores += np.log(p[np.arange(step)[:, None], values]).sum(axis=0)
        rank = int(np.sum(scores > scores[real_key_byte]))
    return rank


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    spec = importlib.util.spec_from_file_location("config", args.config)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    C = mod.CONFIG

    seed = int(C["seed"])
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)

    run_dir = ROOT / C["run_dir"]
    res_dir = run_dir / "results"
    ckpt_path = res_dir / "e0_model.weights.h5"
    hist_path = res_dir / "history.json"
    res_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 数据 ----------
    with h5py.File(C["source_file"], "r") as f:
        Xp = f["Profiling_traces/traces"][:C["profiling_limit"]].astype(np.float32)
        Yp = f["Profiling_traces/labels"][:C["profiling_limit"]].astype(np.int64)
        meta_a = f["Attack_traces/metadata"][:]
        Xa = f["Attack_traces/traces"][:C["attack_limit"]].astype(np.float32)

    def zscore(x):
        return ((x - x.mean(axis=1, keepdims=True)) /
                (x.std(axis=1, keepdims=True) + 1e-8)).reshape(-1, x.shape[1], 1)

    def reshape(x):
        return x.reshape(-1, x.shape[1], 1)

    mode = C.get("normalize", "none")
    if mode == "zscore_per_trace":
        prep = zscore
        Xp = prep(Xp); Xa = prep(Xa)
    elif mode == "feature_standardize":
        # 统计量只在训练集上估计，再套用到验证集与攻击集
        mu = Xp[:Xp.shape[0] - int(C["val_size"])].mean(axis=0, keepdims=True)
        sd = Xp[:Xp.shape[0] - int(C["val_size"])].std(axis=0, keepdims=True) + 1e-8
        np.save(str(res_dir / "input_mean.npy"), mu)
        np.save(str(res_dir / "input_std.npy"), sd)
        Xp = ((Xp - mu) / sd).reshape(-1, Xp.shape[1], 1)
        Xa = ((Xa - mu) / sd).reshape(-1, Xa.shape[1], 1)
    else:
        Xp = reshape(Xp); Xa = reshape(Xa)
    n_val = int(C["val_size"])
    X_train, Y_train = Xp[:-n_val], Yp[:-n_val]
    X_val, Y_val = Xp[-n_val:], Yp[-n_val:]
    byte = int(C["target_byte"])
    pt_attack = np.asarray(meta_a["plaintext"])
    real_key = int(meta_a["key"][0][byte])

    # ---------- 模型 ----------
    fe = build_feature_extractor((Xp.shape[1], 1))
    clf = build_classifier(feature_dim=fe.output_shape[-1], num_classes=C["num_classes"])
    model = tf.keras.Sequential([fe, clf])
    model.compile(optimizer=tf.keras.optimizers.Adam(C["learning_rate"]),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    history = {"epochs": [], "loss": [], "accuracy": [], "val_loss": [],
               "val_accuracy": [], "ge_probe": [], "epoch_seconds": []}
    start_epoch = 0
    opt_state_path = res_dir / "optimizer_state.npz"
    if ckpt_path.exists() and hist_path.exists():
        model.load_weights(str(ckpt_path))
        history = json.loads(hist_path.read_text(encoding="utf-8"))
        start_epoch = len(history["epochs"])
        print(f"[resume] from epoch {start_epoch}", flush=True)
        # 恢复 Adam 一阶/二阶动量，避免分段续训反复冷启动优化器
        if opt_state_path.exists():
            model.fit(X_train[:2], Y_train[:2], batch_size=2, epochs=1, verbose=0)
            saved = np.load(str(opt_state_path))
            for v, key in zip(model.optimizer.variables, sorted(saved.files)):
                v.assign(saved[key])
            print("[resume] optimizer state restored", flush=True)

    if start_epoch >= int(C["epochs"]):
        print("[done] target epochs reached, nothing to do", flush=True)
        return

    # ---------- 训练一段 ----------
    end_epoch = min(start_epoch + int(C["epochs_per_run"]), int(C["epochs"]))
    t0 = time.time()
    h = model.fit(X_train, Y_train, batch_size=int(C["batch_size"]),
                  initial_epoch=start_epoch, epochs=end_epoch,
                  validation_data=(X_val, Y_val), verbose=0, shuffle=True)
    elapsed = time.time() - t0

    for k in range(len(h.history["loss"])):
        history["epochs"].append(start_epoch + k + 1)
        history["loss"].append(float(h.history["loss"][k]))
        history["accuracy"].append(float(h.history["accuracy"][k]))
        history["val_loss"].append(float(h.history["val_loss"][k]))
        history["val_accuracy"].append(float(h.history["val_accuracy"][k]))
        history["epoch_seconds"].append(elapsed / len(h.history["loss"]))

    probe_rank = None
    if C.get("early_stop_ge_probe", True):
        probe_rank = ge_probe(model, Xa, pt_attack, real_key, byte,
                              int(C["ge_probe_traces"]))
    history["ge_probe"].append(probe_rank)

    model.save_weights(str(ckpt_path))
    np.savez(str(opt_state_path),
             **{f"v{i:04d}": v.numpy() for i, v in enumerate(model.optimizer.variables)})
    hist_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    done_epochs = history["epochs"][-1]
    recent = history["ge_probe"][-int(C["early_stop_patience"]):]
    converged = (len(recent) == int(C["early_stop_patience"])
                 and all(r == 0 for r in recent))
    print(json.dumps({
        "epoch_done": done_epochs,
        "loss": history["loss"][-1],
        "val_accuracy": history["val_accuracy"][-1],
        "ge_probe_last500": probe_rank,
        "epoch_seconds": history["epoch_seconds"][-1],
        "converged": converged,
        "finished": done_epochs >= int(C["epochs"]) or converged,
    }, ensure_ascii=False), flush=True)

    # ---------- 收尾：全量攻击集预测 ----------
    if done_epochs >= int(C["epochs"]) or converged:
        probs = model.predict(Xa, batch_size=500, verbose=0)
        np.save(str(res_dir / "predictions_attack.npy"), probs)
        print("[final] saved predictions_attack.npy", flush=True)


if __name__ == "__main__":
    main()
