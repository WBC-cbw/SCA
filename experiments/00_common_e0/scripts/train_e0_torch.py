#!/usr/bin/env python3
"""E0 (GPU/PyTorch 版): ASCAD fixed-key Source Only, 官方五层 CNN.

与 TF 版 train_e0.py 同配置、同指标、同归档。架构逐层对应仓库
src/framework/models/feature_extractor.py + classifier.py:
  Conv1D(64,11)+AvgPool -> Conv1D(128,11)+AvgPool -> Conv1D(256,11)+AvgPool
  -> Conv1D(512,11)+AvgPool -> Conv1D(512,11)+AvgPool
  -> Flatten -> Dense(4096,relu)          [feature extractor]
  -> Dense(4096,relu) -> Dense(256)       [classifier]

用法:
    python experiments/00_common_e0/scripts/train_e0_torch.py \
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

import h5py
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[3]

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


class OfficialCNN(nn.Module):
    def __init__(self, num_classes=256):
        super().__init__()
        blocks = []
        c_in = 1
        for c_out in [64, 128, 256, 512, 512]:
            blocks += [nn.Conv1d(c_in, c_out, 11, padding=5), nn.ReLU(),
                       nn.AvgPool1d(2, 2)]
            c_in = c_out
        self.convs = nn.Sequential(*blocks)
        self.fc1 = nn.Linear(21 * 512, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)
        self.relu = nn.ReLU()

    def features(self, x):
        h = self.convs(x)
        h = h.flatten(1)
        return self.relu(self.fc1(h))

    def forward(self, x):
        return self.fc3(self.relu(self.fc2(self.features(x))))


def ge_probe(model, x_attack, plaintexts, real_key_byte, byte, n_traces, step=100,
             device="cuda", batch=2000):
    model.eval()
    probs = []
    with torch.no_grad():
        for s in range(0, n_traces, batch):
            xb = torch.from_numpy(x_attack[s:s + batch]).to(device)
            probs.append(torch.softmax(model(xb), dim=1).cpu().numpy())
    probs = np.concatenate(probs)[:n_traces]
    scores = np.zeros(256, dtype=np.float64)
    rank = 255
    for end in range(step, n_traces + 1, step):
        p = np.clip(probs[end - step:end], 1e-36, 1.0)
        values = AES_Sbox[np.bitwise_xor(plaintexts[end - step:end, byte, None],
                                         np.arange(256, dtype=np.uint8))]
        scores += np.log(p[np.arange(step)[:, None], values]).sum(axis=0)
        rank = int(np.sum(scores > scores[real_key_byte]))
    return rank, probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    spec = importlib.util.spec_from_file_location("config", args.config)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    C = mod.CONFIG

    seed = int(C["seed"])
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    res_dir = ROOT / C["run_dir"] / "results"
    res_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(C["source_file"], "r") as f:
        Xp = f["Profiling_traces/traces"][:C["profiling_limit"]].astype(np.float32)
        Yp = f["Profiling_traces/labels"][:C["profiling_limit"]].astype(np.int64)
        meta_a = f["Attack_traces/metadata"][:]
        Xa = f["Attack_traces/traces"][:C["attack_limit"]].astype(np.float32)

    n_val = int(C["val_size"])
    mode = C.get("normalize", "none")
    if mode == "feature_standardize":
        mu = Xp[:-n_val].mean(axis=0, keepdims=True)
        sd = Xp[:-n_val].std(axis=0, keepdims=True) + 1e-8
        np.save(str(res_dir / "input_mean.npy"), mu)
        np.save(str(res_dir / "input_std.npy"), sd)
        Xp = (Xp - mu) / sd
        Xa = (Xa - mu) / sd
    Xp = Xp[:, None, :]          # (N,1,700)
    Xa = Xa[:, None, :]

    X_train, Y_train = Xp[:-n_val], Yp[:-n_val]
    X_val, Y_val = Xp[-n_val:], Yp[-n_val:]
    byte = int(C["target_byte"])
    pt_attack = np.asarray(meta_a["plaintext"])
    real_key = int(meta_a["key"][0][byte])

    model = OfficialCNN(int(C["num_classes"])).to(device)
    if str(C.get("optimizer", "adam")).lower() == "rmsprop":
        opt = torch.optim.RMSprop(model.parameters(), lr=float(C["learning_rate"]))
    else:
        opt = torch.optim.Adam(model.parameters(), lr=float(C["learning_rate"]))
    loss_fn = nn.CrossEntropyLoss()

    n = len(X_train)
    bs = int(C["batch_size"])
    history = {"epochs": [], "loss": [], "accuracy": [], "val_loss": [],
               "val_accuracy": [], "ge_probe": [], "epoch_seconds": []}
    probe_n = int(C["ge_probe_traces"])
    patience = int(C["early_stop_patience"])
    max_epochs = int(C["epochs"])

    start_epoch = 0
    ckpt = res_dir / "e0_model_torch.pt"
    hist_file = res_dir / "history.json"
    if ckpt.exists() and hist_file.exists():
        model.load_state_dict(torch.load(str(ckpt), weights_only=True))
        history = json.loads(hist_file.read_text(encoding="utf-8"))
        start_epoch = history["epochs"][-1]
        print(f"[resume] from epoch {start_epoch}", flush=True)

    x_val_t = torch.from_numpy(X_val).to(device)
    y_val_t = torch.from_numpy(Y_val).to(device)

    t_all = time.time()
    converged = False
    for epoch in range(start_epoch + 1, max_epochs + 1):
        model.train()
        t0 = time.time()
        perm = np.random.permutation(n)
        ep_loss, ep_correct, ep_cnt = 0.0, 0, 0
        for s in range(0, n, bs):
            idx = perm[s:s + bs]
            xb = torch.from_numpy(X_train[idx]).to(device)
            yb = torch.from_numpy(Y_train[idx]).to(device)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item() * len(idx)
            ep_correct += int((logits.argmax(1) == yb).sum().item())
            ep_cnt += len(idx)

        model.eval()
        with torch.no_grad():
            v_logits = model(x_val_t)
            v_loss = loss_fn(v_logits, y_val_t).item()
            v_acc = float((v_logits.argmax(1) == y_val_t).float().mean().item())

        probe_rank = None
        if C.get("early_stop_ge_probe", True) and (epoch % 2 == 0 or epoch == 1):
            probe_rank, _ = ge_probe(model, Xa, pt_attack, real_key, byte, probe_n,
                                     device=device)

        history["epochs"].append(epoch)
        history["loss"].append(ep_loss / ep_cnt)
        history["accuracy"].append(ep_correct / ep_cnt)
        history["val_loss"].append(v_loss)
        history["val_accuracy"].append(v_acc)
        history["ge_probe"].append(probe_rank)
        history["epoch_seconds"].append(time.time() - t0)
        # 每个 epoch 落盘一次，防止长跑被中断丢进度
        (res_dir / "history.json").write_text(json.dumps(history, indent=2),
                                              encoding="utf-8")
        torch.save(model.state_dict(), str(res_dir / "e0_model_torch.pt"))
        print(json.dumps({"epoch": epoch, "loss": round(ep_loss / ep_cnt, 4),
                          "val_acc": round(v_acc, 4), "ge_probe": probe_rank,
                          "sec": round(time.time() - t0, 1)}), flush=True)

        recent = [r for r in history["ge_probe"][-patience:] if r is not None]
        if len(recent) == patience and all(r == 0 for r in recent):
            converged = True
            break

    # ---------- 收尾 ----------
    torch.save(model.state_dict(), str(res_dir / "e0_model_torch.pt"))
    _, probs_full = ge_probe(model, Xa, pt_attack, real_key, byte,
                             len(Xa), device=device)
    np.save(str(res_dir / "predictions_attack.npy"), probs_full)
    history["total_seconds"] = time.time() - t_all
    history["converged"] = converged
    history["device"] = torch.cuda.get_device_name(0) if device == "cuda" else "cpu"
    (res_dir / "history.json").write_text(json.dumps(history, indent=2),
                                          encoding="utf-8")
    print(json.dumps({"finished_epochs": history["epochs"][-1],
                      "converged": converged,
                      "total_seconds": round(history["total_seconds"], 1),
                      "device": history["device"]}), flush=True)


if __name__ == "__main__":
    main()
