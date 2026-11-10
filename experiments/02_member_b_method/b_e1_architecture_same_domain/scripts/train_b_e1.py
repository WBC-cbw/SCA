#!/usr/bin/env python3
"""B-E1 结构筛选：MLP / 官方 CNN / ResNet1D 在 ASCAD fixed-key 上的同域 Source Only。

统一训练配方（与 E0 official 配置逐项一致，只改变 backbone 一个变量）：
  优化器 RMSprop lr=1e-5，batch=200，75 epochs，原始 int8 幅度输入（不归一化）
  数据：Profiling 前 50000（前 45000 = source_train，末 5000 = source_val），
        Attack 10000 = target_attack，target_byte = 2，无 adaptation 数据

官方 CNN 不重写：直接从 experiments/00_common_e0/scripts/train_e0_torch.py
导入 OfficialCNN / ge_probe，保证与 E0 完全同一份实现。

用法:
  python scripts/train_b_e1.py --arch cnn --seed 42
  python scripts/train_b_e1.py --arch mlp --seed 42 --epochs 1 --train-limit 2000 --tag smoke
"""
import argparse
import importlib.util
import json
import random
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[4]   # .../sca_repo
E0_TRAIN = ROOT / "experiments/00_common_e0/scripts/train_e0_torch.py"
HERE = Path(__file__).resolve().parents[1]


def load_e0_module():
    """加载 E0 训练脚本作为模块，复用 OfficialCNN 与 GE 探针（不复制代码）。"""
    spec = importlib.util.spec_from_file_location("e0_train", str(E0_TRAIN))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


E0 = load_e0_module()
OfficialCNN = E0.OfficialCNN
ge_probe = E0.ge_probe


class MLP(nn.Module):
    """Flatten -> Dense(512,relu) -> Dense(512,relu) -> Dense(256)。"""

    def __init__(self, num_classes=256, input_len=700, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_len, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class ResBlock1D(nn.Module):
    """两个 Conv1D(k=11) + 残差；通道数或步长变化时用 1x1 卷积投影 shortcut。"""

    def __init__(self, c_in, c_out, stride):
        super().__init__()
        self.conv1 = nn.Conv1d(c_in, c_out, 11, stride=stride, padding=5)
        self.conv2 = nn.Conv1d(c_out, c_out, 11, stride=1, padding=5)
        self.relu = nn.ReLU()
        self.proj = None
        if c_in != c_out or stride != 1:
            self.proj = nn.Conv1d(c_in, c_out, 1, stride=stride)

    def forward(self, x):
        identity = x if self.proj is None else self.proj(x)
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return self.relu(out + identity)


class ResNet1D(nn.Module):
    """3 个残差块，通道 64/128/256，下采样 stride 1/2/2，GlobalAvgPool -> Dense(256)。"""

    def __init__(self, num_classes=256, widths=(64, 128, 256), strides=(1, 2, 2)):
        super().__init__()
        blocks = []
        c_in = 1
        for c_out, st in zip(widths, strides):
            blocks.append(ResBlock1D(c_in, c_out, st))
            c_in = c_out
        self.blocks = nn.Sequential(*blocks)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(c_in, num_classes)

    def features(self, x):
        h = self.blocks(x)
        return self.gap(h).flatten(1)

    def forward(self, x):
        return self.fc(self.features(x))


ARCHS = {"mlp": MLP, "cnn": OfficialCNN, "resnet1d": ResNet1D}


def count_params(model):
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=sorted(ARCHS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=75)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--optimizer", default="rmsprop", choices=["rmsprop", "adam"])
    ap.add_argument("--batch-size", type=int, default=200)
    ap.add_argument("--train-limit", type=int, default=45000)
    ap.add_argument("--val-size", type=int, default=5000)
    ap.add_argument("--probe-traces", type=int, default=500)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out-root", default=str(HERE / "results"))
    ap.add_argument("--dataset",
                    default="D:/sca_project/ASCAD_data/ASCAD.h5")
    args = ap.parse_args()

    run_name = f"{args.arch}_s{args.seed}" + (f"_{args.tag}" if args.tag else "")
    out_dir = Path(args.out_root) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 固定随机种子（与 E0 相同方式：random / numpy / torch / cuda）----
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- 数据：与 E0 完全一致的划分 ----
    with h5py.File(args.dataset, "r") as f:
        n_prof = args.train_limit + args.val_size
        Xp = f["Profiling_traces/traces"][:n_prof].astype(np.float32)
        Yp = f["Profiling_traces/labels"][:n_prof].astype(np.int64)
        meta_a = f["Attack_traces/metadata"][:]
        Xa = f["Attack_traces/traces"][:].astype(np.float32)

    n_val = args.val_size
    Xp = Xp[:, None, :]              # (N,1,700)
    Xa = Xa[:, None, :]
    X_train, Y_train = Xp[:-n_val], Yp[:-n_val]
    X_val, Y_val = Xp[-n_val:], Yp[-n_val:]
    byte = 2
    pt_attack = np.asarray(meta_a["plaintext"])
    real_key = int(meta_a["key"][0][byte])

    model = ARCHS[args.arch]().to(device)
    n_params = count_params(model)
    if args.optimizer == "rmsprop":
        opt = torch.optim.RMSprop(model.parameters(), lr=args.lr)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    n = len(X_train)
    bs = args.batch_size
    history = {"epochs": [], "loss": [], "accuracy": [], "val_loss": [],
               "val_accuracy": [], "ge_probe": [], "epoch_seconds": []}
    ckpt = out_dir / "ckpt.pt"
    hist_file = out_dir / "history.json"
    start_epoch = 0
    if ckpt.exists() and hist_file.exists():
        model.load_state_dict(torch.load(str(ckpt), map_location=device, weights_only=True))
        history = json.loads(hist_file.read_text(encoding="utf-8"))
        start_epoch = history["epochs"][-1]
        print(f"[resume] {run_name} from epoch {start_epoch}", flush=True)

    x_val_t = torch.from_numpy(X_val).to(device)
    y_val_t = torch.from_numpy(Y_val).to(device)

    meta = {
        "run": run_name, "arch": args.arch, "seed": args.seed,
        "epochs_planned": args.epochs, "batch_size": bs, "learning_rate": args.lr,
        "optimizer": args.optimizer, "normalize": "none",
        "input_shape": [1, 700], "num_classes": 256,
        "n_params": n_params, "train_size": int(len(X_train)),
        "val_size": int(len(X_val)), "attack_size": int(len(Xa)),
        "target_byte": byte, "real_key_byte": real_key,
        "dataset": args.dataset,
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "torch": torch.__version__, "numpy": np.__version__,
        "train_limit": args.train_limit,
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps({"start": meta}), flush=True)

    t_all = time.time()
    for epoch in range(start_epoch + 1, args.epochs + 1):
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
        # 与 E0 一致：第 1 epoch 及之后每 2 个 epoch 用 500 条攻击轨迹探 GE
        if epoch % 2 == 0 or epoch == 1:
            probe_rank, _ = ge_probe(model, Xa, pt_attack, real_key, byte,
                                     min(args.probe_traces, len(Xa)), device=device)

        history["epochs"].append(epoch)
        history["loss"].append(ep_loss / ep_cnt)
        history["accuracy"].append(ep_correct / ep_cnt)
        history["val_loss"].append(v_loss)
        history["val_accuracy"].append(v_acc)
        history["ge_probe"].append(probe_rank)
        history["epoch_seconds"].append(time.time() - t0)
        hist_file.write_text(json.dumps(history, indent=2), encoding="utf-8")
        torch.save(model.state_dict(), str(ckpt))
        print(json.dumps({"epoch": epoch, "loss": round(ep_loss / ep_cnt, 4),
                          "acc": round(ep_correct / ep_cnt, 4),
                          "val_acc": round(v_acc, 4), "ge_probe": probe_rank,
                          "sec": round(time.time() - t0, 1)}), flush=True)

    # ---- 收尾：全量 attack 预测概率（256 类）----
    model.eval()
    probs = []
    with torch.no_grad():
        for s in range(0, len(Xa), 2000):
            xb = torch.from_numpy(Xa[s:s + 2000]).to(device)
            probs.append(torch.softmax(model(xb), dim=1).cpu().numpy())
    probs = np.concatenate(probs)
    np.save(str(out_dir / "predictions_attack.npy"), probs)

    meta.update({
        "epochs_done": history["epochs"][-1],
        "train_seconds": float(sum(history["epoch_seconds"])),
        "wall_seconds_total": time.time() - t_all,
        "sec_per_epoch_mean": float(np.mean(history["epoch_seconds"])),
        "final_train_acc": float(history["accuracy"][-1]),
        "final_val_acc": float(history["val_accuracy"][-1]),
        "final_train_loss": float(history["loss"][-1]),
        "final_val_loss": float(history["val_loss"][-1]),
        "first_ge_zero_probe_epoch": next(
            (e for e, r in zip(history["epochs"], history["ge_probe"]) if r == 0), None),
        "predictions_file": str(out_dir / "predictions_attack.npy"),
    })
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps({"done": run_name, "params": n_params,
                      "epochs": meta["epochs_done"],
                      "train_seconds": round(meta["train_seconds"], 1),
                      "val_acc": round(meta["final_val_acc"], 4)}), flush=True)


if __name__ == "__main__":
    main()
