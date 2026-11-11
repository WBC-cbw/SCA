#!/usr/bin/env python3
"""A-E3：CDPA-MMD（预训练 + MMD 微调）在 ASCAD fixed -> desync 上的统一 trainer。

协议（与 A-E2 对齐，便于与 Source Only / DANN / CDAN 同表比较）：
  source_train = fixed Profiling[0:45000]（有标签）
  source_val   = fixed Profiling[45000:50000]（只做监控）
  target_adapt = desync Profiling[0:n_adapt]（无标签，只进入 MMD 项）
  target_attack= desync Attack[0:10000]（只做最终 GE/SR 与辅助 accuracy）
  预训练阶段 = B-E1 的 Source Only 权重（75 epochs，官方配方）
  微调阶段 = 75 epochs，RMSprop lr=1e-5，batch 200，L = CE(source) + lambda_mmd * MMD
             （lambda_mmd=0.1 取自官方 CDPA 配置；MMD 作用在 conv/fc1/fc2 三处表示）

与官方 CDPA 的差异（报告 8.1.5 会逐条列出）：
  主干 OfficialCNN（官方为自带小 CNN，含 BN）；优化器 RMSprop 1e-5 / 75 epochs
  （官方 Adam 1e-3 / 15 epochs）；target_adapt 45000 条（官方 200 条）。

用法:
  python scripts/train_cdpa.py --shift desync50 --seed 42
  python scripts/train_cdpa.py --shift desync50 --seed 42 --epochs 2 \
      --limit-source 2000 --limit-target 2000 --tag smoke
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from cdpa_common import (B_E1_MODELS, CDPANet, MMD_LAYERS_DEFAULT, ROOT,
                         accuracy, count_params, grad_norm_ratio,
                         load_b_e1_init, load_domain_data, predict_probs,
                         save_json, seed_everything)

HERE = Path(__file__).resolve().parents[1]          # .../a_e3_cdpa_or_adabn
SHIFT_FILES = {
    "desync50": "D:/sca_project/ASCAD_data/ASCAD_desync50.h5",
    "desync100": "D:/sca_project/ASCAD_data/ASCAD_desync100.h5",
}


def ge_curve(probs, plaintext, key, byte, step=100):
    from importlib.util import module_from_spec, spec_from_file_location
    spec = spec_from_file_location(
        "ev", str(ROOT / "experiments/cdan-sca-diagnosis/scripts/evaluate_ascad.py"))
    ev = module_from_spec(spec)
    spec.loader.exec_module(ev)
    return ev.ranks(probs, plaintext, key, byte, step)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shift", required=True, choices=sorted(SHIFT_FILES))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=75)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch-size", type=int, default=200)
    ap.add_argument("--lambda-mmd", type=float, default=0.1)
    ap.add_argument("--mmd-layers", default=",".join(MMD_LAYERS_DEFAULT))
    ap.add_argument("--n-adapt", type=int, default=45000)
    ap.add_argument("--n-source-train", type=int, default=45000)
    ap.add_argument("--limit-source", type=int, default=0)
    ap.add_argument("--limit-target", type=int, default=0)
    ap.add_argument("--probe-traces", type=int, default=500)
    ap.add_argument("--probe-every", type=int, default=2)
    ap.add_argument("--attack-acc-every", type=int, default=5)
    ap.add_argument("--grad-probe-epochs", default="1,20,50",
                    help="在这些 epoch 的第一个 batch 上测 lambda*MMD 与 CE 的梯度范数比")
    ap.add_argument("--init-ckpt", default="")
    ap.add_argument("--out-root", default=str(HERE / "results"))
    ap.add_argument("--models-root", default=str(ROOT / "outputs/models/a_e3"))
    ap.add_argument("--no-tf32", action="store_true")
    ap.add_argument("--save-every", type=int, default=5)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    run = f"cdpa_{args.shift}_s{args.seed}" + (f"_{args.tag}" if args.tag else "")
    out_dir = Path(args.out_root) / run
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(args.models_root) / run
    model_dir.mkdir(parents=True, exist_ok=True)
    init_ckpt = Path(args.init_ckpt) if args.init_ckpt else B_E1_MODELS / f"cnn_s{args.seed}" / "ckpt.pt"
    layers = tuple(s.strip() for s in args.mmd_layers.split(",") if s.strip())
    probe_epochs = {int(s) for s in args.grad_probe_epochs.split(",") if s.strip()}

    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        if not args.no_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    data = load_domain_data("D:/sca_project/ASCAD_data/ASCAD.h5", SHIFT_FILES[args.shift],
                            n_source_train=args.n_source_train,
                            n_target_adapt=args.n_adapt)
    xs = data["source_train_x"]; ys = data["source_train_y"]
    xv = data["source_val_x"];  yv = data["source_val_y"]
    xt = data["target_adapt_x"]
    xa = data["attack_x"]; ya = data["attack_true_label"]
    if args.limit_source:
        xs, ys = xs[:args.limit_source], ys[:args.limit_source]
        xv, yv = xv[:args.limit_source], yv[:args.limit_source]
    if args.limit_target:
        xt = xt[:args.limit_target]

    model = CDPANet(num_classes=256).to(device)
    n_loaded = load_b_e1_init(model, init_ckpt)
    n_params = count_params(model)
    opt = torch.optim.RMSprop(model.parameters(), lr=args.lr)
    ce = nn.CrossEntropyLoss()
    lam = args.lambda_mmd

    history = {"epochs": [], "cls_loss": [], "mmd_loss": [],
               "mmd_conv": [], "mmd_fc1": [], "mmd_fc2": [],
               "val_acc": [], "attack_acc": [], "ge_probe": [],
               "epoch_seconds": []}
    start_epoch = 0
    ckpt_path = model_dir / "ckpt.pt"
    hist_path = out_dir / "history.json"
    if ckpt_path.exists() and hist_path.exists():
        state = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        history = state["history"]
        start_epoch = state["epoch"]
        print(f"[resume] {run} from epoch {start_epoch}", flush=True)

    meta = {
        "run": run, "method": "cdpa_mmd", "shift": args.shift, "seed": args.seed,
        "epochs_planned": args.epochs, "lr": args.lr, "optimizer": "RMSprop",
        "batch_size": args.batch_size, "lambda_mmd": lam,
        "mmd_layers": list(layers),
        "mmd_kernel": {"kernel_mul": 2.0, "kernel_num": 5,
                       "bandwidth": "批内平均平方距离 / kernel_mul**(kernel_num//2)（官方实现）"},
        "loss": "CE(source) + lambda_mmd * sum_layers MMD(source, target)",
        "mmd_impl_note": "距离矩阵用 ||x||^2+||y||^2-2xy 计算，与官方 expand 写法数值等价，避免 400x400x4096 张量",
        "pretrain_stage": "B-E1 Source Only（75 epochs, 官方配方）",
        "official_deviations": [
            "主干：OfficialCNN（conv5+fc3）而非官方自带小 CNN（含 BatchNorm）",
            "优化器/轮数：RMSprop lr=1e-5 / 75 epochs，而非 Adam lr=1e-3 / 15 epochs",
            "target_adapt：45000 条无标签轨迹，而非官方 target_finetune_num=200",
            "无 BN 可冻结（官方在微调时 model.eval() 冻结 BN 统计）",
        ],
        "normalize": "none（原始 int8 幅度）",
        "init_ckpt": str(init_ckpt), "init_keys_loaded": n_loaded,
        "n_params_total": n_params,
        "source_train": int(len(xs)), "source_val": int(len(xv)),
        "target_adapt": int(len(xt)), "target_attack": int(len(xa)),
        "target_byte": 2, "real_key_byte": int(data["real_key_byte"]),
        "target_file": SHIFT_FILES[args.shift],
        "dataset_keys_equal": bool(data["keys_equal"]),
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "torch": torch.__version__,
        "tf32": bool(torch.backends.cuda.matmul.allow_tf32) if device == "cuda" else None,
        "protocol_note": "target_adapt 为无标签 desync Profiling；target_domain_val 不参与训练也不使用",
    }
    save_json(out_dir / "run_meta.json", meta)
    print(json.dumps({"start": {k: meta[k] for k in
          ("run", "shift", "seed", "n_params_total", "lambda_mmd", "target_adapt")}}), flush=True)

    n = len(xs)
    bs = args.batch_size
    nt = len(xt)
    t_start = time.time()
    probe_history = []
    if (out_dir / "grad_probe.json").exists():
        try:
            probe_history = json.loads((out_dir / "grad_probe.json").read_text(encoding="utf-8"))
        except Exception:
            probe_history = []
    for epoch in range(start_epoch + 1, args.epochs + 1):
        model.train()
        t0 = time.time()
        perm = np.random.permutation(n)
        tgt_perm = np.random.permutation(nt)
        tgt_ptr = 0
        ep = {"cls": 0.0, "mmd": 0.0, "conv": 0.0, "fc1": 0.0, "fc2": 0.0, "cnt": 0}
        probe_batch = None
        for s in range(0, n, bs):
            idx = perm[s:s + bs]
            k = len(idx)
            if tgt_ptr + k > nt:
                tgt_perm = np.random.permutation(nt)
                tgt_ptr = 0
            tix = tgt_perm[tgt_ptr:tgt_ptr + k]
            tgt_ptr += k

            xb = torch.from_numpy(xs[idx]).to(device)
            yb = torch.from_numpy(ys[idx]).to(device)
            tb = torch.from_numpy(xt[tix]).to(device)

            if probe_batch is None and epoch in probe_epochs:
                probe_batch = (xb, yb, tb)

            logits_s, terms = model.forward_pair(xb, tb, layers)
            mmd = sum(terms.values())
            cls_loss = ce(logits_s, yb)
            loss = cls_loss + lam * mmd

            opt.zero_grad()
            loss.backward()
            opt.step()

            ep["cls"] += float(cls_loss.item()) * k
            ep["mmd"] += float(mmd.item()) * k
            for name in ("conv", "fc1", "fc2"):
                if name in terms:
                    ep[name] += float(terms[name].item()) * k
            ep["cnt"] += k

        if probe_batch is not None:
            probe = grad_norm_ratio(model, *probe_batch, lam, layers, ce)
            probe["epoch"] = epoch
            probe_history.append({"lambda_mmd": lam, **probe})
            save_json(out_dir / "grad_probe.json", probe_history)
            print(json.dumps({"grad_probe": probe}), flush=True)

        model.eval()
        val_acc = accuracy(model, xv, yv, device=device)
        attack_acc = None
        if epoch % args.attack_acc_every == 0 or epoch == args.epochs:
            attack_acc = accuracy(model, xa, ya, device=device)
        probe_rank = None
        if epoch % args.probe_every == 0 or epoch == args.epochs:
            probs = predict_probs(model, xa[:args.probe_traces], device=device)
            curve = ge_curve(probs, data["attack_plaintext"][:args.probe_traces],
                             data["attack_key"], 2, step=100)
            probe_rank = curve[-1][1] if curve else None

        sec = time.time() - t0
        history["epochs"].append(epoch)
        history["cls_loss"].append(ep["cls"] / ep["cnt"])
        history["mmd_loss"].append(ep["mmd"] / ep["cnt"])
        history["mmd_conv"].append(ep["conv"] / ep["cnt"] if "conv" in layers else None)
        history["mmd_fc1"].append(ep["fc1"] / ep["cnt"] if "fc1" in layers else None)
        history["mmd_fc2"].append(ep["fc2"] / ep["cnt"] if "fc2" in layers else None)
        history["val_acc"].append(float(val_acc))
        history["attack_acc"].append(None if attack_acc is None else float(attack_acc))
        history["ge_probe"].append(probe_rank)
        history["epoch_seconds"].append(sec)
        save_json(hist_path, history)
        if epoch % args.save_every == 0 or epoch == args.epochs:
            torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                        "epoch": epoch, "history": history}, str(ckpt_path))
        print(json.dumps({"epoch": epoch, "cls": round(history["cls_loss"][-1], 4),
                          "mmd": round(history["mmd_loss"][-1], 6),
                          "mmd_conv": None if history["mmd_conv"][-1] is None else round(history["mmd_conv"][-1], 6),
                          "val_acc": round(history["val_acc"][-1], 4),
                          "ge_probe": probe_rank, "sec": round(sec, 1)}), flush=True)

    probs = predict_probs(model, xa, device=device)
    np.save(str(out_dir / "predictions_attack.npy"), probs)
    final_attack_acc = float((probs.argmax(1) == ya).mean())
    meta.update({
        "epochs_done": history["epochs"][-1],
        "train_seconds": float(sum(history["epoch_seconds"])),
        "wall_seconds_total": time.time() - t_start,
        "sec_per_epoch_mean": float(np.mean(history["epoch_seconds"])),
        "final_train_cls_loss": history["cls_loss"][-1],
        "final_mmd_loss": history["mmd_loss"][-1],
        "final_val_acc": history["val_acc"][-1],
        "final_attack_acc": final_attack_acc,
        "max_memory_allocated_mb": round(torch.cuda.max_memory_allocated() / 1e6, 1) if device == "cuda" else None,
        "predictions_file": str(out_dir / "predictions_attack.npy"),
        "ckpt_file": str(ckpt_path),
    })
    save_json(out_dir / "run_meta.json", meta)
    print(json.dumps({"done": run, "sec": round(time.time() - t_start, 1),
                      "attack_acc": round(final_attack_acc, 4)}), flush=True)


if __name__ == "__main__":
    main()
