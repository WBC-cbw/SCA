#!/usr/bin/env python3
"""A-E2：DANN / CDAN 统一 UDA trainer（PyTorch，冻结主干选择 + B-E1 初始化）。

协议：
  source_train = fixed Profiling[0:45000]，source_val = fixed Profiling[45000:50000]
  target_adapt = desync Profiling[0:n_adapt]（无标签，仅用于对抗项）
  target_domain_val = desync Profiling[n_adapt:]（只做域判别器诊断）
  target_attack = desync Attack[0:10000]（只做最终 GE/SR 评估与辅助 accuracy）
训练损失：L = L_cls(source) + L_domain，λ_adv 通过 GRL 施加在特征/条件表示上
（判别器本身按未加权的域损失更新，与 DANN 原文一致；见 run_meta.json 的
lambda_application 字段）。CDAN 不加熵权重。

用法:
  python scripts/train_uda.py --method dann --shift desync50 --seed 42
  python scripts/train_uda.py --method cdan --shift desync50 --seed 42 --epochs 2 \
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

from uda_common import (B_E1_MODELS, ROOT, UDANet, accuracy, count_params,
                        domain_accuracy, load_b_e1_init, load_domain_data,
                        predict_probs, save_json, seed_everything)

HERE = Path(__file__).resolve().parents[1]          # .../a_e2_uda_baselines
SHIFT_FILES = {
    "desync50": "D:/sca_project/ASCAD_data/ASCAD_desync50.h5",
    "desync100": "D:/sca_project/ASCAD_data/ASCAD_desync100.h5",
}


def ge_curve(probs, plaintext, key, byte, step=100):
    """与 evaluate_ascad.py 同一个 ranks 口径；返回 [(traces, rank)]。"""
    from importlib.util import module_from_spec, spec_from_file_location
    spec = spec_from_file_location(
        "ev", str(ROOT / "experiments/cdan-sca-diagnosis/scripts/evaluate_ascad.py"))
    ev = module_from_spec(spec)
    spec.loader.exec_module(ev)
    return ev.ranks(probs, plaintext, key, byte, step)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["dann", "cdan"])
    ap.add_argument("--shift", required=True, choices=sorted(SHIFT_FILES))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=75)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch-size", type=int, default=200)
    ap.add_argument("--lambda-adv", type=float, default=0.1)
    ap.add_argument("--proj-dim", type=int, default=1024)
    ap.add_argument("--n-adapt", type=int, default=45000)
    ap.add_argument("--n-source-train", type=int, default=45000)
    ap.add_argument("--limit-source", type=int, default=0)
    ap.add_argument("--limit-target", type=int, default=0)
    ap.add_argument("--probe-traces", type=int, default=500)
    ap.add_argument("--probe-every", type=int, default=2)
    ap.add_argument("--attack-acc-every", type=int, default=5)
    ap.add_argument("--init-ckpt", default="")
    ap.add_argument("--out-root", default=str(HERE / "results"))
    ap.add_argument("--models-root", default=str(ROOT / "outputs/models/a_e2"))
    ap.add_argument("--no-tf32", action="store_true", help="关闭 TF32（默认开启，记录在 run_meta）")
    ap.add_argument("--save-every", type=int, default=5, help="每 N 个 epoch 写一次 checkpoint（末轮必写）")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    run = f"{args.method}_{args.shift}_s{args.seed}" + (f"_{args.tag}" if args.tag else "")
    out_dir = Path(args.out_root) / run
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(args.models_root) / run
    model_dir.mkdir(parents=True, exist_ok=True)
    init_ckpt = Path(args.init_ckpt) if args.init_ckpt else B_E1_MODELS / f"cnn_s{args.seed}" / "ckpt.pt"

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
    xdv = data["target_domain_val_x"]
    xa = data["attack_x"]; ya = data["attack_true_label"]
    if args.limit_source:
        xs, ys = xs[:args.limit_source], ys[:args.limit_source]
    if args.limit_target:
        xt = xt[:args.limit_target]
        xdv = xdv[:args.limit_target]
    if args.limit_source:
        xv, yv = xv[:args.limit_source], yv[:args.limit_source]

    model = UDANet(args.method, proj_dim=args.proj_dim,
                   map_seed=args.seed + 1000).to(device)
    n_loaded = load_b_e1_init(model, init_ckpt)
    n_params = count_params(model)
    n_disc = count_params(model.discriminator)
    opt = torch.optim.RMSprop(model.parameters(), lr=args.lr)
    ce = nn.CrossEntropyLoss()
    bce = nn.BCEWithLogitsLoss()
    lam = args.lambda_adv

    history = {"epochs": [], "cls_loss": [], "dom_loss": [], "d_acc_src": [],
               "d_acc_tgt": [], "val_acc": [], "attack_acc": [], "ge_probe": [],
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

    # 训练前域判别诊断（随机初始化/初始特征）
    diag_before = domain_accuracy(model, xv, xdv, lam, device=device)
    diag_before["epoch"] = start_epoch
    print(json.dumps({"diag_before": diag_before}), flush=True)

    meta = {
        "run": run, "method": args.method, "shift": args.shift, "seed": args.seed,
        "epochs_planned": args.epochs, "lr": args.lr, "optimizer": "RMSprop",
        "batch_size": args.batch_size, "lambda_adv": lam,
        "lambda_application": "GRL 反转系数作用于特征/条件表示；判别器按未加权域损失更新",
        "loss": "L_cls(source) + L_domain(GRL(lambda_adv))",
        "proj_dim": args.proj_dim if args.method == "cdan" else None,
        "entropy_weighting": False,
        "normalize": "none",
        "disc_input_norm": "L2-unit * sqrt(4096)（DANN）/ 条件表示 T=(Rf^T f_unit*sqrt(d))⊙(Rg^T g)（CDAN）",
        "init_ckpt": str(init_ckpt), "init_keys_loaded": n_loaded,
        "n_params_total": n_params, "n_params_discriminator": n_disc,
        "n_params_backbone_classifier": n_params - n_disc,
        "source_train": int(len(xs)), "source_val": int(len(xv)),
        "target_adapt": int(len(xt)), "target_domain_val": int(len(xdv)),
        "target_attack": int(len(xa)), "target_byte": 2,
        "real_key_byte": int(data["real_key_byte"]),
        "target_file": SHIFT_FILES[args.shift],
        "dataset_keys_equal": bool(data["keys_equal"]),
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "torch": torch.__version__,
        "tf32": bool(torch.backends.cuda.matmul.allow_tf32) if device == "cuda" else None,
        "protocol_note": "target_adapt 为无标签 desync Profiling；末 5000 条仅用于域判别诊断，不参与训练",
    }
    save_json(out_dir / "run_meta.json", meta)
    print(json.dumps({"start": {k: meta[k] for k in
          ("run", "method", "shift", "seed", "n_params_total", "target_adapt")}}), flush=True)

    n = len(xs)
    bs = args.batch_size
    nt = len(xt)
    t_start = time.time()
    for epoch in range(start_epoch + 1, args.epochs + 1):
        model.train()
        t0 = time.time()
        perm = np.random.permutation(n)
        tgt_perm = np.random.permutation(nt)
        tgt_ptr = 0
        ep = {"cls": 0.0, "dom": 0.0, "dsrc": 0, "dtgt": 0, "cnt": 0}
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

            fs = model.features(xb)
            logits_s = model.logits_from_features(fs)
            ft = model.features(tb)
            logits_t = model.logits_from_features(ft)

            cls_loss = ce(logits_s, yb)
            d_s = model.domain_logits(fs, logits_s, lam)
            d_t = model.domain_logits(ft, logits_t, lam)
            d_logits = torch.cat([d_s, d_t])
            d_labels = torch.cat([torch.zeros(k, device=device),
                                  torch.ones(k, device=device)])
            dom_loss = bce(d_logits, d_labels)
            loss = cls_loss + dom_loss

            opt.zero_grad()
            loss.backward()
            opt.step()

            ep["cls"] += float(cls_loss.item()) * k
            ep["dom"] += float(dom_loss.item()) * k
            ep["dsrc"] += int(((d_s > 0).logical_not()).sum().item())
            ep["dtgt"] += int((d_t > 0).sum().item())
            ep["cnt"] += k

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
        history["dom_loss"].append(ep["dom"] / ep["cnt"])
        history["d_acc_src"].append(ep["dsrc"] / ep["cnt"])
        history["d_acc_tgt"].append(ep["dtgt"] / ep["cnt"])
        history["val_acc"].append(float(val_acc))
        history["attack_acc"].append(None if attack_acc is None else float(attack_acc))
        history["ge_probe"].append(probe_rank)
        history["epoch_seconds"].append(sec)
        save_json(hist_path, history)
        if epoch % args.save_every == 0 or epoch == args.epochs:
            torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                        "epoch": epoch, "history": history}, str(ckpt_path))
        print(json.dumps({"epoch": epoch, "cls": round(history["cls_loss"][-1], 4),
                          "dom": round(history["dom_loss"][-1], 4),
                          "d_src": round(history["d_acc_src"][-1], 3),
                          "d_tgt": round(history["d_acc_tgt"][-1], 3),
                          "val_acc": round(history["val_acc"][-1], 4),
                          "attack_acc": history["attack_acc"][-1],
                          "ge_probe": probe_rank, "sec": round(sec, 1)}), flush=True)

    # 收尾评估
    probs = predict_probs(model, xa, device=device)
    np.save(str(out_dir / "predictions_attack.npy"), probs)
    final_attack_acc = float((probs.argmax(1) == ya).mean())
    diag_after = domain_accuracy(model, xv, xdv, lam, device=device)
    diag_after["epoch"] = history["epochs"][-1]
    save_json(out_dir / "domain_diag.json",
              {"before": diag_before, "after": diag_after,
               "diag_sets": {"source": "fixed Profiling[45000:50000]",
                             "target": "desync Profiling[45000:50000]"}})
    meta.update({
        "epochs_done": history["epochs"][-1],
        "train_seconds": float(sum(history["epoch_seconds"])),
        "wall_seconds_total": time.time() - t_start,
        "sec_per_epoch_mean": float(np.mean(history["epoch_seconds"])),
        "final_train_cls_loss": history["cls_loss"][-1],
        "final_train_dom_loss": history["dom_loss"][-1],
        "final_val_acc": history["val_acc"][-1],
        "final_attack_acc": final_attack_acc,
        "domain_diag_before": diag_before,
        "domain_diag_after": diag_after,
        "max_memory_allocated_mb": round(torch.cuda.max_memory_allocated() / 1e6, 1) if device == "cuda" else None,
        "predictions_file": str(out_dir / "predictions_attack.npy"),
        "ckpt_file": str(ckpt_path),
    })
    save_json(out_dir / "run_meta.json", meta)
    print(json.dumps({"done": run, "sec": round(time.time() - t_start, 1),
                      "attack_acc": round(final_attack_acc, 4),
                      "domain_before": diag_before, "domain_after": diag_after}), flush=True)


if __name__ == "__main__":
    main()
