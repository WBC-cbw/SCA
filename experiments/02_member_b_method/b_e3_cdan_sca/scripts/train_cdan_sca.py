#!/usr/bin/env python3
"""B-E3：CDAN-SCA full / 消融 1（marginal + 熵权重）trainer。

与 A-E2 train_uda.py 的差异只有两点，其余数据/主干/配方/判别器完全一致：
  1. mode=full     -> 条件表示 T(f,g)（随机多线性映射，proj_dim=1024）+ GRL
     mode=marginal -> 仅特征 f 的边缘对齐（DANN 式判别器输入），无 T(f,g)
  2. 目标域域对抗损失乘熵权重 w = 1 - H(g)/log(256)（两种 mode 都乘）

消融 2（去熵权重 = CDAN）直接复用 A-E2 的 6 个 run，不在本脚本重跑。

训练期监控（每 epoch）：target_adapt 固定 2000 条子集上的 top1 类别占比、
预测熵、w 分布；域判别源/目标准确率；GE 探针（500 条，每 2 epoch）。
top1 占比 >30% 打 WARN、>50% 打 COLLAPSE，写入 history 与日志，不早停。

用法：
  python scripts/train_cdan_sca.py --mode full --shift desync50 --seed 42
  python scripts/train_cdan_sca.py --mode marginal --shift desync100 --seed 43
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "experiments/01_member_a_baselines/a_e2_uda_baselines/scripts"))
sys.path.insert(0, str(HERE / "scripts"))

from uda_common import (B_E1_MODELS, UDANet, accuracy, count_params, domain_accuracy,
                        load_b_e1_init, load_domain_data, predict_probs, save_json,
                        seed_everything)  # noqa: E402
from cdan_sca_common import (LOG256, calibration_stats, collapse_level, collapse_level,
                             entropy_nats, entropy_weight, load_target_adapt_labels,
                             monitor_stats)  # noqa: E402

SHIFT_FILES = {
    "desync50": "D:/sca_project/ASCAD_data/ASCAD_desync50.h5",
    "desync100": "D:/sca_project/ASCAD_data/ASCAD_desync100.h5",
}
MODE_NET = {"full": "cdan", "marginal": "dann"}
MODE_TAG = {"full": "full", "marginal": "marg"}


def ge_curve(probs, plaintext, key, byte, step=100):
    from importlib.util import module_from_spec, spec_from_file_location
    spec = spec_from_file_location(
        "ev", str(ROOT / "experiments/cdan-sca-diagnosis/scripts/evaluate_ascad.py"))
    ev = module_from_spec(spec)
    spec.loader.exec_module(ev)
    return ev.ranks(probs, plaintext, key, byte, step)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["full", "marginal"])
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
    ap.add_argument("--monitor-n", type=int, default=2000)
    ap.add_argument("--diag-n", type=int, default=5000)
    ap.add_argument("--init-ckpt", default="")
    ap.add_argument("--out-root", default=str(HERE / "results"))
    ap.add_argument("--models-root", default=str(ROOT / "outputs/models/b_e3"))
    ap.add_argument("--save-every", type=int, default=5)
    ap.add_argument("--tag", default="")
    ap.add_argument("--no-tf32", action="store_true")
    args = ap.parse_args()

    run = f"cdan_sca_{MODE_TAG[args.mode]}_{args.shift}_s{args.seed}" + (f"_{args.tag}" if args.tag else "")
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
        xv, yv = xv[:args.limit_source], yv[:args.limit_source]
    if args.limit_target:
        xt = xt[:args.limit_target]
        xdv = xdv[:args.limit_target]

    # 诊断标签：只在训练结束后用于校准分析，绝不参与损失或模型选择
    adapt_labels = load_target_adapt_labels(SHIFT_FILES[args.shift], len(xt), byte=2)

    model = UDANet(MODE_NET[args.mode], proj_dim=args.proj_dim,
                   map_seed=args.seed + 1000).to(device)
    n_loaded = load_b_e1_init(model, init_ckpt)
    n_params = count_params(model)
    n_disc = count_params(model.discriminator)
    opt = torch.optim.RMSprop(model.parameters(), lr=args.lr)
    ce = nn.CrossEntropyLoss()
    bce = nn.BCEWithLogitsLoss()
    lam = args.lambda_adv

    monitor_x = xt[:min(args.monitor_n, len(xt))]
    diag_x = xt[:min(args.diag_n, len(xt))]
    diag_labels = adapt_labels[:min(args.diag_n, len(adapt_labels))]

    history = {"epochs": [], "cls_loss": [], "dom_loss": [], "d_acc_src": [], "d_acc_tgt": [],
               "val_acc": [], "attack_acc": [], "ge_probe": [], "epoch_seconds": [],
               "monitor": [], "collapse_flags": []}
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

    diag_before = domain_accuracy(model, xv, xdv, lam, device=device)
    diag_before["epoch"] = start_epoch
    print(json.dumps({"diag_before": diag_before}), flush=True)

    meta = {
        "run": run, "method": f"cdan_sca_{args.mode}", "mode": args.mode,
        "conditional_input": args.mode == "full",
        "entropy_weighting": True,
        "entropy_weight_formula": "w(g) = 1 - H(g)/log(256), H in nats; target-domain adversarial loss only; w detached",
        "shift": args.shift, "seed": args.seed, "epochs_planned": args.epochs,
        "lr": args.lr, "optimizer": "RMSprop", "batch_size": args.batch_size,
        "lambda_adv": lam, "lambda_application": "GRL 反转系数作用于条件表示/特征；判别器按（熵加权的目标域）域损失更新",
        "loss": ("L_cls(source) + BCE_src + mean_t[w(g_t) * BCE_tgt]" if args.mode == "full" else
                 "L_cls(source) + BCE_src + mean_t[w(g_t) * BCE_tgt]（判别器输入为 L2 缩放特征，无 T(f,g)）"),
        "proj_dim": args.proj_dim if args.mode == "full" else None,
        "normalize": "none",
        "disc_input_norm": ("L2-unit*sqrt(4096) 的 f（marginal）"
                            if args.mode == "marginal" else
                            "T=(Rf^T f_unit*sqrt(d))⊙(Rg^T g)，proj_dim=1024"),
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
        "protocol_note": "target_adapt 无标签（desync Profiling 前 45000），末 5000 条仅域判别诊断；"
                         "target_adapt 的标签仅在训练结束后用于校准诊断，不参与训练与模型选择",
        "monitor": {"n_target_adapt_subset": int(len(monitor_x)), "collapse_warn_top1_share": 0.30,
                    "collapse_hard_top1_share": 0.50, "probe_traces": args.probe_traces},
        "diagnostic_labels_use": "target_adapt / target_attack 标签只用于事后 ECE 与 confident-wrong 统计",
    }
    save_json(out_dir / "run_meta.json", meta)
    print(json.dumps({"start": {k: meta[k] for k in ("run", "method", "shift", "seed",
                                                     "n_params_total", "target_adapt")}}), flush=True)

    n = len(xs); bs = args.batch_size; nt = len(xt)
    t_start = time.time()
    for epoch in range(start_epoch + 1, args.epochs + 1):
        model.train()
        t0 = time.time()
        perm = np.random.permutation(n)
        tgt_perm = np.random.permutation(nt); tgt_ptr = 0
        ep = {"cls": 0.0, "dom": 0.0, "dsrc": 0, "dtgt": 0, "cnt": 0, "w": 0.0, "wcnt": 0}
        for s in range(0, n, bs):
            idx = perm[s:s + bs]; k = len(idx)
            if tgt_ptr + k > nt:
                tgt_perm = np.random.permutation(nt); tgt_ptr = 0
            tix = tgt_perm[tgt_ptr:tgt_ptr + k]; tgt_ptr += k

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
            bce_s = bce(d_s, torch.zeros(k, device=device))
            with torch.no_grad():
                probs_t = torch.softmax(logits_t.detach(), dim=1)
                w_t = (1.0 - (-(probs_t * torch.log(probs_t.clamp_min(1e-12))).sum(1)) / LOG256)
            bce_t = F.binary_cross_entropy_with_logits(
                d_t, torch.ones(k, device=device), reduction="none")
            loss = cls_loss + bce_s + (w_t * bce_t).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

            ep["cls"] += float(cls_loss.item()) * k
            ep["dom"] += float(bce_s.item() + (w_t * bce_t).mean().item()) * k
            ep["dsrc"] += int(((d_s > 0).logical_not()).sum().item())
            ep["dtgt"] += int((d_t > 0).sum().item())
            ep["w"] += float(w_t.mean().item()) * k
            ep["wcnt"] += k
            ep["cnt"] += k

        model.eval()
        val_acc = accuracy(model, xv, yv, device=device)
        attack_acc = None
        if epoch % args.attack_acc_every == 0 or epoch == args.epochs:
            attack_acc = accuracy(model, xa, ya, device=device)

        probe_rank = None
        if epoch % args.probe_every == 0 or epoch == args.epochs:
            probs_p = predict_probs(model, xa[:args.probe_traces], device=device)
            curve = ge_curve(probs_p, data["attack_plaintext"][:args.probe_traces],
                             data["attack_key"], 2, step=100)
            probe_rank = curve[-1][1] if curve else None

        mon = monitor_stats(predict_probs(model, monitor_x, device=device))
        flag = collapse_level(mon["top1_share"])
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
        history["monitor"].append(mon)
        history["collapse_flags"].append(flag)
        save_json(hist_path, history)
        if epoch % args.save_every == 0 or epoch == args.epochs:
            torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                        "epoch": epoch, "history": history}, str(ckpt_path))
        rec = {"epoch": epoch, "cls": round(history["cls_loss"][-1], 4),
               "dom": round(history["dom_loss"][-1], 4),
               "d_src": round(history["d_acc_src"][-1], 3),
               "d_tgt": round(history["d_acc_tgt"][-1], 3),
               "w": round(ep["w"] / ep["wcnt"], 3),
               "top1": round(mon["top1_share"], 3), "H": round(mon["pred_entropy_mean"], 3),
               "val_acc": round(history["val_acc"][-1], 4),
               "ge_probe": probe_rank, "flag": flag, "sec": round(sec, 1)}
        print(json.dumps(rec), flush=True)
        if flag != "OK":
            print(f"[{flag}] epoch {epoch} top1_share={mon['top1_share']:.3f} "
                  f"top1_class={mon['top1_class']} H={mon['pred_entropy_mean']:.3f}", flush=True)

    probs_attack = predict_probs(model, xa, device=device)
    np.save(str(out_dir / "predictions_attack.npy"), probs_attack)
    probs_diag = predict_probs(model, diag_x, device=device)
    np.savez_compressed(str(out_dir / "diag_target_adapt.npz"),
                        probs=probs_diag.astype(np.float32),
                        labels=diag_labels.astype(np.int64),
                        entropy=entropy_nats(probs_diag).astype(np.float32),
                        w=entropy_weight(probs_diag).astype(np.float32))
    diag_stats = calibration_stats(probs_diag, diag_labels)
    diag_stats["subset"] = f"target_adapt[0:{len(diag_x)}]"
    diag_stats["note"] = "标签仅用于事后诊断"
    save_json(out_dir / "entropy_diag.json", diag_stats)

    final_attack_acc = float((probs_attack.argmax(1) == ya).mean())
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
        "final_monitor": history["monitor"][-1],
        "collapse_epochs": int(sum(1 for f in history["collapse_flags"] if f == "COLLAPSE")),
        "warn_epochs": int(sum(1 for f in history["collapse_flags"] if f == "WARN")),
        "max_top1_share": float(max(m["top1_share"] for m in history["monitor"])),
        "domain_diag_before": diag_before,
        "domain_diag_after": diag_after,
        "max_memory_allocated_mb": round(torch.cuda.max_memory_allocated() / 1e6, 1) if device == "cuda" else None,
        "predictions_file": str(out_dir / "predictions_attack.npy"),
        "diag_file": str(out_dir / "diag_target_adapt.npz"),
        "ckpt_file": str(ckpt_path),
    })
    save_json(out_dir / "run_meta.json", meta)
    print(json.dumps({"done": run, "sec": round(time.time() - t_start, 1),
                      "attack_acc": round(final_attack_acc, 4),
                      "top1": round(meta["final_monitor"]["top1_share"], 3),
                      "collapse_epochs": meta["collapse_epochs"],
                      "domain_before": diag_before, "domain_after": diag_after}), flush=True)


if __name__ == "__main__":
    main()
