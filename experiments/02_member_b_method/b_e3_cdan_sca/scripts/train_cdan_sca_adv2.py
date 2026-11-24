#!/usr/bin/env python3
"""CDAN-SCA adv2：把对抗训练"做稳"的加强版（B-E3 full 的改进线）。

相对 B-E3 full（train_cdan_sca.py）只改对抗相关部分，数据/主干/配方/判别器结构不变：
  1) 条件输入一致性过滤：目标批做两次随机扰动（时移 ±aug_shift、幅度缩放、噪声），
     仅当两次预测 argmax 一致（可选再要求置信 > gate_tau）的样本参与条件对抗；
     早期预测不可靠时 keep 比例低，对抗项自然"软启动"。
  2) λ 斜坡：前 ramp_epochs 个 epoch 内 λ_eff = λ * epoch/ramp，先纯源域分类再逐步加对抗。
  3) 目标域先验约束：EMA 估计目标域边缘预测分布 p̄，加 KL(p̄ || uniform)，
     直接压制单类坍塌（top1 share 高 → KL 大 → 被惩罚）。
  4) 判别器降速 + 标签平滑：判别器参数单独 param group（lr = lr * d_lr_mult），
     域标签用 label_smooth 软化，避免 D 两个 epoch 就饱和。
  5) 监控：keep_frac、p̄ 熵、KL、λ_eff，以及每 grad_probe_every epoch 在固定小批上
     计算 CE 梯度与对抗梯度（经 GRL 后）在特征提取器参数上的余弦相似度（冲突诊断）。

产物与 B-E3 对齐：predictions_attack.npy / diag_target_adapt.npz / entropy_diag.json /
history.json / run_meta.json / domain_diag.json / ckpt.pt。

用法：
  python scripts/train_cdan_sca_adv2.py --shift desync100 --seed 42
  python scripts/train_cdan_sca_adv2.py --shift desync50 --seed 43 --ramp-epochs 15
"""
from __future__ import annotations

import argparse
import json
import math
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
                        grad_reverse, load_b_e1_init, load_domain_data, predict_probs,
                        save_json, seed_everything)  # noqa: E402
from cdan_sca_common import (LOG256, calibration_stats, collapse_level,
                             entropy_nats, entropy_weight, load_target_adapt_labels,
                             monitor_stats)  # noqa: E402

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


def augment(x, max_shift=8, amp=0.05, noise=0.01, gen=None):
    """轻量标签保持扰动：时移（roll，边缘仅污染 max_shift 点）+ 幅度缩放 + 高斯噪声。"""
    if max_shift > 0:
        s = int(torch.randint(-max_shift, max_shift + 1, (1,), device=x.device).item())
        if s:
            x = torch.roll(x, shifts=s, dims=-1)
    if amp > 0:
        scale = 1.0 + (torch.rand(x.shape[0], 1, 1, device=x.device) * 2 - 1) * amp
        x = x * scale
    if noise > 0:
        x = x + torch.randn_like(x) * noise * x.std().clamp_min(1.0)
    return x


def domain_logits(model, f, logits, lam, detach_cond=True, reverse=True):
    """条件表示上的判别器前向；默认 detach 条件输入，避免对抗梯度直接改写分类头。"""
    g = torch.softmax(logits.detach() if detach_cond else logits, dim=1)
    z = model.cdan_map(f, g)
    if reverse:
        z = grad_reverse(z, lam)
    return model.discriminator(z)


def grad_cosine(model, ce, xb, yb, tb, lam_eff):
    """CE 梯度与对抗梯度（GRL 之后）在 feature_extractor 参数上的余弦相似度。

    负值 = 两项在特征层互相拉扯；接近 0 = 基本正交。
    """
    params = [p for p in model.feature_extractor.parameters() if p.requires_grad]
    fs = model.features(xb)
    ls = model.logits_from_features(fs)
    lc = ce(ls, yb)
    ft = model.features(tb)
    lt = model.logits_from_features(ft)
    d_t = model.domain_logits(ft, lt, lam_eff)
    la = F.binary_cross_entropy_with_logits(d_t, torch.full_like(d_t, 0.95))
    g1 = torch.autograd.grad(lc, params, retain_graph=True, allow_unused=True)
    g2 = torch.autograd.grad(la, params, retain_graph=True, allow_unused=True)
    v1 = torch.cat([g.reshape(-1) for g in g1 if g is not None])
    v2 = torch.cat([g.reshape(-1) for g in g2 if g is not None])
    denom = (v1.norm() * v2.norm()).clamp_min(1e-12)
    return float((v1 @ v2 / denom).item()), float(v1.norm().item()), float(v2.norm().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shift", required=True, choices=sorted(SHIFT_FILES))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=75)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch-size", type=int, default=200)
    ap.add_argument("--lambda-adv", type=float, default=0.1)
    ap.add_argument("--ramp-epochs", type=int, default=20)
    ap.add_argument("--reg-weight", type=float, default=0.3)
    ap.add_argument("--consistency", type=int, default=1)
    ap.add_argument("--gate-tau", type=float, default=0.0)
    ap.add_argument("--keep-min-frac", type=float, default=0.05)
    ap.add_argument("--aux-full", type=float, default=0.5,
                    help="判别器在完整目标批（特征 detach）上的辅助域损失权重")
    ap.add_argument("--detach-cond", type=int, default=1)
    ap.add_argument("--clip-grad", type=float, default=5.0)
    ap.add_argument("--aug-shift", type=int, default=8)
    ap.add_argument("--aug-amp", type=float, default=0.05)
    ap.add_argument("--aug-noise", type=float, default=0.01)
    ap.add_argument("--d-lr-mult", type=float, default=1.0)
    ap.add_argument("--label-smooth", type=float, default=0.1)
    ap.add_argument("--prior-momentum", type=float, default=0.9)
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
    ap.add_argument("--grad-probe-every", type=int, default=10)
    ap.add_argument("--init-ckpt", default="")
    ap.add_argument("--out-root", default=str(HERE / "results_adv2"))
    ap.add_argument("--models-root", default=str(ROOT / "outputs/models/b_e3_adv2"))
    ap.add_argument("--save-every", type=int, default=5)
    ap.add_argument("--tag", default="")
    ap.add_argument("--no-tf32", action="store_true")
    args = ap.parse_args()

    run = f"cdan_sca_adv2_{args.shift}_s{args.seed}" + (f"_{args.tag}" if args.tag else "")
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
                            n_source_train=args.n_source_train, n_target_adapt=args.n_adapt)
    xs = data["source_train_x"]; ys = data["source_train_y"]
    xv = data["source_val_x"];  yv = data["source_val_y"]
    xt = data["target_adapt_x"]; xdv = data["target_domain_val_x"]
    xa = data["attack_x"]; ya = data["attack_true_label"]
    if args.limit_source:
        xs, ys = xs[:args.limit_source], ys[:args.limit_source]
        xv, yv = xv[:args.limit_source], yv[:args.limit_source]
    if args.limit_target:
        xt = xt[:args.limit_target]
        xdv = xdv[:args.limit_target]
    adapt_labels = load_target_adapt_labels(SHIFT_FILES[args.shift], len(xt), byte=2)

    model = UDANet("cdan", proj_dim=args.proj_dim, map_seed=args.seed + 1000).to(device)
    n_loaded = load_b_e1_init(model, init_ckpt)
    n_params = count_params(model)
    n_disc = count_params(model.discriminator)

    disc_params = list(model.discriminator.parameters())
    disc_ids = {id(p) for p in disc_params}
    rest_params = [p for p in model.parameters() if p.requires_grad and id(p) not in disc_ids]
    opt = torch.optim.RMSprop([
        {"params": rest_params, "lr": args.lr},
        {"params": disc_params, "lr": args.lr * args.d_lr_mult},
    ])
    ce = nn.CrossEntropyLoss()

    monitor_x = xt[:min(args.monitor_n, len(xt))]
    diag_x = xt[:min(args.diag_n, len(xt))]
    diag_labels = adapt_labels[:min(args.diag_n, len(adapt_labels))]
    probe_batch = (torch.from_numpy(xs[:100]).to(device), torch.from_numpy(ys[:100]).to(device),
                   torch.from_numpy(xt[:100]).to(device)) if args.grad_probe_every > 0 else None

    prior_ema = torch.full((256,), 1.0 / 256.0, device=device)
    history = {"epochs": [], "cls_loss": [], "dom_loss": [], "adv_loss": [], "aux_loss": [], "reg_loss": [],
               "lambda_eff": [], "d_acc_src": [], "d_acc_tgt": [], "keep_frac": [],
               "prior_entropy": [], "prior_kl": [], "w_mean": [], "grad_cos": [], "grad_norm_cls": [],
               "grad_norm_adv": [], "val_acc": [], "attack_acc": [], "ge_probe": [],
               "epoch_seconds": [], "monitor": [], "collapse_flags": []}
    start_epoch = 0
    ckpt_path = model_dir / "ckpt.pt"
    hist_path = out_dir / "history.json"
    if ckpt_path.exists() and hist_path.exists():
        state = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        history = state["history"]
        start_epoch = state["epoch"]
        if "prior_ema" in state:
            prior_ema = state["prior_ema"].to(device)
        print(f"[resume] {run} from epoch {start_epoch}", flush=True)

    diag_before = domain_accuracy(model, xv, xdv, args.lambda_adv, device=device)
    diag_before["epoch"] = start_epoch
    print(json.dumps({"diag_before": diag_before}), flush=True)

    meta = {
        "run": run, "method": "cdan_sca_adv2", "base_method": "cdan_sca_full",
        "conditional_input": True, "entropy_weighting": True,
        "adv2_changes": {
            "consistency_gated_conditioning": bool(args.consistency),
            "consistency_views": 2, "aug_shift": args.aug_shift,
            "aug_amp": args.aug_amp, "aug_noise": args.aug_noise, "gate_tau": args.gate_tau,
            "keep_min_frac": args.keep_min_frac, "aux_full": args.aux_full,
            "detach_cond": bool(args.detach_cond), "clip_grad": args.clip_grad,
            "lambda_ramp_epochs": args.ramp_epochs,
            "target_prior_kl_weight": args.reg_weight,
            "prior_momentum": args.prior_momentum,
            "disc_lr_mult": args.d_lr_mult, "label_smooth": args.label_smooth,
        },
        "shift": args.shift, "seed": args.seed, "epochs_planned": args.epochs,
        "ramp_epochs": args.ramp_epochs, "reg_weight": args.reg_weight,
        "lr": args.lr, "optimizer": "RMSprop (two param groups: rest, discriminator)",
        "batch_size": args.batch_size, "lambda_adv": args.lambda_adv,
        "lambda_schedule": f"lambda_eff = lambda * min(1, epoch/{args.ramp_epochs})",
        "loss": "L_cls(src) + BCE_src + mean_keep[w(g_t) * BCE_tgt] + reg_weight * KL(p_bar || uniform)",
        "proj_dim": args.proj_dim, "normalize": "none",
        "init_ckpt": str(init_ckpt), "init_keys_loaded": n_loaded,
        "n_params_total": n_params, "n_params_discriminator": n_disc,
        "source_train": int(len(xs)), "source_val": int(len(xv)),
        "target_adapt": int(len(xt)), "target_domain_val": int(len(xdv)),
        "target_attack": int(len(xa)), "target_byte": 2,
        "real_key_byte": int(data["real_key_byte"]), "target_file": SHIFT_FILES[args.shift],
        "dataset_keys_equal": bool(data["keys_equal"]),
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "torch": torch.__version__,
        "protocol_note": "target_adapt 无标签；其标签仅训练后用于校准诊断；attack 轨迹只做诊断，不参与选模",
    }
    save_json(out_dir / "run_meta.json", meta)
    print(json.dumps({"start": {k: meta[k] for k in ("run", "shift", "seed", "n_params_total",
                                                     "lambda_adv", "ramp_epochs", "reg_weight")}}), flush=True)

    n = len(xs); bs = args.batch_size; nt = len(xt)
    t_start = time.time()
    for epoch in range(start_epoch + 1, args.epochs + 1):
        model.train()
        t0 = time.time()
        lam_eff = args.lambda_adv * min(1.0, epoch / max(1, args.ramp_epochs))
        perm = np.random.permutation(n)
        tgt_perm = np.random.permutation(nt); tgt_ptr = 0
        ep = {"cls": 0.0, "dom": 0.0, "adv": 0.0, "reg": 0.0, "dsrc": 0, "dtgt": 0,
              "cnt": 0, "keep": 0.0, "kcnt": 0, "w": 0.0, "H": 0.0, "kl": 0.0}
        for s in range(0, n, bs):
            idx = perm[s:s + bs]; k = len(idx)
            if tgt_ptr + k > nt:
                tgt_perm = np.random.permutation(nt); tgt_ptr = 0
            tix = tgt_perm[tgt_ptr:tgt_ptr + k]; tgt_ptr += k

            xb = torch.from_numpy(xs[idx]).to(device)
            yb = torch.from_numpy(ys[idx]).to(device)
            tb = torch.from_numpy(xt[tix]).to(device)

            tb_a = augment(tb, args.aug_shift, args.aug_amp, args.aug_noise)
            fs = model.features(xb)
            logits_s = model.logits_from_features(fs)
            ft = model.features(tb_a)
            logits_t = model.logits_from_features(ft)
            cls_loss = ce(logits_s, yb)

            # 一致性硬门控（不建图）：只让"两视图一致"的目标样本产生对抗梯度
            with torch.no_grad():
                probe_probs = torch.softmax(logits_t, dim=1)
                if args.consistency:
                    tb_b = augment(tb, args.aug_shift, args.aug_amp, args.aug_noise)
                    logits_t2 = model.logits_from_features(model.features(tb_b))
                    agree = probe_probs.argmax(1) == logits_t2.argmax(1)
                    if args.gate_tau > 0:
                        agree = agree & (probe_probs.max(1).values >= args.gate_tau)
                else:
                    agree = torch.ones(k, dtype=torch.bool, device=device)
                keep_frac = float(agree.float().mean().item())
                if keep_frac < args.keep_min_frac:
                    agree = torch.ones(k, dtype=torch.bool, device=device)
                    keep_frac = 1.0
                w_t = 1.0 - (-(probe_probs * torch.log(probe_probs.clamp_min(1e-12))).sum(1)) / LOG256

            d_s = domain_logits(model, fs, logits_s, lam_eff, args.detach_cond)
            d_t = domain_logits(model, ft[agree], logits_t[agree], lam_eff, args.detach_cond)
            bce_s = F.binary_cross_entropy_with_logits(
                d_s, torch.full_like(d_s, args.label_smooth))
            adv_t = F.binary_cross_entropy_with_logits(
                d_t, torch.full_like(d_t, 1.0 - args.label_smooth), reduction="none")
            adv_loss = (w_t[agree] * adv_t).mean()
            if args.aux_full > 0:
                d_aux = domain_logits(model, ft.detach(), logits_t.detach(), 0.0, True, False)
                aux_loss = F.binary_cross_entropy_with_logits(
                    d_aux, torch.full_like(d_aux, 1.0 - args.label_smooth))
            else:
                aux_loss = torch.zeros((), device=device)

            # 目标域先验（EMA，不建图）→ KL(p_bar || uniform) = log256 - H(p_bar)
            with torch.no_grad():
                p_bar_batch = probe_probs.mean(0)
                prior_ema.mul_(args.prior_momentum).add_(p_bar_batch * (1 - args.prior_momentum))
                prior_ema.clamp_min_(1e-8)
                prior_entropy = float(-(prior_ema * torch.log(prior_ema)).sum().item())
                prior_kl = float(LOG256 - prior_entropy)
            reg_loss = (prior_ema * (torch.log(prior_ema) - math.log(1.0 / 256.0))).sum()

            loss = cls_loss + bce_s + adv_loss + args.aux_full * aux_loss + args.reg_weight * reg_loss
            opt.zero_grad()
            loss.backward()
            if args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            opt.step()

            with torch.no_grad():
                ep["cls"] += float(cls_loss.item()) * k
                ep["dom"] += float(bce_s.item()) * k
                ep["adv"] += float(adv_loss.item()) * k
                ep["aux"] = ep.get("aux", 0.0) + float(aux_loss.item()) * k
                ep["reg"] += float(reg_loss.item()) * k
                ep["dsrc"] += int(((d_s > 0).logical_not()).sum().item())
                ep["dtgt"] += int((d_t > 0).sum().item())
                ep["keep"] += keep_frac * k
                ep["kcnt"] += k
                ep["w"] += float(w_t.mean().item()) * k
                ep["H"] += prior_entropy * k
                ep["kl"] += prior_kl * k
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

        gcos = gnorm_c = gnorm_a = None
        if probe_batch is not None and (epoch % args.grad_probe_every == 0 or epoch == args.epochs):
            model.train()
            gcos, gnorm_c, gnorm_a = grad_cosine(model, ce, probe_batch[0], probe_batch[1],
                                                 probe_batch[2], lam_eff)
            model.eval()

        history["epochs"].append(epoch)
        history["cls_loss"].append(ep["cls"] / ep["cnt"])
        history["dom_loss"].append(ep["dom"] / ep["cnt"])
        history["adv_loss"].append(ep["adv"] / ep["cnt"])
        history["aux_loss"].append(ep.get("aux", 0.0) / ep["cnt"])
        history["reg_loss"].append(ep["reg"] / ep["cnt"])
        history["lambda_eff"].append(lam_eff)
        history["d_acc_src"].append(ep["dsrc"] / ep["cnt"])
        history["d_acc_tgt"].append(ep["dtgt"] / ep["cnt"])
        history["keep_frac"].append(ep["keep"] / ep["kcnt"])
        history["prior_entropy"].append(ep["H"] / ep["cnt"])
        history["prior_kl"].append(ep["kl"] / ep["cnt"])
        history["w_mean"].append(ep["w"] / ep["cnt"])
        history["grad_cos"].append(gcos)
        history["grad_norm_cls"].append(gnorm_c)
        history["grad_norm_adv"].append(gnorm_a)
        history["val_acc"].append(float(val_acc))
        history["attack_acc"].append(None if attack_acc is None else float(attack_acc))
        history["ge_probe"].append(probe_rank)
        history["epoch_seconds"].append(sec)
        history["monitor"].append(mon)
        history["collapse_flags"].append(flag)
        save_json(hist_path, history)
        if epoch % args.save_every == 0 or epoch == args.epochs:
            torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                        "epoch": epoch, "history": history, "prior_ema": prior_ema.cpu()},
                       str(ckpt_path))
        rec = {"epoch": epoch, "cls": round(history["cls_loss"][-1], 4),
               "adv": round(history["adv_loss"][-1], 4),
               "reg": round(history["reg_loss"][-1], 4),
               "lam": round(lam_eff, 4),
               "d_src": round(history["d_acc_src"][-1], 3),
               "d_tgt": round(history["d_acc_tgt"][-1], 3),
               "keep": round(history["keep_frac"][-1], 3),
               "w": round(history["w_mean"][-1], 3),
               "Hp": round(history["prior_entropy"][-1], 3),
               "top1": round(mon["top1_share"], 3), "H": round(mon["pred_entropy_mean"], 3),
               "val_acc": round(history["val_acc"][-1], 4),
               "ge_probe": probe_rank, "gcos": None if gcos is None else round(gcos, 3),
               "flag": flag, "sec": round(sec, 1)}
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
    diag_after = domain_accuracy(model, xv, xdv, args.lambda_adv, device=device)
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
        "final_train_adv_loss": history["adv_loss"][-1],
        "final_val_acc": history["val_acc"][-1],
        "final_attack_acc": final_attack_acc,
        "final_monitor": history["monitor"][-1],
        "final_keep_frac": history["keep_frac"][-1],
        "final_prior_entropy": history["prior_entropy"][-1],
        "collapse_epochs": int(sum(1 for f in history["collapse_flags"] if f == "COLLAPSE")),
        "warn_epochs": int(sum(1 for f in history["collapse_flags"] if f == "WARN")),
        "max_top1_share": float(max(m["top1_share"] for m in history["monitor"])),
        "max_prior_kl": float(max(history["prior_kl"])),
        "final_grad_cos": next((g for g in reversed(history["grad_cos"]) if g is not None), None),
        "domain_diag_before": diag_before,
        "domain_diag_after": diag_after,
        "predictions_file": str(out_dir / "predictions_attack.npy"),
        "ckpt_file": str(ckpt_path),
    })
    save_json(out_dir / "run_meta.json", meta)
    print(json.dumps({"done": run, "sec": round(time.time() - t_start, 1),
                      "attack_acc": round(final_attack_acc, 4),
                      "top1": round(meta["final_monitor"]["top1_share"], 3),
                      "keep": round(history["keep_frac"][-1], 3),
                      "collapse_epochs": meta["collapse_epochs"],
                      "domain_after": diag_after}), flush=True)


if __name__ == "__main__":
    main()
