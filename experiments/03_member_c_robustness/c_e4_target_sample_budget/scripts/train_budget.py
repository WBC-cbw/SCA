#!/usr/bin/env python3
"""C-E4：target_adapt 样本量敏感性训练器（CDPA-MMD / DANN / CDAN-SCA full）。

唯一改变的自变量是 target_adapt 的条数（50 / 500 / 5000；45000 复用已有 run）。
其余全部冻结：source_train=fixed Profiling[0:45000]，source_val=[45000:50000]，
target_attack=desync Attack[0:10000]，OfficialCNN 主干，B-E1 同 seed 初始化，
RMSprop lr=1e-5，batch 200，75 epochs，原始 int8 输入。

样本量适配（关键实现）：
- 子集索引来自 configs/target_subsets.json（与各方法共享，同一 seed 完全相同）。
- 每个 epoch 的目标域 batch 数仍为 ceil(45000/200)=225，即"源域一个 epoch 的量"。
  子集小于 45000 时，按独立随机置换反复循环抽取目标索引并拼接截断到 45000，
  保证源域训练量与全量档完全一致，只有"见过的不同目标样本数"变化。
- 子集与 target_domain_val（Profiling[45000:50000]）不重叠，训练前断言检查。

监控（不改变损失）：每 epoch 在子集上算 top1 占比/预测熵/w 分布/collapse 级别；
每 2 epoch 500 条 attack GE 探针；结束时保存 attack 概率与 target_adapt 诊断 npz。

用法：
  python scripts/train_budget.py --method cdpa --shift desync50 --budget 50 --seed 42
  python scripts/train_budget.py --method dann --budget 500 --seed 43 --epochs 2 \
      --limit-source 2000 --tag smoke
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parents[1]          # .../c_e4_target_sample_budget
ROOT = HERE.parents[2]                                # .../sca_repo
sys.path.insert(0, str(ROOT / "experiments/01_member_a_baselines/a_e2_uda_baselines/scripts"))
sys.path.insert(0, str(ROOT / "experiments/01_member_a_baselines/a_e3_cdpa_or_adabn/scripts"))
sys.path.insert(0, str(ROOT / "experiments/02_member_b_method/b_e3_cdan_sca/scripts"))

import uda_common as UDA  # noqa: E402
import cdpa_common as CDPA  # noqa: E402
from cdan_sca_common import (LOG256, calibration_stats, collapse_level,  # noqa: E402
                             entropy_nats, entropy_weight, load_target_adapt_labels,
                             monitor_stats)

SHIFT_FILES = {
    "desync50": "D:/sca_project/ASCAD_data/ASCAD_desync50.h5",
    "desync100": "D:/sca_project/ASCAD_data/ASCAD_desync100.h5",
}
SUBSET_FILE = HERE / "configs/target_subsets.json"
TAG = {"cdpa": "cdpa", "dann": "dann", "cdan_sca": "cdan_sca"}
TARGET_POOL = 45000


def ge_curve(probs, plaintext, key, byte, step=100):
    from importlib.util import module_from_spec, spec_from_file_location
    spec = spec_from_file_location(
        "ev", str(ROOT / "experiments/cdan-sca-diagnosis/scripts/evaluate_ascad.py"))
    ev = module_from_spec(spec)
    spec.loader.exec_module(ev)
    return ev.ranks(probs, plaintext, key, byte, step)


def load_subset(shift, seed, budget):
    obj = json.loads(SUBSET_FILE.read_text(encoding="utf-8"))
    idx = np.asarray(obj["subsets"][shift][f"s{seed}"][str(budget)], dtype=np.int64)
    assert len(idx) == budget, (shift, seed, budget, len(idx))
    assert len(np.unique(idx)) == budget
    assert int(idx.min()) >= 0 and int(idx.max()) < TARGET_POOL, "子集侵入 target_domain_val"
    sha = hashlib.sha256(SUBSET_FILE.read_bytes()).hexdigest()
    return idx, sha


def epoch_target_indices(nt, n_needed):
    """独立随机置换循环拼接，返回长度 n_needed 的目标索引序列（与源 batch 对齐）。"""
    chunks, got = [], 0
    while got < n_needed:
        p = np.random.permutation(nt)
        chunks.append(p)
        got += nt
    return np.concatenate(chunks)[:n_needed]


def make_meta(args, run, idx, subset_sha, data, xs, xt, xa, init_ckpt, n_loaded,
              n_params, extra=None):
    meta = {
        "run": run, "method": args.method, "shift": args.shift, "seed": args.seed,
        "budget": int(args.budget),
        "epochs_planned": args.epochs, "lr": args.lr, "optimizer": "RMSprop",
        "batch_size": args.batch_size, "normalize": "none（原始 int8 幅度）",
        "epochs_batches_per_epoch": int(np.ceil(len(xs) / args.batch_size)),
        "target_samples_per_epoch": int(np.ceil(len(xs) / args.batch_size) * args.batch_size),
        "target_batch_cycles_per_epoch": int(np.ceil(
            np.ceil(len(xs) / args.batch_size) * args.batch_size / max(len(xt), 1))),
        "target_batch_note": "目标域每 epoch 仍取 ceil(45000/200)=225 个 batch；子集小于该量时按独立随机置换循环采样",
        "subset_file": str(SUBSET_FILE), "subset_sha256": subset_sha,
        "subset_indices_preview": [int(v) for v in idx[:10]],
        "subset_indices_sha256": hashlib.sha256(idx.tobytes()).hexdigest(),
        "n_adapt": int(len(xt)), "target_pool": TARGET_POOL,
        "target_domain_val": int(len(data["target_domain_val_x"])),
        "target_domain_val_note": "Profiling[45000:50000]，只做域判别诊断，不参与训练/评估",
        "source_train": int(len(xs)), "source_val": int(len(data["source_val_x"])),
        "target_attack": int(len(xa)), "target_byte": 2,
        "real_key_byte": int(data["real_key_byte"]),
        "dataset_keys_equal": bool(data["keys_equal"]),
        "target_file": SHIFT_FILES[args.shift],
        "init_ckpt": str(init_ckpt), "init_keys_loaded": n_loaded,
        "n_params_total": n_params,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "torch": torch.__version__,
        "tf32": bool(torch.backends.cuda.matmul.allow_tf32) if torch.cuda.is_available() else None,
        "target_adapt_labels": "只用于训练结束后的 ECE / confident-wrong 诊断，不参与损失或模型选择",
        "cost_accounting": {
            "n_attack_at_ge1": "首个 GE=0 的攻击轨迹数",
            "n_total_at_ge1": "budget + n_attack_at_ge1（适应轨迹是否计入总成本的双口径之一）",
            "censor": "10000 条内未出现 GE=0 记 None，聚合时截尾 10000（NTGE）/10000+budget（N_total）",
        },
        "reference_full_budget": "budget=45000 不重跑，复用 A-E3 CDPA / A-E2 DANN / B-E3 CDAN-SCA full 的 predictions",
    }
    if extra:
        meta.update(extra)
    return meta


def finalize_common(args, run, out_dir, model, data, xt_diag, diag_labels, meta,
                    history, predict_fn, ckpt_path, t_start):
    xa = data["attack_x"]
    ya = data["attack_true_label"]
    probs_attack = predict_fn(model, xa)
    np.save(str(out_dir / "predictions_attack.npy"), probs_attack)
    probs_diag = predict_fn(model, xt_diag)
    np.savez_compressed(str(out_dir / "diag_target_adapt.npz"),
                        probs=probs_diag.astype(np.float32),
                        labels=diag_labels.astype(np.int64),
                        entropy=entropy_nats(probs_diag).astype(np.float32),
                        w=entropy_weight(probs_diag).astype(np.float32))
    diag_stats = calibration_stats(probs_diag, diag_labels)
    diag_stats["subset"] = f"target_adapt subset (budget={args.budget}, n={len(xt_diag)})"
    diag_stats["note"] = "标签仅用于事后诊断"
    UDA.save_json(out_dir / "entropy_diag.json", diag_stats)

    final_attack_acc = float((probs_attack.argmax(1) == ya).mean())
    mon = history["monitor"][-1] if history.get("monitor") else None
    meta.update({
        "epochs_done": history["epochs"][-1],
        "train_seconds": float(sum(history["epoch_seconds"])),
        "wall_seconds_total": time.time() - t_start,
        "sec_per_epoch_mean": float(np.mean(history["epoch_seconds"])),
        "final_train_cls_loss": history["cls_loss"][-1],
        "final_val_acc": history["val_acc"][-1],
        "final_attack_acc": final_attack_acc,
        "final_monitor": mon,
        "collapse_epochs": int(sum(1 for f in history["collapse_flags"] if f == "COLLAPSE")),
        "warn_epochs": int(sum(1 for f in history["collapse_flags"] if f == "WARN")),
        "max_top1_share": float(max(m["top1_share"] for m in history["monitor"])),
        "max_memory_allocated_mb": round(torch.cuda.max_memory_allocated() / 1e6, 1)
        if torch.cuda.is_available() else None,
        "predictions_file": str(out_dir / "predictions_attack.npy"),
        "diag_file": str(out_dir / "diag_target_adapt.npz"),
        "ckpt_file": str(ckpt_path),
    })
    UDA.save_json(out_dir / "run_meta.json", meta)
    print(json.dumps({"done": run, "sec": round(time.time() - t_start, 1),
                      "attack_acc": round(final_attack_acc, 4),
                      "top1": None if mon is None else round(mon["top1_share"], 3),
                      "collapse_epochs": meta["collapse_epochs"]}), flush=True)


def run_adversarial(args, kind):
    """DANN（kind=dann）或 CDAN-SCA full（kind=cdan_sca）。损失与 A-E2/B-E3 完全一致。"""
    run = f"{TAG[args.method]}_{args.shift}_b{args.budget}_s{args.seed}" + (
        f"_{args.tag}" if args.tag else "")
    out_dir = HERE / "results" / run
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = ROOT / "outputs/models/c_e4" / run
    model_dir.mkdir(parents=True, exist_ok=True)
    init_ckpt = Path(args.init_ckpt) if args.init_ckpt else UDA.B_E1_MODELS / f"cnn_s{args.seed}" / "ckpt.pt"
    idx, subset_sha = load_subset(args.shift, args.seed, args.budget)

    data = UDA.load_domain_data("D:/sca_project/ASCAD_data/ASCAD.h5", SHIFT_FILES[args.shift],
                                n_source_train=args.n_source_train,
                                n_target_adapt=TARGET_POOL)
    xs, ys = data["source_train_x"], data["source_train_y"]
    xv, yv = data["source_val_x"], data["source_val_y"]
    xa, ya = data["attack_x"], data["attack_true_label"]
    xdv = data["target_domain_val_x"]
    if args.limit_source:
        xs, ys = xs[:args.limit_source], ys[:args.limit_source]
        xv, yv = xv[:args.limit_source], yv[:args.limit_source]
    xt = data["target_adapt_x"][idx].copy()
    all_labels = load_target_adapt_labels(SHIFT_FILES[args.shift], TARGET_POOL, byte=2)
    adapt_labels = all_labels[idx].copy()
    assert len(xt) == args.budget

    monitor_x = xt[:min(2000, len(xt))]
    diag_x = xt[:min(5000, len(xt))]
    diag_labels = adapt_labels[:min(5000, len(adapt_labels))]

    model = UDA.UDANet("cdan" if kind == "cdan_sca" else "dann", proj_dim=args.proj_dim,
                       map_seed=args.seed + 1000).to("cuda" if torch.cuda.is_available() else "cpu")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_loaded = UDA.load_b_e1_init(model, init_ckpt)
    n_params = UDA.count_params(model)
    opt = torch.optim.RMSprop(model.parameters(), lr=args.lr)
    ce = nn.CrossEntropyLoss()
    bce = nn.BCEWithLogitsLoss()
    lam = args.lambda_adv

    history = {"epochs": [], "cls_loss": [], "dom_loss": [], "d_acc_src": [], "d_acc_tgt": [],
               "val_acc": [], "attack_acc": [], "ge_probe": [], "epoch_seconds": [],
               "monitor": [], "collapse_flags": []}
    start_epoch = 0
    ckpt_path = model_dir / "ckpt.pt"
    hist_path = out_dir / "history.json"
    resumed = False
    if ckpt_path.exists() and hist_path.exists():
        state = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        history = state["history"]
        start_epoch = state["epoch"]
        resumed = True
        print(f"[resume] {run} from epoch {start_epoch}", flush=True)

    meta = make_meta(args, run, idx, subset_sha, data, xs, xt, xa, init_ckpt, n_loaded,
                     n_params, extra={
                         "conditional_input": kind == "cdan_sca",
                         "entropy_weighting": kind == "cdan_sca",
                         "entropy_weight_formula": "w=1-H/log(256)，仅目标域对抗项，w detached" if kind == "cdan_sca" else None,
                         "lambda_adv": lam, "proj_dim": args.proj_dim if kind == "cdan_sca" else None,
                         "loss": ("CE(src)+BCE_src+mean_t[w*BCE_tgt]" if kind == "cdan_sca"
                                  else "CE(src)+BCE_src+BCE_tgt（判别器输入为 L2 缩放特征）"),
                         "resumed_at_start": resumed,
                     })
    UDA.save_json(out_dir / "run_meta.json", meta)
    diag_before = UDA.domain_accuracy(model, xv, xdv, lam, device=device)
    diag_before["epoch"] = start_epoch
    print(json.dumps({"start": {k: meta[k] for k in ("run", "method", "budget", "n_adapt",
                                                     "n_params_total", "subset_sha256")},
                      "diag_before": diag_before}), flush=True)

    n, bs, nt = len(xs), args.batch_size, len(xt)
    n_needed = int(np.ceil(n / bs) * bs)
    t_start = time.time()
    for epoch in range(start_epoch + 1, args.epochs + 1):
        model.train()
        t0 = time.time()
        perm = np.random.permutation(n)
        tgt_all = epoch_target_indices(nt, n_needed)
        ep = {"cls": 0.0, "dom": 0.0, "dsrc": 0, "dtgt": 0, "cnt": 0, "w": 0.0}
        for bi, s in enumerate(range(0, n, bs)):
            idxb = perm[s:s + bs]
            k = len(idxb)
            tix = tgt_all[bi * bs:bi * bs + k]
            xb = torch.from_numpy(xs[idxb]).to(device)
            yb = torch.from_numpy(ys[idxb]).to(device)
            tb = torch.from_numpy(xt[tix]).to(device)

            fs = model.features(xb)
            logits_s = model.logits_from_features(fs)
            ft = model.features(tb)
            logits_t = model.logits_from_features(ft)
            cls_loss = ce(logits_s, yb)
            d_s = model.domain_logits(fs, logits_s, lam)
            d_t = model.domain_logits(ft, logits_t, lam)
            bce_s = bce(d_s, torch.zeros(k, device=device))
            if kind == "cdan_sca":
                with torch.no_grad():
                    probs_t = torch.softmax(logits_t.detach(), dim=1)
                    w_t = 1.0 - (-(probs_t * torch.log(probs_t.clamp_min(1e-12))).sum(1)) / LOG256
                bce_t = F.binary_cross_entropy_with_logits(
                    d_t, torch.ones(k, device=device), reduction="none")
                loss = cls_loss + bce_s + (w_t * bce_t).mean()
                dom_term = bce_s.item() + (w_t * bce_t).mean().item()
                ep["w"] += float(w_t.mean().item()) * k
            else:
                bce_t = bce(d_t, torch.ones(k, device=device))
                loss = cls_loss + bce_s + bce_t
                dom_term = bce_s.item() + bce_t.item()

            opt.zero_grad()
            loss.backward()
            opt.step()

            ep["cls"] += float(cls_loss.item()) * k
            ep["dom"] += float(dom_term) * k
            ep["dsrc"] += int(((d_s > 0).logical_not()).sum().item())
            ep["dtgt"] += int((d_t > 0).sum().item())
            ep["cnt"] += k

        model.eval()
        val_acc = UDA.accuracy(model, xv, yv, device=device)
        attack_acc = None
        if epoch % args.attack_acc_every == 0 or epoch == args.epochs:
            attack_acc = UDA.accuracy(model, xa, ya, device=device)
        probe_rank = None
        if epoch % args.probe_every == 0 or epoch == args.epochs:
            probs_p = UDA.predict_probs(model, xa[:args.probe_traces], device=device)
            curve = ge_curve(probs_p, data["attack_plaintext"][:args.probe_traces],
                             data["attack_key"], 2, step=100)
            probe_rank = curve[-1][1] if curve else None
        mon = monitor_stats(UDA.predict_probs(model, monitor_x, device=device))
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
        UDA.save_json(hist_path, history)
        if epoch % args.save_every == 0 or epoch == args.epochs:
            torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                        "epoch": epoch, "history": history}, str(ckpt_path))
        rec = {"epoch": epoch, "cls": round(history["cls_loss"][-1], 4),
               "dom": round(history["dom_loss"][-1], 4),
               "d_src": round(history["d_acc_src"][-1], 3),
               "d_tgt": round(history["d_acc_tgt"][-1], 3),
               "w": round(ep["w"] / ep["cnt"], 3) if kind == "cdan_sca" else None,
               "top1": round(mon["top1_share"], 3), "H": round(mon["pred_entropy_mean"], 3),
               "val_acc": round(history["val_acc"][-1], 4),
               "ge_probe": probe_rank, "flag": flag, "sec": round(sec, 1)}
        print(json.dumps(rec), flush=True)
        if flag != "OK":
            print(f"[{flag}] epoch {epoch} top1_share={mon['top1_share']:.3f} "
                  f"top1_class={mon['top1_class']} H={mon['pred_entropy_mean']:.3f}", flush=True)

    diag_after = UDA.domain_accuracy(model, xv, xdv, lam, device=device)
    diag_after["epoch"] = history["epochs"][-1]
    UDA.save_json(out_dir / "domain_diag.json",
                  {"before": diag_before, "after": diag_after,
                   "diag_sets": {"source": "fixed Profiling[45000:50000]",
                                 "target": "desync Profiling[45000:50000]"}})
    finalize_common(args, run, out_dir, model, data, diag_x, diag_labels, meta, history,
                    lambda m, x: UDA.predict_probs(m, x, device=device), ckpt_path, t_start)


def run_cdpa(args):
    run = f"cdpa_{args.shift}_b{args.budget}_s{args.seed}" + (f"_{args.tag}" if args.tag else "")
    out_dir = HERE / "results" / run
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = ROOT / "outputs/models/c_e4" / run
    model_dir.mkdir(parents=True, exist_ok=True)
    init_ckpt = Path(args.init_ckpt) if args.init_ckpt else CDPA.B_E1_MODELS / f"cnn_s{args.seed}" / "ckpt.pt"
    idx, subset_sha = load_subset(args.shift, args.seed, args.budget)
    layers = tuple(s.strip() for s in args.mmd_layers.split(",") if s.strip())
    probe_epochs = {int(s) for s in args.grad_probe_epochs.split(",") if s.strip()}

    data = CDPA.load_domain_data("D:/sca_project/ASCAD_data/ASCAD.h5", SHIFT_FILES[args.shift],
                                 n_source_train=args.n_source_train,
                                 n_target_adapt=TARGET_POOL)
    xs, ys = data["source_train_x"], data["source_train_y"]
    xv, yv = data["source_val_x"], data["source_val_y"]
    xa, ya = data["attack_x"], data["attack_true_label"]
    if args.limit_source:
        xs, ys = xs[:args.limit_source], ys[:args.limit_source]
        xv, yv = xv[:args.limit_source], yv[:args.limit_source]
    xt = data["target_adapt_x"][idx].copy()
    all_labels = load_target_adapt_labels(SHIFT_FILES[args.shift], TARGET_POOL, byte=2)
    adapt_labels = all_labels[idx].copy()
    assert len(xt) == args.budget

    monitor_x = xt[:min(2000, len(xt))]
    diag_x = xt[:min(5000, len(xt))]
    diag_labels = adapt_labels[:min(5000, len(adapt_labels))]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CDPA.CDPANet(num_classes=256).to(device)
    n_loaded = CDPA.load_b_e1_init(model, init_ckpt)
    n_params = CDPA.count_params(model)
    opt = torch.optim.RMSprop(model.parameters(), lr=args.lr)
    ce = nn.CrossEntropyLoss()
    lam = args.lambda_mmd

    history = {"epochs": [], "cls_loss": [], "mmd_loss": [], "mmd_conv": [], "mmd_fc1": [],
               "mmd_fc2": [], "val_acc": [], "attack_acc": [], "ge_probe": [],
               "epoch_seconds": [], "monitor": [], "collapse_flags": []}
    start_epoch = 0
    ckpt_path = model_dir / "ckpt.pt"
    hist_path = out_dir / "history.json"
    resumed = False
    if ckpt_path.exists() and hist_path.exists():
        state = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        history = state["history"]
        start_epoch = state["epoch"]
        resumed = True
        print(f"[resume] {run} from epoch {start_epoch}", flush=True)

    meta = make_meta(args, run, idx, subset_sha, data, xs, xt, xa, init_ckpt, n_loaded,
                     n_params, extra={
                         "lambda_mmd": lam, "mmd_layers": list(layers),
                         "mmd_kernel": {"kernel_mul": 2.0, "kernel_num": 5,
                                        "bandwidth": "批内平均平方距离 / kernel_mul**(kernel_num//2)（官方实现）"},
                         "loss": "CE(source) + lambda_mmd * sum_layers MMD(source, target)",
                         "resumed_at_start": resumed,
                     })
    CDPA.save_json(out_dir / "run_meta.json", meta)
    print(json.dumps({"start": {k: meta[k] for k in ("run", "method", "budget", "n_adapt",
                                                     "n_params_total", "subset_sha256")}}), flush=True)

    n, bs, nt = len(xs), args.batch_size, len(xt)
    n_needed = int(np.ceil(n / bs) * bs)
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
        tgt_all = epoch_target_indices(nt, n_needed)
        ep = {"cls": 0.0, "mmd": 0.0, "conv": 0.0, "fc1": 0.0, "fc2": 0.0, "cnt": 0}
        probe_batch = None
        for bi, s in enumerate(range(0, n, bs)):
            idxb = perm[s:s + bs]
            k = len(idxb)
            tix = tgt_all[bi * bs:bi * bs + k]
            xb = torch.from_numpy(xs[idxb]).to(device)
            yb = torch.from_numpy(ys[idxb]).to(device)
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
            probe = CDPA.grad_norm_ratio(model, *probe_batch, lam, layers, ce)
            probe["epoch"] = epoch
            probe_history.append({"lambda_mmd": lam, **probe})
            CDPA.save_json(out_dir / "grad_probe.json", probe_history)
            print(json.dumps({"grad_probe": probe}), flush=True)

        model.eval()
        val_acc = CDPA.accuracy(model, xv, yv, device=device)
        attack_acc = None
        if epoch % args.attack_acc_every == 0 or epoch == args.epochs:
            attack_acc = CDPA.accuracy(model, xa, ya, device=device)
        probe_rank = None
        if epoch % args.probe_every == 0 or epoch == args.epochs:
            probs_p = CDPA.predict_probs(model, xa[:args.probe_traces], device=device)
            curve = ge_curve(probs_p, data["attack_plaintext"][:args.probe_traces],
                             data["attack_key"], 2, step=100)
            probe_rank = curve[-1][1] if curve else None
        mon = monitor_stats(CDPA.predict_probs(model, monitor_x, device=device))
        flag = collapse_level(mon["top1_share"])
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
        history["monitor"].append(mon)
        history["collapse_flags"].append(flag)
        CDPA.save_json(hist_path, history)
        if epoch % args.save_every == 0 or epoch == args.epochs:
            torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                        "epoch": epoch, "history": history}, str(ckpt_path))
        rec = {"epoch": epoch, "cls": round(history["cls_loss"][-1], 4),
               "mmd": round(history["mmd_loss"][-1], 6),
               "mmd_conv": None if history["mmd_conv"][-1] is None else round(history["mmd_conv"][-1], 6),
               "top1": round(mon["top1_share"], 3), "H": round(mon["pred_entropy_mean"], 3),
               "val_acc": round(history["val_acc"][-1], 4),
               "ge_probe": probe_rank, "flag": flag, "sec": round(sec, 1)}
        print(json.dumps(rec), flush=True)
        if flag != "OK":
            print(f"[{flag}] epoch {epoch} top1_share={mon['top1_share']:.3f} "
                  f"top1_class={mon['top1_class']} H={mon['pred_entropy_mean']:.3f}", flush=True)

    finalize_common(args, run, out_dir, model, data, diag_x, diag_labels, meta, history,
                    lambda m, x: CDPA.predict_probs(m, x, device=device), ckpt_path, t_start)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=sorted(TAG))
    ap.add_argument("--shift", required=True, choices=sorted(SHIFT_FILES))
    ap.add_argument("--budget", type=int, required=True, choices=[50, 500, 5000])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=75)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch-size", type=int, default=200)
    ap.add_argument("--lambda-adv", type=float, default=0.1)
    ap.add_argument("--lambda-mmd", type=float, default=0.1)
    ap.add_argument("--mmd-layers", default=",".join(CDPA.MMD_LAYERS_DEFAULT))
    ap.add_argument("--proj-dim", type=int, default=1024)
    ap.add_argument("--n-source-train", type=int, default=45000)
    ap.add_argument("--limit-source", type=int, default=0)
    ap.add_argument("--probe-traces", type=int, default=500)
    ap.add_argument("--probe-every", type=int, default=2)
    ap.add_argument("--attack-acc-every", type=int, default=5)
    ap.add_argument("--grad-probe-epochs", default="1,20,50")
    ap.add_argument("--init-ckpt", default="")
    ap.add_argument("--save-every", type=int, default=5)
    ap.add_argument("--tag", default="")
    ap.add_argument("--no-tf32", action="store_true")
    args = ap.parse_args()

    tag_sfx = f"_{args.tag}" if args.tag else ""
    pre = "cdpa" if args.method == "cdpa" else TAG[args.method]
    pre_run = f"{pre}_{args.shift}_b{args.budget}_s{args.seed}{tag_sfx}"
    if not args.tag and (HERE / "results" / pre_run / "predictions_attack.npy").exists():
        print(f"[skip] {pre_run}: predictions_attack.npy exists")
        return

    UDA.seed_everything(args.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        if not args.no_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    if args.method == "cdpa":
        run_cdpa(args)
    else:
        run_adversarial(args, kind=args.method)


if __name__ == "__main__":
    main()
