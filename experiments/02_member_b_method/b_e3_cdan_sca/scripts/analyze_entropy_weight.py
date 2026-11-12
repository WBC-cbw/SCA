#!/usr/bin/env python3
"""B-E3 熵权重诊断：对五个方法统一计算 target_attack 与 target_adapt 上的
accuracy / ECE(15 bin) / confident-wrong / 熵 / 熵权重 w / top1 类别占比。

- target_attack：直接读各 run 的 predictions_attack.npy（CDAN/CDPA/SO/DANN 复用已有预测）。
- target_adapt：A-E2/A-E3 未保存适应集预测，本脚本用它们的 checkpoint 在**固定 5000 条**
  target_adapt 子集上重建预测（只做诊断，不训练、不选模型）。full/marginal 直接用训练时
  保存的 results/<run>/diag_target_adapt.npz。
- 标签仅在诊断中使用：target_adapt 标签由 desync Profiling metadata 现算。

输出：
  results/adapt_diag/<run>.npz           （baseline 的适应集预测）
  results/entropy_weight_analysis.csv    （逐 run：attack + adapt 两行）
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "experiments/01_member_a_baselines/a_e2_uda_baselines/scripts"))
sys.path.insert(0, str(HERE / "scripts"))
sys.path.insert(0, str(ROOT / "experiments/01_member_a_baselines/a_e3_cdpa_or_adabn/scripts"))

from uda_common import UDANet, OfficialCNN, load_domain_data  # noqa: E402
from cdan_sca_common import (SBOX, calibration_stats, entropy_nats, entropy_weight,  # noqa: E402
                             load_target_adapt_labels, monitor_stats)
from cdpa_common import CDPANet, predict_probs as cdpa_predict  # noqa: E402

DATA = Path("D:/sca_project/ASCAD_data")
SHIFTS = {"desync50": DATA / "ASCAD_desync50.h5", "desync100": DATA / "ASCAD_desync100.h5"}
SEEDS = [42, 43, 44]
DIAG_N = 5000
B_E1 = ROOT / "outputs/models/b_e1"
A_E2 = ROOT / "outputs/models/a_e2"
A_E3 = ROOT / "outputs/models/a_e3"
OUT = HERE / "results/adapt_diag"
CSV = HERE / "results/entropy_weight_analysis.csv"

METHODS = ["source_only", "dann", "cdan", "cdpa_mmd", "cdan_sca_full", "cdan_sca_marg"]
LABELS = {"source_only": "Source Only", "dann": "DANN", "cdan": "CDAN",
          "cdpa_mmd": "CDPA-MMD", "cdan_sca_full": "CDAN-SCA full",
          "cdan_sca_marg": "Abl.1 marginal+w"}


def load_attack_labels(target_h5: Path, byte=2):
    with h5py.File(target_h5, "r") as h:
        md = np.asarray(h["Attack_traces/metadata"])
    pt = np.asarray(md["plaintext"]); key = np.asarray(md["key"])
    return SBOX[np.bitwise_xor(pt[:, byte], np.uint8(int(key[0, byte])))].astype(np.int64)


def adapt_subset(target_h5: Path, n=DIAG_N):
    with h5py.File(target_h5, "r") as h:
        x = h["Profiling_traces/traces"][:n].astype(np.float32)[:, None, :]
        md = h["Profiling_traces/metadata"][:n]
        key = np.asarray(h["Profiling_traces/metadata"]["key"])
    pt = np.asarray(md["plaintext"])
    y = SBOX[np.bitwise_xor(pt[:, 2], np.uint8(int(key[0, 2])))].astype(np.int64)
    return x, y


@torch.no_grad()
def probs_of_uda(method, ckpt, x, device):
    model = UDANet(method, proj_dim=1024, map_seed=0).to(device)
    sd = torch.load(str(ckpt), map_location=device, weights_only=False)["model"]
    model.load_state_dict(sd)
    model.eval()
    out = []
    for s in range(0, len(x), 1000):
        xb = torch.from_numpy(x[s:s + 1000]).to(device)
        logits, _ = model(xb)
        out.append(torch.softmax(logits, 1).cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def probs_of_so(ckpt, x, device):
    model = OfficialCNN(num_classes=256).to(device)
    model.load_state_dict(torch.load(str(ckpt), map_location=device, weights_only=True))
    model.eval()
    out = []
    for s in range(0, len(x), 1000):
        xb = torch.from_numpy(x[s:s + 1000]).to(device)
        out.append(torch.softmax(model(xb), 1).cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def probs_of_cdpa(ckpt, x, device):
    model = CDPANet().to(device)
    sd = torch.load(str(ckpt), map_location=device, weights_only=False)["model"]
    model.load_state_dict(sd)
    model.eval()
    return cdpa_predict(model, x, device=device)


def rec(method, shift, seed, side, probs, labels):
    st = calibration_stats(probs, labels)
    w = entropy_weight(probs)
    st.update({"method": method, "label": LABELS[method], "shift": shift, "seed": seed,
               "side": side, "w_mean": float(w.mean()), "w_p50": float(np.median(w)),
               "w_p10": float(np.percentile(w, 10)), "w_p90": float(np.percentile(w, 90))})
    return st


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    attack_labels = {s: load_attack_labels(p) for s, p in SHIFTS.items()}
    adapt = {s: adapt_subset(p) for s, p in SHIFTS.items()}

    for shift in SHIFTS:
        xa, ya = adapt[shift]
        for seed in SEEDS:
            # ---- 已保存的 attack 预测
            paths = {
                "source_only": ROOT / "experiments/02_member_b_method/b_e2_architecture_shift/results/predictions" / shift / f"cnn_s{seed}.npy",
                "dann": ROOT / "experiments/01_member_a_baselines/a_e2_uda_baselines/results" / f"dann_{shift}_s{seed}/predictions_attack.npy",
                "cdan": ROOT / "experiments/01_member_a_baselines/a_e2_uda_baselines/results" / f"cdan_{shift}_s{seed}/predictions_attack.npy",
                "cdpa_mmd": ROOT / "experiments/01_member_a_baselines/a_e3_cdpa_or_adabn/results" / f"cdpa_{shift}_s{seed}/predictions_attack.npy",
            }
            for method, p in paths.items():
                if p.exists():
                    pr = np.load(p).astype(np.float32)
                    rows.append(rec(method, shift, seed, "attack", pr, attack_labels[shift]))
            for method, tag in [("cdan_sca_full", "full"), ("cdan_sca_marg", "marg")]:
                run = f"cdan_sca_{tag}_{shift}_s{seed}"
                d = HERE / "results" / run
                pa = np.load(d / "predictions_attack.npy").astype(np.float32)
                rows.append(rec(method, shift, seed, "attack", pa, attack_labels[shift]))
                npz = np.load(d / "diag_target_adapt.npz")
                rows.append(rec(method, shift, seed, "adapt", npz["probs"].astype(np.float32),
                                npz["labels"]))

            # ---- baseline 的 target_adapt 预测（用 checkpoint 重建，只做诊断）
            baseline_ckpt = {
                "source_only": (B_E1 / f"cnn_s{seed}/ckpt.pt", "so"),
                "dann": (A_E2 / f"dann_{shift}_s{seed}/ckpt.pt", "uda"),
                "cdan": (A_E2 / f"cdan_{shift}_s{seed}/ckpt.pt", "uda"),
                "cdpa_mmd": (A_E3 / f"cdpa_{shift}_s{seed}/ckpt.pt", "cdpa"),
            }
            for method, (ck, kind) in baseline_ckpt.items():
                if not ck.exists():
                    print(f"[skip adapt diag] {method} {shift} s{seed}: {ck}")
                    continue
                out_npz = OUT / f"{method}_{shift}_s{seed}.npz"
                if out_npz.exists():
                    z = np.load(out_npz)
                    pr = z["probs"].astype(np.float32)
                else:
                    if kind == "so":
                        pr = probs_of_so(ck, xa, device)
                    elif kind == "uda":
                        pr = probs_of_uda("dann" if method == "dann" else "cdan", ck, xa, device)
                    else:
                        pr = probs_of_cdpa(ck, xa, device)
                    np.savez_compressed(out_npz, probs=pr.astype(np.float32), labels=ya)
                rows.append(rec(method, shift, seed, "adapt", pr, ya))
            print(f"[done] {shift} s{seed}", flush=True)

    cols = ["method", "label", "shift", "seed", "side", "n", "accuracy", "ece_15",
            "frac_confwrong_ge_0.9", "frac_conf_ge_0.9", "frac_confwrong_ge_0.99",
            "frac_conf_lt_0.1", "mean_entropy_nats", "mean_max_softmax",
            "w_mean", "w_p10", "w_p50", "w_p90", "top1_share", "top1_class",
            "unique_pred_classes"]
    with CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in cols})
    print(f"wrote {len(rows)} rows -> {CSV}")
    for side in ("attack", "adapt"):
        print(f"\n=== {side} ===")
        for shift in SHIFTS:
            for method in METHODS:
                rs = [r for r in rows if r["side"] == side and r["shift"] == shift and r["method"] == method]
                if not rs:
                    continue
                print(f"{shift:9s} {LABELS[method]:20s} acc {np.mean([r['accuracy'] for r in rs]):.4f} "
                      f"ECE {np.mean([r['ece_15'] for r in rs]):.3f} "
                      f"cw {np.mean([r['frac_confwrong_ge_0.9'] for r in rs]):.3f} "
                      f"H {np.mean([r['mean_entropy_nats'] for r in rs]):.3f} "
                      f"w {np.mean([r['w_mean'] for r in rs]):.3f} "
                      f"top1 {np.mean([r['top1_share'] for r in rs]):.3f}")


if __name__ == "__main__":
    main()
