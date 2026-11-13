"""补充诊断（非训练，不产生新 run）：温度到底拟合到了什么？

B-E5 在 source_val（Profiling[45000:50000]，5000 条）上拟合温度，得到 T≈10.9–14.5。
本脚本回答：该 T 是模型属性还是 split 属性？
- 在 source_train 的 5000 条随机子集与 source_val 上分别拟合 T，比较 T、NLL、ECE、accuracy；
- 交叉应用 T（train-fit T 用到 val，val-fit T 用到 train）。

用法：python scripts/diagnose_temperature_transfer.py [--device cpu|cuda]
输出：results/temperature_transfer_diagnostic.json
"""
import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "experiments/01_member_a_baselines/a_e2_uda_baselines/scripts"))
sys.path.insert(0, str(HERE / "scripts"))
import uda_common as U  # noqa: E402
import calib_common as C  # noqa: E402

ASCAD = Path("D:/sca_project/ASCAD_data/ASCAD.h5")
CKPT = REPO / "outputs/models/b_e1/cnn_s42/ckpt.pt"
OUT = HERE / "results/temperature_transfer_diagnostic.json"


def logits_for(model, x, device, batch=500):
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(x), batch):
            xb = torch.from_numpy(x[s:s + batch]).to(device)
            out.append(model(xb).cpu().numpy())
    return np.concatenate(out)


def summarize(logits, labels, T, name):
    probs = C.softmax_np(logits)
    probs_t = C.apply_temperature(probs, T)
    return {
        "split": name,
        "n": int(len(labels)),
        "accuracy": float(C.accuracy_np(logits, labels)),
        "nll_T1": float(C.nll_np(logits, labels, 1.0)),
        "ece_T1": float(C.ece_np(probs, labels)),
        "temperature": float(T),
        "nll_Tfit": float(C.nll_np(logits, labels, T)),
        "ece_Tfit": float(C.ece_np(probs_t, labels)),
        "mean_max_prob_T1": float(probs.max(axis=1).mean()),
        "mean_max_prob_Tfit": float(probs_t.max(axis=1).mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    with h5py.File(ASCAD, "r") as f:
        x = f["Profiling_traces/traces"][:50000].astype(np.float32)[:, None, :]
        y = f["Profiling_traces/labels"][:50000].astype(np.int64)
    rng = np.random.RandomState(0)
    tr_idx = rng.choice(45000, 5000, replace=False)
    x_tr, y_tr = x[tr_idx], y[tr_idx]
    x_va, y_va = x[45000:50000], y[45000:50000]

    model = U.load_e0_module().OfficialCNN(num_classes=256)
    sd = torch.load(str(CKPT), map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={missing} unexpected={unexpected}")
    model.to(device)

    lg_tr = logits_for(model, x_tr, device)
    lg_va = logits_for(model, x_va, device)

    fit_tr = C.fit_temperature(lg_tr, y_tr)
    fit_va = C.fit_temperature(lg_va, y_va)

    rows = [summarize(lg_tr, y_tr, 1.0, "source_train_5k"),
            summarize(lg_va, y_va, 1.0, "source_val_5k"),
            summarize(lg_tr, y_tr, fit_tr["temperature"], "source_train_5k@T_train"),
            summarize(lg_va, y_va, fit_va["temperature"], "source_val_5k@T_val"),
            summarize(lg_tr, y_tr, fit_va["temperature"], "source_train_5k@T_val"),
            summarize(lg_va, y_va, fit_tr["temperature"], "source_val_5k@T_train")]
    out = {
        "purpose": "T fitted on source_val is a property of a near-chance split, not of the classifier",
        "checkpoint": str(CKPT),
        "device": device,
        "fit_on_source_train": fit_tr,
        "fit_on_source_val": fit_va,
        "rows": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2)[:4000])


if __name__ == "__main__":
    main()
