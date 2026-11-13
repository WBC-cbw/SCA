#!/usr/bin/env python3
"""B-E5 预检：在 B-E1 同 seed 初始化权重上拟合源域温度，记录 T 与校准前后的 NLL/ECE。

只影响权重函数的先验，不参与任何训练；输出 results/temperature_probe.json。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "experiments/01_member_a_baselines/a_e2_uda_baselines/scripts"))
sys.path.insert(0, str(HERE / "scripts"))

from uda_common import B_E1_MODELS, UDANet, load_b_e1_init, load_domain_data  # noqa: E402
from calib_common import fit_temperature, softmax_np  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def logits_of(model, x, batch=1000):
    model.eval()
    out = []
    for s in range(0, len(x), batch):
        xb = torch.from_numpy(x[s:s + batch]).to(DEVICE)
        lg, _ = model(xb)
        out.append(lg.cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def main():
    data = load_domain_data("D:/sca_project/ASCAD_data/ASCAD.h5",
                            "D:/sca_project/ASCAD_data/ASCAD_desync50.h5",
                            n_source_train=45000, n_target_adapt=45000)
    xv, yv = data["source_val_x"], data["source_val_y"]
    recs = []
    for seed in (42, 43, 44):
        model = UDANet("dann").to(DEVICE)
        load_b_e1_init(model, B_E1_MODELS / f"cnn_s{seed}" / "ckpt.pt")
        lg = logits_of(model, xv)
        rec = fit_temperature(lg, yv)
        rec["seed"] = seed
        rec["init_ckpt"] = str(B_E1_MODELS / f"cnn_s{seed}" / "ckpt.pt")
        recs.append(rec)
        print(json.dumps(rec), flush=True)
    (HERE / "results/temperature_probe.json").write_text(
        json.dumps({"note": "source_val 温度拟合，模型为 B-E1 同 seed 初始化（训练前）",
                    "runs": recs}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
