#!/usr/bin/env python3
"""E0 收尾：生成训练曲线图、GE 曲线图（归档到 figures/）与运行清单 manifest。"""
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "experiments/00_common_e0"
RES = RUN / "results"
FIG = RUN / "figures"
MAN = RUN / "manifests"
FIG.mkdir(parents=True, exist_ok=True)
MAN.mkdir(parents=True, exist_ok=True)

history = json.loads((RES / "history.json").read_text(encoding="utf-8"))
ge_summary = json.loads((RES / "eval/summary.json").read_text(encoding="utf-8"))
curve = ge_summary["predictions_attack"]["ge_curve"]

# ---- 图 1：GE 曲线 ----
xs = [p["traces"] for p in curve]
ys = [p["ge"] for p in curve]
plt.figure(figsize=(7, 4.2))
plt.plot(xs, ys, color="#1f77b4", lw=1.8, label="Source Only (official CNN)")
plt.axhline(0, color="gray", lw=0.8, ls="--")
plt.xlabel("Number of attack traces")
plt.ylabel("Guessing Entropy (key rank)")
plt.title("E0: ASCAD fixed-key, Source Only, GE vs attack traces")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(FIG / "e0_ge_curve.png", dpi=160)
plt.close()

# ---- 图 2：训练曲线（loss + val acc + GE 探针）----
fig, ax1 = plt.subplots(figsize=(7, 4.2))
ax1.plot(history["epochs"], history["loss"], color="#d62728", lw=1.5, label="train loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Cross-entropy loss", color="#d62728")
ax1.tick_params(axis="y", labelcolor="#d62728")
ax1.grid(alpha=0.3)
ax2 = ax1.twinx()
probes = [(e, g) for e, g in zip(history["epochs"], history["ge_probe"]) if g is not None]
ax2.plot([p[0] for p in probes], [p[1] for p in probes], "o-", color="#1f77b4",
         ms=3, lw=1.2, label="GE probe (500 attack traces)")
ax2.set_ylabel("GE probe rank", color="#1f77b4")
ax2.tick_params(axis="y", labelcolor="#1f77b4")
plt.title("E0 training dynamics (RMSprop lr=1e-5, batch 200)")
fig.tight_layout()
plt.savefig(FIG / "e0_training_dynamics.png", dpi=160)
plt.close()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_rev():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


manifest = {
    "experiment": "E0: ASCAD fixed-key -> ASCAD fixed-key, Source Only",
    "date": "2026-10-10",
    "git_commit": git_rev(),
    "hardware": {"gpu": history.get("device"), "os": platform.platform()},
    "software": {"python": platform.python_version(), "torch": "2.9.1+cu128"},
    "seed": 42,
    "data": {
        "dataset": "D:/sca_project/ASCAD_data/ASCAD.h5 (ASCAD v1 fixed-key)",
        "source_train": "Profiling_traces[0:45000]",
        "source_val": "Profiling_traces[45000:50000]",
        "target_adapt": "无（Source Only 不使用目标域适应数据）",
        "target_attack": "Attack_traces[0:10000]",
        "target_byte": 2,
        "preprocessing": "原始 int8 幅度直接输入，无归一化（官方做法）",
    },
    "model": "官方五层 CNN（Conv1D 64/128/256/512/512 k11 + AvgPool x5 -> Dense4096 -> Dense4096 -> 256），PyTorch 复刻 src/framework",
    "training": {
        "optimizer": "RMSprop", "lr": 1e-5, "batch_size": 200,
        "epochs_run": history["epochs"][-1],
        "total_train_seconds": round(sum(history["epoch_seconds"]), 1),
        "convergence": "GE 探针（500 条攻击轨迹）自 epoch 34 起稳定为 0",
    },
    "metrics": {
        "ntge": ge_summary["predictions_attack"]["ntge"],
        "ge_at_10000": curve[-1]["ge"],
        "sr_at_1000": ge_summary["predictions_attack"]["sr_at_n"]["1000"],
        "sr_at_10000": ge_summary["predictions_attack"]["sr_at_n"]["10000"],
        "final_train_loss": history["loss"][-1],
        "final_val_accuracy": history["val_accuracy"][-1],
    },
    "artifacts": {
        "results/predictions_attack.npy": sha256(RES / "predictions_attack.npy"),
        "results/e0_model_torch.pt": sha256(RES / "e0_model_torch.pt"),
        "results/history.json": sha256(RES / "history.json"),
        "results/eval/summary.json": sha256(RES / "eval/summary.json"),
    },
}
(MAN / "manifest.json").write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
print("figures + manifest done")
print(json.dumps(manifest["metrics"], ensure_ascii=False))
