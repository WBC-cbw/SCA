#!/usr/bin/env python3
"""B-E3 manifest：实验目录产物 + outputs/models/b_e3 的 12 个主 run checkpoint +
复用的 B-E1 初始化权重与 A-E2/A-E3 基线 checkpoint + 三个数据集哈希。

用法: python scripts/make_manifest.py
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[2]
MODELS = REPO / "outputs/models/b_e3"
B1_MANIFEST = REPO / "experiments/02_member_b_method/b_e1_architecture_same_domain/manifests/manifest.json"
A2 = REPO / "outputs/models/a_e2"
A3 = REPO / "outputs/models/a_e3"
DATASETS = {
    "fixed": Path("D:/sca_project/ASCAD_data/ASCAD.h5"),
    "desync50": Path("D:/sca_project/ASCAD_data/ASCAD_desync50.h5"),
    "desync100": Path("D:/sca_project/ASCAD_data/ASCAD_desync100.h5"),
}
SKIP_DIRS = {".mplconfig", "__pycache__", "results_smoke"}


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def git(*args):
    try:
        return subprocess.check_output(["git", *args], cwd=str(REPO), text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def iter_files(root):
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root).as_posix()
        parts = p.relative_to(root).parts
        if rel.startswith("manifests/") or any(part in SKIP_DIRS for part in parts):
            continue
        yield rel, p


def main():
    files = [{"path": rel, "bytes": p.stat().st_size, "sha256": sha256(p)}
             for rel, p in iter_files(HERE)]

    archived = []
    if MODELS.exists():
        for run_dir in sorted(p for p in MODELS.iterdir() if p.is_dir()):
            ck = run_dir / "ckpt.pt"
            if ck.exists():
                archived.append({"run": run_dir.name, "path": ck.relative_to(REPO).as_posix(),
                                 "bytes": ck.stat().st_size, "sha256": sha256(ck)})

    reused = []
    if B1_MANIFEST.exists():
        b1 = json.loads(B1_MANIFEST.read_text(encoding="utf-8"))
        for m in b1.get("archived_models", []):
            if "/cnn_s" in m["path"]:
                reused.append({"role": "pretrained source-only weights (same-seed B-E1 ckpt)",
                               "path": m["path"], "bytes": m["bytes"], "sha256": m["sha256"]})
    for run in [f"cdan_{s}_s{seed}" for s in ("desync50", "desync100") for seed in (42, 43, 44)]:
        ck = A2 / run / "ckpt.pt"
        if ck.exists():
            reused.append({"role": "ablation 2 (CDAN, no entropy weighting) checkpoint",
                           "path": ck.relative_to(REPO).as_posix(),
                           "bytes": ck.stat().st_size, "sha256": sha256(ck)})
    for run in [f"cdpa_{s}_s{seed}" for s in ("desync50", "desync100") for seed in (42, 43, 44)]:
        ck = A3 / run / "ckpt.pt"
        if ck.exists():
            reused.append({"role": "strong baseline CDPA-MMD checkpoint",
                           "path": ck.relative_to(REPO).as_posix(),
                           "bytes": ck.stat().st_size, "sha256": sha256(ck)})

    try:
        import torch, numpy, h5py, matplotlib
        sw = {"python": sys.version.split()[0], "torch": torch.__version__,
              "numpy": numpy.__version__, "h5py": h5py.__version__,
              "matplotlib": matplotlib.__version__}
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:
        sw, gpu = {}, "unknown"

    summary = json.loads((HERE / "results/summary.json").read_text(encoding="utf-8")) \
        if (HERE / "results/summary.json").exists() else None

    datasets = {}
    for name, path in DATASETS.items():
        datasets[name] = {"path": str(path), "bytes": path.stat().st_size if path.exists() else None,
                          "sha256": sha256(path) if path.exists() else None}

    manifest = {
        "experiment_id": "B-E3 / B-E4",
        "experiment_name": "CDAN-SCA full 与核心消融（含熵权重诊断）：ASCAD fixed -> desync50/desync100",
        "owner": "WBC-cbw",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git("rev-parse", "HEAD"),
        "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "run_commands": [
            "python experiments/02_member_b_method/b_e3_cdan_sca/scripts/train_cdan_sca.py "
            "--mode full|marginal --shift desync50|desync100 --seed 42|43|44  (x12 via scripts/run_queue.bat)",
            "python scripts/evaluate_b_e3.py",
            "python scripts/aggregate_b_e3.py",
            "python scripts/analyze_entropy_weight.py",
            "python scripts/permutation_b_e3.py",
            "python scripts/plot_b_e3.py",
            "python scripts/make_manifest.py",
        ],
        "hardware": {"gpu": gpu, "os": platform.platform()},
        "software": sw,
        "datasets": datasets,
        "splits": {"source_train": "fixed Profiling_traces[0:45000] (labeled)",
                   "source_val": "fixed Profiling_traces[45000:50000]",
                   "target_adapt": "target Profiling_traces[0:45000] (unlabeled; entropy weight + adversarial only)",
                   "target_domain_val": "target Profiling_traces[45000:50000] (domain diagnosis only)",
                   "target_attack": "target Attack_traces[0:10000]"},
        "target_byte": 2,
        "seeds": [42, 43, 44],
        "methods": ["source_only", "dann", "cdan", "cdpa_mmd", "cdan_sca_full", "cdan_sca_marg"],
        "method_notes": {
            "cdan_sca_full": "T(f,g) 随机多线性映射 proj_dim=1024 + GRL + 熵权重 w=1-H/log(256)（仅目标域对抗项）",
            "cdan_sca_marg": "消融 1：去条件输入（DANN 式判别器输入 f）+ 同一熵权重",
            "cdan": "消融 2：复用 A-E2 的 6 个 run（无熵权重），不重跑",
            "cdpa_mmd": "最强基线，复用 A-E3 的 6 个 run",
        },
        "hyperparameters": {"optimizer": "RMSprop", "lr": 1e-5, "batch_size": 200,
                            "epochs": 75, "lambda_adv": 0.1, "proj_dim": 1024,
                            "input": "raw int8 amplitude (no normalization)",
                            "entropy_weight": "w = 1 - H/np.log(256), H in nats, detached"},
        "diagnostic_runs": {
            "level": "pilot / debug (excluded from the 12-run main matrix)",
            "runs": ["cdan_sca_full_desync50_s42_smoke", "cdan_sca_marg_desync100_s43_smoke"],
            "location": "results_smoke/ 与 outputs/models/b_e3_smoke/",
            "purpose": "验证两种 mode 的 loss/监控/落盘链路，2 epoch、2000 条",
        },
        "results_summary": summary,
        "archived_models": archived,
        "reused_checkpoints": reused,
        "files": files,
        "total_bytes": sum(f["bytes"] for f in files),
        "notes": {
            "checkpoints": "ckpt.pt 存在 outputs/models/b_e3/<run>/（每 5 epoch 覆盖写）",
            "predictions": "results/<run>/predictions_attack.npy 与 diag_target_adapt.npz 被 .gitignore 忽略（*.npy/*.npz），manifest 仍记录路径与哈希",
            "baselines": "Source Only / DANN / CDAN / CDPA-MMD 均复用前序实验预测，未重训",
            "ablation2": "CDAN 的逐 epoch top1/熵未在 A-E2 记录，因此监控图对 CDAN 只给 epoch 75 终点参考",
            "privacy": "target_adapt 标签仅在训练结束后的诊断中使用",
        },
    }
    out = HERE / "manifests/manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"manifest: {len(files)} files, {manifest['total_bytes']/1e6:.1f} MB; "
          f"models: {len(archived)} ({sum(m['bytes'] for m in archived)/1e9:.2f} GB); "
          f"reused: {len(reused)}; datasets: {len(datasets)} -> {out}")


if __name__ == "__main__":
    main()
