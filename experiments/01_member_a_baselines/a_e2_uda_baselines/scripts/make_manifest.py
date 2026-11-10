#!/usr/bin/env python3
"""A-E2 manifest：实验目录产物 + outputs/models/a_e2 的 12 个主 run + 2 个 λ=0 pilot checkpoint + 三个数据集哈希。

用法: python scripts/make_manifest.py
"""
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[2]
MODELS = REPO / "outputs/models/a_e2"
B1_MANIFEST = REPO / "experiments/02_member_b_method/b_e1_architecture_same_domain/manifests/manifest.json"
DATASETS = {
    "fixed": Path("D:/sca_project/ASCAD_data/ASCAD.h5"),
    "desync50": Path("D:/sca_project/ASCAD_data/ASCAD_desync50.h5"),
    "desync100": Path("D:/sca_project/ASCAD_data/ASCAD_desync100.h5"),
}
SKIP_DIRS = {".mplconfig", "__pycache__"}


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def git(*args):
    try:
        return subprocess.check_output(["git", *args], cwd=str(REPO),
                                       text=True, stderr=subprocess.DEVNULL).strip()
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
                reused.append({"path": m["path"], "bytes": m["bytes"], "sha256": m["sha256"],
                               "role": "initial weights (same-seed B-E1 checkpoint)",
                               "source_manifest": B1_MANIFEST.relative_to(REPO).as_posix()})

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
        "experiment_id": "A-E2",
        "experiment_name": "DANN/CDAN 统一 UDA 基线：ASCAD fixed -> desync50/desync100",
        "owner": "WBC-cbw",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git("rev-parse", "HEAD"),
        "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "run_commands": [
            "python experiments/01_member_a_baselines/a_e2_uda_baselines/scripts/run_all.py --epochs 75",
            "python experiments/01_member_a_baselines/a_e2_uda_baselines/scripts/evaluate_uda.py",
            "python experiments/01_member_a_baselines/a_e2_uda_baselines/scripts/aggregate_uda.py",
            "python experiments/01_member_a_baselines/a_e2_uda_baselines/scripts/plot_uda_ge.py",
            "python experiments/01_member_a_baselines/a_e2_uda_baselines/scripts/analyze_confidence.py",
            "python experiments/01_member_a_baselines/a_e2_uda_baselines/scripts/plot_training_dynamics.py",
            "python experiments/01_member_a_baselines/a_e2_uda_baselines/scripts/plot_domain_diag.py",
            "python experiments/01_member_a_baselines/a_e2_uda_baselines/scripts/run_controls.py",
            "python experiments/01_member_a_baselines/a_e2_uda_baselines/scripts/evaluate_controls.py",
            "python experiments/01_member_a_baselines/a_e2_uda_baselines/scripts/compare_controls.py",
            "python experiments/01_member_a_baselines/a_e2_uda_baselines/scripts/make_manifest.py",
        ],
        "hardware": {"gpu": gpu, "os": platform.platform()},
        "software": sw,
        "datasets": datasets,
        "splits": {"source_train": "fixed Profiling_traces[0:45000]",
                   "source_val": "fixed Profiling_traces[45000:50000]",
                   "target_adapt": "target Profiling_traces[0:45000] (unlabeled)",
                   "target_domain_val": "target Profiling_traces[45000:50000] (diagnostic only)",
                   "target_attack": "target Attack_traces[0:10000]"},
        "target_byte": 2,
        "seeds": [42, 43, 44],
        "methods": ["dann", "cdan"],
        "baseline": "source_only (B-E2 cnn_s<seed> predictions, not retrained)",
        "controls": {
            "purpose": "attribution: separate extra 75 epochs + target forward pass from adversarial alignment",
            "design": "DANN path with lambda_adv=0 (features receive only CE gradient; discriminator still trained)",
            "runs": ["dann_desync50_s42_lam0", "dann_desync100_s42_lam0"],
            "level": "pilot (excluded from the 12-run main matrix)",
        },
        "lambda_adv": 0.1,
        "proj_dim": 1024,
        "results_summary": summary,
        "archived_models": archived,
        "reused_checkpoints": reused,
        "files": files,
        "total_bytes": sum(f["bytes"] for f in files),
        "notes": {
            "checkpoints": "ckpt.pt 存在 outputs/models/a_e2/<run>/，每 5 epoch 写一次；"
                           "含 model/optimizer/epoch/history，体积约 532 MB/run",
            "predictions": "results/<run>/predictions_attack.npy 被 .gitignore 忽略（*.npy），"
                           "manifest 仍记录其路径与哈希",
            "source_only": "Source Only 的预测复用 B-E2 的 cnn_s<seed>.npy，本实验未重训",
            "controls": "λ=0 pilot 的预测与评估在 results/<run>/predictions_attack.npy 与 results/eval_controls/，"
                        "对比表在 results/controls_comparison.{json,md}，图在 figures/a_e2_ge_controls.{png,pdf}",
        },
    }
    out = HERE / "manifests/manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"manifest: {len(files)} files, {manifest['total_bytes']/1e6:.1f} MB; "
          f"models: {len(archived)} ({sum(m['bytes'] for m in archived)/1e9:.2f} GB); "
          f"reused init: {len(reused)}; datasets: {len(datasets)} -> {out}")


if __name__ == "__main__":
    main()
