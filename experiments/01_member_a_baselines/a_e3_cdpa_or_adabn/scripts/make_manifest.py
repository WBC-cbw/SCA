#!/usr/bin/env python3
"""A-E3 manifest：实验目录产物 + outputs/models/a_e3 的 6 个主 run checkpoint +
复用的 B-E1 初始化权重 + 三个数据集哈希。

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
MODELS = REPO / "outputs/models/a_e3"
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
                               "role": "pretrained source-only weights (same-seed B-E1 checkpoint)",
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
        "experiment_id": "A-E3",
        "experiment_name": "CDPA-MMD 领域强基线：ASCAD fixed -> desync50/desync100",
        "owner": "WBC-cbw",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git("rev-parse", "HEAD"),
        "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "run_commands": [
            "python experiments/01_member_a_baselines/a_e3_cdpa_or_adabn/scripts/train_cdpa.py "
            "--shift desync50 --seed 42  (x6 via scripts/run_cdpa_queue.bat)",
            "python experiments/01_member_a_baselines/a_e3_cdpa_or_adabn/scripts/evaluate_a_e3.py",
            "python experiments/01_member_a_baselines/a_e3_cdpa_or_adabn/scripts/aggregate_a_e3.py",
            "python experiments/01_member_a_baselines/a_e3_cdpa_or_adabn/scripts/plot_a_e3.py",
            "python experiments/01_member_a_baselines/a_e3_cdpa_or_adabn/scripts/make_manifest.py",
        ],
        "hardware": {"gpu": gpu, "os": platform.platform()},
        "software": sw,
        "datasets": datasets,
        "splits": {"source_train": "fixed Profiling_traces[0:45000] (labeled)",
                   "source_val": "fixed Profiling_traces[45000:50000]",
                   "target_adapt": "target Profiling_traces[0:45000] (unlabeled, MMD only)",
                   "target_domain_val": "target Profiling_traces[45000:50000] (diagnostic only)",
                   "target_attack": "target Attack_traces[0:10000]"},
        "target_byte": 2,
        "seeds": [42, 43, 44],
        "methods": ["source_only", "dann", "cdan", "cdpa_mmd"],
        "baseline_sources": {
            "source_only": "experiments/02_member_b_method/b_e2_architecture_shift/results/predictions/<shift>/cnn_s<seed>.npy",
            "dann": "experiments/01_member_a_baselines/a_e2_uda_baselines/results/dann_<shift>_s<seed>/predictions_attack.npy",
            "cdan": "experiments/01_member_a_baselines/a_e2_uda_baselines/results/cdan_<shift>_s<seed>/predictions_attack.npy",
            "cdpa_mmd": "experiments/01_member_a_baselines/a_e3_cdpa_or_adabn/results/cdpa_<shift>_s<seed>/predictions_attack.npy",
        },
        "lambda_mmd": 0.1,
        "mmd_layers": ["conv (10752-d flatten)", "fc1 (4096-d relu)", "fc2 (4096-d relu)"],
        "training_recipe": {
            "pretrain": "B-E1 source-only checkpoint (RMSprop lr=1e-5, batch 200, 75 epochs, seed-matched)",
            "finetune": "75 epochs, RMSprop lr=1e-5, batch 200, raw int8 input, lambda_mmd=0.1",
            "adapt_traces": 45000,
            "note": "official CDPA uses its own small CNN + Adam lr=1e-3 + 15 finetune epochs; "
                    "this port keeps the project-frozen backbone/recipe (see README 8.1.5)",
        },
        "diagnostic_runs": {
            "level": "pilot / debug (excluded from the 6-run main matrix)",
            "runs": ["cdpa_desync50_s42_smoke", "cdpa_desync50_s42_lam0", "cdpa_desync50_s42_lr0",
                    "A-E2: dann_desync50_s42_diag_lam0"],
            "purpose": "check MMD implementation stability, lambda=0 behaviour, and first-epoch "
                       "RMSprop transient before starting the formal queue",
        },
        "results_summary": summary,
        "archived_models": archived,
        "reused_checkpoints": reused,
        "files": files,
        "total_bytes": sum(f["bytes"] for f in files),
        "notes": {
            "checkpoints": "ckpt.pt 存在 outputs/models/a_e3/<run>/（每 5 epoch 覆盖写，含 model/optimizer/epoch/history）",
            "predictions": "results/<run>/predictions_attack.npy 被 .gitignore 忽略（*.npy），manifest 仍记录路径与哈希",
            "baselines": "Source Only / DANN / CDAN 的预测直接复用 B-E2 与 A-E2，未在本实验重训；"
                         "评估脚本只读这些预测，因此四方法在同一 GE 实现下可比",
            "diagnostics": "诊断 run 的预测与 history 保留在 results/<run>/ 下，不计入主表",
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
