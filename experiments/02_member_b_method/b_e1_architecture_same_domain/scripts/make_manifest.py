#!/usr/bin/env python3
"""B-E1 manifest：记录每个产物的路径、大小、SHA256，以及数据集哈希与代码版本。

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
DATASET = Path("D:/sca_project/ASCAD_data/ASCAD.h5")


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


SKIP_DIRS = {".mplconfig"}


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

    # 大文件（checkpoint）按归档规范放 repo 的 outputs/models/b_e1/，单独记录
    model_root = REPO / "outputs/models/b_e1"
    archived_models = [{"path": p.relative_to(REPO).as_posix(), "bytes": p.stat().st_size,
                        "sha256": sha256(p)} for _rel, p in iter_files(model_root)] \
        if model_root.exists() else []

    try:
        import torch, numpy, h5py
        sw = {"python": sys.version.split()[0], "torch": torch.__version__,
              "numpy": numpy.__version__, "h5py": h5py.__version__}
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:
        sw, gpu = {}, "unknown"

    summary = json.loads((HERE / "results/summary.json").read_text(encoding="utf-8")) \
        if (HERE / "results/summary.json").exists() else None

    manifest = {
        "experiment_id": "B-E1",
        "experiment_name": "ASCAD fixed-key 同域结构筛选：MLP / Official CNN / ResNet1D",
        "owner": "WBC-cbw",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git("rev-parse", "HEAD"),
        "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "run_commands": [
            "bash experiments/02_member_b_method/b_e1_architecture_same_domain/scripts/run_b_e1_all.sh",
            "python experiments/02_member_b_method/b_e1_architecture_same_domain/scripts/evaluate_b_e1.py",
            "python experiments/02_member_b_method/b_e1_architecture_same_domain/scripts/aggregate_b_e1.py",
            "python experiments/02_member_b_method/b_e1_architecture_same_domain/scripts/plot_b_e1_ge.py",
        ],
        "hardware": {"gpu": gpu, "os": platform.platform()},
        "software": sw,
        "dataset": {
            "path": str(DATASET),
            "bytes": DATASET.stat().st_size if DATASET.exists() else None,
            "sha256": sha256(DATASET) if DATASET.exists() else None,
            "split": {"source_train": "Profiling_traces[0:45000]",
                      "source_val": "Profiling_traces[45000:50000]",
                      "target_adapt": "none",
                      "target_attack": "Attack_traces[0:10000]"},
        },
        "seeds": [42, 43, 44],
        "architectures": ["mlp", "cnn", "resnet1d"],
        "results_summary": summary,
        "archived_models": archived_models,
        "notes": {
            "large_files": "ckpt.pt 已归档到 outputs/models/b_e1/<run>/ckpt.pt（不入 Git）；"
                           "attack 集预测 predictions_attack.npy 留在 results/<run>/（*.npy 由 .gitignore 排除）",
            "pilot": "results_pilot/ 与 results_pilot/aborted_paging_resnet1d_s42/ 为补充诊断与事故残骸，不进入主结果",
        },
        "files": files,
        "total_bytes": sum(f["bytes"] for f in files),
    }
    out = HERE / "manifests/manifest.json"
    manifest["archived_models_total_bytes"] = sum(m["bytes"] for m in archived_models)
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"manifest: {len(files)} files, {manifest['total_bytes']/1e6:.1f} MB; "
          f"archived models: {len(archived_models)} files, {manifest['archived_models_total_bytes']/1e6:.1f} MB -> {out}")


if __name__ == "__main__":
    main()
