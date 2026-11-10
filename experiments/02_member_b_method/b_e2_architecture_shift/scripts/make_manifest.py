#!/usr/bin/env python3
"""B-E2 manifest：记录本实验全部产物的路径、大小与 SHA256，三个数据集的哈希，
以及从 B-E1 复用（不重新训练）的 checkpoint 清单。

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
B1_MANIFEST = REPO / "experiments/02_member_b_method/b_e1_architecture_same_domain/manifests/manifest.json"
DATASETS = {
    "fixed": Path("D:/sca_project/ASCAD_data/ASCAD.h5"),
    "desync50": Path("D:/sca_project/ASCAD_data/ASCAD_desync50.h5"),
    "desync100": Path("D:/sca_project/ASCAD_data/ASCAD_desync100.h5"),
}
SKIP_DIRS = {".mplconfig"}


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

    # B-E2 不产生新权重；checkpoint 直接从 B-E1 归档复用，这里转录其哈希而不重复计算
    reused = []
    if B1_MANIFEST.exists():
        b1 = json.loads(B1_MANIFEST.read_text(encoding="utf-8"))
        for m in b1.get("archived_models", []):
            reused.append({"path": m["path"], "bytes": m["bytes"], "sha256": m["sha256"],
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
        "experiment_id": "B-E2",
        "experiment_name": "三主干在 ASCAD fixed-key 到 desync50/desync100 的零适应迁移",
        "owner": "WBC-cbw",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git("rev-parse", "HEAD"),
        "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "run_commands": [
            "python experiments/02_member_b_method/b_e2_architecture_shift/scripts/run_b_e2_all.py",
            "python experiments/02_member_b_method/b_e2_architecture_shift/scripts/make_manifest.py",
        ],
        "hardware": {"gpu": gpu, "os": platform.platform()},
        "software": sw,
        "datasets": datasets,
        "splits": {"source_train": "ASCAD.h5 Profiling_traces[0:45000]",
                   "source_val": "ASCAD.h5 Profiling_traces[45000:50000]",
                   "target_adapt": "none",
                   "target_attack": "Attack_traces[0:10000] of each dataset"},
        "target_byte": 2,
        "seeds": [42, 43, 44],
        "architectures": ["mlp", "cnn", "resnet1d"],
        "results_summary": summary,
        "reused_checkpoints": reused,
        "files": files,
        "total_bytes": sum(f["bytes"] for f in files),
    }
    out = HERE / "manifests/manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"manifest: {len(files)} files, {manifest['total_bytes']/1e6:.1f} MB; "
          f"reused checkpoints: {len(reused)}; datasets: {len(datasets)} -> {out}")


if __name__ == "__main__":
    main()
