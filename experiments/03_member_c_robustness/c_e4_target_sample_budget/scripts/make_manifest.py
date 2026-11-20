#!/usr/bin/env python3
"""C-E4 manifest：实验目录产物 + outputs/models/c_e4 的 54 个 checkpoint +
复用的 B-E1 初始化权重 + 三个数据集哈希。

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
MODELS = REPO / "outputs/models/c_e4"
B1_MODELS = REPO / "outputs/models/b_e1"
A2_MODELS = REPO / "outputs/models/a_e2"
A3_MODELS = REPO / "outputs/models/a_e3"
B3_MODELS = REPO / "outputs/models/b_e3"
DATASETS = {
    "fixed": Path("D:/sca_project/ASCAD_data/ASCAD.h5"),
    "desync50": Path("D:/sca_project/ASCAD_data/ASCAD_desync50.h5"),
    "desync100": Path("D:/sca_project/ASCAD_data/ASCAD_desync100.h5"),
}
SKIP_DIRS = {".mplconfig", "__pycache__", "results_smoke", "models_smoke", "results_stage"}


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


def scan_models(root, label):
    out = []
    if not root.exists():
        return out
    for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        ck = run_dir / "ckpt.pt"
        if ck.exists():
            out.append({"group": label, "run": run_dir.name,
                        "path": ck.relative_to(REPO).as_posix(),
                        "bytes": ck.stat().st_size, "sha256": sha256(ck)})
    return out


def main():
    files = [{"path": rel, "bytes": p.stat().st_size, "sha256": sha256(p)}
             for rel, p in iter_files(HERE)]

    archived = scan_models(MODELS, "c_e4")
    reuse = []
    for label, root_ in [("b_e1_init", B1_MODELS), ("a_e2_dann", A2_MODELS),
                         ("a_e3_cdpa", A3_MODELS), ("b_e3_cdan_sca", B3_MODELS)]:
        reuse += scan_models(root_, label)

    datasets = []
    for name, p in DATASETS.items():
        if p.exists():
            datasets.append({"name": name, "path": str(p), "bytes": p.stat().st_size,
                             "sha256": sha256(p)})

    man = {
        "experiment": "C-E4",
        "title": "target adaptation sample budget sensitivity (50/500/5000/45000)",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": platform.node(),
        "python": sys.version.split()[0],
        "git_commit": git("rev-parse", "HEAD"),
        "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "root": HERE.relative_to(REPO).as_posix(),
        "files": files, "n_files": len(files),
        "total_bytes": int(sum(f["bytes"] for f in files)),
        "archived_models": archived, "n_models": len(archived),
        "models_total_bytes": int(sum(f["bytes"] for f in archived)),
        "reuse_models": reuse, "n_reuse_models": len(reuse),
        "datasets": datasets,
        "notes": [
            "predictions_attack.npy（10.2 MB/run）与 history/diag 留在 results/<run>/，由 .gitignore 排除大文件。",
            "checkpoint 不入 Git，存 outputs/models/c_e4/<run>/ckpt.pt。",
            "全量档 (budget=45000) 不重跑：CDPA 复用 A-E3，DANN 复用 A-E2，CDAN-SCA full 复用 B-E3。",
            "Source Only 参照复用 B-E2 的 predictions。",
        ],
    }
    out = HERE / "manifests/manifest.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(man, indent=2), encoding="utf-8")
    print(f"wrote {out} : {man['n_files']} files / {man['total_bytes']/1e6:.1f} MB, "
          f"{man['n_models']} models / {man['models_total_bytes']/1e9:.2f} GB, "
          f"{man['n_reuse_models']} reuse models")


if __name__ == "__main__":
    main()
