#!/usr/bin/env python3
"""训练进行中的快速查看：对已产出 predictions_attack.npy 的 C-E4 run 即时算 GE 曲线。
不写 results/，只打印；用于监控趋势，正式结果由 evaluate_c_e4.py 生成。"""
from __future__ import annotations
import importlib.util, json
from pathlib import Path
import h5py, numpy as np

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[2]
SHIFTS = {"desync50": Path("D:/sca_project/ASCAD_data/ASCAD_desync50.h5"),
          "desync100": Path("D:/sca_project/ASCAD_data/ASCAD_desync100.h5")}
spec = importlib.util.spec_from_file_location("ea", str(ROOT / "experiments/cdan-sca-diagnosis/scripts/evaluate_ascad.py"))
EA = importlib.util.module_from_spec(spec); spec.loader.exec_module(EA)

rows = []
for d in sorted((HERE / "results").iterdir()):
    if not d.is_dir():
        continue
    p = d / "predictions_attack.npy"
    if not p.exists():
        continue
    run = d.name
    shift = "desync100" if "desync100" in run else "desync50"
    with h5py.File(SHIFTS[shift], "r") as h:
        meta = np.asarray(h["Attack_traces/metadata"])
    cur = EA.ranks(np.load(p), np.asarray(meta["plaintext"]), np.asarray(meta["key"]), 2, 100)
    g = dict(cur)
    z = [n for n, r in cur if r == 0]
    rows.append({"run": run, "ntge": z[0] if z else None, "stable": (cur[-1][1] == 0),
                 "ge1k": g.get(1000), "ge5k": g.get(5000), "ge10k": g.get(10000), "nzero": len(z)})
for r in rows:
    print(f"{r['run']:32s} NTGE={str(r['ntge']):5s} stable={str(r['stable']):5s} "
          f"GE@1k={str(r['ge1k']):4s} GE@5k={str(r['ge5k']):4s} GE@10k={str(r['ge10k']):4s} n0={r['nzero']}")
print(f"== {len(rows)} runs with predictions")
