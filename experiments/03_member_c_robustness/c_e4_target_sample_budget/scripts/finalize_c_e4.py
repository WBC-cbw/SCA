#!/usr/bin/env python3
"""C-E4 收尾流水线：evaluate -> aggregate -> permutation -> plot -> manifest -> report。

用法（实验根目录下）:
    python scripts/finalize_c_e4.py
可用 --skip-manifest / --skip-permutation 跳过耗时步骤。
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
PY = sys.executable


def run(name, cmd, envs=None):
    import os
    env = dict(os.environ)
    if envs:
        env.update(envs)
    t0 = time.time()
    print(f"\n=== {name}: {' '.join(cmd)}", flush=True)
    p = subprocess.run(cmd, cwd=str(HERE), env=env)
    print(f"=== {name} rc={p.returncode} ({time.time()-t0:.1f}s)", flush=True)
    if p.returncode != 0:
        raise SystemExit(f"{name} failed with rc={p.returncode}")
    return p.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=100)
    ap.add_argument("--skip-manifest", action="store_true")
    ap.add_argument("--skip-permutation", action="store_true")
    ap.add_argument("--skip-report", action="store_true")
    args = ap.parse_args()

    run("evaluate", [PY, "scripts/evaluate_c_e4.py", "--step", str(args.step)])
    run("aggregate", [PY, "scripts/aggregate_c_e4.py"])
    run("tables", [PY, "scripts/make_tables.py"])
    if not args.skip_permutation:
        run("permutation", [PY, "scripts/permutation_c_e4.py"])
    run("plot", [PY, "scripts/plot_c_e4.py"])
    if not args.skip_manifest:
        run("manifest", [PY, "scripts/make_manifest.py"])
    if not args.skip_report:
        run("report", [PY, "scripts/make_report.py"])
    (HERE / "logs").mkdir(exist_ok=True)
    (HERE / "logs/FINALIZE_DONE.txt").write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
    print("\nFINALIZE_DONE")


if __name__ == "__main__":
    main()
