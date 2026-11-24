#!/usr/bin/env python3
"""adv2 收尾流水线：evaluate -> aggregate -> permutation -> plot。"""
from __future__ import annotations
import subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
SCRIPTS = HERE / "scripts"
PY = sys.executable
STEPS = ["eval_adv2.py", "aggregate_adv2.py", "permutation_adv2.py", "plot_adv2.py", "make_report_adv2.py"]


try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def run(script):
    t0 = time.time()
    r = subprocess.run([PY, str(SCRIPTS / script)], cwd=str(HERE),
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(f"[{script}] rc={r.returncode} {time.time()-t0:.1f}s", flush=True)
    if r.stdout.strip():
        print(r.stdout.strip()[-2000:], flush=True)
    if r.returncode != 0:
        print(r.stderr.strip()[-3000:], flush=True)
    return r.returncode


def main():
    rc = 0
    for s in STEPS:
        rc = run(s)
        if rc != 0:
            break
    (HERE / "logs" / ("FINALIZE_ADV2_DONE.txt" if rc == 0 else "FINALIZE_ADV2_FAIL.txt")).write_text(
        f"{time.strftime('%Y-%m-%d %H:%M:%S')} rc={rc}\n", encoding="utf-8")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
