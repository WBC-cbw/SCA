#!/usr/bin/env python
"""B-E2 全流程编排：校验数据集 -> 三偏移零适应推理 -> GE 评估 -> 汇总 -> 绘图。

使用 Windows 原生 Python 子进程（sys.executable），避免 shell 路径转换问题。
不训练、不更新权重；推理耗时单独记录，不计入攻击轨迹成本。
"""
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
LOGS = EXP / "logs"
STATUS = LOGS / "run_all_status.log"


def log_status(msg):
    line = f"{datetime.now().astimezone().isoformat(timespec='seconds')} {msg}"
    print(line, flush=True)
    with STATUS.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_step(name, args, env=None):
    log_status(f"START {name}")
    t0 = datetime.now()
    p = subprocess.run(args, cwd=str(EXP), capture_output=True, text=True,
                       encoding="utf-8", errors="replace", env=env)
    dt = (datetime.now() - t0).total_seconds()
    with (LOGS / f"{name}.log").open("w", encoding="utf-8") as f:
        f.write(f"command: {' '.join(str(a) for a in args)}\n")
        f.write(f"started: {t0.astimezone().isoformat(timespec='seconds')}\n")
        f.write(f"duration_seconds: {dt:.1f}\n")
        f.write(f"returncode: {p.returncode}\n")
        f.write("\n--- stdout ---\n")
        f.write(p.stdout or "")
        f.write("\n--- stderr ---\n")
        f.write(p.stderr or "")
    log_status(f"END {name} rc={p.returncode} {dt:.1f}s")
    return p.returncode


def main():
    LOGS.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    env_info = []
    env_info.append(f"date: {datetime.now().astimezone().isoformat(timespec='seconds')}")
    env_info.append(f"python_executable: {py}")
    env_info.append(f"python_version: {sys.version.split()[0]}")
    try:
        import torch, numpy, h5py, matplotlib
        env_info.append(f"torch: {torch.__version__}")
        env_info.append(f"cuda_available: {torch.cuda.is_available()}")
        env_info.append(f"device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
        env_info.append(f"numpy: {numpy.__version__}")
        env_info.append(f"h5py: {h5py.__version__}")
        env_info.append(f"matplotlib: {matplotlib.__version__}")
    except Exception as exc:  # pragma: no cover
        env_info.append(f"env_probe_error: {exc!r}")
    repo = EXP.parents[2]
    for cmd in (["git", "-C", str(repo), "log", "-1", "--oneline"],
                ["git", "-C", str(repo), "status", "--porcelain"]):
        p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        env_info.append(f"$ {' '.join(cmd)}")
        env_info.append((p.stdout or "").rstrip() or "(empty)")
    (LOGS / "00_environment.log").write_text("\n".join(env_info) + "\n", encoding="utf-8")

    steps = [
        ("01_verify_datasets", [py, str(HERE / "verify_datasets.py")], None),
        ("02_predict_fixed", [py, str(HERE / "predict_b_e2.py"), "--shift", "fixed"], None),
        ("02_predict_desync50", [py, str(HERE / "predict_b_e2.py"), "--shift", "desync50"], None),
        ("02_predict_desync100", [py, str(HERE / "predict_b_e2.py"), "--shift", "desync100"], None),
        ("03_evaluate", [py, str(HERE / "evaluate_b_e2.py")], None),
        ("04_aggregate", [py, str(HERE / "aggregate_b_e2.py")], None),
    ]
    plot_env = dict(os.environ)
    plot_env["MPLCONFIGDIR"] = str(EXP / ".mplconfig")
    steps.append(("05_plot", [py, str(HERE / "plot_b_e2_ge.py")], plot_env))

    (EXP / ".mplconfig").mkdir(exist_ok=True)
    rc_all = 0
    for name, args, env in steps:
        rc = run_step(name, args, env)
        rc_all = rc_all or rc
    log_status(f"ALL DONE rc={rc_all}")
    return rc_all


if __name__ == "__main__":
    sys.exit(main())
