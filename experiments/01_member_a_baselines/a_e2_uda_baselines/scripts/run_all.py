#!/usr/bin/env python
"""A-E2 串行编排器：12 个 DANN/CDAN 训练 run（方法×偏移×seed）。

设计要点
- 串行执行：本机实测 2 进程并发时单 run 变慢 2.4x，总吞吐反而下降。
- 幂等：run_meta.json 显示 epochs_done==epochs 且 predictions_attack.npy 存在则跳过。
- 每 run 独立日志 logs/<run>.log；总状态 logs/run_all_status.json（每步立即落盘）。
- 训练全部成功后，若后处理脚本存在则依次运行 evaluate/aggregate/plot。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
LOGS = EXP / "logs"
RESULTS = EXP / "results"

METHODS = ["dann", "cdan"]
SHIFTS = ["desync50", "desync100"]
SEEDS = [42, 43, 44]


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def run_name(method: str, shift: str, seed: int) -> str:
    return f"{method}_{shift}_s{seed}"


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def is_complete(name: str, epochs: int) -> bool:
    meta = read_json(RESULTS / name / "run_meta.json")
    if not meta:
        return False
    if int(meta.get("epochs_done") or 0) != epochs:
        return False
    return (RESULTS / name / "predictions_attack.npy").exists()


def write_status(path: Path, status: dict) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--shifts", default=",".join(SHIFTS))
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--epochs", type=int, default=75)
    ap.add_argument("--post", dest="post", action="store_true", help="训练后运行 evaluate/aggregate/plot")
    ap.add_argument("--no-post", dest="post", action="store_false")
    ap.set_defaults(post=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    methods = [m for m in args.methods.split(",") if m]
    shifts = [s for s in args.shifts.split(",") if s]
    seeds = [int(s) for s in args.seeds.split(",") if s]

    LOGS.mkdir(parents=True, exist_ok=True)
    status_path = LOGS / "run_all_status.json"
    status = {
        "started_at": now(),
        "epochs": args.epochs,
        "python": sys.executable,
        "runs": {},
        "state": "running",
        "order": [],
    }
    for m in methods:
        for sh in shifts:
            for seed in seeds:
                name = run_name(m, sh, seed)
                status["order"].append(name)
                status["runs"][name] = {"state": "pending"}
    write_status(status_path, status)

    print(f"[{now()}] A-E2 orchestrator start: {len(status['order'])} runs, epochs={args.epochs}", flush=True)

    failures = []
    for name in status["order"]:
        entry = status["runs"][name]
        if is_complete(name, args.epochs):
            entry.update({"state": "skipped_complete", "checked_at": now()})
            write_status(status_path, status)
            print(f"[{now()}] skip (complete): {name}", flush=True)
            continue
        method, shift, seed_s = name.split("_")
        seed = int(seed_s.lstrip("s"))
        cmd = [sys.executable, str(HERE / "train_uda.py"),
               "--method", method, "--shift", shift, "--seed", str(seed),
               "--epochs", str(args.epochs)]
        log_path = LOGS / f"{name}.log"
        entry.update({"state": "running", "command": " ".join(cmd), "log": str(log_path),
                      "started_at": now()})
        write_status(status_path, status)
        print(f"[{now()}] start {name} -> {log_path.name}", flush=True)
        t0 = time.time()
        if args.dry_run:
            entry.update({"state": "dry_run"})
            write_status(status_path, status)
            continue
        with open(log_path, "w", encoding="utf-8") as fh:
            fh.write(f"# command: {' '.join(cmd)}\n# started: {now()}\n")
            fh.flush()
            rc = subprocess.call(cmd, cwd=str(EXP), stdout=fh, stderr=subprocess.STDOUT)
        dt = time.time() - t0
        meta = read_json(RESULTS / name / "run_meta.json") or {}
        entry.update({
            "state": "done" if rc == 0 else "failed",
            "returncode": rc,
            "seconds": round(dt, 1),
            "finished_at": now(),
            "epochs_done": meta.get("epochs_done"),
            "sec_per_epoch_mean": meta.get("sec_per_epoch_mean"),
            "final_attack_acc": meta.get("final_attack_acc"),
        })
        if rc != 0:
            failures.append(name)
        write_status(status_path, status)
        print(f"[{now()}] {entry['state']} {name} rc={rc} {dt/60:.1f} min", flush=True)

    status["finished_at"] = now()
    status["failures"] = failures
    status["state"] = "training_done" if not failures else "training_with_failures"
    write_status(status_path, status)

    if args.post and not failures:
        for script in ["evaluate_uda.py", "aggregate_uda.py", "plot_uda_ge.py"]:
            p = HERE / script
            if not p.exists():
                print(f"[{now()}] post: skip missing {script}", flush=True)
                continue
            log_path = LOGS / f"post_{script.replace('.py', '')}.log"
            print(f"[{now()}] post: {script}", flush=True)
            with open(log_path, "w", encoding="utf-8") as fh:
                rc = subprocess.call([sys.executable, str(p)], cwd=str(EXP),
                                     stdout=fh, stderr=subprocess.STDOUT)
            status.setdefault("post", {})[script] = {"returncode": rc, "log": str(log_path)}
            write_status(status_path, status)
            if rc != 0:
                failures.append(script)
                break

    status["state"] = "complete" if not failures else "completed_with_failures"
    status["finished_at"] = now()
    write_status(status_path, status)
    print(f"[{now()}] orchestrator end: {status['state']}", flush=True)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
