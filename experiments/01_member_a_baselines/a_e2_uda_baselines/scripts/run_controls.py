#!/usr/bin/env python3
"""λ=0 对照 pilot：同初始化、同配方再训 75 epoch，但无对抗梯度（GRL λ=0）。

目的：把 "多训 75 epoch + 目标域前向" 与 "对抗对齐" 分离，用于归因 DANN/CDAN 的 GE 变化。
复用 train_uda.py 的 DANN 代码路径：λ=0 时特征/分类器只收到 CE 梯度（判别器仍更新但不影响特征），
因此等价于 Source Only 继续训练。结果标注为 pilot，不进入主矩阵统计。
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
SHIFTS = ["desync50", "desync100"]
SEEDS = [42]


def main():
    (HERE / "logs").mkdir(exist_ok=True)
    status = {"started_at": time.strftime("%Y-%m-%d %H:%M:%S"), "runs": {}}
    status_path = HERE / "logs/controls_status.json"
    for shift in SHIFTS:
        for seed in SEEDS:
            run = f"dann_{shift}_s{seed}_lam0"
            cmd = [sys.executable, str(HERE / "scripts/train_uda.py"),
                   "--method", "dann", "--shift", shift, "--seed", str(seed),
                   "--epochs", "75", "--lambda-adv", "0", "--tag", "lam0"]
            log_path = HERE / "logs" / f"{run}.log"
            t0 = time.time()
            status["runs"][run] = {"state": "running", "command": " ".join(cmd),
                                   "started_at": time.strftime("%Y-%m-%d %H:%M:%S")}
            status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
            with log_path.open("w", encoding="utf-8") as lf:
                rc = subprocess.call(cmd, cwd=str(HERE), stdout=lf, stderr=subprocess.STDOUT)
            status["runs"][run].update({"state": "done" if rc == 0 else "failed",
                                        "returncode": rc,
                                        "seconds": round(time.time() - t0, 1),
                                        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S")})
            status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
            if rc != 0:
                print(f"[fail] {run} rc={rc}")
                return 1
    # 评估对照
    subprocess.call([sys.executable, str(HERE / "scripts/evaluate_controls.py")], cwd=str(HERE))
    print("controls done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
