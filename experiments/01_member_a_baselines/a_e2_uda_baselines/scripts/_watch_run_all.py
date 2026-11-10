#!/usr/bin/env python3
"""后台观察者：等待 run_all_status.json 进入终态（complete / completed_with_failures）。

每 10 分钟打印一次进度（完成 run 数），供 job_output 查看；退出码 0 表示矩阵结束。
"""
import json
import time
from pathlib import Path

p = Path("logs/run_all_status.json")
last_print = 0.0
while True:
    try:
        st = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        st = None
    if st:
        state = st.get("state")
        if state in ("complete", "completed_with_failures"):
            done = sum(1 for v in st["runs"].values() if v.get("state") in ("done", "skipped_complete"))
            print(f"FINAL state={state} runs_done={done}/{len(st['runs'])} "
                  f"failures={st.get('failures')} finished_at={st.get('finished_at')}", flush=True)
            raise SystemExit(0)
        now = time.time()
        if now - last_print > 600:
            last_print = now
            done = sum(1 for v in st["runs"].values() if v.get("state") in ("done", "skipped_complete"))
            running = [k for k, v in st["runs"].items() if v.get("state") == "running"]
            print(f"{time.strftime('%H:%M:%S')} state={state} done={done}/{len(st['runs'])} "
                  f"running={running}", flush=True)
    time.sleep(30)
