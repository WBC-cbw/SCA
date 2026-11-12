#!/usr/bin/env python3
"""把 12 个 run 的坍塌监控压缩成一张表：首次 WARN/COLLAPSE、最大 top1 占比、WARN 次数。

history.json 里 collapse_flags 是与 monitor/epochs 对齐的字符串列表（OK / WARN / COLLAPSE）。
"""
from __future__ import annotations
import csv, json
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
RES = HERE / "results"
rows = []
for d in sorted(RES.glob("cdan_sca_*")):
    h = d / "history.json"
    if not h.exists():
        continue
    hist = json.loads(h.read_text(encoding="utf-8"))
    m = hist.get("monitor", [])
    epochs = hist.get("epochs", list(range(1, len(m) + 1)))
    flags = hist.get("collapse_flags", [])
    first_warn = next((epochs[i] for i, x in enumerate(flags) if x in ("WARN", "COLLAPSE")), None)
    first_collapse = next((epochs[i] for i, x in enumerate(flags) if x == "COLLAPSE"), None)
    top1s = [x["top1_share"] for x in m]
    row = {
        "run": d.name,
        "epochs": len(m),
        "max_top1_share": round(max(top1s), 4) if top1s else None,
        "argmax_epoch": int(epochs[top1s.index(max(top1s))]) if top1s else None,
        "final_top1_share": round(top1s[-1], 4) if top1s else None,
        "first_warn_epoch": first_warn,
        "first_collapse_epoch": first_collapse,
        "n_warn": sum(1 for x in flags if x == "WARN"),
        "n_collapse": sum(1 for x in flags if x == "COLLAPSE"),
    }
    rows.append(row)
out = RES / "collapse_summary.csv"
with out.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(out, len(rows), "runs")
