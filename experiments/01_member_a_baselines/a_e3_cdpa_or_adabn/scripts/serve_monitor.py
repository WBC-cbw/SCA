#!/usr/bin/env python3
"""A-E3 实时监控页：解析 logs/ 与 results/，在 127.0.0.1:8777 提供自动刷新页面。

用法：
    python scripts/serve_monitor.py            # 前台运行（Ctrl+C 退出）
    python scripts/serve_monitor.py --port 8777

页面每 5 秒自动刷新；只读日志与结果，不改任何实验文件。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
LOGS = HERE / "logs"
RES = HERE / "results"
ORDER = [(s, seed) for s in ("desync50", "desync100") for seed in (42, 43, 44)]
EPOCHS = 75
QUEUE_RE = re.compile(r"\[queue\] (START|END) (desync\d+) s(\d+)(.*)")

CSS = """
body{font-family:-apple-system,'Segoe UI',Roboto,'Microsoft YaHei',sans-serif;
     margin:24px;color:#1d2433;background:#f6f8fb}
h1{font-size:20px;margin:0 0 4px}
.sub{color:#6b7787;font-size:12px;margin-bottom:18px}
.cards{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:18px}
.card{background:#fff;border:1px solid #e3e8ef;border-radius:10px;padding:12px 16px;min-width:150px;
      box-shadow:0 1px 2px rgba(16,24,40,.04)}
.card .k{font-size:11px;color:#6b7787;letter-spacing:.02em}
.card .v{font-size:20px;font-weight:600;margin-top:4px}
table{border-collapse:collapse;background:#fff;border:1px solid #e3e8ef;border-radius:10px;
      overflow:hidden;width:100%;margin-bottom:18px}
th,td{padding:8px 10px;font-size:12.5px;text-align:right;border-bottom:1px solid #eef1f6}
th{background:#f1f4f9;color:#4b5565;font-weight:600;text-align:right}
th:first-child,td:first-child{text-align:left}
tr.done td{color:#6b7787}
tr.running td{background:#eef6ff}
.pill{display:inline-block;padding:2px 8px;border-radius:999px;font-size:11px;font-weight:600}
.p-done{background:#e7f6ec;color:#1a7f37}
.p-run{background:#e8f0fe;color:#1a56db}
.p-wait{background:#f0f2f5;color:#6b7787}
.bar{height:6px;background:#e6ebf2;border-radius:3px;overflow:hidden;width:110px;display:inline-block;
     vertical-align:middle;margin-left:8px}
.bar i{display:block;height:100%;background:#2f6fe4}
.charts{display:flex;gap:16px;flex-wrap:wrap}
.chart{background:#fff;border:1px solid #e3e8ef;border-radius:10px;padding:10px 12px}
.chart h3{margin:0 0 6px;font-size:12px;color:#4b5565;font-weight:600}
pre{background:#0f172a;color:#d6e2f5;padding:12px;border-radius:10px;font-size:11.5px;
    overflow:auto;max-height:240px;line-height:1.45}
.tag{font-size:11px;color:#6b7787;margin-left:6px}
"""


def read_json_lines(path: Path):
    out = []
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def queue_events():
    ev = {}
    p = LOGS / "queue.log"
    if not p.exists():
        return ev
    text = p.read_text(encoding="utf-8", errors="ignore")
    for m in QUEUE_RE.finditer(text):
        kind, shift, seed, rest = m.groups()
        ev[(shift, int(seed))] = (kind, rest.strip())
    return ev


def run_state(shift, seed, qev, now):
    name = f"cdpa_{shift}_s{seed}"
    log = LOGS / f"{name}.log"
    recs = read_json_lines(log)
    epochs = [r["epoch"] for r in recs if "epoch" in r]
    n = max(epochs) if epochs else 0
    last = next((r for r in reversed(recs) if "epoch" in r), None)
    secs = [r["sec"] for r in recs if "epoch" in r and isinstance(r.get("sec"), (int, float))]
    spe = sorted(secs[-5:])[len(secs[-5:]) // 2] if secs else None
    ge = [(r["epoch"], r["ge_probe"]) for r in recs
          if "epoch" in r and isinstance(r.get("ge_probe"), (int, float))]
    start = None
    if log.exists():
        start = datetime.fromtimestamp(os.path.getctime(log))
    pred = RES / name / "predictions_attack.npy"
    qend = qev.get((shift, seed), ("", ""))[0] == "END"
    done = pred.exists() or qend
    if done:
        status = "done"
    elif start and (n > 0 or (log.exists() and now - start < timedelta(minutes=10))):
        status = "running"
    else:
        status = "pending"
    ref = spe or 27.0
    eta = 0.0
    if status == "running":
        eta = max(0, EPOCHS - n) * ref + 25
    elif status == "pending":
        eta = EPOCHS * 27.0 + 25
    return {"name": name, "shift": shift, "seed": seed, "status": status, "epoch": n,
            "last": last, "spe": spe, "ge": ge, "start": start, "eta": eta,
            "elapsed": (now - start).total_seconds() if start else None, "log": log}


def svg_line(points, w, h, ymin, ymax, color, logy=False, xmax=EPOCHS):
    pts = [(x, y) for x, y in points if y is not None]
    if not pts:
        return f'<svg width="{w}" height="{h}"></svg>'
    def sx(x):
        return 34 + (w - 44) * (x / xmax)
    def sy(y):
        if logy:
            import math
            y = max(y, ymin)
            return h - 18 - (h - 30) * ((math.log10(y) - math.log10(ymin)) /
                                        max(1e-9, math.log10(ymax) - math.log10(ymin)))
        return h - 18 - (h - 30) * ((y - ymin) / max(1e-9, ymax - ymin))
    poly = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in pts)
    dots = "".join(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="1.8" fill="{color}"/>'
                   for x, y in pts)
    labels = []
    for frac in (0, 0.5, 1.0):
        if logy:
            import math
            v = 10 ** (math.log10(ymin) + frac * (math.log10(ymax) - math.log10(ymin)))
        else:
            v = ymin + frac * (ymax - ymin)
        ypx = h - 18 - frac * (h - 30)
        labels.append(f'<text x="30" y="{ypx + 3:.0f}" font-size="9" fill="#6b7787" '
                      f'text-anchor="end">{v:.0f}</text>')
    return (f'<svg width="{w}" height="{h}">'
            f'<polyline points="{poly}" fill="none" stroke="{color}" stroke-width="1.6"/>'
            f'{dots}{"".join(labels)}'
            f'<text x="34" y="{h-4}" font-size="9" fill="#6b7787">epoch 1</text>'
            f'<text x="{w-10}" y="{h-4}" font-size="9" fill="#6b7787" text-anchor="end">'
            f'{xmax}</text></svg>')


def fmt(v, nd=3):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def build_page():
    now = datetime.now()
    qev = queue_events()
    states = [run_state(s, seed, qev, now) for s, seed in ORDER]
    done_n = sum(1 for s in states if s["status"] == "done")
    running = next((s for s in states if s["status"] == "running"), None)
    started = next((s["start"] for s in states if s["start"]), None)
    elapsed = (now - started).total_seconds() if started else 0
    remaining = sum(s["eta"] for s in states if s["status"] != "done")
    eta_ts = now + timedelta(seconds=remaining)
    final_done = (LOGS / "FINALIZE_DONE.txt").exists()

    def hms(sec):
        sec = int(max(0, sec))
        return f"{sec // 3600}h{sec % 3600 // 60:02d}m"

    cards = [
        ("总进度", f"{done_n} / 6"),
        ("当前 run", running["name"] if running else ("已完成" if done_n == 6 else "等待启动")),
        ("已用时间", hms(elapsed) if started else "—"),
        ("预计剩余", hms(remaining)),
        ("预计完成", eta_ts.strftime("%H:%M") if started and not final_done else "—"),
        ("收尾流程", "完成" if final_done else ("运行中" if done_n == 6 else "等待队列")),
    ]
    card_html = "".join(
        f'<div class="card"><div class="k">{k}</div><div class="v">{v}</div></div>'
        for k, v in cards)

    rows = []
    for s in states:
        cls = s["last"].get("cls") if s["last"] else None
        mmd = s["last"].get("mmd") if s["last"] else None
        acc = s["last"].get("val_acc") if s["last"] else None
        spe = s["spe"]
        g = s["ge"][-1][1] if s["ge"] else None
        pill = {"done": '<span class="pill p-done">完成</span>',
                "running": '<span class="pill p-run">训练中</span>',
                "pending": '<span class="pill p-wait">排队</span>'}[s["status"]]
        pct = 100 * s["epoch"] / EPOCHS
        bar = (f'<span class="bar"><i style="width:{pct:.0f}%"></i></span>'
               f'<span class="tag">{s["epoch"]}/{EPOCHS}</span>')
        eta = "—" if s["status"] == "done" else hms(s["eta"])
        rows.append(
            f'<tr class="{s["status"]}"><td>{s["shift"]} · s{s["seed"]}</td><td>{pill}</td>'
            f'<td style="text-align:left">{bar}</td><td>{fmt(cls, 3)}</td><td>{fmt(mmd, 4)}</td>'
            f'<td>{fmt(acc, 4)}</td><td>{fmt(g)}</td><td>{fmt(spe, 1)}</td><td>{eta}</td></tr>')
    table = ('<table><tr><th>run</th><th>状态</th><th>epoch</th><th>CE loss</th>'
             '<th>MMD</th><th>val acc</th><th>GE probe</th><th>s/epoch</th><th>剩余</th></tr>'
             + "".join(rows) + "</table>")

    ge_svg = ce_svg = ""
    if running:
        ge_svg = svg_line(running["ge"], 460, 150, 0, 256, "#d62728")
        ce_svg = svg_line([(r["epoch"], r["cls"]) for r in read_json_lines(running["log"])
                           if "epoch" in r], 460, 150,
                          max(0.1, min([r["cls"] for r in read_json_lines(running["log"])
                                        if "epoch" in r] or [0.1]) * 0.8),
                          max([r["cls"] for r in read_json_lines(running["log"])
                               if "epoch" in r] or [5]) * 1.1, "#2f6fe4", logy=True)

    log_tail = ""
    src = running["log"] if running else (LOGS / "finalize.log")
    if src.exists():
        lines = src.read_text(encoding="utf-8", errors="ignore").splitlines()[-10:]
        log_tail = "\n".join(lines)

    html = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<meta http-equiv="refresh" content="5">
<title>A-E3 CDPA-MMD 实时监控</title><style>{CSS}</style></head><body>
<h1>A-E3 CDPA-MMD 实时监控</h1>
<div class="sub">最后更新 {now.strftime('%H:%M:%S')} · 每 5 秒自动刷新 · 数据源 logs/ 与 results/</div>
<div class="cards">{card_html}</div>
{table}
<div class="charts">
<div class="chart"><h3>当前 run 的 GE probe（500 条攻击轨迹，每 2 epoch）</h3>{ge_svg}</div>
<div class="chart"><h3>当前 run 的 CE loss（对数轴）</h3>{ce_svg}</div>
</div>
<h3 style="font-size:13px;color:#4b5565;margin:16px 0 6px">当前日志尾部</h3>
<pre>{log_tail.replace('&', '&amp;').replace('<', '&lt;')}</pre>
</body></html>"""
    return html


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/api/status"):
            body = json.dumps({"time": datetime.now().isoformat(timespec="seconds"),
                               "done": (LOGS / "QUEUE_DONE.txt").exists(),
                               "finalized": (LOGS / "FINALIZE_DONE.txt").exists()},
                              ensure_ascii=False).encode("utf-8")
            ctype = "application/json; charset=utf-8"
        else:
            body = build_page().encode("utf-8")
            ctype = "text/html; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8777)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"monitor: http://{args.host}:{args.port}/  (Ctrl+C to stop)", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
