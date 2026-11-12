#!/usr/bin/env python3
"""B-E3 实时监控页：解析 logs/ 与 results/，在 127.0.0.1:8778 提供自动刷新页面。

用法: python scripts/serve_monitor.py [--port 8778]
只读日志与结果，不改任何实验文件。
"""
from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
LOGS = HERE / "logs"
RES = HERE / "results"
EPOCHS = 75
RUNS = [(mode, shift, seed) for mode in ("full", "marginal")
        for shift in ("desync50", "desync100") for seed in (42, 43, 44)]
QRE = re.compile(r"\[queue\] (START|END) (full|marginal) (desync\d+) s(\d+)")

CSS = """
body{font-family:-apple-system,'Segoe UI',Roboto,'Microsoft YaHei',sans-serif;margin:24px;
     color:#1d2433;background:#f6f8fb}
h1{font-size:20px;margin:0 0 4px}.sub{color:#6b7787;font-size:12px;margin-bottom:16px}
.cards{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:16px}
.card{background:#fff;border:1px solid #e3e8ef;border-radius:10px;padding:10px 14px;min-width:140px}
.card .k{font-size:11px;color:#6b7787}.card .v{font-size:19px;font-weight:600;margin-top:3px}
table{border-collapse:collapse;background:#fff;border:1px solid #e3e8ef;border-radius:10px;
      overflow:hidden;width:100%;margin-bottom:16px}
th,td{padding:6px 8px;font-size:12px;text-align:right;border-bottom:1px solid #eef1f6}
th{background:#f1f4f9;color:#4b5565;text-align:right;font-weight:600}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2),th:nth-child(3),td:nth-child(3){text-align:left}
tr.running td{background:#eef6ff}tr.done td{color:#6b7787}
.pill{display:inline-block;padding:1px 7px;border-radius:999px;font-size:10.5px;font-weight:600}
.p-done{background:#e7f6ec;color:#1a7f37}.p-run{background:#e8f0fe;color:#1a56db}
.p-wait{background:#f0f2f5;color:#6b7787}.p-warn{background:#fef3e2;color:#b54708}
.p-col{background:#fee4e2;color:#b42318}
.chart{background:#fff;border:1px solid #e3e8ef;border-radius:10px;padding:10px 12px;margin-bottom:14px}
h3{margin:0 0 6px;font-size:12px;color:#4b5565}
pre{background:#0f172a;color:#d6e2f5;padding:10px;border-radius:10px;font-size:11px;
    overflow:auto;max-height:200px;line-height:1.4}
"""


def read_json_lines(path: Path):
    out = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def queue_events():
    ev = {}
    p = LOGS / "queue.log"
    if p.exists():
        for m in QRE.finditer(p.read_text(encoding="utf-8", errors="ignore")):
            kind, mode, shift, seed = m.groups()
            ev[(mode, shift, int(seed))] = kind
    return ev


def run_state(mode, shift, seed, qev, now):
    tag = "full" if mode == "full" else "marg"
    name = "cdan_sca_%s_%s_s%d" % (tag, shift, seed)
    log = LOGS / (name + ".log")
    recs = [r for r in read_json_lines(log) if "epoch" in r]
    n = max((r["epoch"] for r in recs), default=0)
    last = recs[-1] if recs else None
    secs = [r["sec"] for r in recs if isinstance(r.get("sec"), (int, float))]
    spe = sorted(secs[-5:])[len(secs[-5:]) // 2] if secs else None
    ge = [(r["epoch"], r["ge_probe"]) for r in recs if isinstance(r.get("ge_probe"), (int, float))]
    start = datetime.fromtimestamp(os.path.getctime(log)) if log.exists() else None
    done = (RES / name / "run_meta.json").exists() or qev.get((mode, shift, seed)) == "END"
    if done:
        status = "done"
    elif n > 0 or (log.exists() and start and now - start < timedelta(minutes=10)):
        status = "running"
    else:
        status = "pending"
    eta = 0 if done else (EPOCHS - n) * (spe or 24.0) + 20
    flag = last.get("flag", "") if last else ""
    return {"name": name, "mode": mode, "shift": shift, "seed": seed, "status": status,
            "epoch": n, "last": last, "spe": spe, "ge": ge, "eta": eta, "flag": flag}


def svg(points, w, h, ymin, ymax, color, xmax=EPOCHS):
    pts = [(x, y) for x, y in points if y is not None]
    if not pts:
        return '<svg width="%d" height="%d"></svg>' % (w, h)

    def sx(x):
        return 36 + (w - 46) * (x / float(xmax))

    def sy(y):
        return h - 20 - (h - 32) * ((y - ymin) / max(ymax - ymin, 1e-9))

    d = " ".join(("M" if i == 0 else "L") + "%.1f,%.1f" % (sx(x), sy(y))
                 for i, (x, y) in enumerate(pts))
    grid = "".join('<line x1="36" y1="%.1f" x2="%d" y2="%.1f" stroke="#eef1f6"/>'
                   % (sy(v), w - 10, sy(v)) for v in (ymin, (ymin + ymax) / 2, ymax))
    labs = "".join('<text x="4" y="%.1f" font-size="9" fill="#6b7787">%.2f</text>'
                   % (sy(v) + 3, v) for v in (ymin, (ymin + ymax) / 2, ymax))
    return ('<svg width="%d" height="%d">%s%s<path d="%s" fill="none" stroke="%s" '
            'stroke-width="1.6"/></svg>' % (w, h, grid, labs, d, color))


def fmt(v, spec="{:.3f}"):
    return "" if v is None else spec.format(v)


def page():
    now = datetime.now()
    qev = queue_events()
    states = [run_state(m, s, sd, qev, now) for m, s, sd in RUNS]
    done = sum(1 for s in states if s["status"] == "done")
    running = next((s for s in states if s["status"] == "running"), None)
    total_eta = sum(s["eta"] for s in states if s["status"] != "done")
    done_spe = [s["spe"] for s in states if s["spe"]]
    ref = sorted(done_spe)[len(done_spe) // 2] if done_spe else 24.0

    rows = []
    for s in states:
        cls = "running" if s["status"] == "running" else ("done" if s["status"] == "done" else "")
        pill = {"done": '<span class="pill p-done">done</span>',
                "running": '<span class="pill p-run">running</span>',
                "pending": '<span class="pill p-wait">pending</span>'}[s["status"]]
        f = s["flag"]
        fl = ('<span class="pill p-col">COLLAPSE</span>' if f == "COLLAPSE"
              else '<span class="pill p-warn">WARN</span>' if f == "WARN" else "")
        last = s["last"] or {}
        rows.append(
            "<tr class='%s'><td>%s</td><td>%s</td><td>%s</td><td>%s %s</td><td>%d/75</td>"
            "<td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%dm</td></tr>" % (
                cls, s["name"], s["mode"], s["shift"], pill, fl, s["epoch"],
                fmt(last.get("top1")), fmt(last.get("H"), "{:.2f}"),
                fmt(last.get("w"), "{:.2f}"),
                "" if last.get("ge_probe") is None else str(last.get("ge_probe")),
                "" if not s["spe"] else "%.1f" % s["spe"], int(s["eta"] / 60)))
    table = ("<table><tr><th>run</th><th>mode</th><th>shift</th><th>status</th><th>epoch</th>"
             "<th>top1</th><th>H</th><th>w</th><th>GE probe</th><th>s/ep</th><th>ETA</th></tr>"
             + "".join(rows) + "</table>")

    cur = running or (states[done - 1] if done else None)
    charts = ""
    if cur:
        recs = [r for r in read_json_lines(LOGS / (cur["name"] + ".log")) if "epoch" in r]
        if recs:
            x = [r["epoch"] for r in recs]
            top1 = svg(list(zip(x, [r.get("top1") for r in recs])), 520, 170, 0, 0.6, "#6a3d9a")
            ent = svg(list(zip(x, [r.get("H") for r in recs])), 520, 170, 0, 5.6, "#2ca02c")
            gx = [r["epoch"] for r in recs if r.get("ge_probe") is not None]
            gy = [r["ge_probe"] for r in recs if r.get("ge_probe") is not None]
            ge = svg(list(zip(gx, gy)), 520, 170, 0, 256, "#d62728")
            charts = ('<div class="chart"><h3>%s: target top-1 share (WARN 0.30 / COLLAPSE 0.50)'
                      '</h3>%s</div><div class="chart"><h3>target prediction entropy (nats)'
                      '</h3>%s</div><div class="chart"><h3>GE probe (500 attack traces)'
                      '</h3>%s</div>' % (cur["name"], top1, ent, ge))

    lp = LOGS / "queue.log"
    tail = lp.read_text(encoding="utf-8", errors="ignore").strip().splitlines()[-8:] if lp.exists() else []
    html = """<!doctype html><html><head><meta charset="utf-8">
<meta http-equiv="refresh" content="10"><title>B-E3 CDAN-SCA monitor</title>
<style>%s</style></head><body>
<h1>B-E3 CDAN-SCA full / ablation-1 training monitor</h1>
<div class="sub">%s &middot; refresh 10s &middot; 12 runs &middot; ref %.1f s/epoch</div>
<div class="cards">
<div class="card"><div class="k">done</div><div class="v">%d/12</div></div>
<div class="card"><div class="k">current</div><div class="v" style="font-size:14px">%s</div></div>
<div class="card"><div class="k">remaining (est.)</div><div class="v">%dh%02dm</div></div>
<div class="card"><div class="k">epoch</div><div class="v">%s</div></div>
</div>
%s
%s
<h3 style="font-size:12px;color:#4b5565">queue.log</h3><pre>%s</pre>
</body></html>""" % (CSS, now.strftime("%Y-%m-%d %H:%M:%S"), ref, done,
                    running["name"] if running else "-", int(total_eta / 3600),
                    int(total_eta % 3600 / 60), running["epoch"] if running else "-",
                    table, charts, "\n".join(tail))
    return html


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8778)
    args = ap.parse_args()

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            body = page().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    print("http://127.0.0.1:%d/" % args.port, flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), H).serve_forever()


if __name__ == "__main__":
    main()
