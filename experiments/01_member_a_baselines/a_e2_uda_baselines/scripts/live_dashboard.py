#!/usr/bin/env python3
"""A-E2 实时看板：每 5 秒生成 reports/live.html（浏览器 meta refresh 自动刷新）。

显示：矩阵状态、当前 run/epoch/进度条/ETA、当前 run 的 loss 与 GE probe 火花线、
已完成 run 的 GE/NTGE/置信度指标、训练日志尾部。
用法：
  python scripts/live_dashboard.py            # 持续运行，矩阵结束后再刷 2 分钟退出
  python scripts/live_dashboard.py --once     # 只生成一次
"""
from __future__ import annotations

import argparse
import html
import json
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
STATUS = HERE / "logs/run_all_status.json"
OUT = HERE / "reports/live.html"
METHODS = [("dann", "DANN"), ("cdan", "CDAN")]
SHIFTS = ["desync50", "desync100"]
SEEDS = [42, 43, 44]
COLOR = {"done": "#2e7d32", "running": "#1565c0", "pending": "#9e9e9e",
         "failed": "#c62828", "skipped_complete": "#6a1b9a"}


def read_json(p, default=None):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:
        return default


def fmt(v, nd=1):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def sparkline(vals, w=560, h=64, color="#1e88e5"):
    vals = [v for v in vals if v is not None]
    if len(vals) < 2:
        return f'<svg width="{w}" height="{h}"></svg>'
    lo, hi = min(vals), max(vals)
    if hi == lo:
        hi = lo + 1
    n = len(vals)
    pts = []
    for i, v in enumerate(vals):
        x = i * (w - 6) / (n - 1) + 3
        y = h - 3 - (v - lo) / (hi - lo) * (h - 6)
        pts.append(f"{x:.1f},{y:.1f}")
    return (f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}">'
            f'<polyline fill="none" stroke="{color}" stroke-width="1.6" points="{" ".join(pts)}"/>'
            f'<text x="4" y="12" font-size="10" fill="#666">{hi:.0f}</text>'
            f'<text x="4" y="{h-4}" font-size="10" fill="#666">{lo:.0f}</text></svg>')


def maybe_eval(done_runs):
    """有完成但未评估的 run 时，静默跑一次 evaluate_uda.py（CPU，约 2 s）。"""
    missing = False
    for run in done_runs:
        try:
            method, shift, seed_s = run.split("_")
        except ValueError:
            continue
        if not (HERE / "results/eval" / shift / f"{run}.json").exists():
            missing = True
            break
    if missing and (HERE / "scripts/evaluate_uda.py").exists():
        subprocess.run([sys.executable, str(HERE / "scripts/evaluate_uda.py")],
                       cwd=str(HERE), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def build(st):
    now = datetime.now()
    runs = st["runs"]; order = st.get("order", list(runs))
    done = [r for r in order if runs[r].get("state") in ("done", "skipped_complete")]
    running = [r for r in order if runs[r].get("state") == "running"]
    pending = [r for r in order if runs[r].get("state") == "pending"]
    total = len(order)
    finished = st.get("state") in ("complete", "completed_with_failures")

    # 预估
    done_secs = [runs[r]["seconds"] for r in done if runs[r].get("seconds")]
    mean_run = sum(done_secs) / len(done_secs) if done_secs else 1980.0
    cur = running[0] if running else None
    cur_hist = read_json(HERE / "results" / cur / "history.json") if cur else None
    cur_epoch = len(cur_hist["epochs"]) if cur_hist else 0
    remaining_epochs = 75 - cur_epoch
    ep_sec = (sum(cur_hist["epoch_seconds"][-5:]) / len(cur_hist["epoch_seconds"][-5:])
              if cur_hist and cur_hist["epoch_seconds"] else 26.5)
    eta_sec = remaining_epochs * ep_sec + max(0, len(pending) - 1) * mean_run if cur else len(pending) * mean_run
    eta = now + timedelta(seconds=eta_sec) if not finished else None

    # 当前 run 刚结束的 epoch 数据
    last = {}
    if cur_hist and cur_hist["epochs"]:
        i = -1
        last = {"epoch": cur_hist["epochs"][i], "cls": cur_hist["cls_loss"][i],
                "dom": cur_hist["dom_loss"][i], "dsrc": cur_hist["d_acc_src"][i],
                "dtgt": cur_hist["d_acc_tgt"][i], "val": cur_hist["val_acc"][i],
                "probe": cur_hist["ge_probe"][i]}

    # 卡片
    cards = [
        ("矩阵状态", "运行中" if not finished else "已结束", COLOR["running"] if not finished else COLOR["done"]),
        ("完成 / 总数", f"{len(done)} / {total}", "#333"),
        ("当前 run", cur or "—", COLOR["running"] if cur else "#666"),
        ("当前 epoch", f"{cur_epoch} / 75" if cur else "—", "#333"),
        ("平均单 run", f"{mean_run/60:.1f} min", "#333"),
        ("预计完成", eta.strftime("%H:%M") if eta else "已结束", "#333"),
    ]
    card_html = "".join(
        f'<div class="card"><div class="k">{html.escape(k)}</div>'
        f'<div class="v" style="color:{c}">{html.escape(str(v))}</div></div>'
        for k, v, c in cards)

    # 进度条
    pct = 100.0 * (len(done) + (cur_epoch / 75 if cur else 0)) / max(total, 1)
    prog = (f'<div class="bar"><div class="fill" style="width:{pct:.1f}%"></div></div>'
            f'<div class="small">{pct:.1f}% · 当前 {html.escape(cur or "")} epoch {cur_epoch}/75</div>')

    # 当前 run 火花线
    sparks = ""
    if cur_hist and cur_hist["epochs"]:
        sparks = (
            '<div class="spark"><div class="lab">当前 run 的 GE probe（500 条攻击轨迹，每 2 epoch，越低越好）</div>'
            + sparkline(cur_hist["ge_probe"], color="#e53935") +
            '</div><div class="spark"><div class="lab">分类损失（源域）</div>'
            + sparkline(cur_hist["cls_loss"], color="#1e88e5") +
            '</div><div class="spark"><div class="lab">批内域判别准确率（源 / 目标）</div>'
            + sparkline(cur_hist["d_acc_src"], color="#43a047") +
            sparkline(cur_hist["d_acc_tgt"], color="#fb8c00") + "</div>")

    # 矩阵表
    rows = []
    for method, name in METHODS:
        for shift in SHIFTS:
            cells = []
            for seed in SEEDS:
                run = f"{method}_{shift}_s{seed}"
                rs = runs.get(run, {})
                state = rs.get("state", "?")
                ev = read_json(HERE / "results/eval" / shift / f"{run}.json")
                if ev:
                    detail = (f'GE@10k <b>{fmt(ev["ge_at_10000"])}</b><br>'
                              f'NTGE {fmt(ev["ntge_first"])}<br>'
                              f'acc {ev["attack_accuracy"]*100:.2f}%')
                elif state == "done":
                    detail = "完成，待评估"
                elif state == "running":
                    detail = f'epoch {cur_epoch}/75'
                else:
                    detail = "排队"
                cells.append(
                    f'<td class="cell" style="border-left:4px solid {COLOR.get(state, "#999")}">'
                    f'<div class="st" style="color:{COLOR.get(state, "#999")}">{html.escape(state)}</div>'
                    f'<div class="small">{detail}</div></td>')
            rows.append(f'<tr><th>{name} {shift}</th>{"".join(cells)}</tr>')
    matrix = ('<table class="matrix"><thead><tr><th></th>'
              + "".join(f"<th>seed {s}</th>" for s in SEEDS) + '</tr></thead><tbody>'
              + "".join(rows) + '</tbody></table>')

    # 已完成 run 指标表
    trs = []
    for run in done:
        parts = run.split("_")
        if len(parts) != 3:
            continue
        method, shift, seed = parts
        ev = read_json(HERE / "results/eval" / shift / f"{run}.json")
        rs = runs[run]
        if ev:
            trs.append(f'<tr><td>{html.escape(run)}</td><td>{rs.get("seconds", 0)/60:.1f}</td>'
                       f'<td>{fmt(ev["ge_at_1000"])}</td><td>{fmt(ev["ge_at_10000"])}</td>'
                       f'<td>{fmt(ev["ntge_first"])}</td><td>{fmt(ev["ntge_stable"])}</td>'
                       f'<td>{ev["attack_accuracy"]*100:.2f}%</td>'
                       f'<td>{ev["mean_max_softmax"]:.3f}</td>'
                       f'<td>{fmt(ev.get("domain_balanced_before"), 3)} → {fmt(ev.get("domain_balanced_after"), 3)}</td></tr>')
        else:
            trs.append(f'<tr><td>{html.escape(run)}</td><td>{rs.get("seconds", 0)/60:.1f}</td>'
                       f'<td colspan="7" class="small">等待评估</td></tr>')
    metrics = ('<table class="metrics"><thead><tr><th>run</th><th>min</th><th>GE@1k</th>'
               '<th>GE@10k</th><th>NTGE first</th><th>NTGE stable</th><th>acc</th>'
               '<th>mean max prob</th><th>D balanced before→after</th></tr></thead><tbody>'
               + "".join(trs) + '</tbody></table>')

    # 日志尾部
    log_tail = ""
    if cur:
        lp = HERE / "logs" / f"{cur}.log"
        try:
            lines = lp.read_text(encoding="utf-8", errors="replace").splitlines()[-12:]
            log_tail = "<pre>" + html.escape("\n".join(lines)) + "</pre>"
        except Exception:
            log_tail = "<pre>（日志读取失败）</pre>"

    return f"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<meta http-equiv="refresh" content="5">
<title>A-E2 实时状态</title>
<style>
body{{margin:0;background:#fafafa;color:#222;font-family:"Segoe UI","Microsoft YaHei",sans-serif}}
.wrap{{max-width:1100px;margin:0 auto;padding:18px 22px 40px}}
h1{{font-size:22px;margin:0 0 2px}} h2{{font-size:16px;margin:26px 0 8px;color:#444}}
.sub{{color:#777;font-size:12px;margin-bottom:14px}}
.cards{{display:flex;gap:10px;flex-wrap:wrap}}
.card{{background:#fff;border:1px solid #e4e4e4;border-radius:8px;padding:10px 14px;min-width:150px}}
.card .k{{font-size:11px;color:#888}} .card .v{{font-size:18px;font-weight:600;margin-top:2px}}
.bar{{height:12px;background:#e8e8e8;border-radius:6px;overflow:hidden;margin:8px 0 4px}}
.fill{{height:100%;background:linear-gradient(90deg,#42a5f5,#1e88e5);transition:width .4s}}
.small{{font-size:12px;color:#666}}
table{{border-collapse:collapse;background:#fff;width:100%;font-size:13px}}
th,td{{border:1px solid #e6e6e6;padding:6px 8px;text-align:left;vertical-align:top}}
thead th{{background:#f3f4f6;font-weight:600}}
.matrix td.cell{{width:26%}} .st{{font-size:11px;font-weight:600;margin-bottom:2px}}
.spark{{background:#fff;border:1px solid #eee;border-radius:8px;padding:8px 10px;margin:6px 0}}
.lab{{font-size:12px;color:#777;margin-bottom:4px}}
pre{{background:#1e1e1e;color:#d4d4d4;padding:10px;border-radius:8px;overflow:auto;font-size:12px;line-height:1.45}}
.foot{{color:#999;font-size:11px;margin-top:24px}}
</style></head><body><div class="wrap">
<h1>A-E2 DANN / CDAN 训练矩阵 · 实时状态</h1>
<div class="sub">更新时间 {now.strftime("%Y-%m-%d %H:%M:%S")} · 每 5 秒自动刷新 · 页面 {html.escape(str(OUT))}</div>
<div class="cards">{card_html}</div>
<div class="bar-wrap">{prog}</div>
<h2>训练矩阵</h2>
{matrix}
<h2>当前 run 训练动态</h2>
{sparks if sparks else '<div class="small">当前没有运行中的 run。</div>'}
<h2>已完成 run 的评估指标（NTGE 未达成留空）</h2>
{metrics}
<h2>当前 run 日志尾部</h2>
{log_tail if log_tail else '<div class="small">—</div>'}
<div class="foot">数据源：logs/run_all_status.json、results/*/history.json、results/eval/。
指标口径与 evaluate_uda.py 一致：GE@10k、NTGE(first)、domain balanced accuracy。</div>
</div></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--interval", type=float, default=5.0)
    args = ap.parse_args()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    idle_since = None
    while True:
        st = read_json(STATUS)
        if not st:
            time.sleep(args.interval)
            continue
        done = [r for r, v in st["runs"].items() if v.get("state") in ("done", "skipped_complete")]
        try:
            maybe_eval(done)
        except Exception:
            pass
        tmp = OUT.with_suffix(".html.tmp")
        tmp.write_text(build(st), encoding="utf-8")
        tmp.replace(OUT)
        state = st.get("state")
        if state in ("complete", "completed_with_failures"):
            if idle_since is None:
                idle_since = time.time()
            elif time.time() - idle_since > 120:
                break
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
