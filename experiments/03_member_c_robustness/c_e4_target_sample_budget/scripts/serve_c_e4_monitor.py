#!/usr/bin/env python3
"""C-E4 实时监控页（stdlib only）：http://127.0.0.1:8779/

只读：logs/<run>.log（每 epoch 一行 JSON）、results/<run>/history.json、
results/<run>/predictions_attack.npy（存在即完成）、logs/queue_d50_status.txt。
GE 曲线按 evaluate_ascad.ranks 同式重算（log 分数累积，step=100）。
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parents[1]
ROOT_OUT = HERE.parents[2] / "outputs"
PORT = int(os.environ.get("C_E4_MON_PORT", "8779"))
DATA = Path("D:/sca_project/ASCAD_data")
SHIFTS = {"desync50": DATA / "ASCAD_desync50.h5", "desync100": DATA / "ASCAD_desync100.h5"}
BYTE = 2
STEP = 100
BUDGETS = [50, 500, 5000]
SEEDS = [42, 43, 44]
METHODS = ["cdpa", "dann", "cdan_sca"]
SBOX = np.array([
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16,
], dtype=np.uint8)

_meta_cache = {}
_curve_cache = {}


def attack_meta(shift):
    if shift not in _meta_cache:
        with h5py.File(SHIFTS[shift], "r") as h:
            meta = np.asarray(h["Attack_traces/metadata"])
        _meta_cache[shift] = (np.asarray(meta["plaintext"]), np.asarray(meta["key"]))
    return _meta_cache[shift]


def ge_curve(probs, shift):
    plaintext, key = attack_meta(shift)
    scores = np.zeros(256, dtype=np.float64)
    out = []
    true = int(key[0, BYTE])
    vals = SBOX[np.bitwise_xor(plaintext[:, BYTE, None], np.arange(256, dtype=np.uint8))]
    for end in range(STEP, len(probs) + 1, STEP):
        p = np.clip(probs[end - STEP:end], 1e-36, 1.0)
        scores += np.log(p[np.arange(STEP)[:, None], vals[end - STEP:end]]).sum(axis=0)
        out.append((end, int(np.sum(scores > scores[true]))))
    return out


def summarize_curve(curve, budget):
    ge_at = {n: r for n, r in curve}
    zeros = [n for n, r in curve if r == 0]
    first = zeros[0] if zeros else None
    stable = None
    tail = None
    for n, r in reversed(curve):
        if r == 0:
            tail = n
        else:
            break
    if tail is not None and curve[-1][1] == 0:
        stable = tail
    return {"ntge": first, "ntge_stable": stable,
            "ge1k": ge_at.get(1000), "ge2k": ge_at.get(2000),
            "ge5k": ge_at.get(5000), "ge10k": ge_at.get(10000),
            "n_total": (first + budget) if first is not None else None}


def run_stats(shift, method, budget, seed, current=None, ahead=False):
    run = f"{method}_{shift}_b{budget}_s{seed}"
    rdir = HERE / "results" / run
    pred = rdir / "predictions_attack.npy"
    st = {"run": run, "method": method, "budget": budget, "seed": seed,
          "status": "pending", "epoch": None}
    log = HERE / "logs" / f"{run}.log"
    hist = rdir / "history.json"
    if pred.exists():
        st["status"] = "done"
        m = pred.stat().st_mtime
        c = _curve_cache.get(str(pred))
        if not c or c[0] < m:
            _curve_cache[str(pred)] = (m, summarize_curve(ge_curve(np.load(pred), shift), budget))
        st.update(_curve_cache[str(pred)][1])
        if hist.exists():
            try:
                h = json.loads(hist.read_text(encoding="utf-8"))
                st["epoch"] = len(h.get("epochs", []))
                es = h.get("epoch_seconds") or []
                st["train_seconds"] = round(float(sum(es)), 1) if es else None
            except Exception:
                pass
        return st
    if log.exists() and not ahead:
        age = time.time() - log.stat().st_mtime
        st["log_age_s"] = round(age)
        if current == run and age < 150:
            st["status"] = "running"
        else:
            st["status"] = "interrupted"
        last = None
        try:
            with log.open("rb") as f:
                f.seek(max(0, log.stat().st_size - 20000))
                block = f.read().decode("utf-8", "replace")
            for line in reversed(block.splitlines()):
                line = line.strip()
                if line.startswith("{"):
                    last = json.loads(line)
                    break
        except Exception:
            last = None
        if last:
            st["epoch"] = last.get("epoch")
            for k in ("cls", "mmd", "dom", "val_acc", "ge_probe", "top1", "H", "flag", "sec", "w"):
                if k in last:
                    st[k] = last[k]
    if hist.exists() and st.get("epoch") is None:
        try:
            h = json.loads(hist.read_text(encoding="utf-8"))
            st["epoch"] = len(h.get("epochs", []))
        except Exception:
            pass
    ck = ROOT_OUT / "models" / "c_e4" / run / "ckpt.pt"
    if st["status"] == "interrupted":
        st["resume"] = bool(ck.exists())
    return st


def queue_status_lines(shift):
    qlog = HERE / "logs" / ("queue_d50_status.txt" if shift == "desync50" else "queue_d100_status.txt")
    if not qlog.exists():
        return []
    raw = qlog.read_bytes()
    try:
        txt = raw.decode("utf-8")
    except UnicodeDecodeError:
        txt = raw.decode("gbk", errors="replace")
    return [ln.strip() for ln in txt.splitlines() if ln.strip()]


def current_run_name(shift):
    """队列日志里最后一条 start 若还没有 ok/FAIL 收尾，就是当前 run。"""
    cur = None
    for ln in queue_status_lines(shift):
        low = ln.lower()
        if " start " in low:
            tail = low.split(" start ", 1)[1].strip()
            cur = tail
        elif (" ok " in low or " fail " in low) and cur is not None:
            if low.endswith(cur):
                cur = None
    if not cur:
        return None
    parts = cur.split()
    if len(parts) != 3:
        return None
    tag, b, s = parts
    method = tag
    if method not in METHODS:
        return None
    return f"{method}_{shift}_b{int(b[1:])}_s{int(s[1:])}"


def plan(shift):
    return [(m, b, s) for b in BUDGETS for s in SEEDS for m in METHODS]


def status(shift="desync50"):
    current = current_run_name(shift)
    pl = plan(shift)
    cur_idx = None
    for i, (m, b, s) in enumerate(pl):
        if f"{m}_{shift}_b{b}_s{s}" == current:
            cur_idx = i
            break
    runs = []
    for i, (m, b, s) in enumerate(pl):
        ahead = cur_idx is not None and i > cur_idx
        runs.append(run_stats(shift, m, b, s, current, ahead))
    done = [r for r in runs if r["status"] == "done"]
    running = [r for r in runs if r["status"] == "running"]
    cells = {}
    for m in METHODS:
        for b in BUDGETS:
            got = [r for r in runs if r["method"] == m and r["budget"] == b and r["status"] == "done"]
            if len(got) != 3:
                continue
            nt = [10000 if r.get("ntge") is None else r["ntge"] for r in got]
            g10 = [r.get("ge10k") for r in got]
            ntot = [10000 + b if r.get("n_total") is None else r["n_total"] for r in got]
            cells[f"{m}|{b}"] = {
                "ntge_mean": float(np.mean(nt)),
                "stable_rate": sum(r.get("ntge_stable") is not None for r in got) / 3.0,
                "ge10k_mean": float(np.mean([x for x in g10 if x is not None])) if any(
                    x is not None for x in g10) else None,
                "n_total_mean": float(np.mean(ntot)),
            }
    tail = queue_status_lines(shift)[-8:]
    durs = [r["train_seconds"] for r in done if r.get("train_seconds")]
    per_run = float(np.median(durs)) if durs else 2000.0
    remaining = len(runs) - len(done)
    cur = running[0] if running else None
    if cur and cur.get("epoch") and cur.get("sec"):
        eta_s = (75 - cur["epoch"]) * float(cur["sec"]) + max(0, remaining - 1) * per_run
    else:
        eta_s = remaining * per_run if remaining else 0.0
    return {"shift": shift, "now": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "n_done": len(done), "n_total": len(runs), "per_run_s": per_run,
            "eta_seconds": eta_s,
            "eta_at": (datetime.now() + timedelta(seconds=eta_s)).strftime("%m-%d %H:%M") if eta_s else None,
            "current": current, "runs": runs, "cells": cells, "queue_tail": tail}


HTML = """<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>C-E4 目标域样本量敏感性 · 实时监控</title>
<style>
:root{--bg:#0e1116;--fg:#e6edf3;--dim:#8b949e;--card:#161b22;--line:#30363d;--ok:#3fb950;--run:#58a6ff}
*{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--fg);font:13px/1.55 -apple-system,Segoe UI,Roboto,Helvetica,Arial,"Microsoft YaHei",sans-serif}
header{padding:14px 20px;border-bottom:1px solid var(--line);display:flex;gap:18px;align-items:baseline;flex-wrap:wrap}
h1{font-size:16px;margin:0;font-weight:600} .dim{color:var(--dim)} .big{font-size:20px;font-weight:600}
main{padding:16px 20px;display:grid;gap:16px}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px 14px}
.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}
th,td{padding:4px 8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th{color:var(--dim);font-weight:500} td:first-child,th:first-child{text-align:left}
tr.running td{background:#0d2a4a} tr.done td{background:#0f2b18} tr.interrupted td{background:#3a1d1d}
.pill{display:inline-block;padding:1px 7px;border-radius:10px;font-size:11px}
.p-done{background:#1f6f3f;color:#d7ffe6} .p-run{background:#1f4f8f;color:#d7e9ff}
.p-pend{background:#30363d;color:#9aa4af} .p-int{background:#7d2a2a;color:#ffd7d7}
.bar{height:10px;background:#21262d;border-radius:5px;overflow:hidden;margin:6px 0}
.bar>i{display:block;height:100%;background:linear-gradient(90deg,#1f6feb,#58a6ff)}
.kv{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}
.kv div{background:#0d1117;border:1px solid var(--line);border-radius:6px;padding:8px}
.kv b{display:block;font-size:16px;margin-top:2px}
small{color:var(--dim)} pre{white-space:pre-wrap;margin:6px 0 0;color:var(--dim)}
</style></head><body>
<header><h1>C-E4 · 目标域样本量敏感性（desync50 第一阶段）</h1>
<span class="dim">刷新于 <span id="now">—</span></span><span class="dim">每 5 s 自动刷新</span>
<span><a href="?shift=desync50" style="color:#58a6ff">desync50</a> · <a href="?shift=desync100" style="color:#58a6ff">desync100</a></span></header>
<main>
<div class="card" id="top"></div>
<div class="grid"><div class="card" id="cur"></div><div class="card" id="aggregate" style="grid-column:span 2"></div></div>
<div class="card" id="runs"></div><div class="card" id="tail"></div>
</main>
<script>
var L={cdpa:"CDPA-MMD",dann:"DANN",cdan_sca:"CDAN-SCA full"};
function fmt(x,d){if(x===null||x===undefined)return "—";if(typeof x==="number")return x.toFixed(d||0);return String(x);}
function pill(s){var m={done:["p-done","完成"],running:["p-run","训练中"],pending:["p-pend","等待"],interrupted:["p-int","中断"]}[s]||["p-pend",s];
 return '<span class="pill '+m[0]+'">'+m[1]+'</span>';}
var SHIFT=new URLSearchParams(location.search).get("shift")||"desync50";
function tick(){
 fetch("/api/status?shift="+SHIFT).then(function(r){return r.json();}).then(function(d){
  document.getElementById("now").textContent=d.now;
  var cur=d.runs.find(function(x){return x.status==="running";})||d.runs.find(function(x){return x.status==="interrupted";});
  var etaTxt=d.eta_at?("预计全部完成 "+d.eta_at):"";
  document.getElementById("top").innerHTML='<div style="display:flex;gap:28px;flex-wrap:wrap;align-items:baseline">'+
   '<div><span class="dim">总进度</span><div class="big">'+d.n_done+' / '+d.n_total+' runs</div></div>'+
   '<div><span class="dim">当前</span><div class="big">'+(cur?cur.run:"—")+'</div></div>'+
   '<div><span class="dim">单 run 中位数</span><div class="big">'+(d.per_run_s/60).toFixed(1)+' min</div></div>'+
   '<div><span class="dim">剩余</span><div class="big">'+(d.eta_seconds?Math.round(d.eta_seconds/3600)+" h":"—")+'</div></div>'+
   '<div class="dim">'+etaTxt+'</div></div>';
  if(cur){
   var ep=cur.epoch||0,pct=Math.min(100,ep/75*100);
   document.getElementById("cur").innerHTML='<div style="display:flex;justify-content:space-between"><b>'+cur.run+'</b>'+pill(cur.status)+'</div>'+
    '<div class="bar"><i style="width:'+pct+'%"></i></div><small>epoch '+ep+'/75 · '+pct.toFixed(0)+'%</small>'+
    '<div class="kv" style="margin-top:8px">'+
    '<div><span class="dim">cls loss</span><b>'+fmt(cur.cls,3)+'</b></div>'+
    '<div><span class="dim">'+(cur.mmd!=null?"MMD":"dom loss")+'</span><b>'+fmt(cur.mmd!=null?cur.mmd:cur.dom,3)+'</b></div>'+
    '<div><span class="dim">val acc</span><b>'+fmt(cur.val_acc,4)+'</b></div>'+
    '<div><span class="dim">GE probe</span><b>'+fmt(cur.ge_probe)+'</b></div>'+
    '<div><span class="dim">target top-1</span><b>'+fmt(cur.top1,3)+'</b></div>'+
    '<div><span class="dim">pred entropy</span><b>'+fmt(cur.H,3)+'</b></div>'+
    '<div><span class="dim">s / epoch</span><b>'+fmt(cur.sec,1)+'</b></div>'+
    '<div><span class="dim">flag</span><b>'+(cur.flag||"—")+'</b></div></div>';
  } else {document.getElementById("cur").innerHTML="<b>当前无运行中的 run</b>";}
  var rows="";
  Object.keys(d.cells).forEach(function(k){
   var p=k.split("|"),v=d.cells[k];
   var cls=v.stable_rate>=1?"p-done":(v.stable_rate>0?"p-run":"p-pend");
   rows+='<tr><td>'+(L[p[0]]||p[0])+'</td><td>'+p[1]+'</td><td>'+fmt(v.ntge_mean)+'</td>'+
    '<td><span class="pill '+cls+'">'+Math.round(v.stable_rate*3)+'/3</span></td>'+
    '<td>'+fmt(v.ge10k_mean,2)+'</td><td>'+fmt(v.n_total_mean)+'</td></tr>';
  });
  document.getElementById("aggregate").innerHTML='<b>已齐 3 seed 的 cell（NTGE 截尾 10k；N_total = NTGE + budget）</b>'+
   '<table><thead><tr><th>方法</th><th>budget</th><th>NTGE 均值</th><th>稳定 GE=0</th><th>GE@10k</th><th>N_total@GE=1</th></tr></thead><tbody>'+
   (rows||'<tr><td colspan="6" class="dim">还没有齐 3 seed 的 cell</td></tr>')+'</tbody></table>';
  var t='<table><thead><tr><th>run</th><th>状态</th><th>epoch</th><th>NTGE</th><th>stable</th><th>GE@1k</th><th>GE@5k</th><th>GE@10k</th><th>N_total</th><th>训练秒</th></tr></thead><tbody>';
  d.runs.forEach(function(r){
   t+='<tr class="'+r.status+'"><td>'+r.run+'</td><td>'+pill(r.status)+'</td><td>'+(r.epoch==null?"—":r.epoch)+'</td>'+
    '<td>'+fmt(r.ntge)+'</td><td>'+(r.ntge_stable==null?"—":"✓")+'</td><td>'+fmt(r.ge1k)+'</td><td>'+fmt(r.ge5k)+'</td>'+
    '<td>'+fmt(r.ge10k)+'</td><td>'+fmt(r.n_total)+'</td><td>'+fmt(r.train_seconds)+'</td></tr>';
  });
  document.getElementById("runs").innerHTML=t+'</tbody></table>';
  document.getElementById("tail").innerHTML='<b>队列日志（尾部）</b><pre>'+((d.queue_tail||[]).join("\\n"))+'</pre>';
 });
}
tick();setInterval(tick,5000);
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/api/status"):
            shift = "desync50"
            if "shift=" in self.path:
                shift = self.path.split("shift=")[1].split("&")[0]
            body = json.dumps(status(shift), ensure_ascii=False, default=str).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return
        body = HTML.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    print("C-E4 monitor on http://127.0.0.1:%d/" % PORT)
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
