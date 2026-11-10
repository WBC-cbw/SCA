#!/usr/bin/env python3
"""从 results/summary.json 与 results/eval/eval_all.json 生成 README 用的 markdown 表。

输出 reports/tables.md（同时打印到 stdout），包含：
  1) 逐 run 明细（方法 x 偏移 x seed）
  2) 方法 x 偏移 的均值±std 主表
  3) 同 seed 配对差（相对 Source Only）
  4) 域判别诊断（balanced acc before/after）
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
RES = HERE / "results"
SUMMARY = json.loads((RES / "summary.json").read_text(encoding="utf-8"))
EVAL = json.loads((RES / "eval/eval_all.json").read_text(encoding="utf-8"))

METHODS = ["source_only", "dann", "cdan"]
LABEL = {"source_only": "Source Only", "dann": "DANN", "cdan": "CDAN"}
SHIFTS = ["desync50", "desync100"]


def fmt(v, nd=1):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def ntge(v):
    return "—" if v is None else str(v)


def mean_std(m, key):
    v, s = m.get(key + "_mean"), m.get(key + "_std")
    if v is None:
        return "—"
    return f"{v:.1f} ± {s:.1f}"


lines = []
w = lines.append

w("### 逐 run 明细")
w("")
w("| 偏移 | 方法 | seed | NTGE(first) | NTGE(stable) | N_total@GE=1 | GE@1k | GE@10k | min GE | transient | acc | conf | H |")
w("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for shift in SHIFTS:
    for method in METHODS:
        rows = sorted([r for r in EVAL if r["method"] == method and r["shift"] == shift],
                      key=lambda r: r["seed"])
        for r in rows:
            n_total = r["n_total_at_ge1"]
            w(f"| {shift} | {LABEL[method]} | {r['seed']} | {ntge(r['ntge_first'])} | "
              f"{ntge(r['ntge_stable'])} | {ntge(n_total)} | {ntge(r['ge_at_1000'])} | "
              f"{ntge(r['ge_at_10000'])} | {r['min_ge']} | {'是' if r['transient_only'] else '否'} | "
              f"{r['attack_accuracy']*100:.2f}% | {r['mean_max_softmax']:.3f} | {r['mean_entropy_nats']:.2f} |")
w("")

w("### 方法 x 偏移 汇总（3 seed，NTGE 未达成按 10000 截尾）")
w("")
w("| 偏移 | 方法 | NTGE(first) | NTGE(stable) | 达成 first | 达成 stable | transient | GE@10k | SR@10k | acc | conf | H | D balanced (before→after) |")
w("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for method in METHODS:
    for shift in SHIFTS:
        m = SUMMARY["metrics"].get(method, {}).get(shift)
        if not m:
            continue
        db = m.get("domain_balanced_before_mean")
        da = m.get("domain_balanced_after_mean")
        diag = "—" if db is None else f"{db:.3f} → {da:.3f}"
        w(f"| {shift} | {LABEL[method]} | {mean_std(m, 'ntge_first')} | {mean_std(m, 'ntge_stable')} | "
          f"{m['ntge_first_reached']}/{m['n_runs']} | {m['ntge_stable_reached']}/{m['n_runs']} | "
          f"{m['transient_runs']} | {mean_std(m, 'ge_at_10000')} | {m['sr_at_10000_mean']:.2f} | "
          f"{m['attack_accuracy_mean']*100:.2f}% | {m['mean_max_softmax_mean']:.3f} | "
          f"{m['mean_entropy_mean']:.2f} | {diag} |")
w("")

w("### 同 seed 配对差（相对 Source Only；负值表示 GE/NTGE 更低）")
w("")
w("| 方法 | 偏移 | seeds | ΔGE@10k 逐 seed | 均值 | ΔNTGE(first) 逐 seed | 均值 | GE 改善/恶化 |")
w("|---|---|---|---|---|---|---|---|")
for method in ["dann", "cdan"]:
    for shift in SHIFTS:
        p = SUMMARY.get("paired_delta_vs_source_only", {}).get(method, {}).get(shift)
        if not p:
            continue
        dg = ", ".join(f"{d:+.0f}" for d in p["delta_ge_at_10000"])
        dn = ", ".join(f"{d:+.0f}" for d in p["delta_ntge_first"])
        w(f"| {LABEL[method]} | {shift} | {p['seeds']} | {dg} | {p['delta_ge_at_10000_mean']:+.1f} | "
          f"{dn} | {p['delta_ntge_first_mean']:+.1f} | {p['improved_ge']}/{p['worsened_ge']} |")
w("")

w("### 域判别诊断（balanced accuracy，source_val vs target_domain_val）")
w("")
w("| 偏移 | 方法 | seed | before | after | Δ |")
w("|---|---|---|---|---|---|")
for shift in SHIFTS:
    for method in ["dann", "cdan"]:
        for r in sorted([r for r in EVAL if r["method"] == method and r["shift"] == shift],
                        key=lambda r: r["seed"]):
            b, a = r.get("domain_balanced_before"), r.get("domain_balanced_after")
            w(f"| {shift} | {LABEL[method]} | {r['seed']} | {fmt(b, 3)} | {fmt(a, 3)} | "
              f"{'—' if b is None or a is None else f'{a-b:+.3f}'} |")
w("")

text = "\n".join(lines)
(HERE / "reports").mkdir(exist_ok=True)
(HERE / "reports/tables.md").write_text(text, encoding="utf-8")
print(text)
