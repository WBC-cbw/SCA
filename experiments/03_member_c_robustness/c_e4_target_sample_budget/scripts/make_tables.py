#!/usr/bin/env python3
"""C-E4：把 aggregate 产出的 CSV 机械转成 markdown 表（避免手抄数字）。"""
from __future__ import annotations

import csv
import os
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
RES = Path(os.environ.get("C_E4_OUT", str(HERE / "results")))

LABEL = {"source_only": "Source Only", "dann": "DANN", "cdan_sca": "CDAN-SCA full",
         "cdpa_mmd": "CDPA-MMD"}
ORDER = ["source_only", "dann", "cdan_sca", "cdpa_mmd"]
BUDGETS = [0, 50, 500, 5000, 45000]


def rows(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fnum(x, nd=1):
    if x in ("", "None", None):
        return "—"
    try:
        v = float(x)
    except ValueError:
        return str(x)
    if v == int(v):
        return f"{int(v)}"
    return f"{v:.{nd}f}"


def main():
    nt = rows(RES / "table_ntge.csv")
    cost = rows(RES / "table_cost.csv")
    diag = rows(RES / "diagnostics.csv")
    met = rows(RES / "metrics.csv")
    out = ["# C-E4 汇总表（由 scripts/make_tables.py 从 results/*.csv 生成）", ""]
    shifts = sorted({r["shift"] for r in nt}) or ["desync50"]
    for shift in shifts:
        out += [f"## {shift}", ""]
        out += ["### 表 A：NTGE 与稳定率", "",
                "| 方法 | budget | NTGE(first) 均值±sd | 触发率 | NTGE(stable) 均值±sd | 稳定率 | GE@10k | SR@10k |",
                "|---|---|---|---|---|---|---|---|"]
        for m in ORDER:
            for b in BUDGETS:
                c = [r for r in nt if r["shift"] == shift and r["method"] == m
                     and int(r["budget"]) == b]
                if not c:
                    continue
                r = c[0]
                first = fnum(r["ntge_first_mean"]) + (" ± " + fnum(r["ntge_first_sd"])
                                                      if r["ntge_first_mean"] != "None" else "")
                st = (fnum(r["ntge_stable_mean"]) + " ± " + fnum(r["ntge_stable_sd"])
                      if r["ntge_stable_mean"] != "None" else "—")
                cens = "†" if r["first_reached"] != r["n_seeds"] else ""
                out.append(f"| {LABEL[m]} | {b if b else '—'} | {first}{cens} | "
                           f"{r['first_reached']}/{r['n_seeds']} | {st} | "
                           f"{r['stable_reached']}/{r['n_seeds']} | {fnum(r['ge_at_10000'], 2)} | "
                           f"{fnum(r['sr_at_10000'], 2)} |")
        out += ["", "† 该 cell 至少一个 seed 未在 10k 内触发 GE=0（NTGE 按 10000 截尾）。", ""]

        out += ["### 表 B：总成本（双口径）", "",
                "| 方法 | budget | N_attack@GE=1 | N_total@GE=1 均值±sd | N_total(stable) 均值±sd |",
                "|---|---|---|---|---|"]
        for m in ORDER:
            for b in BUDGETS:
                c = [r for r in cost if r["shift"] == shift and r["method"] == m
                     and int(r["budget"]) == b]
                if not c:
                    continue
                r = c[0]
                ntot = fnum(r["n_total_at_ge1_mean"]) + " ± " + fnum(r["n_total_at_ge1_sd"])
                nst = (fnum(r["n_total_stable_mean"]) + " ± " + fnum(r["n_total_stable_sd"])
                       if r["n_total_stable_mean"] != "None" else "—")
                cens = "\u2020" if float(r.get("stable_rate") or 0) == 0.0 else ""
                out.append(f"| {LABEL[m]} | {b if b else '—'} | {fnum(r['n_total_at_ge1_mean'])} | "
                           f"{ntot}{cens} | {nst} |")
        out += ["", "N_total = budget + N_attack@GE=1；† 表示稳定率 0（成本为截尾下界）。", ""]

        if diag:
            out += ["### 表 C：目标域诊断（target_adapt）", "",
                    "| 方法 | budget | max top-1 | 坍塌 epoch 数 | adapt ECE | conf-wrong≥0.9 | w 均值 |",
                    "|---|---|---|---|---|---|---|"]
            for m in ORDER:
                for b in BUDGETS:
                    c = [r for r in diag if r["shift"] == shift and r["method"] == m
                         and int(r["budget"]) == b]
                    if not c:
                        continue
                    r = c[0]
                    out.append(f"| {LABEL[m]} | {b if b else '—'} | {fnum(r.get('max_top1'), 3)} | "
                               f"{fnum(r.get('collapse_epochs'), 2)} | {fnum(r.get('adapt_ece'), 3)} | "
                               f"{fnum(r.get('adapt_confwrong09'), 3)} | {fnum(r.get('adapt_w_mean'), 3)} |")
            out += ["", "w = 1 - H/log256，仅 CDAN-SCA 用其加权；其他方法该列为诊断口径（同式换算）。", ""]

        out += ["### 表 D：逐 seed", "",
                "| 方法 | budget | seed | NTGE(first) | NTGE(stable) | GE@10k | SR@10k | train s |",
                "|---|---|---|---|---|---|---|---|"]
        for m in ORDER:
            for b in BUDGETS:
                for r in [x for x in met if x["shift"] == shift and x["method"] == m
                          and int(x["budget"]) == b]:
                    out.append(f"| {LABEL[m]} | {b if b else '—'} | {r['seed']} | "
                               f"{fnum(r.get('ntge_first'))} | {fnum(r.get('ntge_stable'))} | "
                               f"{fnum(r.get('ge_at_10000'), 2)} | {fnum(r.get('sr_at_10000'), 2)} | "
                               f"{fnum(r.get('train_seconds'), 0)} |")
        out.append("")
    (RES / "tables.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("wrote", RES / "tables.md", len(out), "lines")


if __name__ == "__main__":
    main()
