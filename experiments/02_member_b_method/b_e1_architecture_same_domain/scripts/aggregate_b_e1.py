#!/usr/bin/env python3
"""B-E1 汇总：结构 x seed 指标表 + 均值/标准差。

输入: results/<run>/run_meta.json（训练侧）+ results/eval/<run>.json（评估侧）
输出: results/metrics.csv, results/summary.json
"""
import csv
import json
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
RES = HERE / "results"


def main():
    metas = []
    for d in sorted(p for p in RES.iterdir() if p.is_dir() and p.name != "eval"):
        m = d / "run_meta.json"
        e = RES / "eval" / f"{d.name}.json"
        if not (m.exists() and e.exists()):
            continue
        meta = json.loads(m.read_text(encoding="utf-8"))
        ev = json.loads(e.read_text(encoding="utf-8"))
        meta.update({k: ev[k] for k in ("ntge", "ge_at_1000", "ge_at_5000",
                                        "ge_at_10000", "sr_at_1000", "sr_at_10000")})
        meta["n_total_at_ge1"] = ev["ntge"]
        metas.append(meta)

    metas.sort(key=lambda r: (r["arch"], r["seed"]))
    cols = ["arch", "seed", "n_params", "epochs_done", "train_seconds",
            "sec_per_epoch_mean", "final_train_acc", "final_val_acc",
            "ntge", "n_total_at_ge1", "ge_at_1000", "ge_at_5000", "ge_at_10000",
            "sr_at_1000", "sr_at_10000"]
    with (RES / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in metas:
            w.writerow({k: r.get(k) for k in cols})

    summary = {"experiment": "B-E1 architecture screening (same-domain ASCAD fixed-key)",
               "n_runs": len(metas), "per_arch": {}, "runs": []}
    for r in metas:
        summary["runs"].append({k: r.get(k) for k in cols})

    for arch in sorted({r["arch"] for r in metas}):
        rs = [r for r in metas if r["arch"] == arch]
        def ms(key):
            vals = [r[key] for r in rs if r.get(key) is not None]
            if not vals:
                return None
            m = sum(vals) / len(vals)
            sd = st.stdev(vals) if len(vals) > 1 else 0.0
            return {"mean": round(m, 3), "std": round(sd, 3), "values": vals}
        summary["per_arch"][arch] = {
            "n_params": rs[0]["n_params"],
            "epochs_done": ms("epochs_done"),
            "train_seconds": ms("train_seconds"),
            "sec_per_epoch": ms("sec_per_epoch_mean"),
            "final_val_acc": ms("final_val_acc"),
            "final_train_acc": ms("final_train_acc"),
            "ntge": ms("ntge"),
            "ge_at_1000": ms("ge_at_1000"),
            "ge_at_10000": ms("ge_at_10000"),
            "sr_at_1000": ms("sr_at_1000"),
            "sr_at_10000": ms("sr_at_10000"),
        }
    (RES / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    hdr = f"{'arch':10s}{'seed':>5s}{'params':>12s}{'sec/ep':>8s}{'train_s':>9s}" \
          f"{'val_acc':>9s}{'NTGE':>7s}{'GE1k':>7s}{'GE10k':>7s}{'SR1k':>6s}{'SR10k':>6s}"
    print(hdr)
    for r in metas:
        print(f"{r['arch']:10s}{r['seed']:>5d}{r['n_params']:>12,d}"
              f"{r['sec_per_epoch_mean']:>8.1f}{r['train_seconds']:>9.0f}"
              f"{r['final_val_acc']:>9.4f}{str(r['ntge']):>7s}{str(r['ge_at_1000']):>7s}"
              f"{str(r['ge_at_10000']):>7s}{r['sr_at_1000']:>6d}{r['sr_at_10000']:>6d}")
    print()
    def ms_str(d, nd=0):
        if d is None:
            return "n/a"
        return f"{d['mean']:.{nd}f}±{d['std']:.{nd}f}"

    for arch, s in summary["per_arch"].items():
        print(f"{arch:10s} NTGE={ms_str(s['ntge'])} "
              f"GE10k={ms_str(s['ge_at_10000'], 1)} "
              f"SR1k={ms_str(s['sr_at_1000'], 2)} "
              f"val_acc={ms_str(s['final_val_acc'], 4)} "
              f"t={ms_str(s['train_seconds'])}s")


if __name__ == "__main__":
    main()
