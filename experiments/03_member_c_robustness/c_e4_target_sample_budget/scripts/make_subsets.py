#!/usr/bin/env python3
"""C-E4：生成 target_adapt 嵌套子集索引（与主方法/基线共用同一子集）。

设计
----
- 目标域适应池 = 每个 shift 的 Profiling_traces[0:45000]（target_domain_val 用
  [45000:50000]，与训练子集天然不重叠；target_attack 用 Attack_traces 全 10000 条）。
- 每个 (shift, seed) 用一个置换 perm（default_rng(20261116 + seed*100 + shift_code)），
  预算 50/500/5000 分别取 perm[:50]/perm[:500]/perm[:5000]，即嵌套子集：
  b50 ⊂ b500 ⊂ b5000。同一 seed 下三种方法与全部预算共享同一批索引，
  因此方法间差异只来自适应算法，不来自样本选择。
- 存排序后的索引（训练时每 epoch 重新置换，存储顺序不影响训练）。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
CFG = HERE / "configs"
CFG.mkdir(exist_ok=True)

POOL = 45000
BUDGETS = [50, 500, 5000]
SEEDS = [42, 43, 44]
SHIFT_CODE = {"desync50": 1, "desync100": 2}
BASE_SEED = 20261116


def main():
    out = {
        "meta": {
            "experiment": "C-E4 target_adapt sample budget",
            "target_pool": "Profiling_traces[0:45000] of each shift HDF5",
            "target_domain_val": "Profiling_traces[45000:50000] (diagnostic only)",
            "target_attack": "Attack_traces[0:10000]",
            "budgets": BUDGETS,
            "seeds": SEEDS,
            "sampling": "nested: perm = default_rng(BASE_SEED + seed*100 + shift_code).permutation(45000); "
                        "b50=perm[:50], b500=perm[:500], b5000=perm[:5000]; indices sorted before storage",
            "base_seed": BASE_SEED,
            "shared_across_methods": True,
        },
        "subsets": {},
    }
    for shift, code in SHIFT_CODE.items():
        out["subsets"][shift] = {}
        for seed in SEEDS:
            rng = np.random.default_rng(BASE_SEED + seed * 100 + code)
            perm = rng.permutation(POOL)
            entry = {}
            for b in BUDGETS:
                idx = np.sort(perm[:b]).astype(np.int64)
                assert len(idx) == b and len(set(idx.tolist())) == b
                assert int(idx.min()) >= 0 and int(idx.max()) < POOL
                entry[str(b)] = idx.tolist()
            # 嵌套性校验
            for small, big in [(50, 500), (500, 5000), (50, 5000)]:
                assert set(entry[str(small)]).issubset(set(entry[str(big)])), (shift, seed, small, big)
            out["subsets"][shift][f"s{seed}"] = entry
            print(f"{shift} s{seed}: 50/500/5000 nested ok")

    path = CFG / "target_subsets.json"
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    (CFG / "target_subsets.sha256").write_text(sha + "  target_subsets.json\n", encoding="utf-8")
    print(f"wrote {path} ({path.stat().st_size} bytes) sha256={sha}")


if __name__ == "__main__":
    main()
