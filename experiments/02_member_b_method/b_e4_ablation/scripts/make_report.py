#!/usr/bin/env python3
"""由 README.md 生成 reports/b_e4_report.md。"""
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
readme = (HERE / "README.md").read_text(encoding="utf-8").splitlines()
out = []
for i, line in enumerate(readme):
    if i == 0 and line.startswith("# "):
        continue
    if line.startswith("## "):
        out.append("#### " + line[3:])
    elif line.startswith("### "):
        out.append("##### " + line[4:])
    else:
        out.append(line.replace("](figures/", "](../figures/"))
header = """# B-E4 报告：三向消融（CDAN vs 消融 1 marginal+w vs CDAN-SCA full）

- 实验编号：B-E4（数据来自 B-E3 的 12 个 run；消融 2 = A-E2 CDAN，未重跑）
- 归档目录：experiments/02_member_b_method/b_e4_ablation/
- 本报告由 README.md 自动生成。

"""
text = header + "\n".join(out).strip() + "\n"
(HERE / "reports").mkdir(exist_ok=True)
(HERE / "reports/b_e4_report.md").write_text(text, encoding="utf-8", newline="\n")
print("wrote reports/b_e4_report.md (%d chars)" % len(text))
