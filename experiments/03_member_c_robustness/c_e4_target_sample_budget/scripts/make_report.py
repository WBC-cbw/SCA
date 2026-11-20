#!/usr/bin/env python3
"""由 README.md 的 8.1.x 小节生成 reports/c_e4_report.md。"""
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

header = """# C-E4 报告：目标域适应样本量敏感性（target_adapt = 50 / 500 / 5000 / 45000）

- 实验编号：C-E4
- 实验级别：formal（27 次新训 × desync50；desync100 同构；全量档复用 A-E2/A-E3/B-E3）
- 归档目录：experiments/03_member_c_robustness/c_e4_target_sample_budget/
- 主干：Official CNN（B-E1 同 seed 权重初始化）
- 方法：CDPA-MMD / DANN / CDAN-SCA full；参照 Source Only
- 本报告由 README.md 的 8.1.1-8.1.13 自动生成，内容与 README 一致。

"""

text = header + "\n".join(out).strip() + "\n"
(HERE / "reports").mkdir(exist_ok=True)
(HERE / "reports/c_e4_report.md").write_text(text, encoding="utf-8", newline="\n")
print(f"wrote reports/c_e4_report.md ({len(text)} chars, {len(out)} lines)")
