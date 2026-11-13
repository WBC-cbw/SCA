#!/usr/bin/env python3
"""由 README.md 的 8.1.x 小节生成 reports/b_e5_report.md（格式与 E0/A-E2/A-E3/B-E3 一致）。"""
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

header = """# B-E5 报告：校准感知 CDAN-SCA（温度校准权重 vs 裸熵权重）

- 实验编号：B-E5
- 实验级别：formal（6 次 calib 训练；对照 CDPA-MMD / full / 消融 1 / CDAN 全部复用）
- 归档目录：experiments/02_member_b_method/b_e5_calibration_aware/
- 主干：Official CNN（B-E1 同 seed 权重初始化）
- 本报告由 README.md 的 8.1.1-8.1.13 自动生成，内容与 README 一致。

"""

text = header + "\n".join(out).strip() + "\n"
(HERE / "reports").mkdir(exist_ok=True)
(HERE / "reports/b_e5_report.md").write_text(text, encoding="utf-8", newline="\n")
print(f"wrote reports/b_e5_report.md ({len(text)} chars, {len(out)} lines)")
