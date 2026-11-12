#!/usr/bin/env python3
"""由 README.md 的 8.1.x 小节生成 reports/b_e3_report.md（与 E0/A-E2/A-E3 报告格式一致）。"""
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

header = """# B-E3 报告：CDAN-SCA full 与熵权重诊断（ASCAD fixed-key -> desync50/desync100）

- 实验编号：B-E3（主方法）+ B-E4（三向消融）
- 实验级别：formal（6 次 full + 6 次消融 1；消融 2 = A-E2 的 CDAN 6 次复用）
- 归档目录：experiments/02_member_b_method/b_e3_cdan_sca/（消融汇总在 b_e4_ablation/）
- 主干：Official CNN（B-E1 同 seed 权重初始化）
- 本报告由 README.md 的 8.1.1-8.1.13 自动生成，内容与 README 一致。

"""

text = header + "\n".join(out).strip() + "\n"
(HERE / "reports").mkdir(exist_ok=True)
(HERE / "reports/b_e3_report.md").write_text(text, encoding="utf-8", newline="\n")
print(f"wrote reports/b_e3_report.md ({len(text)} chars, {len(out)} lines)")
