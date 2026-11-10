#!/usr/bin/env python3
"""由 README.md 的 8.1.x 小节生成 reports/a_e2_report.md（与 E0/B-E2 报告格式一致）。

规则：
  - 一级标题 "# ..." 保留为文档标题；
  - "## 8.1.x" -> "#### 8.1.x"，"### ..." -> "##### ..."；
  - 图片路径 figures/... -> ../figures/...（报告位于 reports/ 下）。
"""
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

header = """# A-E2 报告：DANN / CDAN 统一 UDA 基线（ASCAD fixed-key → desync50 / desync100）

- 实验编号：A-E2
- 实验级别：formal（主结果 12 次训练 + 2 次 λ=0 归因 pilot + Source Only 复用 6 个单元）
- 归档目录：experiments/01_member_a_baselines/a_e2_uda_baselines/
- 主干：Official CNN（B-E2 冻结结构；B-E1 同 seed 权重初始化）
- 本报告由 README.md 的 8.1.1–8.1.13 自动生成，内容与 README 一致。

"""

text = header + "\n".join(out).strip() + "\n"
(HERE / "reports").mkdir(exist_ok=True)
(HERE / "reports/a_e2_report.md").write_text(text, encoding="utf-8", newline="\n")
print(f"wrote reports/a_e2_report.md ({len(text)} chars, {len(out)} lines)")
