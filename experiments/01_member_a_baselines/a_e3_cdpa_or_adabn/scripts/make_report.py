#!/usr/bin/env python3
"""由 README.md 的 8.1.x 小节生成 reports/a_e3_report.md（与 E0/A-E2 报告格式一致）。

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

header = """# A-E3 报告：CDPA-MMD 领域强基线（ASCAD fixed-key → desync50 / desync100）

- 实验编号：A-E3
- 实验级别：formal（主结果 6 次微调训练；Source Only / DANN / CDAN 复用前序预测）
- 归档目录：experiments/01_member_a_baselines/a_e3_cdpa_or_adabn/
- 主干：Official CNN（B-E1 同 seed 权重做预训练初始化）
- 本报告由 README.md 的 8.1.1–8.1.13 自动生成，内容与 README 一致。

"""

text = header + "\n".join(out).strip() + "\n"
(HERE / "reports").mkdir(exist_ok=True)
(HERE / "reports/a_e3_report.md").write_text(text, encoding="utf-8", newline="\n")
print(f"wrote reports/a_e3_report.md ({len(text)} chars, {len(out)} lines)")
