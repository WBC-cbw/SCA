#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成实验论文初稿（Markdown + DOCX），数据全部来自本仓库各实验的 results/figures。

用法: python docs/research/make_paper_draft.py
"""
from __future__ import annotations
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT_MD = HERE / "cdan_sca_paper_draft.md"
OUT_DOCX = HERE / "cdan_sca_paper_draft.docx"

E = "experiments"
F = {
 "b1_ge": E + "/02_member_b_method/b_e1_architecture_same_domain/figures/b_e1_ge_curves.png",
 "a3_ge50": E + "/01_member_a_baselines/a_e3_cdpa_or_adabn/figures/a_e3_ge_desync50.png",
 "a3_ge100": E + "/01_member_a_baselines/a_e3_cdpa_or_adabn/figures/a_e3_ge_desync100.png",
 "b3_ge50": E + "/02_member_b_method/b_e3_cdan_sca/figures/b_e3_ge_desync50.png",
 "b3_ge100": E + "/02_member_b_method/b_e3_cdan_sca/figures/b_e3_ge_desync100.png",
 "b3_mon": E + "/02_member_b_method/b_e3_cdan_sca/figures/b_e3_monitor_desync100.png",
 "b3_adv2_ge100": E + "/02_member_b_method/b_e3_cdan_sca/figures/b_e3_adv2_ge_desync100.png",
 "b3_adv2_mon": E + "/02_member_b_method/b_e3_cdan_sca/figures/b_e3_adv2_monitor_desync100.png",
 "b5_ge100": E + "/02_member_b_method/b_e5_calibration_aware/figures/b_e5_ge_desync100_all.png",
 "b5_w": E + "/02_member_b_method/b_e5_calibration_aware/figures/b_e5_weights_attack.png",
 "ce4_cost50": E + "/03_member_c_robustness/c_e4_target_sample_budget/figures/c_e4_budget_cost_desync50.png",
 "ce4_cost100": E + "/03_member_c_robustness/c_e4_target_sample_budget/figures/c_e4_budget_cost_desync100.png",
 "ce4_stab100": E + "/03_member_c_robustness/c_e4_target_sample_budget/figures/c_e4_stability_desync100.png",
}

DOC = []
def h1(t): DOC.append(("h1", t))
def h2(t): DOC.append(("h2", t))
def h3(t): DOC.append(("h3", t))
def p(t): DOC.append(("p", t))
def bullets(xs): DOC.append(("bullets", xs))
def table(cap, head, rows, note=None): DOC.append(("table", cap, head, rows, note))
def img(key, cap): DOC.append(("img", key, cap))
def note(t): DOC.append(("note", t))

# ------------------------------------------------------------------ 标题
h1("跨分布深度学习侧信道分析的无监督域适应：实验研究报告（初稿）")
p("说明：本稿汇总仓库内 E0、B-E1、A-E2、A-E3、B-E3/B-E4、B-E5、C-E4 七个实验的最终数值，"
  "所有数字均来自各实验目录下的 results/ 与 figures/，由 docs/research/make_paper_draft.py 自动生成；"
  "文中图号与表号为初稿占位，正式投稿时按期刊模板重排。")

# ------------------------------------------------------------------ 摘要
h2("摘要")
p("本文报告一套围绕跨分布侧信道分析（SCA）的无监督域适应（UDA）展开的可复现实验。在 ASCAD v1 fixed-key 上，"
  "我们首先建立同域五层 CNN 基线（E0），随后在 fixed→desync50/desync100 两个采集偏移上依次比较主干结构（B-E1）、"
  "通用对抗式 UDA（DANN、CDAN，A-E2）、领域强基线 CDPA-MMD（A-E3）、本文主方法 CDAN-SCA 及其核心消融（B-E3/B-E4）、"
  "校准感知变体（B-E5），并进一步测量目标域适应样本量在 50/500/5000/45000 四档下的敏感性（C-E4，54 次训练）。"
  "全部比较使用同一划分、同一评估脚本、同一训练配方与 3 个随机种子；NTGE 同时报告首次与稳定两个口径，"
  "并对出现 GE=0 的运行做 10 次轨迹顺序置换。主要结果：（1）同域 CNN 在 800–1433 条攻击轨迹内稳定恢复密钥，"
  "跨分布下所有方法显著退化；（2）CDPA-MMD 是唯一在两个偏移上均出现 3/3 首次触底的方法（desync50：500 条适应样本、"
  "NTGE 4733±694、3/3 稳定），通用对抗方法与主方法均未超过它；（3）CDAN-SCA 的目标域熵加权放大“自信但错误”样本，"
  "desync100 上 6/6 运行全部坍塌（目标域 top-1 单类占比 0.79，坍塌 35–40 个 epoch）；（4）校准感知变体的 desync100 "
  "GE@10k 降至 0.33，但温度在近乎随机的 source_val 上拟合（T≈12.3），校准后权重均值仅 0.014–0.017，等于关闭对抗项，"
  "不能据此宣称“校准有效”；（5）desync50 的适应样本拐点在 500 条（N_total=5233，较 45000 档省 89.7%），desync100 "
  "不存在稳健拐点，低预算档的超低 NTGE 在轨迹重排后不复现。据此认为：当前协议下对抗式条件域适应并非 SCA 跨分布问题的"
  "稳健收益来源，稳定零点、置换均值与 N_total 双口径比单次 NTGE 点估计更能反映真实能力。")
p("关键词：侧信道分析；无监督域适应；条件对抗网络；猜测熵；置换检验；可复现性")

# ------------------------------------------------------------------ 1 引言
h2("1　引言")
p("基于深度学习的侧信道分析在同分布测试集上已能稳定恢复密钥，但模型对采集条件（对齐偏移、器件、噪声）高度敏感，"
  "在目标域上往往退化为随机猜测。无监督域适应（UDA）因此成为 SCA 领域的重要方向：只使用源域标签与目标域无标签轨迹，"
  "让特征表示在域间对齐。DANN 通过域判别器与梯度反转层对齐边缘分布，CDAN 进一步引入分类器输出的条件信息，"
  "CDPA-MMD 则在多层特征上做最大均值差异（MMD）匹配并被视为该领域的强基线。")
p("本文的主线方法 CDAN-SCA 在 CDAN 基础上加入目标域熵加权：对预测熵低（更“自信”）的目标样本赋予更大对抗权重，"
  "期望把对齐集中在决策边界附近的高置信样本上。围绕“主方法是否在同等条件下优于基线”这一核心问题，"
  "我们完成了从同域基线、主干筛选、通用与领域基线、主方法与消融、校准变体到样本量敏感性的完整实验链，"
  "并系统记录了失败与不一致之处。本文不追求理论最优，而是给出一份可复核的证据清单，说明该方法在什么条件下失效、"
  "失效的机制是什么、现有评价口径有哪些陷阱。")
p("主要贡献有三点：（1）在同一协议下比较 7 种方法/变体在两个采集偏移上的表现，并给出 N_attack 与 N_total 双口径成本；"
  "（2）用逐 epoch 监控、ECE/confident-wrong 诊断与 10 次轨迹置换，把“预测坍塌”“校准失真”“顺序幸运”三类现象量化；"
  "（3）给出目标域适应样本量的敏感性曲线，指出 desync50 的 500 条拐点与 desync100 的无解区间。")

# ------------------------------------------------------------------ 2 设置
h2("2　实验设置")
h3("2.1　数据与四类划分")
p("数据集为 ASCAD v1 fixed-key（Profiling 50000 条、Attack 10000 条、700 采样点、int8 原始幅度），"
  "标签定义为 Sbox[plaintext[2] ⊕ key[2]]（第 3 字节，256 类）。所有实验严格区分四类划分，"
  "target_adapt 与 target_attack 不重叠；C-E4 通过固定随机种子生成嵌套子集，保证同一 seed 下三种方法使用完全相同的轨迹索引。")
table("表 1　数据划分与用途",
      ["角色", "来源", "条数", "用途"],
      [["source_train", "fixed Profiling[0:45000]", "45000", "有标签监督训练"],
       ["source_val", "fixed Profiling[45000:50000]", "5000", "早停/诊断，不参与训练"],
       ["target_adapt", "desync Profiling[0:N]", "N ∈ {50, 500, 5000, 45000}", "无标签适应（C-E4 扫描 N）"],
       ["target_attack", "desync Attack[0:10000]", "10000", "仅评估，绝不参与训练或选模"]],
      "desync50 与 desync100 各自独立重复同一划分。")

h3("2.2　模型与训练配方")
p("主干为仓库官方五层 CNN（Conv1D 64/128/256/512/512，k=11，每层 AvgPool×2；Flatten 10752→Dense 4096→4096→256，"
  "共 66,652,544 参数），与 E0 完全一致。所有适应实验从同 seed 的 B-E1 同域 checkpoint 初始化，保证起点一致。"
  "统一配方：RMSprop，lr=1e-5，batch=200，75 epochs，原始 int8 幅度输入不做归一化。DANN 的域判别器输入为 "
  "normalize(f)·√4096；CDAN 使用 softmax(logits) 与 1024 维随机多线性映射；CDPA-MMD 在 conv/fc1/fc2 三层做 5 核 RBF-MMD，"
  "λ=0.1；CDAN-SCA 的对抗项权重为 w=1−H(p)/log 256。")
p("为控制变量，每轮实验只改变一个主要因素：B-E1 改主干、A-E2/A-E3 改方法、B-E3 改条件输入与权重、B-E5 改权重来源、"
  "C-E4 只改 target_adapt 条数（目标域 batch 循环采样以保持源域 45000 条的训练量不变）。")

h3("2.3　评价指标与统计口径")
p("主指标为猜测熵 GE（key rank）及由其导出的 NTGE（首次达到 GE=0 的攻击轨迹数）、稳定 NTGE（末尾连续 GE=0 段的起点）、"
  "SR@1k/10k 与总轨迹成本。总成本按 N_total@GE=1 = target_adapt 条数 + N_attack@GE=1 计算，两种口径都报告；"
  "10k 内未触发者按 10000 截尾，其 N_total 视为下界（表中以 † 标注）。正式实验使用 seed ∈ {42,43,44}，"
  "报告均值±标准差；对每个出现 GE=0 的运行再做 10 次攻击轨迹顺序置换，以区分“真实恢复”与“顺序幸运”。")
table("表 2　实验矩阵与计算成本（RTX 5060 Laptop，单卡）",
      ["实验", "内容", "训练次数", "GPU·h"],
      [["E0", "同域 fixed-key 基线（seed 42）", "1", "0.33"],
       ["B-E1", "MLP / 官方 CNN / ResNet1D × 3 seeds", "9", "2.68"],
       ["A-E2", "DANN、CDAN × 2 偏移 × 3 seeds + λ=0 对照", "12 + 2", "约 8.4"],
       ["A-E3", "CDPA-MMD × 2 偏移 × 3 seeds", "6", "3.12"],
       ["B-E3/B-E4", "CDAN-SCA full、Abl.1 × 2 偏移 × 3 seeds", "12", "6.29"],
       ["B-E5", "校准感知变体 × 2 偏移 × 3 seeds", "6", "3.15"],
       ["C-E4", "3 方法 × 4 预算档 × 2 偏移 × 3 seeds（45k 档复用）", "54", "27.17"]],
      "合计约 51 GPU·小时；A-E2 按日志均值估算，其余为记录值。")

# ------------------------------------------------------------------ 3 主干
h2("3　实验一：同域基线与主干结构筛选")
p("E0 在 fixed-key 同域上验证了训练配方与评估链路：75 epochs 后 NTGE=800 条，GE@10k=0，SR@1000=100%，"
  "说明五层 CNN 在该任务上具备稳定的同域建模能力，可作为后续跨分布实验的参照上限。")
p("B-E1 在同一划分、同一配方下比较三种主干：MLP（Flatten→512→512→256，0.75M 参数）、官方 CNN（66.65M）与"
  "ResNet1D（3 个残差块、每块 2×Conv1D k=11，通道 64/128/256，stride 1/2/2，GlobalAvgPool→256，1.51M）。"
  "结果差异极大：只有官方 CNN 在 3 个 seed 上全部于 10k 内触底（NTGE 1000/700/2600，GE@10k 全为 0）；"
  "MLP 仅 1 个 seed 偶发触底（700），GE@10k 仍为 87.7；ResNet1D 三个 seed 全部未触底，GE@10k 171.3。")
table("表 3　主干结构同域对比（3 seeds，均值或逐 seed 值）",
      ["主干", "参数量", "NTGE（条）", "GE@10k", "SR@10k", "s/epoch"],
      [["官方 CNN", "66.65 M", "1433±1021（1000/700/2600）", "0", "1.0", "15.8"],
       ["MLP", "0.75 M", "仅 1/3 触底（700）", "87.7（153/51/59）", "0", "0.33"],
       ["ResNet1D", "1.51 M", "0/3 触底", "171.3（163/205/146）", "0", "26.8"]],
      "训练曲线见 figures/b_e1_ge_curves.png。")
img("b1_ge", "图 1　三种主干的同域 GE 曲线（实线为 3 seed 均值，阴影为 min–max；数据：B-E1）")
p("该结果否定了“容量不足”这一最直观解释：参数量最大的 CNN 唯一成功，而更深的 ResNet1D 与更宽的 MLP 均失败。"
  "后续 pilot 显示，ResNet1D 把 lr 提到 1e-3 后 GE@10k 从 146–205 降到 54（仍无 GE=0），MLP 提 lr 反而恶化到 242，"
  "MLP 缩小训练集到 2000 条时最高 train acc 仅约 1.3%、GE@10k=17（更差）。因此失败更可能来自输入尺度与无 BN 的残差堆叠、"
  "以及全连接结构难以对 700 点原始波形做局部条件化，而非模型容量。后续所有跨分布实验统一使用官方 CNN。")
p("需要说明的评价口径问题：表 3 中 MLP 的“1/3 触底”是单次运行的瞬时零点，未做稳定口径判定；"
  "若按稳定口径，MLP 与 ResNet1D 均为 0/3。这也预告了本文后续反复出现的方法学问题——单次 NTGE 点估计不足以下结论。")

# ------------------------------------------------------------------ 4 UDA 基线
h2("4　实验二：跨分布 UDA 基线")
p("A-E2 在 fixed→desync50/desync100 上实现 DANN 与 CDAN，与 Source Only 对照；A-E3 补上领域强基线 CDPA-MMD。"
  "三者的主干、初始化、数据划分、优化器与 epoch 数完全一致，唯一差别是对齐项。")
table("表 4　跨分布基线（3 seeds；NTGE 为首次触底均值，括号为触底 seed 数；稳定率为末尾连续 GE=0 的比例）",
      ["方法", "desync50 NTGE", "desync50 稳定率", "desync50 GE@10k", "desync100 NTGE", "desync100 稳定率", "desync100 GE@10k"],
      [["Source Only", "10000（0/3）", "0/3", "22.7±11.4", "3700（2/3）", "0/3", "23.0±10.6"],
       ["DANN", "9167±1443（1/3）", "1/3", "4.0±5.3", "8300±2944（1/3）", "0/3", "11.0±6.1"],
       ["CDAN", "10000（0/3）", "0/3", "9.3±6.4", "6567±5105（2/3）", "1/3", "13.7±13.1"],
       ["CDPA-MMD", "5700±624（3/3）", "3/3", "0", "3700±5462（2/3）", "1/3", "4.0±4.6"]],
      "45k 全量适应；稳定 NTGE 与 N_total 见正文。")
img("a3_ge50", "图 2　desync50 上四种方法的 GE 曲线（3 seed 均值与 min–max 带；数据：A-E3）")
img("a3_ge100", "图 3　desync100 上四种方法的 GE 曲线（数据：A-E3）")
p("两个偏移呈现不同图景。desync50 上 CDPA-MMD 明显占优：3/3 首次触底（5700±624），3/3 稳定，GE@10k=0、SR@10k=1.0；"
  "DANN 只有 1/3 触底且 GE@10k 仍为 4.0；CDAN 完全未触底。desync100 上所有方法都退化，CDPA-MMD 也只有 2/3 首次触底、"
  "1/3 稳定，GE@10k 4.0，远未达到同域水平（800 条）。也就是说，即使是最强的领域基线，在更强偏移下也没有解决“稳定恢复密钥”。")
p("A-E2 还包含一组重要的归因对照：固定 seed 42、把训练延长到 150 epochs、令对抗权重 λ=0（即只做 150 epoch 的源域训练），"
  "与 DANN（150 epochs，含对抗）比较。desync100 上 λ=0 的 GE@10k=2，反而优于 DANN 的 7，也优于 Source Only（75 epoch）的 15，"
  "说明该设置下“多训 75 epoch”已能解释 DANN 的大部分收益；desync50 上则是 DANN 0 vs λ=0 34，单 seed 下对抗项似乎有益。"
  "这组对照只有 1 个 seed，只能作为方向性证据，但它提示：任何“UDA 有效”的结论都必须排除训练时长与初始化带来的混淆。")
table("表 5　λ=0 归因对照（seed 42，150 epochs；GE@10k）",
      ["偏移", "Source Only（75 ep）", "DANN（150 ep）", "DANN λ=0（150 ep，无对抗）"],
      [["desync50", "32", "0", "34"],
       ["desync100", "15", "7", "2"]],
      "数据：A-E2 results/controls_comparison.md。")

# ------------------------------------------------------------------ 5 主方法
h2("5　实验三：CDAN-SCA 主方法与核心消融")
p("CDAN-SCA full 在 CDAN 的条件对抗项上乘以目标域熵权重 w=1−H/log256，并去掉源域对抗项（只保留目标域被判别为“源”的项）。"
  "消融 1（Abl.1）去掉条件输入（退化为 DANN 结构 + 熵加权），消融 2 直接复用 A-E2 的 CDAN（无权重、无熵加权，"
  "但其判别器输入为 4096 维特征）。三者构成“条件输入”“熵权重”两因素的 2×2 对照中的三格。")
table("表 6　CDAN-SCA 与消融（3 seeds；top-1 为目标域单类最大占比，collapse 为 top-1>50% 的 epoch 数）",
      ["变体（偏移）", "NTGE（首次）", "稳定率", "GE@10k", "SR@10k", "max top-1", "坍塌 epoch"],
      [["CDAN / Abl.2（desync50）", "10000（0/3）", "0/3", "9.3±6.4", "0", "—", "0"],
       ["Abl.1（marginal+w，desync50）", "7533±4272（1/3）", "0/3", "4.7±2.5", "0", "0.161", "0"],
       ["CDAN-SCA full（desync50）", "6000±4716（2/3）", "1/3", "7.7±10.8", "0.33", "0.610", "5.7"],
       ["CDAN / Abl.2（desync100）", "6567±5105（2/3）", "1/3", "13.7±13.1", "0.33", "—", "0"],
       ["Abl.1（marginal+w，desync100）", "4600±4804（2/3）", "0/3", "18.0±26.1", "0", "0.426", "0.7"],
       ["CDAN-SCA full（desync100）", "6000±4583（2/3）", "0/3", "8.7±5.5", "0", "0.791", "35.3"]],
      "CDAN 的 top-1/坍塌列缺失是因为 A-E2 未记录逐 epoch 监控，只能给终点诊断。")
img("b3_ge50", "图 4　desync50 上五方法 GE 曲线（含 CDAN-SCA full 与 Abl.1；数据：B-E3）")
img("b3_ge100", "图 5　desync100 上五方法 GE 曲线（数据：B-E3）")
img("b3_mon", "图 6　desync100 训练动力学：目标域 top-1 单类占比与预测熵（full vs Abl.1；绿点为 CDAN 终点；数据：B-E3）")
p("主方法没有胜出。desync50 上 full 的首次 NTGE 6000（2/3），但稳定率只有 1/3，GE@10k 7.7，均劣于 CDPA-MMD 的"
  "5700/3/3/0；desync100 上 full 首次 6000（2/3）、稳定 0/3、GE@10k 8.7，同样不如 CDPA-MMD 的 3700/1/3/4.0。"
  "配对比较显示，full 仅在 desync50 上稳定优于 Source Only/DANN/CDAN 的均值（差 −4000/−3167/−4000），"
  "面对 CDPA-MMD 则是 1 胜 0 平 2 负（均值 +300），desync100 上对 CDPA-MMD 为 0 胜 1 平 2 负（均值 +2300）。")
p("消融给出了机制线索。desync50 上 full 优于 Abl.1（6000 vs 7533，配对 2 胜 1 平 0 负），说明条件输入在该偏移上有正贡献；"
  "但 desync100 上 Abl.1 的 GE@10k 反而更差（18.0），而 first NTGE 更低（4600），两者方向不一致，说明指标本身在该偏移上不稳定。"
  "更关键的是坍塌统计：full 在 desync100 的四个预算档/两个偏移全部出现严重坍塌（top-1 0.791、35.3 个 epoch），"
  "而 Abl.1 只有 0.7 个 epoch；结合熵权重诊断（w 均值 0.94–0.98，几乎全部样本接近 1），可以推断："
  "低熵样本获得接近 1 的对抗权重，判别器在这些“自信但错误”的目标样本上被持续强化，进而推高其置信度，形成正反馈。"
  "条件输入（分类器输出参与判别）使判别器更容易利用类别结构，从而放大了这一循环。Abl.1 因为缺少条件输入，反馈弱得多，坍塌也轻得多。")
p("另一个值得记录的现象是：full 在 desync50 的 50 档适应样本下坍塌最重（top-1 0.733、14 个 epoch，ECE 0.929、"
  "confident-wrong 0.840），而 500 档反而最轻（0.537、2.0）——样本越少，熵权重越容易把少数错误簇放大。"
  "这与 C-E4 的少样本结论一致（见第 7 节）。")

h3("5.4 补充实验：把对抗训练“做稳”能否翻盘（adv2）")
p("5.3 的归因指向若干工程性缺陷：条件输入 g=softmax(logits) 让对抗梯度直接回传到分类头、判别器在 45k 源域上过早饱和"
  "（d_acc 0.86–0.92）、分类与对抗梯度在特征提取器上互扯。于是做一版定向改造 adv2，只动对抗侧实现，"
  "主干、划分、配方（RMSprop 1e-5、batch 200、75 epoch、int8 原始输入）全部不变：条件输入 detach；"
  "目标批轻增广两次取 argmax 一致的样本才进条件对抗（硬门控，低于 5% 回退全批）；λ 线性斜坡 20 epoch；"
  "目标域先验 EMA 的均匀正则（权重 0.3）；判别器独立参数组、label smoothing 0.1、梯度裁剪 5.0，"
  "并增加全批辅助判别损失（权重 0.5）以保证判别器看到完整目标分布。6 次训练合计 3.79 GPU·h。")
table("表 6b　adv2 与 full、CDPA-MMD 的对照（3 seeds；N_total=NTGE+45000）",
      ["偏移", "方法", "NTGE（首次）", "稳定率", "GE@1k", "GE@10k", "N_total"],
      [["desync50", "CDPA-MMD", "5700±624（3/3）", "3/3", "30.7", "0.0±0.0", "50700"],
       ["desync50", "CDAN-SCA full", "6000±4716（2/3）", "1/3", "44.3", "7.7±10.8", "51000"],
       ["desync50", "CDAN-SCA adv2", "9967±58（0/3）", "0/3", "61.3", "9.3±9.1", "54967"],
       ["desync100", "CDPA-MMD", "3700±5462（2/3）", "1/3", "12.7", "4.0±4.6", "48700"],
       ["desync100", "CDAN-SCA full", "6000±4583（2/3）", "0/3", "51.3", "8.7±5.5", "51000"],
       ["desync100", "CDAN-SCA adv2", "5933±3754（3/3）", "0/3", "16.3", "5.3±1.5", "50933"]])
p("adv2 没有翻盘：desync50 的 GE@10k 反而从 7.7 变差到 9.3、首次 NTGE 9967（0/3 触发）；desync100 有所改善"
  "（GE@10k 5.3 vs 8.7，GE@1k 16.3 vs 51.3），但仍高于 CDPA-MMD 的 4.0，且稳定率 0/3。坍塌被推迟而未被消除："
  "COLLAPSE 首次出现从 full 的 5.7/35.3 个 epoch 变为 12.7/17.7，末段先验熵仍从 5.5 nats 掉到 4.5 以下，"
  "熵权重 w 依旧饱和到 0.96–0.98。这说明“判别器不稳/梯度互扯/选择偏差”只是次要因素，"
  "真正的驱动是熵权重本身：它在目标域熵塌缩后趋近 1，使加权退化为无权，并把对抗梯度集中到“自信但错误”的样本上。")
p("这一补充实验还留下一条方法论教训：在 12 epoch、6k 轨迹的冒烟里 adv2 全程没有坍塌（top-1 ≤ 0.28），"
  "但 full-scale 下第 13–18 个 epoch 就坍塌。**小规模冒烟只能验证管线，不能用来判定稳定性**，"
  "稳定性必须在正式规模上逐 epoch 监控。")
img("b3_adv2_ge100", "图 5b　desync100 上 adv2、full、CDPA-MMD 与 Source Only 的 GE 曲线（数据：B-E3 补充实验）")
img("b3_adv2_mon", "图 6b　desync100 训练动力学：adv2 与 full 的目标域 top-1 占比与预测熵（数据：B-E3 补充实验）")

# ------------------------------------------------------------------ 6 校准
h2("6　实验四：校准感知变体")
p("B-E5 尝试用温度校准修正熵权重：在目标域无标签条件下无法直接估计温度，因此按常规做法在源域上拟合 T，"
  "再用 p^(1/T) 重新计算熵与权重。结果在 desync100 上看似显著改善：NTGE 3133±3402（3/3 首次触底），"
  "GE@10k 0.33±0.58，SR@10k 0.67，N_total 48133，是唯一在均值口径上“达标”的变体；但 desync50 反而变差（8467、1/3）。")
table("表 7　校准感知变体与 full 对比（3 seeds）",
      ["变体（偏移）", "NTGE（首次）", "稳定率", "GE@10k", "N_total", "温度 T", "w 均值"],
      [["CDAN-SCA full（desync50）", "6000（2/3）", "1/3", "7.7±10.8", "51000", "—", "0.939"],
       ["CDAN-SCA-calib（desync50）", "8467±2656（1/3）", "1/3", "1.7±1.5", "53467", "12.31", "0.017"],
       ["CDAN-SCA full（desync100）", "6000（2/3）", "0/3", "8.7±5.5", "51000", "—", "0.977"],
       ["CDAN-SCA-calib（desync100）", "3133±3402（3/3）", "2/3", "0.33±0.58", "48133", "12.31", "0.014"]],
      "稳定率按末尾连续 GE=0 判定；desync100 的 3/3 首次触底中仅 2/3 在约 9200 条处稳定，600 条为瞬态；SR@10k 见正文（full 0 / calib 0.67）。")
img("b5_ge100", "图 7　desync100 上校准感知变体与全部基线的 GE 曲线（数据：B-E5）")
img("b5_w", "图 8　攻击侧熵权重分布：full 的裸权重（0.93）与校准后权重（0.02）；数据：B-E5")
p("这一改善不能归因于“校准得更准”。温度拟合的诊断显示：source_val 上的分类准确率仅 0.56%（接近 256 类随机水平 0.39%），"
  "在其上拟合出的 T*=10.86（全局取 12.31）本质是在拟合一个“几乎均匀”的分布；把它用于目标域后，"
  "校准概率接近均匀，权重 w 从 0.94 塌到 0.02，对抗项几乎被关闭。对照 A-E2 的 λ=0 结果（desync100 无对抗、150 epoch，"
  "GE@10k=2），calib 的 0.33 更像是“关掉对抗项 + 75 epoch”的组合效应，而非校准带来的信息增益。"
  "desync50 上同一温度使 calib 变差（8467 vs 6000），进一步支持“T 是划分属性而非模型属性”的判断："
  "source_val 与目标域差异越大，源域拟合的 T 越没有迁移意义。")
p("同时必须指出，calib 并没有解决坍塌：其目标域 top-1 单类占比仍达 0.815，坍塌 56.7 个 epoch，甚至高于 full。"
  "这说明“预测坍塌”与“攻击失效”不是同一件事：在没有对抗项的情况下，模型可以继续坍塌，但攻击侧仍可能恢复密钥"
  "（或至少不再被对抗项主动破坏）。这一分离现象对后续方法设计很重要——用 top-1 占比单独判断方法好坏会误导。")

# ------------------------------------------------------------------ 7 样本量
h2("7　实验五：目标域适应样本量敏感性")
p("C-E4 把 target_adapt 从 45000 降到 50/500/5000（45000 档复用前序结果），对 CDPA-MMD、DANN、CDAN-SCA full 各跑 3 seeds，"
  "共 54 次训练（desync50 与 desync100 各 27 次）。所有方法在同一 seed 下共用同一嵌套子集；目标域 batch 循环采样，"
  "源域训练量保持 45000 条不变，因此唯一变量是适应样本数。评价以 N_total@GE=1 为主判据。")
table("表 8　desync50 总成本 N_total@GE=1（† 表示 cell 内至少一个 seed 未触发，均值为下界）",
      ["方法", "50", "500", "5000", "45000"],
      [["DANN", "10050†", "9533†", "13633†", "54167†"],
       ["CDAN-SCA full", "10050†", "8467†", "12567†", "51000†"],
       ["CDPA-MMD", "8050†", "5233", "10167", "50700"],
       ["Source Only", "—", "—", "—", "10000"]])
table("表 9　desync100 总成本 N_total@GE=1",
      ["方法", "50", "500", "5000", "45000"],
      [["DANN", "7017†", "10500†", "8433", "53300†"],
       ["CDAN-SCA full", "9383†", "7867†", "11867†", "51000†"],
       ["CDPA-MMD", "3883†", "7233†", "8733†", "48700†"],
       ["Source Only", "—", "—", "—", "3700†"]])
img("ce4_cost50", "图 9　desync50 的预算—总成本曲线（空心标记表示至少一个 seed 删失，成本为下界；数据：C-E4）")
img("ce4_cost100", "图 10　desync100 的预算—总成本曲线（数据：C-E4）")
img("ce4_stab100", "图 11　desync100 各预算档的稳定 GE=0 比例（数据：C-E4）")
p("desync50 呈现清晰的预算效应：CDPA-MMD 在 500 条时就已 3/3 首次触底（4733±694），稳定率 2/3，N_total=5233，"
  "比 45000 档省 89.7%；5000 与 45000 档实现 3/3 稳定（GE@10k=0）。50 条档掉到 1/3 触发，说明“太少”仍不够。"
  "CDAN-SCA full 在任何预算档都没有 3/3 稳定，50 档 0/3 触发且 N_total 下界已比 CDPA-MMD 高 24.8%，"
  "因此少样本生态位不成立。")
p("desync100 则没有可用拐点：没有任何 (方法, 预算) 组合达到 3/3 稳定（最高 1/3：CDPA@5000、CDPA@45000、DANN@5000、CDAN@50）；"
  "GE@10k 最好的是 CDPA-MMD@45000=4.0。更严重的是，低预算档的名义低 NTGE 在置换下不复现——"
  "CDPA@50 的 700 在 10 次轨迹重排中 10/10 变回 10000，CDPA@500 的 200 变为 7340±3506，CDPA@5000 的 400 变为 4700±2661。"
  "因此这些“200–900 条恢复密钥”的读数应判定为顺序幸运造成的假最优，desync100 上不存在“少量适应样本即可恢复”的证据。")
table("表 10　轨迹顺序置换（10 次，step=100）代表性结果",
      ["运行", "原始 NTGE", "置换均值±sd", "区间"],
      [["cdpa desync100 b50 s42", "700", "10000±0", "[10000, 10000]"],
       ["cdpa desync100 b500 s42", "200", "7340±3506", "[900, 10000]"],
       ["cdpa desync100 b5000 s42", "400", "4700±2661", "[100, 7500]"],
       ["dann desync100 b5000 s43", "300", "8250±3268", "[1100, 10000]"],
       ["cdpa desync50 b500 s42", "4100", "3530±2078", "[900, 7400]"],
       ["cdpa desync50 b5000 s44", "5300", "1970±2049", "[300, 5700]"]],
      "全部 25 个出现 GE=0 的运行均做了置换，完整结果见各实验 results/permutation.json。")

# ------------------------------------------------------------------ 8 讨论
h2("8　讨论")
h3("8.1　主方法为何未胜")
p("把第 5、6、7 节的证据放在一起，CDAN-SCA 的失效路径可以概括为三步。第一，目标域熵加权在训练早期就把权重推向 1："
  "目标域预测迅速变得低熵（无论对错），w 均值从 0.7 升到 0.94–0.98，几乎所有样本都获得接近满额的对抗权重，"
  "“按置信度选择样本”的设计意图失效。第二，判别器以条件信息（softmax 输出）为输入，在目标域标签缺失时会把"
  "“自信且一致的错误类别”当作可分的域特征，反过来强化该类别，形成自信错误的正反馈；证据是 full 的 top-1 单调上升"
  "并长期停在 0.6–0.85，而 Abl.1（无条件下输入）坍塌轻得多。第三，对抗梯度与分类梯度方向冲突，"
  "在无 BN、学习率仅 1e-5 的配方下，特征层来不及通过正常分类修正这种偏移，最终表现为攻击侧 rank 不再下降。")
p("相较之下，CDPA-MMD 只在特征层做矩匹配，没有“按样本重加权”的路径，因此不会主动放大错误簇；它的收益也更稳健"
  "（desync50 上 3/3 稳定，desync100 上至少 1/3 稳定且 GE@10k 4.0）。这提示：在 SCA 这类低信噪比、类别极多（256 类）、"
  "目标域无标签的问题上，样本级重加权比分布级匹配更容易失稳。")

h3("8.2　“校准”一词的歧义")
p("B-E5 的结果表面上支持“校准能救对抗 UDA”，但温度拟合的数据源决定了这一解读不成立：在 source_val（准确率 0.56%）上"
  "拟合出的 T 接近 10–12，等价于把预测压成近均匀分布，权重因此接近 0，对抗项被关闭。这是“用校准参数实现了一个软开关”，"
  "不是“校准让权重更可靠”。若要检验校准本身的价值，应在目标域上以无标签方式（如熵最小化、一致性）估计温度，"
  "或在 source_val 准确率足够高的设置下重复本实验；当前证据不足。")

h3("8.3　评价口径比方法差异更大")
p("本文多次出现“同一模型、同一权重、只是攻击轨迹顺序不同”导致 NTGE 从 200 变到 10000 的情况（表 10）。"
  "在 10k 轨迹、256 类的设定下，GE 曲线的首次零点对顺序高度敏感，单次点估计的标准差可达数千条；"
  "因此方法排序必须建立在“是否 3/3 稳定 + 置换均值 + N_total 量级”上，而不是比较两个相近的首次 NTGE。"
  "A-E2 的 λ=0 对照还说明，训练时长（75 vs 150 epoch）本身就能改变 GE@10k 数个名次，任何 UDA 增益都必须与该混淆因素分离。")

# ------------------------------------------------------------------ 9 问题
h2("9　存在的问题与局限")
bullets([
 "P1（核心）主方法没有胜出，也没有少样本生态位：desync50 四档、desync100 四档的 N_total 全部高于 CDPA-MMD；"
 "full 在任一 shift 都没有 3/3 稳定。按任务预设的硬判据（desync50 NTGE≤5700 且 3/3 稳定；desync100 GE@10k≤4.0 且 NTGE≤3700）均未达标。",
 "P2（关键缺失）没有“同 ckpt、同 75 epoch、λ_adv=0”的对照。现有 λ=0 对照是 150 epoch 的 DANN 结构，"
 "无法把“对抗项”与“训练时长/结构”分开；这是当前归因链上唯一的空缺，建议补 6 次训练（约 3.2 GPU·h）。",
 "P3 熵权重的方向性未验证：w=1−H/log256 会把低熵样本权重推到 1，但低熵在目标域往往意味着“自信地错”。"
 "反向加权（高熵优先）或对 w 设上限/截断尚未实验，因此“熵加权本身有害”目前只是强证据下的推断，而非对照结论；"
 "补充实验 adv2（5.4）已经把对抗实现侧的缺陷逐一修掉仍然失败，这加强了该推断，但反向加权仍未做。",
 "P4 校准变体的温度在近乎随机的 source_val 上拟合（acc 0.56%，T≈12.3），校准后 w≈0.014–0.017，实质关闭对抗项；"
 "B-E5 在 desync100 的“WIN”只在均值口径成立，且 600 条首触为瞬态（置换后不复现），不能作为“校准有效”的证据。",
 "P5 NTGE 首零点不稳定：多个运行的原始值（200–900）在 10 次重排后变为 4700–10000；desync100 低预算档的“低成本”是假最优。"
 "主表已改用稳定口径 + 置换均值，但点估计仍出现在部分图表中，正式论文需统一。",
 "P6 统计功效不足：3 seeds、attack 上限 10k，大量 cell 截尾（†），均值与标准差被截尾值主导；"
 "建议补到 5 seeds 并给出 bootstrap 95% CI，或把 attack 集扩到 20k（若数据允许）。",
 "P7 “跨设备”名不副实：fixed→desync50/100 只是同一芯片的采集窗口偏移，不含器件间差异；"
 "要支撑“跨设备 UDA”，需要 CDPA、variable-key 等数据集，相关工作尚未开始。",
 "P8 A-E2 缺少逐 epoch 监控（checkpoint 每 5 epoch 覆盖写），CDAN 的坍塌/熵曲线只能用终点诊断近似，"
 "导致第 5 节的三向对比中 CDAN 一列不完整，削弱了“条件输入导致坍塌”的因果强度。",
 "P9 CDPA-MMD 的移植与原实现有 6 处偏差（主干换官方 CNN、RMSprop 1e-5 替 Adam 1e-3、75 替 15 epoch、"
 "45000 条 target_adapt 替 200 条循环、无 BN 可冻、预训练用 B-E1 同 seed checkpoint），"
 "因此“强基线”的绝对水平不能直接与原文数字比较，只能在同一协议内比较。",
 "P10 GE 计算依赖 key 枚举顺序与轨迹顺序；不同实现的 rank 定义（是否含 key=0、是否用 log 似然累加）会改变曲线形状。"
 "本文统一复用 experiments/cdan-sca-diagnosis/scripts/evaluate_ascad.py，但仍建议在论文附录给出伪代码与置换协议。",
 "P11 计算与存储成本已成瓶颈：C-E4 单实验 54 次训练 27.17 GPU·h、30 GB checkpoint，完整 manifest 647 MB；"
 "继续扩大 seed/数据集前需要先做 checkpoint 淘汰策略（仅保留 GE=0 与最优 run）。",
])

# ------------------------------------------------------------------ 10 结论
h2("10　结论与下一步")
p("在 ASCAD v1 fixed→desync50/desync100 的统一协议下，本文得到三点结论。其一，同域 CNN 能在 800–1433 条攻击轨迹内"
  "稳定恢复密钥，而跨分布下最强方法（CDPA-MMD）在 desync50 需要 500 条适应样本 + 4733 条攻击轨迹、在 desync100 上"
  "仍无法做到 3/3 稳定；域偏移的代价远大于方法差异。其二，CDAN-SCA 的熵加权 + 条件对抗在该任务上没有带来收益："
  "它在两个偏移上都不如 CDPA-MMD，且在 desync100 上 6/6 运行坍塌；校准变体的表面改善来自关闭对抗项，而非校准信息。"
  "其三，评价口径本身是结论的一部分：首次 NTGE 会被轨迹顺序主导，低预算档的“超低成本”在置换下消失，"
  "只有稳定率、置换均值与 N_total 三者同时报告，才能支撑“某方法更好”的判断。")
p("下一步工作按优先级排列：（1）补 λ_adv=0 的同 ckpt、同 75 epoch 对照（6 次训练），完成归因闭环；"
  "（2）补 5 seeds 与 bootstrap CI，重点覆盖 desync50@500 与 desync100@5000/45000；"
  "（3）把 10 次置换均值纳入所有主表，作为默认口径；（4）在 CDPA/variable-key 上验证“跨设备”，"
  "现有 fixed→desync 只能称“跨采集条件”；（5）若继续 CDAN-SCA 主线，先修少样本/强偏移下的过自信"
  "（温度校准改为目标域无标签估计、w 截断、熵下限），并以“3/3 稳定 GE=0 且 N_total 低于 CDPA-MMD 同档”为准入条件；"
  "（6）为控制成本，建立 checkpoint 保留策略并固定随机种子集合。")

# ------------------------------------------------------------------ 附录
h2("附录 A　复现入口与产物")
bullets([
 "E0：experiments/00_common_e0/（train_e0_torch.py、evaluate_ascad.py、results/eval/summary.json）。",
 "B-E1：experiments/02_member_b_method/b_e1_architecture_same_domain/（train_b_e1.py、evaluate_b_e1.py、results/summary.json）。",
 "A-E2：experiments/01_member_a_baselines/a_e2_uda_baselines/（train_uda.py、evaluate_a_e2.py、controls_comparison.md）。",
 "A-E3：experiments/01_member_a_baselines/a_e3_cdpa_or_adabn/（cdpa_common.py、train_cdpa.py、evaluate_a_e3.py、table_methods.csv）。",
 "B-E3/B-E4：experiments/02_member_b_method/b_e3_cdan_sca/ 与 b_e4_ablation/（cdan_sca_common.py、train_cdan_sca.py、"
 "evaluate_b_e3.py、permutation_b_e3.py、results/table_three_way.csv）。",
 "B-E5：experiments/02_member_b_method/b_e5_calibration_aware/（train_cdan_sca.py --mode calib、temperature_transfer_diagnostic.json）。",
 "C-E4：experiments/03_member_c_robustness/c_e4_target_sample_budget/（train_budget.py、aggregate_c_e4.py、"
 "permutation_c_e4.py、make_tables.py、README.md 8.1.1–8.1.13）。",
 "统一评估：experiments/cdan-sca-diagnosis/scripts/evaluate_ascad.py（GE/key rank，step=100，byte=2）。",
 "大文件不入库：checkpoint 位于 outputs/models/<实验>/，predictions_attack.npy 由 .gitignore 排除，"
 "manifest 记录路径、字节数与 SHA256。",
])
note("本文为初稿，数值截至 2026-11-20；图表由各实验 results/ 自动生成，引用时请以对应 results/summary.json 为准。"
     "参考文献待补：ASCAD 数据集、DANN、CDAN、CDPA-MMD、DASA-SCA 等原始论文信息需在投稿前补全并核对卷期。")

# ------------------------------------------------------------------ 渲染
def md_table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)

def render_md():
    lines, tn, fn = [], 0, 0
    for b in DOC:
        if b[0] == "h1":
            lines += ["# " + b[1], ""]
        elif b[0] == "h2":
            lines += ["## " + b[1], ""]
        elif b[0] == "h3":
            lines += ["### " + b[1], ""]
        elif b[0] == "p":
            lines += [b[1], ""]
        elif b[0] == "bullets":
            lines += ["- " + x for x in b[1]] + [""]
        elif b[0] == "table":
            tn += 1
            lines += ["**" + b[1] + "**", "", md_table(b[2], b[3])]
            if b[4]:
                lines += ["", "*" + b[4] + "*"]
            lines += [""]
        elif b[0] == "img":
            fn += 1
            rel = "../../" + F[b[1]]
            lines += ["![" + b[2] + "](" + rel + ")", "", "*" + b[2] + "*", ""]
        elif b[0] == "note":
            lines += ["> " + b[1], ""]
    return "\n".join(lines)

def render_docx():
    from docx import Document
    from docx.shared import Pt, Cm, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.oxml.ns import qn
    from docx.enum.section import WD_SECTION

    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    sec.top_margin = sec.bottom_margin = Cm(2.54)
    sec.left_margin = sec.right_margin = Cm(2.6)

    def set_font(style, latin="Times New Roman", east="宋体", size=None, bold=None, color=None):
        style.font.name = latin
        rpr = style.element.get_or_add_rPr()
        rf = rpr.find(qn("w:rFonts"))
        if rf is None:
            rf = rpr.makeelement(qn("w:rFonts"), {})
            rpr.append(rf)
        rf.set(qn("w:eastAsia"), east)
        rf.set(qn("w:ascii"), latin)
        rf.set(qn("w:hAnsi"), latin)
        if size is not None:
            style.font.size = Pt(size)
        if bold is not None:
            style.font.bold = bold

    st = doc.styles
    set_font(st["Normal"], size=10.5)
    set_font(st["Heading 1"], east="黑体", size=16, bold=True)
    set_font(st["Heading 2"], east="黑体", size=13, bold=True)
    set_font(st["Heading 3"], east="黑体", size=11.5, bold=True)
    st["Normal"].paragraph_format.line_spacing = 1.25
    st["Normal"].paragraph_format.space_after = Pt(4)

    first_h1 = True
    for b in DOC:
        if b[0] == "h1":
            para = doc.add_heading(b[1], level=1)
            para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            if first_h1:
                first_h1 = False
        elif b[0] == "h2":
            doc.add_heading(b[1], level=2)
        elif b[0] == "h3":
            doc.add_heading(b[1], level=3)
        elif b[0] == "p":
            para = doc.add_paragraph(b[1])
            para.paragraph_format.first_line_indent = Pt(21)
            para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        elif b[0] == "bullets":
            for x in b[1]:
                para = doc.add_paragraph("• " + x)
                para.paragraph_format.left_indent = Pt(12)
                para.paragraph_format.first_line_indent = Pt(-6)
                para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        elif b[0] == "table":
            cap = doc.add_paragraph()
            cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = cap.add_run(b[1])
            run.bold = True
            run.font.size = Pt(9)
            t = doc.add_table(rows=1, cols=len(b[2]))
            t.style = "Table Grid"
            t.alignment = WD_TABLE_ALIGNMENT.CENTER
            t.autofit = True
            tblPr = t._tbl.tblPr
            for tag in ("w:tblW",):
                old = tblPr.find(qn(tag))
                if old is not None:
                    tblPr.remove(old)
            tblW = tblPr.makeelement(qn("w:tblW"), {qn("w:type"): "pct", qn("w:w"): "5000"})
            tblPr.append(tblW)
            fs_pt = Pt(7.8) if len(b[2]) >= 6 else Pt(8.5)
            hdr = t.rows[0].cells
            for i, htxt in enumerate(b[2]):
                hdr[i].text = ""
                run = hdr[i].paragraphs[0].add_run(htxt)
                run.bold = True
                run.font.size = fs_pt
                hdr[i].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
            for row in b[3]:
                cells = t.add_row().cells
                for i, val in enumerate(row):
                    cells[i].text = ""
                    run = cells[i].paragraphs[0].add_run(str(val))
                    run.font.size = fs_pt
                    cells[i].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER if i else WD_ALIGN_PARAGRAPH.LEFT
            if b[4]:
                cap2 = doc.add_paragraph()
                cap2.alignment = WD_ALIGN_PARAGRAPH.LEFT
                run = cap2.add_run("注：" + b[4])
                run.font.size = Pt(7.5)
                run.font.color.rgb = RGBColor(0x44, 0x44, 0x44)
            doc.add_paragraph()
        elif b[0] == "img":
            path = REPO / F[b[1]]
            if not path.exists():
                raise SystemExit("missing figure: " + str(path))
            para = doc.add_paragraph()
            para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            para.add_run().add_picture(str(path), width=Cm(14.6))
            cap = doc.add_paragraph()
            cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = cap.add_run(b[2])
            run.font.size = Pt(9)
            run.bold = False
            doc.add_paragraph()
        elif b[0] == "note":
            para = doc.add_paragraph()
            para.paragraph_format.left_indent = Pt(18)
            run = para.add_run("注：" + b[1])
            run.font.size = Pt(8)
            run.font.color.rgb = RGBColor(0x44, 0x44, 0x44)
    doc.core_properties.title = "跨分布深度学习侧信道分析的无监督域适应：实验研究报告（初稿）"
    doc.core_properties.author = "SCA UDA 实验组"
    doc.save(str(OUT_DOCX))

if __name__ == "__main__":
    md = render_md()
    OUT_MD.write_text(md, encoding="utf-8")
    render_docx()
    print("wrote", OUT_MD.name, len(md), "chars")
    print("wrote", OUT_DOCX.name, OUT_DOCX.stat().st_size // 1024, "KB")
