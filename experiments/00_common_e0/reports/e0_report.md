# E0 实验报告：ASCAD fixed-key Source Only 最小端到端实验

**实验编号**：E0（共同最小实验）　**执行人**：王博超　**日期**：2026-10-10

#### 8.1.1 实验级别

Formal（单 seed；按规范 E0 允许每人不同 seed，正式多 seed 比较自 E1 起）

#### 8.1.2 研究问题

数据读取、模型训练、256 类预测、key rank/GE 计算与归档的全流程链路是否打通？Source Only 在 ASCAD fixed-key 上的基线攻击成本是多少？

#### 8.1.3 研究假设

按官方配方（五层 CNN + RMSprop lr=1e-5 + 原始幅度输入）训练，GE 应在千条攻击轨迹内归 0；若使用非官方优化器/预处理组合，收敛可能显著变慢或失败。

#### 8.1.4 数据与划分

- 数据集：ASCAD v1 fixed-key（`ASCAD.h5`，Profiling 50000 / Attack 10000，700 点，int8）
- 源域：ASCAD fixed-key（同域实验，源域=目标域）
- profiling 数据：Profiling_traces[0:45000]（source_train）
- adaptation 数据：无（Source Only 不使用目标域适应数据）
- attack evaluation 数据：Attack_traces[0:10000]（target_attack）
- 验证数据：Profiling_traces[45000:50000]（source_val，仅监控，不参与训练）
- 目标字节：byte=2（第 3 字节）
- 标签定义：Sbox[plaintext[2] ⊕ key[2]]，已对全量 50000+10000 条校验与 metadata 一致

#### 8.1.5 方法与对比

- baseline：Source Only（本实验即基线本身）
- proposed：无
- ablation：无（过程中的失败配方作为异常记录见 8.1.11）

#### 8.1.6 固定变量与自变量

固定：backbone（官方五层 CNN）、标签（ID 256 类）、目标字节、数据划分、评估脚本（`experiments/cdan-sca-diagnosis/scripts/evaluate_ascad.py`，step=100）。
自变量：无（单配置链路验证）。

#### 8.1.7 运行配置

- seed：42
- batch size：200
- epoch：75
- learning rate：1e-5（RMSprop）
- lambda：无
- 代码版本或 commit：见 `manifests/manifest.json`（git_commit 字段）
- 运行命令：

```bash
python experiments/00_common_e0/scripts/train_e0_torch.py \
    --config experiments/00_common_e0/configs/e0_source_only_fixed_official.py
python experiments/cdan-sca-diagnosis/scripts/evaluate_ascad.py \
    --predictions outputs/results/00_common_e0/predictions_attack.npy \
    --dataset <ASCAD.h5 路径> --step 100 \
    --out-dir experiments/00_common_e0/results/eval
python experiments/00_common_e0/scripts/finalize_e0.py
```

#### 8.1.8 指标

- GE/key rank：GE 曲线见 `figures/e0_ge_curve.png`
- SR：SR@1000 = 100%，SR@10000 = 100%
- NTGE：800 条攻击轨迹
- target accuracy：0.78%（ID 模型 256 类，低 accuracy 属预期，GE 为判据）
- domain accuracy：不适用（Source Only 无域判别器）

#### 8.1.9 结果

| 指标 | 值 |
| --- | --- |
| NTGE（GE=0 所需攻击轨迹数） | 800 |
| GE @ 10000 | 0 |
| SR @ 1000 / @10000 | 100% / 100% |
| 最终 train loss / val accuracy | 3.49 / 0.78% |
| 训练耗时 | 约 1170 s（15.6 s/epoch × 75，RTX 5060 Laptop） |

GE 在 800 条处归 0 并稳定保持到 10000 条（`results/e0_metrics.csv`、`results/eval/summary.json`）。训练动态呈典型「平台期—悬崖」模式：loss 平台期至约 epoch 30，GE 探针（500 条攻击轨迹）自 epoch 34 起稳定为 0（`figures/e0_training_dynamics.png`）。

#### 8.1.10 结果解释

- 事实：NTGE=800，GE@10000=0，SR 全通过。
- 解释：结果与 ASCAD 文献中官方 CNN 的正常水平一致，说明数据链路、标签语义、模型实现和 GE 计算全部正确；低 accuracy 下的 GE 收敛符合掩码数据集 ID 模型的已知特征。
- 主张：本机流水线可作为后续全部对照实验的基础设施；本结果作为 M0（Source Only）基线参照点。

#### 8.1.11 失败、限制与异常

1. **失败配方记录（不删除，留作经验）**：Adam(lr=1e-4) + 特征标准化训练 60 epochs，train loss 降至 0.17 但 val accuracy 与 GE 完全不收敛——典型的「记忆训练集」假收敛。结论：后续实验固定使用官方配方（RMSprop lr=1e-5 + 原始幅度输入）。
2. **仓库草稿配置（Adam lr=1e-5）在 CPU 上 6 epochs 无任何收敛迹象**，判定不适用于本机实验节奏。
3. **五月遗留权重（3-block CNN）复测 GE@10000=59**，与五月报告数字不一致，判定不可用，未纳入任何结果（评估脚本 `scripts/eval_legacy_model.py`，图 `figures/e0_ge_curve_legacy.png`）。
4. **实现偏差**：因 TensorFlow ≥2.11 原生 Windows 不支持 GPU，训练侧使用 PyTorch 2.9.1+cu128 逐层复刻 `src/framework` 架构；评估与归档使用仓库标准脚本，与框架无关。
5. 限制：单 seed；source_val 仅监控未用于早停。

#### 8.1.12 结论边界

本结果仅说明同域（ASCAD fixed-key → fixed-key）条件下 Source Only 的攻击成本，不能作为任何跨域方法有效性的证据；跨域结论须以 E1（fixed→desync50/100）及后续实验为准。

#### 8.1.13 下一步实验

E1（对应分工 B-E1/B-E2）：以同一官方配方、同一数据划分复现 MLP / 官方 CNN / ResNet1D 三种主干，同域 + desync50/desync100 迁移比较，各 3 个 seed，筛选并冻结统一主干。

---

**附带文件**（按 §8.2 规范）：

```text
README.md               实验概述与复现说明
configs/                完整配置（official 配置为最终结果）
scripts/                训练/评估/收尾入口
logs/e0_train.log       训练日志
results/                e0_metrics.csv, eval/summary.json, eval/predictions_attack.csv, history.json
figures/                e0_ge_curve.png, e0_training_dynamics.png
manifests/manifest.json 运行命令、seed、环境、产物路径与哈希
```

大型文件（不进 Git，按 §13.8 保存于本地）：
- `outputs/results/00_common_e0/predictions_attack.npy`（9.8 MB，路径/哈希见 manifest）
- `outputs/models/00_common_e0/e0_model_torch.pt`（255 MB，路径/哈希见 manifest）
