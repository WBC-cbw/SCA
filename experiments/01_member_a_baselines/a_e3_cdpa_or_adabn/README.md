# A-E3 CDPA-MMD 领域强基线（ASCAD fixed-key → desync50 / desync100）

本目录把侧信道领域的强基线 CDPA-MMD（Cross-Device Profiled Attack 的 MMD 变体）移植到
本项目冻结的 Official CNN 与官方训练配方上，与 Source Only / DANN / CDAN 在同一数据划分、
同一评估脚本下对照，回答“领域专用对齐方法是否真的比通用 UDA 强”。

**一句话结论**：desync50 上 CDPA-MMD 是唯一 3/3 稳定 GE=0 的方法（NTGE 5700 ± 625，SR 100%），明确强于 Source Only / DANN / CDAN；desync100 上其 NTGE 与 Source Only 打平（3700），但 GE@10k 最低（4.0）、无负迁移，且校准（ECE 0.73、confident-wrong 0.36）明显好于 DANN/CDAN。

**目录约定**：configs/ 配置，scripts/ 代码，results/ 逐 run 产物与评估，figures/ 论文图，
logs/ 训练与编排日志，reports/ 报告与表，manifests/ 清单。

## 8.1.1 实验级别与运行清单

- 级别：**formal**（主结果 6 次微调训练）；另有 smoke / λ=0 / lr 诊断 3 个 pilot（见 8.1.11，不计入主矩阵）。
- 矩阵：CDPA-MMD × 目标域 {desync50, desync100} × seed {42, 43, 44} = 6 次训练。
  Source Only 复用 B-E2 的 `results/predictions/<shift>/cnn_s<seed>.npy`；DANN / CDAN 复用
  A-E2 的 `results/<run>/predictions_attack.npy`。三种基线的预测**不重训**，只在本实验的
  GE 实现下重新评估，保证四方法同一把尺子。
- 每次训练 1726–1951 s（75 epoch，25–26 s/epoch，RTX 5060 Laptop），6 次合计 11239 s ≈ 3.12 GPU 小时。
- 运行环境：Windows 11，venv Python 3.12，torch 2.9.1+cu128，CUDA 可用；TF32 开启；
  其余超参与官方配方一致（见 8.1.5）。
- 编排：`scripts/run_cdpa_queue.bat` 串行跑 6 个 run（Windows 计划任务 AE3_CDPA_queue），
  每 run 独立日志 `logs/cdpa_<shift>_s<seed>.log`，总状态 `logs/queue.log`，完成写 `logs/QUEUE_DONE.txt`。

复现命令：

    python scripts/train_cdpa.py --shift desync50 --seed 42     # 其余 5 个同理
    python scripts/evaluate_a_e3.py
    python scripts/aggregate_a_e3.py
    python scripts/analyze_confidence_a_e3.py
    python scripts/plot_a_e3.py
    python scripts/plot_cdpa_dynamics.py
    python scripts/make_manifest.py

## 8.1.2 目标与假设

1. **H1（领域专用对齐更强）**：CDPA-MMD 用“预训练 + 微调”两阶段，并在多个全连接层上做多核
   RBF MMD，理论上比 DANN（单层边缘对齐）、CDAN（条件化对齐）更贴近 SCA 的跨设备差异。
   若 H1 成立，desync50/desync100 的 NTGE 应低于 Source Only，且在四方法中最好（或至少不差于 DANN/CDAN）。
2. **H2（强基线门槛）**：若 CDPA-MMD 并未超过通用 UDA，则说明“领域专用”并不自动更强，
   后续 CDAN-SCA 主方法的门槛应取四方法在每个偏移上的最好成绩，而不是只盯 DANN/CDAN。
3. **H3（对齐强度）**：在 RMSprop lr=1e-5 + 75 epoch 的官方配方下，λ_mmd=0.1 的 MMD 梯度只占
   分类梯度的约 1%（8.1.4 的梯度探针），可能不足以改变表示；若结果与分析一致，则
   “对齐强度不足”是 CDPA-MMD 在本协议下的主要限制之一。

评估只看 GE/NTGE/SR 与总轨迹成本；accuracy 仅作辅助诊断。

## 8.1.3 数据与划分

与 B-E2 / A-E2 完全一致，标签按 Sbox[plaintext[2] xor key[2]] 现算（B-E2 已对三个数据集
50000 条 Profiling 标签做过 0 mismatch 校验）。

| 划分 | 来源 | 条数 | 用途 |
|---|---|---|---|
| source_train | ASCAD.h5 Profiling_traces[0:45000] | 45000 | 有标签 CE（微调阶段仍保留） |
| source_val | ASCAD.h5 Profiling_traces[45000:50000] | 5000 | 源域验证曲线（只记录） |
| target_adapt | 目标数据集 Profiling_traces[0:45000] | 45000 | 无标签，**仅**进入 MMD 损失 |
| target_domain_val | 目标数据集 Profiling_traces[45000:50000] | 5000 | 不参与训练（本实验未使用） |
| target_attack | 目标数据集 Attack_traces[0:10000] | 10000 | GE/NTGE/SR 评估 |

关键约束：target_adapt 与 target_attack 严格分离；攻击集不参与训练或模型选择
（GE probe 每 2 epoch 记录 500 条攻击轨迹，只写入 history，不早停、不选模型）。
适应轨迹计入总成本：N_total@GE=1 = N_attack@GE=1 + 45000（Source Only 无适应轨迹，N_total = N_attack）。

## 8.1.4 方法与实现

### 官方实现与本项目移植的对应关系

官方代码：`sca_repo/baselines/cdpa/CDPA/Cross-Device-Profiled-Attack-main/Different_Implementations/ASCAD_addDesync100_Demo.ipynb`
（只有 notebook，无 .py）。其网络 `Net` / `CDP_Net` 为：

    Conv1d(1,32,k=1)+SELU+BN1d+AvgPool1d(2,2)
    Conv1d(32,64,k=50)+SELU+BN1d+AvgPool1d(50,50)
    Conv1d(64,128,k=3)+SELU+BN1d+AvgPool1d(2,2)+Flatten      # 256 维
    Linear(256,20)+SELU  ->  Linear(20,20)+SELU  ->  Linear(20,20)+SELU  ->  Linear(20,256)

微调时 `CDP_Net.forward(source, target)` 在三个全连接表示上累加 MMD：
flatten 后 256 维、过 classifier_1 后的 20 维、过 classifier_2 后的 20 维；
**不含** classifier_3 之后、也不在卷积特征上做 MMD。损失 = 分类 CE + `lambda_ * mmd_loss`，
官方 `lambda_ = 0.1`、Adam lr=1e-3、`total_epoch=100`（预训练）、`finetune_epoch=15`、
批 200，微调阶段每步取一个 source batch + 一个 batch 目标轨迹（官方只缓存 `target_finetune_num=200` 条
目标轨迹循环使用），且 `model.eval()` 冻结 BN 与 dropout。

本项目的 `scripts/cdpa_common.py` 把同一套 MMD 目标移植到 **Official CNN**（B-E1 冻结结构）：

- 分类主干：convs（64/128/256/512/512，k=11，AvgPool）+ fc1(10752→4096) + fc2(4096→4096) + fc3(4096→256)；
- MMD 作用在三个表示上：`conv`（flatten 后 10752 维）、`fc1`（ReLU 后 4096 维）、`fc2`（ReLU 后 4096 维），
  与官方“多个全连接层、由浅到深”的设计动机一致（逐层对齐，而不是只对齐最后特征）；
- `mmd_rbf(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None)` 与官方数值等价：
  带宽取批内平均平方距离 `/(n^2-n)`，再除以 `kernel_mul ** (kernel_num // 2)`，5 个核累加；
  损失 = `mean(XX + YY - XY - YX)`；
- 距离用 `||x||^2 + ||y||^2 - 2xy` 计算。官方 notebook 用 `x.expand(...)` 显式展开，
  在 4096 维、batch 200 时单项约 400×400×4096 float32 ≈ 2.6 GB，会 OOM；改成矩阵乘法后
  数值相同（同核函数、同带宽），只避免显存爆炸。这一点写入 `run_meta.json` 的 `mmd_impl_note`。

**预训练阶段**：直接以 B-E1 同 seed 的 source-only checkpoint 作为预训练模型
（`outputs/models/b_e1/cnn_s<seed>/ckpt.pt`），不在本实验重训预训练——B-E1 与 E0 已有
同配方、同 seed 的 75 epoch 结果，重训只会引入不必要的差异。官方是 100 epoch 从头预训练，
这里用“同配方同 seed 的 source-only”替代，属于**有意的协议对齐**，不是遗漏。

**微调阶段**：只使用 target_adapt 的无标签轨迹做 MMD；CE 仍用 source_train 标签；
target 标签不出现任何地方。每步 = 一个 source batch（200）+ 一个 target batch（200），
target 轨迹每 epoch 重新打乱采样（官方是 200 条循环，本实验用全部 45000 条，见下表差异 4）。

### 与官方实现的差异（全部显式记录，见 `run_meta.json.official_deviations`）

| # | 项目 | 官方 | 本实验 | 理由 |
|---|---|---|---|---|
| 1 | 主干 | 小型 3 层 CNN（256 维 flatten） | Official CNN（10752 维 flatten） | 任务要求主干与前序任务完全一致 |
| 2 | 优化器/学习率 | Adam, lr=1e-3 | RMSprop, lr=1e-5 | 项目冻结的官方 SCA 配方（Adam 会“记忆训练集”假收敛） |
| 3 | 微调 epoch | 15 | 75 | 与 Source Only / DANN / CDAN 的控制变量一致（同样多训 75 epoch） |
| 4 | 目标轨迹用法 | 缓存 200 条循环使用 | 45000 条 target_adapt 每 epoch 重采样 | 用满无标签适应集，避免“只适应 200 条”的偶然性 |
| 5 | BN 冻结 | `model.eval()` 冻结 BN | 无 BN 可冻（Official CNN 无 BN） | 结构决定；MMD 对齐承担全部域适应职责 |
| 6 | 预训练 | 100 epoch 从头训练 | B-E1 同 seed 75 epoch checkpoint | 复用已验证配方，减少变量 |

保留官方 `lambda_ = 0.1`、多核 RBF（5 核，mult 2.0）与“逐层 MMD”设计。
**为什么没有降级为 AdaBN**：Official CNN 没有 BatchNorm（E0/B-E1 已冻结结构），
AdaBN 需要 BN 统计量，在不改结构的前提下无法实现；改结构会破坏“主干与前序任务一致”的约束。
因此 AdaBN 不是“成本高”而是“协议上不可用”，本实验按首选路线实现 CDPA-MMD。

### 实现自检（smoke / 诊断，不计入主矩阵）

- 2 epoch、2000 条 source 的 smoke：MMD 数值稳定，3 层 MMD 均在 [0.2, 1.1]，无 NaN/Inf。
- λ_mmd=0 与 lr=0 诊断：确认 MMD 项可关闭、断点续训可复现；首轮 CE 的尖峰
  （3.44 → 20.4 → 数步内回落）来自 RMSprop 首步按 sign 更新乘以特征范数，
  A-E2 的 λ_adv=0 诊断同样出现（首 10 batch 均值 6.81，225 batch 首轮 3.87），
  属于配方本身的瞬态，不是 CDPA 的 bug（详见 8.1.11）。
- 梯度探针（epoch 1，`grad_probe.json`）：λ_mmd·MMD 的参数梯度范数 / CE 的参数梯度范数 = **0.0099**（约 1%）；
  三层 MMD 值 conv 0.932 / fc1 0.584 / fc2 0.230（越深越小，说明域差异主要在浅层卷积特征）。

## 8.1.5 训练配方与成本

| 项目 | 值 |
|---|---|
| 预训练 | B-E1 source-only checkpoint（RMSprop lr=1e-5，batch 200，75 epoch，同 seed） |
| 微调优化器 | RMSprop，lr=1e-5，alpha=0.99，eps=1e-8，weight_decay=0 |
| batch size | 200（source）+ 200（target） |
| epochs | 75 |
| 输入 | 原始 int8 幅度，不归一化 |
| λ_mmd | 0.1（官方值） |
| MMD 层 | conv（10752）、fc1（4096）、fc2（4096） |
| MMD 核 | RBF，kernel_num=5，kernel_mul=2.0，批内中位数带宽 |
| checkpoint | 每 5 epoch 覆盖写 outputs/models/a_e3/<run>/ckpt.pt（约 533 MB/run，含 optimizer） |
| GE probe | 每 2 epoch，500 条攻击轨迹，step=100（只记录，不早停、不选模型） |
| attack acc | 每 5 epoch 与训练结束 |

单 run 实测：75 epoch / 1726–1951 s（25–26 s/epoch）；6 run 合计 11239 s ≈ 3.12 GPU 小时。
推理与评估：每 run 10000 条攻击轨迹前向约 5 s；GE 评估（CPU）每 run 约 1 s。

## 8.1.6 主结果：GE、NTGE 与 SR

口径：NTGE 主口径为**首次 GE=0** 的攻击轨迹数（step=100），未达成的 run 截尾记 10000；
`stable` 口径要求首次 GE=0 之后一直保持 0，用于识别“短暂触底又弹回”。SR@10k 指
GE@10000=0 的 run 比例。所有数字来自 `results/metrics.csv`、`results/summary.json`，
逐 run 原始评估在 `results/eval/`（24 个 JSON）。

### desync50

| 方法 | NTGE（首次） | 达成 | 稳定达成 | GE@1k | GE@10k | SR@10k | attack acc | N_total@GE=1 |
|---|---|---|---|---|---|---|---|---|
| Source Only | 10000 ± 0 | 0/3 | 0/3 | 96.3 | 22.7 ± 11.4 | 0.00 | 0.0040 | 10000 |
| DANN | 9166.7 ± 1443.4 | 1/3 | 1/3 | 68.7 | 4.0 ± 5.3 | 0.33 | 0.0051 | 54166.7 |
| CDAN | 10000 ± 0 | 0/3 | 0/3 | 15.0 | 9.3 ± 6.4 | 0.00 | 0.0042 | 55000 |
| **CDPA-MMD** | **5700.0 ± 624.5** | **3/3** | **3/3** | 30.7 | **0.0 ± 0.0** | **1.00** | 0.0043 | 50700 |

逐 run（GE@10k，seed 42/43/44）：SO 32/10/26（均未到 0）；DANN 0/10/2（仅 s42 在 7500 条触底，稳定点同为 7500）；
CDAN 2/13/13（均未到 0）；CDPA **0/0/0**（首次 6400/5200/5500，稳定 8000/5500/5500）。

desync50 上 CDPA-MMD 是**唯一三个 seed 都稳定达到 GE=0** 的方法：NTGE 5700 ± 625，
相对 10k 评估上限节省 43% 攻击轨迹（相对 DANN 的 7500 节省 24%）；SR@10k = 100%。DANN 只有 s42 达成
（且直到 7500 条才触底），CDAN 的三个 seed 在 10k 内都未触底，尽管其 GE@1k（15.0）明显
低于 CDPA-MMD（30.7）——即 CDAN 早期排序更好，但尾部无法收敛到 0。

### desync100

| 方法 | NTGE（首次） | 达成 | 稳定达成 | GE@1k | GE@10k | SR@10k | attack acc | N_total@GE=1 |
|---|---|---|---|---|---|---|---|---|
| Source Only | 3700.0 ± 5461.7 | 2/3 | 0/3 | 3.0 | 23.0 ± 10.6 | 0.00 | 0.0047 | 3700 |
| DANN | 8300.0 ± 2944.5 | 1/3 | 0/3 | 19.0 | 11.0 ± 6.1 | 0.00 | 0.0042 | 53300 |
| CDAN | 6566.7 ± 5105.2 | 2/3 | 1/3 | 29.7 | 13.7 ± 13.1 | 0.33 | 0.0038 | 51566.7 |
| **CDPA-MMD** | **3700.0 ± 5461.7** | 2/3 | 1/3 | 12.7 | **4.0 ± 4.6** | 0.33 | 0.0040 | 48700 |

逐 run：SO 800/300/未达（GE@10k 15/19/35）；DANN 4900/未达/未达（7/8/18）；
CDAN 9000（稳定 9000）/700/未达（0/26/15）；CDPA **300（稳定 7700）/800（稳定未达）/未达**（0/9/3）。

desync100 上三种适应方法的 NTGE 都**没有超过** Source Only：CDPA 均值相同（3700），
配对差值 -500/+500/0；DANN/CDAN 反而更差（+4600/+2867，负迁移）。但 CDPA 的 GE@10k
最低（4.0 ± 4.6，SO 为 23.0 ± 10.6），且只有它把末态 GE 压在个位数——说明适应并非完全无效，
只是在这个偏移上不足以让排序优势稳定跨过 0 点。

### 一句话读法

desync50：**CDPA-MMD 明显更强**（3/3 稳定 GE=0，NTGE 5700）。
desync100：**CDPA-MMD 与 Source Only 打平（NTGE），但尾部 rank 更低、更稳**；
通用 UDA 的 DANN/CDAN 在该偏移上出现负迁移。

## 8.1.7 图

全部为 3.5 英寸单栏宽、300 dpi、PNG + PDF 矢量、衬线字体；Y 轴 0–256 全量程，
对数版将 rank=0 映射为 0.5 触底（与 E0/B-E1/A-E2 一致）。

| 图 | 文件 | 内容与观察 |
|---|---|---|
| desync50 线性 | `figures/a_e3_ge_desync50.{png,pdf}` | CDPA-MMD（红）约 5700 条触底并保持 0；DANN 在 7500 触底；CDAN 尾部约 10、SO 约 20 |
| desync50 对数 | `figures/a_e3_ge_desync50_log.{png,pdf}` | 同上，可分辨 1–10 量级的尾部差异；CDPA 触底后无回升 |
| desync100 线性 | `figures/a_e3_ge_desync100.{png,pdf}` | CDPA 均值最低（约 3–4）；SO 在 4000–8000 条区间回升到 40+；CDAN/DANN 中段 10–20 |
| desync100 对数 | `figures/a_e3_ge_desync100_log.{png,pdf}` | 同上；SO 的“先降后升”在 log 轴上更清楚（域偏移导致的排序震荡） |
| NTGE 柱状 | `figures/a_e3_ntge.{png,pdf}` | 四方法 × 两偏移，柱内标注稳定达成 seed 数（如 3/3、1/3）；误差棒为 3 seed 标准差，截尾 10000 |
| GE@10k 柱状 | `figures/a_e3_ge10k.{png,pdf}` | desync50：CDPA=0、DANN=4.0、CDAN=9.3、SO=22.7；desync100：CDPA=4.0 最低；虚线为随机猜测 127.5 |
| 训练动力学 | `figures/a_e3_cdpa_dynamics.{png,pdf}` | 四联图：CE（3.6→0.35）、MMD（d50 2.3→0.80；d100 3.4→1.23–1.44）、500 条 GE probe、源域 val acc |

MMD 曲线显示对齐损失在前 10 epoch 快速下降后进入平台；desync100 的残余 MMD（1.23–1.44）
明显高于 desync50（0.80–0.89），与“d100 适应更难”的结论一致。

## 8.1.8 汇总表：四方法 × 两偏移

完整表见 `results/table_methods.csv`；以下为论文用精简版（均值 ± 标准差，3 seeds）。

| 方法 | desync50 NTGE | desync50 GE@10k | desync100 NTGE | desync100 GE@10k |
|---|---|---|---|---|
| Source Only | 10000 ± 0（0/3） | 22.7 ± 11.4 | 3700 ± 5462（2/3） | 23.0 ± 10.6 |
| DANN | 9167 ± 1443（1/3） | 4.0 ± 5.3 | 8300 ± 2944（1/3） | 11.0 ± 6.1 |
| CDAN | 10000 ± 0（0/3） | 9.3 ± 6.4 | 6567 ± 5105（2/3） | 13.7 ± 13.1 |
| **CDPA-MMD** | **5700 ± 625（3/3）** | **0.0 ± 0.0** | **3700 ± 5462（2/3）** | **4.0 ± 4.6** |

配对比较（逐 seed 的 NTGE，未达成按 10000）：
- CDPA vs DANN：desync50 3 胜 0 负；desync100 2 胜 1 平。
- CDPA vs CDAN：desync50 3 胜 0 负；desync100 1 胜 1 平 1 负。
- CDPA vs Source Only：desync50 3 个 seed 全部更优（delta -3600/-4800/-4500）；
  desync100 delta -500/+500/0（净 0）。

## 8.1.9 置信度与校准诊断

沿用 A-E2 的口径：`mean_max_softmax`、`mean_entropy_nats`、15-bin ECE，
以及“confident wrong”比例 = 预测错误且 max softmax ≥ 0.9 的轨迹占比（3 seed 均值）。

| 偏移 | 方法 | attack acc | ECE | confident-wrong (≥0.9) | 平均熵 (nats) |
|---|---|---|---|---|---|
| desync50 | Source Only | 0.0040 | 0.556 | 0.193 | 1.62 |
| desync50 | DANN | 0.0051 | 0.822 | 0.539 | 0.50 |
| desync50 | CDAN | 0.0042 | 0.846 | 0.597 | 0.43 |
| desync50 | **CDPA-MMD** | 0.0043 | **0.718** | **0.333** | 0.85 |
| desync100 | Source Only | 0.0047 | 0.704 | 0.388 | 1.00 |
| desync100 | DANN | 0.0042 | 0.889 | 0.702 | 0.29 |
| desync100 | CDAN | 0.0038 | 0.944 | 0.857 | 0.15 |
| desync100 | **CDPA-MMD** | 0.0040 | **0.732** | **0.355** | 0.80 |

读法：attack acc 都在 0.4% 左右（256 类随机猜测 0.39%），但置信度分布差异极大。
DANN/CDAN 把绝大多数轨迹压到接近 one-hot（熵 0.15–0.50 nats），其中约 54%–86% 是
“高置信度错误”——这正好对应 GE 曲线尾部上翘（尤其是 SO/DANN/CDAN 在 desync100 中段
回升）以及“早期排序好、后期无法收敛到 0”的行为。CDPA-MMD 是三种适应方法里校准最好的
（ECE 0.72–0.73、confident-wrong 0.33–0.36），但仍明显差于同域 Source Only 的 0.19/0.39。
因此 CDPA 的优势至少有一部分来自“没有把错误密钥过度自信化”，而不是纯粹的特征可分性。

## 8.1.10 攻击成本

- 每次适应训练：75 epoch，1726–1951 s（均值 1873 s ≈ 31.2 min，25–26 s/epoch）；
  6 次合计 **11239 s = 3.12 GPU 小时**（RTX 5060 Laptop，含每 2 epoch 的 500 条 GE probe）。
- 适应轨迹成本：三种适应方法各使用 45000 条 target_adapt 无标签轨迹；Source Only 为 0。
- 攻击成本双口径（desync50 / desync100，均值）：
  - N_attack@GE=1 = NTGE：CDPA 5700 / 3700；DANN 9167 / 8300；CDAN 10000 / 6567；SO 10000 / 3700。
  - N_total@GE=1 = N_attack + 45000：CDPA 50700 / 48700；DANN 54167 / 53300；CDAN 55000 / 51567；SO 10000 / 3700。
- 评估成本：每 run 10000 条前向约 5 s（GPU）+ GE 计算约 1 s（CPU）；24 个评估合计 < 3 min。
- 结论：若只按攻击轨迹计，CDPA 在 desync50 上以 5700 条优于所有基线；若把 45000 条
  无标签适应轨迹计入总成本，则 CDPA 的总成本（≈5.1 万条）高于 Source Only 的 1 万条，
  只有在“适应数据可免费获得”的场景下才划算——这一点必须在论文中明写。

## 8.1.11 失败、限制与异常

| 编号 | 现象 | 处理与解释 |
|---|---|---|
| F1 | desync100 s44 在 10k 内未达 GE=0（末态 3，GE@1k 38） | 保留；d100 残余 MMD 1.44 最高、seed 间方差大（NTGE 300/800/未达），不剔除 |
| F2 | desync50 s42 首次 6400 触底，但 6400–8000 之间回升，稳定点 8000 | 主口径记 6400、稳健口径记 8000；两口径都报，不择优 |
| F3 | desync100 s42 在 300 条“侥幸”触底后回升到 7700 才稳定 | 同上；说明单点 GE=0 不等于攻击成功，必须看稳定口径 |
| F4 | DANN/CDAN 在 desync100 负迁移（NTGE 比 SO 差 +4600/+2867） | 保留并归因：对抗对齐在小 λ/无 BN 配方下压不过域差异，且过自信放大错误排序 |
| F5 | 官方 MMD 实现 `x.expand` 在 4096×200 下需 ~2.6 GB，smoke 时 OOM | 改为 `||x||²+||y||²-2xy` 的矩阵乘等价式，数值一致；写入 `run_meta.mmd_impl_note` |
| F6 | AdaBN 备选路线不可用 | Official CNN 无 BatchNorm，不改结构无法实现；理由已写入 8.1.4，未强行改结构 |
| F7 | `ops_bash` 在本机多次 `EXECUTOR_UNAVAILABLE`（niubash hardlink awk.exe） | 全部命令改走 run_code 内 node `child_process.spawnSync`；监控与收尾用 Windows 计划任务 |
| F8 | `analyze_confidence_a_e3.py` 初次运行路径错误（REPO=parents[3]）与 cdpa run 名生成错误 | 修为 parents[2] 与 `cdpa_<shift>_s<seed>`，重跑后 24 行完整；不删除错误版本痕迹，日志保留 |

其它限制：单数据集家族（ASCAD v1 fixed-key）、单源域、3 seeds；GE probe 仅 500 条
（不参与选模型）；MMD 带宽用批内统计，batch=200 时估计有噪声；Official CNN 无 BN，
无法实现 AdaBN / 无法用 BN 统计量做域对齐。

## 8.1.12 结论边界

回答两个问题：

1. **领域专用 vs 通用 UDA 谁更强？** 在 desync50 上 CDPA-MMD 明确更强：唯一 3/3 稳定 GE=0，
   NTGE 5700 ± 625，GE@10k=0，SR=100%；DANN 仅 1/3、CDAN 0/3、SO 0/3。在 desync100 上
   CDPA 的 NTGE 与 SO 打平（3700），但 GE@10k 最低（4.0），且没有出现 DANN/CDAN 的负迁移。
   因此“领域专用更强”只在中等偏移上成立，强偏移上优势退化为“更稳的尾部”。
2. **为 CDAN-SCA 树立的门槛**：desync50 必须超过 **NTGE ≤ 5700 且 3/3 稳定 GE=0**；
   desync100 必须超过 **GE@10k ≤ 4.0、NTGE ≤ 3700（至少 2/3 达成）**，同时不得出现
   DANN/CDAN 式负迁移（NTGE 不得高于 SO）。校准门槛：ECE ≤ 0.73、confident-wrong ≤ 0.36
   （CDPA 的水平），否则即使 GE 略好也可能只是“更自信的错误”。

边界：结论绑定于本协议（Official CNN、RMSprop 1e-5、75 epoch、45000 条适应轨迹、
ASCAD v1 fixed-key → desync50/100）；CDPA 的成本口径依赖“适应轨迹免费”的假设；
3 seeds 的 NTGE 方差在 desync100 上很大（±5462），均值排序需谨慎。

## 8.1.13 下一步实验

1. **CDAN-SCA 主方法对照**：把 CDPA 的多层 MMD（conv+fc1+fc2）作为正则项并入 CDAN-SCA
   （条件对抗 + 熵加权 + 多层 MMD），在完全相同的 6 个 run 上对比 8.1.12 的门槛。
2. **λ_mmd 单变量扫描**：desync100 上取 λ ∈ {0.03, 0.1, 0.3}（每档 2 seeds），检验
   “残余 MMD 1.2–1.4 是否因对齐强度不足”，其余配方不变。
3. **desync100 的适应长度/学习率调度**：在 75 epoch 基础上加 cosine 或后 25 epoch 降 lr，
   检验 F1/F3 的尾部震荡是否来自优化末期步长过大。
4. **校准后处理**：对 DANN/CDAN/CDPA 统一做 temperature scaling（用 target_adapt 拟合，
   不用 attack），看 ECE/confident-wrong 改善是否能转化为 GE 尾部改善。
5. **跨数据集验证**：在 CDPA 官方 ASCAD_desync100 或 ASCAD variable-key 上复跑
   CDPA-MMD 与 Source Only，确认“中等偏移强、强偏移打平”的结论是否可迁移。
