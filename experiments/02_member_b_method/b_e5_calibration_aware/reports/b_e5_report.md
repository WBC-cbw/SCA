# B-E5 报告：校准感知 CDAN-SCA（温度校准权重 vs 裸熵权重）

- 实验编号：B-E5
- 实验级别：formal（6 次 calib 训练；对照 CDPA-MMD / full / 消融 1 / CDAN 全部复用）
- 归档目录：experiments/02_member_b_method/b_e5_calibration_aware/
- 主干：Official CNN（B-E1 同 seed 权重初始化）
- 本报告由 README.md 的 8.1.1-8.1.13 自动生成，内容与 README 一致。

一句话：把 CDAN-SCA 中未校准的裸熵权重 w=1−H(softmax(g))/log256 换成「源域验证集温度校准后的
置信度权重」 w=1−H(softmax(g/T))/log256，在 ASCAD fixed-key → desync50/desync100 上做 6 次正式
训练（2 偏移 × seeds 42/43/44），检验它能否把主方法推过最强基线 CDPA-MMD 的门槛，并用校准/权重/
坍塌三组诊断回答「熵权重为什么失效、校准是否救得回来」。

#### 8.1.1 实验级别、问题与矩阵

- **级别**：formal（6 次正式训练，单变量：权重函数；其余与 B-E3 full 完全一致）。
- **要回答的问题**：
  1. 温度校准能否让熵权重「重新生效」——目标域 w 去饱和、权重出现分化，而不是像 full 那样
     熵坍缩到 0.13 nats、w 饱和到 0.98、90 分位=1.000？
  2. 校准能否减少预测坍塌（top-1 单类占比）与 confident-wrong（ECE）？
  3. 若 GE/NTGE 有改善，能否归因于权重机制本身（唯一变量就是权重函数）？
- **实验矩阵**：CDAN-SCA-calib × {desync50, desync100} × seeds {42, 43, 44} = 6 次训练。
  对照全部复用，不重跑：Source Only（B-E2）、DANN/CDAN（A-E2）、CDPA-MMD（A-E3）、
  CDAN-SCA full 与消融 1（B-E3/B-E4）。
- **预注册的失败模式**（训练前写定）：温度若把源域预测压到近均匀，则目标域校准熵也会接近
  log256，w=1−H/log256 ≈ 0，目标域对抗项几乎关闭，calib 退化为「从同一 B-E1 初始化继续做
  75 epoch 源域训练」。此时无论 GE 好坏都不能算「校准权重生效」；GE 若变好只能归因于多训，
  需用 A-E2 的 λ=0 对照（150 epoch 无对抗）作旁证。

#### 8.1.2 假设与证伪条件

| 编号 | 假设 | 证伪/判据 |
|---|---|---|
| H1 | 温度校准使目标域权重去饱和：校准后 w 的均值落在 (0.05, 0.95)、p90 < 0.999 | 若 w 仍 >0.99 或反而 ≈0，H1 不成立，分别记为「温度未能跨域迁移」/「过度校准」 |
| H2 | 校准减少坍塌与过自信：top-1 占比 <30%、ECE 与 confident-wrong 低于 full | 与 full 的逐 seed 对比，若更高则 H2 不成立 |
| H3 | 若 H1+H2 成立，GE/NTGE 相对 full 改善，并可能打过 CDPA-MMD | d50：NTGE ≤ 5700 且 3/3 seed 稳定 GE=0；d100：GE@10k ≤ 4.0 且 NTGE ≤ 3700；NTGE 打平但 N_total 更高判「未胜出」 |
| H4 | 收益可归因于权重函数（唯一变量） | 若 w≈0 使对抗项关闭，则收益归因于多训而非权重机制，H4 不成立 |

- **允许的补充 pilot**：仅当温度法「仍饱和」（H1 的第二种失败）时，可补 w>0.5 阈值变体，
  只做 d50_s42 与 d100_s42 两次，标 pilot 不进主结果。若失败模式是「过度校准」（w≈0），
  阈值法会几乎选空集，不做 pilot，按纪律记录为方法学结论。
- **稳健性**：达到 GE=0 的 run 做 10 次 attack 轨迹顺序置换，重算 NTGE（首次 GE=0，step=100）。
- **双成本口径**：N_attack@GE=1 = NTGE；N_total@GE=1 = NTGE + 45000 条 target_adapt
  （Source Only 无适应，单列）。

#### 8.1.3 数据划分与统一协议

数据划分与 B-E3/A-E2/A-E3 完全一致（byte=2，标签 = Sbox[plaintext[2] XOR key[2]]）：

| 划分 | 来源 | 条数 | 用途 |
|---|---|---|---|
| source_train | fixed ASCAD.h5 Profiling_traces [0:45000] | 45000 | 有监督分类 + 域判别源侧 |
| source_val | fixed ASCAD.h5 Profiling_traces [45000:50000] | 5000 | 温度 T 拟合；每 epoch val acc 监控 |
| target_adapt | desync50/100 Profiling_traces [0:45000] | 45000 | 无标签域适应（对抗 + 权重） |
| target_domain_val | desync50/100 Profiling_traces [45000:50000] | 5000 | 域判别诊断（不参与训练） |
| target_attack | desync50/100 Attack_traces | 10000 | 只做 GE/NTGE/SR 评估，不参与训练 |

统一配方（与 B-E3 full 逐项相同）：

| 项 | 值 |
|---|---|
| 主干 | Official CNN（B-E1 同 seed checkpoint 初始化，conv 64/128/256/512/512 k11 + AvgPool×5 → Dense4096 → Dense4096 → 256） |
| 优化器 / lr / batch | RMSprop / 1e-5 / 200 |
| epochs | 75（不早停，按固定预算） |
| 输入 | 原始 int8 幅度，不归一化 |
| 域对抗 | GRL + 条件表示 T(f,g)，proj_dim=1024，λ_adv=0.1，源/目标 BCE 各半 |
| 监控 | 每 epoch target_adapt 前 2000 条：top-1 占比、预测熵、权重分位；每 2 epoch 500 条 GE probe |
| 参数量 | 68,227,457（与 full 相同，权重函数不引入参数） |
| 随机性 | seed 42/43/44；同 seed 的初始化与 full 完全相同 |

**与 full 的唯一方法差异**：目标域分类损失的样本权重由裸熵改为温度校准熵（见 8.1.4）。
条件表示 T(f,g) 内部仍用未校准 softmax(g)，λ、投影维度、划分、预算、监控、评估全部不变。

#### 8.1.4 方法定义与训练前温度预检

**温度拟合**（每个 run 训练前一次，B-E1 同 seed 初始化权重上）：

```
T* = argmin_T  mean_{(x,y) in source_val} [ -log softmax(g(x)/T)_y ]
```

对数网格 T ∈ [0.05, 20]，400 点粗搜 + 3 轮加密（scripts/calib_common.py::fit_temperature）。
拟合只用 source_val 的 5000 条带标签轨迹；target_adapt/target_attack 不参与。

**目标域权重**（每个 batch，detach）：

```
g_T = softmax(g_t / T*)
w   = 1 - H(g_T) / log(256)          # H 为 nats，w ∈ [0,1]
L_adv_tgt = mean_t [ w * BCE(d_t, 1) ]
```

**训练前预检事实**（results/temperature_probe.json，训练前已记录，不是事后拟合）：

| seed | source_val acc | T* | NLL before → after | ECE before → after |
|---|---|---|---|---|
| 42 | 0.0058 | 10.864 | 7.525 → 5.514 | 0.178 → 0.0010 |
| 43 | 0.0064 | 11.601 | 7.715 → 5.515 | 0.190 → 0.0004 |
| 44 | 0.0054 | 14.452 | 8.527 → 5.517 | 0.248 → 0.0012 |

解读（预检即已明确）：source_val 上模型接近随机（1/256=0.0039），NLL 最优温度必然把 logits
压到近均匀（NLL→ln256=5.545）。因此校准后的目标域熵会接近 log256，w≈0。smoke（d50_s42，2 epoch）
实测：w=0.021/0.023，w_bare=0.684/0.731，H_cal=5.43/5.42 → 目标域对抗项权重约 2%，实际几乎关闭。
这属于 8.1.1 预注册的「过度校准」失败模式，不是「仍饱和」，故按 8.1.2 不启用阈值 pilot。

**实现映射**：

- scripts/calib_common.py：fit_temperature / apply_temperature / weight_np /
  monitor_stats_calib（在 B-E3 监控字段上增加 T、校准熵、w 与 w_bare）。
- scripts/train_cdan_sca_calib.py：由 B-E3 train_cdan_sca.py 改造（同一 UDANet/GRL/条件映射），
  仅替换 batch 权重；训练前写 temperature.json（断点续训读回，不重拟合）；产物
  predictions_attack.npy、diag_target_adapt.npz（含 probs_calibrated、w、w_bare）、
  entropy_diag.json、history.json、run_meta.json。
- scripts/evaluate_b_e5.py：七方法统一评估（复用 evaluate_ascad.py::ranks）。
- scripts/aggregate_b_e5.py：3 seed 均值±std、双口径、calib vs full/CDPA 配对与硬判据 verdict。
- scripts/analyze_calibration_b_e5.py：attack/adapt 两侧 bare vs calibrated 的 ECE /
  confident-wrong / 熵 / w 分位；scripts/permutation_b_e5.py：GE=0 run 的置换检验。

#### 8.1.5 计算成本

| 项 | desync50（s42/s43/s44） | desync100（s42/s43/s44） |
|---|---|---|
| 单 run 训练时间 (s) | 1861 / 1879 / 1853 | 1928 / 1973 / 1832 |
| 单 run 平均 (s) | 1864.2 | 1911.2 |
| 单 run 收敛预算 | 75 epoch，约 24.8 s/epoch | 75 epoch，约 25.5 s/epoch |
| 合计 | 5593 s | 5733 s |

- 6 次正式训练合计 **11,326 s = 3.15 GPU 小时**（RTX 5060 Laptop，CUDA 12.8，torch 2.9.1）。
- 参数量 68,227,457，与 B-E3 full 完全一致（权重函数不引入参数）。
- 温度拟合额外开销可忽略：每个 run 在 5000×256 的 logits 上做 400 点网格 + 3 轮加密，
  单次 <2 s，已计入训练时间。
- 与 B-E3 的 12 次 full+marginal（22,632 s = 6.29 h）相比，本实验只跑 6 次，成本减半。

#### 8.1.6 主结果

七方法在相同划分、相同主干、相同 75 epoch 预算下的 3 seed 均值±标准差（NTGE 截尾 10000；
括号内为达到 GE=0 / 稳定 GE=0 的 seed 数）。完整逐 run 数值见 results/summary.json、
results/metrics.csv、results/table_methods.csv。

**desync50**

| 方法 | NTGE first | 稳定 NTGE | GE@1k | GE@10k | SR@10k | attack acc | N_total |
|---|---|---|---|---|---|---|---|
| Source Only | 10000 ± 0 (0/3) | 10000 (0/3) | 96.3 | 22.7 ± 11.4 | 0 | 0.0040 | 10000 |
| DANN | 9167 ± 1443 (1/3) | 9167 (1/3) | 68.7 | 4.0 ± 5.3 | 0.333 | 0.0051 | 54167 |
| CDAN | 10000 ± 0 (0/3) | 10000 (0/3) | 15.0 | 9.3 ± 6.4 | 0 | 0.0042 | 55000 |
| CDPA-MMD | **5700 ± 625 (3/3)** | **6333 (3/3)** | 30.7 | **0.0 ± 0.0** | 1.000 | 0.0043 | 50700 |
| CDAN-SCA full | 6000 ± 4716 (2/3) | 7333 (1/3) | 44.3 | 7.7 ± 10.8 | 0.333 | 0.0042 | 51000 |
| Abl.1 marginal+w | 7533 ± 4272 (1/3) | 10000 (0/3) | 25.7 | 4.7 ± 2.5 | 0 | 0.0044 | 52533 |
| **CDAN-SCA-calib** | 8467 ± 2656 (1/3) | 8600 (1/3) | 24.3 | 1.7 ± 1.5 | 0.333 | 0.0047 | 53467 |

**desync100**

| 方法 | NTGE first | 稳定 NTGE | GE@1k | GE@10k | SR@10k | attack acc | N_total |
|---|---|---|---|---|---|---|---|
| Source Only | 3700 ± 5462 (2/3) | 10000 (0/3) | 3.0 | 23.0 ± 10.6 | 0 | 0.0047 | 3700 |
| DANN | 8300 ± 2944 (1/3) | 10000 (0/3) | 19.0 | 11.0 ± 6.1 | 0 | 0.0042 | 53300 |
| CDAN | 6567 ± 5105 (2/3) | 9667 (1/3) | 29.7 | 13.7 ± 13.1 | 0.333 | 0.0038 | 51567 |
| CDPA-MMD | 3700 ± 5462 (2/3) | 9233 (1/3) | 12.7 | 4.0 ± 4.6 | 0.333 | 0.0040 | 48700 |
| CDAN-SCA full | 6000 ± 4583 (2/3) | 10000 (0/3) | 51.3 | 8.7 ± 5.5 | 0 | 0.0044 | 51000 |
| Abl.1 marginal+w | 4600 ± 4804 (2/3) | 10000 (0/3) | 2.0 | 18.0 ± 26.1 | 0 | 0.0053 | 49600 |
| **CDAN-SCA-calib** | **3133 ± 3402 (3/3)** | 9467 ± 462 (2/3) | 12.0 | **0.3 ± 0.6** | 0.667 | 0.0046 | 48133 |

**calib 逐 run 细节**（温度 T、首次/稳定 NTGE、GE@10k、坍塌与权重）：

| run | T | NTGE first | 稳定 NTGE | GE@1k | GE@10k | acc | max top1 | 坍塌 epoch | w 均值 | 训练 s |
|---|---|---|---|---|---|---|---|---|---|---|
| d50 s42 | 10.86 | — | — | 20 | 3 | 0.0047 | 0.658 | 20 | 0.0228 | 1861 |
| d50 s43 | 11.60 | 5400 | 5800 | 1 | 0 | 0.0043 | 0.698 | 21 | 0.0188 | 1879 |
| d50 s44 | 14.45 | — | — | 52 | 2 | 0.0050 | 0.690 | 30 | 0.0096 | 1853 |
| d100 s42 | 10.86 | 600 | — | 0 | 1 | 0.0046 | 0.826 | 57 | 0.0186 | 1928 |
| d100 s43 | 11.60 | 7000 | 9200 | 23 | 0 | 0.0047 | 0.812 | 62 | 0.0158 | 1973 |
| d100 s44 | 14.45 | 1800 | 9200 | 13 | 0 | 0.0046 | 0.806 | 51 | 0.0070 | 1832 |

两点必须同时看：

1. **d50 上 calib 明显弱于 CDPA-MMD 与 full**：NTGE 8467 对 5700/6000，只有 1/3 稳定归零；
   GE@10k 1.7 虽低于 full 的 7.7，但 CDPA 是 0.0；SR@10k 只有 0.333 对 CDPA 的 1.000。
2. **d100 上 calib 的首次 NTGE 均值 3133 < 3700、GE@10k 0.33 < 4.0**，按预注册判据形式达标；
   但 3 个 seed 的 GE=0 全部是「先归零后回升」型（s42 在 600 归零、10k 时回到 1；s43/s44 到
   9200 才稳定），稳定 NTGE 9467 反而略高于 CDPA 的 9233（见 8.1.11）。

#### 8.1.7 训练期监控与坍塌

监控每 epoch 取 target_adapt 前 2000 条（不参与训练），图见
[figures/b_e5_monitor_desync50.png](../figures/b_e5_monitor_desync50.png) 与
[figures/b_e5_monitor_desync100.png](../figures/b_e5_monitor_desync100.png)。

| 指标 | d50 full | d50 calib | d100 full | d100 calib |
|---|---|---|---|---|
| top-1 单类占比最大值 | 0.610 | **0.682** | 0.791 | **0.815** |
| 触发 COLLAPSE(>0.5) 的 epoch 数 | 5.7 | **23.7** | 35.3 | **56.7** |
| 目标域预测熵末值 (nats) | 0.36 | 0.24 | 0.14 | 0.13 |
| 校准后权重 w 均值 | —（w=0.94） | 0.017 | —（w=0.98） | 0.014 |
| w p90 | 1.000 | 0.033 | 1.000 | 0.028 |

- **校准没有减轻坍塌，反而更糟**：d50 的 top-1 占比从 epoch ~20 起在 0.3–0.6 间振荡，
  d100 从 epoch ~10 起爬到 0.7–0.8；COLLAPSE 持续的 epoch 数是 full 的 4 倍（d50）与 1.6 倍（d100）。
- 原因不是权重把对抗项「调小」就应更稳：权重缩小确实让对抗项接近关闭（下节），但目标域
  分类项本身仍在把源域特征强行套到目标域上，熵照样坍缩到 0.13 nats；坍塌与对抗项强弱无关，
  这一点与 B-E5 之前 B-E3 的诊断一致。
- 校准后的熵（H_cal≈5.45）只是把同一组 logits 除以 T 后的数学结果，不代表模型对目标域更有知识；
  它的「稳定」是构造出来的（见 8.1.8）。

#### 8.1.8 校准与权重诊断（三问）

**ECE / confident-wrong 对比**（attack 侧，3 seed 均值；bare = 未校准概率，calibrated = 用 T 缩放后）：

| 方法 | d50 ECE | d50 cw≥0.9 | d50 熵 | d100 ECE | d100 cw≥0.9 | d100 熵 |
|---|---|---|---|---|---|---|
| Source Only | 0.556 | 0.193 | 1.62 | 0.704 | 0.388 | 1.00 |
| DANN | 0.822 | 0.538 | 0.50 | 0.889 | 0.702 | 0.29 |
| CDAN | 0.846 | 0.597 | 0.43 | 0.944 | 0.857 | 0.15 |
| CDPA-MMD | 0.718 | 0.333 | 0.85 | 0.732 | 0.355 | 0.80 |
| CDAN-SCA full | 0.867 | 0.655 | 0.36 | 0.945 | 0.861 | 0.14 |
| Abl.1 marginal+w | 0.802 | 0.491 | 0.56 | 0.881 | 0.686 | 0.31 |
| calib（bare） | **0.913** | **0.782** | 0.24 | **0.950** | **0.877** | 0.13 |
| calib（calibrated） | 0.029 | 0.000 | 5.45 | 0.031 | 0.000 | 5.47 |

**Q1：温度是否让权重重新生效（去饱和、分化）？——否，且方向相反。**

- 拟合出的 T=10.86/11.60/14.45，把目标域预测压到近均匀：校准熵 5.45/5.47 nats，
  为 log256=5.545 的 98.3%/98.6%；平均最大概率只剩 0.033/0.035。
- 权重 w=1−H/log256 因此塌到 **0.017（d50）/0.014（d100）**，p90 仅 0.033/0.028，
  **100% 的样本 w<0.5**（history 的 w_frac_lt_0.5=1.0）；而 bare 权重是 0.957/0.977，
  与 full 的 0.94/0.98 同型。也就是说：校准把「接近饱和的权重」变成了「接近零的权重」，
  对抗项有效强度 λ·w≈0.1×0.017=0.0017，只有名义值的 1.7%，实际等于关闭。
- 权重分布图：[figures/b_e5_weights_adapt.png](../figures/b_e5_weights_adapt.png)、
  [figures/b_e5_weights_attack.png](../figures/b_e5_weights_attack.png)。attack 侧的 calibrated
  权重由保存的预测概率按各 run 的 T 重算（p^(1/T) 归一化），与 analyze 脚本给出的
  0.017/0.014 一致；adapt 侧直接来自 diag_target_adapt.npz 的 w。

**Q2：是否减少坍塌与 confident-wrong？——坍塌没有，confident-wrong 的「降低」是退化的。**

- 坍塌见 8.1.7：更严重，不是更轻。
- bare 概率的 ECE/confident-wrong 是七方法里最差的（0.913/0.782 与 0.950/0.877）。
- calibrated 概率的 ECE 0.029/0.031、cw=0 看似完美，但平均最大概率只有 0.033/0.035，
  准确率仍是 0.0047/0.0046（随机 1/256=0.0039）——这是「把概率压平」换来的校准，
  没有任何判别信息增量。按 8.1.2 的 H2，这不能算校准成功。
- 温度迁移诊断（results/temperature_transfer_diagnostic.json）说明根源：同一模型在
  source_train 5k 子集上 acc=0.2908、T*=0.824（ECE 0.075→0.020），在 source_val 上
  acc=0.0056、T*=10.863（ECE 0.178→0.001）；把 T_val 用到 source_train 会把 NLL 从
  3.146 推到 5.099。**T 是「这份 split 接近随机」的属性，不是模型的属性**；用 source_val
  拟合必然得到「压平」的解，校准感知权重从一开始就被预检判了死刑（8.1.1 预注册的
  「过度校准」失败模式）。

**Q3：收益能否归因于权重机制？——不能。**

- 与 full 相比，calib 唯一改动是权重函数；但该函数把有效对抗权重从 ~0.94 压到 ~0.017，
  所以「calib vs full」实际比较的是「几乎无对抗」vs「强对抗」。
- d100 的改善（GE@10k 0.33 vs full 8.67）与 A-E2 的 λ=0 对照（150 epoch、无对抗，
  d100 GE@10k=2）方向一致；d50 则没有改善（8467 vs full 6000，反而更差）。
  因此 d100 的收益应归因于「关闭对抗对齐」，而不是「温度校准让熵权重更聪明」。
- 这也解释了为什么 d50 输、d100 赢：在 d100 上 DANN/CDAN 的对抗本身就是负迁移
  （DANN 8300、CDAN 6567、full 6000 都远差于 SO 的 3700），关掉它反而回到 SO 附近；
  在 d50 上 CDPA 的分布对齐是有效的（5700/3-3），关掉对抗没有好处。

#### 8.1.9 论文标准图

全部 300 dpi PNG + PDF 矢量、3.5 英寸单栏宽、衬线字体；GE 图 Y 轴 0–256 全量程、
rank=0 映射 0.5 触底、线性与对数双版：

- GE 曲线：[b_e5_ge_desync50.png](../figures/b_e5_ge_desync50.png) /
  [b_e5_ge_desync50_log.png](../figures/b_e5_ge_desync50_log.png)；
  [b_e5_ge_desync100.png](../figures/b_e5_ge_desync100.png) /
  [b_e5_ge_desync100_log.png](../figures/b_e5_ge_desync100_log.png)。
- 七方法总览版（含 DANN/CDAN/Abl.1）：[b_e5_ge_desync50_all.png](../figures/b_e5_ge_desync50_all.png) /
  [b_e5_ge_desync50_all_log.png](../figures/b_e5_ge_desync50_all_log.png)；
  [b_e5_ge_desync100_all.png](../figures/b_e5_ge_desync100_all.png) /
  [b_e5_ge_desync100_all_log.png](../figures/b_e5_ge_desync100_all_log.png)。
- NTGE 柱图（柱内标 k/3，虚线 10k 截尾）：[b_e5_ntge.png](../figures/b_e5_ntge.png)。
- 权重分布：[b_e5_weights_adapt.png](../figures/b_e5_weights_adapt.png)、
  [b_e5_weights_attack.png](../figures/b_e5_weights_attack.png)。
- 训练监控：[b_e5_monitor_desync50.png](../figures/b_e5_monitor_desync50.png)、
  [b_e5_monitor_desync100.png](../figures/b_e5_monitor_desync100.png)。
- 校准对比：[b_e5_calibration_attack.png](../figures/b_e5_calibration_attack.png)、
  [b_e5_calibration_adapt.png](../figures/b_e5_calibration_adapt.png)。

目检记录：d50 的 calib 曲线在 0–3000 轨迹明显高于 CDPA/full、5000 后才贴近 0；
d100 的 calib 在 1500–3000 与 7000–9000 两次触底、10k 时均值 0.3，是全方法最低；
权重图 d50/d100 的 calib 分布分别集中在 0.02/0.01，full 集中在 0.93/0.98。

#### 8.1.10 失败与异常

| 编号 | 事件 | 影响 | 处理 |
|---|---|---|---|
| F1 | d50 s43 在 epoch 50、d100 s44（首轮）在 epoch 41 被 "forrtl: error (200): program aborting due to window-CLOSE event" 杀死 | 该 run 中断，无结果文件 | 从 ckpt 续训（s43 51–75、s44 41–75），日志确认 resume |
| F2 | Daimon runtime 更新瞬间，4 个 run 在 0.03 s 内失败："No Python at '\"D:\\KimiData\\daimon-bundle\\runtime\\python\\cpython-3.12\\python.exe'\"" | 后续 run 全部未启动 | 更新完成后重排队列；未产生假结果 |
| F3 | run_meta.json 在训练开始时就写（无 epochs_done），用「文件存在」判完成导致 s43/s44 被跳过、d100 s42 重复启动 | 重复启动的 d100 s42 只跑了 epoch 1–5 即被杀 | 改用 findstr epochs_done 判完成；d100 s42 从 epoch 5 续训，浪费 5 epoch |
| F4 | run_queue.bat 只把 QUEUE_DONE 写进 queue.log，不创建 logs/QUEUE_DONE.txt | 第一版 watcher 永不触发，finalize 未自动跑 | 新建 watch3_and_finalize.vbs 轮询 QUEUE3_DONE.txt，并用隐藏 wscript+pythonw 规避控制台事件 |
| F5 | 首版 plot_b_e5.py 的 attack 权重面板把 calib 的 bare w 当作校准权重（图例 mean 0.96），monitor 的 w_bare 虚线因键名回退重复画了 w | 图会误导「校准后仍饱和」 | 已在定稿前修正：attack 侧按 T 重算 calibrated w，monitor 虚线改读 w_bare_mean；结果数值不变 |
| F6 | d100 s42 的 SR@1k=1 但 SR@10k=0（600 处 GE=0，10k 时回到 1） | 单看 SR@1k 会高估 | 报告首次/稳定双口径；置换检验给出 7370±3214 的区间 |
| F7 | 预注册的「过度校准」失败模式命中（w≈0.017），不是「仍饱和」 | 按 8.1.2 不启用 w>0.5 阈值 pilot | 无 pilot；按纪律记为方法学结论 |

另：本机 ops_bash 自 2026-11-09 起不可用（niubash 硬链接上限），全部命令经 run_code 内
node child_process 执行；训练用 schtasks + pythonw 托管，属环境限制，不影响数值。

#### 8.1.11 与最强基线 CDPA-MMD 的胜负判定

**d50（判据：NTGE ≤ 5700 且 3/3 稳定 GE=0）——未胜出。**

- calib：NTGE 8467 ± 2656（1/3 触发、1/3 稳定），GE@10k 1.7 ± 1.5，SR@10k 0.333；
- CDPA-MMD：NTGE 5700 ± 625（3/3、3/3），GE@10k 0.0，SR@10k 1.000；
- 配对（calib − CDPA）：+3600 / +200 / +4500，均值 +2767，**0 胜 0 平 3 负**；
- N_total 53467 对 50700，成本也更高。

**d100（判据：GE@10k ≤ 4.0 且 NTGE ≤ 3700）——形式达标，但不能宣称稳定胜出。**

- 形式：NTGE 3133 ± 3402 < 3700，GE@10k 0.33 ± 0.58 ≤ 4.0，聚合脚本按预注册判据给
  verdict="WIN"；SR@10k 0.667 也高于 CDPA 的 0.333；N_total 48133 < 48700。
- 但稳健性三条都不支持「胜出」：
  1. **稳定口径更差**：稳定 NTGE 9467 ± 462（2/3）对 CDPA 9233（1/3）——calib 的 GE=0
     到 9200 才稳定，CDPA 的 1/3 在 7700；按稳定口径 calib 不优于 CDPA。
  2. **配对 1 胜 2 负**：+300 / +6200 / −8200，均值 −567；s43 的 7000 对 CDPA 800 是大幅落后，
     均值优势全靠 s44 的 1800 对 CDPA 的 10000（CDPA 该 seed 未收敛）。
  3. **s42 的 600 是顺序脆弱的**：10 次轨迹置换后 7370 ± 3214（700–10000，10 次里 4 次
     直接 10000）；s43 7000→4210 ± 3094、s44 1800→3410 ± 2682。单一 seed 的 600 不能当
     稳定结论。
- 结论：d100 只能记为「按预注册均值判据压线通过、稳健性检验后与 CDPA 打平（甚至略差）」，
  不能写成「calib 打赢了 CDPA-MMD」。真正的信息是：在 d100 上**关掉对抗对齐**（calib 实际做的）
  比保留对抗好，这与 CDPA 的 MMD 对齐在 d100 上也只有 1/3 稳定互相印证。

#### 8.1.12 结论与边界

**结论：校准感知 CDAN-SCA 没有把主方法推过 CDPA-MMD，不建议确立为正式 Proposed Method。**

1. 温度校准在 source_val 上退化为「压平」：T≈12、w≈0.017、有效对抗强度只剩名义值的 1.7%。
   预注册的 H1（权重去饱和到 (0.05,0.95)）不成立，命中「过度校准」失败模式。
2. 校准没有减少坍塌：COLLAPSE epoch 数 23.7/56.7，是 full 的 4.0/1.6 倍；top-1 占比更高。
   H2 不成立。
3. ECE/confident-wrong 的「改善」是把输出压到近均匀（mean max prob 0.033）换来的，
   准确率不变，属于退化校准，不能作为校准有效的证据。
4. d50 明确输给 CDPA-MMD（3 负、NTGE 差 2767）；d100 仅按均值判据压线通过，
   稳定口径与配对结果都不支持胜出。H3 不成立。
5. 唯一可复用的发现是「d100 上对抗对齐有害」：calib 的 d100 改善与 A-E2 λ=0 对照同向，
   说明收益来自关闭对抗而非权重机制（H4 不成立）。

**边界**：ASCAD v1 fixed-key 单数据集、desync50/100 两个偏移、3 seeds、75 epoch 固定预算、
单字节（byte=2）、Official CNN 单主干、无 BN、无早停；温度只在 source_val 上拟合；
「稳定 NTGE」定义为 GE 归零后不再回升的首次轨迹数。结论不外推到其它数据集/主干/攻击目标。

#### 8.1.13 下一步实验

1. **λ 扫描（最高优先级）**：固定 CDAN-SCA full，λ_adv ∈ {0, 0.01, 0.03, 0.1} × d100 × 3 seeds。
   若 λ=0.01 能复现 calib 的 GE@10k≈0 且稳定 3/3，则真正的问题是「对齐强度过大」，
   应改写论文主张为「弱对齐/对齐强度自适应」，而不是「校准感知熵权重」。
2. **温度对照**：在 source_train 的 5k 子集上拟合 T（acc 0.29、T≈0.82）作为 control，
   证明 T 由 split 的判别力决定；该对照只作诊断，不作为方法。
3. **权重下限/阈值**：若仍想保留熵权重，用 w'=max(w, w0)（如 w0=0.3）或按运行熵归一，
   避免 w 塌到 0；需与 λ 扫描联合，否则无法区分「权重」与「有效 λ」。
4. **加梯度占比探针**：每个 run 记录 adversarial 项与分类项的梯度范数比（B-E5 未采，
   grad_probe 为 null），用有效 λ=λ·mean(w) 解释结果，替代仅凭 w 的推断。
5. **无 BN 的校准替代**：TTABN/CORAL 或输入级归一化，避免温度在无 BN 网络上退化为压平。
6. **指标纪律**：后续正式表以稳定 NTGE 为主、首次 NTGE 为辅，并报告 10 次置换区间；
   d100 这类「先归零后回升」的曲线必须同时给 SR@1k 与 SR@10k。
