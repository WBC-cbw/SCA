# 00_common_e0：公共 E0 流程验证实验

**目的**：验证数据 → 训练 → 攻击 → GE/key rank → 曲线归档全流程闭环，并确立 ASCAD fixed-key 上的 Source Only 基线（NTGE=800 条）。

**结论**：✅ 通过。详见 `reports/e0_report.md`。

## 目录

```text
configs/    e0_source_only_fixed.py（初版，Adam 配方，已弃用）
            e0_source_only_fixed_official.py（最终官方配方，以此为准）
scripts/    train_e0_torch.py   训练（PyTorch/GPU，断点续训）
            train_e0.py         TF 版训练脚本（CPU，已弃用，仅留档）
            eval_legacy_model.py 五月遗留权重评估（判定不可用）
            finalize_e0.py      图表与 manifest 生成
reports/    e0_report.md        实验报告（8.1–8.3）
results/    predictions_attack.npy, e0_model_torch.pt, history.json, eval/
figures/    e0_ge_curve.png, e0_training_dynamics.png
manifests/  manifest.json（数据划分、环境、产物哈希）
```

## 复现

```bash
python experiments/00_common_e0/scripts/train_e0_torch.py \
    --config experiments/00_common_e0/configs/e0_source_only_fixed_official.py
python experiments/cdan-sca-diagnosis/scripts/evaluate_ascad.py \
    --predictions experiments/00_common_e0/results/predictions_attack.npy \
    --dataset <ASCAD.h5 路径> --out-dir experiments/00_common_e0/results/eval
python experiments/00_common_e0/scripts/finalize_e0.py
```
