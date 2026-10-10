CONFIG = {
    # 数据
    "source_file": "D:/sca_project/ASCAD_data/ASCAD.h5",
    "profiling_limit": 30000,      # 取 Profiling_traces 前 30000 条
    "val_size": 5000,              # 其中末 5000 条作 source_val
    "attack_limit": 10000,         # Attack_traces 全量用于最终 GE
    "target_byte": 2,              # ASCAD 标准攻击字节（第 3 字节）
    "normalize": "feature_standardize",  # 用训练集每个时间点的均值/方差做标准化
                                         # （保留轨迹间幅度差异；per-trace z-score 会抹掉泄露）
    # 模型与训练
    "num_classes": 256,
    "epochs": 60,
    "batch_size": 200,
    # 注意：仓库草稿配置为 1e-5，实测收敛过慢；单批过拟合测试表明 lr=1e-3
    # 可正常学习（120 步内 200 样本 acc=99%）。E0 为流程验证实验，采用 1e-3；
    # 正式基线（E1 起）的超参数将在 E1 结构筛选时统一标定。
    "learning_rate": 0.0001,
    "optimizer": "adam",
    "seed": 42,
    # 运行控制（分段续训）
    "epochs_per_run": 2,           # 每次进程运行 2 个 epoch 后保存断点
    "early_stop_ge_probe": True,   # 每段结束后用 500 条攻击轨迹探 GE
    "ge_probe_traces": 500,
    "early_stop_patience": 3,      # GE 探针连续 3 次为 0 则提前收敛
    # 输出
    "run_dir": "experiments/00_common_e0",
}
