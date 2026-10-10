CONFIG = {
    # 数据（严格对齐官方 ASCAD：全量 50k profiling、原始 int8 输入不归一化）
    "source_file": "D:/sca_project/ASCAD_data/ASCAD.h5",
    "profiling_limit": 50000,
    "val_size": 5000,              # 末 5000 条作 source_val，45k 训练
    "attack_limit": 10000,
    "target_byte": 2,
    "normalize": "none",           # 官方做法：原始幅度直接输入
    # 模型与训练（官方配方：RMSprop lr=1e-5，75 epochs，batch 200）
    "num_classes": 256,
    "epochs": 75,
    "batch_size": 200,
    "learning_rate": 0.00001,
    "optimizer": "rmsprop",
    "seed": 42,
    # 运行控制
    "epochs_per_run": 2,
    "early_stop_ge_probe": True,
    "ge_probe_traces": 500,
    "early_stop_patience": 3,
    "run_dir": "experiments/00_common_e0",
}
