# B-E1 结构筛选（同域 ASCAD fixed-key）实验配置
# 只改变 backbone 一个变量，其余全部与 E0 official 配置一致。

BASE = {
    "dataset": "D:/sca_project/ASCAD_data/ASCAD.h5",
    "target_byte": 2,
    "profiling_limit": 50000,     # Profiling_traces[0:50000]
    "train_size": 45000,          # [0:45000]  -> source_train
    "val_size": 5000,             # [45000:50000] -> source_val
    "attack_size": 10000,         # Attack_traces[0:10000] -> target_attack
    "target_adapt": "none (same-domain Source Only)",
    "normalize": "none",          # 原始 int8 幅度输入，不归一化
    "num_classes": 256,
    "optimizer": "rmsprop",
    "learning_rate": 1e-5,
    "batch_size": 200,
    "epochs": 75,
    "seed_list": [42, 43, 44],
    "ge_step": 100,
    "ge_probe_traces": 500,
    "label": "Sbox[pt[2] XOR key[2]]",
}

ARCHS = {
    "mlp": {
        "layers": "Flatten(700) -> Dense(512,relu) -> Dense(512,relu) -> Dense(256)",
        "n_params": 752896,
    },
    "cnn": {
        "layers": ("Conv1D 64/128/256/512/512 (k=11) + AvgPool2 x5 -> Flatten(21*512) "
                   "-> Dense(4096,relu) -> Dense(4096,relu) -> Dense(256)"),
        "n_params": 66652544,
        "source": "experiments/00_common_e0/scripts/train_e0_torch.py::OfficialCNN (直接复用)",
    },
    "resnet1d": {
        "layers": ("3 x ResidualBlock(2 x Conv1D k=11), widths 64/128/256, "
                   "stride 1/2/2, GlobalAvgPool -> Dense(256)"),
        "n_params": 1505600,
    },
}

RUN_MATRIX = [(a, s) for a in ("mlp", "resnet1d", "cnn") for s in BASE["seed_list"]]
