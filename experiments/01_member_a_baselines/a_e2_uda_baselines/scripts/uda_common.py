#!/usr/bin/env python3
"""A-E2 统一 UDA trainer 的共享组件（PyTorch）。

包含：
  - 数据加载：source_train / source_val = fixed Profiling 前 45000 / 末 5000；
    target_adapt = desync Profiling 前 45000（无标签）；target_domain_val = 末 5000
    （仅用于域判别器诊断，不参与训练）；target_attack = desync Attack 10000。
  - 主干：OfficialCNN 的 penultimate 表示（convs + fc1 + fc2，4096 维）作为
    feature_extractor，fc3 作为 256 类 classifier。初始化直接来自 B-E1 的
    Source Only 权重（每个 seed 一份），因此与 Source Only 基线共享起点。
  - 对抗模块：GRL、DomainDiscriminator、RandomMultilinearMap（CDAN 的条件表示）。
  - 特征送入判别器前统一做固定 L2 缩放：f_unit * sqrt(dim)。E0 官方 CNN 没有
    BN，源域特征范数约 413、目标域更小；不缩放时判别器仅凭范数即可 100% 分域，
    对抗信号退化为尺度博弈。该缩放对 DANN/CDAN 一致，记录在配置里。
"""
from __future__ import annotations

import importlib.util
import json
import math
import random
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[4]          # .../sca_repo
E0_TRAIN = ROOT / "experiments/00_common_e0/scripts/train_e0_torch.py"
B_E1_MODELS = ROOT / "outputs/models/b_e1"


def load_e0_module():
    spec = importlib.util.spec_from_file_location("e0_train", str(E0_TRAIN))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


E0 = load_e0_module()
OfficialCNN = E0.OfficialCNN


# ---------------------------------------------------------------- 数据
def load_domain_data(source_h5, target_h5, n_source_train=45000, n_val=5000,
                     n_target_adapt=45000, target_byte=2):
    """读取四个划分；target_adapt 用前 n_target_adapt 条无标签 profiling。"""
    with h5py.File(source_h5, "r") as f:
        n_prof = n_source_train + n_val
        xp = f["Profiling_traces/traces"][:n_prof].astype(np.float32)
        yp = f["Profiling_traces/labels"][:n_prof].astype(np.int64)
        key_src = np.asarray(f["Profiling_traces/metadata"]["key"])
    with h5py.File(target_h5, "r") as f:
        xt = f["Profiling_traces/traces"][:].astype(np.float32)
        xa = f["Attack_traces/traces"][:].astype(np.float32)
        meta_a = np.asarray(f["Attack_traces/metadata"])
        key_tgt = np.asarray(f["Profiling_traces/metadata"]["key"])
    data = {
        "source_train_x": xp[:n_source_train, None, :],
        "source_train_y": yp[:n_source_train],
        "source_val_x": xp[n_source_train:, None, :],
        "source_val_y": yp[n_source_train:],
        "target_adapt_x": xt[:n_target_adapt, None, :],
        "target_domain_val_x": xt[n_target_adapt:, None, :],
        "attack_x": xa[:, None, :],
        "attack_plaintext": np.asarray(meta_a["plaintext"]),
        "attack_key": np.asarray(meta_a["key"]),
        "attack_true_label": None,
        "keys_equal": bool(np.array_equal(key_src, key_tgt)),
        "target_byte": int(target_byte),
    }
    # 目标域攻击标签只用于辅助诊断（不参与训练）：Sbox[pt[byte] ^ key[byte]]
    sbox = E0.AES_Sbox
    key_byte = int(data["attack_key"][0, target_byte])
    data["attack_true_label"] = sbox[np.bitwise_xor(
        data["attack_plaintext"][:, target_byte], key_byte)].astype(np.int64)
    data["real_key_byte"] = key_byte
    return data


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------- 模型
class _GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_adv):
        ctx.lambda_adv = float(lambda_adv)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_adv * grad_output, None


def grad_reverse(x, lambda_adv=1.0):
    return _GRL.apply(x, lambda_adv)


class CNNFeatureExtractor(nn.Module):
    """OfficialCNN 的 convs + fc1 + fc2，输出 4096 维 penultimate 表示。"""

    def __init__(self, cnn: OfficialCNN):
        super().__init__()
        self.convs = cnn.convs
        self.fc1 = cnn.fc1
        self.fc2 = cnn.fc2
        self.relu = nn.ReLU()

    def forward(self, x):
        h = self.convs(x).flatten(1)
        h = self.relu(self.fc1(h))
        return self.relu(self.fc2(h))


class CNNClassifier(nn.Module):
    """OfficialCNN 的 fc3：4096 -> 256。"""

    def __init__(self, cnn: OfficialCNN):
        super().__init__()
        self.fc3 = cnn.fc3

    def forward(self, f):
        return self.fc3(f)


class DomainDiscriminator(nn.Module):
    def __init__(self, in_dim, hidden=1024, mid=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, mid), nn.ReLU(),
            nn.Linear(mid, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


class RandomMultilinearMap(nn.Module):
    """CDAN 的随机多线性映射：T(f, g) = (Rf^T f_unit * sqrt(d)) ⊙ (Rg^T g)。

    Rf / Rg 用固定种子初始化后冻结（不参与训练），维度写进配置。
    """

    def __init__(self, feat_dim, num_classes, proj_dim, seed):
        super().__init__()
        g = torch.Generator().manual_seed(int(seed))
        rf = torch.randn(feat_dim, proj_dim, generator=g) / math.sqrt(feat_dim)
        rg = torch.randn(num_classes, proj_dim, generator=g)
        self.register_buffer("rf", rf)
        self.register_buffer("rg", rg)
        self.feat_dim = feat_dim
        self.proj_dim = proj_dim
        self.scale = math.sqrt(feat_dim)

    def forward(self, f, g):
        f_unit = F.normalize(f, dim=1)
        u = (f_unit @ self.rf) * self.scale
        v = g @ self.rg
        return u * v


class UDANet(nn.Module):
    """feature_extractor + classifier + (DANN/CDAN) domain discriminator。"""

    def __init__(self, method, num_classes=256, proj_dim=1024, map_seed=1234):
        super().__init__()
        cnn = OfficialCNN(num_classes=num_classes)
        self.feature_extractor = CNNFeatureExtractor(cnn)
        self.classifier = CNNClassifier(cnn)
        self.method = method
        self.feat_dim = 4096
        if method == "dann":
            self.discriminator = DomainDiscriminator(self.feat_dim)
            self.cdan_map = None
        elif method == "cdan":
            self.cdan_map = RandomMultilinearMap(self.feat_dim, num_classes, proj_dim, map_seed)
            self.discriminator = DomainDiscriminator(proj_dim)
        else:
            raise ValueError(f"unknown method {method}")
        self.proj_dim = proj_dim

    def features(self, x):
        return self.feature_extractor(x)

    def logits_from_features(self, f):
        return self.classifier(f)

    def forward(self, x):
        f = self.features(x)
        return self.logits_from_features(f), f

    def domain_logits(self, f, logits, lambda_adv):
        if self.method == "dann":
            z = F.normalize(f, dim=1) * math.sqrt(self.feat_dim)
        else:
            g = torch.softmax(logits, dim=1)
            z = self.cdan_map(f, g)
        return self.discriminator(grad_reverse(z, lambda_adv))


def load_b_e1_init(model: UDANet, ckpt_path):
    """把 B-E1 的 OfficialCNN state_dict 映射到 UDANet 的 feature_extractor/classifier。"""
    sd = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    mapped = {}
    for k, v in sd.items():
        if k.startswith("convs.") or k.startswith("fc1.") or k.startswith("fc2."):
            mapped["feature_extractor." + k] = v
        elif k.startswith("fc3."):
            mapped["classifier." + k] = v
    missing, unexpected = model.load_state_dict(mapped, strict=False)
    expected_missing = {n for n in missing if not n.startswith("discriminator.")
                        and (model.cdan_map is None or not n.startswith("cdan_map."))}
    if expected_missing or unexpected:
        raise RuntimeError(f"B-E1 权重映射失败: missing={sorted(expected_missing)} "
                           f"unexpected={sorted(unexpected)}")
    return len(mapped)


def count_params(model):
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


# ---------------------------------------------------------------- 评估辅助
@torch.no_grad()
def predict_probs(model, x, device="cuda", batch=2000):
    model.eval()
    out = []
    for s in range(0, len(x), batch):
        xb = torch.from_numpy(x[s:s + batch]).to(device)
        logits, _ = model(xb)
        out.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def accuracy(model, x, y, device="cuda", batch=1000):
    model.eval()
    correct, total = 0, 0
    for s in range(0, len(x), batch):
        xb = torch.from_numpy(x[s:s + batch]).to(device)
        logits, _ = model(xb)
        correct += int((logits.argmax(1).cpu().numpy() == y[s:s + batch]).sum())
        total += len(xb)
    return correct / max(total, 1)


@torch.no_grad()
def domain_accuracy(model, x_src, x_tgt, lambda_adv, device="cuda", batch=500):
    """域判别器在来源已知的保留集上的平衡准确率（源=0，目标=1）。"""
    model.eval()

    def score(x, want_target):
        correct, total = 0, 0
        for s in range(0, len(x), batch):
            xb = torch.from_numpy(x[s:s + batch]).to(device)
            logits, f = model(xb)
            d = model.domain_logits(f, logits, lambda_adv)
            pred = (torch.sigmoid(d) > 0.5)
            correct += int((pred == want_target).sum().item())
            total += len(xb)
        return correct / max(total, 1)

    a_src = score(x_src, False)
    a_tgt = score(x_tgt, True)
    return {"src_acc": a_src, "tgt_acc": a_tgt, "balanced_acc": 0.5 * (a_src + a_tgt)}


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
