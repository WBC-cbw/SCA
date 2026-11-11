#!/usr/bin/env python3
"""A-E3：CDPA-MMD 在 OfficialCNN 上的移植组件（PyTorch）。

依据官方实现
  sca_repo/baselines/cdpa/CDPA/Cross-Device-Profiled-Attack-main/
  Different_Implementations/ASCAD_addDesync100_Demo.ipynb
的事实：
  - CDP_Net.forward(source, target) 在 features 展平表示、classifier_1 输出、
    classifier_2 输出三处各算一次多核 RBF MMD，三项直接相加（official cell 4）；
  - mmd_rbf：guassian_kernel 的带宽 = 批内平均平方距离，再除以
    kernel_mul**(kernel_num//2)，取 5 个核 [bw/4, bw/2, bw, 2bw, 4bw]，
    loss = mean(K_ss + K_tt - K_st - K_ts)（official cell 2）；
  - 微调损失 = CE(source) + lambda_ * mmd，official lambda_ = 0.1（cell 3/5）。

移植映射（OfficialCNN 没有 BatchNorm）：
  official features(256-d)  -> conv 展平表示（10752-d）
  official classifier_1(20) -> fc1 后 ReLU（4096-d）
  official classifier_2(20) -> fc2 后 ReLU（4096-d）
  official classifier_3     -> 不参与 MMD（与官方一致：第三层之后不算）
距离矩阵用 ||x-y||^2 = ||x||^2 + ||y||^2 - 2x·y 计算。官方展开写法在 4096 维、
batch=200 时会生成 400x400x4096 的张量（约 2.6 GB/项），数学上完全等价但会 OOM，
因此这里换等价实现并在 run_meta 里记录。
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[4]          # .../sca_repo
A_E2_SCRIPTS = ROOT / "experiments/01_member_a_baselines/a_e2_uda_baselines/scripts"
B_E1_MODELS = ROOT / "outputs/models/b_e1"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


UDA = _load("uda_common_a_e2", A_E2_SCRIPTS / "uda_common.py")
OfficialCNN = UDA.OfficialCNN
load_domain_data = UDA.load_domain_data
seed_everything = UDA.seed_everything
count_params = UDA.count_params
save_json = UDA.save_json

MMD_LAYERS_DEFAULT = ("conv", "fc1", "fc2")


class CDPANet(nn.Module):
    """OfficialCNN + 三处表示输出，供 CDPA 的 MMD 使用。"""

    def __init__(self, num_classes=256):
        super().__init__()
        cnn = OfficialCNN(num_classes=num_classes)
        self.convs = cnn.convs
        self.fc1 = cnn.fc1
        self.fc2 = cnn.fc2
        self.fc3 = cnn.fc3
        self.relu = nn.ReLU()
        self.num_classes = num_classes

    def representations(self, x):
        h = self.convs(x).flatten(1)
        h1 = self.relu(self.fc1(h))
        h2 = self.relu(self.fc2(h1))
        return {"conv": h, "fc1": h1, "fc2": h2}

    def logits_from_reps(self, reps):
        return self.fc3(reps["fc2"])

    def forward(self, x):
        return self.logits_from_reps(self.representations(x))

    def forward_pair(self, xs, xt, layers=MMD_LAYERS_DEFAULT):
        """返回 (source logits, {layer: mmd})；target 不产生分类损失。"""
        rs = self.representations(xs)
        rt = self.representations(xt)
        terms = {k: mmd_rbf(rs[k], rt[k]) for k in layers}
        return self.logits_from_reps(rs), terms


def _sq_dists(source, target):
    n = source.size(0)
    m = target.size(0)
    total = torch.cat([source, target], dim=0)
    sq = total.pow(2).sum(dim=1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (total @ total.t())
    return d2.clamp_min_(0.0), n, m


def mmd_rbf(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """与官方 guassian_kernel + mmd_rbf 数值等价的实现（见模块 docstring）。"""
    d2, n, m = _sq_dists(source, target)
    n_samples = n + m
    if fix_sigma is not None:
        bandwidth = float(fix_sigma)
    else:
        denom = max(n_samples * n_samples - n_samples, 1)
        bandwidth = float(d2.detach().sum().item() / denom)
        if not np.isfinite(bandwidth) or bandwidth <= 0:
            bandwidth = 1.0
    bandwidth /= kernel_mul ** (kernel_num // 2)
    kernels = None
    for i in range(kernel_num):
        bw = bandwidth * (kernel_mul ** i)
        k = torch.exp(-d2 / bw)
        kernels = k if kernels is None else kernels + k
    XX = kernels[:n, :n]
    YY = kernels[n:, n:]
    XY = kernels[:n, n:]
    YX = kernels[n:, :n]
    return torch.mean(XX + YY - XY - YX)


def load_b_e1_init(model, ckpt_path):
    sd = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"B-E1 权重映射失败: missing={sorted(missing)} "
                           f"unexpected={sorted(unexpected)}")
    return len(sd)


@torch.no_grad()
def predict_probs(model, x, device="cuda", batch=1000):
    model.eval()
    out = []
    for s in range(0, len(x), batch):
        xb = torch.from_numpy(x[s:s + batch]).to(device)
        out.append(torch.softmax(model(xb), dim=1).cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def accuracy(model, x, y, device="cuda", batch=1000):
    model.eval()
    correct = 0
    for s in range(0, len(x), batch):
        xb = torch.from_numpy(x[s:s + batch]).to(device)
        correct += int((model(xb).argmax(1).cpu().numpy() == y[s:s + batch]).sum())
    return correct / max(len(x), 1)


def grad_norm_ratio(model, xb, yb, tb, lam, layers, ce):
    """诊断：lambda*MMD 与 CE 对参数梯度的范数比（单 batch，不改参数）。"""
    params = [p for p in model.parameters() if p.requires_grad]
    model.zero_grad(set_to_none=True)
    logits, terms = model.forward_pair(xb, tb, layers)
    mmd = sum(terms.values())
    cls = ce(logits, yb)
    g_mmd = torch.autograd.grad(lam * mmd, params, retain_graph=True, allow_unused=True)
    g_cls = torch.autograd.grad(cls, params, allow_unused=True)
    n_mmd = float(torch.sqrt(sum((g.float() ** 2).sum() for g in g_mmd if g is not None)))
    n_cls = float(torch.sqrt(sum((g.float() ** 2).sum() for g in g_cls if g is not None)))
    model.zero_grad(set_to_none=True)
    return {"mmd_grad_norm": n_mmd, "cls_grad_norm": n_cls,
            "ratio": (n_mmd / n_cls) if n_cls > 0 else None,
            "mmd_terms": {k: float(v.detach()) for k, v in terms.items()}}
