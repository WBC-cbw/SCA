#!/usr/bin/env python3
"""B-E2 数据校验：目标字节、标签公式与 desync 偏移语义。

校验项（对应任务要求第 2 条“desync 数据集标签与目标字节需重新校验”）：
 1) 三个数据集 Attack_traces 的 plaintext / key 是否完全一致（同批明文、同一真实密钥）
 2) Profiling_traces/labels（数据集自带）是否 == Sbox[plaintext[2] ^ key[2]]，全量 50000 条
 3) Attack_traces 的标签按同公式现算，检查类别覆盖（256 类应全部出现）
 4) metadata['desync'] 分布（逐轨迹偏移量：fixed 全 0 / desync50 0..49 / desync100 0..99）
 5) 抽样攻击轨迹的归一化互相关峰值滞后与 metadata['desync'] 的对应关系

输出: results/verification/dataset_check.json
"""
import json
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parents[1]
DATA = Path("D:/sca_project/ASCAD_data")
FILES = {
    "fixed": DATA / "ASCAD.h5",
    "desync50": DATA / "ASCAD_desync50.h5",
    "desync100": DATA / "ASCAD_desync100.h5",
}
BYTE = 2

SBOX = np.array([
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16,
], dtype=np.uint8)


def peak_lag(x, y, maxlag):
    """返回使归一化互相关最大的滞后 k：c[k] = <y[t], x[t+k]> / (||y|| ||x||)。"""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x -= x.mean()
    y -= y.mean()
    best_k, best_c = 0, -2.0
    n = len(x)
    for k in range(-maxlag, maxlag + 1):
        if k >= 0:
            a, b = 0, n - k
            seg_x = x[k:k + b]
            seg_y = y[:b]
        else:
            a, b = -k, n
            seg_x = x[:b + k]
            seg_y = y[a:]
        denom = np.linalg.norm(seg_y) * np.linalg.norm(seg_x)
        if denom == 0:
            continue
        c = float(np.dot(seg_y, seg_x) / denom)
        if c > best_c:
            best_c, best_k = c, k
    return best_k, best_c


def main():
    out_dir = HERE / "results/verification"
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {"target_byte": BYTE, "sbox": "AES Sbox, 256 entries",
              "files": {}, "checks": {}}
    plaintexts, keys, desync_values = {}, {}, {}

    for name, path in FILES.items():
        with h5py.File(path, "r") as h:
            ptr = np.asarray(h["Profiling_traces/metadata"])
            atr = np.asarray(h["Attack_traces/metadata"])
            labels_prof = np.asarray(h["Profiling_traces/labels"]).astype(np.int64)
            pt_p = np.asarray(ptr["plaintext"])
            key_p = np.asarray(ptr["key"])
            pt_a = np.asarray(atr["plaintext"])
            key_a = np.asarray(atr["key"])
            d_p = np.asarray(ptr["desync"]).reshape(-1)
            d_a = np.asarray(atr["desync"]).reshape(-1)

        key_byte = int(key_p[0, BYTE])
        lab_formula = SBOX[np.bitwise_xor(pt_p[:, BYTE], np.uint8(key_byte))].astype(np.int64)
        lab_ok = bool(np.array_equal(lab_formula, labels_prof))

        # Attack 标签现算（数据集不带 label 字段）
        lab_attack = SBOX[np.bitwise_xor(pt_a[:, BYTE], np.uint8(int(key_a[0, BYTE])))]
        coverage = int(len(np.unique(lab_attack)))

        report["files"][name] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "attack_traces": int(len(pt_a)),
            "profiling_traces": int(len(pt_p)),
            "key_row0": [int(v) for v in key_a[0][:4]],
            "key_byte_2": int(key_a[0, BYTE]),
            "profiling_labels_equal_sbox_formula": lab_ok,
            "profiling_label_mismatches": int((lab_formula != labels_prof).sum()),
            "attack_label_classes_present": coverage,
            "attack_label_counts_min": int(np.bincount(lab_attack, minlength=256).min()),
            "attack_label_counts_max": int(np.bincount(lab_attack, minlength=256).max()),
            "desync_metadata": {
                "profiling_min": int(d_p.min()), "profiling_max": int(d_p.max()),
                "profiling_unique": int(len(np.unique(d_p))),
                "attack_min": int(d_a.min()), "attack_max": int(d_a.max()),
                "attack_unique": int(len(np.unique(d_a))),
                "attack_mean": round(float(d_a.mean()), 3),
            },
        }
        plaintexts[name], keys[name], desync_values[name] = pt_a, key_a, d_a

    report["checks"]["attack_plaintext_identical_across_datasets"] = bool(
        np.array_equal(plaintexts["fixed"], plaintexts["desync50"]) and
        np.array_equal(plaintexts["fixed"], plaintexts["desync100"]))
    report["checks"]["attack_key_identical_across_datasets"] = bool(
        np.array_equal(keys["fixed"], keys["desync50"]) and
        np.array_equal(keys["fixed"], keys["desync100"]))
    report["checks"]["all_profiling_labels_match_sbox_formula"] = all(
        v["profiling_labels_equal_sbox_formula"] for v in report["files"].values())

    # 5) 互相关验证：抽样 30 条固定轨迹，与 desync 轨迹比较峰值滞后
    rng = np.random.default_rng(42)
    with h5py.File(FILES["fixed"], "r") as h:
        X_fixed = np.asarray(h["Attack_traces/traces"][:], dtype=np.float32)
    idx = rng.choice(len(X_fixed), size=30, replace=False)
    report["lags"] = {}
    for name in ["desync50", "desync100"]:
        with h5py.File(FILES[name], "r") as h:
            X = np.asarray(h["Attack_traces/traces"][:], dtype=np.float32)
        maxlag = int(desync_values[name].max()) + 5
        est, d_true, cc = [], [], []
        for i in idx:
            k, c = peak_lag(X_fixed[i], X[i], maxlag)
            est.append(k)
            d_true.append(int(desync_values[name][i]))
            cc.append(c)
        est = np.array(est)
        d_true = np.array(d_true)
        err_plus = np.abs(est - d_true)
        err_minus = np.abs(est + d_true)
        sign = 1 if err_plus.mean() <= err_minus.mean() else -1
        report["lags"][name] = {
            "n_sampled": int(len(idx)),
            "peak_lag_estimate": [int(v) for v in est],
            "metadata_desync": [int(v) for v in d_true],
            "mean_abs_err_same_sign": round(float(err_plus.mean()), 3),
            "mean_abs_err_opposite_sign": round(float(err_minus.mean()), 3),
            "best_sign": sign,
            "mean_abs_err_best_sign": round(float((est - sign * d_true).mean()), 3) if False else round(float(np.abs(est - sign * d_true).mean()), 3),
            "frac_within_2_samples": round(float((np.abs(est - sign * d_true) <= 2).mean()), 3),
            "mean_peak_corr": round(float(np.mean(cc)), 4),
        }

    (out_dir / "dataset_check.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
