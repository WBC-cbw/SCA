### λ=0 归因对照（seed 42）

| 偏移 | 条件 | 训练 epoch | GE@1k | GE@10k | NTGE(first) | NTGE(stable) | acc | conf | H |
|---|---|---|---|---|---|---|---|---|---|
| desync50 | Source Only | 75 | 109 | 32 | — | — | 0.39% | 0.524 | 1.78 |
| desync50 | DANN | 150 | 30 | 0 | 7500 | 7500 | 0.50% | 0.815 | 0.53 |
| desync50 | DANN λ=0（无对抗） | 150 | 39 | 34 | — | — | 0.52% | 0.850 | 0.41 |
| desync100 | Source Only | 75 | 0 | 15 | 800 | — | 0.41% | 0.679 | 1.12 |
| desync100 | DANN | 150 | 2 | 7 | 4900 | — | 0.47% | 0.883 | 0.32 |
| desync100 | DANN λ=0（无对抗） | 150 | 1 | 2 | 400 | — | 0.38% | 0.908 | 0.25 |

- desync50: ΔGE@10k (λ=0 − Source Only) = +2; (DANN − Source Only) = -32; (DANN − λ=0) = -34
- desync100: ΔGE@10k (λ=0 − Source Only) = -13; (DANN − Source Only) = -8; (DANN − λ=0) = +5