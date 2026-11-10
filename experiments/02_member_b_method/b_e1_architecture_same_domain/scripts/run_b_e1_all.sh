#!/usr/bin/env bash
# B-E1: 9 次训练（3 结构 x 3 seed），统一配方，串行执行避免 GPU 争用影响耗时统计。
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO="$(cd "$EXP_DIR/../../.." && pwd)"
WS="$(cd "$REPO/.." && pwd)"
PY="${PYTHON:-$WS/venv_exp/Scripts/python.exe}"
STATUS="$EXP_DIR/logs/run_all_status.log"

mkdir -p "$EXP_DIR/results" "$EXP_DIR/logs"

run_one() {
  local arch="$1" seed="$2"
  local run="${arch}_s${seed}"
  local out="$EXP_DIR/results/$run"
  if [ -f "$out/predictions_attack.npy" ]; then
    echo "$(date -Is) SKIP $run (already done)" | tee -a "$STATUS"
    return 0
  fi
  echo "$(date -Is) START $run" | tee -a "$STATUS"
  "$PY" "$SCRIPT_DIR/train_b_e1.py" --arch "$arch" --seed "$seed" \
      --out-root "$EXP_DIR/results" >> "$EXP_DIR/logs/$run.log" 2>&1
  local rc=$?
  echo "$(date -Is) END $run rc=$rc" | tee -a "$STATUS"
  return 0
}

for ARCH in mlp resnet1d cnn; do
  for SEED in 42 43 44; do
    run_one "$ARCH" "$SEED"
  done
done

echo "$(date -Is) ALL DONE" | tee -a "$STATUS"
