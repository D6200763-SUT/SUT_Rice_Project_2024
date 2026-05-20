#!/bin/bash

set -eo pipefail

# ค้นหา directory ของ script นี้ (ใช้งานได้ไม่ว่าจะรันจาก directory ไหน)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="/home/ai-station/my_project/.tf251p310/bin/python"

# ==============================
# รับค่าจากผู้ใช้ตอนรัน
# ==============================
NPZ="$1"
ROOT_OUT="$2"

# ==============================
# ตรวจสอบว่าระบุ input ครบหรือไม่
# ==============================
if [ -z "$NPZ" ] || [ -z "$ROOT_OUT" ]; then
  echo "Usage:"
  echo "  ./scripts/run_train_auto.sh <npz_file> <output_dir>"
  echo ""
  echo "Example:"
  echo "  ./scripts/run_train_auto.sh results/out_feature_sets_w60_h7/core/sequences_window60_h7.npz results/out_train_w60_h7"
  exit 1
fi

# ==============================
# ตรวจสอบว่าไฟล์ NPZ มีจริงหรือไม่
# ==============================
if [ ! -f "$NPZ" ]; then
  echo "ERROR: NPZ file not found:"
  echo "$NPZ"
  exit 1
fi

mkdir -p "$ROOT_OUT/logs"

echo "======================================"
echo "Start Training"
echo "NPZ      : $NPZ"
echo "OUT ROOT : $ROOT_OUT"
echo "======================================"

echo ""
echo "======================================"
echo "1) Train LSTM"
echo "======================================"

$PYTHON "$SCRIPT_DIR/train_lstm.py" \
  --npz "$NPZ" \
  --out_dir "$ROOT_OUT/lstm_core" \
  --epochs 120 --batch_size 128 --lr 0.0005 --patience 20 \
  --clipnorm 1.0 --lstm_units 64 --dropout 0.2 \
  2>&1 | tee "$ROOT_OUT/logs/01_lstm_core.log"

echo ""
echo "======================================"
echo "2) Train CNN-LSTM"
echo "======================================"

$PYTHON "$SCRIPT_DIR/train_cnn_lstm.py" \
  --npz "$NPZ" \
  --out_dir "$ROOT_OUT/cnn_lstm_core" \
  --epochs 150 --batch_size 128 --lr 0.0005 --patience 25 \
  --clipnorm 1.0 --conv_filters 64 --kernel_size 5 \
  --lstm_units 64 --dropout 0.25 \
  2>&1 | tee "$ROOT_OUT/logs/02_cnn_lstm_core.log"

echo ""
echo "======================================"
echo "3) Train Transformer"
echo "======================================"

$PYTHON "$SCRIPT_DIR/train_transformer.py" \
  --npz "$NPZ" \
  --out_dir "$ROOT_OUT/transformer_core" \
  --epochs 200 --batch_size 128 --lr 0.0005 --patience 30 \
  --clipnorm 1.0 --d_model 64 --num_layers 2 --num_heads 4 \
  --ff_dim 128 --dropout 0.2 \
  2>&1 | tee "$ROOT_OUT/logs/03_transformer_core.log"

echo ""
echo "======================================"
echo "4) Summarize Runs"
echo "======================================"

$PYTHON "$SCRIPT_DIR/summarize_runs.py" \
  --root "$ROOT_OUT" \
  --out "$ROOT_OUT/summary" \
  2>&1 | tee "$ROOT_OUT/logs/04_summary.log"

echo ""
echo "======================================"
echo "Completed"
echo "NPZ      : $NPZ"
echo "OUT ROOT : $ROOT_OUT"
echo "Summary  : $ROOT_OUT/summary"
echo "Logs     : $ROOT_OUT/logs"
echo "======================================"