#!/bin/bash
# run_walk_forward.sh
# Walk-Forward CV pipeline: build folds → Stage 1 (classifier) → Stage 2 (regressor)
#
# Usage (from BPH_Research/):
#   chmod +x scripts/run_walk_forward.sh
#   nohup ./scripts/run_walk_forward.sh > walk_forward_main.log 2>&1 &
#   tail -f walk_forward_main.log

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="/home/ai-station/my_project/.tf251p310/bin/python"
ROOT="results/walk_forward"
CLEANED="results/out_quality_gate/cleaned_raw.csv"

# -----------------------------------------------------------------------
# Guard: must be run from BPH_Research/
# -----------------------------------------------------------------------
if [ ! -f "$CLEANED" ]; then
  echo "[ERROR] $CLEANED not found. Run from BPH_Research/ directory."
  exit 1
fi

echo "======================================================="
echo " BPH Walk-Forward CV Pipeline"
echo " $(date)"
echo "======================================================="

# -----------------------------------------------------------------------
# Step 1: Build all folds
# -----------------------------------------------------------------------
echo ""
echo "=== STEP 1: BUILD WALK-FORWARD FOLDS ==="
$PYTHON "$SCRIPT_DIR/build_walk_forward_folds.py" \
  --input_csv "$CLEANED" \
  --out_dir   "$ROOT" \
  --folds 2,3,4 \
  --min_train_years 2 \
  --window 30 --horizon 7 --roll_days 7 \
  --feature_set context \
  --spike_threshold 100

echo "[OK] Fold NPZs built."

# -----------------------------------------------------------------------
# Step 2 & 3: Train Stage 1 (Classifier) + Stage 2 (Regressor) per fold
# -----------------------------------------------------------------------
for FOLD in 2 3 4; do
  NPZ="$ROOT/fold${FOLD}/sequences_wf.npz"

  if [ ! -f "$NPZ" ]; then
    echo "[ERROR] $NPZ not found — build step may have failed."
    exit 1
  fi

  echo ""
  echo "======================================================="
  echo "=== FOLD ${FOLD}: Stage 1 — Binary Classifier ==="
  echo "======================================================="
  $PYTHON "$SCRIPT_DIR/train_cnn_lstm_classifier.py" \
    --npz      "$NPZ" \
    --out_dir  "$ROOT/fold${FOLD}/stage1" \
    --epochs 150 --batch_size 128 --lr 0.0005 --patience 25 \
    --conv_filters 64 --kernel_size 5 --lstm_units 64 --dropout 0.25 \
    --class_weight auto --recall_target 0.80 \
    2>&1 | tee "$ROOT/fold${FOLD}/stage1_train.log"

  echo ""
  echo "======================================================="
  echo "=== FOLD ${FOLD}: Stage 2 — Regressor (spike_only) ==="
  echo "======================================================="
  $PYTHON "$SCRIPT_DIR/train_cnn_lstm.py" \
    --npz      "$NPZ" \
    --out_dir  "$ROOT/fold${FOLD}/stage2" \
    --spike_only \
    --epochs 150 --batch_size 64 --lr 0.0005 --patience 25 \
    --conv_filters 64 --kernel_size 5 --lstm_units 64 --dropout 0.25 \
    2>&1 | tee "$ROOT/fold${FOLD}/stage2_train.log"

done

# -----------------------------------------------------------------------
# Step 4: Summarize results
# -----------------------------------------------------------------------
echo ""
echo "======================================================="
echo "=== STEP 4: SUMMARIZE WALK-FORWARD RESULTS ==="
echo "======================================================="
$PYTHON "$SCRIPT_DIR/summarize_walk_forward.py" \
  --root   "$ROOT" \
  --folds  2,3,4 \
  --out    "$ROOT/cv_summary.csv"

echo ""
echo "======================================================="
echo " ALL FOLDS DONE — $(date)"
echo "======================================================="
echo ""
echo "Results:"
echo "  CSV   : $ROOT/cv_summary.csv"
echo "  Report: $ROOT/cv_report.md"
echo "  Plot  : $ROOT/figures/cv_r2_barplot.png"
echo ""
cat "$ROOT/cv_report.md" 2>/dev/null || true
