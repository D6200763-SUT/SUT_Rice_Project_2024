#!/bin/bash
# =============================================================
# run_two_stage.sh  —  รัน Two-stage experiments หลาย quantile
# =============================================================
# Usage:
#   chmod +x scripts/run_two_stage.sh
#   ./scripts/run_two_stage.sh                    # รันทุก config
#   ./scripts/run_two_stage.sh --quick            # รันแค่ config หลัก
#   ./scripts/run_two_stage.sh --npz path/to.npz  # ระบุ NPZ เอง
# =============================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORK_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$WORK_DIR"

PYTHON="/home/ai-station/my_project/.tf251p310/bin/python"
NPZ="results/out_feature_sets_w30_h1/context/sequences_window30_h1.npz"
OUT_BASE="results/out_train_two_stage"
QUICK=false
LOG="results/two_stage_runs.log"

while [[ $# -gt 0 ]]; do
  case $1 in
    --quick)   QUICK=true;  shift ;;
    --npz)     NPZ="$2";    shift 2 ;;
    --out)     OUT_BASE="$2"; shift 2 ;;
    *) echo "Unknown: $1"; exit 1 ;;
  esac
done

mkdir -p results
exec > >(tee -a "$LOG") 2>&1

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }
sep() { echo ""; echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; }

sep
echo "[$(timestamp)] Two-stage experiments START"
echo "  NPZ: $NPZ"
echo "  Quick: $QUICK"

# ==============================
# Config ที่จะทดสอบ
# ==============================
# format: "spike_quantile:label"
if [[ "$QUICK" == "true" ]]; then
  CONFIGS=("0.75:q75")
else
  CONFIGS=("0.75:q75" "0.80:q80" "0.85:q85" "0.90:q90")
fi

for CONFIG in "${CONFIGS[@]}"; do
  IFS=':' read -r QUANTILE LABEL <<< "$CONFIG"
  OUT_DIR="${OUT_BASE}_${LABEL}"

  sep
  echo "[$(timestamp)] Running: quantile=$QUANTILE  out=$OUT_DIR"

  $PYTHON scripts/train_two_stage.py \
    --npz "$NPZ" \
    --out_dir "$OUT_DIR" \
    --spike_quantile "$QUANTILE" \
    --spike_threshold 0.5 \
    --epochs_s1 120 --patience_s1 20 \
    --epochs_s2 150 --patience_s2 25 \
    --batch_size 128 \
    --lr 0.0005 \
    --lr_s2 0.0003 \
    --conv_filters 64 \
    --lstm_units 64 \
    --dropout 0.25

  echo "[$(timestamp)] Done: $OUT_DIR"
done

# ==============================
# สรุปผลเปรียบเทียบทุก config
# ==============================
sep
echo "[$(timestamp)] Summarizing results..."

$PYTHON - << 'PYEOF'
import json, os
from pathlib import Path

base = Path("results")
rows = []
for d in sorted(base.glob("out_train_two_stage_*")):
    mf = d / "metrics.json"
    if not mf.exists():
        continue
    m = json.loads(mf.read_text())
    cfg = m.get("config", {})
    clf = m.get("stage1_classifier", {})
    log = m.get("log1p", {})
    raw = m.get("raw", {})
    imp = m.get("improvement_vs_single_stage", {})
    rows.append({
        "exp":         d.name,
        "Q":           cfg.get("spike_quantile"),
        "spike_ratio": f"{cfg.get('spike_ratio_train', 0):.1%}",
        "clf_f1":      f"{clf.get('f1', 0):.4f}",
        "clf_auc":     f"{clf.get('auc_roc', 0):.4f}",
        "r2_log1p":    f"{log.get('r2', 0):.4f}",
        "rmse_log1p":  f"{log.get('rmse', 0):.4f}",
        "rmse_raw":    f"{raw.get('rmse', 0):.4f}",
        "delta_r2":    f"{imp.get('delta_r2', 0):+.4f}" if imp.get('delta_r2') is not None else "N/A",
    })

if not rows:
    print("No results found")
    exit()

# Print table
header = ["exp","Q","spike_ratio","clf_f1","clf_auc","r2_log1p","rmse_log1p","rmse_raw","delta_r2"]
col_w = {h: max(len(h), max(len(str(r[h])) for r in rows)) for h in header}
fmt = "  " + "  ".join(f"{{:<{col_w[h]}}}" for h in header)
print()
print(fmt.format(*header))
print("  " + "  ".join("─" * col_w[h] for h in header))
for r in rows:
    print(fmt.format(*[str(r[h]) for h in header]))

# Best by r2_log1p
best = max(rows, key=lambda x: float(x["r2_log1p"]))
print(f"\n  Best: {best['exp']}  R²={best['r2_log1p']}  Δ={best['delta_r2']}")

# Save CSV
import csv
csv_path = Path("results/two_stage_comparison.csv")
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=header)
    w.writeheader()
    w.writerows(rows)
print(f"\n  Saved: {csv_path}")
PYEOF

sep
echo "[$(timestamp)] All two-stage experiments done"
echo "  Log: $LOG"
echo "  Summary: results/two_stage_comparison.csv"
