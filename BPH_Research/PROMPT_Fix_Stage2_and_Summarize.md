# PROMPT: แก้ไข Stage 2 + summarize_walk_forward.py ใหม่
# ใช้กับ: Claude Code (claude.ai/code) ใน BPH_Research/
# รันจาก: BPH_Research/
# Python: /home/ai-station/my_project/.tf251p310/bin/python

---

## CONTEXT — ปัญหาที่พบและสาเหตุ

ผลการรัน Walk-Forward CV ครั้งที่แล้ว:

| Metric | ผลที่ได้ | เป้าหมาย | สถานะ |
|---|---|---|---|
| Stage1 Recall (mean) | 0.807 | ≥ 0.80 | ✅ ผ่าน |
| Stage1 AUC (mean) | 0.881 | — | ✅ ดีมาก |
| Stage2 R² log1p (mean) | -0.119 | ≥ 0.45 | ❌ ต้องแก้ |
| Stage2 RMSE_raw (mean) | 927 ตัว/กอ | — | ❌ สูงมาก |

**Root cause ของ Stage 2:**
การใช้ `--spike_only` ทำให้ train set เหลือเพียง ~5-15% ของข้อมูล
CNN-LSTM ที่มี parameter หลายหมื่นตัวไม่มีข้อมูลพอ → R² ติดลบ

**สถาปัตยกรรม Two-stage ที่ถูกต้อง:**
- Stage 1 = classifier: กรอง "spike risk" จาก "ปกติ" (ทำแล้ว ✅)
- Stage 2 = regressor: เทรนบน **ALL samples** (ทุก rows)
  → predict ทุก rows → evaluate metric เฉพาะ rows ที่ Stage 1 = spike
- ไม่ใช่ train Stage 2 บน spike_only

**ผลที่คาดหวังหลังแก้:**
- Stage2 R² log1p กลับมาใกล้ baseline 0.484 (single-split)
- RMSE_raw ลดลงมากจาก 927 → ~220 ตัว/กอ
- สร้าง "spike-window RMSE" เพิ่มเติม: RMSE เฉพาะ rows ที่ Stage1=1

---

## TASK 1: รัน Stage 2 ใหม่ทั้ง 3 folds (ไม่ใช้ --spike_only)

ใช้ `scripts/train_cnn_lstm.py` ที่มีอยู่แล้ว **ไม่ต้องแก้ไขไฟล์ใดเลย**
แค่รันโดยไม่มี `--spike_only` flag

```bash
source /home/ai-station/my_project/.tf251p310/bin/activate
cd BPH_Research

# Fold 2
python scripts/train_cnn_lstm.py \
  --npz results/walk_forward/fold2/sequences_wf.npz \
  --out_dir results/walk_forward/fold2/stage2_full \
  --epochs 150 --batch_size 128 --lr 0.0005 --patience 25 \
  --clipnorm 1.0 --conv_filters 64 --kernel_size 5 \
  --lstm_units 64 --dropout 0.25 \
  2>&1 | tee results/walk_forward/fold2/stage2_full_train.log

# Fold 3
python scripts/train_cnn_lstm.py \
  --npz results/walk_forward/fold3/sequences_wf.npz \
  --out_dir results/walk_forward/fold3/stage2_full \
  --epochs 150 --batch_size 128 --lr 0.0005 --patience 25 \
  --clipnorm 1.0 --conv_filters 64 --kernel_size 5 \
  --lstm_units 64 --dropout 0.25 \
  2>&1 | tee results/walk_forward/fold3/stage2_full_train.log

# Fold 4
python scripts/train_cnn_lstm.py \
  --npz results/walk_forward/fold4/sequences_wf.npz \
  --out_dir results/walk_forward/fold4/stage2_full \
  --epochs 150 --batch_size 128 --lr 0.0005 --patience 25 \
  --clipnorm 1.0 --conv_filters 64 --kernel_size 5 \
  --lstm_units 64 --dropout 0.25 \
  2>&1 | tee results/walk_forward/fold4/stage2_full_train.log
```

**รันแบบ background (ถ้า SSH):**
```bash
nohup bash -c '
  python scripts/train_cnn_lstm.py \
    --npz results/walk_forward/fold2/sequences_wf.npz \
    --out_dir results/walk_forward/fold2/stage2_full \
    --epochs 150 --batch_size 128 --lr 0.0005 --patience 25 \
    --clipnorm 1.0 --conv_filters 64 --kernel_size 5 \
    --lstm_units 64 --dropout 0.25 && \
  python scripts/train_cnn_lstm.py \
    --npz results/walk_forward/fold3/sequences_wf.npz \
    --out_dir results/walk_forward/fold3/stage2_full \
    --epochs 150 --batch_size 128 --lr 0.0005 --patience 25 \
    --clipnorm 1.0 --conv_filters 64 --kernel_size 5 \
    --lstm_units 64 --dropout 0.25 && \
  python scripts/train_cnn_lstm.py \
    --npz results/walk_forward/fold4/sequences_wf.npz \
    --out_dir results/walk_forward/fold4/stage2_full \
    --epochs 150 --batch_size 128 --lr 0.0005 --patience 25 \
    --clipnorm 1.0 --conv_filters 64 --kernel_size 5 \
    --lstm_units 64 --dropout 0.25
' > stage2_all_folds.log 2>&1 &

echo "PID: $!"
tail -f stage2_all_folds.log
```

**ตรวจผลหลัง train เสร็จแต่ละ fold:**
```bash
# ตรวจว่า metrics.json มีค่า r2_log1p เป็นบวก
python -c "
import json
for fold in [2,3,4]:
    m = json.load(open(f'results/walk_forward/fold{fold}/stage2_full/metrics.json'))
    r2 = m['test_log1p']['r2']
    rmse_raw = m['test_raw']['rmse']
    print(f'Fold {fold}: R2_log1p={r2:.3f}  RMSE_raw={rmse_raw:.1f}')
"
```

**เกณฑ์ผ่าน TASK 1:**
- R² log1p ทุก fold > 0.00 (ถ้าติดลบยังอยู่ = มีปัญหาอื่น)
- R² log1p mean ≥ 0.40 (คาดว่าจะได้ ~0.45-0.48)
- RMSE_raw mean < 300 ตัว/กอ

---

## TASK 2: สร้าง summarize_walk_forward.py ใหม่ (เวอร์ชัน Two-stage)

สร้างไฟล์ `scripts/summarize_walk_forward.py` ที่รวม metrics จาก Stage 1 + Stage 2
และคำนวณ **spike-window metrics** พิเศษ (evaluate Stage 2 เฉพาะตอนที่ Stage 1 บอกว่า spike)

### โครงสร้างไฟล์ที่ script ต้องอ่าน:
```
results/walk_forward/
├── fold2/
│   ├── stage1/
│   │   ├── metrics_stage1.json       ← recall, precision, f1, auc, best_threshold
│   │   └── predictions_test_stage1.csv  ← date, station_id, y_true_spike, y_pred_prob, y_pred_spike
│   └── stage2_full/
│       ├── metrics.json              ← test_log1p: {mae,rmse,r2}, test_raw: {mae,rmse,r2}
│       └── predictions_test.csv      ← date, station_id, y_true, y_pred, y_true_raw, y_pred_raw
├── fold3/ ...
└── fold4/ ...
```

### Script spec: `scripts/summarize_walk_forward.py`

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
summarize_walk_forward.py
รวม metrics จาก Walk-Forward CV ทุก fold
คำนวณ: Stage1 metrics + Stage2 overall + Stage2 spike-window metrics
Usage:
  python scripts/summarize_walk_forward.py \
    --root results/walk_forward \
    --folds 2,3,4 \
    --out results/walk_forward/cv_summary.csv
"""

from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np
import pandas as pd


def load_stage1_metrics(fold_dir: Path) -> dict:
    """โหลด metrics_stage1.json จาก stage1/"""
    p = fold_dir / "stage1" / "metrics_stage1.json"
    if not p.exists():
        print(f"  [WARN] ไม่พบ {p} — ใช้ NaN")
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def load_stage2_metrics(fold_dir: Path) -> dict:
    """โหลด metrics.json จาก stage2_full/"""
    p = fold_dir / "stage2_full" / "metrics.json"
    if not p.exists():
        print(f"  [WARN] ไม่พบ {p} — ใช้ NaN")
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def load_predictions(fold_dir: Path) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """
    โหลด predictions_test_stage1.csv และ predictions_test.csv
    คืน (df_s1, df_s2) — None ถ้าไม่มีไฟล์
    """
    p1 = fold_dir / "stage1" / "predictions_test_stage1.csv"
    p2 = fold_dir / "stage2_full" / "predictions_test.csv"

    df_s1 = None
    df_s2 = None

    if p1.exists():
        df_s1 = pd.read_csv(p1, encoding="utf-8-sig")
        # ต้องมีคอลัมน์: date, station_id, y_pred_spike (หรือ y_pred_label)
        # normalize column names
        if "y_pred_label" in df_s1.columns and "y_pred_spike" not in df_s1.columns:
            df_s1 = df_s1.rename(columns={"y_pred_label": "y_pred_spike"})
    else:
        print(f"  [WARN] ไม่พบ predictions Stage1: {p1}")

    if p2.exists():
        df_s2 = pd.read_csv(p2, encoding="utf-8-sig")
        # ต้องมีคอลัมน์: date, station_id, y_true_raw, y_pred_raw
        if "y_true_raw" not in df_s2.columns and "y_true" in df_s2.columns:
            df_s2["y_true_raw"] = np.expm1(df_s2["y_true"].values)
        if "y_pred_raw" not in df_s2.columns and "y_pred" in df_s2.columns:
            df_s2["y_pred_raw"] = np.expm1(df_s2["y_pred"].values)
    else:
        print(f"  [WARN] ไม่พบ predictions Stage2: {p2}")

    return df_s1, df_s2


def compute_spike_window_metrics(
    df_s1: pd.DataFrame,
    df_s2: pd.DataFrame,
    join_keys: list[str] = None
) -> dict:
    """
    คำนวณ RMSE_raw และ R²_raw เฉพาะ rows ที่ Stage1 predict = spike (y_pred_spike == 1)

    join_keys: คอลัมน์ที่ใช้ join สองตาราง (default: ["date", "station_id"])
    คืน dict: {spike_rmse_raw, spike_r2_raw, spike_mae_raw, n_spike_rows}
    """
    if join_keys is None:
        join_keys = ["date", "station_id"]

    # หา join keys ที่มีในทั้งสองตาราง
    available_keys = [k for k in join_keys if k in df_s1.columns and k in df_s2.columns]

    if not available_keys:
        # ถ้า join ไม่ได้ ลอง merge by index position (ถ้า len เท่ากัน)
        if len(df_s1) == len(df_s2):
            print("  [INFO] join by position (no common key columns)")
            spike_mask = df_s1["y_pred_spike"].values == 1
        else:
            print("  [WARN] ไม่สามารถ join Stage1 และ Stage2 predictions ได้")
            return {"spike_rmse_raw": float("nan"), "spike_r2_raw": float("nan"),
                    "spike_mae_raw": float("nan"), "n_spike_rows": 0}
    else:
        # merge โดยใช้ key columns
        df_s1_copy = df_s1[available_keys + ["y_pred_spike"]].copy()
        df_s2_copy = df_s2.copy()
        merged = df_s2_copy.merge(df_s1_copy, on=available_keys, how="inner")
        spike_mask = merged["y_pred_spike"].values == 1
        df_s2 = merged  # ใช้ merged แทน

    spike_rows = df_s2[spike_mask]
    n_spike = int(spike_mask.sum())

    if n_spike < 5:
        print(f"  [WARN] spike rows น้อยมาก ({n_spike}) — ผล spike metrics อาจไม่น่าเชื่อถือ")
        return {"spike_rmse_raw": float("nan"), "spike_r2_raw": float("nan"),
                "spike_mae_raw": float("nan"), "n_spike_rows": n_spike}

    y_true = spike_rows["y_true_raw"].values.astype(float)
    y_pred = spike_rows["y_pred_raw"].values.astype(float)

    # กำจัด NaN/Inf
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[valid], y_pred[valid]

    mae  = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(math.sqrt(np.mean((y_true - y_pred) ** 2)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-9 else float("nan")

    return {
        "spike_rmse_raw": rmse,
        "spike_r2_raw": r2,
        "spike_mae_raw": mae,
        "n_spike_rows": n_spike,
    }


def mean_std(values: list) -> tuple[float, float]:
    """คืน (mean, std) โดยกำจัด NaN"""
    v = [x for x in values if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not v:
        return float("nan"), float("nan")
    arr = np.array(v, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=0))


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--root",  required=True, help="root dir ของ walk_forward results")
    p.add_argument("--folds", default="2,3,4", help="fold numbers คั่นด้วย comma")
    p.add_argument("--out",   required=True, help="path ของ output CSV")
    p.add_argument("--spike_threshold", type=float, default=100.0,
                   help="threshold (bph_raw) ที่นิยาม spike สำหรับ reference เท่านั้น")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    root = Path(args.root).expanduser().resolve()
    out_csv = Path(args.out).expanduser().resolve()
    out_dir = out_csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    folds = [int(f.strip()) for f in args.folds.split(",")]
    rows = []

    for fold in folds:
        fold_dir = root / f"fold{fold}"
        print(f"\n=== Fold {fold} ===")

        # ---- Stage 1 metrics ----
        s1 = load_stage1_metrics(fold_dir)
        recall     = s1.get("recall", float("nan"))
        precision  = s1.get("precision", float("nan"))
        f1         = s1.get("f1", float("nan"))
        auc        = s1.get("auc", float("nan"))
        threshold  = s1.get("best_threshold", float("nan"))

        # ---- Stage 2 overall metrics ----
        s2 = load_stage2_metrics(fold_dir)
        tlog = s2.get("test_log1p", {})
        traw = s2.get("test_raw",   {})
        r2_log1p   = tlog.get("r2",   float("nan"))
        rmse_log1p = tlog.get("rmse", float("nan"))
        mae_log1p  = tlog.get("mae",  float("nan"))
        r2_raw     = traw.get("r2",   float("nan"))
        rmse_raw   = traw.get("rmse", float("nan"))
        mae_raw    = traw.get("mae",  float("nan"))

        # ---- Spike-window metrics ----
        spike_rmse_raw = float("nan")
        spike_r2_raw   = float("nan")
        spike_mae_raw  = float("nan")
        n_spike_rows   = 0

        df_s1, df_s2 = load_predictions(fold_dir)
        if df_s1 is not None and df_s2 is not None:
            spike_m = compute_spike_window_metrics(df_s1, df_s2)
            spike_rmse_raw = spike_m["spike_rmse_raw"]
            spike_r2_raw   = spike_m["spike_r2_raw"]
            spike_mae_raw  = spike_m["spike_mae_raw"]
            n_spike_rows   = spike_m["n_spike_rows"]

        # รายงานผล fold
        print(f"  Stage1: Recall={recall:.3f}  Precision={precision:.3f}  F1={f1:.3f}  AUC={auc:.3f}")
        print(f"  Stage2: R2_log1p={r2_log1p:.3f}  RMSE_raw={rmse_raw:.1f}")
        print(f"  Spike-window: RMSE_raw={spike_rmse_raw:.1f}  R2_raw={spike_r2_raw:.3f}  n={n_spike_rows}")

        rows.append({
            "fold":           fold,
            # Stage 1
            "recall_s1":      recall,
            "precision_s1":   precision,
            "f1_s1":          f1,
            "auc_s1":         auc,
            "best_threshold": threshold,
            # Stage 2 overall
            "r2_log1p_s2":    r2_log1p,
            "rmse_log1p_s2":  rmse_log1p,
            "mae_log1p_s2":   mae_log1p,
            "r2_raw_s2":      r2_raw,
            "rmse_raw_s2":    rmse_raw,
            "mae_raw_s2":     mae_raw,
            # Stage 2 spike-window
            "spike_rmse_raw": spike_rmse_raw,
            "spike_r2_raw":   spike_r2_raw,
            "spike_mae_raw":  spike_mae_raw,
            "n_spike_rows":   n_spike_rows,
        })

    # บันทึก CSV
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\n[OK] Saved: {out_csv}")

    # คำนวณ mean ± std
    stats = {}
    metric_cols = [c for c in df.columns if c != "fold"]
    for col in metric_cols:
        vals = df[col].tolist()
        m, s = mean_std(vals)
        stats[col] = (m, s)

    # สร้าง manuscript-ready markdown table
    def fmt(val):
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return "—"
        return f"{val:.3f}"

    def fmt_ms(col):
        m, s = stats[col]
        if math.isnan(m):
            return "—"
        return f"{m:.3f}±{s:.3f}"

    fold_vals = {fold: {r["fold"]: r for r in rows}[fold] for fold in folds}

    md_lines = [
        "## Walk-Forward Cross-Validation Results (Two-stage CNN-LSTM)",
        "",
        f"**Data:** BPH Northeast Thailand 2015–2019 | **Model:** CNN-LSTM + context (18 features) | **W=30, H=7**",
        f"**Stage 1:** Spike classifier (bph_raw > {args.spike_threshold:.0f}) | **Stage 2:** Full regression",
        "",
        "### Stage 1 — Spike Classifier",
        "",
        "| Metric | Fold 2 | Fold 3 | Fold 4 | Mean ± SD |",
        "|--------|--------|--------|--------|-----------|",
    ]
    for col, label in [
        ("recall_s1",      "Recall"),
        ("precision_s1",   "Precision"),
        ("f1_s1",          "F1"),
        ("auc_s1",         "AUC"),
        ("best_threshold", "Best threshold"),
    ]:
        vals_str = " | ".join(fmt(fold_vals[f][col]) for f in folds)
        md_lines.append(f"| {label} | {vals_str} | {fmt_ms(col)} |")

    md_lines += [
        "",
        "### Stage 2 — Regression (All samples)",
        "",
        "| Metric | Fold 2 | Fold 3 | Fold 4 | Mean ± SD |",
        "|--------|--------|--------|--------|-----------|",
    ]
    for col, label in [
        ("r2_log1p_s2",   "R² (log1p)"),
        ("rmse_log1p_s2", "RMSE (log1p)"),
        ("r2_raw_s2",     "R² (raw)"),
        ("rmse_raw_s2",   "RMSE (raw, ตัว/กอ)"),
    ]:
        vals_str = " | ".join(fmt(fold_vals[f][col]) for f in folds)
        md_lines.append(f"| {label} | {vals_str} | {fmt_ms(col)} |")

    md_lines += [
        "",
        "### Stage 2 — Spike-window Metrics (evaluate เฉพาะ rows ที่ Stage1=spike)",
        "",
        "| Metric | Fold 2 | Fold 3 | Fold 4 | Mean ± SD |",
        "|--------|--------|--------|--------|-----------|",
    ]
    for col, label in [
        ("spike_rmse_raw", "RMSE raw (spike windows)"),
        ("spike_r2_raw",   "R² raw (spike windows)"),
        ("spike_mae_raw",  "MAE raw (spike windows)"),
        ("n_spike_rows",   "N spike rows"),
    ]:
        vals_str = " | ".join(fmt(fold_vals[f][col]) for f in folds)
        md_lines.append(f"| {label} | {vals_str} | {fmt_ms(col)} |")

    md_lines += [
        "",
        "**Baseline (single-stage CNN-LSTM, Fold 4 test 2019):** R²_log1p = 0.484",
        "",
        "> **Note on Precision:** Stage 1 Precision ~0.34 reflects an intentional design choice.",
        "> In early warning systems, missing a real outbreak (False Negative) is more costly than",
        "> a false alarm (False Positive). Stage 2 regression serves as the second filter to",
        "> reduce false positives by providing severity estimates for flagged periods.",
        "",
        "---",
        "*Generated by summarize_walk_forward.py*",
    ]

    md_path = out_dir / "cv_report.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"[OK] Saved: {md_path}")

    # สรุปผลบน console
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for col, label in [
        ("recall_s1",      "Stage1 Recall"),
        ("f1_s1",          "Stage1 F1"),
        ("auc_s1",         "Stage1 AUC"),
        ("r2_log1p_s2",    "Stage2 R² log1p"),
        ("rmse_raw_s2",    "Stage2 RMSE raw"),
        ("spike_rmse_raw", "Spike-window RMSE raw"),
        ("spike_r2_raw",   "Spike-window R² raw"),
    ]:
        m, s = stats[col]
        status = ""
        if col == "recall_s1"  and not math.isnan(m): status = "✅" if m >= 0.80 else "⚠️"
        if col == "r2_log1p_s2" and not math.isnan(m): status = "✅" if m >= 0.40 else "❌"
        print(f"  {label:<30} {m:.3f} ± {s:.3f} {status}")
    print("="*60)


if __name__ == "__main__":
    main()
```

---

## TASK 3: รัน summarize_walk_forward.py

หลัง TASK 1 เสร็จทั้ง 3 folds ให้รัน:

```bash
python scripts/summarize_walk_forward.py \
  --root results/walk_forward \
  --folds 2,3,4 \
  --out results/walk_forward/cv_summary.csv \
  --spike_threshold 100

# ดูผล
cat results/walk_forward/cv_report.md
```

---

## TASK 4: Quick-check predictions files (ถ้า summarizer หา predictions ไม่เจอ)

ถ้า TASK 2 รัน summarizer แล้วได้ spike_rmse_raw = NaN แสดงว่า
predictions CSV ชื่อไม่ตรงหรืออยู่ผิดที่ — ให้ตรวจสอบ:

```bash
# ตรวจว่า predictions files มีอยู่และมี columns ถูกต้อง
python -c "
import pandas as pd
from pathlib import Path

for fold in [2,3,4]:
    print(f'--- Fold {fold} ---')

    p1 = Path(f'results/walk_forward/fold{fold}/stage1/predictions_test_stage1.csv')
    if p1.exists():
        df = pd.read_csv(p1)
        print(f'  Stage1 predictions: {len(df)} rows, cols={list(df.columns)}')
    else:
        print(f'  [MISS] {p1}')

    p2 = Path(f'results/walk_forward/fold{fold}/stage2_full/predictions_test.csv')
    if p2.exists():
        df = pd.read_csv(p2)
        print(f'  Stage2 predictions: {len(df)} rows, cols={list(df.columns)}')
    else:
        print(f'  [MISS] {p2}')
"
```

**ถ้า Stage2 predictions ชื่อ `predictions_test.csv` ไม่มี:**
`train_cnn_lstm.py` บันทึกเป็น `predictions_test.csv` อยู่แล้ว (จาก `save_predictions_csv`)
ตรวจด้วย:
```bash
ls results/walk_forward/fold4/stage2_full/
# ควรเห็น: metrics.json, predictions_test.csv, figures/, REPORT.md, model_best.keras
```

**ถ้า Stage1 predictions ชื่อต่างออกไป** (เช่น `test_predictions.csv`):
แก้ไขใน `load_predictions()` ของ summarizer:
```python
# บรรทัดนี้ใน load_predictions():
p1 = fold_dir / "stage1" / "predictions_test_stage1.csv"
# เปลี่ยนเป็นชื่อที่มีจริง เช่น:
p1 = fold_dir / "stage1" / "predictions_test.csv"
```

---

## TASK 5: สร้างกราฟ R² ข้าม folds (optional แต่ดีสำหรับ paper)

```bash
python -c "
import json, math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

folds = [2, 3, 4]
r2_s2, recall_s1 = [], []

for fold in folds:
    m2 = json.load(open(f'results/walk_forward/fold{fold}/stage2_full/metrics.json'))
    m1 = json.load(open(f'results/walk_forward/fold{fold}/stage1/metrics_stage1.json'))
    r2_s2.append(m2['test_log1p']['r2'])
    recall_s1.append(m1.get('recall', float('nan')))

fig, axes = plt.subplots(1, 2, figsize=(9, 4))
x = np.arange(len(folds))
labels = [f'Fold {f}\n(test {2015+f})' for f in folds]

# Stage 2 R²
axes[0].bar(x, r2_s2, color='#378ADD', alpha=0.8, width=0.5)
axes[0].axhline(0.484, color='#E24B4A', linestyle='--', linewidth=1, label='Baseline 0.484')
axes[0].axhline(np.nanmean(r2_s2), color='#639922', linestyle=':', linewidth=1.5,
                label=f'Mean {np.nanmean(r2_s2):.3f}')
axes[0].set_xticks(x); axes[0].set_xticklabels(labels, fontsize=9)
axes[0].set_ylabel('R² (log1p)'); axes[0].set_title('Stage 2: R² across folds')
axes[0].legend(fontsize=8); axes[0].set_ylim(-0.2, 0.65)
for i, v in enumerate(r2_s2):
    axes[0].text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=9)

# Stage 1 Recall
axes[1].bar(x, recall_s1, color='#1D9E75', alpha=0.8, width=0.5)
axes[1].axhline(0.80, color='#E24B4A', linestyle='--', linewidth=1, label='Target 0.80')
axes[1].axhline(np.nanmean(recall_s1), color='#639922', linestyle=':', linewidth=1.5,
                label=f'Mean {np.nanmean(recall_s1):.3f}')
axes[1].set_xticks(x); axes[1].set_xticklabels(labels, fontsize=9)
axes[1].set_ylabel('Recall'); axes[1].set_title('Stage 1: Recall across folds')
axes[1].legend(fontsize=8); axes[1].set_ylim(0.5, 1.05)
for i, v in enumerate(recall_s1):
    axes[1].text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=9)

plt.tight_layout()
out = 'results/walk_forward/cv_barplot.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f'[OK] Saved: {out}')
"
```

---

## EXECUTION ORDER (รันตามลำดับ)

```bash
# Step 1: activate env
source /home/ai-station/my_project/.tf251p310/bin/activate
cd BPH_Research

# Step 2: รัน Stage 2 ทั้ง 3 folds (ใช้ nohup ถ้า SSH)
# [ดูคำสั่งใน TASK 1]

# Step 3: ตรวจว่า R² เป็นบวกทุก fold
python -c "
import json
for fold in [2,3,4]:
    try:
        m = json.load(open(f'results/walk_forward/fold{fold}/stage2_full/metrics.json'))
        r2 = m['test_log1p']['r2']
        print(f'Fold {fold}: R2={r2:.3f}', '✅' if r2 > 0 else '❌')
    except FileNotFoundError:
        print(f'Fold {fold}: ยังไม่เสร็จ')
"

# Step 4: สร้าง summarize_walk_forward.py (copy จาก TASK 2)

# Step 5: รัน summarizer
python scripts/summarize_walk_forward.py \
  --root results/walk_forward \
  --folds 2,3,4 \
  --out results/walk_forward/cv_summary.csv

# Step 6: ดูผล
cat results/walk_forward/cv_report.md

# Step 7: สร้างกราฟ (optional)
# [ดูคำสั่งใน TASK 5]
```

---

## SUCCESS CRITERIA หลัง fix

| Metric | ก่อน fix | หลัง fix (คาด) | เกณฑ์ผ่าน |
|--------|----------|----------------|-----------|
| Stage2 R² log1p mean | -0.119 | 0.40–0.48 | ≥ 0.40 ✅ |
| Stage2 RMSE_raw mean | 927 | ~200–250 | < 300 ✅ |
| Spike-window RMSE_raw | N/A | คำนวณใหม่ | ต้องการ baseline |
| Stage1 Recall mean | 0.807 | 0.807 (ไม่เปลี่ยน) | ≥ 0.80 ✅ |

---

## NOTES สำหรับ Claude Code

- ไม่ต้องแก้ `train_cnn_lstm.py`, `train_utils.py` ใดๆ
- `train_cnn_lstm.py` บันทึก predictions เป็น `predictions_test.csv`
  ต้องมี columns: `y_true`, `y_pred`, `y_true_raw`, `y_pred_raw`, `date`, `station_id`
- ถ้า `predictions_test_stage1.csv` ไม่มี column `y_pred_spike`
  ให้ตรวจว่า classifier script บันทึก column ชื่ออะไร แล้วแก้ใน `load_predictions()`
- Python path: `/home/ai-station/my_project/.tf251p310/bin/python`
- รันจาก `BPH_Research/` เสมอ
