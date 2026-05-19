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
    p = fold_dir / "stage1" / "metrics_stage1.json"
    if not p.exists():
        print(f"  [WARN] ไม่พบ {p} — ใช้ NaN")
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def load_stage2_metrics(fold_dir: Path) -> dict:
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
        if "y_pred_label" in df_s1.columns and "y_pred_spike" not in df_s1.columns:
            df_s1 = df_s1.rename(columns={"y_pred_label": "y_pred_spike"})
    else:
        print(f"  [WARN] ไม่พบ predictions Stage1: {p1}")

    if p2.exists():
        df_s2 = pd.read_csv(p2, encoding="utf-8-sig")
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
    """
    if join_keys is None:
        join_keys = ["date", "station_id"]

    available_keys = [k for k in join_keys if k in df_s1.columns and k in df_s2.columns]

    if not available_keys:
        if len(df_s1) == len(df_s2):
            print("  [INFO] join by position (no common key columns)")
            df_merged = df_s2.copy()
            df_merged["y_pred_spike"] = df_s1["y_pred_spike"].values
        else:
            print("  [WARN] ไม่สามารถ join Stage1 และ Stage2 predictions ได้")
            return {"spike_rmse_raw": float("nan"), "spike_r2_raw": float("nan"),
                    "spike_mae_raw": float("nan"), "n_spike_rows": 0}
    else:
        df_s1_copy = df_s1[available_keys + ["y_pred_spike"]].copy()
        df_merged = df_s2.merge(df_s1_copy, on=available_keys, how="inner")

    spike_mask = df_merged["y_pred_spike"].values == 1
    spike_rows = df_merged[spike_mask]
    n_spike = int(spike_mask.sum())

    if n_spike < 5:
        print(f"  [WARN] spike rows น้อยมาก ({n_spike}) — ผล spike metrics อาจไม่น่าเชื่อถือ")
        return {"spike_rmse_raw": float("nan"), "spike_r2_raw": float("nan"),
                "spike_mae_raw": float("nan"), "n_spike_rows": n_spike}

    y_true = spike_rows["y_true_raw"].values.astype(float)
    y_pred = spike_rows["y_pred_raw"].values.astype(float)

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

        # Stage 1 metrics
        s1 = load_stage1_metrics(fold_dir)
        recall     = s1.get("recall",          float("nan"))
        precision  = s1.get("precision",       float("nan"))
        f1         = s1.get("f1",              float("nan"))
        auc        = s1.get("auc",             float("nan"))
        threshold  = s1.get("best_threshold",  float("nan"))

        # Stage 2 overall metrics
        s2 = load_stage2_metrics(fold_dir)
        tlog = s2.get("test_log1p", {})
        traw = s2.get("test_raw",   {})
        r2_log1p   = tlog.get("r2",   float("nan"))
        rmse_log1p = tlog.get("rmse", float("nan"))
        mae_log1p  = tlog.get("mae",  float("nan"))
        r2_raw     = traw.get("r2",   float("nan"))
        rmse_raw   = traw.get("rmse", float("nan"))
        mae_raw    = traw.get("mae",  float("nan"))

        # Spike-window metrics
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

        print(f"  Stage1: Recall={recall:.3f}  Precision={precision:.3f}  F1={f1:.3f}  AUC={auc:.3f}")
        print(f"  Stage2: R2_log1p={r2_log1p:.3f}  RMSE_raw={rmse_raw:.1f}")
        print(f"  Spike-window: RMSE_raw={spike_rmse_raw:.1f}  R2_raw={spike_r2_raw:.3f}  n={n_spike_rows}")

        rows.append({
            "fold":           fold,
            "recall_s1":      recall,
            "precision_s1":   precision,
            "f1_s1":          f1,
            "auc_s1":         auc,
            "best_threshold": threshold,
            "r2_log1p_s2":    r2_log1p,
            "rmse_log1p_s2":  rmse_log1p,
            "mae_log1p_s2":   mae_log1p,
            "r2_raw_s2":      r2_raw,
            "rmse_raw_s2":    rmse_raw,
            "mae_raw_s2":     mae_raw,
            "spike_rmse_raw": spike_rmse_raw,
            "spike_r2_raw":   spike_r2_raw,
            "spike_mae_raw":  spike_mae_raw,
            "n_spike_rows":   n_spike_rows,
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\n[OK] Saved: {out_csv}")

    stats = {}
    metric_cols = [c for c in df.columns if c != "fold"]
    for col in metric_cols:
        vals = df[col].tolist()
        m, s = mean_std(vals)
        stats[col] = (m, s)

    def fmt(val):
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return "—"
        if isinstance(val, (int, np.integer)):
            return str(int(val))
        return f"{val:.3f}"

    def fmt_ms(col):
        m, s = stats[col]
        if math.isnan(m):
            return "—"
        return f"{m:.3f}±{s:.3f}"

    fold_vals = {fold: row for fold, row in zip(folds, rows)}

    md_lines = [
        "## Walk-Forward Cross-Validation Results (Two-stage CNN-LSTM)",
        "",
        f"**Data:** BPH Northeast Thailand 2015–2019 | **Model:** CNN-LSTM + context (18 features) | **W=30, H=7**",
        f"**Stage 1:** Spike classifier (bph_raw > {args.spike_threshold:.0f}) | **Stage 2:** Full regression (all samples)",
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
        if col == "recall_s1"   and not math.isnan(m): status = "✅" if m >= 0.80 else "⚠️"
        if col == "r2_log1p_s2" and not math.isnan(m): status = "✅" if m >= 0.40 else "❌"
        if math.isnan(m):
            print(f"  {label:<30} —")
        else:
            print(f"  {label:<30} {m:.3f} ± {s:.3f}  {status}")
    print("="*60)


if __name__ == "__main__":
    main()
