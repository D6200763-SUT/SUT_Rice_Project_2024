#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_twostage_threshold.py
Sweep pred_threshold for Stage 1 → measure impact on full two-stage metrics.
No retraining — uses existing Stage 1 + Stage 2 model_best.keras.

Usage:
/home/ai-station/my_project/.tf251p310/bin/python scripts/evaluate_twostage_threshold.py \
  --npz results/out_feature_sets_w30_h7/context/sequences_window30_h7.npz \
  --stage1_dir results/out_train_stage1 \
  --stage2_dir results/out_train_stage2 \
  --out_dir results/out_threshold_sweep \
  --thresholds 0.30 0.35 0.40 0.45 0.50
"""

from __future__ import annotations
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from train_utils import (
    set_seed, try_enable_tf_memory_growth,
    load_sequences_npz, make_out_dir,
    assert_all_finite, save_json,
    inverse_log1p,
)

BASELINE = {"r2_log1p": 0.4798, "r2_raw": 0.0041, "rmse_raw": 219.51}


# ── Metrics ───────────────────────────────────────────────────────────────────

def _rmse(y_true, y_pred):
    return float(math.sqrt(float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))))


def reg_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    from sklearn.metrics import mean_absolute_error, r2_score
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return {
        "mae":  float(mean_absolute_error(y_true, y_pred)),
        "rmse": _rmse(y_true, y_pred),
        "r2":   float(r2_score(y_true, y_pred)),
    }


def clf_metrics(y_true: np.ndarray, y_pred_label: np.ndarray) -> dict:
    from sklearn.metrics import precision_score, recall_score, f1_score
    y_true  = y_true.astype(int)
    y_label = y_pred_label.astype(int)
    return {
        "precision": float(precision_score(y_true, y_label, zero_division=0)),
        "recall":    float(recall_score(y_true, y_label, zero_division=0)),
        "f1":        float(f1_score(y_true, y_label, zero_division=0)),
    }


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_sweep(df: pd.DataFrame, out_png: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # R² log1p
    axes[0].plot(df["threshold"], df["r2_log1p"], marker="o", color="steelblue", label="R² log1p")
    axes[0].axhline(BASELINE["r2_log1p"], color="red", ls="--", lw=1, label=f"baseline {BASELINE['r2_log1p']}")
    axes[0].set_title("R² (log1p)"); axes[0].set_xlabel("threshold")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    # R² raw
    axes[1].plot(df["threshold"], df["r2_raw"], marker="s", color="tomato", label="R² raw")
    axes[1].axhline(BASELINE["r2_raw"], color="red", ls="--", lw=1, label=f"baseline {BASELINE['r2_raw']}")
    axes[1].set_title("R² (raw)"); axes[1].set_xlabel("threshold")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    # F1 / Recall / Precision
    axes[2].plot(df["threshold"], df["f1"],        marker="^", label="F1")
    axes[2].plot(df["threshold"], df["recall"],    marker="v", label="Recall")
    axes[2].plot(df["threshold"], df["precision"], marker="D", label="Precision")
    axes[2].set_title("Stage 1 Classification"); axes[2].set_xlabel("threshold")
    axes[2].legend(); axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Threshold sweep for two-stage BPH model")
    p.add_argument("--npz",        required=True)
    p.add_argument("--stage1_dir", required=True)
    p.add_argument("--stage2_dir", required=True)
    p.add_argument("--out_dir",    required=True)
    p.add_argument("--thresholds", type=float, nargs="+",
                   default=[0.30, 0.35, 0.40, 0.45, 0.50])
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--seed",       type=int, default=42)
    return p.parse_args(argv)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(argv=None):
    args = parse_args(argv)
    set_seed(args.seed)
    try_enable_tf_memory_growth()

    out_dir = make_out_dir(args.out_dir)
    stage1_dir = Path(args.stage1_dir)
    stage2_dir = Path(args.stage2_dir)

    # ── Load data ─────────────────────────────────────────────────────────────
    seq = load_sequences_npz(args.npz)
    Xte, yte = seq.X_test, seq.y_test.reshape(-1)
    assert_all_finite("X_test", Xte); assert_all_finite("y_test", yte)

    with open(stage1_dir / "threshold.json") as f:
        spike_threshold = json.load(f)["spike_threshold"]
    true_spike_label = (yte > spike_threshold).astype(int)

    print(f"[Sweep] Test samples: {len(yte)}  "
          f"True spikes: {true_spike_label.sum()} ({true_spike_label.mean()*100:.1f}%)")

    # ── Load models ───────────────────────────────────────────────────────────
    import tensorflow as tf

    print("[Sweep] Loading Stage 1 model...")
    s1_model = tf.keras.models.load_model(str(stage1_dir / "model_best.keras"))

    print("[Sweep] Loading Stage 2 model...")
    s2_model = tf.keras.models.load_model(str(stage2_dir / "model_best.keras"))

    # ── Stage 1: get probabilities once ───────────────────────────────────────
    print("[Sweep] Running Stage 1 inference...")
    s1_probs = s1_model.predict(Xte, batch_size=args.batch_size, verbose=0).reshape(-1)

    # ── Stage 2: get regression predictions on all test samples ───────────────
    # (we'll mask per threshold without re-predicting)
    print("[Sweep] Running Stage 2 inference on all test samples...")
    s2_preds = s2_model.predict(Xte, batch_size=args.batch_size, verbose=0).reshape(-1)

    yte_raw = inverse_log1p(yte)

    # ── Sweep ─────────────────────────────────────────────────────────────────
    rows = []
    thresholds = sorted(args.thresholds)

    print(f"\n{'Threshold':>10} {'n_spike':>8} {'%spike':>7} "
          f"{'Prec':>6} {'Recall':>7} {'F1':>6} "
          f"{'R²_log1p':>9} {'R²_raw':>8} {'RMSE_raw':>9}")
    print("-" * 80)

    for thr in thresholds:
        mask = s1_probs >= thr

        # Two-stage inference: mask=0 → 0.0, mask=1 → Stage 2 pred
        y_pred_final = np.where(mask, s2_preds, 0.0)

        # Regression metrics (log1p)
        m_log1p = reg_metrics(yte, y_pred_final)

        # Regression metrics (raw)
        ypred_raw = inverse_log1p(y_pred_final)
        m_raw = reg_metrics(yte_raw, ypred_raw)

        # Stage 1 classification metrics (vs true spike label)
        m_clf = clf_metrics(true_spike_label, mask.astype(int))

        row = {
            "threshold":       thr,
            "n_spike_predicted": int(mask.sum()),
            "pct_spike":       float(mask.mean() * 100),
            "precision":       m_clf["precision"],
            "recall":          m_clf["recall"],
            "f1":              m_clf["f1"],
            "mae_log1p":       m_log1p["mae"],
            "rmse_log1p":      m_log1p["rmse"],
            "r2_log1p":        m_log1p["r2"],
            "mae_raw":         m_raw["mae"],
            "rmse_raw":        m_raw["rmse"],
            "r2_raw":          m_raw["r2"],
            "beat_r2_log1p":   m_log1p["r2"] > BASELINE["r2_log1p"],
            "beat_r2_raw":     m_raw["r2"]   > BASELINE["r2_raw"],
            "beat_rmse_raw":   m_raw["rmse"] < BASELINE["rmse_raw"],
        }
        rows.append(row)

        print(f"{thr:>10.2f} {int(mask.sum()):>8} {mask.mean()*100:>6.1f}% "
              f"{m_clf['precision']:>6.3f} {m_clf['recall']:>7.3f} {m_clf['f1']:>6.3f} "
              f"{m_log1p['r2']:>9.4f} {m_raw['r2']:>8.4f} {m_raw['rmse']:>9.2f}")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    csv_path = out_dir / "threshold_sweep.csv"
    df.to_csv(csv_path, index=False, float_format="%.6f")
    print(f"\n[OK] threshold_sweep.csv → {csv_path}")

    # ── Best thresholds ───────────────────────────────────────────────────────
    best_idx_log1p = int(df["r2_log1p"].idxmax())
    best_idx_raw   = int(df["r2_raw"].idxmax())
    best_log1p = rows[best_idx_log1p]
    best_raw   = rows[best_idx_raw]

    summary = {
        "baseline": BASELINE,
        "best_by_r2_log1p": best_log1p,
        "best_by_r2_raw":   best_raw,
        "all_results":      rows,
    }
    save_json(out_dir / "best_threshold_summary.json", summary)

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_sweep(df, out_dir / "threshold_sweep_plot.png")

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print(f"Baseline (single CNN-LSTM):  R²_log1p={BASELINE['r2_log1p']}  "
          f"R²_raw={BASELINE['r2_raw']}  RMSE_raw={BASELINE['rmse_raw']}")
    print("=" * 68)
    print(f"Best threshold by R²_log1p : {best_log1p['threshold']:.2f}  "
          f"→ R²={best_log1p['r2_log1p']:.4f}  "
          f"{'BETTER' if best_log1p['beat_r2_log1p'] else 'WORSE'} than baseline")
    print(f"Best threshold by R²_raw   : {best_raw['threshold']:.2f}  "
          f"→ R²={best_raw['r2_raw']:.4f}  "
          f"{'BETTER' if best_raw['beat_r2_raw'] else 'WORSE'} than baseline")
    print("=" * 68)
    print(f"[OK] best_threshold_summary.json → {out_dir / 'best_threshold_summary.json'}")
    print(f"[OK] threshold_sweep_plot.png    → {out_dir / 'threshold_sweep_plot.png'}")


if __name__ == "__main__":
    main()
