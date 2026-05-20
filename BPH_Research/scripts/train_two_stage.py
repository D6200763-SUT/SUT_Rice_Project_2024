#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_two_stage.py  —  BPH Two-stage Model
===========================================
Stage 1: CNN-LSTM binary classifier  → spike / no-spike
Stage 2: CNN-LSTM regressor          → ทำนายเฉพาะ sample ที่ classifier บอกว่า spike

หลักการ:
- ข้อมูล BPH เป็น zero-inflated: วันส่วนใหญ่ BPH ต่ำ มีเพียงไม่กี่ช่วงที่พีคสูง
- Single-stage regression ต้อง "ประนีประนอม" ระหว่างสองโหมด ทำให้ R² ติด ceiling
- Two-stage แยกปัญหา: จัดการ low/high แยกกัน → ขจัด ceiling นั้น

Usage:
  python scripts/train_two_stage.py \\
    --npz results/out_feature_sets_w30_h1/context/sequences_window30_h1.npz \\
    --out_dir results/out_train_two_stage_w30_h1_context \\
    --spike_quantile 0.75 \\
    --epochs_s1 120 --epochs_s2 150 \\
    --batch_size 128 --lr 0.0005
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from train_utils import (
    set_seed, try_enable_tf_memory_growth,
    load_sequences_npz, make_out_dir,
    compute_metrics, compute_smape, inverse_log1p, assert_all_finite,
    save_predictions_csv, pick_station_for_plot, subset_by_station,
    save_json, write_text_report,
    plot_training_history, plot_scatter_true_pred,
    plot_residual_hist, plot_timeseries_one_station,
)


# ============================================================
# Stage 1: Spike Classifier (CNN-LSTM + sigmoid)
# ============================================================

def build_classifier(window: int, n_features: int,
                     conv_filters: int = 64, kernel_size: int = 5,
                     lstm_units: int = 64, dropout: float = 0.25,
                     lr: float = 0.0005, clipnorm: float = 1.0):
    """
    CNN-LSTM classifier: output = P(spike) ∈ [0,1]
    Loss: Binary Crossentropy (class-weighted ใน fit)
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models, optimizers

    inp = layers.Input(shape=(window, n_features))
    x = layers.Conv1D(conv_filters, kernel_size, padding="same", activation="relu")(inp)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(dropout)(x)
    x = layers.LSTM(lstm_units, dropout=dropout)(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)   # probability of spike

    model = models.Model(inp, out)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr,
                                  clipnorm=clipnorm if clipnorm > 0 else None),
        loss="binary_crossentropy",
        metrics=["accuracy",
                 tf.keras.metrics.AUC(name="auc"),
                 tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")],
    )
    return model


# ============================================================
# Stage 2: Spike Regressor (CNN-LSTM + linear — ทำนาย log1p BPH)
# ============================================================

def build_regressor(window: int, n_features: int,
                    conv_filters: int = 64, kernel_size: int = 5,
                    lstm_units: int = 64, dropout: float = 0.25,
                    lr: float = 0.0003, clipnorm: float = 1.0):
    """
    CNN-LSTM regressor เหมือน best model เดิม แต่เทรนเฉพาะ spike samples
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models, optimizers

    inp = layers.Input(shape=(window, n_features))
    x = layers.Conv1D(conv_filters, kernel_size, padding="same", activation="relu")(inp)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Dropout(dropout)(x)
    x = layers.LSTM(lstm_units, dropout=dropout)(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(1)(x)

    model = models.Model(inp, out)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr,
                                  clipnorm=clipnorm if clipnorm > 0 else None),
        loss=tf.keras.losses.Huber(),
    )
    return model


# ============================================================
# Two-stage inference: รวม Stage 1 + Stage 2
# ============================================================

def two_stage_predict(clf, reg, X: np.ndarray,
                      spike_threshold: float = 0.5,
                      spike_floor: float = 0.0) -> np.ndarray:
    """
    รวม prediction ของทั้งสอง stage:
    - ถ้า P(spike) >= threshold → ใช้ค่าจาก regressor
    - ถ้า P(spike) < threshold  → ใช้ spike_floor (log1p scale ≈ log1p(0) = 0)
    คืน array รูป (n,) ใน log1p scale
    """
    prob = clf.predict(X, verbose=0).flatten()          # P(spike)
    reg_pred = reg.predict(X, verbose=0).flatten()      # log1p BPH

    pred = np.where(prob >= spike_threshold, reg_pred, spike_floor)
    pred = np.maximum(pred, 0.0)                        # clamp ไม่ให้ติดลบ
    return pred, prob


# ============================================================
# Metrics สำหรับ classifier
# ============================================================

def classifier_metrics(y_true_bin: np.ndarray,
                       y_prob: np.ndarray,
                       threshold: float = 0.5) -> dict:
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score, confusion_matrix,
    )
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true_bin, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (cm[0, 0], 0, 0, 0)

    return {
        "accuracy":     float(accuracy_score(y_true_bin, y_pred)),
        "precision":    float(precision_score(y_true_bin, y_pred, zero_division=0)),
        "recall":       float(recall_score(y_true_bin, y_pred, zero_division=0)),
        "f1":           float(f1_score(y_true_bin, y_pred, zero_division=0)),
        "auc_roc":      float(roc_auc_score(y_true_bin, y_prob)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "spike_threshold": threshold,
    }


# ============================================================
# Main training loop
# ============================================================

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="BPH Two-stage Model Trainer")
    p.add_argument("--npz",        required=True,  help="path to .npz sequences file")
    p.add_argument("--out_dir",    required=True,  help="output directory")

    # Spike definition
    p.add_argument("--spike_quantile", type=float, default=0.75,
                   help="quantile ของ y_train (log1p) ที่ถือว่าเป็น spike (default 0.75)")
    p.add_argument("--spike_threshold", type=float, default=0.5,
                   help="probability threshold สำหรับ classify spike (default 0.5)")

    # Stage 1 hyperparameters
    p.add_argument("--epochs_s1",    type=int,   default=120)
    p.add_argument("--patience_s1",  type=int,   default=20)

    # Stage 2 hyperparameters
    p.add_argument("--epochs_s2",    type=int,   default=150)
    p.add_argument("--patience_s2",  type=int,   default=25)

    # Shared hyperparameters
    p.add_argument("--batch_size",   type=int,   default=128)
    p.add_argument("--lr",           type=float, default=0.0005)
    p.add_argument("--lr_s2",        type=float, default=0.0003,
                   help="lr ของ Stage 2 (ค่าต่ำกว่าเพราะเทรนบน subset)")
    p.add_argument("--conv_filters", type=int,   default=64)
    p.add_argument("--kernel_size",  type=int,   default=5)
    p.add_argument("--lstm_units",   type=int,   default=64)
    p.add_argument("--dropout",      type=float, default=0.25)
    p.add_argument("--clipnorm",     type=float, default=1.0)

    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--plot_station", default="")
    return p.parse_args(argv)


def main(argv=None):
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

    args = parse_args(argv)
    set_seed(args.seed)
    try_enable_tf_memory_growth()

    out_dir = make_out_dir(args.out_dir)
    print(f"\n{'='*60}")
    print(f"  BPH Two-stage Model")
    print(f"  NPZ      : {args.npz}")
    print(f"  Output   : {out_dir}")
    print(f"  Spike Q  : {args.spike_quantile} (top {(1-args.spike_quantile)*100:.0f}%)")
    print(f"{'='*60}\n")

    # ---- Load data ----
    data = load_sequences_npz(args.npz)
    X_train, y_train = data.X_train, data.y_train.flatten()
    X_val,   y_val   = data.X_val,   data.y_val.flatten()
    X_test,  y_test  = data.X_test,  data.y_test.flatten()
    assert_all_finite(X_train, "X_train")

    window, n_features = X_train.shape[1], X_train.shape[2]
    print(f"  Data     : train={len(X_train)} val={len(X_val)} test={len(X_test)}")
    print(f"  Shape    : ({window}, {n_features})\n")

    # ============================================================
    # กำหนด spike label จาก quantile ของ y_train
    # ============================================================
    spike_cutoff = float(np.quantile(y_train, args.spike_quantile))
    print(f"  Spike cutoff (Q{args.spike_quantile:.0%}) = {spike_cutoff:.4f} "
          f"(raw ≈ {inverse_log1p(spike_cutoff):.1f} ตัว/กับดัก)")

    y_bin_train = (y_train >= spike_cutoff).astype(np.float32)
    y_bin_val   = (y_val   >= spike_cutoff).astype(np.float32)
    y_bin_test  = (y_test  >= spike_cutoff).astype(np.float32)

    spike_ratio_train = y_bin_train.mean()
    print(f"  Spike ratio (train) : {spike_ratio_train:.1%}  "
          f"({y_bin_train.sum():.0f}/{len(y_bin_train)} samples)")

    # class weight สำหรับ imbalanced binary classification
    neg_weight = 1.0
    pos_weight = (1.0 - spike_ratio_train) / max(spike_ratio_train, 1e-6)
    class_weight = {0: neg_weight, 1: pos_weight}
    print(f"  Class weights: {{0: {neg_weight:.2f}, 1: {pos_weight:.2f}}}\n")

    # ============================================================
    # Stage 1 — Train Classifier
    # ============================================================
    print(f"{'─'*60}")
    print(f"  STAGE 1: Spike Classifier")
    print(f"{'─'*60}")

    clf = build_classifier(
        window=window, n_features=n_features,
        conv_filters=args.conv_filters, kernel_size=args.kernel_size,
        lstm_units=args.lstm_units, dropout=args.dropout,
        lr=args.lr, clipnorm=args.clipnorm,
    )
    clf.summary(print_fn=lambda x: print(f"  {x}"))

    clf_ckpt = str(out_dir / "stage1_classifier.keras")
    history_s1 = clf.fit(
        X_train, y_bin_train,
        validation_data=(X_val, y_bin_val),
        epochs=args.epochs_s1,
        batch_size=args.batch_size,
        class_weight=class_weight,
        callbacks=[
            EarlyStopping(patience=args.patience_s1, restore_best_weights=True,
                          monitor="val_auc", mode="max"),
            ModelCheckpoint(clf_ckpt, save_best_only=True,
                            monitor="val_auc", mode="max"),
        ],
        verbose=1,
    )

    # Eval classifier
    y_prob_test = clf.predict(X_test, verbose=0).flatten()
    clf_metrics = classifier_metrics(y_bin_test, y_prob_test, args.spike_threshold)
    print(f"\n  Classifier (test):")
    for k, v in clf_metrics.items():
        print(f"    {k}: {v}")

    plot_training_history(history_s1.history, out_dir / "stage1_loss_curve.png")

    # ============================================================
    # Stage 2 — Train Regressor (spike samples เท่านั้น)
    # ============================================================
    print(f"\n{'─'*60}")
    print(f"  STAGE 2: Spike Regressor (train on spike samples only)")
    print(f"{'─'*60}")

    # กรองเฉพาะ spike samples
    spike_mask_train = y_bin_train == 1
    spike_mask_val   = y_bin_val   == 1
    X_spike_train = X_train[spike_mask_train]
    y_spike_train = y_train[spike_mask_train]
    X_spike_val   = X_val[spike_mask_val]
    y_spike_val   = y_val[spike_mask_val]

    print(f"  Spike samples: train={len(X_spike_train)} val={len(X_spike_val)}")

    reg = build_regressor(
        window=window, n_features=n_features,
        conv_filters=args.conv_filters, kernel_size=args.kernel_size,
        lstm_units=args.lstm_units, dropout=args.dropout,
        lr=args.lr_s2, clipnorm=args.clipnorm,
    )

    reg_ckpt = str(out_dir / "stage2_regressor.keras")
    history_s2 = reg.fit(
        X_spike_train, y_spike_train,
        validation_data=(X_spike_val, y_spike_val),
        epochs=args.epochs_s2,
        batch_size=min(args.batch_size, len(X_spike_train)),
        callbacks=[
            EarlyStopping(patience=args.patience_s2, restore_best_weights=True,
                          monitor="val_loss", mode="min"),
            ModelCheckpoint(reg_ckpt, save_best_only=True,
                            monitor="val_loss", mode="min"),
        ],
        verbose=1,
    )

    plot_training_history(history_s2.history, out_dir / "stage2_loss_curve.png")

    # ============================================================
    # Two-stage Inference บน Test set ทั้งหมด
    # ============================================================
    print(f"\n{'─'*60}")
    print(f"  TWO-STAGE INFERENCE (full test set)")
    print(f"{'─'*60}")

    y_pred_log1p, y_prob = two_stage_predict(
        clf, reg, X_test,
        spike_threshold=args.spike_threshold,
        spike_floor=0.0,
    )

    # Metrics ใน log1p space
    metrics_log1p = compute_metrics(y_test, y_pred_log1p)
    metrics_log1p["smape"] = compute_smape(y_test, y_pred_log1p)

    # Metrics ใน raw space
    y_test_raw = inverse_log1p(y_test)
    y_pred_raw = inverse_log1p(y_pred_log1p)
    metrics_raw = compute_metrics(y_test_raw, y_pred_raw)
    metrics_raw["smape"] = compute_smape(y_test_raw, y_pred_raw)

    # Metrics เฉพาะ spike samples (top-25% ของ test)
    spike_mask_test = y_bin_test == 1
    if spike_mask_test.sum() > 0:
        spike_met = compute_metrics(y_test[spike_mask_test],
                                    y_pred_log1p[spike_mask_test])
        spike_met_raw = compute_metrics(y_test_raw[spike_mask_test],
                                        y_pred_raw[spike_mask_test])
    else:
        spike_met = spike_met_raw = {}

    print(f"\n  Regression metrics (log1p):")
    for k, v in metrics_log1p.items():
        print(f"    {k}: {v:.4f}")
    print(f"\n  Regression metrics (raw BPH count):")
    for k, v in metrics_raw.items():
        print(f"    {k}: {v:.4f}")
    if spike_met:
        print(f"\n  Spike-only metrics (log1p, top {(1-args.spike_quantile)*100:.0f}%):")
        for k, v in spike_met.items():
            print(f"    {k}: {v:.4f}")

    # ============================================================
    # เปรียบเทียบกับ single-stage baseline (ถ้ามี)
    # ============================================================
    baseline_path = Path(args.npz).parent.parent.parent
    baseline_candidates = list(baseline_path.glob("*/cnn_lstm_*/metrics.json"))
    single_stage_r2 = None
    if baseline_candidates:
        try:
            bm = json.loads(baseline_candidates[0].read_text())
            single_stage_r2 = bm.get("r2_log1p") or bm.get("test", {}).get("r2_log1p")
        except Exception:
            pass

    # ============================================================
    # บันทึกผล
    # ============================================================
    all_metrics = {
        "model": "two_stage_cnn_lstm",
        "stage1_classifier": clf_metrics,
        "log1p": metrics_log1p,
        "raw":   metrics_raw,
        "spike_only_log1p": spike_met,
        "spike_only_raw":   spike_met_raw,
        "config": {
            "spike_quantile":  args.spike_quantile,
            "spike_cutoff":    spike_cutoff,
            "spike_threshold": args.spike_threshold,
            "spike_ratio_train": float(spike_ratio_train),
            "window": window, "n_features": n_features,
        },
        "improvement_vs_single_stage": {
            "single_stage_r2_log1p": single_stage_r2,
            "two_stage_r2_log1p":    metrics_log1p.get("r2"),
            "delta_r2": (metrics_log1p.get("r2", 0) - single_stage_r2)
                        if single_stage_r2 else None,
        },
    }
    save_json(out_dir / "metrics.json", all_metrics)

    # ============================================================
    # Plots
    # ============================================================
    plot_scatter_true_pred(y_test, y_pred_log1p,
                           out_dir / "scatter_two_stage_log1p.png",
                           title="Two-stage: True vs Predicted (log1p)")
    plot_scatter_true_pred(y_test_raw, y_pred_raw,
                           out_dir / "scatter_two_stage_raw.png",
                           title="Two-stage: True vs Predicted (raw BPH)")
    plot_residual_hist(y_test, y_pred_log1p,
                       out_dir / "residual_two_stage.png",
                       title="Two-stage: Residuals (log1p)")

    # Time-series plot รายสถานี
    station_col = data.station_test
    plot_station = args.plot_station or pick_station_for_plot(station_col)
    if plot_station:
        df_st = subset_by_station(y_test, y_pred_log1p, data.y_date_test, station_col, plot_station)
        if len(df_st) > 0:
            plot_timeseries_one_station(
                df_st,
                out_dir / f"timeseries_{plot_station}.png",
                f"Two-stage: Station {plot_station} (log1p)",
            )

    # ============================================================
    # Report
    # ============================================================
    delta_r2 = all_metrics["improvement_vs_single_stage"]["delta_r2"]
    delta_str = (f"+{delta_r2:.4f}" if delta_r2 and delta_r2 > 0
                 else f"{delta_r2:.4f}" if delta_r2 else "N/A")

    report = f"""# Two-stage BPH Model — Results

## Configuration
- NPZ: {args.npz}
- Spike quantile: Q{args.spike_quantile:.0%} = {spike_cutoff:.4f} (raw ≈ {inverse_log1p(spike_cutoff):.1f} ตัว)
- Spike ratio (train): {spike_ratio_train:.1%}
- Spike threshold (inference): {args.spike_threshold}

## Stage 1: Classifier
| Metric     | Value  |
|------------|--------|
| Accuracy   | {clf_metrics.get('accuracy', 0):.4f} |
| Precision  | {clf_metrics.get('precision', 0):.4f} |
| Recall     | {clf_metrics.get('recall', 0):.4f} |
| F1         | {clf_metrics.get('f1', 0):.4f} |
| AUC-ROC    | {clf_metrics.get('auc_roc', 0):.4f} |
| TP / FP / FN / TN | {clf_metrics.get('tp')} / {clf_metrics.get('fp')} / {clf_metrics.get('fn')} / {clf_metrics.get('tn')} |

## Stage 2 + Combined: Regression (Test Set)
| Metric       | log1p  | raw BPH |
|-------------|--------|---------|
| R²          | {metrics_log1p.get('r2', 0):.4f} | {metrics_raw.get('r2', 0):.4f} |
| MAE         | {metrics_log1p.get('mae', 0):.4f} | {metrics_raw.get('mae', 0):.4f} |
| RMSE        | {metrics_log1p.get('rmse', 0):.4f} | {metrics_raw.get('rmse', 0):.4f} |
| SMAPE       | {metrics_log1p.get('smape', 0):.4f} | {metrics_raw.get('smape', 0):.4f} |

## เปรียบเทียบกับ Single-stage
| | Single-stage CNN-LSTM | Two-stage |
|---|---|---|
| R² (log1p) | {single_stage_r2 or 'N/A'} | {metrics_log1p.get('r2', 0):.4f} |
| Delta R²   | — | **{delta_str}** |

## Spike-only Performance (top {(1-args.spike_quantile)*100:.0f}%)
| Metric | log1p | raw |
|--------|-------|-----|
| R²     | {spike_met.get('r2', 0):.4f} | {spike_met_raw.get('r2', 0):.4f} |
| RMSE   | {spike_met.get('rmse', 0):.4f} | {spike_met_raw.get('rmse', 0):.4f} |
"""
    write_text_report(out_dir / "REPORT.md", report.splitlines())
    print(f"\n  Report saved → {out_dir / 'REPORT.md'}")
    print(f"\n  ╔══════════════════════════════════════════╗")
    print(f"  ║  Two-stage R² (log1p) = {metrics_log1p.get('r2', 0):.4f}              ║")
    print(f"  ║  vs Single-stage      = {single_stage_r2 or 'N/A'}              ║")
    print(f"  ║  Delta R²             = {delta_str}              ║")
    print(f"  ╚══════════════════════════════════════════╝\n")


if __name__ == "__main__":
    main()
