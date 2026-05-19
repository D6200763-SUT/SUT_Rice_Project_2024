#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_cnn_lstm_weighted.py
CNN-LSTM regression with spike-weighted sample loss.

Same architecture as train_cnn_lstm.py (Optuna best params).
Only difference: sample_weight boosts spike samples during training.
Trains one run per spike_weight value, then writes sweep_summary.csv.

Usage:
/home/ai-station/my_project/.tf251p310/bin/python scripts/train_cnn_lstm_weighted.py \
  --npz results/out_feature_sets_w30_h7/context/sequences_window30_h7.npz \
  --stage1_threshold_json results/out_train_stage1/threshold.json \
  --out_dir results/out_train_weighted \
  --spike_weights 2.0 3.0 5.0 8.0 \
  --epochs 200 --patience 35 --seed 42
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
    assert_all_finite, save_json, write_text_report,
    compute_metrics, compute_smape, inverse_log1p,
    save_predictions_csv, pick_station_for_plot, subset_by_station,
    plot_training_history, plot_scatter_true_pred,
    plot_residual_hist, plot_timeseries_one_station,
)

BASELINE = {"r2_log1p": 0.484, "r2_raw": 0.0041, "rmse_raw": 219.51}


# ── Architecture (identical to train_cnn_lstm.py) ─────────────────────────────

def build_model(window: int, n_features: int,
                conv_filters: int, kernel_size: int,
                lstm_units: int, dropout: float,
                lr: float, clipnorm: float):
    import tensorflow as tf
    from tensorflow.keras import layers, models, optimizers

    inp = layers.Input(shape=(window, n_features))
    x = layers.Conv1D(filters=conv_filters, kernel_size=kernel_size,
                      padding="same", activation="relu")(inp)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(dropout)(x)
    x = layers.LSTM(lstm_units, dropout=dropout)(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(1)(x)

    opt = optimizers.Adam(learning_rate=lr, clipnorm=clipnorm if clipnorm > 0 else None)
    model = models.Model(inp, out)
    model.compile(optimizer=opt, loss=tf.keras.losses.Huber())
    return model


# ── Spike-specific metrics ────────────────────────────────────────────────────

def spike_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                  threshold: float) -> dict:
    """R² and RMSE on samples where y_true > threshold (spike region only)."""
    mask = y_true > threshold
    n = int(mask.sum())
    if n == 0:
        return {"r2": float("nan"), "rmse": float("nan"), "n_samples": 0}
    yt = y_true[mask]; yp = y_pred[mask]
    m = compute_metrics(yt, yp)
    return {"r2": m["r2"], "rmse": m["rmse"], "mae": m["mae"], "n_samples": n}


# ── Single run ────────────────────────────────────────────────────────────────

def run_one(spike_weight: float, args, seq, spike_threshold: float,
            Xtr, ytr, Xva, yva, Xte, yte) -> dict:
    """Train one CNN-LSTM with given spike_weight; return metrics dict."""
    import tensorflow as tf
    from tensorflow.keras.callbacks import (
        EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TerminateOnNaN,
    )

    tag = f"weight_{spike_weight:.1f}".replace(".", "p")
    out_dir = make_out_dir(Path(args.out_dir) / tag)

    print(f"\n{'='*60}")
    print(f"[Weighted] spike_weight={spike_weight:.1f}  → {out_dir.name}")
    print(f"{'='*60}")

    set_seed(args.seed)

    window     = int(Xtr.shape[1])
    n_features = int(Xtr.shape[2])

    # Sample weights — spike samples get spike_weight, rest get 1.0
    spike_mask_tr = (ytr > spike_threshold)
    sample_weight = np.where(spike_mask_tr, spike_weight, 1.0).astype(np.float32)
    n_spike = int(spike_mask_tr.sum())
    print(f"[Weighted] train spikes: {n_spike}/{len(ytr)} ({n_spike/len(ytr)*100:.1f}%)  "
          f"weight={spike_weight:.1f}")

    model = build_model(
        window, n_features,
        conv_filters=args.conv_filters,
        kernel_size=args.kernel_size,
        lstm_units=args.lstm_units,
        dropout=args.dropout,
        lr=args.lr,
        clipnorm=args.clipnorm,
    )

    ckpt_path = out_dir / "model_best.keras"
    callbacks = [
        TerminateOnNaN(),
        EarlyStopping(monitor="val_loss", patience=args.patience, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=max(2, args.patience // 4), min_lr=1e-6),
        ModelCheckpoint(filepath=str(ckpt_path), monitor="val_loss", save_best_only=True),
    ]

    hist = model.fit(
        Xtr, ytr,
        validation_data=(Xva, yva),
        sample_weight=sample_weight,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=2,
        callbacks=callbacks,
    )

    # ── Predictions ───────────────────────────────────────────────────────────
    yhat_tr = model.predict(Xtr, batch_size=args.batch_size, verbose=0).reshape(-1)
    yhat_va = model.predict(Xva, batch_size=args.batch_size, verbose=0).reshape(-1)
    yhat_te = model.predict(Xte, batch_size=args.batch_size, verbose=0).reshape(-1)

    # ── Standard metrics ──────────────────────────────────────────────────────
    m_tr = compute_metrics(ytr, yhat_tr)
    m_va = compute_metrics(yva, yhat_va)
    m_te = compute_metrics(yte, yhat_te)
    m_te["smape"] = compute_smape(yte, yhat_te)

    yte_raw   = inverse_log1p(yte)
    yhat_raw  = inverse_log1p(yhat_te)
    m_te_raw  = compute_metrics(yte_raw, yhat_raw)
    m_te_raw["smape"] = compute_smape(yte_raw, yhat_raw)

    # ── Spike-specific metrics ────────────────────────────────────────────────
    m_spike_val  = spike_metrics(yva,  yhat_va,  spike_threshold)
    m_spike_test = spike_metrics(yte,  yhat_te,  spike_threshold)

    metrics = {
        "model":          "CNN-LSTM-Weighted",
        "spike_weight":   spike_weight,
        "spike_threshold": spike_threshold,
        "n_train_spikes": n_spike,
        "window":         int(window),
        "n_features":     int(n_features),
        "train":          m_tr,
        "val":            m_va,
        "test_log1p":     m_te,
        "test_raw":       m_te_raw,
        "spike_only_val":  m_spike_val,
        "spike_only_test": m_spike_test,
        "hyperparams":    vars(args),
        "beat_baseline": {
            "r2_log1p": m_te["r2"]     > BASELINE["r2_log1p"],
            "r2_raw":   m_te_raw["r2"] > BASELINE["r2_raw"],
            "rmse_raw": m_te_raw["rmse"] < BASELINE["rmse_raw"],
        },
    }
    save_json(out_dir / "metrics.json", metrics)

    # ── Predictions CSV ───────────────────────────────────────────────────────
    save_predictions_csv(out_dir, "train", ytr, yhat_tr,
                         seq.y_date_train, seq.station_train, also_raw=True)
    save_predictions_csv(out_dir, "val",   yva, yhat_va,
                         seq.y_date_val,   seq.station_val,   also_raw=True)
    save_predictions_csv(out_dir, "test",  yte, yhat_te,
                         seq.y_date_test,  seq.station_test,  also_raw=True)

    # ── Figures ───────────────────────────────────────────────────────────────
    figs = out_dir / "figures"
    plot_training_history(hist.history,  figs / "loss_curve.png")
    plot_scatter_true_pred(yte, yhat_te, figs / "scatter_log1p.png",
                           f"Test true vs pred (log1p)  spike_w={spike_weight:.1f}")
    plot_residual_hist(yte, yhat_te,     figs / "residual_hist.png",
                       f"Test residuals (log1p)  spike_w={spike_weight:.1f}")

    st = pick_station_for_plot(seq.station_test)
    if st and len(seq.y_date_test) == len(yte):
        df_st = subset_by_station(yte, yhat_te, seq.y_date_test, seq.station_test, st)
        if len(df_st) > 0:
            plot_timeseries_one_station(
                df_st, figs / f"timeseries_{st}.png",
                f"Test station={st}  spike_w={spike_weight:.1f}",
            )

    # ── Report ────────────────────────────────────────────────────────────────
    def _beat(flag): return "BETTER" if flag else "WORSE"

    write_text_report(out_dir / "REPORT.md", [
        f"# CNN-LSTM Weighted Loss  (spike_weight={spike_weight:.1f})",
        "",
        f"spike_threshold (log1p): {spike_threshold:.4f}",
        f"Train spikes: {n_spike}/{len(ytr)} ({n_spike/len(ytr)*100:.1f}%)",
        "",
        "## Test Metrics (log1p)",
        f"- R²:   {m_te['r2']:.4f}  ({_beat(m_te['r2'] > BASELINE['r2_log1p'])} "
        f"vs baseline {BASELINE['r2_log1p']})",
        f"- MAE:  {m_te['mae']:.4f}",
        f"- RMSE: {m_te['rmse']:.4f}",
        f"- sMAPE:{m_te['smape']:.4f}",
        "",
        "## Test Metrics (raw)",
        f"- R²:   {m_te_raw['r2']:.4f}  ({_beat(m_te_raw['r2'] > BASELINE['r2_raw'])} "
        f"vs baseline {BASELINE['r2_raw']})",
        f"- MAE:  {m_te_raw['mae']:.2f}",
        f"- RMSE: {m_te_raw['rmse']:.2f}  ({_beat(m_te_raw['rmse'] < BASELINE['rmse_raw'])} "
        f"vs baseline {BASELINE['rmse_raw']})",
        "",
        "## Spike-only Test Metrics (y_true > threshold)",
        f"- R²:   {m_spike_test['r2']:.4f}",
        f"- RMSE: {m_spike_test['rmse']:.4f}",
        f"- MAE:  {m_spike_test['mae']:.4f}",
        f"- n:    {m_spike_test['n_samples']}",
    ])

    model.save(out_dir / "model_final.keras")

    r2_log = m_te["r2"]; r2_raw = m_te_raw["r2"]; rmse_raw = m_te_raw["rmse"]
    print(f"[Done spike_w={spike_weight:.1f}]  "
          f"R²_log1p={r2_log:.4f} {'✓' if r2_log > BASELINE['r2_log1p'] else '✗'}  "
          f"R²_raw={r2_raw:.4f} {'✓' if r2_raw > BASELINE['r2_raw'] else '✗'}  "
          f"RMSE_raw={rmse_raw:.2f} {'✓' if rmse_raw < BASELINE['rmse_raw'] else '✗'}")

    return {
        "spike_weight":      spike_weight,
        "r2_log1p":          r2_log,
        "r2_raw":            r2_raw,
        "rmse_log1p":        m_te["rmse"],
        "mae_log1p":         m_te["mae"],
        "rmse_raw":          rmse_raw,
        "mae_raw":           m_te_raw["mae"],
        "r2_spike_only":     m_spike_test["r2"],
        "rmse_spike_only":   m_spike_test["rmse"],
        "n_spike_test":      m_spike_test["n_samples"],
        "beat_baseline_log1p": m_te["r2"]     > BASELINE["r2_log1p"],
        "beat_baseline_raw":   m_te_raw["r2"] > BASELINE["r2_raw"],
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="CNN-LSTM with spike-weighted loss sweep")
    p.add_argument("--npz",                    required=True)
    p.add_argument("--stage1_threshold_json",  required=True,
                   help="Path to threshold.json from Stage 1")
    p.add_argument("--out_dir",                required=True)
    p.add_argument("--spike_weights",          type=float, nargs="+",
                   default=[2.0, 3.0, 5.0, 8.0])
    p.add_argument("--epochs",                 type=int,   default=200)
    p.add_argument("--patience",               type=int,   default=35)
    p.add_argument("--batch_size",             type=int,   default=256)
    # Optuna best hyperparams as defaults
    p.add_argument("--lr",                     type=float, default=0.0004329)
    p.add_argument("--conv_filters",           type=int,   default=128)
    p.add_argument("--kernel_size",            type=int,   default=5)
    p.add_argument("--lstm_units",             type=int,   default=32)
    p.add_argument("--dropout",                type=float, default=0.1094)
    p.add_argument("--clipnorm",               type=float, default=2.0)
    p.add_argument("--seed",                   type=int,   default=42)
    return p.parse_args(argv)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(argv=None):
    args = parse_args(argv)
    try_enable_tf_memory_growth()

    # ── Load data once ────────────────────────────────────────────────────────
    seq = load_sequences_npz(args.npz)

    Xtr, ytr = seq.X_train, seq.y_train.reshape(-1)
    Xva, yva = seq.X_val,   seq.y_val.reshape(-1)
    Xte, yte = seq.X_test,  seq.y_test.reshape(-1)

    assert_all_finite("X_train", Xtr); assert_all_finite("y_train", ytr)
    assert_all_finite("X_val",   Xva); assert_all_finite("y_val",   yva)
    assert_all_finite("X_test",  Xte); assert_all_finite("y_test",  yte)

    # ── Spike threshold ───────────────────────────────────────────────────────
    with open(args.stage1_threshold_json) as f:
        spike_threshold = json.load(f)["spike_threshold"]
    print(f"[Weighted] spike_threshold (log1p): {spike_threshold:.4f}")
    print(f"[Weighted] spike_weights to sweep: {args.spike_weights}")
    print(f"[Weighted] Baseline to beat: R²_log1p={BASELINE['r2_log1p']}  "
          f"R²_raw={BASELINE['r2_raw']}  RMSE_raw={BASELINE['rmse_raw']}")

    # ── Sweep ─────────────────────────────────────────────────────────────────
    summary_rows = []
    for w in args.spike_weights:
        row = run_one(w, args, seq, spike_threshold, Xtr, ytr, Xva, yva, Xte, yte)
        summary_rows.append(row)

    # ── sweep_summary.csv ─────────────────────────────────────────────────────
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(summary_rows)
    csv_path = out_root / "sweep_summary.csv"
    df.to_csv(csv_path, index=False, float_format="%.6f")

    # ── Best runs ─────────────────────────────────────────────────────────────
    best_log1p = summary_rows[int(df["r2_log1p"].idxmax())]
    best_raw   = summary_rows[int(df["r2_raw"].idxmax())]

    save_json(out_root / "sweep_best.json", {
        "baseline":           BASELINE,
        "best_by_r2_log1p":   best_log1p,
        "best_by_r2_raw":     best_raw,
    })

    # ── Final table ───────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print(f"{'weight':>8}  {'R²_log1p':>10}  {'R²_raw':>8}  "
          f"{'RMSE_raw':>9}  {'R²_spike':>9}  {'beat_log1p':>10}  {'beat_raw':>8}")
    print("-" * 78)
    for r in summary_rows:
        print(f"{r['spike_weight']:>8.1f}  {r['r2_log1p']:>10.4f}  {r['r2_raw']:>8.4f}  "
              f"{r['rmse_raw']:>9.2f}  {r['r2_spike_only']:>9.4f}  "
              f"{'YES' if r['beat_baseline_log1p'] else 'no':>10}  "
              f"{'YES' if r['beat_baseline_raw'] else 'no':>8}")
    print("-" * 78)
    print(f"{'baseline':>8}  {BASELINE['r2_log1p']:>10.4f}  {BASELINE['r2_raw']:>8.4f}  "
          f"{BASELINE['rmse_raw']:>9.2f}")
    print("=" * 78)
    print(f"Best R²_log1p: spike_weight={best_log1p['spike_weight']:.1f}  "
          f"R²={best_log1p['r2_log1p']:.4f}")
    print(f"Best R²_raw  : spike_weight={best_raw['spike_weight']:.1f}  "
          f"R²={best_raw['r2_raw']:.4f}")
    print(f"[OK] sweep_summary.csv → {csv_path}")


if __name__ == "__main__":
    main()
