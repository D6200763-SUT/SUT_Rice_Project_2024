#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_stage2_regression.py
Stage 2 of Two-stage BPH model: CNN-LSTM regression on spike samples only.

Pipeline:
1. Load NPZ + Stage 1 model
2. Run Stage 1 → binary mask (spike / no-spike)
3. Filter to spike samples for Stage 2 training
4. Train CNN-LSTM regression (Huber loss) — Optuna best hyperparams
5. Two-stage inference: Stage1=0 → 0.0 | Stage1=1 → Stage2_pred
6. Compute full metrics vs single-stage baseline

Usage:
/home/ai-station/my_project/.tf251p310/bin/python scripts/train_stage2_regression.py \
  --npz results/out_feature_sets_w30_h7/context/sequences_window30_h7.npz \
  --stage1_dir results/out_train_stage1 \
  --out_dir results/out_train_stage2 \
  --epochs 200 --patience 35 --seed 42
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from train_utils import (
    set_seed, try_enable_tf_memory_growth,
    load_sequences_npz, make_out_dir,
    assert_all_finite, save_json, write_text_report,
    compute_metrics, compute_smape, inverse_log1p,
    plot_training_history,
)

# Single-stage CNN-LSTM baseline (W30 H7 context) to beat
BASELINE = {"r2_log1p": 0.4798, "r2_raw": 0.0041, "rmse_raw": 219.51}


# ── Architecture ──────────────────────────────────────────────────────────────

def build_stage2_model(window: int, n_features: int,
                       conv_filters: int = 128, kernel_size: int = 5,
                       lstm_units: int = 32, dropout: float = 0.1094,
                       lr: float = 0.0004329, clipnorm: float = 2.0):
    """CNN-LSTM regression — same backbone as best single-stage model."""
    import tensorflow as tf
    from tensorflow.keras import layers, models, optimizers

    inp = layers.Input(shape=(window, n_features))
    x = layers.Conv1D(conv_filters, kernel_size, padding="same", activation="relu")(inp)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(dropout)(x)
    x = layers.LSTM(lstm_units, dropout=dropout)(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(1)(x)

    model = models.Model(inp, out)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr, clipnorm=clipnorm),
        loss=tf.keras.losses.Huber(),
    )
    return model


# ── Stage 1 helpers ───────────────────────────────────────────────────────────

def load_stage1_model(stage1_dir: Path):
    import tensorflow as tf
    ckpt = stage1_dir / "model_best.keras"
    if not ckpt.exists():
        raise FileNotFoundError(f"Stage 1 model not found: {ckpt}")
    print(f"[Stage2] Loading Stage 1 model from {ckpt}")
    return tf.keras.models.load_model(str(ckpt))


def get_spike_mask(stage1_model, X: np.ndarray,
                   batch_size: int = 256, pred_threshold: float = 0.5) -> np.ndarray:
    """Boolean mask — True where Stage 1 probability >= pred_threshold."""
    probs = stage1_model.predict(X, batch_size=batch_size, verbose=0).reshape(-1)
    return probs >= pred_threshold


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_scatter_full(y_true: np.ndarray, y_pred: np.ndarray,
                      out_png: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_true, y_pred, s=5, alpha=0.35, rasterized=True)
    lo = min(y_true.min(), y_pred.min()); hi = max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "r--", lw=1, label="y=x")
    ax.set_xlabel("True (log1p)"); ax.set_ylabel("Pred (log1p)")
    ax.set_title(title); ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def plot_timeseries_combined(df: pd.DataFrame, station_id: str,
                              out_png: Path, title: str) -> None:
    """True vs two-stage pred for one station; orange ticks = Stage 1 spike predictions."""
    import matplotlib.pyplot as plt

    sub = df[df["station_id"] == station_id].sort_values("date")
    if len(sub) == 0:
        return

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(sub["date"], sub["y_true_log1p"], label="True", color="steelblue", lw=1.5)
    ax.plot(sub["date"], sub["y_pred_final_log1p"],
            label="Two-stage Pred", color="tomato", lw=1.5, linestyle="--")

    spike_dates = sub.loc[sub["stage1_label"] == 1, "date"]
    for d in spike_dates:
        ax.axvline(d, color="orange", alpha=0.18, lw=0.7)

    ax.set_title(f"{title}  |  station={station_id}")
    ax.set_xlabel("date"); ax.set_ylabel("BPH log1p")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Stage 2: BPH spike regression (CNN-LSTM)")
    p.add_argument("--npz",            required=True,           help="Path to sequences NPZ")
    p.add_argument("--stage1_dir",     required=True,           help="Stage 1 output directory")
    p.add_argument("--out_dir",        required=True,           help="Stage 2 output directory")
    p.add_argument("--epochs",         type=int,   default=200)
    p.add_argument("--patience",       type=int,   default=35)
    p.add_argument("--batch_size",     type=int,   default=256)
    p.add_argument("--lr",             type=float, default=0.0004329)
    p.add_argument("--conv_filters",   type=int,   default=128)
    p.add_argument("--kernel_size",    type=int,   default=5)
    p.add_argument("--lstm_units",     type=int,   default=32)
    p.add_argument("--dropout",        type=float, default=0.1094)
    p.add_argument("--clipnorm",       type=float, default=2.0)
    p.add_argument("--pred_threshold", type=float, default=0.5,
                   help="Stage 1 probability threshold for spike mask")
    p.add_argument("--seed",           type=int,   default=42)
    return p.parse_args(argv)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(argv=None):
    args = parse_args(argv)
    set_seed(args.seed)
    try_enable_tf_memory_growth()

    stage1_dir = Path(args.stage1_dir)
    out_dir    = make_out_dir(args.out_dir)

    # ── Load data ─────────────────────────────────────────────────────────────
    seq = load_sequences_npz(args.npz)

    Xtr, ytr = seq.X_train, seq.y_train.reshape(-1)
    Xva, yva = seq.X_val,   seq.y_val.reshape(-1)
    Xte, yte = seq.X_test,  seq.y_test.reshape(-1)

    assert_all_finite("X_train", Xtr); assert_all_finite("y_train", ytr)
    assert_all_finite("X_val",   Xva); assert_all_finite("y_val",   yva)
    assert_all_finite("X_test",  Xte); assert_all_finite("y_test",  yte)

    window     = int(Xtr.shape[1])
    n_features = int(Xtr.shape[2])

    # ── Spike threshold from Stage 1 ──────────────────────────────────────────
    with open(stage1_dir / "threshold.json") as f:
        thresh_data = json.load(f)
    spike_threshold = thresh_data["spike_threshold"]
    print(f"[Stage2] spike_threshold (log1p): {spike_threshold:.4f}")

    # ── Stage 1 binary mask ───────────────────────────────────────────────────
    stage1_model = load_stage1_model(stage1_dir)

    mask_tr = get_spike_mask(stage1_model, Xtr, args.batch_size, args.pred_threshold)
    mask_va = get_spike_mask(stage1_model, Xva, args.batch_size, args.pred_threshold)
    mask_te = get_spike_mask(stage1_model, Xte, args.batch_size, args.pred_threshold)

    print(f"[Stage2] Spike mask — "
          f"train: {mask_tr.sum()}/{len(mask_tr)} ({mask_tr.mean()*100:.1f}%)  "
          f"val: {mask_va.sum()}/{len(mask_va)} ({mask_va.mean()*100:.1f}%)  "
          f"test: {mask_te.sum()}/{len(mask_te)} ({mask_te.mean()*100:.1f}%)")

    # ── Filter to spike samples ───────────────────────────────────────────────
    Xtr_sp, ytr_sp = Xtr[mask_tr], ytr[mask_tr]
    Xva_sp, yva_sp = Xva[mask_va], yva[mask_va]

    if len(Xtr_sp) == 0:
        raise RuntimeError("Stage 1 produced no spike predictions on train set. "
                           "Lower --pred_threshold or check Stage 1 model.")

    # Fallback: if val has no spikes, borrow a slice from train for monitoring
    _val_fallback = False
    if len(Xva_sp) == 0:
        print("[Stage2] WARNING: no spike predictions on val — using train tail as fallback val")
        Xva_sp = Xtr_sp[-max(50, len(Xtr_sp) // 10):]
        yva_sp = ytr_sp[-max(50, len(ytr_sp) // 10):]
        _val_fallback = True

    print(f"[Stage2] Training Stage 2 on {len(Xtr_sp)} spike samples (val: {len(Xva_sp)})")

    # ── Build and train Stage 2 ───────────────────────────────────────────────
    import tensorflow as tf
    from tensorflow.keras.callbacks import (
        EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TerminateOnNaN,
    )

    model = build_stage2_model(
        window, n_features,
        conv_filters=args.conv_filters,
        kernel_size=args.kernel_size,
        lstm_units=args.lstm_units,
        dropout=args.dropout,
        lr=args.lr,
        clipnorm=args.clipnorm,
    )
    model.summary()

    ckpt_path = out_dir / "model_best.keras"
    callbacks = [
        TerminateOnNaN(),
        EarlyStopping(monitor="val_loss", patience=args.patience, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=max(2, args.patience // 4), min_lr=1e-6),
        ModelCheckpoint(filepath=str(ckpt_path), monitor="val_loss", save_best_only=True),
    ]

    hist = model.fit(
        Xtr_sp, ytr_sp,
        validation_data=(Xva_sp, yva_sp),
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=2,
        callbacks=callbacks,
    )

    # ── Spike-only metrics ────────────────────────────────────────────────────
    if not _val_fallback and len(Xva_sp) > 0:
        yhat_va_sp = model.predict(Xva_sp, batch_size=args.batch_size, verbose=0).reshape(-1)
        m_val_spike = compute_metrics(yva_sp, yhat_va_sp)
        m_val_spike["smape"] = compute_smape(yva_sp, yhat_va_sp)
        m_val_spike["n_samples"] = int(len(yva_sp))
    else:
        m_val_spike = {"note": "val fallback used — metrics not meaningful"}

    Xte_sp, yte_sp = Xte[mask_te], yte[mask_te]
    if len(Xte_sp) > 0:
        yhat_te_sp = model.predict(Xte_sp, batch_size=args.batch_size, verbose=0).reshape(-1)
        m_test_spike = compute_metrics(yte_sp, yhat_te_sp)
        m_test_spike["smape"] = compute_smape(yte_sp, yhat_te_sp)
        m_test_spike["n_samples"] = int(len(yte_sp))
    else:
        yhat_te_sp = np.array([])
        m_test_spike = {"r2": float("nan"), "n_samples": 0}

    # ── Two-stage inference (full test set) ───────────────────────────────────
    # Stage1=0 → predict 0.0 (no outbreak); Stage1=1 → Stage2 regression
    y_pred_final = np.zeros(len(yte), dtype=np.float64)
    if mask_te.sum() > 0:
        y_pred_final[mask_te] = model.predict(
            Xte[mask_te], batch_size=args.batch_size, verbose=0
        ).reshape(-1)

    m_full_log1p = compute_metrics(yte, y_pred_final)
    m_full_log1p["smape"] = compute_smape(yte, y_pred_final)

    yte_raw    = inverse_log1p(yte)
    ypred_raw  = inverse_log1p(y_pred_final)
    m_full_raw = compute_metrics(yte_raw, ypred_raw)
    m_full_raw["smape"] = compute_smape(yte_raw, ypred_raw)

    # ── Comparison vs baseline ────────────────────────────────────────────────
    beat_r2_log1p = m_full_log1p["r2"] > BASELINE["r2_log1p"]
    beat_r2_raw   = m_full_raw["r2"]   > BASELINE["r2_raw"]
    beat_rmse_raw = m_full_raw["rmse"] < BASELINE["rmse_raw"]

    vs_baseline = {
        "baseline_r2_log1p":    BASELINE["r2_log1p"],
        "baseline_r2_raw":      BASELINE["r2_raw"],
        "baseline_rmse_raw":    BASELINE["rmse_raw"],
        "two_stage_r2_log1p":   float(m_full_log1p["r2"]),
        "two_stage_r2_raw":     float(m_full_raw["r2"]),
        "two_stage_rmse_raw":   float(m_full_raw["rmse"]),
        "beat_r2_log1p":        beat_r2_log1p,
        "beat_r2_raw":          beat_r2_raw,
        "beat_rmse_raw":        beat_rmse_raw,
    }

    # ── Save metrics_stage2.json ──────────────────────────────────────────────
    save_json(out_dir / "metrics_stage2.json", {
        "model":       "Two-stage CNN-LSTM (Stage2=Regression)",
        "stage":       2,
        "npz":         str(Path(args.npz).resolve()),
        "window":      window,
        "n_features":  n_features,
        "spike_threshold":       spike_threshold,
        "pred_threshold_stage1": args.pred_threshold,
        "spike_mask_counts": {
            "train": int(mask_tr.sum()),
            "val":   int(mask_va.sum()),
            "test":  int(mask_te.sum()),
        },
        "spike_only_val":  m_val_spike,
        "spike_only_test": m_test_spike,
        "full_test_log1p": m_full_log1p,
        "full_test_raw":   m_full_raw,
        "vs_baseline":     vs_baseline,
        "meta":            seq.meta,
        "hyperparams":     vars(args),
    })

    # ── predictions_test_full.csv ─────────────────────────────────────────────
    # stage2_pred = NaN where Stage 1 predicted no-spike (not passed to Stage 2)
    stage2_col = np.where(mask_te, y_pred_final, np.nan)

    df_out = pd.DataFrame()
    if len(seq.y_date_test) == len(yte):
        df_out["date"] = pd.to_datetime(seq.y_date_test)
    if len(seq.station_test) == len(yte):
        df_out["station_id"] = seq.station_test.astype(str)
    df_out["y_true_log1p"]       = yte
    df_out["y_pred_final_log1p"] = y_pred_final
    df_out["stage1_label"]       = mask_te.astype(int)
    df_out["stage2_pred"]        = stage2_col
    df_out["y_true_raw"]         = yte_raw
    df_out["y_pred_final_raw"]   = ypred_raw

    if "date" in df_out.columns and "station_id" in df_out.columns:
        df_out = df_out.sort_values(["station_id", "date"])

    df_out.to_csv(out_dir / "predictions_test_full.csv", index=False, encoding="utf-8-sig")

    # ── Figures ───────────────────────────────────────────────────────────────
    plot_training_history(hist.history, out_dir / "figures" / "loss_curve.png")

    plot_scatter_full(
        yte, y_pred_final,
        out_dir / "figures" / "scatter_full.png",
        f"Two-stage Test: true vs pred (log1p)  R²={m_full_log1p['r2']:.4f}",
    )

    if "station_id" in df_out.columns and "date" in df_out.columns:
        best_st = str(df_out["station_id"].value_counts().index[0])
        plot_timeseries_combined(
            df_out, best_st,
            out_dir / "figures" / "timeseries_combined.png",
            "Two-stage: True vs Pred",
        )

    # ── REPORT_stage2.md ──────────────────────────────────────────────────────
    def _vs(flag: bool, label: str, val: float, base: float, better_when: str = "higher") -> str:
        arrow = "BETTER" if flag else "WORSE"
        delta = val - base if better_when == "higher" else base - val
        return f"  {label}: {val:.4f}  (baseline {base}) [{arrow} Δ={delta:+.4f}]"

    write_text_report(out_dir / "REPORT_stage2.md", [
        "# Stage 2: BPH Two-stage Regression — Report",
        "",
        f"NPZ: {Path(args.npz).resolve()}",
        f"Window={window}, Features={n_features}",
        f"Spike threshold (log1p): {spike_threshold:.4f}",
        f"Stage 1 pred_threshold:  {args.pred_threshold}",
        "",
        "## Stage 2 Architecture: CNN-LSTM Regression (Optuna best params)",
        f"- Conv1D({args.conv_filters}, k={args.kernel_size}) + MaxPool(2) + Dropout({args.dropout:.4f})",
        f"- LSTM({args.lstm_units}, dropout={args.dropout:.4f})",
        "- Dense(64, relu) + Dense(1)  — Huber loss",
        f"- lr={args.lr:.6f}, clipnorm={args.clipnorm}, batch_size={args.batch_size}",
        "",
        "## Spike Sample Counts (Stage 1 predicted)",
        f"- Train: {mask_tr.sum()}/{len(mask_tr)} ({mask_tr.mean()*100:.1f}%)",
        f"- Val:   {mask_va.sum()}/{len(mask_va)} ({mask_va.mean()*100:.1f}%)",
        f"- Test:  {mask_te.sum()}/{len(mask_te)} ({mask_te.mean()*100:.1f}%)",
        "",
        "## Two-stage Inference Logic",
        "- Stage 1 = 0 (no-spike) → y_pred_final = 0.0",
        "- Stage 1 = 1 (spike)    → y_pred_final = Stage 2 regression output",
        "",
        "## Spike-only Metrics (Stage 2 on spike test samples)",
        f"- R²  log1p: {m_test_spike.get('r2', float('nan')):.4f}",
        f"- MAE log1p: {m_test_spike.get('mae', float('nan')):.4f}",
        f"- n_samples: {m_test_spike.get('n_samples', 0)}",
        "",
        "## Full Two-stage Metrics (all test samples)",
        _vs(beat_r2_log1p, "R²  log1p", m_full_log1p["r2"], BASELINE["r2_log1p"]),
        f"  MAE log1p:  {m_full_log1p['mae']:.4f}",
        f"  RMSE log1p: {m_full_log1p['rmse']:.4f}",
        _vs(beat_r2_raw,   "R²  raw",   m_full_raw["r2"],   BASELINE["r2_raw"]),
        f"  MAE raw:    {m_full_raw['mae']:.2f}",
        _vs(beat_rmse_raw, "RMSE raw",  m_full_raw["rmse"], BASELINE["rmse_raw"],
            better_when="lower"),
        "",
        "## Next Steps",
        "- If two-stage R² > 0.4798: include in paper as main result",
        "- Consider optimizing pred_threshold (try 0.3–0.4 for higher recall)",
        "- Consider training Stage 2 on oracle spike labels (y_true > threshold) for comparison",
    ])

    model.save(out_dir / "model_final.keras")

    # ── Summary ───────────────────────────────────────────────────────────────
    def _mark(flag: bool) -> str:
        return "BETTER" if flag else "WORSE"

    print("\n" + "=" * 68)
    print("[Two-stage — Full Test Results]")
    print(f"  R²  log1p : {m_full_log1p['r2']:.4f}  "
          f"(baseline {BASELINE['r2_log1p']})  {_mark(beat_r2_log1p)}")
    print(f"  R²  raw   : {m_full_raw['r2']:.4f}  "
          f"(baseline {BASELINE['r2_raw']})   {_mark(beat_r2_raw)}")
    print(f"  RMSE raw  : {m_full_raw['rmse']:.2f}  "
          f"(baseline {BASELINE['rmse_raw']})  {_mark(beat_rmse_raw)}")
    print(f"  MAE log1p : {m_full_log1p['mae']:.4f}")
    print(f"  MAE raw   : {m_full_raw['mae']:.2f}")
    print("=" * 68)
    print(f"[OK] metrics_stage2.json → {out_dir / 'metrics_stage2.json'}")
    print(f"[OK] Outputs saved to:     {out_dir}")


if __name__ == "__main__":
    main()
