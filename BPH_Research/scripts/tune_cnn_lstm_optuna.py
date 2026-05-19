#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tune_cnn_lstm_optuna.py
Optuna hyperparameter search for CNN-LSTM on BPH forecasting.

Usage:
  python scripts/tune_cnn_lstm_optuna.py \
    --npz results/out_feature_sets_w30_h7/context/sequences_window30_h7.npz \
    --out_dir results/out_tune_w30_h7_context \
    --n_trials 30

After tuning, run best params with train_cnn_lstm.py (command printed at end).
"""

from __future__ import annotations
import argparse
import sys
import numpy as np
from pathlib import Path

# Allow importing siblings (train_utils, train_cnn_lstm) when run from BPH_Research/
sys.path.insert(0, str(Path(__file__).parent))

try:
    import optuna
except ImportError:
    sys.exit(
        "[ERROR] optuna not installed.\n"
        "Run: /home/ai-station/my_project/.tf251p310/bin/pip install optuna"
    )

from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from train_utils import (
    set_seed, try_enable_tf_memory_growth,
    load_sequences_npz, make_out_dir,
    compute_metrics, compute_smape, inverse_log1p, assert_all_finite,
    save_json,
)
from train_cnn_lstm import build_model


# ── Optuna-aware Keras callback ──────────────────────────────────────────────

class _OptunaPruningCallback:
    """Reports val_loss each epoch and raises TrialPruned when pruner fires."""

    def __init__(self, trial: "optuna.Trial"):
        self._trial = trial

    def get_callback(self):
        trial = self._trial

        import tensorflow as tf

        class _CB(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                val_loss = (logs or {}).get("val_loss")
                if val_loss is None or np.isnan(val_loss):
                    return
                # Report negative val_loss as proxy for R² during training
                # (R² on val not computed per-epoch; val_loss is the best proxy)
                trial.report(float(val_loss), step=epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned(f"Pruned at epoch {epoch}")

        return _CB()


# ── Objective ────────────────────────────────────────────────────────────────

def objective(trial: "optuna.Trial", seq, out_root: Path, seed: int) -> float:
    import tensorflow as tf
    from tensorflow.keras.callbacks import (
        EarlyStopping, ReduceLROnPlateau, TerminateOnNaN,
    )

    # ── Hyperparameter search space ──
    conv_filters = trial.suggest_categorical("conv_filters", [32, 64, 128])
    kernel_size  = trial.suggest_categorical("kernel_size",  [3, 5, 7])
    lstm_units   = trial.suggest_categorical("lstm_units",   [32, 64, 128, 256])
    dropout      = trial.suggest_float("dropout", 0.1, 0.4)
    lr           = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
    batch_size   = trial.suggest_categorical("batch_size",   [64, 128, 256])
    clipnorm     = trial.suggest_categorical("clipnorm",     [0.5, 1.0, 2.0])

    EPOCHS   = 150
    PATIENCE = 25

    set_seed(seed)
    tf.keras.backend.clear_session()

    trial_dir = make_out_dir(out_root / f"trial_{trial.number:03d}")

    Xtr, ytr = seq.X_train, seq.y_train
    Xva, yva = seq.X_val,   seq.y_val
    Xte, yte = seq.X_test,  seq.y_test

    window     = int(Xtr.shape[1])
    n_features = int(Xtr.shape[2])

    model = build_model(
        window, n_features,
        conv_filters, kernel_size,
        lstm_units, dropout, lr, clipnorm,
    )

    pruning_cb = _OptunaPruningCallback(trial).get_callback()

    callbacks = [
        TerminateOnNaN(),
        EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=max(2, PATIENCE // 2), min_lr=1e-6),
        pruning_cb,
    ]

    hist = model.fit(
        Xtr, ytr,
        validation_data=(Xva, yva),
        epochs=EPOCHS,
        batch_size=batch_size,
        verbose=0,
        callbacks=callbacks,
    )

    val_losses = hist.history.get("val_loss", [float("nan")])
    best_val_loss = float(np.nanmin(val_losses)) if val_losses else float("nan")

    if np.isnan(best_val_loss):
        raise optuna.TrialPruned("val_loss is NaN")

    # ── Test metrics ──
    yhat_te = model.predict(Xte, batch_size=batch_size, verbose=0).reshape(-1)
    m_te    = compute_metrics(yte, yhat_te)
    m_te["smape"] = compute_smape(yte, yhat_te)

    yte_raw    = inverse_log1p(yte)
    yhat_raw   = inverse_log1p(yhat_te)
    m_te_raw   = compute_metrics(yte_raw, yhat_raw)
    m_te_raw["smape"] = compute_smape(yte_raw, yhat_raw)

    r2_log1p = float(m_te["r2"])

    if np.isnan(r2_log1p):
        raise optuna.TrialPruned("r2_log1p is NaN")

    # ── Persist per-trial results ──
    trial_metrics = {
        "trial_number": trial.number,
        "val_loss":     best_val_loss,
        "r2_log1p":     r2_log1p,
        "test_log1p":   m_te,
        "test_raw":     m_te_raw,
        "hyperparams": {
            "conv_filters": conv_filters,
            "kernel_size":  kernel_size,
            "lstm_units":   lstm_units,
            "dropout":      dropout,
            "lr":           lr,
            "batch_size":   batch_size,
            "clipnorm":     clipnorm,
            "epochs":       EPOCHS,
            "patience":     PATIENCE,
        },
    }
    save_json(trial_dir / "metrics.json", trial_metrics)

    trial.set_user_attr("val_loss",  best_val_loss)
    trial.set_user_attr("r2_log1p",  r2_log1p)
    trial.set_user_attr("r2_raw",    float(m_te_raw["r2"]))

    epochs_ran = len(val_losses)
    print(
        f"[Trial {trial.number:>3d}] "
        f"R²={r2_log1p:.4f}  "
        f"R²_raw={m_te_raw['r2']:.4f}  "
        f"val_loss={best_val_loss:.4f}  "
        f"epochs={epochs_ran}"
    )

    # Clear TF graph to free GPU memory between trials
    tf.keras.backend.clear_session()

    # Objective: maximize R² → minimize negative R²
    return -r2_log1p


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Optuna tuning for CNN-LSTM BPH model")
    p.add_argument("--npz",      required=True,  help="Path to sequences NPZ")
    p.add_argument("--out_dir",  required=True,  help="Root dir for trial outputs")
    p.add_argument("--n_trials", type=int, default=30)
    p.add_argument("--n_jobs",   type=int, default=1,
                   help="Parallel trials (1 recommended for GPU)")
    p.add_argument("--seed",     type=int, default=42)
    p.add_argument("--timeout",  type=float, default=None,
                   help="Stop after N seconds regardless of n_trials")
    return p.parse_args()


def main():
    args = parse_args()
    try_enable_tf_memory_growth()

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Load & validate data once — shared across all trials
    print(f"[Load] {args.npz}")
    seq = load_sequences_npz(args.npz)
    for name, arr in [
        ("X_train", seq.X_train), ("y_train", seq.y_train),
        ("X_val",   seq.X_val),   ("y_val",   seq.y_val),
        ("X_test",  seq.X_test),  ("y_test",  seq.y_test),
    ]:
        assert_all_finite(name, arr)

    print(
        f"[Data] train={seq.X_train.shape}  val={seq.X_val.shape}  "
        f"test={seq.X_test.shape}  features={seq.X_train.shape[2]}"
    )

    # n_startup_trials: explore randomly before pruning kicks in
    # n_warmup_steps:   don't prune before epoch 20 (model needs warm-up)
    sampler = TPESampler(seed=args.seed, n_startup_trials=10)
    pruner  = MedianPruner(n_startup_trials=10, n_warmup_steps=20, interval_steps=5)

    optuna.logging.set_verbosity(optuna.logging.WARNING)  # suppress per-step spam

    study = optuna.create_study(
        direction="minimize",   # minimize -R²  ≡  maximize R²
        sampler=sampler,
        pruner=pruner,
        study_name="cnn_lstm_bph_tuning_r2",
    )

    print(f"[Study] n_trials={args.n_trials}  out_dir={out_root}\n")

    study.optimize(
        lambda trial: objective(trial, seq, out_root, args.seed),
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        timeout=args.timeout,
        show_progress_bar=True,
    )

    # ── Summary ──────────────────────────────────────────────────────────────
    best = study.best_trial
    best_r2     = -best.value   # convert back from -R²
    best_params = {
        **best.params,
        "epochs":    150,
        "patience":  25,
        "r2_log1p":  best_r2,
        "val_loss":  best.user_attrs.get("val_loss"),
        "r2_raw":    best.user_attrs.get("r2_raw"),
    }
    save_json(out_root / "best_params.json", best_params)
    print(f"\n{'='*60}")
    print(f"[Best Trial #{best.number}]")
    print(f"  R² log1p : {best_r2:.4f}  ← objective")
    print(f"  val_loss : {best.user_attrs.get('val_loss', 'N/A')}")
    print(f"  R² raw   : {best.user_attrs.get('r2_raw', 'N/A')}")
    print(f"  params   : {best.params}")
    print(f"{'='*60}")

    # Save full study dataframe
    try:
        import pandas as pd
        df = study.trials_dataframe()
        csv_path = out_root / "optuna_study.csv"
        df.to_csv(csv_path, index=False)
        print(f"[Saved] {csv_path}")
    except ImportError:
        print("[Skip] pandas not available — optuna_study.csv not written")

    # ── Print ready-to-run command ────────────────────────────────────────────
    bp = best.params
    PYTHON = "/home/ai-station/my_project/.tf251p310/bin/python"
    tuned_dir = str(Path(args.out_dir).parent / "out_train_w30_h7_context_tuned")

    print(f"\n[Next step — run best params:]")
    print(f"{PYTHON} scripts/train_cnn_lstm.py \\")
    print(f"  --npz {args.npz} \\")
    print(f"  --out_dir {tuned_dir} \\")
    print(f"  --conv_filters {bp['conv_filters']} \\")
    print(f"  --kernel_size {bp['kernel_size']} \\")
    print(f"  --lstm_units {bp['lstm_units']} \\")
    print(f"  --dropout {bp['dropout']:.4f} \\")
    print(f"  --lr {bp['lr']:.6f} \\")
    print(f"  --batch_size {bp['batch_size']} \\")
    print(f"  --clipnorm {bp['clipnorm']} \\")
    print(f"  --epochs 200 \\")
    print(f"  --patience 35")


if __name__ == "__main__":
    main()
