#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_cnn_lstm_classifier.py
CNN-LSTM binary classifier for BPH spike detection (Stage 1 of two-stage pipeline).

Reads y_spike_train / y_spike_val / y_spike_test from NPZ built by
build_walk_forward_folds.py and trains a binary classifier.

Usage:
python scripts/train_cnn_lstm_classifier.py \
  --npz results/walk_forward/fold4/sequences_wf.npz \
  --out_dir results/walk_forward/fold4/stage1 \
  --epochs 150 --batch_size 128 --lr 0.0005 --patience 25 \
  --conv_filters 64 --kernel_size 5 --lstm_units 64 --dropout 0.25 \
  --class_weight auto --recall_target 0.80
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
    assert_all_finite, save_json, plot_training_history,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_spike_labels(npz_path: str):
    """Load binary spike label arrays from NPZ."""
    data = np.load(Path(npz_path).expanduser().resolve(), allow_pickle=True)
    required = ["y_spike_train", "y_spike_val", "y_spike_test"]
    missing = [k for k in required if k not in data.files]
    if missing:
        raise RuntimeError(
            f"NPZ missing spike keys: {missing}. "
            "Build the NPZ with build_walk_forward_folds.py first."
        )
    return (
        data["y_spike_train"].astype(np.float32),
        data["y_spike_val"].astype(np.float32),
        data["y_spike_test"].astype(np.float32),
    )


def build_classifier(
    window: int, n_features: int,
    conv_filters: int, kernel_size: int, lstm_units: int,
    dropout: float, lr: float, clipnorm: float,
):
    import tensorflow as tf
    from tensorflow.keras import layers, models, optimizers

    inp = layers.Input(shape=(window, n_features))
    x = layers.Conv1D(filters=conv_filters, kernel_size=kernel_size,
                      padding="same", activation="relu")(inp)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(dropout)(x)
    x = layers.LSTM(lstm_units, dropout=dropout)(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)

    opt = optimizers.Adam(
        learning_rate=lr,
        clipnorm=clipnorm if clipnorm > 0 else None,
    )
    model = models.Model(inp, out)
    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])
    return model


def clf_metrics_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
    y_pred = (y_prob >= threshold).astype(int)
    yi = y_true.astype(int)
    return {
        "threshold":        threshold,
        "precision":        float(precision_score(yi, y_pred, zero_division=0)),
        "recall":           float(recall_score(yi, y_pred, zero_division=0)),
        "f1":               float(f1_score(yi, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(yi, y_pred).tolist(),
    }


def select_best_threshold(
    y_true: np.ndarray, y_prob: np.ndarray,
    thresholds: list, recall_target: float,
) -> dict:
    """Return threshold with highest Precision where Recall >= recall_target.

    Falls back to the threshold with highest Recall if no threshold satisfies
    the recall constraint.
    """
    candidates = [clf_metrics_at_threshold(y_true, y_prob, t) for t in thresholds]
    qualifying = [m for m in candidates if m["recall"] >= recall_target]
    if qualifying:
        return max(qualifying, key=lambda m: m["precision"])

    best = max(candidates, key=lambda m: m["recall"])
    print(
        f"[WARNING] No threshold achieves Recall >= {recall_target:.2f}. "
        f"Using threshold={best['threshold']:.2f} (Recall={best['recall']:.3f})"
    )
    return best


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, out_png: Path) -> float:
    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt

    fpr, tpr, _ = roc_curve(y_true.astype(int), y_prob)
    roc_auc = float(auc(fpr, tpr))
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve (Test)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    return roc_auc


def plot_pr_curve(y_true: np.ndarray, y_prob: np.ndarray, out_png: Path) -> None:
    from sklearn.metrics import precision_recall_curve, average_precision_score
    import matplotlib.pyplot as plt

    prec, rec, _ = precision_recall_curve(y_true.astype(int), y_prob)
    ap = float(average_precision_score(y_true.astype(int), y_prob))
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(rec, prec, label=f"AP = {ap:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve (Test)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def plot_confusion_matrix(cm_list: list, out_png: Path, title: str = "Confusion Matrix") -> None:
    import matplotlib.pyplot as plt

    cm = np.array(cm_list)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Non-spike", "Spike"])
    ax.set_yticklabels(["Non-spike", "Spike"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=14, color="black")
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="CNN-LSTM Stage 1 binary classifier")
    p.add_argument("--npz",          required=True)
    p.add_argument("--out_dir",      required=True)
    p.add_argument("--epochs",       type=int,   default=150)
    p.add_argument("--batch_size",   type=int,   default=128)
    p.add_argument("--lr",           type=float, default=5e-4)
    p.add_argument("--clipnorm",     type=float, default=1.0)
    p.add_argument("--conv_filters", type=int,   default=64)
    p.add_argument("--kernel_size",  type=int,   default=5)
    p.add_argument("--lstm_units",   type=int,   default=64)
    p.add_argument("--dropout",      type=float, default=0.25)
    p.add_argument("--patience",     type=int,   default=25)
    p.add_argument("--class_weight", default="auto", choices=["auto", "none"])
    p.add_argument("--recall_target", type=float, default=0.80)
    p.add_argument("--seed",         type=int,   default=42)
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    args = parse_args(argv)
    set_seed(args.seed)
    try_enable_tf_memory_growth()

    seq = load_sequences_npz(args.npz)
    y_spike_train, y_spike_val, y_spike_test = load_spike_labels(args.npz)
    out_dir = make_out_dir(args.out_dir)

    Xtr = seq.X_train
    Xva = seq.X_val
    Xte = seq.X_test

    assert_all_finite("X_train", Xtr)
    assert_all_finite("X_val",   Xva)
    assert_all_finite("X_test",  Xte)

    window     = int(Xtr.shape[1])
    n_features = int(Xtr.shape[2])

    print(f"[Stage1] window={window}  features={n_features}")
    print(f"[Stage1] train={len(Xtr)}  val={len(Xva)}  test={len(Xte)}")
    print(f"[Stage1] spike ratio — train={y_spike_train.mean():.3f}  "
          f"val={y_spike_val.mean():.3f}  test={y_spike_test.mean():.3f}")

    # ---- Class weight ----
    class_weight_dict = None
    if args.class_weight == "auto":
        from sklearn.utils.class_weight import compute_class_weight
        weights = compute_class_weight(
            "balanced", classes=np.array([0, 1]),
            y=y_spike_train.astype(int),
        )
        class_weight_dict = {0: float(weights[0]), 1: float(weights[1])}
        print(f"[Stage1] class_weight = {class_weight_dict}")

    import tensorflow as tf
    from tensorflow.keras.callbacks import (
        EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TerminateOnNaN,
    )

    model = build_classifier(
        window, n_features,
        args.conv_filters, args.kernel_size, args.lstm_units,
        args.dropout, args.lr, args.clipnorm,
    )

    ckpt_path = out_dir / "model_stage1_best.keras"
    callbacks = [
        TerminateOnNaN(),
        EarlyStopping(monitor="val_loss", patience=args.patience, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=max(2, args.patience // 2), min_lr=1e-6),
        ModelCheckpoint(filepath=str(ckpt_path), monitor="val_loss", save_best_only=True),
    ]

    hist = model.fit(
        Xtr, y_spike_train,
        validation_data=(Xva, y_spike_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=2,
    )

    # ---- Predict probabilities ----
    yprob_te = model.predict(Xte, batch_size=args.batch_size, verbose=0).reshape(-1)
    yprob_va = model.predict(Xva, batch_size=args.batch_size, verbose=0).reshape(-1)

    # ---- Select best threshold on validation set ----
    thresholds = [0.30, 0.35, 0.40, 0.45, 0.50]
    best_va = select_best_threshold(y_spike_val, yprob_va, thresholds, args.recall_target)
    best_thr = best_va["threshold"]

    # ---- Test metrics at best threshold ----
    m_te = clf_metrics_at_threshold(y_spike_test, yprob_te, best_thr)

    # ---- Plots ----
    auc_val = plot_roc_curve(y_spike_test, yprob_te, out_dir / "figures" / "roc_curve.png")
    plot_pr_curve(y_spike_test, yprob_te, out_dir / "figures" / "precision_recall_curve.png")
    plot_confusion_matrix(
        m_te["confusion_matrix"],
        out_dir / "figures" / "confusion_matrix.png",
        f"Confusion Matrix (Test, thr={best_thr:.2f})",
    )
    plot_training_history(hist.history, out_dir / "figures" / "loss_curve.png")

    # ---- Save metrics ----
    metrics = {
        "model":            "CNN-LSTM-Classifier",
        "npz":              str(Path(args.npz).resolve()),
        "recall":           m_te["recall"],
        "precision":        m_te["precision"],
        "f1":               m_te["f1"],
        "auc":              auc_val,
        "best_threshold":   best_thr,
        "confusion_matrix": m_te["confusion_matrix"],
        "class_weight":     class_weight_dict,
        "val_at_best_thr":  best_va,
        "all_thresholds":   [clf_metrics_at_threshold(y_spike_test, yprob_te, t) for t in thresholds],
        "hyperparams":      vars(args),
    }
    save_json(out_dir / "metrics_stage1.json", metrics)

    # ---- Save predictions CSV ----
    df_pred = pd.DataFrame({
        "y_true_spike": y_spike_test.astype(int),
        "y_pred_prob":  yprob_te,
        "y_pred_spike": (yprob_te >= best_thr).astype(int),
    })
    if len(seq.y_date_test) == len(df_pred):
        df_pred["date"] = pd.to_datetime(seq.y_date_test)
    if len(seq.station_test) == len(df_pred):
        df_pred["station_id"] = seq.station_test.astype(str)
    df_pred.to_csv(out_dir / "predictions_test_stage1.csv", index=False, encoding="utf-8-sig")

    model.save(out_dir / "model_stage1_final.keras")

    print(
        f"\n[Stage1] Test Recall={m_te['recall']:.3f}  "
        f"Precision={m_te['precision']:.3f}  "
        f"F1={m_te['f1']:.3f}  AUC={auc_val:.3f}"
    )
    print(f"[Stage1] Best threshold={best_thr:.2f} (Recall>={args.recall_target:.2f})")
    print(f"[OK] Saved: {ckpt_path}")
    print(f"[OK] Saved: {out_dir / 'metrics_stage1.json'}")


if __name__ == "__main__":
    main()
