#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_stage1_classifier.py
Stage 1 of Two-stage BPH model: binary classifier (spike vs no-spike).

Architecture: CNN-LSTM (same backbone as best regression model)
  - Why: CNN-LSTM proved best on this dataset (R²=0.484 W30 H7 context)
  - Conv1D extracts local temporal patterns that precede spike events
  - LSTM captures sequential context across the window
  - BatchNorm stabilizes training under class imbalance
  - sigmoid output + binary_crossentropy loss

Usage:
/home/ai-station/my_project/.tf251p310/bin/python scripts/train_stage1_classifier.py \
  --npz results/out_feature_sets_w30_h7/context/sequences_window30_h7.npz \
  --out_dir results/out_train_stage1 \
  --quantile 0.75 \
  --epochs 150 --patience 25 --batch_size 128 --lr 0.0005 --seed 42
"""

from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from train_utils import (
    set_seed, try_enable_tf_memory_growth,
    load_sequences_npz, make_out_dir,
    assert_all_finite, save_json, write_text_report,
)


# ── Architecture ──────────────────────────────────────────────────────────────

def build_classifier(window: int, n_features: int,
                     conv_filters: int = 64, kernel_size: int = 5,
                     lstm_units: int = 64, dropout: float = 0.25,
                     lr: float = 5e-4):
    import tensorflow as tf
    from tensorflow.keras import layers, models, optimizers

    inp = layers.Input(shape=(window, n_features))

    # CNN block 1 — local feature extraction
    x = layers.Conv1D(conv_filters, kernel_size, padding="same", activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(dropout)(x)

    # CNN block 2 — finer feature maps
    x = layers.Conv1D(conv_filters // 2, kernel_size, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)

    # LSTM — temporal sequence context
    x = layers.LSTM(lstm_units, dropout=dropout)(x)

    # Classification head
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(dropout / 2)(x)
    out = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inp, out)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr, clipnorm=1.0),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_clf_metrics(y_true: np.ndarray, y_prob: np.ndarray,
                        pred_threshold: float = 0.5) -> dict:
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    )
    y_pred = (y_prob >= pred_threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    return {
        "accuracy":         float(accuracy_score(y_true, y_pred)),
        "precision":        float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":           float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":               float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": cm.tolist(),
        "n_pos":            int(y_true.sum()),
        "n_neg":            int((1 - y_true).sum()),
    }


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_confusion_matrix(cm: list, out_png: Path, title: str = "Confusion Matrix") -> None:
    import matplotlib.pyplot as plt

    cm_arr = np.array(cm)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm_arr, cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred: No spike", "Pred: Spike"])
    ax.set_yticklabels(["True: No spike", "True: Spike"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm_arr[i, j]), ha="center", va="center",
                    fontsize=14, color="black")
    ax.set_title(title)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def plot_loss_curve(history: dict, out_png: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.get("loss", []), label="train")
    axes[0].plot(history.get("val_loss", []), label="val")
    axes[0].set_title("Binary Cross-Entropy Loss")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss")
    axes[0].legend()

    axes[1].plot(history.get("accuracy", []), label="train")
    axes[1].plot(history.get("val_accuracy", []), label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("accuracy")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Stage 1: BPH spike binary classifier (CNN-LSTM)")
    p.add_argument("--npz",           required=True,           help="Path to sequences NPZ")
    p.add_argument("--out_dir",       required=True,           help="Output directory")
    p.add_argument("--quantile",      type=float, default=0.75, help="Spike threshold quantile (y_train)")
    p.add_argument("--epochs",        type=int,   default=150)
    p.add_argument("--patience",      type=int,   default=25)
    p.add_argument("--batch_size",    type=int,   default=128)
    p.add_argument("--lr",            type=float, default=5e-4)
    p.add_argument("--conv_filters",  type=int,   default=64)
    p.add_argument("--kernel_size",   type=int,   default=5)
    p.add_argument("--lstm_units",    type=int,   default=64)
    p.add_argument("--dropout",       type=float, default=0.25)
    p.add_argument("--pred_threshold",type=float, default=0.5,
                   help="Probability threshold for binary prediction (default=0.5)")
    p.add_argument("--seed",          type=int,   default=42)
    return p.parse_args(argv)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(argv=None):
    args = parse_args(argv)
    set_seed(args.seed)
    try_enable_tf_memory_growth()

    seq = load_sequences_npz(args.npz)
    out_dir = make_out_dir(args.out_dir)

    Xtr, ytr = seq.X_train, seq.y_train.reshape(-1)
    Xva, yva = seq.X_val,   seq.y_val.reshape(-1)
    Xte, yte = seq.X_test,  seq.y_test.reshape(-1)

    assert_all_finite("X_train", Xtr); assert_all_finite("y_train", ytr)
    assert_all_finite("X_val",   Xva); assert_all_finite("y_val",   yva)
    assert_all_finite("X_test",  Xte); assert_all_finite("y_test",  yte)

    window     = int(Xtr.shape[1])
    n_features = int(Xtr.shape[2])

    # ── Spike threshold (train only) ──────────────────────────────────────────
    spike_threshold = float(np.quantile(ytr, args.quantile))
    print(f"[Stage1] Spike threshold (Q{args.quantile:.0%} of y_train, log1p): {spike_threshold:.4f}")

    save_json(out_dir / "threshold.json", {
        "spike_threshold": spike_threshold,
        "quantile":        args.quantile,
        "computed_from":   "y_train",
        "note":            "log1p(bph_count) — use spike_threshold in Stage 2 to define spike labels",
    })

    # Binary labels
    ltr = (ytr > spike_threshold).astype(np.float32)
    lva = (yva > spike_threshold).astype(np.float32)
    lte = (yte > spike_threshold).astype(np.float32)

    print(f"[Stage1] Train — spike={ltr.mean()*100:.1f}%  ({int(ltr.sum())}/{len(ltr)})")
    print(f"[Stage1] Val   — spike={lva.mean()*100:.1f}%  ({int(lva.sum())}/{len(lva)})")
    print(f"[Stage1] Test  — spike={lte.mean()*100:.1f}%  ({int(lte.sum())}/{len(lte)})")

    # ── Class weight (auto from train set) ────────────────────────────────────
    n_neg = int((ltr == 0).sum())
    n_pos = int((ltr == 1).sum())
    w_pos = n_neg / n_pos if n_pos > 0 else 1.0
    class_weight = {0: 1.0, 1: w_pos}
    print(f"[Stage1] class_weight: {{0: 1.00, 1: {w_pos:.2f}}}")

    # ── Build and train ───────────────────────────────────────────────────────
    import tensorflow as tf
    from tensorflow.keras.callbacks import (
        EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TerminateOnNaN,
    )

    model = build_classifier(
        window, n_features,
        conv_filters=args.conv_filters,
        kernel_size=args.kernel_size,
        lstm_units=args.lstm_units,
        dropout=args.dropout,
        lr=args.lr,
    )
    model.summary()

    ckpt_path = out_dir / "model_best.keras"
    callbacks = [
        TerminateOnNaN(),
        EarlyStopping(monitor="val_loss", patience=args.patience, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=max(2, args.patience // 3), min_lr=1e-6),
        ModelCheckpoint(filepath=str(ckpt_path), monitor="val_loss", save_best_only=True),
    ]

    hist = model.fit(
        Xtr, ltr,
        validation_data=(Xva, lva),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weight,
        verbose=2,
        callbacks=callbacks,
    )

    # ── Predictions ───────────────────────────────────────────────────────────
    prob_tr = model.predict(Xtr, batch_size=args.batch_size, verbose=0).reshape(-1)
    prob_va = model.predict(Xva, batch_size=args.batch_size, verbose=0).reshape(-1)
    prob_te = model.predict(Xte, batch_size=args.batch_size, verbose=0).reshape(-1)

    m_tr = compute_clf_metrics(ltr.astype(int), prob_tr, args.pred_threshold)
    m_va = compute_clf_metrics(lva.astype(int), prob_va, args.pred_threshold)
    m_te = compute_clf_metrics(lte.astype(int), prob_te, args.pred_threshold)

    # ── Save metrics_stage1.json ──────────────────────────────────────────────
    save_json(out_dir / "metrics_stage1.json", {
        "model":            "CNN-LSTM-Classifier",
        "stage":            1,
        "npz":              str(Path(args.npz).resolve()),
        "window":           window,
        "n_features":       n_features,
        "spike_threshold":  spike_threshold,
        "quantile":         args.quantile,
        "pred_threshold":   args.pred_threshold,
        "class_weight":     {"0": 1.0, "1": w_pos},
        "train":            m_tr,
        "val":              m_va,
        "test":             m_te,
        "meta":             seq.meta,
        "hyperparams":      vars(args),
    })

    # ── predictions_test.csv ──────────────────────────────────────────────────
    df_pred = pd.DataFrame({
        "y_true_label":  lte.astype(int),
        "y_pred_label":  (prob_te >= args.pred_threshold).astype(int),
        "y_pred_prob":   prob_te,
        "y_true_log1p":  yte,
    })
    if len(seq.y_date_test) == len(yte):
        df_pred.insert(0, "date", pd.to_datetime(seq.y_date_test))
    if len(seq.station_test) == len(yte):
        df_pred.insert(0, "station_id", seq.station_test.astype(str))
    if "date" in df_pred.columns and "station_id" in df_pred.columns:
        df_pred = df_pred.sort_values(["station_id", "date"])
    df_pred.to_csv(out_dir / "predictions_test.csv", index=False, encoding="utf-8-sig")

    # ── Figures ───────────────────────────────────────────────────────────────
    plot_confusion_matrix(
        m_te["confusion_matrix"],
        out_dir / "figures" / "confusion_matrix.png",
        "Test — Confusion Matrix",
    )
    plot_loss_curve(hist.history, out_dir / "figures" / "loss_curve.png")

    # ── REPORT_stage1.md ──────────────────────────────────────────────────────
    def _section(m: dict, name: str) -> list:
        return [
            f"### {name}",
            f"- Accuracy:  {m['accuracy']:.4f}",
            f"- Precision: {m['precision']:.4f}",
            f"- Recall:    {m['recall']:.4f}",
            f"- F1:        {m['f1']:.4f}",
            f"- Samples:   pos={m['n_pos']}  neg={m['n_neg']}",
            f"- Confusion matrix: {m['confusion_matrix']}",
            "",
        ]

    write_text_report(out_dir / "REPORT_stage1.md", [
        "# Stage 1: BPH Spike Binary Classifier",
        "",
        f"NPZ: {Path(args.npz).resolve()}",
        f"Window={window}, Features={n_features}",
        f"Spike threshold (Q{args.quantile:.0%} of y_train, log1p): {spike_threshold:.4f}",
        f"Class weight: {{0: 1.00, 1: {w_pos:.2f}}}",
        "",
        "## Architecture: CNN-LSTM Binary Classifier",
        "- Conv1D(64, k=5) + BatchNorm + MaxPool(2) + Dropout(0.25)",
        "- Conv1D(32, k=5) + BatchNorm + Dropout(0.25)",
        "- LSTM(64, dropout=0.25)",
        "- Dense(32, relu) + Dropout(0.125) + Dense(1, sigmoid)",
        "",
        "Rationale: CNN-LSTM is the best model on this dataset (R²=0.484 W30 H7).",
        "CNN blocks capture local temporal patterns preceding spikes.",
        "BatchNorm stabilizes training under class imbalance.",
        "",
        "## Metrics",
        *_section(m_tr, "Train"),
        *_section(m_va, "Val"),
        *_section(m_te, "Test"),
        "## Thresholds",
        f"- Spike threshold saved: {out_dir}/threshold.json",
        f"- Binary decision threshold (pred_threshold): {args.pred_threshold}",
        "",
        "## Next Steps",
        "- F1 >= 0.55 and Recall >= 0.60 → proceed to Stage 2 regression",
        "- Otherwise: tune --quantile (try 0.70) or increase --patience",
    ])

    model.save(out_dir / "model_final.keras")

    # ── Summary ───────────────────────────────────────────────────────────────
    cm = np.array(m_te["confusion_matrix"])
    print("\n" + "=" * 60)
    print("[Stage 1 — Test Results]")
    print(f"  Accuracy:  {m_te['accuracy']:.4f}")
    print(f"  Precision: {m_te['precision']:.4f}")
    print(f"  Recall:    {m_te['recall']:.4f}")
    print(f"  F1:        {m_te['f1']:.4f}")
    print(f"  Spike threshold (log1p): {spike_threshold:.4f}  (Q{args.quantile:.0%})")
    print(f"  Confusion matrix:\n    TN={cm[0,0]}  FP={cm[0,1]}\n    FN={cm[1,0]}  TP={cm[1,1]}")
    print("=" * 60)
    print(f"[OK] threshold.json  → {out_dir / 'threshold.json'}")
    print(f"[OK] metrics_stage1.json → {out_dir / 'metrics_stage1.json'}")
    print(f"[OK] Outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
