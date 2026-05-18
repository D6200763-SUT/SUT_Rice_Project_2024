#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
12_train_cnn_lstm_compat.py
CNN-LSTM trainer compatible with older scikit-learn (uses train_seq_utils_compat.py).

Usage:
python code/12_train_cnn_lstm_compat.py --npz out_feature_sets/core/sequences_window30_h1.npz --out_dir out_train/cnn_lstm_core
"""

from __future__ import annotations
import argparse
from pathlib import Path

from train_utils import (
    set_seed, try_enable_tf_memory_growth,
    load_sequences_npz, make_out_dir,
    compute_metrics, compute_smape, inverse_log1p, assert_all_finite,
    save_predictions_csv, pick_station_for_plot, subset_by_station,
    save_json, write_text_report,
    plot_training_history, plot_scatter_true_pred, plot_residual_hist, plot_timeseries_one_station
)

def build_model(window: int, n_features: int, conv_filters: int, kernel_size: int,
                lstm_units: int, dropout: float, lr: float, clipnorm: float):
    import tensorflow as tf
    from tensorflow.keras import layers, models, optimizers

    inp = layers.Input(shape=(window, n_features))
    x = layers.Conv1D(filters=conv_filters, kernel_size=kernel_size, padding="same", activation="relu")(inp)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(dropout)(x)
    x = layers.LSTM(lstm_units, dropout=dropout)(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(1)(x)

    opt = optimizers.Adam(learning_rate=lr, clipnorm=clipnorm if clipnorm > 0 else None)
    model = models.Model(inp, out)
    model.compile(optimizer=opt, loss=tf.keras.losses.Huber())
    return model

def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--npz", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--clipnorm", type=float, default=1.0)
    p.add_argument("--conv_filters", type=int, default=64)
    p.add_argument("--kernel_size", type=int, default=5)
    p.add_argument("--lstm_units", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.25)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--plot_station", default="")
    return p.parse_args(argv)

def main(argv=None):
    args = parse_args(argv)
    set_seed(args.seed)
    try_enable_tf_memory_growth()
    seq = load_sequences_npz(args.npz)
    out_dir = make_out_dir(args.out_dir)

    Xtr,ytr = seq.X_train, seq.y_train
    Xva,yva = seq.X_val, seq.y_val
    Xte,yte = seq.X_test, seq.y_test

    assert_all_finite("X_train", Xtr); assert_all_finite("y_train", ytr)
    assert_all_finite("X_val", Xva);   assert_all_finite("y_val", yva)
    assert_all_finite("X_test", Xte);  assert_all_finite("y_test", yte)

    window = int(Xtr.shape[1]); n_features = int(Xtr.shape[2])

    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TerminateOnNaN

    model = build_model(window, n_features, args.conv_filters, args.kernel_size,
                        args.lstm_units, args.dropout, args.lr, args.clipnorm)

    ckpt_path = out_dir / "model_best.keras"
    callbacks = [
        TerminateOnNaN(),
        EarlyStopping(monitor="val_loss", patience=args.patience, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=max(2, args.patience//2), min_lr=1e-6),
        ModelCheckpoint(filepath=str(ckpt_path), monitor="val_loss", save_best_only=True),
    ]

    hist = model.fit(Xtr, ytr, validation_data=(Xva, yva),
                     epochs=args.epochs, batch_size=args.batch_size,
                     verbose=2, callbacks=callbacks)

    yhat_tr = model.predict(Xtr, batch_size=args.batch_size, verbose=0).reshape(-1)
    yhat_va = model.predict(Xva, batch_size=args.batch_size, verbose=0).reshape(-1)
    yhat_te = model.predict(Xte, batch_size=args.batch_size, verbose=0).reshape(-1)

    assert_all_finite("y_pred_test", yhat_te)

    m_tr = compute_metrics(ytr, yhat_tr)
    m_va = compute_metrics(yva, yhat_va)
    m_te = compute_metrics(yte, yhat_te); m_te["smape"] = compute_smape(yte, yhat_te)

    yte_raw = inverse_log1p(yte); yhat_te_raw = inverse_log1p(yhat_te)
    m_te_raw = compute_metrics(yte_raw, yhat_te_raw); m_te_raw["smape"] = compute_smape(yte_raw, yhat_te_raw)

    metrics = {
        "model":"CNN-LSTM",
        "npz": str(Path(args.npz).resolve()),
        "window": window,
        "n_features": n_features,
        "train": m_tr,
        "val": m_va,
        "test_log1p": m_te,
        "test_raw": m_te_raw,
        "meta": seq.meta,
        "hyperparams": vars(args),
    }
    save_json(out_dir/"metrics.json", metrics)

    save_predictions_csv(out_dir,"train",ytr,yhat_tr,seq.y_date_train,seq.station_train,also_raw=True)
    save_predictions_csv(out_dir,"val",yva,yhat_va,seq.y_date_val,seq.station_val,also_raw=True)
    pred_test = save_predictions_csv(out_dir,"test",yte,yhat_te,seq.y_date_test,seq.station_test,also_raw=True)

    plot_training_history(hist.history, out_dir/"figures"/"loss_curve.png")
    plot_scatter_true_pred(yte,yhat_te,out_dir/"figures"/"scatter_true_pred_log1p.png","Test: true vs pred (log1p)")
    plot_residual_hist(yte,yhat_te,out_dir/"figures"/"residual_hist_log1p.png","Test residuals (log1p)")

    st = args.plot_station.strip() or pick_station_for_plot(seq.station_test)
    if st and len(seq.y_date_test)==len(yte) and len(seq.station_test)==len(yte):
        df_st = subset_by_station(yte,yhat_te,seq.y_date_test,seq.station_test,st)
        if len(df_st)>0:
            plot_timeseries_one_station(df_st,out_dir/"figures"/f"timeseries_test_{st}.png",f"Test station={st}: true vs pred")

    write_text_report(out_dir/"REPORT.md",[
        "# Training report: CNN-LSTM",
        f"NPZ: {Path(args.npz).resolve()}",
        f"Window={window}, Horizon={seq.meta.get('horizon','?')}, Features={n_features}",
        "",
        "## Metrics (log1p)",
        f"- MAE:  {m_te['mae']:.4f}",
        f"- RMSE: {m_te['rmse']:.4f}",
        f"- R2:   {m_te['r2']:.4f}",
        f"- sMAPE:{m_te['smape']:.4f}",
        "",
        "## Metrics (raw)",
        f"- MAE:  {m_te_raw['mae']:.2f}",
        f"- RMSE: {m_te_raw['rmse']:.2f}",
        f"- R2:   {m_te_raw['r2']:.4f}",
        f"- sMAPE:{m_te_raw['smape']:.4f}",
    ])

    model.save(out_dir/"model_final.keras")
    print("[OK] Done:", out_dir)

if __name__=="__main__":
    main()
