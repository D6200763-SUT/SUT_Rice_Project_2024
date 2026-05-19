#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_walk_forward_folds.py
Walk-Forward Temporal Cross-Validation fold builder for BPH research.

Creates NPZ files for each fold with regression + classification targets.

Usage:
python scripts/build_walk_forward_folds.py \
  --input_csv results/out_quality_gate/cleaned_raw.csv \
  --out_dir results/walk_forward \
  --folds 2,3,4 \
  --min_train_years 2 \
  --window 30 --horizon 7 --roll_days 7 \
  --feature_set context \
  --spike_threshold 100
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.preprocessing import MinMaxScaler

from build_sequences import (
    read_csv_any, pick_first_existing, ensure_datetime, safe_numeric,
    add_time_features, add_bph_targets, add_temp_range, add_wind_uv,
    add_delta_and_rolling, select_feature_set, sanitize_features,
)
from train_utils import assert_all_finite

# ---------------------------------------------------------------------------
# Fold date-range definitions
# Fold 4 mirrors the original single-split for direct comparison.
# ---------------------------------------------------------------------------
FOLD_CONFIGS: Dict[int, Dict[str, str]] = {
    2: {
        "train_start": "2015-01-01", "train_end": "2016-12-31",
        "val_start":   "2017-01-01", "val_end":   "2017-06-30",
        "test_start":  "2017-07-01", "test_end":  "2017-12-31",
    },
    3: {
        "train_start": "2015-01-01", "train_end": "2017-12-31",
        "val_start":   "2018-01-01", "val_end":   "2018-06-30",
        "test_start":  "2018-07-01", "test_end":  "2018-12-31",
    },
    4: {
        "train_start": "2015-01-01", "train_end": "2018-07-01",
        "val_start":   "2018-07-02", "val_end":   "2019-03-31",
        "test_start":  "2019-04-01", "test_end":  "2019-12-31",
    },
}


def build_sequences_wf(
    df: pd.DataFrame,
    station_col: str,
    date_col: str,
    feature_cols: List[str],
    target_col: str,
    spike_col: str,
    window: int,
    horizon: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build sliding-window sequences from the full fold dataset.

    Window consecutive-date check is applied per window slice so that
    sequences spanning the train/val boundary remain valid (enabling
    the val set to look back into train data).

    Returns: X, y_log1p, y_spike, y_date, station
    """
    X_all, y_all, yspike_all, yd_all, st_all = [], [], [], [], []

    for st, g in df.groupby(station_col):
        g2 = g.sort_values(date_col).reset_index(drop=True)
        dates = g2[date_col].values
        feat = g2[feature_cols].values.astype(np.float32)
        yv = g2[target_col].values.astype(np.float32)
        spike_v = g2[spike_col].values.astype(np.float32)

        n = len(g2)
        if n < window + horizon:
            continue

        for i in range(window - 1, n - horizon):
            # Require consecutive dates inside the window
            w_dates = dates[i - window + 1: i + 1]
            diffs = np.diff(w_dates.astype("datetime64[D]").astype("int64"))
            if not np.all(diffs == 1):
                continue
            X_all.append(feat[i - window + 1: i + 1, :])
            y_all.append(yv[i + horizon])
            yspike_all.append(spike_v[i + horizon])
            yd_all.append(dates[i + horizon])
            st_all.append(str(st))

    if not X_all:
        raise RuntimeError("No sequences built — check fold data range and parameters.")

    return (
        np.stack(X_all, axis=0),
        np.array(y_all,      dtype=np.float32),
        np.array(yspike_all, dtype=np.float32),
        np.array(yd_all,     dtype="datetime64[ns]"),
        np.array(st_all,     dtype=object),
    )


def build_fold(
    df_full: pd.DataFrame,
    fold: int,
    cfg: Dict[str, str],
    station_col: str,
    date_col: str,
    feature_cols: List[str],
    window: int,
    horizon: int,
    spike_threshold: float,
    out_dir: Path,
    min_train_years: int,
    feature_set_name: str,
) -> None:
    fold_dir = out_dir / f"fold{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    train_start = pd.Timestamp(cfg["train_start"])
    train_end   = pd.Timestamp(cfg["train_end"])
    val_start   = pd.Timestamp(cfg["val_start"])
    val_end     = pd.Timestamp(cfg["val_end"])
    test_start  = pd.Timestamp(cfg["test_start"])
    test_end    = pd.Timestamp(cfg["test_end"])

    train_years = (train_end - train_start).days / 365.25
    # Allow 0.02-year tolerance (≈7 days) to handle leap-year edge cases
    if train_years < min_train_years - 0.02:
        raise RuntimeError(
            f"Fold {fold}: train period too short "
            f"({train_years:.2f} yrs < {min_train_years} required)"
        )

    # Include a lookback window so sequences at val/test boundary can use
    # the tail of the preceding period.
    lookback_start = train_start - pd.Timedelta(days=window + horizon + 1)
    df_fold = df_full[
        (df_full[date_col] >= lookback_start) & (df_full[date_col] <= test_end)
    ].copy()

    if len(df_fold) == 0:
        raise RuntimeError(f"Fold {fold}: no data in date range")

    # Add binary spike label column
    df_fold["spike_label"] = (df_fold["bph_raw"] > spike_threshold).astype(np.float32)

    # Sanitize features (ffill -> bfill -> 0 within station)
    df_fold = sanitize_features(df_fold, station_col, date_col, feature_cols)

    # Fit MinMaxScaler on TRAIN portion only
    train_mask = (df_fold[date_col] >= train_start) & (df_fold[date_col] <= train_end)
    X_train_raw = df_fold.loc[train_mask, feature_cols].values.astype(np.float32)
    if len(X_train_raw) == 0:
        raise RuntimeError(f"Fold {fold}: no training rows for scaler")

    scaler = MinMaxScaler()
    scaler.fit(X_train_raw)

    df_scaled = df_fold.copy()
    df_scaled[feature_cols] = scaler.transform(
        df_fold[feature_cols].values.astype(np.float32)
    )

    # Build sequences from the full (scaled) fold dataset
    X, y, y_spike, y_date, station = build_sequences_wf(
        df_scaled, station_col, date_col,
        feature_cols, "bph_log1p", "spike_label",
        window, horizon,
    )

    y_dates_ts = pd.to_datetime(y_date)

    tr_mask = (y_dates_ts >= train_start) & (y_dates_ts <= train_end)
    va_mask = (y_dates_ts >= val_start)   & (y_dates_ts <= val_end)
    te_mask = (y_dates_ts >= test_start)  & (y_dates_ts <= test_end)

    X_train, y_train, y_spike_train = X[tr_mask], y[tr_mask], y_spike[tr_mask]
    X_val,   y_val,   y_spike_val   = X[va_mask], y[va_mask], y_spike[va_mask]
    X_test,  y_test,  y_spike_test  = X[te_mask], y[te_mask], y_spike[te_mask]
    y_date_train, y_date_val, y_date_test = y_date[tr_mask], y_date[va_mask], y_date[te_mask]
    station_train, station_val, station_test = station[tr_mask], station[va_mask], station[te_mask]

    # ---- Validations ----
    n_features = int(X_train.shape[2])
    if feature_set_name == "context":
        assert n_features == 18, f"Expected 18 context features, got {n_features}"

    assert set(np.unique(y_spike_train)).issubset({0.0, 1.0}), \
        "y_spike_train is not binary {0, 1}"

    for name, arr in [
        ("X_train", X_train), ("y_train", y_train),
        ("X_val",   X_val),   ("y_val",   y_val),
        ("X_test",  X_test),  ("y_test",  y_test),
    ]:
        assert_all_finite(name, arr)

    # Spike ratio check (warning only)
    for split_name, sp_arr, total in [
        ("train", y_spike_train, len(y_train)),
        ("val",   y_spike_val,   len(y_val)),
        ("test",  y_spike_test,  len(y_test)),
    ]:
        n_sp = int(sp_arr.sum())
        ratio = n_sp / total if total > 0 else 0.0
        tag = f"Fold {fold} {split_name}: spike={n_sp}/{total} ({ratio:.1%})"
        if ratio < 0.05:
            print(f"[WARNING] {tag} — spike ratio < 0.05 (very rare spikes)")
        elif ratio > 0.40:
            print(f"[WARNING] {tag} — spike ratio > 0.40 (very frequent spikes)")
        else:
            print(f"[OK]      {tag}")

    # ---- Save NPZ ----
    meta = {
        "fold": fold,
        "feature_set": feature_set_name,
        "feature_cols": feature_cols,
        "window": window,
        "horizon": horizon,
        "spike_threshold": spike_threshold,
        "train_start": cfg["train_start"], "train_end": cfg["train_end"],
        "val_start":   cfg["val_start"],   "val_end":   cfg["val_end"],
        "test_start":  cfg["test_start"],  "test_end":  cfg["test_end"],
        "counts": {
            "train": int(len(y_train)),
            "val":   int(len(y_val)),
            "test":  int(len(y_test)),
        },
    }

    np.savez_compressed(
        fold_dir / "sequences_wf.npz",
        X_train=X_train, y_train=y_train, y_spike_train=y_spike_train,
        y_date_train=y_date_train, station_train=station_train,
        X_val=X_val,     y_val=y_val,     y_spike_val=y_spike_val,
        y_date_val=y_date_val,     station_val=station_val,
        X_test=X_test,   y_test=y_test,   y_spike_test=y_spike_test,
        y_date_test=y_date_test,   station_test=station_test,
        meta=np.array([json.dumps(meta, ensure_ascii=False)], dtype=object),
    )

    dump({"scaler": scaler, "feature_cols": feature_cols}, fold_dir / "scaler_minmax.joblib")

    fold_info = {
        "fold": fold,
        "train_start": cfg["train_start"], "train_end": cfg["train_end"],
        "val_start":   cfg["val_start"],   "val_end":   cfg["val_end"],
        "test_start":  cfg["test_start"],  "test_end":  cfg["test_end"],
        "spike_threshold": spike_threshold,
        "n_features": n_features,
        "n_spike_train": int(y_spike_train.sum()),
        "n_total_train": int(len(y_train)),
        "n_spike_val":   int(y_spike_val.sum()),
        "n_total_val":   int(len(y_val)),
        "n_spike_test":  int(y_spike_test.sum()),
        "n_total_test":  int(len(y_test)),
        "feature_cols":  feature_cols,
    }
    (fold_dir / "fold_info.json").write_text(
        json.dumps(fold_info, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"[OK] Fold {fold} saved → {fold_dir}")
    print(f"     train={len(y_train)}  val={len(y_val)}  test={len(y_test)}  features={n_features}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Build walk-forward CV folds for BPH research")
    p.add_argument("--input_csv",        required=True, help="Path to cleaned_raw.csv")
    p.add_argument("--out_dir",          required=True, help="Output root (e.g. results/walk_forward)")
    p.add_argument("--folds",            default="2,3,4", help="Comma-separated fold numbers")
    p.add_argument("--min_train_years",  type=int, default=2)
    p.add_argument("--window",           type=int, default=30)
    p.add_argument("--horizon",          type=int, default=7)
    p.add_argument("--roll_days",        type=int, default=7)
    p.add_argument("--feature_set",      default="context", choices=["core", "context", "full"])
    p.add_argument("--spike_threshold",  type=float, default=100.0,
                   help="bph_raw > spike_threshold → spike=1")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    folds = [int(f.strip()) for f in args.folds.split(",")]
    unknown = [f for f in folds if f not in FOLD_CONFIGS]
    if unknown:
        raise ValueError(f"Unknown fold(s): {unknown}. Available: {list(FOLD_CONFIGS.keys())}")

    # ------------------------------------------------------------------
    # Load & preprocess the full dataset once, then slice per fold
    # ------------------------------------------------------------------
    input_csv = Path(args.input_csv).expanduser().resolve()
    df = read_csv_any(input_csv)

    date_col    = pick_first_existing(df, ["date", "Date", "datetime", "timestamp"])
    station_col = pick_first_existing(df, ["station_id", "StationID", "station", "stationId"])
    bph_col     = pick_first_existing(df, ["bph_count", "BPH_count", "bph_raw"])
    temp_col    = pick_first_existing(df, ["temp", "temperature", "tmean"])
    hum_col     = pick_first_existing(df, ["humidity", "rh"])
    rain_col    = pick_first_existing(df, ["rainfall", "rain"])
    ws_col      = pick_first_existing(df, ["wind_speed", "windspeed", "ws"])
    wd_col      = pick_first_existing(df, ["wind_direction", "wind_dir", "wd", "winddirection"])
    mint_col    = pick_first_existing(df, ["mint_temp", "tmin", "min_temp"])
    maxt_col    = pick_first_existing(df, ["maxt_temp", "tmax", "max_temp"])

    if not date_col or not station_col:
        raise RuntimeError("Cannot detect date / station columns")
    if not bph_col:
        raise RuntimeError("Cannot detect bph_count column")

    df = ensure_datetime(df, date_col)
    df = safe_numeric(df, [bph_col, temp_col, hum_col, rain_col, ws_col, wd_col, mint_col, maxt_col])
    df = df.dropna(subset=[bph_col])

    df = add_bph_targets(df,   base_col=bph_col)
    df = add_time_features(df, date_col=date_col)
    df = add_temp_range(df,    mint_col=mint_col, maxt_col=maxt_col)
    df = add_wind_uv(df,       wind_speed_col=ws_col, wind_dir_col=wd_col, wind_from_convention=False)
    df = add_delta_and_rolling(df, station_col, date_col, temp_col, hum_col, rain_col, int(args.roll_days))

    feature_cols = select_feature_set(
        df, args.feature_set, int(args.roll_days),
        include_year=False, include_distance_km=False, variety_keep_cols=[],
    )
    print(f"[INFO] Feature set '{args.feature_set}': {len(feature_cols)} features")
    print(f"       {feature_cols}")

    for fold in folds:
        cfg = FOLD_CONFIGS[fold]
        print(f"\n{'='*50}")
        print(f"Building Fold {fold}")
        print(f"  train : {cfg['train_start']} ~ {cfg['train_end']}")
        print(f"  val   : {cfg['val_start']} ~ {cfg['val_end']}")
        print(f"  test  : {cfg['test_start']} ~ {cfg['test_end']}")
        print(f"  spike_threshold: {args.spike_threshold}")

        build_fold(
            df_full=df,
            fold=fold,
            cfg=cfg,
            station_col=station_col,
            date_col=date_col,
            feature_cols=feature_cols,
            window=int(args.window),
            horizon=int(args.horizon),
            spike_threshold=float(args.spike_threshold),
            out_dir=out_dir,
            min_train_years=int(args.min_train_years),
            feature_set_name=args.feature_set,
        )

    print(f"\n{'='*50}")
    print("[DONE] All folds built successfully.")


if __name__ == "__main__":
    main()
