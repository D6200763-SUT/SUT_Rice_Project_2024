#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02_make_model_ready_from_cleaned.py
-----------------------------------
From a *cleaned* daily station-level CSV (output of quality_gate.py),
build model-ready features, split by time (70/15/15, no shuffle),
fit MinMaxScaler on TRAIN only, and build sliding windows for deep learning.

Outputs (in --out_dir)
- model_ready_daily.csv                 (feature-engineered, unscaled)
- model_ready_daily_scaled.csv          (scaled features; identifiers/targets kept)
- scaler_minmax.joblib                  (fitted on TRAIN only)
- sequences_window{W}_h{H}.npz          (X/y for train/val/test + metadata arrays)
- report_model_ready.json               (summary, columns, split dates, counts)

Example
python code/02_make_model_ready_from_cleaned.py \
  --input_csv out_quality_gate/cleaned_raw.csv \
  --out_dir out_model_ready \
  --target bph_log1p \
  --window 30 \
  --horizon 1 \
  --roll_days 7

Notes
- This script is cross-platform (Pathlib).
- Wind components assume wind_direction is in degrees (0–360).
  We use:
      theta = radians(wind_direction)
      wind_u = wind_speed * cos(theta)
      wind_v = wind_speed * sin(theta)
  If your wind_direction uses meteorological "from" convention, you can adjust by adding 180 degrees.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.preprocessing import MinMaxScaler


# ----------------------------
# I/O helpers
# ----------------------------
ENCODINGS = ("utf-8-sig", "utf-8", "cp874", "tis-620")

def read_csv_any(path: str | Path, **kwargs) -> pd.DataFrame:
    """Read CSV trying multiple encodings (Thai-friendly)."""
    path = str(path)
    last_err = None
    for enc in ENCODINGS:
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except Exception as e:
            last_err = e
    try:
        return pd.read_csv(path, **kwargs)
    except Exception as e:
        raise RuntimeError(f"Cannot read CSV: {path} (last_err={last_err})") from e


def pick_first_existing(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def ensure_datetime(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out = out.dropna(subset=[date_col])
    return out


def safe_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


# ----------------------------
# Feature engineering
# ----------------------------
def add_time_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    out = df.copy()
    dt = out[date_col]
    out["doy"] = dt.dt.dayofyear.astype(int)
    out["month"] = dt.dt.month.astype(int)

    # cyclic encodings
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12.0)

    out["doy_sin"] = np.sin(2 * np.pi * out["doy"] / 365.0)
    out["doy_cos"] = np.cos(2 * np.pi * out["doy"] / 365.0)
    return out


def add_bph_targets(df: pd.DataFrame, base_col: str) -> pd.DataFrame:
    out = df.copy()
    out["bph_raw"] = pd.to_numeric(out[base_col], errors="coerce").clip(lower=0)
    out["bph_log1p"] = np.log1p(out["bph_raw"])
    return out


def add_wind_uv(df: pd.DataFrame, wind_speed_col: str, wind_dir_col: str) -> pd.DataFrame:
    out = df.copy()
    ws = pd.to_numeric(out[wind_speed_col], errors="coerce")
    wd = pd.to_numeric(out[wind_dir_col], errors="coerce")
    theta = np.deg2rad(wd % 360.0)
    out["wind_u"] = ws * np.cos(theta)
    out["wind_v"] = ws * np.sin(theta)
    return out


def add_delta_and_rolling(
    df: pd.DataFrame,
    station_col: str,
    date_col: str,
    temp_col: str,
    humidity_col: str,
    rainfall_col: str,
    roll_days: int
) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values([station_col, date_col])

    # delta temp (day-to-day change per station)
    out["delta_temp"] = out.groupby(station_col)[temp_col].diff()

    # rolling features (use past roll_days including current day)
    out[f"temp_{roll_days}d_mean"] = out.groupby(station_col)[temp_col].rolling(window=roll_days, min_periods=1).mean().reset_index(level=0, drop=True)
    out[f"humidity_{roll_days}d_mean"] = out.groupby(station_col)[humidity_col].rolling(window=roll_days, min_periods=1).mean().reset_index(level=0, drop=True)
    out[f"rain_{roll_days}d_sum"] = out.groupby(station_col)[rainfall_col].rolling(window=roll_days, min_periods=1).sum().reset_index(level=0, drop=True)
    return out


# ----------------------------
# Split by time (no shuffle)
# ----------------------------
def split_dates_by_ratio(unique_dates: pd.Series, train_ratio: float, val_ratio: float) -> Dict[str, Any]:
    dates = pd.Series(pd.to_datetime(unique_dates)).sort_values().reset_index(drop=True)
    n = len(dates)
    n_train = int(math.floor(n * train_ratio))
    n_val = int(math.floor(n * val_ratio))
    n_test = n - n_train - n_val
    if n_train <= 0 or n_val <= 0 or n_test <= 0:
        raise ValueError(f"Invalid split sizes: n={n} -> train={n_train} val={n_val} test={n_test}")

    train_end = dates.iloc[n_train - 1]
    val_end = dates.iloc[n_train + n_val - 1]

    return {
        "n_dates": n,
        "n_train_dates": n_train,
        "n_val_dates": n_val,
        "n_test_dates": n_test,
        "train_start": dates.iloc[0].date().isoformat(),
        "train_end": train_end.date().isoformat(),
        "val_start": dates.iloc[n_train].date().isoformat(),
        "val_end": val_end.date().isoformat(),
        "test_start": dates.iloc[n_train + n_val].date().isoformat(),
        "test_end": dates.iloc[-1].date().isoformat(),
        "train_end_ts": train_end,
        "val_end_ts": val_end,
    }


def assign_split(df: pd.DataFrame, date_col: str, split_info: Dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    train_end = split_info["train_end_ts"]
    val_end = split_info["val_end_ts"]
    dt = out[date_col]
    out["split"] = np.where(dt <= train_end, "train",
                     np.where(dt <= val_end, "val", "test"))
    return out


# ----------------------------
# Sliding windows
# ----------------------------
def _dates_are_consecutive(dates: np.ndarray) -> bool:
    if len(dates) < 2:
        return True
    diffs = np.diff(dates.astype("datetime64[D]").astype("int64"))
    return bool(np.all(diffs == 1))


def build_sequences_for_station(
    g: pd.DataFrame,
    date_col: str,
    feature_cols: List[str],
    target_col: str,
    window: int,
    horizon: int,
    require_consecutive: bool
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    g2 = g.sort_values(date_col).reset_index(drop=True)
    dates = g2[date_col].values

    if require_consecutive and not _dates_are_consecutive(dates):
        return np.empty((0, window, len(feature_cols))), np.empty((0,), dtype=np.float32), np.empty((0,), dtype="datetime64[ns]")

    feat = g2[feature_cols].values.astype(np.float32)
    yv = g2[target_col].values.astype(np.float32)

    X_list, y_list, yd_list = [], [], []
    n = len(g2)
    max_i = n - 1 - horizon
    for i in range(window - 1, max_i + 1):
        X_list.append(feat[i - window + 1: i + 1, :])
        y_list.append(yv[i + horizon])
        yd_list.append(dates[i + horizon])

    if not X_list:
        return np.empty((0, window, len(feature_cols))), np.empty((0,), dtype=np.float32), np.empty((0,), dtype="datetime64[ns]")

    return np.stack(X_list, axis=0), np.array(y_list, dtype=np.float32), np.array(yd_list, dtype="datetime64[ns]")


def build_sequences(
    df: pd.DataFrame,
    station_col: str,
    date_col: str,
    feature_cols: List[str],
    target_col: str,
    window: int,
    horizon: int,
    require_consecutive: bool
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for split_name in ["train", "val", "test"]:
        d = df[df["split"] == split_name].copy()

        X_all, y_all, yd_all, st_all = [], [], [], []
        for st, g in d.groupby(station_col):
            X, y, yd = build_sequences_for_station(
                g=g,
                date_col=date_col,
                feature_cols=feature_cols,
                target_col=target_col,
                window=window,
                horizon=horizon,
                require_consecutive=require_consecutive
            )
            if len(X) == 0:
                continue
            X_all.append(X)
            y_all.append(y)
            yd_all.append(yd)
            st_all.append(np.array([str(st)] * len(y), dtype=object))

        if X_all:
            out[f"X_{split_name}"] = np.concatenate(X_all, axis=0)
            out[f"y_{split_name}"] = np.concatenate(y_all, axis=0)
            out[f"y_date_{split_name}"] = np.concatenate(yd_all, axis=0)
            out[f"station_{split_name}"] = np.concatenate(st_all, axis=0)
        else:
            out[f"X_{split_name}"] = np.empty((0, window, len(feature_cols)), dtype=np.float32)
            out[f"y_{split_name}"] = np.empty((0,), dtype=np.float32)
            out[f"y_date_{split_name}"] = np.empty((0,), dtype="datetime64[ns]")
            out[f"station_{split_name}"] = np.empty((0,), dtype=object)

    return out


# ----------------------------
# CLI
# ----------------------------
@dataclass
class Args:
    input_csv: str
    out_dir: str
    target: str
    window: int
    horizon: int
    roll_days: int
    train_ratio: float
    val_ratio: float
    require_consecutive: bool


def parse_args(argv=None) -> Args:
    p = argparse.ArgumentParser(description="Build model-ready data + sliding windows from cleaned raw CSV")
    p.add_argument("--input_csv", required=True, help="Input cleaned CSV path (from quality_gate.py)")
    p.add_argument("--out_dir", required=True, help="Output directory")
    p.add_argument("--target", default="bph_log1p", choices=["bph_log1p", "bph_raw"], help="Target to learn")
    p.add_argument("--window", type=int, default=30, help="Sliding window length (days)")
    p.add_argument("--horizon", type=int, default=1, help="Forecast horizon (days ahead). horizon=1 predicts next day")
    p.add_argument("--roll_days", type=int, default=7, help="Rolling window size for means/sums")
    p.add_argument("--train_ratio", type=float, default=0.70, help="Train ratio by date (no shuffle)")
    p.add_argument("--val_ratio", type=float, default=0.15, help="Validation ratio by date (no shuffle)")
    p.add_argument("--require_consecutive", action="store_true", help="Require strictly consecutive daily dates per station within each split")
    args = p.parse_args(argv)
    return Args(**vars(args))


def main(argv=None):
    args = parse_args(argv)
    input_csv = Path(args.input_csv).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = read_csv_any(input_csv)

    # detect key columns
    date_col = pick_first_existing(df, ["date", "Date", "datetime", "timestamp"])
    station_col = pick_first_existing(df, ["station_id", "StationID", "station", "stationId"])
    if not date_col or not station_col:
        raise RuntimeError(f"Cannot detect date/station columns. date={date_col}, station={station_col}")

    # detect target base
    target_base_col = pick_first_existing(df, ["bph_count", "BPH_count", "bph_raw"])
    if not target_base_col:
        raise RuntimeError("Cannot detect BPH count column. Expected one of: bph_count, BPH_count, bph_raw")

    # weather cols
    temp_col = pick_first_existing(df, ["temp", "temperature", "tmean"])
    hum_col = pick_first_existing(df, ["humidity", "rh"])
    rain_col = pick_first_existing(df, ["rainfall", "rain"])
    ws_col = pick_first_existing(df, ["wind_speed", "windspeed", "ws"])
    wd_col = pick_first_existing(df, ["wind_direction", "wind_dir", "wd", "winddirection"])

    missing = [name for name, col in [("temp", temp_col), ("humidity", hum_col), ("rainfall", rain_col), ("wind_speed", ws_col), ("wind_direction", wd_col)] if col is None]
    if missing:
        raise RuntimeError(f"Missing required weather columns: {missing}")

    # normalize types
    df = ensure_datetime(df, date_col)
    df = safe_numeric(df, [target_base_col, temp_col, hum_col, rain_col, ws_col, wd_col])

    # targets + features
    df = add_bph_targets(df, base_col=target_base_col)
    df = add_time_features(df, date_col=date_col)
    df = add_wind_uv(df, wind_speed_col=ws_col, wind_dir_col=wd_col)
    df = add_delta_and_rolling(
        df,
        station_col=station_col,
        date_col=date_col,
        temp_col=temp_col,
        humidity_col=hum_col,
        rainfall_col=rain_col,
        roll_days=args.roll_days
    )

    target_col = args.target

    # feature columns: numeric and not identifiers/targets
    exclude = {date_col, station_col, "split", "bph_raw", "bph_log1p", target_base_col}
    for c in ["province_en", "province_th", "qa_flag", "method"]:
        if c in df.columns:
            exclude.add(c)

    candidate_cols = [c for c in df.columns if c not in exclude]
    feature_cols: List[str] = []
    for c in candidate_cols:
        if pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]):
            feature_cols.append(c)

    # split by date
    unique_dates = df[date_col].dropna().sort_values().unique()
    split_info = split_dates_by_ratio(unique_dates, train_ratio=args.train_ratio, val_ratio=args.val_ratio)
    df = assign_split(df, date_col=date_col, split_info=split_info)

    # save unscaled
    model_ready_path = out_dir / "model_ready_daily.csv"
    df.to_csv(model_ready_path, index=False, encoding="utf-8-sig")

    # fit scaler on TRAIN only
    train_mask = (df["split"] == "train")
    scaler = MinMaxScaler()
    scaler.fit(df.loc[train_mask, feature_cols].values.astype(np.float32))

    scaled = df.copy()
    scaled_vals = scaler.transform(df[feature_cols].values.astype(np.float32))
    scaled.loc[:, feature_cols] = scaled_vals

    scaler_path = out_dir / "scaler_minmax.joblib"
    dump({"scaler": scaler, "feature_cols": feature_cols}, scaler_path)

    scaled_path = out_dir / "model_ready_daily_scaled.csv"
    scaled.to_csv(scaled_path, index=False, encoding="utf-8-sig")

    # sequences
    seq = build_sequences(
        df=scaled,
        station_col=station_col,
        date_col=date_col,
        feature_cols=feature_cols,
        target_col=target_col,
        window=args.window,
        horizon=args.horizon,
        require_consecutive=args.require_consecutive
    )

    meta = {
        "input_csv": str(input_csv),
        "date_col": date_col,
        "station_col": station_col,
        "target_base_col": target_base_col,
        "target_used": target_col,
        "feature_cols": feature_cols,
        "window": int(args.window),
        "horizon": int(args.horizon),
        "roll_days": int(args.roll_days),
        "split": {k: v for k, v in split_info.items() if not k.endswith("_ts")},
        "require_consecutive": bool(args.require_consecutive),
        "counts": {
            "train_samples": int(seq["X_train"].shape[0]),
            "val_samples": int(seq["X_val"].shape[0]),
            "test_samples": int(seq["X_test"].shape[0]),
        }
    }

    npz_path = out_dir / f"sequences_window{args.window}_h{args.horizon}.npz"
    np.savez_compressed(
        npz_path,
        **seq,
        meta=np.array([json.dumps(meta, ensure_ascii=False)], dtype=object)
    )

    report_path = out_dir / "report_model_ready.json"
    report_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] model_ready_daily.csv: {model_ready_path}")
    print(f"[OK] model_ready_daily_scaled.csv: {scaled_path}")
    print(f"[OK] scaler_minmax.joblib: {scaler_path}")
    print(f"[OK] sequences: {npz_path}")
    print(f"[OK] report: {report_path}")


if __name__ == "__main__":
    main()
