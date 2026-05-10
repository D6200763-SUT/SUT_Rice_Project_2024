#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_build_feature_sets_and_sequences.py
-------------------------------------
Build "feature sets" (core / context / full) from a cleaned daily CSV,
then create model-ready scaled tables + sliding-window sequences for each set.

Why?
- Your feature_cols are many and contain redundant/compositional groups.
- This script lets you run **ablation** experiments reproducibly:
    core -> core+context -> full (variety reduced)

Feature sets (default)
1) core:
   - time cyclic: month_sin/cos, doy_sin/cos (+ optional year)
   - weather: temp, humidity, rainfall
   - wind: wind_u, wind_v
   - dynamics: delta_temp, rolling means/sums
   - optional: temp_range (maxt-mint) if available

2) context:
   - core + province monthly context: area_rai_in_season/off_season
   - + station spatial: latitude, longitude
   - optional: distance_km (mapping QA)

3) full:
   - context + variety shares (Top-K per season + "other")
   - optionally drop one var_share per season to remove linear dependency.

Outputs (for each set under --out_dir/<set_name>/)
- model_ready_daily.csv
- model_ready_daily_scaled.csv
- scaler_minmax.joblib
- sequences_window{W}_h{H}.npz
- report_model_ready.json

Example
python code/03_build_feature_sets_and_sequences.py \
  --input_csv out_quality_gate/cleaned_raw.csv \
  --out_dir out_feature_sets \
  --window 30 --horizon 1 --roll_days 7 \
  --topk_var 8 \
  --all_sets

Or run only core:
python code/03_build_feature_sets_and_sequences.py \
  --input_csv out_quality_gate/cleaned_raw.csv \
  --out_dir out_feature_sets \
  --set core

Notes
- Cross-platform paths (Windows/macOS/Linux) via pathlib.
- If your wind_direction is "from" (meteorological), you can set --wind_from_convention
  to rotate 180 degrees before computing u/v.
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


ENCODINGS = ("utf-8-sig", "utf-8", "cp874", "tis-620")


# ----------------------------
# I/O helpers
# ----------------------------
def read_csv_any(path: str | Path, **kwargs) -> pd.DataFrame:
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
        if c and c in out.columns:
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
    out["year"] = dt.dt.year.astype(int)

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


def add_temp_range_if_possible(df: pd.DataFrame, mint_col: Optional[str], maxt_col: Optional[str]) -> pd.DataFrame:
    out = df.copy()
    if mint_col and maxt_col and mint_col in out.columns and maxt_col in out.columns:
        out["temp_range"] = pd.to_numeric(out[maxt_col], errors="coerce") - pd.to_numeric(out[mint_col], errors="coerce")
    return out


def add_wind_uv(df: pd.DataFrame, wind_speed_col: str, wind_dir_col: str, wind_from_convention: bool) -> pd.DataFrame:
    out = df.copy()
    ws = pd.to_numeric(out[wind_speed_col], errors="coerce")
    wd = pd.to_numeric(out[wind_dir_col], errors="coerce") % 360.0
    if wind_from_convention:
        wd = (wd + 180.0) % 360.0
    theta = np.deg2rad(wd)
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
    out["delta_temp"] = out.groupby(station_col)[temp_col].diff()
    out[f"temp_{roll_days}d_mean"] = out.groupby(station_col)[temp_col].rolling(roll_days, min_periods=1).mean().reset_index(level=0, drop=True)
    out[f"humidity_{roll_days}d_mean"] = out.groupby(station_col)[humidity_col].rolling(roll_days, min_periods=1).mean().reset_index(level=0, drop=True)
    out[f"rain_{roll_days}d_sum"] = out.groupby(station_col)[rainfall_col].rolling(roll_days, min_periods=1).sum().reset_index(level=0, drop=True)
    return out


def detect_var_share_cols(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    cols = df.columns.tolist()
    in_cols = [c for c in cols if c.startswith("var_share_") and c.endswith("_in_season")]
    off_cols = [c for c in cols if c.startswith("var_share_") and c.endswith("_off_season")]
    return in_cols, off_cols


def build_variety_topk(
    df: pd.DataFrame,
    in_cols: List[str],
    off_cols: List[str],
    topk: int,
    drop_one: bool
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    out = df.copy()
    info: Dict[str, Any] = {"topk": int(topk), "drop_one": bool(drop_one)}

    def _topk_cols(cols: List[str]) -> List[str]:
        if not cols:
            return []
        means = out[cols].apply(pd.to_numeric, errors="coerce").mean(axis=0).sort_values(ascending=False)
        return means.head(topk).index.tolist()

    top_in = _topk_cols(in_cols)
    top_off = _topk_cols(off_cols)
    info["top_in"] = top_in
    info["top_off"] = top_off

    keep_cols: List[str] = []

    if top_in:
        s = out[top_in].apply(pd.to_numeric, errors="coerce").sum(axis=1)
        out["var_other_in_season"] = (1.0 - s).clip(lower=0.0, upper=1.0)
        keep_cols += top_in + ["var_other_in_season"]

    if top_off:
        s_top = out[top_off].apply(pd.to_numeric, errors="coerce").sum(axis=1)
        total_off = out[off_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1) if off_cols else 0.0
        other = (total_off - s_top).clip(lower=0.0)
        other_frac = np.where(total_off > 0, (other / total_off).clip(0.0, 1.0), 0.0)
        out["var_other_off_season"] = other_frac
        keep_cols += top_off + ["var_other_off_season"]

    dropped = {"in": None, "off": None}
    if drop_one:
        if top_in and keep_cols:
            dropped["in"] = keep_cols[0]
            keep_cols = [c for c in keep_cols if c != dropped["in"]]
        if top_off:
            off_candidates = [c for c in keep_cols if c.endswith("_off_season") or c == "var_other_off_season"]
            if off_candidates:
                dropped["off"] = off_candidates[0]
                keep_cols = [c for c in keep_cols if c != dropped["off"]]

    info["dropped_one"] = dropped
    info["keep_cols"] = keep_cols
    return out, info


# ----------------------------
# Split by time (no shuffle)
# ----------------------------
def split_dates_by_ratio(unique_dates: np.ndarray, train_ratio: float, val_ratio: float) -> Dict[str, Any]:
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
        "n_dates": int(n),
        "n_train_dates": int(n_train),
        "n_val_dates": int(n_val),
        "n_test_dates": int(n_test),
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
# Feature set selection
# ----------------------------
def select_feature_set(
    df: pd.DataFrame,
    set_name: str,
    roll_days: int,
    include_year: bool,
    include_distance_km: bool,
    variety_keep_cols: List[str]
) -> List[str]:
    cyc = ["month_sin", "month_cos", "doy_sin", "doy_cos"]
    core: List[str] = [c for c in cyc if c in df.columns]
    if include_year and "year" in df.columns:
        core.append("year")

    for c in ["temp", "humidity", "rainfall"]:
        if c in df.columns:
            core.append(c)

    for c in ["wind_u", "wind_v"]:
        if c in df.columns:
            core.append(c)

    for c in ["delta_temp", f"temp_{roll_days}d_mean", f"humidity_{roll_days}d_mean", f"rain_{roll_days}d_sum"]:
        if c in df.columns:
            core.append(c)

    if "temp_range" in df.columns:
        core.append("temp_range")

    if set_name == "core":
        return core

    context = core.copy()
    for c in ["area_rai_in_season", "area_rai_off_season"]:
        if c in df.columns:
            context.append(c)

    for c in ["latitude", "longitude"]:
        if c in df.columns:
            context.append(c)

    if include_distance_km and "distance_km" in df.columns:
        context.append("distance_km")

    if set_name == "context":
        return context

    full = context.copy()
    for c in variety_keep_cols:
        if c in df.columns:
            full.append(c)
    return full


# ----------------------------
# Runner for one feature set
# ----------------------------
def run_one_set(
    df_base: pd.DataFrame,
    set_name: str,
    date_col: str,
    station_col: str,
    target_col: str,
    feature_cols: List[str],
    window: int,
    horizon: int,
    train_ratio: float,
    val_ratio: float,
    require_consecutive: bool,
    out_dir: Path
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    unique_dates = df_base[date_col].dropna().sort_values().unique()
    split_info = split_dates_by_ratio(unique_dates, train_ratio=train_ratio, val_ratio=val_ratio)
    df = assign_split(df_base, date_col=date_col, split_info=split_info)

    model_ready_path = out_dir / "model_ready_daily.csv"
    df.to_csv(model_ready_path, index=False, encoding="utf-8-sig")

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

    seq = build_sequences(
        df=scaled,
        station_col=station_col,
        date_col=date_col,
        feature_cols=feature_cols,
        target_col=target_col,
        window=window,
        horizon=horizon,
        require_consecutive=require_consecutive
    )

    meta = {
        "set_name": set_name,
        "date_col": date_col,
        "station_col": station_col,
        "target_used": target_col,
        "feature_cols": feature_cols,
        "window": int(window),
        "horizon": int(horizon),
        "split": {k: v for k, v in split_info.items() if not k.endswith("_ts")},
        "require_consecutive": bool(require_consecutive),
        "counts": {
            "train_samples": int(seq["X_train"].shape[0]),
            "val_samples": int(seq["X_val"].shape[0]),
            "test_samples": int(seq["X_test"].shape[0]),
        }
    }

    npz_path = out_dir / f"sequences_window{window}_h{horizon}.npz"
    np.savez_compressed(
        npz_path,
        **seq,
        meta=np.array([json.dumps(meta, ensure_ascii=False)], dtype=object)
    )

    report_path = out_dir / "report_model_ready.json"
    report_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "set_name": set_name,
        "out_dir": str(out_dir),
        "npz": str(npz_path),
        "report": str(report_path),
        **meta["counts"]
    }


# ----------------------------
# CLI
# ----------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Build feature sets and sequences for ablation experiments")
    p.add_argument("--input_csv", required=True, help="Input cleaned CSV (from quality_gate.py)")
    p.add_argument("--out_dir", required=True, help="Output directory root")
    p.add_argument("--set", default="core", choices=["core", "context", "full"], help="Which feature set to build")
    p.add_argument("--all_sets", action="store_true", help="Build core, context, full in one run")

    p.add_argument("--target", default="bph_log1p", choices=["bph_log1p", "bph_raw"], help="Target to learn")
    p.add_argument("--window", type=int, default=30, help="Sliding window length (days)")
    p.add_argument("--horizon", type=int, default=1, help="Forecast horizon (days ahead)")
    p.add_argument("--roll_days", type=int, default=7, help="Rolling window size for means/sums")

    p.add_argument("--train_ratio", type=float, default=0.70, help="Train ratio by date (no shuffle)")
    p.add_argument("--val_ratio", type=float, default=0.15, help="Validation ratio by date (no shuffle)")
    p.add_argument("--require_consecutive", action="store_true", help="Require strictly consecutive daily dates per station within each split")

    p.add_argument("--include_year", action="store_true", help="Include 'year' in feature set")
    p.add_argument("--include_distance_km", action="store_true", help="Include 'distance_km' (mapping QA) in context/full")
    p.add_argument("--wind_from_convention", action="store_true", help="Treat wind_direction as meteorological FROM direction (rotate 180°)")

    p.add_argument("--topk_var", type=int, default=8, help="Top-K varieties to keep per season (full set)")
    p.add_argument("--drop_one_var_share", action="store_true", help="Drop one var_share col per season to remove linear dependency (recommended)")

    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    input_csv = Path(args.input_csv).expanduser().resolve()
    out_root = Path(args.out_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    df = read_csv_any(input_csv)

    date_col = pick_first_existing(df, ["date", "Date", "datetime", "timestamp"])
    station_col = pick_first_existing(df, ["station_id", "StationID", "station", "stationId"])
    if not date_col or not station_col:
        raise RuntimeError(f"Cannot detect date/station columns. date={date_col}, station={station_col}")

    base_bph_col = pick_first_existing(df, ["bph_count", "BPH_count", "bph_raw"])
    if not base_bph_col:
        raise RuntimeError("Cannot detect BPH count column. Expected one of: bph_count, BPH_count, bph_raw")

    temp_col = pick_first_existing(df, ["temp", "temperature", "tmean"])
    hum_col = pick_first_existing(df, ["humidity", "rh"])
    rain_col = pick_first_existing(df, ["rainfall", "rain"])
    ws_col = pick_first_existing(df, ["wind_speed", "windspeed", "ws"])
    wd_col = pick_first_existing(df, ["wind_direction", "wind_dir", "wd", "winddirection"])

    mint_col = pick_first_existing(df, ["mint_temp", "tmin", "min_temp"])
    maxt_col = pick_first_existing(df, ["maxt_temp", "tmax", "max_temp"])

    missing = [name for name, col in [("temp", temp_col), ("humidity", hum_col), ("rainfall", rain_col), ("wind_speed", ws_col), ("wind_direction", wd_col)] if col is None]
    if missing:
        raise RuntimeError(f"Missing required weather columns: {missing}")

    df = ensure_datetime(df, date_col)
    df = safe_numeric(df, [base_bph_col, temp_col, hum_col, rain_col, ws_col, wd_col, mint_col, maxt_col])

    df = add_bph_targets(df, base_col=base_bph_col)
    df = add_time_features(df, date_col=date_col)
    df = add_temp_range_if_possible(df, mint_col=mint_col, maxt_col=maxt_col)
    df = add_wind_uv(df, wind_speed_col=ws_col, wind_dir_col=wd_col, wind_from_convention=bool(args.wind_from_convention))
    df = add_delta_and_rolling(df, station_col=station_col, date_col=date_col, temp_col=temp_col, humidity_col=hum_col, rainfall_col=rain_col, roll_days=int(args.roll_days))

    in_cols, off_cols = detect_var_share_cols(df)
    df_var = df
    var_info: Dict[str, Any] = {}
    variety_keep_cols: List[str] = []
    if args.all_sets or args.set == "full":
        df_var, var_info = build_variety_topk(df, in_cols, off_cols, topk=int(args.topk_var), drop_one=bool(args.drop_one_var_share))
        variety_keep_cols = var_info.get("keep_cols", [])

    sets = ["core", "context", "full"] if args.all_sets else [args.set]

    summary = {"input_csv": str(input_csv), "sets": [], "variety_info": var_info}
    for s in sets:
        feature_cols = select_feature_set(
            df=df_var,
            set_name=s,
            roll_days=int(args.roll_days),
            include_year=bool(args.include_year),
            include_distance_km=bool(args.include_distance_km),
            variety_keep_cols=variety_keep_cols
        )

        # Keep only needed columns (identifiers + targets + selected features)
        keep_id = [date_col, station_col]
        for c in ["latitude", "longitude", "province_en", "province_th"]:
            if c in df_var.columns and c not in keep_id:
                keep_id.append(c)

        base_cols = list(dict.fromkeys(keep_id + ["bph_raw", "bph_log1p"] + feature_cols))
        df_base = df_var[[c for c in base_cols if c in df_var.columns]].copy()

        out_dir = out_root / s
        res = run_one_set(
            df_base=df_base,
            set_name=s,
            date_col=date_col,
            station_col=station_col,
            target_col=str(args.target),
            feature_cols=feature_cols,
            window=int(args.window),
            horizon=int(args.horizon),
            train_ratio=float(args.train_ratio),
            val_ratio=float(args.val_ratio),
            require_consecutive=bool(args.require_consecutive),
            out_dir=out_dir
        )
        summary["sets"].append(res)
        print(f"[OK] Built set={s} -> train={res['train_samples']} val={res['val_samples']} test={res['test_samples']}")

    (out_root / "summary_feature_sets.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Summary: {out_root / 'summary_feature_sets.json'}")


if __name__ == "__main__":
    main()
