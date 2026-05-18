#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_build_feature_sets_and_sequences_v2_nan_safe.py
-------------------------------------------------
Same as 03_build_feature_sets_and_sequences.py but with explicit NaN/Inf handling.

Key fixes:
- Drop rows where target base (bph_count) is missing
- Fill delta_temp NaN (first day per station) with 0
- Fill temp_range NaN with 0
- After selecting feature_cols: per-station forward-fill -> back-fill -> remaining NaN -> 0
- Replace +/-Inf with NaN before filling

This prevents NaNs from leaking into scaler / sequences, which is the #1 cause of:
ValueError: Input contains NaN (during sklearn metrics) and/or NaN loss.

Usage:
python code/03_build_feature_sets_and_sequences_v2_nan_safe.py \
  --input_csv out_quality_gate/cleaned_raw.csv \
  --out_dir out_feature_sets \
  --all_sets --require_consecutive
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable, Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.preprocessing import MinMaxScaler

ENCODINGS = ("utf-8-sig", "utf-8", "cp874", "tis-620")


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


def safe_numeric(df: pd.DataFrame, cols: Iterable[Optional[str]]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c and c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


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


def add_temp_range(df: pd.DataFrame, mint_col: Optional[str], maxt_col: Optional[str]) -> pd.DataFrame:
    out = df.copy()
    if mint_col and maxt_col and mint_col in out.columns and maxt_col in out.columns:
        out["temp_range"] = pd.to_numeric(out[maxt_col], errors="coerce") - pd.to_numeric(out[mint_col], errors="coerce")
        out["temp_range"] = out["temp_range"].fillna(0.0)
    return out


def add_wind_uv(df: pd.DataFrame, wind_speed_col: str, wind_dir_col: str, wind_from_convention: bool) -> pd.DataFrame:
    out = df.copy()
    ws = pd.to_numeric(out[wind_speed_col], errors="coerce")
    wd = pd.to_numeric(out[wind_dir_col], errors="coerce") % 360.0
    if wind_from_convention:
        wd = (wd + 180.0) % 360.0
    theta = np.deg2rad(wd)
    out["wind_u"] = (ws * np.cos(theta)).fillna(0.0)
    out["wind_v"] = (ws * np.sin(theta)).fillna(0.0)
    return out


def add_delta_and_rolling(df: pd.DataFrame, station_col: str, date_col: str,
                          temp_col: str, humidity_col: str, rainfall_col: str, roll_days: int) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values([station_col, date_col])

    out["delta_temp"] = out.groupby(station_col)[temp_col].diff().fillna(0.0)

    out[f"temp_{roll_days}d_mean"] = (
        out.groupby(station_col)[temp_col]
        .rolling(roll_days, min_periods=1).mean()
        .reset_index(level=0, drop=True)
    )

    out[f"humidity_{roll_days}d_mean"] = (
        out.groupby(station_col)[humidity_col]
        .rolling(roll_days, min_periods=1).mean()
        .reset_index(level=0, drop=True)
    )

    out[f"rain_{roll_days}d_sum"] = (
        out.groupby(station_col)[rainfall_col]
        .rolling(roll_days, min_periods=1).sum()
        .reset_index(level=0, drop=True)
    )

    # Fill any NaN caused by missing source values
    for c in [f"temp_{roll_days}d_mean", f"humidity_{roll_days}d_mean", f"rain_{roll_days}d_sum"]:
        out[c] = out[c].fillna(0.0)

    return out


def detect_var_share_cols(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    cols = df.columns.tolist()
    in_cols = [c for c in cols if c.startswith("var_share_") and c.endswith("_in_season")]
    off_cols = [c for c in cols if c.startswith("var_share_") and c.endswith("_off_season")]
    return in_cols, off_cols


def build_variety_topk(df: pd.DataFrame, in_cols: List[str], off_cols: List[str],
                       topk: int, drop_one: bool) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    out = df.copy()
    info: Dict[str, Any] = {"topk": int(topk), "drop_one": bool(drop_one)}

    def _topk(cols: List[str]) -> List[str]:
        if not cols:
            return []
        means = out[cols].apply(pd.to_numeric, errors="coerce").mean(axis=0).sort_values(ascending=False)
        return means.head(topk).index.tolist()

    top_in = _topk(in_cols)
    top_off = _topk(off_cols)
    info["top_in"] = top_in
    info["top_off"] = top_off

    keep_cols: List[str] = []

    if top_in:
        s = out[top_in].apply(pd.to_numeric, errors="coerce").sum(axis=1)
        out["var_other_in_season"] = (1.0 - s).clip(lower=0.0, upper=1.0).fillna(0.0)
        keep_cols += top_in + ["var_other_in_season"]

    if top_off:
        s_top = out[top_off].apply(pd.to_numeric, errors="coerce").sum(axis=1)
        total_off = out[off_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1) if off_cols else 0.0
        other = (total_off - s_top).clip(lower=0.0)
        other_frac = np.where(total_off > 0, (other / total_off).clip(0.0, 1.0), 0.0)
        out["var_other_off_season"] = pd.Series(other_frac).fillna(0.0)
        keep_cols += top_off + ["var_other_off_season"]

    dropped = {"in": None, "off": None}
    if drop_one:
        if keep_cols:
            if top_in:
                dropped["in"] = keep_cols[0]
                keep_cols = [c for c in keep_cols if c != dropped["in"]]
            off_candidates = [c for c in keep_cols if c.endswith("_off_season") or c == "var_other_off_season"]
            if off_candidates:
                dropped["off"] = off_candidates[0]
                keep_cols = [c for c in keep_cols if c != dropped["off"]]

    info["dropped_one"] = dropped
    info["keep_cols"] = keep_cols
    return out, info


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


def _dates_are_consecutive(dates: np.ndarray) -> bool:
    if len(dates) < 2:
        return True
    diffs = np.diff(dates.astype("datetime64[D]").astype("int64"))
    return bool(np.all(diffs == 1))


def build_sequences_for_station(g: pd.DataFrame, date_col: str, feature_cols: List[str], target_col: str,
                                window: int, horizon: int, require_consecutive: bool):
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


def build_sequences(df: pd.DataFrame, station_col: str, date_col: str, feature_cols: List[str], target_col: str,
                    window: int, horizon: int, require_consecutive: bool) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for split_name in ["train","val","test"]:
        d = df[df["split"] == split_name].copy()
        X_all, y_all, yd_all, st_all = [], [], [], []
        for st, g in d.groupby(station_col):
            X, y, yd = build_sequences_for_station(g, date_col, feature_cols, target_col, window, horizon, require_consecutive)
            if len(X) == 0:
                continue
            X_all.append(X); y_all.append(y); yd_all.append(yd)
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


def select_feature_set(df: pd.DataFrame, set_name: str, roll_days: int, include_year: bool,
                       include_distance_km: bool, variety_keep_cols: List[str]) -> List[str]:
    cyc = ["month_sin","month_cos","doy_sin","doy_cos"]
    core: List[str] = [c for c in cyc if c in df.columns]
    if include_year and "year" in df.columns:
        core.append("year")
    for c in ["temp","humidity","rainfall","wind_u","wind_v","delta_temp",f"temp_{roll_days}d_mean",f"humidity_{roll_days}d_mean",f"rain_{roll_days}d_sum","temp_range"]:
        if c in df.columns:
            core.append(c)
    if set_name == "core":
        return core

    context = core.copy()
    for c in ["area_rai_in_season","area_rai_off_season","latitude","longitude"]:
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


def sanitize_features(df: pd.DataFrame, station_col: str, date_col: str, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values([station_col, date_col])
    out[cols] = out[cols].replace([np.inf, -np.inf], np.nan)
    # fill using past values within station, then backfill, then zeros
    out[cols] = out.groupby(station_col)[cols].ffill()
    out[cols] = out.groupby(station_col)[cols].bfill()
    out[cols] = out[cols].fillna(0.0)
    return out


def run_one_set(df: pd.DataFrame, set_name: str, date_col: str, station_col: str, target_col: str, feature_cols: List[str],
                window: int, horizon: int, train_ratio: float, val_ratio: float, require_consecutive: bool, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    unique_dates = df[date_col].dropna().sort_values().unique()
    split_info = split_dates_by_ratio(unique_dates, train_ratio=train_ratio, val_ratio=val_ratio)
    df2 = assign_split(df, date_col=date_col, split_info=split_info)

    # sanitize chosen features
    df2 = sanitize_features(df2, station_col, date_col, feature_cols)

    # drop rows with missing target (must be finite)
    df2[target_col] = df2[target_col].replace([np.inf, -np.inf], np.nan)
    df2 = df2.dropna(subset=[target_col])

    # save unscaled
    df2.to_csv(out_dir / "model_ready_daily.csv", index=False, encoding="utf-8-sig")

    # scale features (fit on TRAIN)
    train_mask = (df2["split"] == "train")
    scaler = MinMaxScaler()
    scaler.fit(df2.loc[train_mask, feature_cols].values.astype(np.float32))

    scaled = df2.copy()
    scaled.loc[:, feature_cols] = scaler.transform(df2[feature_cols].values.astype(np.float32))
    dump({"scaler": scaler, "feature_cols": feature_cols}, out_dir / "scaler_minmax.joblib")
    scaled.to_csv(out_dir / "model_ready_daily_scaled.csv", index=False, encoding="utf-8-sig")

    seq = build_sequences(scaled, station_col, date_col, feature_cols, target_col, window, horizon, require_consecutive)

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
    np.savez_compressed(out_dir / f"sequences_window{window}_h{horizon}.npz", **seq, meta=np.array([json.dumps(meta, ensure_ascii=False)], dtype=object))
    (out_dir / "report_model_ready.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--input_csv", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--set", default="core", choices=["core","context","full"])
    p.add_argument("--all_sets", action="store_true")

    p.add_argument("--target", default="bph_log1p", choices=["bph_log1p","bph_raw"])
    p.add_argument("--window", type=int, default=30)
    p.add_argument("--horizon", type=int, default=1)
    p.add_argument("--roll_days", type=int, default=7)

    p.add_argument("--train_ratio", type=float, default=0.70)
    p.add_argument("--val_ratio", type=float, default=0.15)
    p.add_argument("--require_consecutive", action="store_true")

    p.add_argument("--include_year", action="store_true")
    p.add_argument("--include_distance_km", action="store_true")
    p.add_argument("--wind_from_convention", action="store_true")

    p.add_argument("--topk_var", type=int, default=8)
    p.add_argument("--drop_one_var_share", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    input_csv = Path(args.input_csv).expanduser().resolve()
    out_root = Path(args.out_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    df = read_csv_any(input_csv)

    date_col = pick_first_existing(df, ["date","Date","datetime","timestamp"])
    station_col = pick_first_existing(df, ["station_id","StationID","station","stationId"])
    if not date_col or not station_col:
        raise RuntimeError("Cannot detect date/station columns")

    base_bph_col = pick_first_existing(df, ["bph_count","BPH_count","bph_raw"])
    if not base_bph_col:
        raise RuntimeError("Cannot detect bph_count column")

    temp_col = pick_first_existing(df, ["temp","temperature","tmean"])
    hum_col = pick_first_existing(df, ["humidity","rh"])
    rain_col = pick_first_existing(df, ["rainfall","rain"])
    ws_col = pick_first_existing(df, ["wind_speed","windspeed","ws"])
    wd_col = pick_first_existing(df, ["wind_direction","wind_dir","wd","winddirection"])
    mint_col = pick_first_existing(df, ["mint_temp","tmin","min_temp"])
    maxt_col = pick_first_existing(df, ["maxt_temp","tmax","max_temp"])

    df = ensure_datetime(df, date_col)
    df = safe_numeric(df, [base_bph_col,temp_col,hum_col,rain_col,ws_col,wd_col,mint_col,maxt_col])

    # IMPORTANT: drop missing target base
    df = df.dropna(subset=[base_bph_col])

    df = add_bph_targets(df, base_col=base_bph_col)
    df = add_time_features(df, date_col=date_col)
    df = add_temp_range(df, mint_col=mint_col, maxt_col=maxt_col)
    df = add_wind_uv(df, wind_speed_col=ws_col, wind_dir_col=wd_col, wind_from_convention=bool(args.wind_from_convention))
    df = add_delta_and_rolling(df, station_col, date_col, temp_col, hum_col, rain_col, int(args.roll_days))

    in_cols, off_cols = detect_var_share_cols(df)
    var_info: Dict[str, Any] = {}
    variety_keep_cols: List[str] = []
    if args.all_sets or args.set == "full":
        df, var_info = build_variety_topk(df, in_cols, off_cols, int(args.topk_var), bool(args.drop_one_var_share))
        variety_keep_cols = var_info.get("keep_cols", [])

    sets = ["core","context","full"] if args.all_sets else [args.set]
    summary = {"input_csv": str(input_csv), "sets": [], "variety_info": var_info}

    for s in sets:
        feature_cols = select_feature_set(df, s, int(args.roll_days), bool(args.include_year), bool(args.include_distance_km), variety_keep_cols)
        keep_cols = list(dict.fromkeys([date_col, station_col, "bph_raw", "bph_log1p"] + feature_cols))
        df_base = df[[c for c in keep_cols if c in df.columns]].copy()

        out_dir = out_root / s
        meta = run_one_set(df_base, s, date_col, station_col, str(args.target), feature_cols,
                           int(args.window), int(args.horizon), float(args.train_ratio), float(args.val_ratio),
                           bool(args.require_consecutive), out_dir)
        summary["sets"].append(meta)
        print(f"[OK] {s}: {meta['counts']}")

    (out_root / "summary_feature_sets.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[OK] summary_feature_sets.json written")

if __name__ == "__main__":
    main()
