#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""01_make_model_ready_v3_stream.py

โปรแกรม A) นำเข้า RAW CSV -> สร้างฟีเจอร์อนุพันธ์ -> split ตามเวลา -> scale -> sliding window

อินพุต:
  --raw_csv  ไฟล์ RAW รายวัน (เช่น env_daily_with_rice_monthly_raw.csv)
เอาต์พุต (ใน --out_dir):
  - model_ready_daily_scaled_mincols.csv
  - scaler_minmax.joblib
  - feature_columns.json
  - build_report.json
  - sequences/X_train.npy, y_train.npy, X_val.npy, y_val.npy, X_test.npy, y_test.npy
  - sequences/sequences_meta.json

หมายเหตุเรื่อง split + window:
- split ทำตาม "วันของ target" (target date อยู่ train/val/test อันไหน ก็เก็บ sample ไปอันนั้น)
- window อนุญาตให้ย้อนกลับไปใช้วันก่อนหน้าได้ แม้จะอยู่ก่อนขอบเขต split (ใช้บริบทอดีตได้เหมือนงานพยากรณ์จริง)
- ถ้าต้องการเข้มงวดให้ window อยู่ใน split เดียวกันทั้งหมด ใช้ --strict_split_windows
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ENCODINGS = ("utf-8-sig", "utf-8", "cp874", "tis-620")


def read_csv_any(path: str | Path) -> pd.DataFrame:
    """อ่าน CSV โดยลอง encoding หลายแบบ (กันไฟล์ไทย cp874/tis-620)"""
    path = str(path)
    last_err = None
    for enc in ENCODINGS:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e
    # fallback
    try:
        return pd.read_csv(path)
    except Exception:
        raise RuntimeError(f"Cannot read CSV: {path} (last_err={last_err})")


def ensure_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    """ตรวจคอลัมน์สำคัญ และจัดชนิดข้อมูลเบื้องต้น"""
    required = {"date", "station_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"RAW file missing required columns: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()
    df["station_id"] = df["station_id"].astype(str).str.strip()
    return df


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """สร้าง bph_raw และ bph_log1p จาก bph_count"""
    if "bph_count" not in df.columns:
        # ถ้าไม่มี ก็สร้างเป็น NaN ไว้ก่อน (ผู้ใช้สามารถแก้ชื่อคอลัมน์ได้)
        df["bph_count"] = np.nan

    df["bph_raw"] = pd.to_numeric(df["bph_count"], errors="coerce").astype(float)
    df["bph_raw"] = df["bph_raw"].clip(lower=0)
    df["bph_log1p"] = np.log1p(df["bph_raw"])
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """สร้าง month/doy + sin/cos จากคอลัมน์ date"""
    df["year"] = df["date"].dt.year
    df["month_num"] = df["date"].dt.month
    df["doy"] = df["date"].dt.dayofyear

    # cyclic features
    df["month_sin"] = np.sin(2 * np.pi * df["month_num"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month_num"] / 12.0)
    df["doy_sin"] = np.sin(2 * np.pi * df["doy"] / 365.0)
    df["doy_cos"] = np.cos(2 * np.pi * df["doy"] / 365.0)
    return df


def add_wind_uv(df: pd.DataFrame) -> pd.DataFrame:
    """สร้าง wind_u/wind_v จาก wind_speed และ wind_direction (องศา)"""
    if "wind_speed" in df.columns and "wind_direction" in df.columns:
        ws = pd.to_numeric(df["wind_speed"], errors="coerce")
        wd = pd.to_numeric(df["wind_direction"], errors="coerce")
        theta = np.deg2rad(wd)
        df["wind_u"] = ws * np.sin(theta)
        df["wind_v"] = ws * np.cos(theta)
    else:
        df["wind_u"] = np.nan
        df["wind_v"] = np.nan
    return df

def add_station_derivatives(df: pd.DataFrame, roll_days: int) -> pd.DataFrame:
    """สร้างฟีเจอร์อนุพันธ์แบบรายสถานี (diff/rolling)

    - delta_temp = temp(t) - temp(t-1) ภายในสถานีเดียวกัน
    - temp_{roll_days}d_mean, humidity_{roll_days}d_mean
    - rain_{roll_days}d_sum
    """
    if "station_id" not in df.columns:
        raise ValueError("RAW ต้องมีคอลัมน์ station_id")

    df = df.sort_values(["station_id", "date"]).reset_index(drop=True)

    # temp
    if "temp" in df.columns:
        df["temp"] = pd.to_numeric(df["temp"], errors="coerce")
        df["delta_temp"] = df.groupby("station_id")["temp"].diff()
        df[f"temp_{roll_days}d_mean"] = df.groupby("station_id")["temp"].transform(
            lambda s: s.rolling(roll_days, min_periods=1).mean()
        )
    else:
        df["delta_temp"] = np.nan
        df[f"temp_{roll_days}d_mean"] = np.nan

    # humidity
    if "humidity" in df.columns:
        df["humidity"] = pd.to_numeric(df["humidity"], errors="coerce")
        df[f"humidity_{roll_days}d_mean"] = df.groupby("station_id")["humidity"].transform(
            lambda s: s.rolling(roll_days, min_periods=1).mean()
        )
    else:
        df[f"humidity_{roll_days}d_mean"] = np.nan

    # rainfall
    if "rainfall" in df.columns:
        df["rainfall"] = pd.to_numeric(df["rainfall"], errors="coerce")
        df[f"rain_{roll_days}d_sum"] = df.groupby("station_id")["rainfall"].transform(
            lambda s: s.rolling(roll_days, min_periods=1).sum()
        )
    else:
        df[f"rain_{roll_days}d_sum"] = np.nan

    return df

# -----------------------------
# Plot helpers (A-D)
# -----------------------------

def _auto_pick_station(df: pd.DataFrame) -> str:
    """Pick a representative station: station with the most records."""
    vc = df["station_id"].value_counts(dropna=True)
    return str(vc.index[0]) if len(vc) else ""


def _sample_df(df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    """Downsample rows for faster plotting."""
    if n <= 0 or len(df) <= n:
        return df
    return df.sample(n=n, random_state=seed)


def _save_fig(fig, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def make_plots_A_to_D(
    df: pd.DataFrame,
    out_dir: Path,
    target_col: str,
    roll_days: int,
    plot_station: str = "",
    sample_n: int = 5000,
) -> Path:
    """Create presentation-friendly plots for parts A-D.

    A) load/target overview
    B) time cyclic features (seasonality)
    C) wind vector features
    D) derivatives/rolling features
    """
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    station = plot_station.strip() or _auto_pick_station(df)
    dft = df.copy()
    if "date" in dft.columns:
        dft = dft.sort_values("date")

    created = []

    # ---------- A: target overview ----------
    if target_col in dft.columns:
        dft_y = dft.dropna(subset=[target_col])

        # A1) timeseries for one station
        if station and "station_id" in dft_y.columns:
            d1 = dft_y[dft_y["station_id"].astype(str) == station].sort_values("date")
            if len(d1) >= 10:
                fig = plt.figure()
                plt.plot(d1["date"], d1[target_col])
                plt.xlabel("date")
                plt.ylabel(target_col)
                plt.title(f"Target timeseries (station={station})")
                p = fig_dir / "A1_timeseries_target_sample_station.png"
                _save_fig(fig, p)
                created.append(str(p))

        # A2) target distribution
        if len(dft_y) >= 10:
            fig = plt.figure()
            plt.hist(dft_y[target_col].astype(float), bins=50)
            plt.xlabel(target_col)
            plt.ylabel("count")
            plt.title("Target distribution")
            p = fig_dir / "A2_target_distribution.png"
            _save_fig(fig, p)
            created.append(str(p))

    # A3) missingness (key columns)
    key_cols = [c for c in ["temp", "humidity", "rainfall", "wind_speed", "wind_direction", target_col] if c in dft.columns]
    if key_cols:
        miss_pct = (dft[key_cols].isna().mean() * 100).sort_values(ascending=False)
        fig = plt.figure()
        plt.bar(miss_pct.index.astype(str), miss_pct.values)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("missing (%)")
        plt.title("Missingness of key columns")
        p = fig_dir / "A3_missingness_key_columns.png"
        _save_fig(fig, p)
        created.append(str(p))

    # ---------- B: seasonality (month) ----------
    if "month_num" in dft.columns and target_col in dft.columns:
        dft_y = dft.dropna(subset=[target_col])
        if len(dft_y) >= 10:
            g = dft_y.groupby("month_num")[target_col].agg(["mean", "median"]).reset_index()
            fig = plt.figure()
            plt.plot(g["month_num"], g["mean"], marker="o", label="mean")
            plt.plot(g["month_num"], g["median"], marker="o", label="median")
            plt.xlabel("month")
            plt.ylabel(target_col)
            plt.title("Seasonality by month")
            plt.legend()
            p = fig_dir / "B1_seasonality_by_month.png"
            _save_fig(fig, p)
            created.append(str(p))

    # ---------- C: wind features ----------
    if "wind_direction" in dft.columns:
        wd = pd.to_numeric(dft["wind_direction"], errors="coerce").dropna()
        if len(wd) >= 10:
            fig = plt.figure()
            plt.hist(wd, bins=36)
            plt.xlabel("wind_direction (deg)")
            plt.ylabel("count")
            plt.title("Wind direction distribution")
            p = fig_dir / "C1_wind_direction_hist.png"
            _save_fig(fig, p)
            created.append(str(p))

    if "wind_u" in dft.columns and "wind_v" in dft.columns:
        d2 = dft[["wind_u", "wind_v"]].copy()
        d2["wind_u"] = pd.to_numeric(d2["wind_u"], errors="coerce")
        d2["wind_v"] = pd.to_numeric(d2["wind_v"], errors="coerce")
        d2 = d2.dropna()
        d2 = _sample_df(d2, sample_n)
        if len(d2) >= 10:
            fig = plt.figure()
            plt.scatter(d2["wind_u"], d2["wind_v"], s=8)
            plt.xlabel("wind_u")
            plt.ylabel("wind_v")
            plt.title("Wind vector scatter (u vs v)")
            p = fig_dir / "C2_wind_uv_scatter.png"
            _save_fig(fig, p)
            created.append(str(p))

    # ---------- D: derivatives / rolling ----------
    if "delta_temp" in dft.columns:
        dt = pd.to_numeric(dft["delta_temp"], errors="coerce").dropna()
        if len(dt) >= 10:
            fig = plt.figure()
            plt.hist(dt, bins=60)
            plt.xlabel("delta_temp")
            plt.ylabel("count")
            plt.title("Distribution of delta_temp")
            p = fig_dir / "D1_delta_temp_distribution.png"
            _save_fig(fig, p)
            created.append(str(p))

    # D2) temp vs rolling mean timeseries (one station)
    roll_temp_col = f"temp_{roll_days}d_mean"
    if station and all(c in dft.columns for c in ["station_id", "date", "temp", roll_temp_col]):
        d1 = dft[dft["station_id"].astype(str) == station].sort_values("date")
        d1 = d1.dropna(subset=["temp", roll_temp_col])
        if len(d1) >= 10:
            fig = plt.figure()
            plt.plot(d1["date"], d1["temp"], label="temp")
            plt.plot(d1["date"], d1[roll_temp_col], label=roll_temp_col)
            plt.xlabel("date")
            plt.ylabel("temp")
            plt.title(f"Temp vs rolling mean (station={station})")
            plt.legend()
            p = fig_dir / "D2_temp_and_rolling_mean_timeseries.png"
            _save_fig(fig, p)
            created.append(str(p))

    # D3) rainfall vs rolling sum timeseries (one station)
    roll_rain_col = f"rain_{roll_days}d_sum"
    if station and all(c in dft.columns for c in ["station_id", "date", "rainfall", roll_rain_col]):
        d1 = dft[dft["station_id"].astype(str) == station].sort_values("date")
        d1 = d1.dropna(subset=["rainfall", roll_rain_col])
        if len(d1) >= 10:
            fig = plt.figure()
            plt.plot(d1["date"], d1["rainfall"], label="rainfall")
            plt.plot(d1["date"], d1[roll_rain_col], label=roll_rain_col)
            plt.xlabel("date")
            plt.ylabel("rain")
            plt.title(f"Rainfall vs rolling sum (station={station})")
            plt.legend()
            p = fig_dir / "D3_rain_and_rolling_sum_timeseries.png"
            _save_fig(fig, p)
            created.append(str(p))

    # D4) scatter temp vs target (sample)
    if target_col in dft.columns and "temp" in dft.columns:
        d3 = dft[["temp", target_col]].copy()
        d3["temp"] = pd.to_numeric(d3["temp"], errors="coerce")
        d3[target_col] = pd.to_numeric(d3[target_col], errors="coerce")
        d3 = d3.dropna()
        d3 = _sample_df(d3, sample_n)
        if len(d3) >= 10:
            fig = plt.figure()
            plt.scatter(d3["temp"], d3[target_col], s=8)
            plt.xlabel("temp")
            plt.ylabel(target_col)
            plt.title("Scatter: temp vs target")
            p = fig_dir / "D4_scatter_temp_vs_target.png"
            _save_fig(fig, p)
            created.append(str(p))

    summary = {
        "plot_station": station,
        "target": target_col,
        "roll_days": roll_days,
        "created": created,
    }
    summary_path = out_dir / "plots_A_to_D_summary.json"
    import json as _json
    summary_path.write_text(_json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary_path




def time_split_boundaries(df: pd.DataFrame, train_frac: float, val_frac: float):
    """หาวันสิ้นสุด train/val จากลิสต์วันทั้งหมด (global timeline)"""
    dates = np.array(sorted(df["date"].dt.normalize().unique()))
    n = len(dates)
    if n < 10:
        raise ValueError("ข้อมูลวันน้อยเกินไปสำหรับ split")

    train_idx = max(0, int(np.floor(n * train_frac)) - 1)
    val_idx = max(0, int(np.floor(n * (train_frac + val_frac))) - 1)

    train_end = pd.Timestamp(dates[train_idx])
    val_end = pd.Timestamp(dates[val_idx])
    test_end = pd.Timestamp(dates[-1])
    return train_end, val_end, test_end


def assign_split(df: pd.DataFrame, train_end: pd.Timestamp, val_end: pd.Timestamp) -> pd.DataFrame:
    """เพิ่มคอลัมน์ split = train/val/test ตามวันของแถว"""
    d = df["date"].dt.normalize()
    df["split"] = np.where(d <= train_end, "train", np.where(d <= val_end, "val", "test"))
    return df


def get_feature_columns(df: pd.DataFrame, target_col: str) -> list[str]:
    """เลือกเฉพาะคอลัมน์ numeric ที่จะใช้เป็น feature

    - ตัดคอลัมน์ ID/ข้อความ และตัดคอลัมน์ target/ฉลากทิ้ง
    - เก็บ monthly/variety share (ซึ่งเป็น numeric) ไว้ทั้งหมด
    """
    exclude = {
        "date",
        "station_id",
        "province_en",
        "province_th",
        "province_th_pm",
        "split",
        "bph_count",
        "bph_raw",
        "bph_log1p",
        target_col,
    }

    feature_cols = [
        c
        for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]
    # จัดลำดับให้คงที่
    feature_cols = sorted(feature_cols)
    return feature_cols


def scale_minmax(df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, MinMaxScaler]:
    """MinMaxScaler fit เฉพาะ TRAIN แล้ว transform ทุก split"""
    train_mask = df["split"] == "train"
    scaler = MinMaxScaler()

    X_train = df.loc[train_mask, feature_cols].astype(float)
    scaler.fit(X_train)

    df_scaled = df.copy()
    df_scaled[feature_cols] = scaler.transform(df_scaled[feature_cols].astype(float))
    return df_scaled, scaler


def _count_sequences_station(
    g: pd.DataFrame,
    window: int,
    horizon: int,
    target_col: str,
    strict_split_windows: bool,
) -> dict[str, int]:
    """นับจำนวน sample ต่อ split ภายในสถานีเดียว (pass แรก)"""
    counts = {"train": 0, "val": 0, "test": 0}
    n = len(g)
    splits = g["split"].astype(str).to_numpy()
    y = pd.to_numeric(g[target_col], errors="coerce").to_numpy()

    for i in range(window, n - horizon + 1):
        target_idx = i + horizon - 1
        split = splits[target_idx]
        if np.isnan(y[target_idx]):
            continue
        if strict_split_windows:
            if len(set(splits[i - window : target_idx + 1])) != 1:
                continue
        counts[split] += 1

    return counts


def build_sequences_memmap(
    df_scaled: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    window: int,
    horizon: int,
    out_dir: Path,
    strict_split_windows: bool,
):
    """สร้าง sliding window แล้วบันทึกเป็น .npy (memmap) แยก train/val/test

    X shape = [samples, window, n_features]
    y shape = [samples]
    """
    seq_dir = out_dir / "sequences"
    seq_dir.mkdir(parents=True, exist_ok=True)

    df_scaled = df_scaled.sort_values(["station_id", "date"]).reset_index(drop=True)
    n_feat = len(feature_cols)

    # pass 1: count
    totals = {"train": 0, "val": 0, "test": 0}
    for _, g in df_scaled.groupby("station_id", sort=False):
        g = g.reset_index(drop=True)
        c = _count_sequences_station(g, window, horizon, target_col, strict_split_windows)
        for k in totals:
            totals[k] += c[k]

    # allocate memmap
    arrays = {}
    for split in ("train", "val", "test"):
        X_path = seq_dir / f"X_{split}.npy"
        y_path = seq_dir / f"y_{split}.npy"
        X = np.lib.format.open_memmap(
            X_path, mode="w+", dtype=np.float32, shape=(totals[split], window, n_feat)
        )
        y = np.lib.format.open_memmap(
            y_path, mode="w+", dtype=np.float32, shape=(totals[split],)
        )
        arrays[split] = {"X": X, "y": y, "pos": 0, "X_path": str(X_path), "y_path": str(y_path)}

    # pass 2: fill
    for _, g in df_scaled.groupby("station_id", sort=False):
        g = g.reset_index(drop=True)
        feats = g[feature_cols].astype(np.float32).to_numpy()
        y_all = pd.to_numeric(g[target_col], errors="coerce").astype("float32").to_numpy()
        splits = g["split"].astype(str).to_numpy()

        for i in range(window, len(g) - horizon + 1):
            target_idx = i + horizon - 1
            split = splits[target_idx]
            yv = y_all[target_idx]
            if np.isnan(yv):
                continue

            if strict_split_windows:
                if len(set(splits[i - window : target_idx + 1])) != 1:
                    continue

            Xwin = feats[i - window : i, :]
            pos = arrays[split]["pos"]
            arrays[split]["X"][pos, :, :] = Xwin
            arrays[split]["y"][pos] = yv
            arrays[split]["pos"] += 1

    # flush + sanity
    for split in arrays:
        arrays[split]["X"].flush()
        arrays[split]["y"].flush()
        if arrays[split]["pos"] != totals[split]:
            print(
                f"WARN: filled {arrays[split]['pos']} but expected {totals[split]} for {split}"
            )

    meta = {
        "window": window,
        "horizon": horizon,
        "target": target_col,
        "n_features": n_feat,
        "feature_columns": feature_cols,
        "strict_split_windows": strict_split_windows,
        "samples": totals,
        "files": {
            s: {"X": arrays[s]["X_path"], "y": arrays[s]["y_path"]}
            for s in ("train", "val", "test")
        },
    }

    meta_path = seq_dir / "sequences_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return totals, meta_path


def main():
    ap = argparse.ArgumentParser(
        description="Build model-ready dataset + sliding windows from raw daily CSV"
    )
    ap.add_argument("--raw_csv", required=True, help="Path to raw merged daily CSV")
    ap.add_argument("--out_dir", required=True, help="Output directory")

    ap.add_argument(
        "--target",
        default="bph_log1p",
        choices=["bph_log1p", "bph_raw"],
        help="Target column",
    )
    ap.add_argument("--window", type=int, default=30, help="Sliding window length (days)")
    ap.add_argument("--horizon", type=int, default=1, help="Forecast horizon (days)")
    ap.add_argument("--roll_days", type=int, default=7, help="Rolling window days")

    ap.add_argument("--train_frac", type=float, default=0.70)
    ap.add_argument("--val_frac", type=float, default=0.15)

    ap.add_argument(
        "--strict_split_windows",
        action="store_true",
        help="If set: window+target must be entirely inside same split",
    )
    ap.add_argument(
        "--save_unscaled_csv",
        action="store_true",
        help="If set: save unscaled model-ready daily csv too",
    )

    ap.add_argument(
        "--make_plots",
        action="store_true",
        help="If set: create presentation plots for parts A-D",
    )
    ap.add_argument(
        "--plot_station",
        default="",
        help="Optional station_id for plots (default: auto pick)",
    )
    ap.add_argument(
        "--plot_sample_n",
        type=int,
        default=5000,
        help="Downsample size for scatter plots",
    )

    args = ap.parse_args()

    raw_csv = Path(args.raw_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) read
    df = read_csv_any(raw_csv)

    # 2) normalize minimal columns
    if "date" not in df.columns:
        raise ValueError("RAW ต้องมีคอลัมน์ date")
    if "station_id" not in df.columns:
        raise ValueError("RAW ต้องมีคอลัมน์ station_id")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()

    # 3) target + features
    df = add_targets(df)
    df = add_time_features(df)
    df = add_wind_uv(df)
    df = add_station_derivatives(df, roll_days=args.roll_days)

    # (optional) plots for parts A-D
    if args.make_plots:
        make_plots_A_to_D(
            df=df,
            out_dir=out_dir,
            target_col=args.target,
            roll_days=args.roll_days,
            plot_station=args.plot_station,
            sample_n=args.plot_sample_n,
        )

    # เลือก target
    target_col = args.target

    # 4) split
    train_end, val_end, test_end = time_split_boundaries(df, args.train_frac, args.val_frac)
    df = assign_split(df, train_end, val_end)

    # 5) choose feature columns
    feature_cols = get_feature_columns(df, target_col=target_col)

    # ตัดแถวที่ไม่มี target (เพื่อไม่ให้มี y=NaN)
    df = df.dropna(subset=[target_col]).copy()

    # 6) scale (fit train only)
    df_scaled, scaler = scale_minmax(df, feature_cols)

    # 7) save tables + scaler + feature list
    if args.save_unscaled_csv:
        unscaled_path = out_dir / "model_ready_daily_unscaled.csv"
        df.to_csv(unscaled_path, index=False, encoding="utf-8-sig")

    # เก็บเฉพาะคอลัมน์สำคัญ + feature + target + split
    keep_cols = ["date", "station_id", "split", target_col] + feature_cols
    scaled_path = out_dir / "model_ready_daily_scaled_mincols.csv"
    df_scaled[keep_cols].to_csv(scaled_path, index=False, encoding="utf-8-sig")

    joblib.dump(scaler, out_dir / "scaler_minmax.joblib")
    with open(out_dir / "feature_columns.json", "w", encoding="utf-8") as f:
        json.dump({"feature_columns": feature_cols}, f, ensure_ascii=False, indent=2)

    # 8) sliding windows
    totals, meta_path = build_sequences_memmap(
        df_scaled,
        feature_cols=feature_cols,
        target_col=target_col,
        window=args.window,
        horizon=args.horizon,
        out_dir=out_dir,
        strict_split_windows=args.strict_split_windows,
    )

    # 9) report
    report = {
        "raw_csv": str(raw_csv),
        "rows_after_dropna_date": int(len(df)),
        "stations": int(df["station_id"].nunique()),
        "date_min": str(df["date"].min().date()),
        "date_max": str(df["date"].max().date()),
        "split_boundaries": {
            "train_end": str(train_end.date()),
            "val_end": str(val_end.date()),
            "test_end": str(test_end.date()),
        },
        "target": target_col,
        "window": args.window,
        "horizon": args.horizon,
        "roll_days": args.roll_days,
        "n_features": len(feature_cols),
        "sequence_samples": {k: int(v) for k, v in totals.items()},
        "outputs": {
            "scaled_daily_csv": str(scaled_path),
            "scaler": str(out_dir / "scaler_minmax.joblib"),
            "feature_columns": str(out_dir / "feature_columns.json"),
            "sequences_meta": str(meta_path),
        },
    }

    with open(out_dir / "build_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("DONE")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
