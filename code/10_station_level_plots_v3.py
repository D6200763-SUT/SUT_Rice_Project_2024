#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Station-level EDA plots for BPH time-series (raw or model-ready).

This script focuses on *station-level* views to help you present:
- data quality per station,
- target behavior per station,
- feature vs target patterns per station,
- per-station correlations (Pearson/Spearman).

Outputs are saved into <out_dir>/figures and tables into <out_dir>.

Usage example:
python 10_station_level_plots.py \
  --input_csv "Import_Dataset/env_daily_with_rice_monthly_raw.csv" \
  --out_dir "out_station_eda" \
  --target bph_log1p \
  --features temp,humidity,rainfall,wind_speed,rain_7d_sum,temp_7d_mean \
  --top_k_stations 8 \
  --plot_sample_n 5000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless backend for saving PNG
import matplotlib.pyplot as plt


# ----------------------------
# Part 0: Robust CSV reader
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


# ----------------------------
# Part 1: Column auto-detection
# ----------------------------
def pick_first_existing(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def detect_columns(df: pd.DataFrame, target: str) -> dict:
    date_col = pick_first_existing(df, ["date", "Date", "datetime", "timestamp"])
    station_col = pick_first_existing(df, ["station_id", "StationID", "station", "stationId"])
    if target in df.columns:
        target_col = target
    else:
        target_col = pick_first_existing(df, ["bph_log1p", "bph_raw", "bph_count", "BPH_count"])
    lat_col = pick_first_existing(df, ["latitude", "lat", "Latitude", "LAT"])
    lon_col = pick_first_existing(df, ["longitude", "lon", "Longitude", "LON"])
    return {
        "date_col": date_col,
        "station_col": station_col,
        "target_col": target_col,
        "lat_col": lat_col,
        "lon_col": lon_col,
    }


# ----------------------------
# Part 2: Utility helpers
# ----------------------------
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

def auto_make_bph_targets(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Create bph_raw / bph_log1p if missing but bph_count exists."""
    out = df.copy()
    if target_col == "bph_log1p" and "bph_log1p" not in out.columns:
        base = pick_first_existing(out, ["bph_raw", "bph_count", "BPH_count"])
        if base is not None:
            out["bph_raw"] = pd.to_numeric(out[base], errors="coerce").clip(lower=0)
            out["bph_log1p"] = np.log1p(out["bph_raw"])
    if target_col == "bph_raw" and "bph_raw" not in out.columns:
        base = pick_first_existing(out, ["bph_count", "BPH_count"])
        if base is not None:
            out["bph_raw"] = pd.to_numeric(out[base], errors="coerce").clip(lower=0)
    return out

def sample_df(df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    if n is None or n <= 0 or len(df) <= n:
        return df
    return df.sample(n=n, random_state=seed)

def save_fig(fig: plt.Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


# ----------------------------
# Part 2b: Station list helpers (NEW)
# ----------------------------
def parse_station_list_arg(stations_arg: str) -> list[str]:
    """
    Parse --stations argument: comma-separated station IDs.
    Example: --stations 101,205,330
    """
    if not stations_arg:
        return []
    items = [x.strip() for x in str(stations_arg).split(",")]
    return [x for x in items if x]

def load_stations_from_file(path: str | Path) -> list[str]:
    """
    Load station IDs from a txt/csv file.

    - .txt: one station per line OR comma-separated in one line
    - .csv: must contain a column like station_id/station/StationID, otherwise uses the first column
    """
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"stations_file not found: {p}")

    if p.suffix.lower() in [".txt", ".list"]:
        txt = p.read_text(encoding="utf-8", errors="ignore")
        raw = []
        for line in txt.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            raw.extend([x.strip() for x in line.split(",") if x.strip()])
        return [x for x in raw if x]

    sep = "\t" if p.suffix.lower() == ".tsv" else ","
    df = read_csv_any(p, sep=sep)
    if df.empty:
        return []

    col = pick_first_existing(df, ["station_id", "station", "StationID", "stationId"])
    if col is None:
        col = df.columns[0]

    vals = df[col].astype(str).str.strip()
    vals = vals[vals != ""]
    out = []
    for v in vals.tolist():
        out.extend([x.strip() for x in v.split(",") if x.strip()])
    return out

def unique_preserve_order(items: list[str]) -> list[str]:
    seen = set()
    out = []
    for x in items:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out

# ----------------------------
# Part 3: Station selection
# ----------------------------
def pick_stations(df: pd.DataFrame, station_col: str, how: str, k: int) -> list[str]:
    """Pick stations to visualize."""
    if how == "most_rows":
        vc = df[station_col].value_counts(dropna=True)
        return [str(x) for x in vc.index[:k].tolist()]
    return []


# ----------------------------
# Part 4: Plots (A1-A5)
# ----------------------------
def plot_timeseries_target(df, date_col, station_col, target_col, stations, out_dir) -> str:
    """A1: Time-series of target for selected stations."""
    fig, ax = plt.subplots(figsize=(12, 5))
    created = []
    for st in stations:
        g = df[df[station_col].astype(str) == st].sort_values(date_col)
        g = g.dropna(subset=[target_col])
        if len(g) == 0:
            continue
        ax.plot(g[date_col], g[target_col], label=st)
        created.append(st)
    ax.set_title(f"Target time-series by station ({target_col})")
    ax.set_xlabel("Date")
    ax.set_ylabel(target_col)
    if len(created) <= 10 and len(created) > 0:
        ax.legend(loc="best", fontsize=8)
    out_path = out_dir / "figures" / "A1_timeseries_target_by_station.png"
    save_fig(fig, out_path)
    return str(out_path)

def plot_timeseries_target_per_station(
    df: pd.DataFrame,
    date_col: str,
    station_col: str,
    target_col: str,
    stations: list[str],
    out_dir: Path,
    max_points: int = 0
) -> list[str]:
    """
    A1b: Time-series of target, saved as ONE FIGURE PER STATION.

    - stations: list of station_id to plot
    - max_points: if > 0, downsample by taking evenly-spaced points to reduce file size
    Returns list of created file paths (as strings).
    """
    out_paths: list[str] = []
    fig_dir = out_dir / "figures" / "A1_per_station"
    fig_dir.mkdir(parents=True, exist_ok=True)

    for st in stations:
        g = df[df[station_col].astype(str) == str(st)].sort_values(date_col)
        g = g.dropna(subset=[target_col])
        if len(g) == 0:
            continue

        # optional downsample (evenly spaced)
        if max_points and max_points > 0 and len(g) > max_points:
            step = max(1, len(g) // max_points)
            g = g.iloc[::step, :]

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(g[date_col], g[target_col])
        ax.set_title(f"{target_col} time-series | station={st}")
        ax.set_xlabel("Date")
        ax.set_ylabel(target_col)

        # safer filename
        safe_st = str(st).replace("/", "_").replace("\\", "_").replace(" ", "_")
        out_path = fig_dir / f"A1_timeseries_{safe_st}.png"
        save_fig(fig, out_path)
        out_paths.append(str(out_path))

    return out_paths


def plot_missingness_per_station(df, station_col, key_cols, out_dir) -> str:
    """A2: Missingness (%) per station for key columns."""
    key_cols = [c for c in key_cols if c in df.columns]
    if len(key_cols) == 0:
        return ""

    grp = df.groupby(station_col)
    miss = grp[key_cols].apply(lambda x: x.isna().mean() * 100.0).reset_index()
    miss_long = miss.melt(id_vars=[station_col], var_name="column", value_name="missing_pct")

    fig, ax = plt.subplots(figsize=(12, 5))
    stations = miss[station_col].astype(str).tolist()
    x = np.arange(len(stations))
    for col in key_cols:
        y = miss_long[miss_long["column"] == col]["missing_pct"].values
        ax.plot(x, y, marker="o", linewidth=1, label=col)
    ax.set_title("Missingness (%) per station for key columns")
    ax.set_xlabel("Station (index)")
    ax.set_ylabel("Missing (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(stations, rotation=90, fontsize=7)
    ax.legend(loc="best", fontsize=8)
    out_path = out_dir / "figures" / "A2_missingness_per_station.png"
    save_fig(fig, out_path)
    return str(out_path)

def plot_target_boxplot_by_station(
    df: pd.DataFrame,
    station_col: str,
    target_col: str,
    top_k: int,
    out_dir: Path,
    stations_override: Optional[list[str]] = None
) -> str:
    """A3: Boxplot of target by station.

    - If stations_override is provided, use that station list (up to top_k).
    - Otherwise, use top_k stations by row count.
    """
    # pick stations to show in boxplot
    if stations_override and len(stations_override) > 0:
        stations = [str(s) for s in stations_override][:top_k]
    else:
        stations = df[station_col].value_counts(dropna=True).index[:top_k].astype(str).tolist()

    data, labels = [], []
    for st in stations:
        vals = df[df[station_col].astype(str) == st][target_col].dropna().values
        if len(vals) == 0:
            continue
        data.append(vals)
        labels.append(st)

    if len(data) == 0:
        return ""

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.boxplot(data, labels=labels, showfliers=True)
    ax.set_title(f"Boxplot of {target_col} by station (top {top_k})")
    ax.set_xlabel("Station")
    ax.set_ylabel(target_col)
    ax.tick_params(axis="x", rotation=90, labelsize=7)

    out_path = out_dir / "figures" / "A3_boxplot_target_by_station_topk.png"
    save_fig(fig, out_path)
    return str(out_path)


def plot_feature_vs_target_scatter(df, station_col, target_col, feature, stations, sample_n, out_dir) -> str:
    """A4: Scatter feature vs target for selected stations (sampled)."""
    if feature not in df.columns:
        return ""

    fig, ax = plt.subplots(figsize=(7, 6))
    created = []
    for st in stations:
        g = df[df[station_col].astype(str) == st][[feature, target_col]].dropna()
        g = sample_df(g, sample_n)
        if len(g) == 0:
            continue
        ax.scatter(g[feature].values, g[target_col].values, s=10, alpha=0.6, label=st)
        created.append(st)
    ax.set_title(f"{feature} vs {target_col} (sampled)")
    ax.set_xlabel(feature)
    ax.set_ylabel(target_col)
    if len(created) <= 10 and len(created) > 0:
        ax.legend(loc="best", fontsize=8)
    out_path = out_dir / "figures" / f"A4_scatter_{feature}_vs_{target_col}.png"
    save_fig(fig, out_path)
    return str(out_path)

def corr_per_station(df, station_col, target_col, features, method="pearson") -> pd.DataFrame:
    """A5: Correlation per station (pearson/spearman)."""
    out_rows = []
    for st, g in df.groupby(station_col):
        g = g.dropna(subset=[target_col])
        if len(g) < 10:
            continue
        for f in features:
            if f not in g.columns:
                continue
            gg = g[[f, target_col]].dropna()
            n = len(gg)
            if n < 10:
                continue
            corr = gg[f].corr(gg[target_col], method=("spearman" if method == "spearman" else "pearson"))
            out_rows.append({"station_id": str(st), "feature": f, "corr": corr, "n": int(n), "method": method})
    return pd.DataFrame(out_rows)

def plot_corr_heatmap_station_feature(corr_df, out_dir, method) -> str:
    """Heatmap: stations x features correlation (matplotlib imshow)."""
    if corr_df.empty:
        return ""
    pivot = corr_df[corr_df["method"] == method].pivot_table(
        index="station_id", columns="feature", values="corr", aggfunc="mean"
    )
    pivot = pivot.loc[pivot.abs().mean(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(max(8, 0.6 * pivot.shape[1] + 3), max(6, 0.35 * pivot.shape[0] + 3)))
    im = ax.imshow(pivot.values, aspect="auto")
    ax.set_title(f"Correlation heatmap (stations x features) - {method}")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Station")
    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns.tolist(), rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels(pivot.index.tolist(), fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    out_path = out_dir / "figures" / f"A5_corr_heatmap_{method}.png"
    save_fig(fig, out_path)
    return str(out_path)


# ----------------------------
# Part 5: Main
# ----------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Station-level plots for BPH dataset")
    p.add_argument("--input_csv", required=True, help="Path to daily CSV")
    p.add_argument("--out_dir", required=True, help="Output directory for plots and tables")
    p.add_argument("--target", default="bph_log1p", help="Target column (bph_log1p or bph_raw)")
    p.add_argument(
        "--features",
        default="temp,humidity,rainfall,wind_speed,wind_u,wind_v,delta_temp,temp_7d_mean,humidity_7d_mean,rain_7d_sum",
        help="Comma-separated feature columns"
    )
    p.add_argument("--top_k_stations", type=int, default=6, help="Stations to plot in time-series/scatter")
    p.add_argument("--pick_stations_how", default="most_rows", choices=["most_rows"], help="Station selection rule")
    p.add_argument("--plot_sample_n", type=int, default=5000, help="Max points per station in scatter")
    p.add_argument("--per_station_timeseries", action="store_true",
                   help="If set, save A1 time-series as 1 figure per station (in figures/A1_per_station/).")
    p.add_argument("--per_station_max_points", type=int, default=0,
                   help="Optional downsample: max points per station time-series (0 = no downsample).")
    p.add_argument("--boxplot_top_k", type=int, default=10, help="Stations to include in target boxplot")
    # NEW: choose specific stations (override auto top-K)
    p.add_argument(
        "--stations",
        default="",
        help='Comma-separated station IDs to include (overrides auto selection). Example: --stations "101,205,330"',
    )
    p.add_argument(
        "--stations_file",
        default="",
        help="Path to a txt/csv containing station IDs to include (overrides auto selection).",
    )
    return p.parse_args(argv)

def main(argv=None):
    args = parse_args(argv)
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = read_csv_any(args.input_csv)
    cols = detect_columns(df, target=args.target)

    if not cols["date_col"] or not cols["station_col"]:
        raise RuntimeError(f"Required columns not found. Detected: {cols}")

    df = ensure_datetime(df, cols["date_col"])
    df = auto_make_bph_targets(df, target_col=args.target)

    # re-detect after creating targets
    cols = detect_columns(df, target=args.target)
    if not cols["target_col"]:
        raise RuntimeError(f"Target column '{args.target}' not found and could not be created.")
    target_col = cols["target_col"]

    features = [c.strip() for c in args.features.split(",") if c.strip()]
    df = safe_numeric(df, features + [target_col])

        # Station selection:
    # If --stations or --stations_file is provided, we *filter* the dataset to those stations
    # and use that list for all plots/tables.
    stations_arg = parse_station_list_arg(args.stations)
    stations_file = load_stations_from_file(args.stations_file) if args.stations_file else []
    stations_req = unique_preserve_order(stations_arg + stations_file)

    if stations_req:
        all_stations = set(df[cols["station_col"]].astype(str).unique().tolist())
        stations = [s for s in stations_req if str(s) in all_stations]
        missing = [s for s in stations_req if str(s) not in all_stations]
        if missing:
            print(f"[WARN] Requested stations not found and will be ignored: {missing[:20]}{'...' if len(missing)>20 else ''}")

        if not stations:
            raise RuntimeError("No requested stations were found in the dataset. Please check --stations/--stations_file.")
        df = df[df[cols["station_col"]].astype(str).isin(stations)].copy()
        print(f"[OK] Station filter applied: {len(stations)} stations, rows={len(df)}")
    else:
        stations = pick_stations(df, cols["station_col"], how=args.pick_stations_how, k=args.top_k_stations)

    created = {}

    created["A1_timeseries"] = plot_timeseries_target(
        df, cols["date_col"], cols["station_col"], target_col, stations, out_dir
    )

    # Optional: A1 per-station figures
    if args.per_station_timeseries:
        created["A1_timeseries_per_station"] = plot_timeseries_target_per_station(
            df=df,
            date_col=cols["date_col"],
            station_col=cols["station_col"],
            target_col=target_col,
            stations=stations,
            out_dir=out_dir,
            max_points=args.per_station_max_points
        )

    key_cols = [target_col]
    for c in ["temp", "humidity", "rainfall", "wind_speed", "wind_direction", "wind_u", "wind_v",
              "delta_temp", "temp_7d_mean", "humidity_7d_mean", "rain_7d_sum"]:
        if c in df.columns and c not in key_cols:
            key_cols.append(c)

    created["A2_missingness"] = plot_missingness_per_station(df, cols["station_col"], key_cols, out_dir)
    created["A3_boxplot"] = plot_target_boxplot_by_station(
        df=df,
        station_col=cols["station_col"],
        target_col=target_col,
        top_k=args.boxplot_top_k,
        out_dir=out_dir,
        stations_override=stations
    )

    scatter_features = [f for f in features if f in df.columns and f != target_col][:3]
    created["A4_scatter"] = []
    for f in scatter_features:
        pth = plot_feature_vs_target_scatter(df, cols["station_col"], target_col, f, stations, args.plot_sample_n, out_dir)
        if pth:
            created["A4_scatter"].append(pth)

    corr_features = [f for f in features if f in df.columns and f != target_col]
    pear = corr_per_station(df, cols["station_col"], target_col, corr_features, method="pearson")
    spear = corr_per_station(df, cols["station_col"], target_col, corr_features, method="spearman")
    corr_all = pd.concat([pear, spear], ignore_index=True) if (not pear.empty or not spear.empty) else pd.DataFrame()

    corr_csv = out_dir / "corr_per_station_long.csv"
    corr_all.to_csv(corr_csv, index=False, encoding="utf-8-sig")
    created["corr_table_csv"] = str(corr_csv)

    created["A5_heatmap_pearson"] = plot_corr_heatmap_station_feature(corr_all, out_dir, method="pearson")
    created["A5_heatmap_spearman"] = plot_corr_heatmap_station_feature(corr_all, out_dir, method="spearman")

    report = {
        "input_csv": str(Path(args.input_csv).expanduser()),
        "detected_columns": cols,
        "target_used": target_col,
        "selected_stations": stations,
        "stations_arg": args.stations,
        "stations_file": args.stations_file,
        "station_filter_applied": bool(stations_req),
        "figures": created,
        "n_rows": int(len(df)),
        "n_stations": int(df[cols["station_col"]].nunique(dropna=True)),
    }
    report_path = out_dir / "station_eda_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] Saved figures: {out_dir / 'figures'}")
    print(f"[OK] Saved report:  {report_path}")
    print(f"[OK] Saved corr:    {corr_csv}")

if __name__ == "__main__":
    main()
