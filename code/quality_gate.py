#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
quality_gate.py
---------------
Data Contract / Quality Gate for BPH daily dataset (station-level time series).

What it does
1) Read a CSV (Thai-friendly encodings)
2) Validate data contract rules (schema / key uniqueness / ranges / missingness / continuity / variety-share sums)
3) Optionally apply "soft fixes" (drop duplicates, coerce types, drop rows with missing mandatory fields)
4) Write:
   - cleaned_raw.csv
   - quality_report.json

Typical usage
python quality_gate.py \
  --input_csv data/env_daily_with_rice_monthly_raw.csv \
  --out_dir out_quality_gate \
  --mode strict

Soft mode (tries to fix duplicates, etc.)
python quality_gate.py \
  --input_csv data/env_daily_with_rice_monthly_raw.csv \
  --out_dir out_quality_gate \
  --mode soft \
  --drop_rows_missing_mandatory
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Iterable, Dict, Any, List, Tuple

import numpy as np
import pandas as pd


# ----------------------------
# Config
# ----------------------------
ENCODINGS = ("utf-8-sig", "utf-8", "cp874", "tis-620")

DEFAULT_MANDATORY_COLS = [
    "date",
    "station_id",
    "latitude",
    "longitude",
    "temp",
    "humidity",
    "rainfall",
    "wind_speed",
    "wind_direction",
    "bph_count",
    "province_en",
]

# Range rules (None means no bound)
RANGE_RULES = {
    "humidity": (0.0, 100.0),
    "rainfall": (0.0, None),
    "wind_speed": (0.0, None),
    "wind_direction": (0.0, 360.0),
    "temp": (-5.0, 50.0),       # generous for Thailand
    "mint_temp": (-10.0, 45.0),
    "maxt_temp": (-5.0, 55.0),
    "dew_point": (-20.0, 40.0),
    "bph_count": (0.0, None),
    "bph_raw": (0.0, None),
}

SUSPECT_ZERO_COLS_DEFAULT = ["temp", "humidity", "mint_temp", "maxt_temp"]  # configurable


# ----------------------------
# Helpers
# ----------------------------
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


def coerce_datetime(df: pd.DataFrame, date_col: str) -> Tuple[pd.DataFrame, int]:
    out = df.copy()
    before = len(out)
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out = out.dropna(subset=[date_col])
    dropped = before - len(out)
    return out, dropped


def coerce_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def find_var_share_cols(df: pd.DataFrame) -> Dict[str, List[str]]:
    cols = df.columns.tolist()
    in_cols = [c for c in cols if c.startswith("var_share_") and c.endswith("_in_season")]
    off_cols = [c for c in cols if c.startswith("var_share_") and c.endswith("_off_season")]
    return {"in_season": in_cols, "off_season": off_cols}


def compute_duplicates(df: pd.DataFrame, station_col: str, date_col: str) -> pd.Series:
    return df.duplicated(subset=[station_col, date_col], keep=False)


def continuity_by_station(df: pd.DataFrame, station_col: str, date_col: str) -> pd.DataFrame:
    g = df.groupby(station_col)[date_col]
    res = g.agg(min_date="min", max_date="max", observed_days="nunique").reset_index()
    res["expected_days"] = (res["max_date"] - res["min_date"]).dt.days + 1
    res["missing_days"] = res["expected_days"] - res["observed_days"]
    res["continuity"] = res["observed_days"] / res["expected_days"].replace(0, np.nan)
    return res


def station_coordinate_stability(df: pd.DataFrame, station_col: str, lat_col: str, lon_col: str) -> pd.DataFrame:
    tmp = df[[station_col, lat_col, lon_col]].copy()
    tmp[lat_col] = pd.to_numeric(tmp[lat_col], errors="coerce")
    tmp[lon_col] = pd.to_numeric(tmp[lon_col], errors="coerce")
    g = tmp.groupby(station_col)
    out = g.agg(
        unique_lat=(lat_col, lambda s: s.dropna().nunique()),
        unique_lon=(lon_col, lambda s: s.dropna().nunique()),
        min_lat=(lat_col, "min"),
        max_lat=(lat_col, "max"),
        min_lon=(lon_col, "min"),
        max_lon=(lon_col, "max"),
    ).reset_index()
    out["max_abs_lat_diff"] = (out["max_lat"] - out["min_lat"]).abs()
    out["max_abs_lon_diff"] = (out["max_lon"] - out["min_lon"]).abs()
    return out


def range_violations(df: pd.DataFrame, rules: Dict[str, Tuple[Optional[float], Optional[float]]]) -> Dict[str, int]:
    violations = {}
    for col, (lo, hi) in rules.items():
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        mask = pd.Series(False, index=df.index)
        if lo is not None:
            mask |= (s < lo)
        if hi is not None:
            mask |= (s > hi)
        violations[col] = int(mask.sum())
    return violations


def suspect_zero_counts(df: pd.DataFrame, cols: List[str]) -> Dict[str, int]:
    out = {}
    for c in cols:
        if c not in df.columns:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        out[c] = int((s == 0).sum())
    return out


def var_share_sum_checks(df: pd.DataFrame, cols_in: List[str], cols_off: List[str], tol: float) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if cols_in:
        s = df[cols_in].apply(pd.to_numeric, errors="coerce").sum(axis=1)
        out["in_season_sum_min"] = float(np.nanmin(s.values))
        out["in_season_sum_max"] = float(np.nanmax(s.values))
        out["in_season_sum_mean"] = float(np.nanmean(s.values))
        out["in_season_bad_count"] = int(((s < (1 - tol)) | (s > (1 + tol))).sum())
    else:
        out["in_season_bad_count"] = None

    if cols_off:
        s = df[cols_off].apply(pd.to_numeric, errors="coerce").sum(axis=1)
        out["off_season_sum_min"] = float(np.nanmin(s.values))
        out["off_season_sum_max"] = float(np.nanmax(s.values))
        out["off_season_sum_mean"] = float(np.nanmean(s.values))
        out["off_season_bad_count"] = int(((s < -tol) | (s > (1 + tol))).sum())
    else:
        out["off_season_bad_count"] = None

    return out


# ----------------------------
# Gate logic
# ----------------------------
@dataclass
class GateConfig:
    mode: str  # strict | soft
    max_missing_pct: float
    min_continuity: float
    var_share_tol: float
    coord_max_drift: float  # degrees (very small number)
    suspect_zero_cols: List[str]
    drop_rows_missing_mandatory: bool


def validate_and_optionally_fix(
    df: pd.DataFrame,
    date_col: str,
    station_col: str,
    mandatory_cols: List[str],
    cfg: GateConfig,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    report: Dict[str, Any] = {}
    report["mode"] = cfg.mode
    report["n_rows_input"] = int(len(df))
    report["n_cols_input"] = int(df.shape[1])

    # 1) Schema checks (presence)
    missing_mandatory = [c for c in mandatory_cols if c not in df.columns]
    report["missing_mandatory_columns"] = missing_mandatory

    # 2) Coerce datetime
    df2, dropped_bad_date = coerce_datetime(df, date_col)
    report["dropped_rows_bad_date"] = int(dropped_bad_date)

    # 3) Coerce numeric for known columns
    numeric_candidates = list(set([c for c in RANGE_RULES.keys() if c in df2.columns] + cfg.suspect_zero_cols))
    df2 = coerce_numeric(df2, numeric_candidates)

    # 4) Key duplicates
    dup_mask = compute_duplicates(df2, station_col, date_col)
    n_dup_rows = int(dup_mask.sum())
    report["duplicate_key_rows"] = n_dup_rows

    if n_dup_rows > 0 and cfg.mode == "soft":
        before = len(df2)
        df2 = df2.drop_duplicates(subset=[station_col, date_col], keep="first")
        report["soft_fix_dropped_duplicate_rows"] = int(before - len(df2))
    else:
        report["soft_fix_dropped_duplicate_rows"] = 0

    # 5) Missingness (%)
    mand_present = [c for c in mandatory_cols if c in df2.columns]
    miss_pct = (df2[mand_present].isna().mean() * 100.0).to_dict()
    report["missing_pct_mandatory"] = {k: float(v) for k, v in miss_pct.items()}

    if cfg.drop_rows_missing_mandatory and cfg.mode == "soft" and mand_present:
        before = len(df2)
        df2 = df2.dropna(subset=mand_present)
        report["soft_fix_dropped_rows_missing_mandatory"] = int(before - len(df2))
    else:
        report["soft_fix_dropped_rows_missing_mandatory"] = 0

    # 6) Range violations
    rv = range_violations(df2, RANGE_RULES)
    report["range_violations_count"] = rv

    # 7) Suspect zeros
    report["suspect_zero_counts"] = suspect_zero_counts(df2, cfg.suspect_zero_cols)

    # 8) Continuity per station
    cont = continuity_by_station(df2, station_col, date_col)
    report["n_stations"] = int(cont[station_col].nunique())
    report["continuity_summary"] = {
        "min": float(cont["continuity"].min()),
        "p50": float(cont["continuity"].median()),
        "mean": float(cont["continuity"].mean()),
        "p95": float(cont["continuity"].quantile(0.95)),
        "max": float(cont["continuity"].max()),
    }
    bad_stations = cont.loc[cont["continuity"] < cfg.min_continuity, station_col].astype(str).tolist()
    report["stations_below_min_continuity"] = bad_stations

    # 9) Coordinate stability (if lat/lon exist)
    lat_col = pick_first_existing(df2, ["latitude", "lat", "Latitude", "LAT"])
    lon_col = pick_first_existing(df2, ["longitude", "lon", "Longitude", "LON"])
    report["lat_col"] = lat_col
    report["lon_col"] = lon_col

    if lat_col and lon_col:
        stab = station_coordinate_stability(df2, station_col, lat_col, lon_col)
        drift_mask = (stab["max_abs_lat_diff"] > cfg.coord_max_drift) | (stab["max_abs_lon_diff"] > cfg.coord_max_drift)
        report["stations_coordinate_drift"] = stab.loc[drift_mask, station_col].astype(str).tolist()
        report["coordinate_drift_threshold_deg"] = float(cfg.coord_max_drift)
    else:
        report["stations_coordinate_drift"] = []

    # 10) Variety share checks
    vs = find_var_share_cols(df2)
    report["var_share_columns_count"] = {k: int(len(v)) for k, v in vs.items()}
    report["var_share_sum_checks"] = var_share_sum_checks(df2, vs["in_season"], vs["off_season"], tol=cfg.var_share_tol)
    report["var_share_tol"] = float(cfg.var_share_tol)

    # Decide PASS / FAIL
    fails: List[str] = []

    if missing_mandatory:
        fails.append(f"Missing mandatory columns: {missing_mandatory}")

    if n_dup_rows > 0 and cfg.mode == "strict":
        fails.append(f"Duplicate (station_id,date) rows: {n_dup_rows}")

    for c, pct in report["missing_pct_mandatory"].items():
        if pct > cfg.max_missing_pct:
            fails.append(f"Missingness too high: {c}={pct:.3f}% > {cfg.max_missing_pct}%")

    if bad_stations:
        fails.append(f"Stations below min continuity ({cfg.min_continuity}): {bad_stations[:10]}{'...' if len(bad_stations)>10 else ''}")

    if cfg.mode == "strict":
        bad_ranges = {k: v for k, v in rv.items() if v and v > 0}
        if bad_ranges:
            fails.append(f"Range violations exist: {bad_ranges}")

        in_bad = report["var_share_sum_checks"].get("in_season_bad_count")
        if in_bad is not None and int(in_bad) > 0:
            fails.append(f"In-season var_share sum not ~1: bad_rows={in_bad}")

        off_bad = report["var_share_sum_checks"].get("off_season_bad_count")
        if off_bad is not None and int(off_bad) > 0:
            fails.append(f"Off-season var_share sum outside [0,1]: bad_rows={off_bad}")

    report["fail_reasons"] = fails
    report["pass"] = (len(fails) == 0)
    report["n_rows_output"] = int(len(df2))
    report["n_cols_output"] = int(df2.shape[1])

    return df2, report


# ----------------------------
# CLI
# ----------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Data Contract / Quality Gate for BPH daily dataset")
    p.add_argument("--input_csv", required=True, help="Input CSV path")
    p.add_argument("--out_dir", required=True, help="Output directory")
    p.add_argument("--mode", default="strict", choices=["strict", "soft"], help="strict = fail on violations; soft = apply limited fixes")
    p.add_argument("--max_missing_pct", type=float, default=1.0, help="Max missing percent allowed for mandatory columns")
    p.add_argument("--min_continuity", type=float, default=0.95, help="Min continuity per station (observed_days/expected_days)")
    p.add_argument("--var_share_tol", type=float, default=1e-3, help="Tolerance for var_share sums")
    p.add_argument("--coord_max_drift", type=float, default=1e-6, help="Max allowed lat/lon drift per station (degrees)")
    p.add_argument("--suspect_zero_cols", default=",".join(SUSPECT_ZERO_COLS_DEFAULT), help="Comma-separated columns to flag when value==0")
    p.add_argument("--drop_rows_missing_mandatory", action="store_true",
                   help="(soft mode) Drop rows that have NA in any mandatory column present in file.")
    p.add_argument("--mandatory_cols", default=",".join(DEFAULT_MANDATORY_COLS),
                   help="Comma-separated list of mandatory columns (default fits merged raw).")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    input_csv = Path(args.input_csv).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = read_csv_any(input_csv)
    date_col = pick_first_existing(df, ["date", "Date", "datetime", "timestamp"])
    station_col = pick_first_existing(df, ["station_id", "StationID", "station", "stationId"])
    if not date_col or not station_col:
        raise RuntimeError(f"Cannot detect date/station columns. Found date={date_col}, station={station_col}")

    mandatory_cols = [c.strip() for c in args.mandatory_cols.split(",") if c.strip()]
    suspect_zero_cols = [c.strip() for c in args.suspect_zero_cols.split(",") if c.strip()]

    cfg = GateConfig(
        mode=args.mode,
        max_missing_pct=float(args.max_missing_pct),
        min_continuity=float(args.min_continuity),
        var_share_tol=float(args.var_share_tol),
        coord_max_drift=float(args.coord_max_drift),
        suspect_zero_cols=suspect_zero_cols,
        drop_rows_missing_mandatory=bool(args.drop_rows_missing_mandatory),
    )

    cleaned, report = validate_and_optionally_fix(
        df=df,
        date_col=date_col,
        station_col=station_col,
        mandatory_cols=mandatory_cols,
        cfg=cfg,
    )

    cleaned_path = out_dir / "cleaned_raw.csv"
    report_path = out_dir / "quality_report.json"
    cleaned.to_csv(cleaned_path, index=False, encoding="utf-8-sig")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] cleaned_raw.csv: {cleaned_path}")
    print(f"[OK] quality_report.json: {report_path}")
    print(f"[RESULT] PASS={report['pass']}")
    if not report["pass"]:
        print("[FAIL REASONS]")
        for r in report["fail_reasons"]:
            print(" -", r)

    if cfg.mode == "strict" and not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
