#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import re
from pathlib import Path

import pandas as pd

ENCODINGS = ("cp874", "utf-8-sig", "utf-8")


def read_csv_any(path: Path) -> pd.DataFrame:
    last_err = None
    for enc in ENCODINGS:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Cannot read {path.name} with {ENCODINGS}: {last_err}")


def parse_year_season(filename: str):
    m = re.search(r"(19\d{2}|20\d{2})", filename)
    if not m:
        raise ValueError(f"Year not found in filename: {filename}")
    year = int(m.group(1))
    low = filename.lower()
    if "in-season" in low:
        season = "in_season"
    elif "off-season" in low:
        season = "off_season"
    else:
        raise ValueError(f"Season not found in filename: {filename}")
    return year, season


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Build a mapping based on lowercased, space-normalized names
    def key(c: str) -> str:
        return re.sub(r"\s+", " ", str(c).strip().lower())

    cols = {key(c): c for c in df.columns}

    rename = {}
    # province Thai / Eng
    for cand in ("province thai", "province_thai", "province th", "prov th"):
        if cand in cols:
            rename[cols[cand]] = "province_th"
            break
    for cand in ("province eng", "province english", "province_en", "province en", "prov en"):
        if cand in cols:
            rename[cols[cand]] = "province_en"
            break
    # coordinates
    for cand in ("lat", "latitude"):
        if cand in cols:
            rename[cols[cand]] = "lat"
            break
    for cand in ("long", "lon", "longitude"):
        if cand in cols:
            rename[cols[cand]] = "lon"
            break

    return df.rename(columns=rename)


def norm_variety_name(s: str) -> str:
    if s is None:
        return ""
    x = str(s)
    x = x.replace("\u2013", "-").replace("\u2014", "-")  # en/em dash
    x = re.sub(r"\s+", " ", x.strip())
    return x


def to_long(df: pd.DataFrame, year: int, season: str, source_file: str) -> pd.DataFrame:
    df = normalize_columns(df)

    id_cols = [c for c in ["province_th", "province_en", "lat", "lon"] if c in df.columns]
    if "province_en" not in id_cols and "province_th" not in id_cols:
        raise ValueError(f"Province columns not found in {source_file}")

    variety_cols = [c for c in df.columns if c not in id_cols]

    long = df.melt(
        id_vars=id_cols,
        value_vars=variety_cols,
        var_name="variety_raw",
        value_name="area_rai",
    )

    long["year"] = year
    long["crop_season"] = season
    long["variety_norm"] = long["variety_raw"].map(norm_variety_name)
    long["area_rai"] = pd.to_numeric(long["area_rai"], errors="coerce")
    long["source_file"] = Path(source_file).name

    return long


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default="/mnt/data")
    ap.add_argument("--years", default="2015,2016,2017,2018,2019")
    ap.add_argument("--out_csv", default="/mnt/data/variety_rice_area_2015_2019_raw.csv")
    ap.add_argument("--out_report", default="/mnt/data/variety_rice_area_2015_2019_report.json")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    years = [int(x) for x in args.years.split(",") if x.strip()]

    frames = []
    file_summaries = []

    for y in years:
        # Search for both seasons
        patterns = [
            f"Varieties-production volume of rice in-season Rai(unit) {y}.csv",
            f"Varieties-production volume of rice off-season Rai(unit) {y}.csv",
        ]
        for pat in patterns:
            path = input_dir / pat
            if not path.exists():
                continue
            year, season = parse_year_season(path.name)
            df = read_csv_any(path)
            long = to_long(df, year, season, path.name)
            frames.append(long)
            file_summaries.append(
                {
                    "file": path.name,
                    "year": year,
                    "season": season,
                    "rows_wide": int(df.shape[0]),
                    "cols_wide": int(df.shape[1]),
                    "rows_long": int(long.shape[0]),
                    "n_varieties": int(long["variety_raw"].nunique()),
                }
            )

    if not frames:
        raise SystemExit("No variety files found. Check input_dir and filenames.")

    out = pd.concat(frames, ignore_index=True)

    # Drop empty variety names (should not happen)
    out = out[out["variety_norm"].astype(str).str.len() > 0].copy()

    # Merge duplicates if any
    key_cols = [c for c in ["province_th", "province_en", "lat", "lon"] if c in out.columns]
    key = key_cols + ["year", "crop_season", "variety_norm"]
    if out.duplicated(subset=key).any():
        out = (
            out.groupby(key, as_index=False)["area_rai"].sum()
            .merge(out[key + ["variety_raw"]].drop_duplicates(subset=key), on=key, how="left")
        )

    # Basic report
    report = {
        "files_processed": file_summaries,
        "total_rows": int(out.shape[0]),
        "years": sorted(out["year"].unique().tolist()),
        "seasons": sorted(out["crop_season"].unique().tolist()),
        "n_provinces": int(out["province_en"].nunique()) if "province_en" in out.columns else None,
        "n_varieties": int(out["variety_norm"].nunique()),
        "missing_area_rai": int(out["area_rai"].isna().sum()),
        "area_rai_min": None if out["area_rai"].dropna().empty else float(out["area_rai"].min()),
        "area_rai_max": None if out["area_rai"].dropna().empty else float(out["area_rai"].max()),
    }

    out_csv = Path(args.out_csv)
    out_report = Path(args.out_report)
    out.to_csv(out_csv, index=False, encoding="utf-8-sig")
    out_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved: {out_csv}")
    print(f"Saved: {out_report}")
    print(f"Rows: {out.shape[0]:,} | Varieties: {out['variety_norm'].nunique():,}")


if __name__ == "__main__":
    main()
