from pathlib import Path
import pandas as pd

import subprocess
import sys

from make_model_ready_v3_stream_with_plots import read_csv_any,add_targets,add_time_features,add_wind_uv,add_station_derivatives
6
# โฟลเดอร์ที่ไฟล์ .py ตัวนี้อยู่
HERE = Path(__file__).resolve().parent

# โฟลเดอร์โปรเจกต์ (ถ้า scripts อยู่ใต้โปรเจกต์)
PROJECT_ROOT = HERE.parent

raw_csv = PROJECT_ROOT / "data" / "env_daily_with_rice_monthly_raw.csv"
out_dir = PROJECT_ROOT / "data" / "out_model_ready_v1"

print("RAW_CSV =", raw_csv)
print("OUT_DIR =", out_dir)

raw_csv = Path(raw_csv)
out_dir = Path(out_dir)
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
df = add_station_derivatives(df, roll_days=7)

print(df.head())
print(df.shape)





