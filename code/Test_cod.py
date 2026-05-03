from pathlib import Path
import pandas as pd

# scripts/run_pipeline.py
from pathlib import Path
from make_model_ready_v3_stream_with_plots import (
    add_station_derivatives,
    add_targets,
    add_time_features,
    add_wind_uv,
    read_csv_any,
)
# โฟลเดอร์ที่ไฟล์ .py ตัวนี้อยู่
HERE = Path(__file__).resolve().parent

# โฟลเดอร์โปรเจกต์ (ถ้า scripts อยู่ใต้โปรเจกต์)
PROJECT_ROOT = HERE.parent

RAW_CSV = PROJECT_ROOT / "data"/"env_daily_with_rice_monthly_raw.csv"
OUT_DIR = PROJECT_ROOT / "out_model_ready_v1"

print("RAW_CSV =", RAW_CSV)
print("OUT_DIR =", OUT_DIR)


#df = read_csv_any(RAW_CSV)
#print(df.head())
#print(df.shape)

df = read_csv_any(RAW_CSV)
if "date" not in df.columns:
    raise ValueError("RAW ต้องมีคอลัมน์ date")
if "station_id" not in df.columns:
    raise ValueError("RAW ต้องมีคอลัมน์ station_id")

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"]).copy()

df = add_targets(df)
df = add_time_features(df)
df = add_wind_uv(df)
df = add_station_derivatives(df, roll_days=7)


print(df.shape)
print(df.head())
print(df.info())
print(df.describe(include="all"))

