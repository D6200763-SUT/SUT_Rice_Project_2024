from pathlib import Path
import pandas as pd

# โฟลเดอร์ที่ไฟล์ .py ตัวนี้อยู่
HERE = Path(__file__).resolve().parent

# โฟลเดอร์โปรเจกต์ (ถ้า scripts อยู่ใต้โปรเจกต์)
PROJECT_ROOT = HERE.parent

RAW_CSV = PROJECT_ROOT / "Import_Dataset" / "env_daily_with_rice_monthly_raw.csv"
OUT_DIR = PROJECT_ROOT / "out_model_ready_v3"

print("RAW_CSV =", RAW_CSV)
print("OUT_DIR =", OUT_DIR)

df = pd.read_csv(RAW_CSV)
df = pd.read_csv(RAW_CSV, parse_dates=["date"])
df = df.sort_values("date")
print(df.head())
print(df.shape)