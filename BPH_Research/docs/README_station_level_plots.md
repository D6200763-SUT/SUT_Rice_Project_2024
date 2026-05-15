# Station-level EDA Plots for BPH (Brown Planthopper)
**Script:** `code/10_station_level_plots_v2.py`  
**Purpose:** สร้างกราฟ/ตารางสำหรับ “นำเสนอข้อมูลเชิงรายสถานี (station-level)” เพื่ออธิบายพฤติกรรมของ BPH และคุณภาพข้อมูลก่อนทำโมเดล time-series (CNN-LSTM/Transformer ฯลฯ)

---

## 1) สิ่งที่สคริปต์ทำ (A1–A5)
- **A1**: Target time-series หลายสถานีในรูปเดียว (ภาพรวมเปรียบเทียบ)
- **A1b**: (ตัวเลือก) แยก A1 **1 รูปต่อ 1 สถานี** เพื่อแนบรายงาน/ภาคผนวก
- **A2**: Missingness (%) ต่อสถานี (ตรวจคุณภาพ/ความครบถ้วนของข้อมูล)
- **A3**: Boxplot ของ Target ต่อสถานี (เปรียบเทียบการกระจาย/Outlier)
- **A4**: Scatter ระหว่าง **Feature vs Target** (แยกสีตามสถานี)
- **A5**: Correlation ต่อสถานี (**Pearson + Spearman**) + Heatmap (stations × features)

นอกจากนี้ยังสร้างไฟล์รายงานและตาราง correlation เพื่อนำไปสรุปผลเชิงงานวิจัยได้ต่อทันที

---

## 2) Requirements (ติดตั้งไลบรารี)
แนะนำให้ใช้ environment เดิมของโปรเจกต์ (เช่น `tf_cb`) แล้วติดตั้งแพ็กเกจพื้นฐาน:

```bash
pip install pandas numpy matplotlib scikit-learn joblib
```

> ใช้ `matplotlib` เท่านั้น (ไม่ใช้ seaborn) เพื่อความเบาและรันได้ง่ายทุกเครื่อง

---

## 3) Input data (ไฟล์ข้อมูลที่ต้องมี)
### 3.1 ไฟล์รายวัน (จำเป็น)
`--input_csv` เช่น
- `data/env_daily_with_rice_monthly_raw.csv`

ต้องมีอย่างน้อย:
- คอลัมน์วันเวลา: `date` (หรือ `Date/datetime/timestamp`)
- คอลัมน์สถานี: `station_id` (หรือ `StationID/station/stationId`)
- คอลัมน์เป้าหมาย (เลือกอย่างใดอย่างหนึ่ง):
  - `bph_log1p` (แนะนำ)
  - `bph_raw`
  - หรือถ้ามีแค่ `bph_count` สคริปต์จะสร้าง `bph_raw` และ `bph_log1p` ให้อัตโนมัติ (ถ้าพบคอลัมน์นั้น)

### 3.2 ไฟล์รายการสถานี (ตัวเลือก)
ใช้กับ `--stations_file` ได้ทั้ง `.txt` หรือ `.csv`

**ตัวอย่าง `data/stations.csv`**
```csv
station_id
Chachoengsao Rice research Center
Ratchaburi Rice Seed Center 1
Phatthalung Rice research Center
```

**ตัวอย่าง `data/stations.txt`**
```
# one station per line
Chachoengsao Rice research Center
Ratchaburi Rice Seed Center 1
Phatthalung Rice research Center
```

---

## 4) Output structure (ผลลัพธ์ที่ได้)
เมื่อรันแล้วจะได้โครงสร้างประมาณนี้:

```
<out_dir>/
  figures/
    A1_timeseries_target_by_station.png
    A2_missingness_per_station.png
    A3_boxplot_target_by_station_topk.png
    A4_scatter_<feature>_vs_<target>.png
    A5_corr_heatmap_pearson.png
    A5_corr_heatmap_spearman.png
    A1_per_station/                       # (ถ้าเปิด --per_station_timeseries)
      A1_timeseries_<station>.png
  corr_per_station_long.csv               # ตาราง corr รายสถานี (long format)
  station_eda_report.json                 # รายงานสรุปพาธไฟล์/คอลัมน์/สถานีที่ใช้
```

### อธิบายไฟล์สำคัญ
- **`corr_per_station_long.csv`**  
  คอลัมน์หลัก: `station_id, feature, corr, n, method`  
  ใช้ทำสรุป “ฟีเจอร์ไหนสัมพันธ์กับ target มากสุด” รายสถานี หรือสรุปข้ามสถานี
- **`station_eda_report.json`**  
  เก็บ:
  - คอลัมน์ที่ตรวจพบ (date/station/target)
  - รายชื่อสถานีที่ใช้จริง (หลังกรอง)
  - รายการไฟล์รูปที่สร้าง
  - จำนวนแถว/จำนวนสถานี  
  เหมาะสำหรับ reproducibility ในบทที่ 3/4

---

## 5) Quick start (รันพื้นฐาน)
จากโฟลเดอร์โปรเจกต์:

```bash
python code/10_station_level_plots_v2.py \
  --input_csv data/env_daily_with_rice_monthly_raw.csv \
  --out_dir out_station_eda \
  --target bph_log1p
```

---

## 6) คำสั่งรัน “ทุกแบบ” (Examples)

### 6.1 เลือก target เป็น `bph_raw`
```bash
python code/10_station_level_plots_v2.py \
  --input_csv data/env_daily_with_rice_monthly_raw.csv \
  --out_dir out_station_eda_raw \
  --target bph_raw
```

### 6.2 กำหนด features เอง (ใช้กับ A4/A5)
```bash
python code/10_station_level_plots_v2.py \
  --input_csv data/env_daily_with_rice_monthly_raw.csv \
  --out_dir out_station_eda_custom \
  --target bph_log1p \
  --features temp,humidity,rainfall,wind_u,wind_v,delta_temp,temp_7d_mean,humidity_7d_mean,rain_7d_sum
```

### 6.3 เปลี่ยนจำนวนสถานีที่เลือกมาโชว์ (Top-K)
```bash
python code/10_station_level_plots_v2.py \
  --input_csv data/env_daily_with_rice_monthly_raw.csv \
  --out_dir out_station_eda_top10 \
  --target bph_log1p \
  --top_k_stations 10
```

### 6.4 แยกกราฟ A1 เป็น 1 รูปต่อ 1 สถานี (A1b)
```bash
python code/10_station_level_plots_v2.py \
  --input_csv data/env_daily_with_rice_monthly_raw.csv \
  --out_dir out_station_eda_per_station \
  --target bph_log1p \
  --top_k_stations 10 \
  --per_station_timeseries
```

### 6.5 แยก A1 รายสถานี + ลดจำนวนจุด (ไฟล์เล็ก/เปิดเร็ว)
```bash
python code/10_station_level_plots_v2.py \
  --input_csv data/env_daily_with_rice_monthly_raw.csv \
  --out_dir out_station_eda_per_station_ds \
  --target bph_log1p \
  --top_k_stations 10 \
  --per_station_timeseries \
  --per_station_max_points 2000
```

### 6.6 ปรับจำนวนจุดใน scatter (A4) ต่อสถานี
```bash
python code/10_station_level_plots_v2.py \
  --input_csv data/env_daily_with_rice_monthly_raw.csv \
  --out_dir out_station_eda_scatter1000 \
  --target bph_log1p \
  --plot_sample_n 1000
```

### 6.7 ปรับจำนวนสถานีใน boxplot (A3)
```bash
python code/10_station_level_plots_v2.py \
  --input_csv data/env_daily_with_rice_monthly_raw.csv \
  --out_dir out_station_eda_box20 \
  --target bph_log1p \
  --boxplot_top_k 20
```

---

## 7) เลือก “เฉพาะสถานี” (NEW)

เมื่อใช้ `--stations` หรือ `--stations_file` สคริปต์จะ:
1) กรองข้อมูลให้เหลือเฉพาะสถานีที่ระบุ  
2) ทำกราฟ/ตารางทั้งหมดจาก subset นั้น

### 7.1 ระบุในคำสั่งด้วย `--stations`
```bash
python code/10_station_level_plots_v2.py \
  --input_csv data/env_daily_with_rice_monthly_raw.csv \
  --out_dir out_station_eda_selected \
  --target bph_log1p \
  --stations "Chachoengsao Rice research Center,Ratchaburi Rice Seed Center 1"
```

### 7.2 ระบุผ่านไฟล์ `--stations_file` (.csv/.txt)
```bash
python code/10_station_level_plots_v2.py \
  --input_csv data/env_daily_with_rice_monthly_raw.csv \
  --out_dir out_station_eda_selected_file \
  --target bph_log1p \
  --stations_file data/stations.csv
```

> ถ้ามี station ที่ไม่พบในข้อมูล สคริปต์จะแจ้ง `[WARN]` แล้วข้าม station นั้น

---

## 8) จัดกลุ่มสถานีตาม “จังหวัด/ภูมิภาค” (สำหรับทำกราฟเป็นกลุ่ม)
ใช้สคริปต์: `code/20_build_station_groups.py` (สร้างไฟล์รายการสถานีต่อกลุ่ม ให้ใช้กับ `--stations_file` ได้ทันที)

### 8.1 สร้างไฟล์กลุ่ม (ครั้งเดียว)
```bash
python code/20_build_station_groups.py \
  --mapping_csv data/station_province_mapping_with_override.csv \
  --out_dir out_station_groups
```

### 8.2 รันกราฟตาม “ภูมิภาค” (ตัวอย่าง Northeast)
```bash
python code/10_station_level_plots_v2.py \
  --input_csv data/env_daily_with_rice_monthly_raw.csv \
  --out_dir out_eda_region_northeast \
  --target bph_log1p \
  --stations_file out_station_groups/station_lists/by_region/stations_region_Northeast.csv \
  --per_station_timeseries
```

### 8.3 รันกราฟตาม “จังหวัด” (ตัวอย่าง Chiang Rai)
```bash
python code/10_station_level_plots_v2.py \
  --input_csv data/env_daily_with_rice_monthly_raw.csv \
  --out_dir out_eda_province_chiang_rai \
  --target bph_log1p \
  --stations_file out_station_groups/station_lists/by_province/stations_province_Chiang_Rai.csv \
  --per_station_timeseries
```

---

## 9) อธิบาย “การวิเคราะห์กราฟ” แต่ละภาพ (Interpretation)

### A1 — `A1_timeseries_target_by_station.png` (ภาพรวมหลายสถานี)
**ดูอะไร**
- Seasonality / ช่วงพีคของการระบาด (spike)
- ความต่างของระดับ target ระหว่างสถานี  
**ใช้ต่อ**
- เลือกสถานีตัวอย่างเล่าเรื่องในบทนำเสนอ
- ตรวจช่วงข้อมูลหาย/ผิดปกติ ก่อนทำ sliding window

### A1b — `A1_per_station/A1_timeseries_<station>.png` (รายสถานี)
**ดูอะไร**
- ความต่อเนื่องของข้อมูลรายสถานี (continuity)
- Outlier และ “พีคซ้ำปีต่อปี”  
**ใช้ต่อ**
- แนบภาคผนวก / ทำกรณีศึกษา (case study) รายสถานี
- ใช้ตัดสินใจว่าจะตัด/เก็บสถานีนี้เพื่อเทรน

### A2 — `A2_missingness_per_station.png` (Missingness)
**ดูอะไร**
- สถานีไหนข้อมูลขาดมากในคอลัมน์สำคัญ (ฝน/ลม/target ฯลฯ)  
**ใช้ต่อ**
- ถ้าขาดมาก: พิจารณา
  - ตัดสถานีออก
  - ทำ imputation (เติมค่า)
  - หรือปรับฟีเจอร์ rolling เพื่อลดผล missing

### A3 — `A3_boxplot_target_by_station_topk.png` (Boxplot target)
**ดูอะไร**
- Median/IQR/Outliers ของ target ต่อสถานี  
**ใช้ต่อ**
- สถานี median สูง ⇒ baseline risk สูง
- outlier เยอะ ⇒ เหตุการณ์ระบาดเฉียบพลัน หรือ data issue (ต้องตรวจ)

### A4 — `A4_scatter_<feature>_vs_<target>.png` (Feature vs Target)
**ดูอะไร**
- ความสัมพันธ์แบบภาพรวมระหว่าง feature กับ target
- การแยกกลุ่มตามสถานี (cluster by station)  
**ใช้ต่อ**
- ถ้าเห็น threshold/โค้ง ⇒ สนับสนุนโมเดล non-linear หรือเพิ่ม transform
- ถ้าคลัสเตอร์ตามสถานี ⇒ เพิ่ม spatial/station features หรือ station embedding (ถ้าทำ deep learning)

### A5 — Heatmap correlation (Pearson/Spearman)
ไฟล์:
- `A5_corr_heatmap_pearson.png`
- `A5_corr_heatmap_spearman.png`

**Pearson**: เน้นสัมพันธ์เชิงเส้น  
**Spearman**: เน้นสัมพันธ์แบบ monotonic (ไม่ต้องเป็นเส้นตรง)

**ใช้ต่อ**
- เลือก features ที่สัมพันธ์ “สม่ำเสมอหลายสถานี”
- สถานีที่ corr “กลับทิศ” บางฟีเจอร์ ⇒ ตรวจ microclimate หรือคุณภาพข้อมูล

---

## 10) Troubleshooting (ปัญหาพบบ่อย)
### 10.1 หาไฟล์ไม่เจอ
- ตรวจว่า path ถูก: `data/env_daily_with_rice_monthly_raw.csv`
```bash
ls -lah data/env_daily_with_rice_monthly_raw.csv
```

### 10.2 ชื่อคอลัมน์ไม่ตรง
- เปิดดูหัวตาราง:
```bash
python - <<'PY'
import pandas as pd
df = pd.read_csv("data/env_daily_with_rice_monthly_raw.csv", encoding="utf-8-sig")
print(df.columns.tolist()[:50])
PY
```

### 10.3 ขอรายชื่อสถานีทั้งหมดในไฟล์
```bash
python - <<'PY'
import pandas as pd
df = pd.read_csv("data/env_daily_with_rice_monthly_raw.csv", encoding="utf-8-sig")
print("n_stations =", df["station_id"].nunique())
print(df["station_id"].astype(str).value_counts().head(20))
PY
```

---

## 11) Help
ดูพารามิเตอร์ทั้งหมด:
```bash
python code/10_station_level_plots_v2.py -h
```
