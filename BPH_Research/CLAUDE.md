# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

งานวิจัยการพยากรณ์ **Brown Planthopper (BPH / เพลี้ยกระโดดสีน้ำตาล)** ในนาข้าวภาคตะวันออกเฉียงเหนือของไทย  
ใช้ข้อมูลสภาพแวดล้อมรายวันร่วมกับโมเดล Deep Learning 3 ประเภท: **LSTM**, **CNN-LSTM**, **Transformer**

## โครงสร้างโฟลเดอร์

```
BPH_Research/
├── scripts/       ← Python scripts + run_train_auto.sh
├── data/          ← ข้อมูล CSV ดิบ (BPH + สภาพอากาศ + ข้าว)
├── results/       ← output ทั้งหมด (flat)
│   ├── out_quality_gate/
│   ├── out_feature_sets_w{W}_h{H}/
│   └── out_train_w{W}_h{H}/
├── manuscript/    ← บทความ .docx
├── docs/          ← คู่มือเพิ่มเติม
└── CLAUDE.md
```

> **Working directory:** รันทุกคำสั่งจาก `BPH_Research/` เสมอ

## Pipeline (รันตามลำดับ)

```
data/ CSV  →  quality_gate  →  build NPZ  →  train  →  summarize
```

### 1) Quality Gate
```bash
python scripts/quality_gate.py \
  --input_csv data/env_daily_with_rice_monthly_raw.csv \
  --out_dir results/out_quality_gate \
  --mode soft --drop_rows_missing_mandatory
```
ผลลัพธ์: `results/out_quality_gate/cleaned_raw.csv` และ `quality_report.json`

### 2) สร้าง Sliding Window Sequences (NPZ)
ใช้ nan-safe เสมอ — ตัวอย่าง 3 ชุดหลัก:
```bash
python scripts/03_build_feature_sets_and_sequences_v2_nan_safe.py \
  --input_csv results/out_quality_gate/cleaned_raw.csv \
  --out_dir results/out_feature_sets_w30_h1 \
  --window 30 --horizon 1 --roll_days 7 \
  --all_sets --require_consecutive

python scripts/03_build_feature_sets_and_sequences_v2_nan_safe.py \
  --input_csv results/out_quality_gate/cleaned_raw.csv \
  --out_dir results/out_feature_sets_w60_h7 \
  --window 60 --horizon 7 --roll_days 7 \
  --all_sets --require_consecutive

python scripts/03_build_feature_sets_and_sequences_v2_nan_safe.py \
  --input_csv results/out_quality_gate/cleaned_raw.csv \
  --out_dir results/out_feature_sets_w90_h14 \
  --window 90 --horizon 14 --roll_days 14 \
  --all_sets --require_consecutive
```
`--all_sets` สร้างทั้ง 3 feature set: `core/`, `context/`, `full/`

### 3) ตรวจ NaN/Inf ก่อนเทรนทุกครั้ง
```bash
python scripts/inspect_npz_for_nan.py \
  --npz results/out_feature_sets_w30_h1/core/sequences_window30_h1.npz
```
ถ้า `nan>0` หรือ `inf>0` → rebuild NPZ ใหม่

### 4) เทรนโมเดลทั้ง 3 (แนะนำ)
```bash
chmod +x scripts/run_train_auto.sh   # ครั้งแรกครั้งเดียว

./scripts/run_train_auto.sh \
  results/out_feature_sets_w30_h1/core/sequences_window30_h1.npz \
  results/out_train_w30_h1
```

รันแบบ background (SSH-safe):
```bash
nohup ./scripts/run_train_auto.sh \
  results/out_feature_sets_w60_h7/core/sequences_window60_h7.npz \
  results/out_train_w60_h7 > train_w60_h7.log 2>&1 &

tail -f train_w60_h7.log
```

### 5) สรุปผลเปรียบเทียบ
```bash
python scripts/summarize_runs.py \
  --root results/out_train_w30_h1 \
  --out results/out_train_w30_h1/summary
```
ได้: `summary/comparison.csv` และ `summary/comparison.md`

### 6) Visualization (optional)
```bash
python scripts/main_program.py \
  --plot_columns temp bph_raw \
  --station_id 101 --tail 365
```

## แนวทางเลือก Window / Horizon / โมเดล

| Horizon (H) | Window (W) | roll_days | โมเดลแนะนำ |
|---:|---:|---:|---|
| 1 วัน | 30–45 | 7 | LSTM, CNN-LSTM |
| 7 วัน | 45–60 | 7 | CNN-LSTM |
| 14 วัน | 60–90 | 14 | Transformer |

## โครงสร้าง results/

```
results/out_feature_sets_w{W}_h{H}/
  core/       ← สภาพอากาศ + target
  context/    ← core + rice variety context
  full/       ← core + context + เพิ่มเติม

results/out_train_w{W}_h{H}/
  lstm_core/
  cnn_lstm_core/
  transformer_core/
  summary/
  logs/
    01_lstm_core.log
    02_cnn_lstm_core.log
    03_transformer_core.log
    04_summary.log
```

## Scripts หลัก

| ไฟล์ | หน้าที่ |
|---|---|
| `scripts/quality_gate.py` | ตรวจสอบ + ทำความสะอาดข้อมูลดิบ (data contract) |
| `scripts/03_build_feature_sets_and_sequences_v2_nan_safe.py` | สร้าง sliding window NPZ (nan-safe) |
| `scripts/11_train_lstm_compat.py` | เทรน LSTM |
| `scripts/12_train_cnn_lstm_compat.py` | เทรน CNN-LSTM |
| `scripts/13_train_transformer_compat.py` | เทรน Transformer |
| `scripts/train_seq_utils_compat.py` | shared utilities (load NPZ, metrics, plots) |
| `scripts/summarize_runs.py` | รวม metrics ทุก run เป็นตาราง |
| `scripts/inspect_npz_for_nan.py` | ตรวจ NaN/Inf ใน NPZ |
| `scripts/run_train_auto.sh` | รันโมเดลทั้ง 3 ต่อเนื่อง + สรุปผล |
| `scripts/main_program.py` | plot time-series รายสถานี |

## NPZ Format

Shape: `(samples, window, features)` — keys: `X_train`, `y_train`, `X_val`, `y_val`, `X_test`, `y_test`, `y_date_*`, `station_*`, `meta`  
Target คือ `log1p(bph_count)`

## Training Presets

| Mode | epochs | patience | batch_size | lr |
|---|---|---|---|---|
| FAST | 50 | 10 | 256 | 0.001 |
| BALANCED (แนะนำ) | 120–150 | 20–30 | 128 | 0.0005 |
| QUALITY | 200–260 | 35–45 | 64–128 | 0.0003–0.0005 |

ถ้า RAM/VRAM ตึง: ลด `--batch_size`, `--d_model`/`--num_layers` (Transformer) หรือ `--conv_filters` (CNN-LSTM)

## หยุดการเทรน

```bash
pkill -f "train_"       # หยุดทุก Python process ที่กำลังเทรน
ps aux | grep python    # ดู PID
kill <PID>
```
