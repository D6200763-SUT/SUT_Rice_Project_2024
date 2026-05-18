# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

งานวิจัยการพยากรณ์ **Brown Planthopper (BPH / เพลี้ยกระโดดสีน้ำตาล)** 
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
python scripts/build_sequences.py \
  --input_csv results/out_quality_gate/cleaned_raw.csv \
  --out_dir results/out_feature_sets_w30_h1 \
  --window 30 --horizon 1 --roll_days 7 \
  --all_sets --require_consecutive

python scripts/build_sequences.py \
  --input_csv results/out_quality_gate/cleaned_raw.csv \
  --out_dir results/out_feature_sets_w60_h7 \
  --window 60 --horizon 7 --roll_days 7 \
  --all_sets --require_consecutive

python scripts/build_sequences.py \
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
| `scripts/build_sequences.py` | สร้าง sliding window NPZ (nan-safe) |
| `scripts/train_lstm.py` | เทรน LSTM |
| `scripts/train_cnn_lstm.py` | เทรน CNN-LSTM |
| `scripts/train_transformer.py` | เทรน Transformer |
| `scripts/train_utils.py` | shared utilities (load NPZ, metrics, plots) |
| `scripts/summarize_runs.py` | รวม metrics ทุก run เป็นตาราง |
| `scripts/inspect_npz_for_nan.py` | ตรวจ NaN/Inf ใน NPZ |
| `scripts/run_train_auto.sh` | รันโมเดลทั้ง 3 ต่อเนื่อง + สรุปผล |

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

---

## บันทึกการทดลอง — 2026-05-18

### สิ่งที่ทำไปแล้ว

#### NPZ ที่สร้างแล้ว (ครบทุก feature set)
| Window | Horizon | ที่เก็บ |
|---|---|---|
| W=30 | H=1 | `results/out_feature_sets_w30_h1/` |
| W=30 | H=7 | `results/out_feature_sets_w30_h7/` ← สร้างวันนี้ |
| W=60 | H=7 | `results/out_feature_sets_w60_h7/` |
| W=90 | H=14 | `results/out_feature_sets_w90_h14/` |

`out_feature_sets_w30_h7/` มี trimmed/ เพิ่มเติม (top-10 features จาก permutation importance)

#### โมเดลที่เทรนแล้ว (ครบทุก run)
| Run folder | โมเดล | Feature set | W | H | R² (log1p) |
|---|---|---|---|---|---|
| `out_train_w60_h7/` | LSTM, CNN-LSTM, Transformer | core | 60 | 7 | 0.135–0.202 |
| `out_train_w60_h7_context/` | LSTM, CNN-LSTM, Transformer | context | 60 | 7 | 0.264–0.463 |
| `out_train_w60_h7_full/` | LSTM, CNN-LSTM, Transformer | full | 60 | 7 | 0.192–0.334 |
| `out_train_w30_h7/lstm_context` | LSTM | context | 30 | 7 | 0.363 |
| `out_train_w30_h7/cnn_lstm_context` | CNN-LSTM | context | 30 | 7 | 0.484 |
| `out_train_w30_h7/transformer_context` | Transformer | context | 30 | 7 | 0.368 |
| `out_train_w30_h7/cnn_lstm_context_quality` | CNN-LSTM (QUALITY preset) | context | 30 | 7 | 0.452 |
| `out_train_w30_h7/cnn_lstm_context_weighted` | CNN-LSTM + spike weight α=3.0 | context | 30 | 7 | -0.370 (แย่) |
| `out_train_w30_h7/cnn_lstm_context_weighted2` | CNN-LSTM + spike weight α=1.5 | context | 30 | 7 | 0.175 (แย่) |
| `out_train_w30_h7/cnn_lstm_trimmed` | CNN-LSTM | trimmed (10f) | 30 | 7 | 0.470 |

#### ผลสรุปสุดท้าย — เปรียบเทียบยุติธรรม W=30 H=7 context
| อันดับ | โมเดล | Feature set | W | H | R² | r2_raw |
|---|---|---|---|---|---|---|
| 🥇 | CNN-LSTM | context | 30 | 1 | **0.500** | +0.011 |
| 🥈 | CNN-LSTM | context | 30 | 7 | **0.484** | +0.004 |
| 🥉 | CNN-LSTM | trimmed (10f) | 30 | 7 | 0.470 | **+0.007** |
| 4 | Transformer | context | 30 | 7 | 0.368 | +0.002 |
| 5 | LSTM | context | 30 | 7 | 0.363 | +0.009 |

CNN-LSTM ชนะทุกโมเดลบน W=30 H=7 context อย่างชัดเจน (+12 R² points)

ดูตารางเต็มได้ที่: `results/summary_final/comparison.csv`

### บทเรียนสำคัญ
- **CNN-LSTM + context (18f) ดีที่สุดเสมอ** — rice variety context ช่วยได้จริง
- **W=30 ดีกว่า W=60/90** สำหรับโมเดลนี้
- **full (36f) แพ้ context (18f)** — feature เพิ่มใน full เป็น noise
- **Weighted loss ไม่ช่วย** — log1p compress spike อยู่แล้ว
- **Trimmed (10f) ทำให้ r2_raw เป็นบวกครั้งแรก** — lat/lon/temp_range สำคัญที่สุด
- **Feature สำคัญ (permutation importance):** longitude > latitude > temp_range > month_sin > doy_sin
- **CNN-LSTM ชนะ LSTM/Transformer ชัดเจนบน W30 H7** — ยืนยันว่า CNN-LSTM เป็น best model

### Manuscript
- `manuscript/BPH_DeepLearning_Results_2026-05-18.md` — สรุปผลและวิเคราะห์ครบทุก run พร้อม feature importance, fair comparison, และแนวทางต่อ

### สิ่งที่ยังไม่ได้ทำ (แนวทางต่อ)
- [ ] ลอง Two-stage model: classify spike/no-spike ก่อน แล้ว regression
- [ ] Spatial model: Graph Neural Network หรือ ConvLSTM ระหว่างสถานี
- [ ] Cross-validation แบบ temporal (ปัจจุบัน split เดียว train 2015–2018)
- [ ] เพิ่ม feature จาก NDVI หรือ soil moisture (ถ้ามีข้อมูล)
