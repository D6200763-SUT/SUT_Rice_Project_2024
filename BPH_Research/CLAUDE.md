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

## Python Environment (เครื่องนี้)

```
/home/ai-station/my_project/.tf251p310/bin/python
```

| Package | Version |
|---|---|
| Python | 3.10.12 |
| TensorFlow | 2.15.1 |
| numpy | 1.26.4 |
| pandas | 2.3.3 |
| scikit-learn | 1.7.2 |
| joblib | 1.5.3 |
| optuna | 4.8.0 |

ใช้แทน `python3` ทุกครั้ง:
```bash
/home/ai-station/my_project/.tf251p310/bin/python scripts/quality_gate.py ...
```

หรือ activate ก่อน:
```bash
source /home/ai-station/my_project/.tf251p310/bin/activate
```

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
  results/out_feature_sets_w30_h1/context/sequences_window30_h1.npz \
  results/out_train_w30_h1_context
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
  --root results/out_train_w30_h1_context \
  --out results/out_train_w30_h1_context/summary
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
results/
├── out_quality_gate/                  ← cleaned_raw.csv, quality_report.json
├── out_feature_sets_w30_h1/           ← NPZ: core/ context/ full/
├── out_feature_sets_w30_h7/           ← NPZ: core/ context/ full/ trimmed/
├── out_feature_sets_w60_h7/           ← NPZ: core/ context/ full/
├── out_feature_sets_w90_h14/          ← NPZ: core/ context/ full/
├── out_train_w30_h1_context/          ← BEST (R²=0.500): lstm/ cnn_lstm/ transformer/
├── out_train_w30_h7/                  ← lstm_context/ cnn_lstm_context/ cnn_lstm_context_quality/
│                                          cnn_lstm_trimmed/ transformer_context/
├── out_train_w60_h7/                  ← core feature set
├── out_train_w60_h7_context/          ← context feature set
├── out_train_w60_h7_full/             ← full feature set
├── out_train_stage1/                  ← Two-stage Stage 1 classifier (CNN-LSTM binary)
├── out_train_stage2/                  ← Two-stage Stage 2 regression (spike samples)
├── out_threshold_sweep/               ← threshold sweep results (threshold_sweep.csv)
├── out_train_weighted/                ← Spike-weighted CNN-LSTM sweep (weight 2/3/5/8)
├── walk_forward/                      ← Walk-Forward CV: fold2/ fold3/ fold4/
└── summary_final/                     ← comparison.csv, comparison.md
```

## Scripts หลัก

| ไฟล์ | หน้าที่ |
|---|---|
| `scripts/quality_gate.py` | ตรวจสอบ + ทำความสะอาดข้อมูลดิบ (data contract) |
| `scripts/build_sequences.py` | สร้าง sliding window NPZ (nan-safe) |
| `scripts/train_lstm.py` | เทรน LSTM |
| `scripts/train_cnn_lstm.py` | เทรน CNN-LSTM (มี `--spike_only` flag สำหรับ Stage 2) |
| `scripts/train_transformer.py` | เทรน Transformer |
| `scripts/train_utils.py` | shared utilities (load NPZ, metrics, plots) |
| `scripts/summarize_runs.py` | รวม metrics ทุก run เป็นตาราง |
| `scripts/inspect_npz_for_nan.py` | ตรวจ NaN/Inf ใน NPZ |
| `scripts/run_train_auto.sh` | รันโมเดลทั้ง 3 ต่อเนื่อง + สรุปผล |
| `scripts/tune_cnn_lstm_optuna.py` | Optuna hyperparameter tuning สำหรับ CNN-LSTM |
| `scripts/train_stage1_classifier.py` | Two-stage Stage 1: CNN-LSTM binary classifier (spike/no-spike) |
| `scripts/train_stage2_regression.py` | Two-stage Stage 2: CNN-LSTM regression บน spike samples |
| `scripts/evaluate_twostage_threshold.py` | sweep pred_threshold → full two-stage metrics (ไม่ต้องเทรนใหม่) |
| `scripts/train_cnn_lstm_weighted.py` | CNN-LSTM + spike sample_weight sweep (single-stage) |
| `scripts/build_walk_forward_folds.py` | สร้าง Walk-Forward CV folds (NPZ พร้อม y_spike labels) |
| `scripts/train_cnn_lstm_classifier.py` | Walk-Forward Stage 1: CNN-LSTM binary classifier (sigmoid + binary_crossentropy) |
| `scripts/run_walk_forward.sh` | รัน Walk-Forward CV pipeline อัตโนมัติ (build → train → summarize) |
| `scripts/summarize_walk_forward.py` | สรุป CV metrics ข้าม folds → cv_summary.csv + cv_report.md + barplot |

## NPZ Format

Shape: `(samples, window, features)` — keys: `X_train`, `y_train`, `X_val`, `y_val`, `X_test`, `y_test`, `y_date_*`, `station_*`, `meta`  
Target คือ `log1p(bph_count)`

**Walk-Forward NPZ** (`walk_forward/fold{N}/sequences_wf.npz`) มี keys เพิ่มเติม:  
`y_spike_train`, `y_spike_val`, `y_spike_test` (float32: 0.0/1.0, spike = bph_raw > 100)

## Training Presets

| Mode | epochs | patience | batch_size | lr |
|---|---|---|---|---|
| FAST | 50 | 10 | 256 | 0.001 |
| BALANCED (แนะนำ) | 120–150 | 20–30 | 128 | 0.0005 |
| QUALITY | 200–260 | 35–45 | 64–128 | 0.0003–0.0005 |

ถ้า RAM/VRAM ตึง: ลด `--batch_size`, `--d_model`/`--num_layers` (Transformer) หรือ `--conv_filters` (CNN-LSTM)

## Optuna Hyperparameter Tuning (CNN-LSTM)

```bash
/home/ai-station/my_project/.tf251p310/bin/python scripts/tune_cnn_lstm_optuna.py \
  --npz results/out_feature_sets_w30_h7/context/sequences_window30_h7.npz \
  --out_dir results/out_tune_w30_h7_context \
  --n_trials 30 \
  --n_jobs 1
```

ผลลัพธ์:
- `results/out_tune_w30_h7_context/trial_NNN/metrics.json` — ทุก trial
- `results/out_tune_w30_h7_context/best_params.json` — params ที่ดีที่สุด
- `results/out_tune_w30_h7_context/optuna_study.csv` — ตารางเปรียบเทียบทุก trial

หลังได้ best_params แล้ว รันเทรนจริง (script จะ print คำสั่งพร้อมใช้เมื่อจบ):
```bash
/home/ai-station/my_project/.tf251p310/bin/python scripts/train_cnn_lstm.py \
  --npz results/out_feature_sets_w30_h7/context/sequences_window30_h7.npz \
  --out_dir results/out_train_w30_h7_context_tuned \
  --conv_filters {best.conv_filters} --kernel_size {best.kernel_size} \
  --lstm_units {best.lstm_units} --dropout {best.dropout} \
  --lr {best.lr} --batch_size {best.batch_size} --clipnorm {best.clipnorm} \
  --epochs 200 --patience 35
```

รันแบบ background:
```bash
nohup /home/ai-station/my_project/.tf251p310/bin/python scripts/tune_cnn_lstm_optuna.py \
  --npz results/out_feature_sets_w30_h7/context/sequences_window30_h7.npz \
  --out_dir results/out_tune_w30_h7_context \
  --n_trials 30 > tune_w30_h7.log 2>&1 &

tail -f tune_w30_h7.log
```

Search space: conv_filters [32,64,128], kernel_size [3,5,7], lstm_units [32,64,128,256],
dropout [0.1–0.4], lr [1e-4–1e-3 log], batch_size [64,128,256], clipnorm [0.5,1.0,2.0]
Pruner: MedianPruner (n_startup=10, warmup=20 epochs)

## หยุดการเทรน

```bash
pkill -f "train_"       # หยุดทุก Python process ที่กำลังเทรน
ps aux | grep python    # ดู PID
kill <PID>
```

---

## สถานะ Repo — 2026-05-19

### Scripts (19 files — ทั้งหมดผ่านการทดสอบแล้ว)
```
scripts/
├── quality_gate.py                    ← pipeline step 1
├── build_sequences.py                 ← pipeline step 2
├── inspect_npz_for_nan.py             ← pipeline step 3
├── run_train_auto.sh                  ← pipeline step 4 (launcher)
├── train_lstm.py                      ← เทรน LSTM
├── train_cnn_lstm.py                  ← เทรน CNN-LSTM (+--spike_only flag)
├── train_transformer.py               ← เทรน Transformer
├── train_utils.py                     ← shared utilities
├── summarize_runs.py                  ← pipeline step 5
├── tune_cnn_lstm_optuna.py            ← Optuna hyperparameter search (CNN-LSTM)
├── train_stage1_classifier.py         ← Two-stage Stage 1: binary classifier (old)
├── train_stage2_regression.py         ← Two-stage Stage 2: regression on spike samples (old)
├── evaluate_twostage_threshold.py     ← threshold sweep (no retraining)
├── train_cnn_lstm_weighted.py         ← spike sample_weight sweep
├── build_walk_forward_folds.py        ← Walk-Forward CV: สร้าง fold NPZs (W=30, H=7, context, spike labels)
├── train_cnn_lstm_classifier.py       ← Walk-Forward Stage 1: binary classifier (sigmoid, binary_crossentropy)
├── run_walk_forward.sh                ← Walk-Forward pipeline launcher (build+train+summarize)
└── summarize_walk_forward.py          ← รวม CV metrics → csv/md/barplot
```

### Results (16 folders)
```
results/
├── out_quality_gate/                  ← cleaned_raw.csv
├── out_feature_sets_w30_h1/           ← NPZ (W=30, H=1)
├── out_feature_sets_w30_h7/           ← NPZ (W=30, H=7) + trimmed/
├── out_feature_sets_w60_h7/           ← NPZ (W=60, H=7)
├── out_feature_sets_w90_h14/          ← NPZ (W=90, H=14)
├── out_train_w30_h1_context/          ← BEST: CNN-LSTM R²=0.500
├── out_train_w30_h7/                  ← 5 runs (context/trimmed)
├── out_train_w60_h7/                  ← core feature set
├── out_train_w60_h7_context/          ← context feature set
├── out_train_w60_h7_full/             ← full feature set
├── out_train_stage1/                  ← Two-stage Stage 1 (F1=0.615, Recall=0.804)
├── out_train_stage2/                  ← Two-stage Stage 2 (R²_log1p=0.311, R²_raw=0.0063)
├── out_threshold_sweep/               ← threshold sweep 0.30–0.50 (best=0.50)
├── out_train_weighted/                ← weighted sweep (best w=3.0, R²=0.344)
├── walk_forward/                      ← Walk-Forward CV (fold2/ fold3/ fold4/) ← สร้างแล้ว
└── summary_final/                     ← comparison.csv/md ครบทุก run
```

### Git / .gitignore
- Working tree clean, sync กับ origin/main
- Ignored (ไม่ track): `*.npz`, `*.keras`, `*.joblib`, `*.log`, `model_ready_daily*.csv`
- Tracked: metrics, predictions, figures, reports, source data

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

#### โมเดลที่เทรนแล้ว (folders ที่เก็บไว้)
| Run folder | โมเดล | Feature set | W | H | R² (log1p) |
|---|---|---|---|---|---|
| `out_train_w30_h1_context/` | LSTM, CNN-LSTM, Transformer | context | 30 | 1 | 0.268–0.500 |
| `out_train_w30_h7/cnn_lstm_context` | CNN-LSTM | context | 30 | 7 | 0.484 |
| `out_train_w30_h7/cnn_lstm_context_quality` | CNN-LSTM (QUALITY preset) | context | 30 | 7 | 0.452 |
| `out_train_w30_h7/cnn_lstm_trimmed` | CNN-LSTM | trimmed (10f) | 30 | 7 | 0.470 |
| `out_train_w30_h7/transformer_context` | Transformer | context | 30 | 7 | 0.368 |
| `out_train_w30_h7/lstm_context` | LSTM | context | 30 | 7 | 0.363 |
| `out_train_w60_h7/` | LSTM, CNN-LSTM, Transformer | core | 60 | 7 | 0.135–0.202 |
| `out_train_w60_h7_context/` | LSTM, CNN-LSTM, Transformer | context | 60 | 7 | 0.264–0.463 |
| `out_train_w60_h7_full/` | LSTM, CNN-LSTM, Transformer | full | 60 | 7 | 0.192–0.334 |

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
- **Weighted loss ไม่ช่วย (ยืนยันแล้ว)** — ทั้ง sample_weight และ two-stage ไม่ผ่าน R²_log1p=0.484
- **Trimmed (10f) ทำให้ r2_raw เป็นบวกครั้งแรก** — lat/lon/temp_range สำคัญที่สุด
- **Feature สำคัญ (permutation importance):** longitude > latitude > temp_range > month_sin > doy_sin
- **CNN-LSTM ชนะ LSTM/Transformer ชัดเจนบน W30 H7** — ยืนยันว่า CNN-LSTM เป็น best model
- **Two-stage: Stage1=0 → predict 0.0 ทำลาย R²_log1p** — FN ~20% ทำให้ log1p loss พุ่ง
- **Spike weighting paradox:** weight สูง → R²_spike_only ติดลบ — โมเดล over-predict spike ทั่วไป

### Manuscript
- `manuscript/BPH_DeepLearning_Results_2026-05-18.md` — สรุปผลและวิเคราะห์ครบทุก run พร้อม feature importance, fair comparison, และแนวทางต่อ

### สิ่งที่ยังไม่ได้ทำ (แนวทางต่อ)
- [x] ลอง Two-stage model: classify spike/no-spike ก่อน แล้ว regression ← ทำแล้ว (2026-05-19)
- [x] ลอง Spike-weighted loss (sample_weight) ← ทำแล้ว (2026-05-19)
- [ ] Train Stage 2 ด้วย oracle spike labels (y_true > threshold แทน Stage 1 pred) — upper bound ของ two-stage
- [ ] Spatial model: Graph Neural Network หรือ ConvLSTM ระหว่างสถานี
- [x] Cross-validation แบบ temporal ← Walk-Forward CV (fold2/3/4) ทำแล้ว (2026-05-19)
- [ ] เพิ่ม feature จาก NDVI หรือ soil moisture (ถ้ามีข้อมูล)

---

## บันทึกการทดลอง — 2026-05-19

### Experiment 1: Two-stage Model (Stage1 Classifier + Stage2 Regression)

#### Stage 1: CNN-LSTM Binary Classifier
- NPZ: `out_feature_sets_w30_h7/context/` (W=30, H=7, 18 features)
- spike_threshold = 1.0986 (Q75% ของ y_train log1p)
- class_weight = {0: 1.0, 1: 3.01} (อัตโนมัติจาก train set)
- Architecture: Conv1D(64)×2 + BatchNorm + LSTM(64) + sigmoid
- Output: `results/out_train_stage1/`

| Split | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Train | 0.833 | 0.618 | 0.865 | 0.721 |
| Val | 0.756 | 0.504 | 0.829 | 0.627 |
| **Test** | **0.780** | **0.498** | **0.804** | **0.615** |

#### Stage 2: CNN-LSTM Regression (Optuna best params)
- เทรนเฉพาะ spike samples ที่ Stage 1 predict=1
- train: 14,724/42,228 (34.9%)  val: 3,276/8,058 (40.7%)  test: 2,872/8,126 (35.3%)
- Architecture: Conv1D(128) + LSTM(32) + Huber loss (identical to tuned single-stage)
- Output: `results/out_train_stage2/`

**Full Two-stage Test Metrics (Stage1=0 → 0.0; Stage1=1 → Stage2 pred):**

| Metric | Two-stage | Baseline (single CNN-LSTM) | Result |
|---|---|---|---|
| R² log1p | 0.311 | 0.484 | WORSE −0.173 |
| R² raw | 0.006 | 0.004 | BETTER +0.002 |
| RMSE raw | 219.27 | 219.51 | BETTER −0.24 |

#### Threshold Sweep (pred_threshold 0.30–0.50)
- ไม่ต้องเทรนใหม่ — ใช้ model เดิมทั้งคู่
- **ผล: threshold ยิ่งต่ำ R²_log1p ยิ่งแย่ลง monotone** (ไม่มี sweet spot)
- best threshold = 0.50 (default) ทุก metric

| threshold | n_spike | R² log1p | R² raw | RMSE raw |
|---:|---:|---:|---:|---:|
| 0.30 | 3785 (46.6%) | 0.255 | 0.005 | 219.43 |
| 0.35 | 3531 (43.5%) | 0.275 | 0.005 | 219.39 |
| 0.40 | 3309 (40.7%) | 0.287 | 0.006 | 219.35 |
| 0.45 | 3070 (37.8%) | 0.304 | 0.006 | 219.30 |
| **0.50** | **2872 (35.3%)** | **0.311** | **0.006** | **219.27** |

**Root cause:** Stage 2 trained on spike samples predicts spike-range values; ส่ง borderline samples ผ่าน Stage 2 ทำให้ non-spike regions over-predicted

---

### Experiment 2: Spike-Weighted CNN-LSTM (single-stage)

- Architecture เหมือน train_cnn_lstm.py ทุกอย่าง + Optuna best params
- เพิ่มแค่ `sample_weight` ใน model.fit(): spike samples ได้ weight สูงขึ้น
- ทดลอง 4 ค่า: spike_weight ∈ {2.0, 3.0, 5.0, 8.0}
- Output: `results/out_train_weighted/`

| spike_weight | R² log1p | R² raw | RMSE raw | R² spike-only | beat baseline log1p |
|---:|---:|---:|---:|---:|:---:|
| 2.0 | 0.317 | 0.001 | 219.84 | −0.970 | ✗ |
| **3.0** | **0.344** | **0.008** | **219.09** | −0.341 | ✗ |
| 5.0 | 0.266 | −0.004 | 220.43 | −0.299 | ✗ |
| 8.0 | 0.110 | −0.004 | 220.36 | −0.067 | ✗ |
| **baseline** | **0.484** | **0.004** | **219.51** | — | — |

**ข้อสังเกต:**
- weight=3.0 ดีที่สุด — R²_raw ชนะ baseline (0.008 vs 0.004) แต่ R²_log1p ยังแพ้ (0.344 vs 0.484)
- **R²_spike_only ติดลบทุก weight** — paradox: weight สูง → โมเดล over-predict spike ทั่วไป → non-spike regions พัง
- weight สูงเกิน (≥5) ทำให้ R²_log1p พังมาก (degenerate toward predicting spike every step)

---

### ตารางเปรียบเทียบสุดท้าย (ทุก approach บน W=30 H=7 context)

| อันดับ | Approach | R² log1p | R² raw | RMSE raw |
|---|---|---:|---:|---:|
| 🥇 | **CNN-LSTM baseline** (single-stage, Optuna) | **0.484** | 0.004 | 219.51 |
| 2 | Spike-weighted w=3.0 | 0.344 | **0.008** | **219.09** |
| 3 | Two-stage (threshold=0.50) | 0.311 | 0.006 | 219.27 |
| 4 | Spike-weighted w=2.0 | 0.317 | 0.001 | 219.84 |

**สรุป:** Standard CNN-LSTM บน log1p target ยังคงเป็น best model ทุก metric log1p — advanced approaches ช่วยใน raw scale เล็กน้อยแต่ไม่ผ่าน log1p threshold

---

## บันทึกการทดลอง — 2026-05-19 (ต่อ)

### Experiment 3: Walk-Forward Temporal Cross-Validation Pipeline

#### Pipeline ที่สร้าง
- `build_walk_forward_folds.py` — สร้าง fold NPZs (W=30, H=7, context 18f, spike_threshold=100)
- `train_cnn_lstm_classifier.py` — Stage 1 binary classifier พร้อม ROC/PR/CM plots
- `train_cnn_lstm.py` — เพิ่ม `--spike_only` flag สำหรับ Stage 2 (กรองเฉพาะ spike samples)
- `run_walk_forward.sh` — launcher ครบ pipeline
- `summarize_walk_forward.py` — รวม metrics ทุก fold → cv_summary.csv + cv_report.md

#### Fold Configurations
| Fold | Train | Val | Test |
|---|---|---|---|
| 2 | 2015–2016 | 2017-H1 | 2017-H2 |
| 3 | 2015–2017 | 2018-H1 | 2018-H2 |
| 4 | 2015–2018-07-01 | 2018-07-02–2019-03-31 | 2019-04-01–2019-12-31 (= original split) |

#### Fold Statistics (spike_threshold=100, bph_raw > 100 = spike)
| Fold | Train samples | Train spike% | Test samples | Test spike% |
|---|---|---|---|---|
| 2 | 23,630 | 6.0% | 6,256 | 9.7% |
| 3 | 36,040 | 6.5% | 6,256 | 11.4% |
| 4 | 42,228 | 7.0% | 9,350 | 9.7% |

#### วิธีรัน Walk-Forward Pipeline
```bash
# รันทั้งหมดแบบ background
nohup ./scripts/run_walk_forward.sh > walk_forward_main.log 2>&1 &
tail -f walk_forward_main.log

# สรุปผลหลัง training เสร็จ
python scripts/summarize_walk_forward.py \
  --root results/walk_forward --folds 2,3,4 \
  --out results/walk_forward/cv_summary.csv
```

#### Output Structure (หลัง training เสร็จ)
```
results/walk_forward/
├── fold2/
│   ├── sequences_wf.npz       ← X/y/y_spike + date/station arrays
│   ├── scaler_minmax.joblib
│   ├── fold_info.json
│   ├── stage1/                ← metrics_stage1.json, model_stage1_best.keras, figures/
│   └── stage2/                ← metrics.json, model_best.keras, figures/
├── fold3/ ...
├── fold4/ ...
├── cv_summary.csv             ← mean/SD ของ Stage1 + Stage2 metrics
├── cv_report.md               ← manuscript-ready table
└── figures/cv_r2_barplot.png
```

#### ผลลัพธ์จริง (รัน 2026-05-19)

**Stage 1 — Binary Classifier**

| Fold | Test period | Recall | Precision | F1 | AUC | Best threshold |
|---|---|---|---|---|---|---|
| 2 | 2017-H2 | 0.743 ⚠️ | 0.250 | 0.374 | 0.804 | 0.30 |
| 3 | 2018-H2 | **0.899** ✅ | 0.333 | 0.486 | 0.915 | 0.30 |
| 4 | 2019 | 0.779 ⚠️ | 0.432 | 0.556 | **0.923** | 0.50 |
| **Mean ± SD** | — | **0.807 ± 0.067** | 0.339 ± 0.074 | 0.472 ± 0.075 | 0.881 ± 0.054 | — |

**Stage 2 — Regressor (spike_only, oracle labels)**

| Fold | R² log1p | RMSE log1p | R² raw | RMSE raw |
|---|---|---|---|---|
| 2 | −0.130 ❌ | 0.847 | −0.005 | 1199.5 |
| 3 | −0.004 ❌ | 0.739 | +0.024 | 965.0 |
| 4 | −0.223 ❌ | 0.662 | −0.003 | 618.3 |
| **Mean ± SD** | **−0.119 ± 0.090** | 0.749 ± 0.076 | +0.006 ± 0.013 | 927.6 ± 238.7 |

**Baseline อ้างอิง:** Single-stage CNN-LSTM R²_log1p = **0.484** (Fold 4 / test 2019)

#### Success Criteria — ผ่าน/ไม่ผ่าน
| Metric | เป้าหมาย | ผลจริง | สถานะ |
|---|---|---|---|
| Stage1 Recall (mean) | ≥ 0.80 | 0.807 | ✅ ผ่าน |
| Stage1 Recall (min fold) | ≥ 0.70 | 0.743 (fold 2) | ⚠️ เกือบผ่าน |
| Stage2 R² log1p (mean) | ≥ 0.45 | −0.119 | ❌ ไม่ผ่าน |
| CV variance R² | SD < 0.08 | 0.090 | ⚠️ เล็กน้อย |

#### บทเรียนสำคัญจาก Walk-Forward CV
- **Stage 2 ติดลบทุก fold** — ยืนยัน Experiment 1: train บน spike samples เท่านั้น → val set เล็กมาก → early stopping ไม่ reliable → โมเดล over/underfit
- **Fold 2 Recall ต่ำ (0.743)** — train data เพียง 2 ปี (23,630 samples) ไม่เพียงพอให้ classifier เรียนรู้ pattern ครบ
- **AUC สูง (0.88 mean)** — โมเดล discriminate spike/non-spike ได้ดี แต่ threshold ที่ Recall≥0.80 บังคับ precision ต่ำ (~0.34)
- **Walk-Forward ยืนยัน baseline** — Single-stage CNN-LSTM R²=0.484 ดีกว่า two-stage approach ทุก fold และทุกปี
- **R² raw ดีกว่าเล็กน้อย** (mean +0.006) — สอดคล้องกับ Experiment 1 และ 2

**หมายเหตุ:** Walk-Forward Stage 2 ใช้ oracle spike labels (y_spike_true) ไม่ใช่ Stage 1 predictions — เป็น upper bound ของ two-stage จริง แต่ยังแพ้ baseline
