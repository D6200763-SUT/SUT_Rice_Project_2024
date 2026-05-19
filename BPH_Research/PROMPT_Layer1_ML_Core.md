# PROMPT: ชั้น 1 — ML Core (Two-stage CNN-LSTM + Walk-Forward Temporal CV)
# ใช้กับ: Claude Code (claude.ai/code) ใน BPH_Research/
# รันจาก: BPH_Research/
# Python: /home/ai-station/my_project/.tf251p310/bin/python

---

## CONTEXT (บริบทที่ Claude Code ต้องรู้ก่อน)

งานวิจัยพยากรณ์เพลี้ยกระโดดสีน้ำตาล (BPH) ในภาคอีสาน ไทย
ข้อมูล: รายวัน 33 สถานี ปี 2015–2019 (cleaned_raw.csv)
Best model ที่ผ่านมา: CNN-LSTM + context (18 features) + W=30, H=7 → R²=0.484

Scripts ที่มีอยู่แล้ว (ห้ามแก้ไข):
- scripts/quality_gate.py
- scripts/build_sequences.py
- scripts/train_cnn_lstm.py          ← Architecture: Conv1D(64,k=5)→MaxPool1D(2)→Dropout(0.25)→LSTM(64)→Dense(64,relu)→Dense(1)
- scripts/train_lstm.py
- scripts/train_transformer.py
- scripts/train_utils.py             ← shared: load_sequences_npz, compute_metrics, inverse_log1p, etc.
- scripts/summarize_runs.py
- scripts/inspect_npz_for_nan.py
- scripts/run_train_auto.sh

Python env: /home/ai-station/my_project/.tf251p310/bin/python
TF: 2.15.1 | numpy: 1.26.4 | pandas: 2.3.3 | sklearn: 1.7.2

---

## TASK 1: สร้าง build_walk_forward_folds.py

สร้างไฟล์ `scripts/build_walk_forward_folds.py` ที่ทำหน้าที่:

**Input:**
- `--input_csv results/out_quality_gate/cleaned_raw.csv`
- `--out_dir results/walk_forward`
- `--folds 2,3,4`             (Fold2=test2017, Fold3=test2018, Fold4=test2019)
- `--min_train_years 2`       (train ต้องมีข้อมูลอย่างน้อย 2 ปีเต็ม)
- `--window 30`
- `--horizon 7`
- `--roll_days 7`
- `--feature_set context`     (18 features)
- `--spike_threshold 100`     (bph_raw > 100 = spike)

**Logic การตัด fold:**
```
Fold 2: train=2015-01-01~2016-12-31 | val=2017-01-01~2017-06-30 | test=2017-07-01~2017-12-31
Fold 3: train=2015-01-01~2017-12-31 | val=2018-01-01~2018-06-30 | test=2018-07-01~2018-12-31
Fold 4: train=2015-01-01~2018-07-01 | val=2018-07-02~2019-03-31 | test=2019-04-01~2019-12-31
```
(Fold 4 = split เดิมที่เคยใช้ เพื่อให้เปรียบเทียบกับผลเก่าได้)

**สิ่งที่ต้องทำในแต่ละ fold:**
1. filter cleaned_raw.csv ตาม date range ที่กำหนด
2. เพิ่ม feature engineering เหมือน build_sequences.py:
   - cyclical: month_sin, month_cos, doy_sin, doy_cos
   - rolling: temp_7d_mean, humidity_7d_mean, rain_7d_sum, temp_range, delta_temp, wind_u, wind_v
   - context: area_rai_in_season, area_rai_off_season, latitude, longitude
3. เพิ่ม column `spike_label` = (bph_raw > spike_threshold).astype(int) สำหรับ Stage 1
4. สร้าง sliding window sequences (W=30, H=7) เหมือน build_sequences.py
5. scale features ด้วย MinMaxScaler fit บน train set เท่านั้น
6. บันทึก NPZ:
   - keys สำหรับ regression: X_train, y_train (log1p), X_val, y_val, X_test, y_test
   - keys สำหรับ classifier: y_spike_train, y_spike_val, y_spike_test (binary 0/1)
   - keys เพิ่มเติม: y_date_*, station_*, meta
7. บันทึก scaler: fold{N}/scaler_minmax.joblib
8. บันทึก fold_info.json: {fold, train_start, train_end, val_start, val_end, test_start, test_end, spike_threshold, n_spike_train, n_spike_test}

**Output structure:**
```
results/walk_forward/
├── fold2/
│   ├── sequences_wf.npz
│   ├── scaler_minmax.joblib
│   └── fold_info.json
├── fold3/
│   └── ...
└── fold4/
    └── ...
```

**Example run:**
```bash
python scripts/build_walk_forward_folds.py \
  --input_csv results/out_quality_gate/cleaned_raw.csv \
  --out_dir results/walk_forward \
  --folds 2,3,4 \
  --min_train_years 2 \
  --window 30 --horizon 7 --roll_days 7 \
  --feature_set context \
  --spike_threshold 100
```

**Validation ที่ต้องผ่านก่อน exit:**
- assert X_train.shape[2] == 18  (context features)
- assert set(np.unique(y_spike_train)).issubset({0, 1})
- assert no NaN/Inf ใน X_*, y_*
- print spike ratio: n_spike / n_total ต้องอยู่ระหว่าง 0.05–0.40
- ถ้า spike ratio < 0.05 ใน fold ใด → print WARNING แต่ไม่ exit

---

## TASK 2: สร้าง train_cnn_lstm_classifier.py

สร้างไฟล์ `scripts/train_cnn_lstm_classifier.py` โดย copy จาก `train_cnn_lstm.py` แล้วดัดแปลง:

**สิ่งที่เปลี่ยนจาก train_cnn_lstm.py:**

1. **Model head:** เปลี่ยนจาก `Dense(1)` เป็น `Dense(1, activation='sigmoid')` 
2. **Loss:** เปลี่ยนจาก `Huber()` เป็ `binary_crossentropy`
3. **Target:** ใช้ `y_spike_train`, `y_spike_val`, `y_spike_test` จาก NPZ (ไม่ใช่ y_train)
4. **Class weight:** เพิ่ม `--class_weight` flag
   - `auto`: คำนวณจาก `sklearn.utils.class_weight.compute_class_weight`
   - `none`: ไม่ใช้ class weight
5. **Metrics:** วัด F1, Precision, Recall บน test set ที่ threshold ต่างๆ
   - ทดสอบ threshold: 0.3, 0.4, 0.5 แล้วเลือก threshold ที่ให้ Recall ≥ 0.80 และ Precision สูงสุด
   - บันทึก best_threshold ลงใน metrics_stage1.json
6. **Output files เพิ่มเติม:**
   - `metrics_stage1.json`: {recall, precision, f1, auc, best_threshold, confusion_matrix}
   - `figures/roc_curve.png`
   - `figures/precision_recall_curve.png`
   - `figures/confusion_matrix.png`
   - `model_stage1_best.keras`
   - `predictions_test_stage1.csv`: {date, station_id, y_true_spike, y_pred_prob, y_pred_spike}

**Args เพิ่ม/เปลี่ยน:**
```
--npz         (เหมือนเดิม)
--out_dir     (เหมือนเดิม)
--epochs      150    (default)
--batch_size  128    (default)
--lr          0.0005 (default)
--patience    25     (default)
--conv_filters 64
--kernel_size  5
--lstm_units   64
--dropout      0.25
--clipnorm     1.0
--class_weight auto  (new)
--recall_target 0.80 (new — threshold ที่จะ select)
--seed        42
```

**Example run:**
```bash
python scripts/train_cnn_lstm_classifier.py \
  --npz results/walk_forward/fold4/sequences_wf.npz \
  --out_dir results/walk_forward/fold4/stage1 \
  --epochs 150 --batch_size 128 --lr 0.0005 --patience 25 \
  --conv_filters 64 --kernel_size 5 --lstm_units 64 --dropout 0.25 \
  --class_weight auto --recall_target 0.80
```

**Expected output หลัง run:**
```
[Stage1] Test Recall=0.804 Precision=0.512 F1=0.626 AUC=0.871
[Stage1] Best threshold=0.35 (Recall≥0.80)
[OK] Saved: results/walk_forward/fold4/stage1/model_stage1_best.keras
[OK] Saved: results/walk_forward/fold4/stage1/metrics_stage1.json
```

---

## TASK 3: แก้ไข train_cnn_lstm.py — เพิ่ม --spike_only flag

แก้ไข `scripts/train_cnn_lstm.py` เพิ่ม 2 arguments:
- `--spike_only` (store_true): ถ้า True ให้กรองเฉพาะ samples ที่ y_spike=1 ก่อนเทรน
- `--spike_threshold` (float, default=100): ใช้สร้าง spike mask ถ้า y_spike ไม่มีใน NPZ

**Logic เพิ่มใน main():**
```python
if args.spike_only:
    # โหลด y_spike จาก NPZ ถ้ามี
    if "y_spike_train" in data.files:
        mask_tr = seq.y_spike_train == 1
        mask_va = seq.y_spike_val == 1
        mask_te = seq.y_spike_test == 1
    else:
        # fallback: ใช้ inverse_log1p แล้วตรวจ threshold
        mask_tr = inverse_log1p(seq.y_train) > args.spike_threshold
        mask_va = inverse_log1p(seq.y_val)   > args.spike_threshold
        mask_te = inverse_log1p(seq.y_test)  > args.spike_threshold
    
    Xtr, ytr = Xtr[mask_tr], ytr[mask_tr]
    Xva, yva = Xva[mask_va], yva[mask_va]
    Xte, yte = Xte[mask_te], yte[mask_te]
    
    print(f"[spike_only] train={mask_tr.sum()} val={mask_va.sum()} test={mask_te.sum()}")
    
    if mask_tr.sum() < 50:
        raise RuntimeError(f"spike_only: train samples too few ({mask_tr.sum()}). Lower --spike_threshold.")
```

**Example run (Stage 2):**
```bash
python scripts/train_cnn_lstm.py \
  --npz results/walk_forward/fold4/sequences_wf.npz \
  --out_dir results/walk_forward/fold4/stage2 \
  --spike_only \
  --epochs 150 --batch_size 64 --lr 0.0005 --patience 25 \
  --conv_filters 64 --kernel_size 5 --lstm_units 64 --dropout 0.25
```

---

## TASK 4: สร้าง run_walk_forward.sh

สร้างไฟล์ `scripts/run_walk_forward.sh` ที่รัน Task 1–3 ทั้งหมดต่อเนื่องอัตโนมัติ:

```bash
#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="/home/ai-station/my_project/.tf251p310/bin/python"
ROOT="results/walk_forward"
CLEANED="results/out_quality_gate/cleaned_raw.csv"

# Step 1: Build all folds
echo "=== BUILD WALK-FORWARD FOLDS ==="
$PYTHON "$SCRIPT_DIR/build_walk_forward_folds.py" \
  --input_csv "$CLEANED" \
  --out_dir "$ROOT" \
  --folds 2,3,4 --min_train_years 2 \
  --window 30 --horizon 7 --roll_days 7 \
  --feature_set context --spike_threshold 100

# Step 2 & 3: Train Stage1 + Stage2 per fold
for FOLD in 2 3 4; do
  NPZ="$ROOT/fold${FOLD}/sequences_wf.npz"
  echo "=== FOLD ${FOLD}: Stage 1 (Classifier) ==="
  $PYTHON "$SCRIPT_DIR/train_cnn_lstm_classifier.py" \
    --npz "$NPZ" \
    --out_dir "$ROOT/fold${FOLD}/stage1" \
    --epochs 150 --batch_size 128 --lr 0.0005 --patience 25 \
    --conv_filters 64 --kernel_size 5 --lstm_units 64 --dropout 0.25 \
    --class_weight auto --recall_target 0.80 \
    2>&1 | tee "$ROOT/fold${FOLD}/stage1_train.log"

  echo "=== FOLD ${FOLD}: Stage 2 (Regressor, spike_only) ==="
  $PYTHON "$SCRIPT_DIR/train_cnn_lstm.py" \
    --npz "$NPZ" \
    --out_dir "$ROOT/fold${FOLD}/stage2" \
    --spike_only \
    --epochs 150 --batch_size 64 --lr 0.0005 --patience 25 \
    --conv_filters 64 --kernel_size 5 --lstm_units 64 --dropout 0.25 \
    2>&1 | tee "$ROOT/fold${FOLD}/stage2_train.log"
done

echo "=== ALL FOLDS DONE ==="
```

---

## TASK 5: สร้าง summarize_walk_forward.py

สร้างไฟล์ `scripts/summarize_walk_forward.py`:

**Input:**
- `--root results/walk_forward`
- `--folds 2,3,4`
- `--out results/walk_forward/cv_summary.csv`

**สิ่งที่ต้องรวบรวมจากแต่ละ fold:**

จาก `fold{N}/stage1/metrics_stage1.json`:
- recall, precision, f1, auc, best_threshold

จาก `fold{N}/stage2/metrics.json`:
- test_log1p: mae, rmse, r2
- test_raw: mae, rmse, r2

**Output 1: cv_summary.csv**
```
fold, recall_s1, precision_s1, f1_s1, auc_s1, best_threshold,
      r2_log1p_s2, rmse_log1p_s2, r2_raw_s2, rmse_raw_s2
```

**Output 2: cv_report.md** — manuscript-ready table:
```markdown
## Walk-Forward Cross-Validation Results

| Metric          | Fold 2 | Fold 3 | Fold 4 | Mean ± SD |
|-----------------|--------|--------|--------|-----------|
| Stage1 Recall   | x.xxx  | x.xxx  | x.xxx  | x.xxx±x.xxx |
| Stage1 F1       | ...    | ...    | ...    | ...       |
| Stage2 R² (log) | ...    | ...    | ...    | ...       |
| Stage2 RMSE_raw | ...    | ...    | ...    | ...       |

**Note:** Fold 4 test period (2019) corresponds to the original single-split evaluation.
Best threshold selected to achieve Recall ≥ 0.80 per fold.
```

**Output 3: figures/cv_r2_barplot.png** — bar chart เปรียบเทียบ R² ข้าม 3 folds

**Example run:**
```bash
python scripts/summarize_walk_forward.py \
  --root results/walk_forward \
  --folds 2,3,4 \
  --out results/walk_forward/cv_summary.csv
```

---

## EXECUTION ORDER (รันตามลำดับนี้)

```bash
# 0. activate env
source /home/ai-station/my_project/.tf251p310/bin/activate
cd BPH_Research

# 1. ตรวจ cleaned_raw.csv มีอยู่แล้ว
ls results/out_quality_gate/cleaned_raw.csv

# 2. รัน pipeline ทั้งหมด (background-safe)
chmod +x scripts/run_walk_forward.sh
nohup ./scripts/run_walk_forward.sh > walk_forward_main.log 2>&1 &
tail -f walk_forward_main.log

# 3. สรุปผลหลัง training เสร็จ
python scripts/summarize_walk_forward.py \
  --root results/walk_forward \
  --folds 2,3,4 \
  --out results/walk_forward/cv_summary.csv

# 4. ตรวจผลสรุป
cat results/walk_forward/cv_report.md
```

---

## SUCCESS CRITERIA (เป้าหมายที่ต้องผ่าน)

| Metric | เป้าหมาย | เหตุผล |
|--------|----------|--------|
| Stage1 Recall (mean) | ≥ 0.80 | ห้าม miss outbreak จริง |
| Stage1 Recall (min fold) | ≥ 0.70 | stable ข้าม fold |
| Stage1 F1 (mean) | ≥ 0.55 | สมดุล precision-recall |
| Stage2 R² log1p (mean) | ≥ 0.45 | เทียบกับ baseline 0.484 |
| Stage2 r²_raw (mean) | > 0.00 | ดีขึ้นจาก 0.007 |
| CV variance R² | SD < 0.08 | stable ข้ามปี |

ถ้าผลไม่ผ่าน Stage1 Recall ใน fold ใด:
1. ลด threshold จาก 0.4 → 0.3 แล้วประเมินใหม่
2. ถ้ายังไม่ผ่าน → เพิ่ม class_weight spike:non-spike ratio ขึ้นเป็น 3:1
3. บันทึกเหตุผลใน cv_report.md

---

## NOTES สำหรับ Claude Code

- รันทุก script จาก `BPH_Research/` เสมอ (ไม่ใช่ scripts/)
- ใช้ Python path เต็ม: `/home/ai-station/my_project/.tf251p310/bin/python`
- train_utils.py import pattern: `from train_utils import ...` (ไม่ใช่ scripts.train_utils)
- NPZ key สำหรับ spike: `y_spike_train`, `y_spike_val`, `y_spike_test` (float32: 0.0 หรือ 1.0)
- อย่าแก้ train_utils.py, build_sequences.py, quality_gate.py — ใช้เป็น dependency เท่านั้น
- ถ้า GPU OOM: ลด batch_size เป็น 64 หรือ 32 ก่อน
- ถ้า fold 2 มี train samples น้อยเกินไป → ใช้ preset FAST (epochs=50, patience=10) แทน BALANCED
