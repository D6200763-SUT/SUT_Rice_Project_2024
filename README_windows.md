# README (Windows cmd/PowerShell) — Time-Series Training (LSTM / CNN-LSTM / Transformer)

> ใช้ `^` สำหรับ Windows cmd / PowerShell เพื่อขึ้นบรรทัดใหม่  
> โฟลเดอร์ตัวอย่าง: `D:\AI_Projects\SUT_Rice_Project_2024\`

---

## 0) โครงสร้างโฟลเดอร์ที่แนะนำ
```text
SUT_Rice_Project_2024\
  code\
    03_build_feature_sets_and_sequences_v2_nan_safe.py
    11_train_lstm_compat.py
    12_train_cnn_lstm_compat.py
    13_train_transformer_compat.py
    inspect_npz_for_nan.py
    summarize_runs.py
  out_quality_gate\
    cleaned_raw.csv
  out_feature_sets\
    core\sequences_window30_h1.npz
    context\...
    full\...
  out_train\
```

---

# 1) ปรับ `horizon` แล้วควรตั้ง `window`
| Horizon (H) | Window (W) แนะนำ |
|---:|---:|
| 1 | 30–45 |
| 7 | 45–60 |
| 14 | 60–90 |

---

# 2) Rolling features ให้เข้ากับ horizon
- **H=1** → ใช้ `roll_days=7`
- **H=7** → `roll_days=7` (เสริม 14 วันถ้าจะทำ multi-scale)
- **H=14** → `roll_days=14` (แนะนำ 7+14 ถ้าจะทำ multi-scale)

---

# 3) โมเดลไหนเหมาะกับ horizon แบบไหน
- **H=1**: LSTM / CNN-LSTM
- **H=7**: CNN-LSTM (เด่นกับ pattern ระยะสั้น)
- **H=14**: Transformer (คุ้มขึ้นเมื่อ W=60–90)

---

# 4) สร้าง NPZ (Sliding window) — แนะนำใช้เวอร์ชัน nan-safe
```bat
python code\03_build_feature_sets_and_sequences_v2_nan_safe.py ^
  --input_csv out_quality_gate\cleaned_raw.csv ^
  --out_dir out_feature_sets_h7_w60 ^
  --window 60 --horizon 7 --roll_days 7 ^
  --all_sets ^
  --require_consecutive
```

---

# 5) ตรวจ NaN/Inf ในไฟล์ NPZ
```bat
python code\inspect_npz_for_nan.py ^
  --npz out_feature_sets\core\sequences_window30_h1.npz
```

---

# 6) เทรนโมเดล (H=1, W=30) — Core set
## 6.1 LSTM
```bat
python code\11_train_lstm_compat.py ^
  --npz out_feature_sets\core\sequences_window30_h1.npz ^
  --out_dir out_train\lstm_core ^
  --epochs 120 --batch_size 128 --lr 0.0005 --patience 20 ^
  --clipnorm 1.0 --lstm_units 64 --dropout 0.2
```

## 6.2 CNN-LSTM
```bat
python code\12_train_cnn_lstm_compat.py ^
  --npz out_feature_sets\core\sequences_window30_h1.npz ^
  --out_dir out_train\cnn_lstm_core ^
  --epochs 150 --batch_size 128 --lr 0.0005 --patience 25 ^
  --clipnorm 1.0 --conv_filters 64 --kernel_size 5 ^
  --lstm_units 64 --dropout 0.25
```

## 6.3 Transformer
```bat
python code\13_train_transformer_compat.py ^
  --npz out_feature_sets\core\sequences_window30_h1.npz ^
  --out_dir out_train\transformer_core ^
  --epochs 200 --batch_size 128 --lr 0.0005 --patience 30 ^
  --clipnorm 1.0 --d_model 64 --num_layers 2 --num_heads 4 ^
  --ff_dim 128 --dropout 0.2
```

---

# 7) รวมผลทุกการเทรนเป็นตาราง
```bat
python code\summarize_runs.py ^
  --root out_train ^
  --out out_train\summary
```

Outputs:
- `out_train\summary\comparison.csv`
- `out_train\summary\comparison.md`

---

# 8) Presets 3 ระดับ (เร็ว / สมดุล / เน้นคุณภาพ)
## FAST (ลองไว)
- `epochs=50`, `patience=10`, `batch_size=256`, `lr=0.001`

## BALANCED (แนะนำ)
- `epochs=120–150`, `patience=20–30`, `batch_size=128`, `lr=0.0005`, `clipnorm=1.0`

## QUALITY (เน้นสรุปลงรายงาน)
- `epochs=200–260`, `patience=35–45`, `batch_size=128 (หรือ 64)`, `lr=0.0003–0.0005`

---

## Tips: ถ้า RAM/VRAM ตึง
- ลด `--batch_size 128 → 64`
- ลด `--d_model` หรือ `--num_layers` (Transformer)
- ลด `--conv_filters` (CNN-LSTM)
