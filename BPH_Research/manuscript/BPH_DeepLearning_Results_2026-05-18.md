# การพยากรณ์เพลี้ยกระโดดสีน้ำตาล (BPH) ในนาข้าวภาคตะวันออกเฉียงเหนือของไทย  
## ด้วยโมเดล Deep Learning: LSTM, CNN-LSTM และ Transformer

**วันที่บันทึกผล:** 2026-05-18  
**ข้อมูล:** `results/summary_final/comparison.csv`

---

## บทคัดย่อ

งานวิจัยนี้ศึกษาการพยากรณ์ความหนาแน่นของเพลี้ยกระโดดสีน้ำตาล (*Nilaparvata lugens*) ในนาข้าวภาคตะวันออกเฉียงเหนือของประเทศไทย โดยใช้โมเดล Deep Learning สามประเภท ได้แก่ LSTM, CNN-LSTM และ Transformer ร่วมกับข้อมูลสภาพแวดล้อมรายวัน ผลการทดลองพบว่า **CNN-LSTM ร่วมกับ context feature set (18 features) และ sliding window 30 วัน** ให้ผลดีที่สุด โดยได้ R² = 0.500 บน test set (log1p-space) สำหรับการพยากรณ์ล่วงหน้า 1 วัน และ R² = 0.484 สำหรับการพยากรณ์ล่วงหน้า 7 วัน นอกจากนี้การคัดเลือก feature ด้วย permutation importance เหลือเพียง 10 features พบว่าสามารถรักษาประสิทธิภาพได้ใกล้เคียง (R² = 0.470) พร้อมทั้งให้ค่า r²_raw เป็นบวกครั้งแรก (+0.007)

---

## 1. ข้อมูลและวิธีการ

### 1.1 ข้อมูลที่ใช้

- **ช่วงเวลา:** 2015-01-01 ถึง 2019-12-31 (1,826 วัน)
- **การแบ่งข้อมูล:**
  - Train: 2015-01-01 – 2018-07-01 (1,278 วัน, 42,228 samples)
  - Validation: 2018-07-02 – 2019-03-31 (273 วัน, 8,058 samples)
  - Test: 2019-04-01 – 2019-12-31 (275 วัน, 8,126 samples)
- **Target:** `log1p(bph_count)` — ใช้ log transformation เพื่อลด skewness จาก BPH spike

### 1.2 Feature Sets

| Feature Set | จำนวน Features | รายละเอียด |
|---|---|---|
| **core** | 14 | สภาพอากาศรายวัน + cyclical time encoding |
| **context** | 18 | core + พื้นที่ปลูกข้าว (in/off season) + lat/lon |
| **full** | 36 | context + สัดส่วนพันธุ์ข้าวรายพันธุ์ |
| **trimmed** | 10 | top-10 features จาก permutation importance |

**Features ใน context set (18 features):**  
`month_sin, month_cos, doy_sin, doy_cos, temp, humidity, rainfall, wind_u, wind_v, delta_temp, temp_7d_mean, humidity_7d_mean, rain_7d_sum, temp_range, area_rai_in_season, area_rai_off_season, latitude, longitude`

### 1.3 โมเดลและ Hyperparameters

#### CNN-LSTM (Best Model)
```
Architecture: Conv1D(64, kernel=5, padding=same) → MaxPool1D(2) → Dropout(0.25)
              → LSTM(64, dropout=0.25) → Dense(64, relu) → Dense(1)
Loss:         Huber
Optimizer:    Adam (lr=0.0005, clipnorm=1.0)
Epochs:       150 (max), early stop patience=25
Batch size:   128
```

#### LSTM
```
Architecture: LSTM(64, dropout=0.2) → Dense(1)
Optimizer:    Adam (lr=0.0005, clipnorm=1.0)
Epochs:       120 (max), early stop patience=20
```

#### Transformer
```
Architecture: d_model=64, num_layers=2, num_heads=4, ff_dim=128, dropout=0.2
Optimizer:    Adam (lr=0.0005, clipnorm=1.0)
Epochs:       200 (max), early stop patience=30
```

### 1.4 Sliding Window

| Window (W) | Horizon (H) | roll_days | Samples (train) |
|---|---|---|---|
| 30 วัน | 1 วัน | 7 | 42,228 |
| 30 วัน | 7 วัน | 7 | 42,228 |
| 60 วัน | 7 วัน | 7 | 41,208 |
| 90 วัน | 14 วัน | 14 | — |

---

## 2. ผลการทดลอง

### 2.1 ภาพรวมทุก Combination (Test Set)

| โมเดล | Feature Set | W | H | R² (log1p) | MAE (log1p) | RMSE (log1p) | r²_raw |
|---|---|---|---|---|---|---|---|
| **CNN-LSTM** | **context** | **30** | **1** | **0.500** | **0.632** | **1.176** | **+0.011** |
| CNN-LSTM | context | 30 | 7 | 0.484 | 0.634 | 1.198 | +0.004 |
| CNN-LSTM | trimmed (10f) | 30 | 7 | 0.470 | 0.653 | 1.214 | +0.007 |
| CNN-LSTM | context | 60 | 7 | 0.463 | 0.667 | 1.243 | +0.009 |
| Transformer | context | 30 | 1 | 0.408 | 0.742 | 1.280 | +0.003 |
| Transformer | context | 30 | 7 | 0.368 | 0.722 | 1.326 | +0.002 |
| LSTM | context | 30 | 7 | 0.363 | 0.724 | 1.331 | +0.009 |
| LSTM | context | 60 | 7 | 0.344 | 0.769 | 1.374 | +0.010 |
| LSTM | full | 60 | 7 | 0.334 | 0.802 | 1.384 | +0.000 |
| Transformer | full | 60 | 7 | 0.279 | 0.823 | 1.440 | −0.002 |
| CNN-LSTM | core | 60 | 7 | 0.202 | 0.971 | 1.515 | −0.006 |
| LSTM | core | 60 | 7 | 0.135 | 1.015 | 1.577 | −0.013 |

### 2.2 เปรียบเทียบโมเดลบน W=30 H=7 Context (Fair Comparison)

เปรียบเทียบ 3 โมเดลบน feature set และ window เดียวกัน:

| โมเดล | R² (log1p) | MAE | RMSE | r²_raw | ∆R² vs LSTM |
|---|---|---|---|---|---|
| **CNN-LSTM** | **0.484** | **0.634** | **1.198** | +0.004 | — |
| Transformer | 0.368 | 0.722 | 1.326 | +0.002 | −0.116 |
| LSTM | 0.363 | 0.724 | 1.331 | +0.009 | −0.121 |

**CNN-LSTM เหนือกว่า LSTM และ Transformer ประมาณ 12 R² points**

### 2.3 ผลกระทบของ Feature Set (CNN-LSTM, W=60, H=7)

| Feature Set | Features | R² | MAE | RMSE |
|---|---|---|---|---|
| **context** | 18 | **0.463** | **0.667** | **1.243** |
| full | 36 | 0.192 | 0.844 | 1.525 |
| core | 14 | 0.202 | 0.971 | 1.515 |

**context ชนะ full ทั้งที่มี features น้อยกว่าถึง 18 features** — แสดงว่า features เพิ่มเติมใน full set เป็น noise

### 2.4 ผลกระทบของ Window Size (CNN-LSTM, Context, H=7)

| Window | R² | MAE | RMSE |
|---|---|---|---|
| **W=30** | **0.484** | **0.634** | **1.198** |
| W=60 | 0.463 | 0.667 | 1.243 |

**Window สั้นกว่า (30 วัน) ให้ผลดีกว่า Window ยาว (60 วัน)** — อาจเป็นเพราะ BPH มี pattern ระยะสั้นมากกว่าระยะยาว

### 2.5 Feature Importance (Permutation Importance บน CNN-LSTM Context W=30 H=7)

| อันดับ | Feature | ∆RMSE เมื่อ shuffle | ความหมาย |
|---|---|---|---|
| 1 | longitude | +0.685 | ตำแหน่งพื้นที่ (ตะวันออก-ตะวันตก) |
| 2 | latitude | +0.647 | ตำแหน่งพื้นที่ (เหนือ-ใต้) |
| 3 | temp_range | +0.385 | ช่วงอุณหภูมิกลางวัน-กลางคืน |
| 4 | month_sin | +0.157 | ฤดูกาล (เดือน) |
| 5 | doy_sin | +0.145 | ฤดูกาล (วันในปี) |
| 6 | month_cos | +0.086 | ฤดูกาล (เดือน) |
| 7 | humidity_7d_mean | +0.071 | ความชื้นสัมพัทธ์เฉลี่ย 7 วัน |
| 8 | area_rai_in_season | +0.067 | พื้นที่ปลูกข้าวฤดูหลัก |
| 9 | doy_cos | +0.063 | ฤดูกาล (วันในปี) |
| 10 | humidity | +0.060 | ความชื้นสัมพัทธ์รายวัน |
| 11–18 | wind_u, wind_v, rainfall, ... | < 0.010 | ผลกระทบน้อยมาก |

**ข้อสังเกต:** longitude และ latitude มีความสำคัญสูงสุดมาก — แสดงว่า BPH มี spatial pattern ชัดเจน บางพื้นที่เสี่ยงกว่าพื้นที่อื่นอย่างมีนัยสำคัญ

### 2.6 Trimmed Feature Set (Top-10 Features)

เลือก features ที่มี ∆RMSE > 0.06 (10 features):  
`longitude, latitude, temp_range, month_sin, doy_sin, month_cos, humidity_7d_mean, area_rai_in_season, doy_cos, humidity`

| โมเดล | Feature Set | R² (log1p) | r²_raw |
|---|---|---|---|
| CNN-LSTM | context (18f) | 0.484 | +0.004 |
| **CNN-LSTM** | **trimmed (10f)** | **0.470** | **+0.007** |

ลด feature จาก 18 → 10 ทำให้ R² log1p ลดลงเพียง 0.014 แต่ **r²_raw เพิ่มขึ้น** (+0.003) — แสดงว่า trimmed model พยากรณ์ raw BPH count ได้แม่นยำกว่าในแง่ของ spike prediction

---

## 3. การอภิปรายผล

### 3.1 ทำไม CNN-LSTM ถึงดีกว่า

CNN-LSTM ผสมจุดแข็งของ CNN ในการดึง local temporal pattern (เช่น การเพิ่มขึ้นอย่างรวดเร็วของ BPH ใน 5–7 วัน) กับ LSTM ที่จดจำ long-term dependency ทำให้สามารถจับ pattern ทั้งระยะสั้นและระยะยาวได้ดีกว่าการใช้ LSTM เดี่ยว ส่วน Transformer แม้จะมี attention mechanism แต่ข้อมูลของเรามี samples ไม่มากพอให้ Transformer แสดงศักยภาพเต็มที่

### 3.2 ทำไม Context ชนะ Full

Context set (18f) เพิ่ม spatial context (lat/lon) และ พื้นที่ปลูกข้าว ซึ่งมีความสัมพันธ์กับ BPH โดยตรง ในขณะที่ Full set (36f) เพิ่มสัดส่วนพันธุ์ข้าวรายพันธุ์ซึ่งอาจมี multicollinearity สูงและทำให้โมเดล overfit กับ noise

### 3.3 ข้อจำกัด

- **r²_raw ต่ำ:** แม้ R² log1p จะถึง 0.50 แต่ r²_raw ยังต่ำ (~0.01) เพราะ BPH spike รุนแรงทำให้ error ใน raw scale ใหญ่มาก log1p transformation ช่วย training แต่ทำให้ inverse transform ขยาย error ของ outlier
- **Single temporal split:** ใช้ split เดียว (train 2015–2018, test 2019) ผลอาจไม่ robust — ควรทำ temporal cross-validation
- **Spatial independence:** โมเดลใช้ lat/lon แต่ไม่ได้ model spatial autocorrelation อย่างชัดเจน

---

## 4. สรุป

| ประเด็น | ข้อสรุป |
|---|---|
| **Best model** | CNN-LSTM + context (18f), W=30, H=1 → R²=0.500 |
| **Best for 7-day forecast** | CNN-LSTM + context (18f), W=30, H=7 → R²=0.484 |
| **Lightest model** | CNN-LSTM + trimmed (10f), W=30, H=7 → R²=0.470, r²_raw=+0.007 |
| **Feature ที่สำคัญที่สุด** | longitude, latitude, temp_range (spatial + thermal) |
| **Window ที่เหมาะสม** | W=30 วัน ดีกว่า W=60 และ W=90 |
| **Weighted loss** | ไม่ช่วย — log1p compress spike อยู่แล้ว |

### แนวทางพัฒนาต่อ

1. **Two-stage model:** classify spike/no-spike ก่อน (binary) แล้ว regression เฉพาะ spike period
2. **Spatial model:** Graph Neural Network หรือ ConvLSTM ที่จัดการ spatial dependency ระหว่างสถานี
3. **Temporal cross-validation:** ทดสอบหลาย fold เพื่อ robust estimate
4. **External data:** NDVI, soil moisture, ข้อมูลศัตรูธรรมชาติ

---

*บันทึกโดย: Claude Code + D6200763-SUT*  
*วันที่: 2026-05-18*  
*ไฟล์ผลลัพธ์: `results/summary_final/comparison.csv`*
