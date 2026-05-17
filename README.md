# SUT Rice Project 2024

งานวิจัยการพยากรณ์ **เพลี้ยกระโดดสีน้ำตาล (BPH)**   
ด้วยโมเดล Deep Learning: LSTM, CNN-LSTM, Transformer

---

## โครงสร้างโปรเจกต์

```
SUT_Rice_Project_2024/
└── BPH_Research/
    ├── scripts/       ← Python scripts + run_train_auto.sh
    ├── data/          ← ข้อมูล CSV ดิบ (BPH + สภาพอากาศ + ข้าว)
    ├── results/       ← output ทั้งหมด (quality gate, feature sets, training)
    ├── manuscript/    ← บทความ .docx
    ├── docs/          ← คู่มือเพิ่มเติม
    └── CLAUDE.md      ← คำสั่งและรายละเอียด pipeline ทั้งหมด
```

---

## เริ่มต้นใช้งาน

```bash
cd BPH_Research
```

ดูคำสั่ง pipeline ทั้งหมดได้ที่ **[BPH_Research/CLAUDE.md](BPH_Research/CLAUDE.md)**

---

## Pipeline โดยย่อ

| ขั้นตอน | คำสั่ง |
|---|---|
| 1. ตรวจสอบข้อมูล | `python scripts/quality_gate.py ...` |
| 2. สร้าง sequences | `python scripts/03_build_feature_sets_and_sequences_v2_nan_safe.py ...` |
| 3. เทรนโมเดล | `./scripts/run_train_auto.sh results/out_feature_sets_.../... results/out_train_...` |
| 4. สรุปผล | `python scripts/summarize_runs.py ...` |

> รันทุกคำสั่งจาก `BPH_Research/` เสมอ
