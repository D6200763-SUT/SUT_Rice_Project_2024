# Layer 2 — Agent Command Reference

ไฟล์หลัก: `scripts/agent_next_step.py`  
Loop runner: `scripts/agent_loop.sh`  
Dashboard: `scripts/agent_status.py`  
State file: `results/agent_state.json`  
รันทุกคำสั่งจาก `BPH_Research/`

---

## 1. คำสั่งพื้นฐาน

### ดูการตัดสินใจถัดไป
```bash
python scripts/agent_next_step.py
```
Agent อ่าน state แล้วพิมพ์ว่าต้องรันคำสั่งใดต่อ พร้อม experiment ที่เลือก

### ใช้ state file อื่น
```bash
python scripts/agent_next_step.py --state results/my_state.json
```

### ทดสอบโดยไม่เปลี่ยน state
```bash
python scripts/agent_next_step.py --dry-run
```

### ดู state ทั้งหมด (JSON)
```bash
python scripts/agent_next_step.py --show-state
```

---

## 2. คำสั่ง Mark (บอก Agent ว่าขั้นนี้เสร็จแล้ว)

รันหลังจากแต่ละขั้นตอนสำเร็จ เพื่อให้ Agent ก้าวไปขั้นถัดไป

| คำสั่ง | ใช้หลังจาก |
|---|---|
| `--mark-quality-gate-done` | รัน `quality_gate.py` สำเร็จ |
| `--mark-build-done 30_1` | รัน `build_sequences.py` W=30, H=1 สำเร็จ |
| `--mark-inspect-done w30_h1_context` | รัน `inspect_npz_for_nan.py` ไม่พบ NaN |
| `--mark-training-done w30_h1_context` | รัน `run_train_auto.sh` เสร็จ |
| `--mark-summarize-done w30_h1_context` | รัน `summarize_runs.py` เสร็จ (อ่าน metrics อัตโนมัติ) |

```bash
# ตัวอย่างลำดับการใช้งานจริง
python scripts/agent_next_step.py --mark-quality-gate-done
python scripts/agent_next_step.py --mark-build-done 30_1
python scripts/agent_next_step.py --mark-inspect-done w30_h1_context
python scripts/agent_next_step.py --mark-training-done w30_h1_context
python scripts/agent_next_step.py --mark-summarize-done w30_h1_context
```

> `--mark-summarize-done` จะอ่าน `metrics.json` อัตโนมัติ และอัปเดต `current_best` ถ้าผลดีขึ้น

---

## 3. คำสั่ง Reset (เมื่อมีข้อมูลใหม่หรืออยากเริ่มใหม่)

### Reset เพื่อรับข้อมูลชุดใหม่ (แนะนำ)
```bash
python scripts/agent_next_step.py --reset-data
```
สิ่งที่เกิดขึ้น:
- ลบ `quality_gate_done` และ `build_w*` flags ทั้งหมด
- Reset `inspect_done` ของทุก experiment (NPZ ต้อง rebuild)
- **เก็บ `baseline_score` ไว้** เพื่อเปรียบเทียบกับผลใหม่
- คง `current_best` ไว้เป็น reference

### Reset พร้อมระบุ CSV ใหม่
```bash
python scripts/agent_next_step.py --reset-data-csv data/env_2020_2024.csv
```
เหมือน `--reset-data` แต่บันทึก path ของ CSV ใหม่ใน state ด้วย

### Reset ทุกอย่างกลับเป็น fresh (เริ่มใหม่หมดเลย)
```bash
python scripts/agent_next_step.py --reset-all
```
ล้าง experiments, flags และ scores ทั้งหมด — ใช้เมื่ออยากเริ่มต้นใหม่ทั้งหมด

---

## 4. Dashboard

```bash
python scripts/agent_status.py
```
แสดง progress bar, สถานะทุก experiment, log ล่าสุด

```bash
python scripts/agent_status.py --state results/my_state.json
```

ตัวอย่าง output:
```
╔══════════════════════════════════════════════════════════╗
║          🤖  BPH Agent Status Dashboard                  ║
╚══════════════════════════════════════════════════════════╝
  Status  : running
  Goal    : r2_log1p ≥ 0.5
  Progress: [██████████████░░░░░░░░░░░░░░░░] 47%
  Best    : w30_h1_context → 0.4700

  Experiments:
  ✅ w30_h1_context    done      0.4700 ⭐
  🔄 w30_h7_context    training_done  —
  ⬜ w60_h7_context    pending        —
```

---

## 5. Auto Loop

### รันอัตโนมัติจนถึงเป้าหมาย
```bash
./scripts/agent_loop.sh
```

### จำกัดจำนวน iteration (แนะนำสำหรับทดสอบครั้งแรก)
```bash
./scripts/agent_loop.sh --max-iter 3
```

### ทดสอบโดยไม่รันจริง
```bash
./scripts/agent_loop.sh --dry-run
```

### ใช้ state file อื่น
```bash
./scripts/agent_loop.sh --state results/my_state.json
```

### รัน background (SSH-safe)
```bash
nohup ./scripts/agent_loop.sh > results/agent_loop.log 2>&1 &
tail -f results/agent_loop.log   # ติดตามความคืบหน้า
```

---

## 6. Flow การทำงานทั้งหมด

```
เริ่มต้น (fresh state)
    ↓
quality_gate.py  →  --mark-quality-gate-done
    ↓
build_sequences.py  →  --mark-build-done {W}_{H}
    ↓
inspect_npz_for_nan.py  →  --mark-inspect-done {EXP_ID}
    ↓
run_train_auto.sh  →  --mark-training-done {EXP_ID}
    ↓
summarize_runs.py  →  --mark-summarize-done {EXP_ID}
    ↓
[Agent เปรียบเทียบ score กับ goal]
    ↓
    ├─ ยังไม่ถึงเป้า → เลือก experiment ถัดไปตาม priority → วนซ้ำ
    └─ ถึงเป้าแล้ว  → CONVERGED ✅
```

---

## 7. Search Space (ลำดับ experiment ที่ Agent จะรัน)

| ลำดับ | Experiment ID | W | H | Feature Set | Preset |
|---|---|---|---|---|---|
| 1 | w30_h1_context | 30 | 1 | context | balanced |
| 2 | w30_h7_context | 30 | 7 | context | balanced |
| 3 | w60_h7_context | 60 | 7 | context | balanced |
| 4 | w60_h7_full | 60 | 7 | full | balanced |
| 5 | w90_h14_context | 90 | 14 | context | balanced |
| 6 | w90_h14_full | 90 | 14 | full | balanced |
| 7 | w30_h7_context_quality | 30 | 7 | context | quality |
| 8 | w60_h7_context_quality | 60 | 7 | context | quality |
| 9 | w30_h1_full | 30 | 1 | full | balanced |
| 10 | w45_h7_context | 45 | 7 | context | balanced |

---

## 8. Training Presets

| Preset | Epochs (CNN) | Patience | Batch Size | Learning Rate |
|---|---|---|---|---|
| fast | 50 | 10 | 256 | 0.001 |
| **balanced** ★ | 150 | 25 | 128 | 0.0005 |
| quality | 260 | 45 | 128 | 0.0003 |

★ = ค่า default ที่แนะนำ

---

## 9. ความแตกต่างระหว่าง Reset

| | `--reset-data` | `--reset-data-csv` | `--reset-all` |
|---|---|---|---|
| ลบ pipeline flags | ✅ | ✅ | ✅ |
| เก็บ experiment scores | ✅ baseline | ✅ baseline | ❌ |
| อัปเดต CSV path | ❌ | ✅ | ❌ |
| ล้าง experiments | ❌ | ❌ | ✅ |
| ใช้เมื่อ | **ข้อมูลใหม่มา** | ข้อมูลใหม่ + ระบุไฟล์ | เริ่มใหม่ทั้งหมด |
