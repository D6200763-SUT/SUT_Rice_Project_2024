# Layer 3 — Agentic Command Reference

ไฟล์หลัก: `scripts/agent_agentic.py`  
State file: `results/agent_state_l3.json`  
รันทุกคำสั่งจาก `BPH_Research/`

---

## 1. คำสั่งพื้นฐาน

### ดูการตัดสินใจถัดไป
```bash
python scripts/agent_agentic.py
```
Agent อ่าน state แล้วพิมพ์ว่าต้องรันคำสั่งใดต่อ พร้อม phase ปัจจุบัน

### ใช้ state file อื่น
```bash
python scripts/agent_agentic.py --state results/my_experiment.json
```

### ทดสอบโดยไม่เปลี่ยน state
```bash
python scripts/agent_agentic.py --dry-run
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
| `--mark-summarize-done w30_h1_context` | รัน `summarize_runs.py` เสร็จ |

```bash
# ตัวอย่างการใช้งานจริง
python scripts/agent_agentic.py --mark-quality-gate-done
python scripts/agent_agentic.py --mark-build-done 30_1
python scripts/agent_agentic.py --mark-inspect-done w30_h1_context
python scripts/agent_agentic.py --mark-training-done w30_h1_context
python scripts/agent_agentic.py --mark-summarize-done w30_h1_context
```

> หลัง `--mark-summarize-done` Agent จะ **re-evaluate phase อัตโนมัติ** และอาจสร้าง dynamic experiments ใหม่

---

## 3. คำสั่งปรับเป้าหมาย (Layer 3 ใหม่)

### เปลี่ยนเป้าหมาย R²
```bash
python scripts/agent_agentic.py --set-goal-target 0.55
```
Agent จะพยายามหา experiment จนได้ R² ≥ 0.55

### เปลี่ยนเป้าหมาย Peak RMSE (outbreak)
```bash
python scripts/agent_agentic.py --set-peak-target 250.0
```
Agent จะ switch peak_focus mode เมื่อ peak RMSE > 250

---

## 4. คำสั่งควบคุม Phase (Layer 3 ใหม่)

Phase คือสถานะการค้นหาของ Agent มี 4 ระดับ

| Phase | ความหมาย | Agent จะทำ |
|---|---|---|
| `explore` | ยังห่างเป้ามาก | ลอง window/feature หลายชุด |
| `exploit` | ใกล้เป้าแล้ว | โฟกัสปรับ hyperparameter |
| `peak_focus` | R² ดีแต่ outbreak แย่ | เทรนด้วย weighted loss |
| `done` | เสร็จสิ้น | หยุดทำงาน |

```bash
# บังคับเปลี่ยน phase (ใช้สำหรับ debug หรือควบคุมเอง)
python scripts/agent_agentic.py --force-phase explore
python scripts/agent_agentic.py --force-phase exploit
python scripts/agent_agentic.py --force-phase peak_focus
python scripts/agent_agentic.py --force-phase done
```

---

## 5. คำสั่งดู State และ Queue (Layer 3 ใหม่)

### ดู state ทั้งหมด (JSON)
```bash
python scripts/agent_agentic.py --show-state
```

### ดู dynamic queue ที่ Agent สร้างเอง
```bash
python scripts/agent_agentic.py --show-queue
```
ตัวอย่าง output:
```
  Dynamic Queue (3 items):
    dyn_w15_h1_context  W=15 H=1 feat=context src=dynamic_window_search
    dyn_w45_h1_context  W=45 H=1 feat=context src=dynamic_window_search
    dyn_hp_w30_h1_context_lr00003  W=30 H=1 feat=context src=hyperopt
```

---

## 6. Auto Loop (รันอัตโนมัติจนเสร็จ)

```bash
# รันปกติ (จนถึง converge หรือ max 30 รอบ)
./scripts/agent_agentic_loop.sh

# จำกัดจำนวน iteration
./scripts/agent_agentic_loop.sh --max-iter 5

# ทดสอบโดยไม่รันจริง
./scripts/agent_agentic_loop.sh --dry-run

# ใช้ state file อื่น
./scripts/agent_agentic_loop.sh --state results/my_experiment.json

# รัน background (SSH-safe)
nohup ./scripts/agent_agentic_loop.sh > results/agentic.log 2>&1 &
tail -f results/agentic.log
```

---

## 7. Flow การทำงานทั้งหมด

```
เริ่มต้น
    ↓
quality_gate  →  --mark-quality-gate-done
    ↓
build NPZ     →  --mark-build-done {W}_{H}
    ↓
inspect NaN   →  --mark-inspect-done {EXP_ID}
    ↓
train models  →  --mark-training-done {EXP_ID}
    ↓
summarize     →  --mark-summarize-done {EXP_ID}
    ↓
[Agent re-evaluates phase & generates dynamic experiments]
    ↓
    ├─ explore  → ลอง W/feature ชุดใหม่ → วนซ้ำ
    ├─ exploit  → ปรับ hyperparameter → วนซ้ำ
    ├─ peak_focus → train weighted loss → วนซ้ำ
    └─ converged → เสร็จ ✅
```

---

## 8. ความแตกต่างจาก Layer 2

| ความสามารถ | Layer 2 (`agent_next_step.py`) | Layer 3 (`agent_agentic.py`) |
|---|---|---|
| รันตาม priority คงที่ | ✅ | ✅ |
| สร้าง experiment ใหม่เอง | ❌ | ✅ Dynamic Search Space |
| ปรับ hyperparameter อัตโนมัติ | ❌ | ✅ Hyperopt |
| โฟกัส outbreak | ❌ | ✅ Peak Focus Mode |
| ปรับเป้าหมายระหว่างทาง | ❌ | ✅ `--set-goal-target` |
| ควบคุม phase | ❌ | ✅ `--force-phase` |
| ดู dynamic queue | ❌ | ✅ `--show-queue` |
| State file | `agent_state.json` | `agent_state_l3.json` |
