# วิธีทดสอบ Layer 3 — Agentic

ทดสอบ 4 ระดับตามลำดับความซับซ้อน  
รันทุกคำสั่งจาก `BPH_Research/`

---

## ก่อนเริ่ม — ติดตั้งไฟล์

```bash
cp agent_agentic.py        scripts/
cp agent_agentic_loop.sh   scripts/
chmod +x scripts/agent_agentic_loop.sh
```

---

## ระดับ 1: ทดสอบ State ทำงานถูกต้อง (~5 นาที)

ตรวจว่า Agent สร้าง state ใหม่ได้และตัดสินใจ quality_gate เป็นขั้นแรก

```bash
# สร้าง state ใหม่ ดูการตัดสินใจ
python scripts/agent_agentic.py --state results/test_l3.json
```

ผลที่คาดหวัง:
```
Action  : QUALITY_GATE
Best    : (none yet)
Goal    : R² ≥ 0.55  |  Peak RMSE ≤ 300.0
```

```bash
# ดู state JSON ที่สร้างขึ้น
python scripts/agent_agentic.py --state results/test_l3.json --show-state
```

ตรวจสอบว่ามีครบ: `goal`, `experiments`, `dynamic_queue`, `phase`, `hp_history`

---

## ระดับ 2: ทดสอบ Decision Flow ด้วยข้อมูลจำลอง (~10 นาที)

ใช้ผลจริงจาก Layer 1-2 ที่มีอยู่แล้ว inject เข้า state แทนการเทรนใหม่

```bash
# สร้าง state พร้อม baseline จากผลที่ทำไปแล้ว
python - << 'EOF'
import json
from pathlib import Path
from datetime import datetime

state = {
    "created": datetime.now().isoformat(),
    "goal": {"metric": "r2_log1p", "target": 0.55,
             "min_acceptable": 0.40, "peak_target": 300.0},
    "quality_gate_done": True,
    "build_w30_h1": True,
    "experiments": {
        "w30_h1_context": {
            "status": "done",
            "best_score": 0.500,
            "best_peak_rmse": 216.1,
            "best_model": "cnn_lstm_core",
            "inspect_done": True,
            "summarize_done": True,
        },
        "w30_h7_context": {
            "status": "done",
            "best_score": 0.484,
            "best_peak_rmse": 219.6,
            "best_model": "cnn_lstm_context",
            "inspect_done": True,
            "summarize_done": True,
        },
    },
    "current_best": "w30_h1_context",
    "current_best_score": 0.500,
    "current_best_peak": 216.1,
    "dynamic_queue": [],
    "hp_history": [],
    "phase": "explore",
    "peak_mode": False,
    "iteration": 2,
    "status": "running",
    "log": ["[2026-05-20] Injected from Layer 1-2 results"]
}

p = Path("results/test_l3_baseline.json")
p.parent.mkdir(exist_ok=True)
p.write_text(json.dumps(state, indent=2, ensure_ascii=False))
print("State created:", p)
print(f"Best: {state['current_best']} R²={state['current_best_score']}")
print(f"Gap to goal: {state['goal']['target'] - state['current_best_score']:+.3f}")
EOF
```

```bash
# ดูว่า Agent ตัดสินใจอะไรต่อ
python scripts/agent_agentic.py --state results/test_l3_baseline.json
```

ผลที่คาดหวัง: Agent เห็น gap=+0.05 → phase=explore → เสนอ experiment ถัดไป (w60_h7_context)

```bash
# ดู dynamic queue ที่ Agent สร้างเอง
python scripts/agent_agentic.py --state results/test_l3_baseline.json --show-queue
```

ผลที่คาดหวัง: มี dynamic experiments อย่างน้อย 2-3 รายการ เช่น `dyn_w15_h1_context`, `dyn_w45_h1_context`

---

## ระดับ 3: ทดสอบ Phase Transitions (~15 นาที)

ตรวจว่า Agent เปลี่ยน phase ถูกต้องตาม score

### 3a — ทดสอบ explore → exploit

```bash
# จำลองว่า score เข้าใกล้เป้า (gap <= 0.05)
python - << 'EOF'
import json
from pathlib import Path

p = Path("results/test_l3_baseline.json")
s = json.loads(p.read_text())
# ดัน score ขึ้นใกล้เป้า
s["experiments"]["w60_h7_context"] = {
    "status": "done", "best_score": 0.512,
    "best_peak_rmse": 210.0, "inspect_done": True, "summarize_done": True
}
s["current_best"] = "w60_h7_context"
s["current_best_score"] = 0.512
s["iteration"] = 3
p.write_text(json.dumps(s, indent=2, ensure_ascii=False))
print("Score updated: R²=0.512  gap=+0.038 → should trigger exploit")
EOF

python scripts/agent_agentic.py --state results/test_l3_baseline.json
```

ผลที่คาดหวัง: `phase: explore → exploit` และ Agent เสนอ quality preset หรือ hyperopt

### 3b — ทดสอบ exploit → peak_focus

```bash
# จำลองว่า R² ถึงเป้าแต่ peak_rmse ยังสูง
python - << 'EOF'
import json
from pathlib import Path

p = Path("results/test_l3_baseline.json")
s = json.loads(p.read_text())
s["experiments"]["w30_h7_context_quality"] = {
    "status": "done", "best_score": 0.560,
    "best_peak_rmse": 340.0,   # ← สูงกว่า peak_target=300
    "inspect_done": True, "summarize_done": True
}
s["current_best"] = "w30_h7_context_quality"
s["current_best_score"] = 0.560
s["current_best_peak"] = 340.0
s["iteration"] = 4
p.write_text(json.dumps(s, indent=2, ensure_ascii=False))
print("R²=0.560 (ถึงเป้า) แต่ peak_rmse=340 > 300 → should trigger peak_focus")
EOF

python scripts/agent_agentic.py --state results/test_l3_baseline.json
```

ผลที่คาดหวัง: `phase: peak_focus` และ Agent เสนอ train_peak ด้วย weighted loss

### 3c — ทดสอบ converged

```bash
# จำลองว่าทั้ง R² และ peak_rmse ดีแล้ว
python - << 'EOF'
import json
from pathlib import Path

p = Path("results/test_l3_baseline.json")
s = json.loads(p.read_text())
s["current_best_peak"] = 250.0   # ← ต่ำกว่า peak_target=300 แล้ว
s["iteration"] = 5
p.write_text(json.dumps(s, indent=2, ensure_ascii=False))
print("R²=0.560, peak_rmse=250 → should CONVERGE")
EOF

python scripts/agent_agentic.py --state results/test_l3_baseline.json
```

ผลที่คาดหวัง: `Action: CONVERGED` และ Agent หยุดทำงาน

---

## ระดับ 4: ทดสอบการรันจริงบน Pipeline (รันบนเครื่อง AI Station)

ทดสอบกับ data จริง 1 experiment โดยใช้ NPZ ที่มีอยู่แล้ว

### 4a — เตรียม state จากผลที่มีอยู่จริง

```bash
# สร้าง state ที่รู้ว่า NPZ และ quality_gate เสร็จแล้ว
python - << 'EOF'
import json
from pathlib import Path
from datetime import datetime

state = {
    "created": datetime.now().isoformat(),
    "goal": {"metric": "r2_log1p", "target": 0.55,
             "min_acceptable": 0.40, "peak_target": 300.0},
    "quality_gate_done": True,
    "build_w30_h1": True,
    "build_w30_h7": True,
    "build_w60_h7": True,
    "experiments": {
        "w30_h1_context": {
            "status": "done",
            "best_score": 0.500,
            "best_peak_rmse": 216.1,
            "inspect_done": True,
            "summarize_done": True,
        },
    },
    "current_best": "w30_h1_context",
    "current_best_score": 0.500,
    "current_best_peak": 216.1,
    "dynamic_queue": [],
    "hp_history": [],
    "phase": "explore",
    "peak_mode": False,
    "iteration": 1,
    "status": "running",
    "log": []
}

p = Path("results/agent_state_l3.json")
p.write_text(json.dumps(state, indent=2, ensure_ascii=False))
print("State ready. NPZ ที่มีอยู่แล้ว จะถูก skip build")
EOF
```

### 4b — รัน Agent แบบ dry-run ก่อน

```bash
./scripts/agent_agentic_loop.sh --max-iter 3 --dry-run
```

ตรวจสอบ output: Agent ควรเสนอ experiment ถัดไปจาก w30_h1_context ที่เสร็จแล้ว

### 4c — รันจริง 1 รอบ

```bash
# รันแค่ 1 iteration ดูผล
./scripts/agent_agentic_loop.sh --max-iter 1
```

### 4d — รัน Auto Loop เต็ม (background)

```bash
nohup ./scripts/agent_agentic_loop.sh \
  --state results/agent_state_l3.json \
  --max-iter 10 \
  > results/agentic_loop.log 2>&1 &

echo "PID: $!"
tail -f results/agentic_loop.log
```

---

## สิ่งที่ต้องตรวจสอบหลังรัน

```bash
# 1. ดู state ปัจจุบัน
python scripts/agent_agentic.py --state results/agent_state_l3.json --show-state

# 2. ดู dynamic queue
python scripts/agent_agentic.py --state results/agent_state_l3.json --show-queue

# 3. ดู log
python - << 'EOF'
import json
from pathlib import Path
s = json.loads(Path("results/agent_state_l3.json").read_text())
print(f"Phase    : {s.get('phase')}")
print(f"Iteration: {s.get('iteration')}")
print(f"Best R²  : {s.get('current_best_score')}")
print(f"Best exp : {s.get('current_best')}")
print(f"Peak RMSE: {s.get('current_best_peak')}")
print(f"Dyn queue: {len(s.get('dynamic_queue', []))} items")
print()
print("Log (last 5):")
for line in s.get("log", [])[-5:]:
    print(" ", line)
EOF
```

---

## ตารางสรุปผลที่คาดหวัง

| ระดับ | เวลา | สิ่งที่ตรวจ | ผลที่คาดหวัง |
|---|---|---|---|
| 1 — State | 5 นาที | สร้าง state, ตัดสินใจ quality_gate | Action=QUALITY_GATE |
| 2 — Baseline | 10 นาที | inject ผลเก่า, dynamic queue | queue มี ≥2 experiments |
| 3a — Exploit | 5 นาที | score ใกล้เป้า | phase → exploit |
| 3b — Peak | 5 นาที | R² ดีแต่ peak แย่ | phase → peak_focus |
| 3c — Converge | 2 นาที | ทั้งคู่ดี | Action=CONVERGED |
| 4 — จริง | ขึ้นกับ training | รันบน pipeline จริง | Dynamic exp ใหม่ปรากฏ |
