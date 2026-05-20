#!/usr/bin/env python3
"""
agent_agentic.py  —  BPH Layer 3: Agentic Decision Engine
=============================================================
ต่างจาก Layer 2 (agent_next_step.py) ตรงที่:
  - Dynamic Search Space: ปรับชุดทดลองตามผลจริง ไม่ใช่แค่รันตาม priority คงที่
  - Auto Hyperparameter Tuning: วิเคราะห์ loss pattern แล้วแนะนำ lr/dropout/batch
  - Peak Detection: ตรวจ outbreak pattern แล้ว switch weighted loss อัตโนมัติ
  - Self-Eval Loop: ตัดสินใจ iterate หรือ stop โดยไม่รอคน

รัน: python scripts/agent_agentic.py [--state results/agent_state_l3.json]
"""

import argparse
import json
import math
import os
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional


PYTHON = "/home/ai-station/my_project/.tf251p310/bin/python"
DEFAULT_STATE = "results/agent_state_l3.json"

# ============================================================
# Base Search Space (seed experiments — Agent จะขยายเองจากนี้)
# ============================================================
BASE_EXPERIMENTS = [
    # id,                  window, horizon, feature_set, roll_days, preset, priority
    ("w30_h1_context",     30,  1,  "context", 7,  "balanced", 1),
    ("w30_h7_context",     30,  7,  "context", 7,  "balanced", 2),
    ("w60_h7_context",     60,  7,  "context", 7,  "balanced", 3),
    ("w90_h14_context",    90,  14, "context", 14, "balanced", 4),
    # Two-stage experiments (จะรันหลัง single-stage หลายชุดเสร็จแล้ว)
    ("two_stage_w30_h1_context_q75", 30, 1, "context", 7, "two_stage_q75", 8),
    ("two_stage_w30_h1_context_q80", 30, 1, "context", 7, "two_stage_q80", 9),
]

# Hyperparameter search ranges
HP_RANGES = {
    "lr":          [0.001, 0.0005, 0.0003, 0.0001],
    "dropout":     [0.1, 0.2, 0.3, 0.4],
    "batch_size":  [64, 128, 256],
    "lstm_units":  [32, 64, 128],
    "conv_filters":[32, 64, 128],
}

# Peak outbreak threshold (top quantile ที่ถือว่าเป็น outbreak)
PEAK_QUANTILE = 0.90
PEAK_WEIGHT   = 3.0   # น้ำหนักของ sample ในช่วงพีค

# Two-stage architecture switching thresholds
TWO_STAGE_WIN_THRESHOLD  =  0.02   # delta_r2 > +0.02 → สลับไป two-stage
TWO_STAGE_LOSE_THRESHOLD = -0.01   # delta_r2 < -0.01 → คง single_stage

# ============================================================
# State helpers
# ============================================================

def load_state(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {
        "created": datetime.now().isoformat(),
        "goal": {
            "metric": "r2_log1p",
            "target": 0.55,
            "min_acceptable": 0.40,
            "peak_metric": "peak_rmse",   # ตัวชี้วัด outbreak เพิ่มเติม
            "peak_target": 300.0,          # RMSE บน top-10% ต้องต่ำกว่านี้
        },
        "experiments": {},
        "dynamic_queue": [],   # ← ใหม่: คิวที่ Agent สร้างขึ้นเองจากผลจริง
        "hp_history": [],      # ← ใหม่: ประวัติ hyperparameter ที่ลองแล้ว
        "peak_mode": False,    # ← ใหม่: กำลังโฟกัส outbreak หรือไม่
        "current_best": None,
        "current_best_score": None,
        "current_best_peak": None,
        "iteration": 0,
        "phase": "explore",    # explore → exploit → peak_focus → done
        "status": "running",
        "quality_gate_done": False,
        "preferred_arch": "single_stage",   # ← single_stage | two_stage
        "arch_decision_log": [],             # ← ประวัติการตัดสินใจเปลี่ยน arch
        "two_stage_tested": False,           # ← ทดสอบ two-stage แล้วหรือยัง
        "log": [],
    }


def save_state(state: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, ensure_ascii=False))


def log_event(state: dict, msg: str, level: str = "INFO"):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{ts}][{level}] {msg}"
    state["log"].append(entry)
    icon = {"INFO": "ℹ", "WARN": "⚠", "GOOD": "✅", "ACT": "⚡"}
    print(f"  {icon.get(level,'·')} {msg}")


# ============================================================
# Metrics reader (รองรับหลายรูปแบบ metrics.json)
# ============================================================

def read_all_metrics(run_dir: Path) -> dict:
    """อ่าน metrics ทุก model ใน run_dir และหาค่าดีที่สุด"""
    results = {}
    for mf in run_dir.glob("*/metrics.json"):
        model_name = mf.parent.name
        try:
            data = json.loads(mf.read_text())
            # normalize: รองรับ flat หรือ nested {test: {...}}
            flat = data.get("test", data)
            results[model_name] = {
                "r2_log1p":   flat.get("r2_log1p"),
                "rmse_log1p": flat.get("rmse_log1p"),
                "mae_log1p":  flat.get("mae_log1p"),
                "rmse_raw":   flat.get("rmse_raw"),
                "r2_raw":     flat.get("r2_raw"),
                "peak_rmse":  flat.get("peak_rmse"),   # อาจมีหรือไม่มี
            }
        except Exception:
            pass
    return results


def read_two_stage_metrics(run_dir: Path,
                            single_stage_r2: float | None = None) -> dict:
    """
    อ่าน metrics.json ของ two-stage model และคืน delta_r2 เทียบกับ single-stage
    - ถ้า metrics.json มี single_stage_r2_log1p → ใช้ค่านั้น
    - ถ้าเป็น null → ใช้ single_stage_r2 ที่ส่งมา (จาก state)
    - ถ้าทั้งคู่ไม่มี → delta = None (ยังตัดสินใจไม่ได้)
    """
    mf = run_dir / "metrics.json"
    if not mf.exists():
        # ลองหาใน subdirectory (two-stage อาจเก็บใน run_dir/metrics.json โดยตรง)
        for candidate in [run_dir / "metrics.json",
                          run_dir.parent / f"{run_dir.name}_metrics.json"]:
            if candidate.exists():
                mf = candidate
                break
        else:
            return {}
    try:
        data = json.loads(mf.read_text())
        imp  = data.get("improvement_vs_single_stage", {})

        ts_r2   = data.get("log1p", {}).get("r2")
        ss_r2   = imp.get("single_stage_r2_log1p") or single_stage_r2
        delta   = imp.get("delta_r2")

        # คำนวณ delta เองถ้า metrics.json ไม่มีแต่เรารู้ทั้งสอง R²
        if delta is None and ts_r2 is not None and ss_r2 is not None:
            delta = round(float(ts_r2) - float(ss_r2), 6)

        return {
            "two_stage_r2":    ts_r2,
            "single_stage_r2": ss_r2,
            "delta_r2":        delta,
            "classifier_f1":   data.get("stage1_classifier", {}).get("f1"),
            "classifier_auc":  data.get("stage1_classifier", {}).get("auc_roc"),
        }
    except Exception:
        return {}


def best_from_metrics(metrics: dict, metric: str = "r2_log1p") -> tuple[str, float]:
    """คืน (model_name, best_value)"""
    best_model, best_val = None, None
    for model, m in metrics.items():
        v = m.get(metric)
        if v is not None:
            if best_val is None or v > best_val:
                best_val, best_model = v, model
    return best_model, best_val


# ============================================================
# Layer 3 Core: Dynamic Search Space Generator
# ============================================================

def generate_next_experiments(state: dict) -> list[dict]:
    """
    วิเคราะห์ผลที่ผ่านมาแล้วสร้าง experiment ชุดต่อไป
    หลักการ: Exploit ถ้า gap เล็ก, Explore ถ้า gap ใหญ่
    """
    experiments = state.get("experiments", {})
    goal = state["goal"]
    best_score = state.get("current_best_score") or 0
    best_exp_id = state.get("current_best")
    new_exps = []

    done_exps = {eid: e for eid, e in experiments.items()
                 if e.get("status") == "done" and e.get("best_score") is not None}

    if not done_exps:
        return []

    # ---- วิเคราะห์แนวโน้ม ----
    scores = [(eid, e["best_score"]) for eid, e in done_exps.items()]
    scores.sort(key=lambda x: x[1], reverse=True)

    top_eid, top_score = scores[0]
    top_exp = experiments[top_eid]

    gap = goal["target"] - best_score
    phase = state.get("phase", "explore")

    # ---- กลยุทธ์ตาม phase ----
    if phase == "explore" and gap > 0.05:
        # ยังห่างเป้ามาก → explore window/feature combinations ใหม่
        w = top_exp.get("window", 30)
        h = top_exp.get("horizon", 1)
        feat = top_exp.get("feature_set", "context")

        candidates = []
        # ลอง window ใกล้เคียง
        for dw in [-15, +15, +30]:
            nw = max(15, w + dw)
            if nw != w:
                candidates.append({
                    "id": f"dyn_w{nw}_h{h}_{feat}",
                    "window": nw, "horizon": h,
                    "feature_set": feat, "roll_days": 7 if h <= 7 else 14,
                    "preset": "balanced", "source": "dynamic_window_search"
                })
        # ลอง feature set อื่น
        for nf in ["context", "full", "core"]:
            if nf != feat:
                candidates.append({
                    "id": f"dyn_w{w}_h{h}_{nf}",
                    "window": w, "horizon": h,
                    "feature_set": nf, "roll_days": 7,
                    "preset": "balanced", "source": "dynamic_feature_search"
                })
        new_exps = candidates[:3]  # เพิ่มครั้งละไม่เกิน 3

    elif phase == "exploit" or gap <= 0.05:
        # ใกล้เป้า → exploit: ปรับ quality preset + hyperopt รอบ best
        w = top_exp.get("window", 30)
        h = top_exp.get("horizon", 1)
        feat = top_exp.get("feature_set", "context")

        # เพิ่ม quality preset ถ้ายังไม่มี
        qual_id = f"dyn_w{w}_h{h}_{feat}_quality"
        if qual_id not in experiments:
            new_exps.append({
                "id": qual_id,
                "window": w, "horizon": h,
                "feature_set": feat, "roll_days": 7,
                "preset": "quality", "source": "exploit_quality"
            })

        # เพิ่ม hyperopt experiment
        hp_exp = suggest_hyperopt(state, w, h, feat)
        if hp_exp:
            new_exps.append(hp_exp)

    # ---- กรอง experiment ที่มีแล้ว ----
    existing = set(experiments.keys()) | {e["id"] for e in state.get("dynamic_queue", [])}
    new_exps = [e for e in new_exps if e["id"] not in existing]

    return new_exps


# ============================================================
# Layer 3 Core: Auto Hyperparameter Tuning
# ============================================================

def suggest_hyperopt(state: dict, window: int, horizon: int,
                     feature_set: str) -> Optional[dict]:
    """
    วิเคราะห์ loss history แล้วแนะนำ hyperparameter ชุดต่อไป
    ใช้ simple bandit: ลอง lr ต่ำลงถ้า val_loss oscillate,
    เพิ่ม dropout ถ้า overfit
    """
    experiments = state.get("experiments", {})
    hp_history  = state.get("hp_history", [])

    # รวบรวม hp ที่ลองแล้วในกลุ่ม w/h/feat เดียวกัน
    tried_lrs = set()
    tried_dropouts = set()
    for hp in hp_history:
        if hp.get("window") == window and hp.get("horizon") == horizon:
            tried_lrs.add(hp.get("lr"))
            tried_dropouts.add(hp.get("dropout"))

    # เลือก lr ที่ยังไม่ลอง
    next_lr = None
    for lr in HP_RANGES["lr"]:
        if lr not in tried_lrs:
            next_lr = lr
            break
    if next_lr is None:
        return None   # ลอง lr ทุกค่าแล้ว

    # เลือก dropout
    next_dropout = 0.25
    for d in HP_RANGES["dropout"]:
        if d not in tried_dropouts:
            next_dropout = d
            break

    hp_id = f"dyn_hp_w{window}_h{horizon}_{feature_set}_lr{str(next_lr).replace('.','')}"
    return {
        "id": hp_id,
        "window": window, "horizon": horizon,
        "feature_set": feature_set, "roll_days": 7 if horizon <= 7 else 14,
        "preset": "custom",
        "custom_hp": {"lr": next_lr, "dropout": next_dropout, "batch_size": 128},
        "source": "hyperopt"
    }


# ============================================================
# Layer 3 Core: Peak / Outbreak Detection
# ============================================================

def check_peak_mode(state: dict) -> bool:
    """
    ตรวจว่าควร switch ไป peak_focus mode หรือไม่
    เงื่อนไข: R²(log1p) ดีพอ (>= min_acceptable)
              แต่ peak_rmse ยังสูง (> peak_target)
    """
    goal = state["goal"]
    best_score = state.get("current_best_score") or 0
    best_peak  = state.get("current_best_peak")

    min_ok = best_score >= goal.get("min_acceptable", 0.40)
    peak_bad = (best_peak is None or
                best_peak > goal.get("peak_target", 300.0))

    return min_ok and peak_bad


def make_peak_train_cmd(exp: dict) -> str:
    """สร้างคำสั่งเทรนแบบ weighted loss เพื่อโฟกัส outbreak"""
    w, h, feat = exp["window"], exp["horizon"], exp["feature_set"]
    npz = f"results/out_feature_sets_w{w}_h{h}/{feat}/sequences_window{w}_h{h}.npz"
    out = f"results/out_train_{exp['id']}"
    hp  = exp.get("custom_hp", {})
    lr  = hp.get("lr", 0.0003)
    dr  = hp.get("dropout", 0.25)
    bs  = hp.get("batch_size", 128)

    return (
        f"{PYTHON} scripts/train_cnn_lstm.py \\\n"
        f"  --npz {npz} \\\n"
        f"  --out_dir {out}/cnn_lstm_peak \\\n"
        f"  --epochs 260 --batch_size {bs} --lr {lr} \\\n"
        f"  --patience 45 --clipnorm 1.0 \\\n"
        f"  --conv_filters 64 --kernel_size 5 \\\n"
        f"  --lstm_units 64 --dropout {dr} \\\n"
        f"  --peak_quantile {PEAK_QUANTILE} --peak_weight {PEAK_WEIGHT}"
    )


# ============================================================
# Layer 3 Core: Self-Evaluation & Phase Controller
# ============================================================

def evaluate_preferred_arch(state: dict):
    """
    ตรวจผล two-stage ทุกครั้งที่ summarize เสร็จ แล้วตัดสินใจ:
      delta_r2 > TWO_STAGE_WIN_THRESHOLD  → preferred_arch = "two_stage"
      delta_r2 < TWO_STAGE_LOSE_THRESHOLD → preferred_arch = "single_stage"
      ไม่อยู่ในช่วงนั้น → คงค่าเดิม (undecided)
    """
    experiments = state.get("experiments", {})

    # หา two-stage experiment ที่ done แล้ว
    ts_results = []
    ss_r2 = state.get("current_best_score")   # single-stage baseline จาก state
    for eid, exp_data in experiments.items():
        if "two_stage" not in eid:
            continue
        if exp_data.get("status") != "done":
            continue

        # อ่าน delta จาก exp_data ที่ inject ไว้ก่อน (เร็วกว่า)
        cached_delta = exp_data.get("delta_r2")
        cached_ts_r2 = exp_data.get("best_score")
        if cached_delta is not None:
            ts_m = {
                "two_stage_r2":    cached_ts_r2,
                "single_stage_r2": ss_r2,
                "delta_r2":        cached_delta,
                "classifier_f1":   exp_data.get("classifier_f1"),
                "classifier_auc":  exp_data.get("classifier_auc"),
            }
            ts_results.append((eid, ts_m))
            continue

        # ถ้าไม่มีใน exp_data → อ่านจาก metrics.json
        run_dir = Path(f"results/out_train_{eid}")
        ts_m = read_two_stage_metrics(run_dir, single_stage_r2=ss_r2)
        if ts_m.get("delta_r2") is not None:
            ts_results.append((eid, ts_m))

    if not ts_results:
        return  # ยังไม่มีผล two-stage

    state["two_stage_tested"] = True

    # ใช้ผลล่าสุด (delta_r2 สูงสุด = ดีที่สุดที่ two-stage ทำได้)
    best_eid, best_ts = max(ts_results, key=lambda x: x[1].get("delta_r2", -999))
    delta = best_ts["delta_r2"]
    old_arch = state.get("preferred_arch", "single_stage")

    if delta > TWO_STAGE_WIN_THRESHOLD:
        new_arch = "two_stage"
        reason = f"delta_r2={delta:+.4f} > {TWO_STAGE_WIN_THRESHOLD}"
    elif delta < TWO_STAGE_LOSE_THRESHOLD:
        new_arch = "single_stage"
        reason = f"delta_r2={delta:+.4f} < {TWO_STAGE_LOSE_THRESHOLD}"
    else:
        reason = f"delta_r2={delta:+.4f} อยู่ในช่วง undecided — คงไว้"
        log_event(state, f"arch undecided: {reason}", "WARN")
        return

    if new_arch != old_arch:
        state["preferred_arch"] = new_arch
        rec = {
            "iteration": state.get("iteration", 0),
            "from": old_arch, "to": new_arch,
            "reason": reason,
            "exp_id": best_eid,
            "delta_r2": delta,
            "ts": datetime.now().isoformat(),
        }
        state.setdefault("arch_decision_log", []).append(rec)
        level = "GOOD" if new_arch == "two_stage" else "INFO"
        log_event(state, f"ARCH SWITCH: {old_arch} → {new_arch}  ({reason})", level)
    else:
        log_event(state, f"ARCH confirmed: {new_arch}  ({reason})", "INFO")


def evaluate_and_update_phase(state: dict):
    """
    อ่าน metrics ทุก experiment ที่ done แล้ว update:
    - current_best / current_best_score / current_best_peak
    - phase (explore → exploit → peak_focus → done)
    - dynamic_queue (เพิ่ม experiments ใหม่ถ้าจำเป็น)
    """
    experiments = state.get("experiments", {})
    goal = state["goal"]

    # อัปเดต best scores
    for eid, exp_data in experiments.items():
        if exp_data.get("status") != "done":
            continue
        score = exp_data.get("best_score")
        peak  = exp_data.get("best_peak_rmse")
        if score is not None:
            old = state.get("current_best_score")
            if old is None or score > old:
                state["current_best"] = eid
                state["current_best_score"] = score
                log_event(state, f"NEW BEST: {eid} → R²={score:.4f}", "GOOD")
        if peak is not None:
            old_peak = state.get("current_best_peak")
            if old_peak is None or peak < old_peak:
                state["current_best_peak"] = peak

    # อัปเดต current_best_peak จาก best experiment (ไม่ใช่ min ทุก exp)
    best_exp_id = state.get("current_best")
    if best_exp_id and best_exp_id in experiments:
        pk = experiments[best_exp_id].get("best_peak_rmse")
        if pk is not None:
            state["current_best_peak"] = pk

    best  = state.get("current_best_score") or 0
    gap   = goal["target"] - best
    phase = state.get("phase", "explore")

    # Phase transitions
    if best >= goal["target"]:
        if check_peak_mode(state):
            new_phase = "peak_focus"
        else:
            new_phase = "done"
    elif gap <= 0.05:
        new_phase = "exploit"
    else:
        new_phase = "explore"

    if new_phase != phase:
        log_event(state, f"Phase: {phase} → {new_phase}", "ACT")
        state["phase"] = new_phase

    # สร้าง dynamic experiments
    new_exps = generate_next_experiments(state)
    if new_exps:
        q = state.setdefault("dynamic_queue", [])
        for e in new_exps:
            q.append(e)
            log_event(state, f"DYNAMIC: เพิ่ม {e['id']} (source={e.get('source','-')})", "ACT")

    # ตรวจสอบ two-stage architecture preference
    evaluate_preferred_arch(state)

    # อัปเดต peak mode
    state["peak_mode"] = check_peak_mode(state)


# ============================================================
# Build commands (extended for custom HP)
# ============================================================

def make_build_cmd(exp: dict) -> str:
    w, h, feat, roll = exp["window"], exp["horizon"], exp["feature_set"], exp["roll_days"]
    return (
        f"{PYTHON} scripts/build_sequences.py \\\n"
        f"  --input_csv results/out_quality_gate/cleaned_raw.csv \\\n"
        f"  --out_dir results/out_feature_sets_w{w}_h{h} \\\n"
        f"  --window {w} --horizon {h} --roll_days {roll} \\\n"
        f"  --all_sets --require_consecutive"
    )


def make_inspect_cmd(exp: dict) -> str:
    w, h, feat = exp["window"], exp["horizon"], exp["feature_set"]
    return (f"{PYTHON} scripts/inspect_npz_for_nan.py \\\n"
            f"  --npz results/out_feature_sets_w{w}_h{h}/{feat}/"
            f"sequences_window{w}_h{h}.npz")


PRESETS_HP = {
    "fast":     dict(lstm_ep=50,  lstm_pat=10,  cnn_ep=50,  cnn_pat=10,
                     tf_ep=50,  tf_pat=10,  bs=256, lr=0.001,  dropout=0.2),
    "balanced": dict(lstm_ep=120, lstm_pat=20,  cnn_ep=150, cnn_pat=25,
                     tf_ep=200, tf_pat=30,  bs=128, lr=0.0005, dropout=0.25),
    "quality":  dict(lstm_ep=200, lstm_pat=35,  cnn_ep=260, cnn_pat=45,
                     tf_ep=260, tf_pat=45,  bs=128, lr=0.0003, dropout=0.3),
}


def make_train_cmd(exp: dict, peak_mode: bool = False) -> str:
    w, h, feat = exp["window"], exp["horizon"], exp["feature_set"]
    out = f"results/out_train_{exp['id']}"
    npz = f"results/out_feature_sets_w{w}_h{h}/{feat}/sequences_window{w}_h{h}.npz"

    if peak_mode and exp.get("source") == "peak_focus":
        return make_peak_train_cmd(exp)

    # ── Two-stage experiment ──
    preset_name = exp.get("preset", "balanced")
    if preset_name.startswith("two_stage"):
        quantile = 0.80 if "q80" in preset_name else 0.75
        return (
            f"# Two-stage experiment: {exp['id']}\n"
            f"{PYTHON} scripts/train_two_stage.py \\\n"
            f"  --npz {npz} \\\n"
            f"  --out_dir {out} \\\n"
            f"  --spike_quantile {quantile} \\\n"
            f"  --spike_threshold 0.5 \\\n"
            f"  --epochs_s1 120 --patience_s1 20 \\\n"
            f"  --epochs_s2 150 --patience_s2 25 \\\n"
            f"  --batch_size 128 --lr 0.0005 --lr_s2 0.0003"
        )

    # ── Custom HP experiment ──
    if preset_name == "custom" and exp.get("custom_hp"):
        hp = exp["custom_hp"]
        lr      = hp.get("lr", 0.0005)
        dropout = hp.get("dropout", 0.25)
        bs      = hp.get("batch_size", 128)
        return (
            f"# Custom hyperparameter experiment: {exp['id']}\n"
            f"{PYTHON} scripts/train_cnn_lstm.py \\\n"
            f"  --npz {npz} \\\n"
            f"  --out_dir {out}/cnn_lstm_custom \\\n"
            f"  --epochs 200 --batch_size {bs} --lr {lr} \\\n"
            f"  --patience 40 --clipnorm 1.0 \\\n"
            f"  --conv_filters 64 --kernel_size 5 \\\n"
            f"  --lstm_units 64 --dropout {dropout}"
        )

    return (
        f"./scripts/run_train_auto.sh \\\n"
        f"  {npz} \\\n"
        f"  {out}"
    )


def make_summarize_cmd(exp: dict) -> str:
    out = f"results/out_train_{exp['id']}"
    return (
        f"{PYTHON} scripts/summarize_runs.py \\\n"
        f"  --root {out} \\\n"
        f"  --out {out}/summary"
    )


# ============================================================
# Decision logic (Layer 3 version)
# ============================================================

def get_next_pending(state: dict) -> Optional[dict]:
    """
    หา experiment ถัดไปที่ต้องทำ:
    1. ตรวจ base experiments ก่อน
    2. ตรวจ dynamic_queue (สร้างโดย Agent เอง)
    """
    experiments = state.get("experiments", {})
    queue = state.get("dynamic_queue", [])

    # รวมทุก experiment เรียงตาม priority
    all_exps = []
    for eid, w, h, feat, roll, preset, pri in BASE_EXPERIMENTS:
        all_exps.append((pri, {
            "id": eid, "window": w, "horizon": h,
            "feature_set": feat, "roll_days": roll, "preset": preset
        }))
    for i, exp in enumerate(queue):
        all_exps.append((100 + i, exp))  # dynamic queue ตาม phase

    all_exps.sort(key=lambda x: x[0])

    for _, exp in all_exps:
        eid = exp["id"]
        exp_state = experiments.get(eid, {})
        status = exp_state.get("status", "pending")

        if status in ("done", "failed"):
            continue

        w, h, feat = exp["window"], exp["horizon"], exp["feature_set"]
        npz_path = Path(f"results/out_feature_sets_w{w}_h{h}/{feat}/sequences_window{w}_h{h}.npz")

        if status == "pending":
            if not npz_path.exists():
                build_key = f"build_w{w}_h{h}"
                if not state.get(build_key):
                    return ("build", exp)
                if not exp_state.get("inspect_done"):
                    return ("inspect", exp)
            if not exp_state.get("inspect_done"):
                return ("inspect", exp)
            return ("train", exp)

        if status == "training_done" and not exp_state.get("summarize_done"):
            return ("summarize", exp)

    return None


def decide_l3(state: dict) -> tuple[str, Optional[dict]]:
    """Layer 3 decision: รวม pipeline + dynamic + peak"""
    if not state.get("quality_gate_done"):
        return ("quality_gate", None)

    goal = state["goal"]
    best = state.get("current_best_score") or 0

    # ตรวจ convergence
    if best >= goal["target"] and not check_peak_mode(state):
        return ("converged", None)

    # ถ้าอยู่ phase peak_focus → train peak model
    if state.get("phase") == "peak_focus":
        best_eid = state.get("current_best")
        if best_eid:
            best_exp_data = state.get("experiments", {}).get(best_eid, {})
            peak_id = f"{best_eid}_peak"
            if peak_id not in state.get("experiments", {}):
                be = state.get("experiments", {}).get(best_eid, {})
                peak_exp = {
                    "id": peak_id,
                    "window": be.get("window", 30),
                    "horizon": be.get("horizon", 1),
                    "feature_set": be.get("feature_set", "context"),
                    "roll_days": 7,
                    "preset": "custom",
                    "custom_hp": {"lr": 0.0003, "dropout": 0.3, "batch_size": 64},
                    "source": "peak_focus"
                }
                return ("train_peak", peak_exp)

    # ตรวจ dynamic experiments
    result = get_next_pending(state)
    if result:
        return result

    # ถ้าไม่มีอะไรเหลือ → evaluate แล้วอาจสร้าง dynamic ใหม่
    done_count = sum(1 for e in state.get("experiments", {}).values()
                     if e.get("status") == "done")
    if done_count > 0:
        new_exps = generate_next_experiments(state)
        if new_exps:
            q = state.setdefault("dynamic_queue", [])
            for e in new_exps:
                if e["id"] not in state.get("experiments", {}):
                    q.append(e)
            return decide_l3(state)  # recursive: ลองอีกครั้ง

    return ("done", None)


# ============================================================
# Mark helpers
# ============================================================

def mark_quality_gate(state: dict):
    state["quality_gate_done"] = True
    log_event(state, "quality_gate DONE", "GOOD")


def mark_build(state: dict, w: int, h: int):
    state[f"build_w{w}_h{h}"] = True
    log_event(state, f"build W={w},H={h} DONE", "GOOD")


def mark_inspect(state: dict, eid: str):
    state.setdefault("experiments", {}).setdefault(eid, {})["inspect_done"] = True
    log_event(state, f"inspect[{eid}] DONE", "GOOD")


def mark_training(state: dict, eid: str, exp: Optional[dict] = None):
    exps = state.setdefault("experiments", {})
    exps.setdefault(eid, {})["status"] = "training_done"
    exps[eid]["train_finished_at"] = datetime.now().isoformat()
    if exp and exp.get("custom_hp"):
        # บันทึก hp ที่ใช้ลงใน hp_history
        rec = {**exp["custom_hp"],
               "window": exp.get("window"),
               "horizon": exp.get("horizon"),
               "exp_id": eid}
        state.setdefault("hp_history", []).append(rec)
    log_event(state, f"train[{eid}] DONE", "GOOD")


def mark_summarize(state: dict, eid: str):
    exps = state.setdefault("experiments", {})
    exps.setdefault(eid, {})["summarize_done"] = True
    exps[eid]["status"] = "done"

    # อ่าน metrics จริง
    run_dir = Path(f"results/out_train_{eid}")
    all_m = read_all_metrics(run_dir)
    if all_m:
        best_model, best_r2 = best_from_metrics(all_m, "r2_log1p")
        _, best_peak = best_from_metrics(all_m, "peak_rmse")
        exps[eid]["best_score"] = best_r2
        exps[eid]["best_peak_rmse"] = best_peak
        exps[eid]["best_model"] = best_model
        log_event(state, f"summarize[{eid}] R²={best_r2} peak_rmse={best_peak}", "GOOD")
    else:
        log_event(state, f"summarize[{eid}] done (metrics not found)", "WARN")

    state["iteration"] = state.get("iteration", 0) + 1
    # Re-evaluate phase after every summarize
    evaluate_and_update_phase(state)


# ============================================================
# Printer
# ============================================================

def print_decision(action: str, exp: Optional[dict], state: dict, state_path: str):
    sep = "=" * 62
    goal = state["goal"]
    best = state.get("current_best_score")
    phase = state.get("phase", "explore")
    peak_mode = state.get("peak_mode", False)

    print(f"\n{sep}")
    print(f"  🤖 AGENTIC DECISION  (iter={state.get('iteration',0)}  phase={phase})")
    print(sep)
    print(f"  Action  : {action.upper()}")
    if best is not None:
        gap = goal["target"] - best
        print(f"  Best    : {state.get('current_best')} → R²={best:.4f}  (gap={gap:+.4f})")
    else:
        print("  Best    : (none yet)")
    print(f"  Goal    : R² ≥ {goal['target']}  |  Peak RMSE ≤ {goal.get('peak_target','?')}")
    arch = state.get("preferred_arch", "single_stage")
    ts_tested = state.get("two_stage_tested", False)
    arch_icon = "★" if arch == "two_stage" else "·"
    ts_label = f"tested (delta logged)" if ts_tested else "not tested yet"
    print(f"  Arch    : {arch_icon} {arch}  |  two-stage: {ts_label}")
    if peak_mode:
        print("  ⚠️  PEAK MODE ACTIVE — กำลังโฟกัส outbreak performance")

    dq = state.get("dynamic_queue", [])
    pending_dyn = [e for e in dq if e["id"] not in state.get("experiments", {})]
    if pending_dyn:
        print(f"  Queue   : {len(pending_dyn)} dynamic experiments รอดำเนินการ")
    print()

    if action == "converged":
        print(f"  ✅ GOAL REACHED!  R²={best:.4f} ≥ {goal['target']}")
        print(f"  Best experiment: {state.get('current_best')}")
        return
    if action == "done":
        print("  ⚠️  Search space + dynamic queue exhausted")
        print(f"  Best achieved: {best}")
        return
    if action == "quality_gate":
        cmd = (f"{PYTHON} scripts/quality_gate.py \\\n"
               f"  --input_csv data/env_daily_with_rice_monthly_raw.csv \\\n"
               f"  --out_dir results/out_quality_gate \\\n"
               f"  --mode soft --drop_rows_missing_mandatory")
        print(f"  📋 COMMAND:\n\n{cmd}\n")
        print(f"  After success:\n    python scripts/agent_agentic.py --state {state_path} --mark-quality-gate-done")
        return

    if exp:
        print(f"  Experiment : {exp['id']}")
        print(f"  Config     : W={exp['window']}, H={exp['horizon']}, "
              f"feat={exp['feature_set']}, preset={exp.get('preset','?')}")
        if exp.get("custom_hp"):
            hp = exp["custom_hp"]
            print(f"  Custom HP  : lr={hp.get('lr')} dropout={hp.get('dropout')} bs={hp.get('batch_size')}")
        if exp.get("source"):
            print(f"  Source     : {exp['source']}")
        print()

    eid = exp["id"] if exp else "?"

    if action == "build":
        cmd = make_build_cmd(exp)
        w, h = exp["window"], exp["horizon"]
        mark = f"--mark-build-done {w}_{h}"
    elif action == "inspect":
        cmd = make_inspect_cmd(exp)
        mark = f"--mark-inspect-done {eid}"
    elif action in ("train", "train_peak"):
        cmd = make_train_cmd(exp, peak_mode=(action == "train_peak"))
        mark = f"--mark-training-done {eid}"
    elif action == "summarize":
        cmd = make_summarize_cmd(exp)
        mark = f"--mark-summarize-done {eid}"
    else:
        cmd = "# unknown action"
        mark = ""

    print(f"  📋 COMMAND:\n\n{cmd}\n")
    if mark:
        print(f"  After success:\n    python scripts/agent_agentic.py --state {state_path} {mark}")
    print(sep + "\n")


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser(description="BPH Agent Layer 3: Agentic Decision Engine")
    ap.add_argument("--state", default=DEFAULT_STATE)
    ap.add_argument("--mark-quality-gate-done", action="store_true")
    ap.add_argument("--mark-build-done",    metavar="W_H")
    ap.add_argument("--mark-inspect-done",  metavar="EXP_ID")
    ap.add_argument("--mark-training-done", metavar="EXP_ID")
    ap.add_argument("--mark-summarize-done",metavar="EXP_ID")
    ap.add_argument("--set-goal-target",    metavar="FLOAT",  type=float,
                    help="เปลี่ยนเป้าหมาย R² เช่น 0.55")
    ap.add_argument("--set-peak-target",    metavar="FLOAT",  type=float,
                    help="เปลี่ยนเป้าหมาย peak RMSE เช่น 250.0")
    ap.add_argument("--inject-two-stage",   metavar="EXP_ID",
                    help="inject ผล two-stage จาก metrics.json เพื่อ trigger arch decision ทันที")
    ap.add_argument("--show-arch",   action="store_true",
                    help="แสดงสถานะ preferred_arch และ arch_decision_log")
    ap.add_argument("--force-arch",  metavar="ARCH",
                    choices=["single_stage", "two_stage"],
                    help="บังคับเปลี่ยน preferred_arch โดยไม่ต้องรอผล")
    ap.add_argument("--force-phase",        metavar="PHASE",
                    choices=["explore","exploit","peak_focus","done"],
                    help="บังคับ phase (ใช้สำหรับ debug)")
    ap.add_argument("--show-state",  action="store_true")
    ap.add_argument("--show-queue",  action="store_true")
    ap.add_argument("--dry-run",     action="store_true")
    args = ap.parse_args()

    state_path = Path(args.state)
    state = load_state(state_path)

    # ---- show state ----
    if args.show_state:
        print(json.dumps(state, indent=2, ensure_ascii=False))
        return

    if args.show_queue:
        q = state.get("dynamic_queue", [])
        exps = state.get("experiments", {})
        pending = [e for e in q if e["id"] not in exps or exps[e["id"]].get("status") == "pending"]
        print(f"\n  Dynamic Queue ({len(pending)} pending):")
        for e in pending:
            print(f"    {e['id']}  W={e['window']} H={e['horizon']} "
                  f"feat={e['feature_set']} src={e.get('source','-')}")
        print()
        return

    # ---- goal update ----
    if args.set_goal_target:
        state["goal"]["target"] = args.set_goal_target
        log_event(state, f"Goal target → R²={args.set_goal_target}", "ACT")
        if not args.dry_run:
            save_state(state, state_path)
        return

    if args.set_peak_target:
        state["goal"]["peak_target"] = args.set_peak_target
        log_event(state, f"Peak target → RMSE={args.set_peak_target}", "ACT")
        if not args.dry_run:
            save_state(state, state_path)
        return

    if args.show_arch:
        arch = state.get("preferred_arch", "single_stage")
        tested = state.get("two_stage_tested", False)
        print(f"\n  preferred_arch  : {arch}")
        print(f"  two_stage_tested: {tested}")
        log = state.get("arch_decision_log", [])
        if log:
            print(f"  arch_decision_log ({len(log)} entries):")
            for rec in log:
                print(f"    [{rec.get('ts','?')[:19]}] "
                      f"{rec['from']} → {rec['to']}  ({rec['reason']})")
        else:
            print("  arch_decision_log: (empty — no switch yet)")
        print()
        return

    if args.force_arch:
        old = state.get("preferred_arch", "single_stage")
        state["preferred_arch"] = args.force_arch
        rec = {
            "iteration": state.get("iteration", 0),
            "from": old, "to": args.force_arch,
            "reason": "forced by --force-arch flag",
            "ts": datetime.now().isoformat(),
        }
        state.setdefault("arch_decision_log", []).append(rec)
        log_event(state, f"ARCH FORCED: {old} → {args.force_arch}", "ACT")
        if not args.dry_run: save_state(state, state_path)
        return

    if args.inject_two_stage:
        eid = args.inject_two_stage
        exps = state.setdefault("experiments", {})
        exps.setdefault(eid, {})["status"] = "done"
        exps[eid]["summarize_done"] = True

        # ดึง single_stage_r2 จาก current_best_score ใน state
        ss_r2 = state.get("current_best_score")

        run_dir = Path(f"results/out_train_{eid}")
        ts_m = read_two_stage_metrics(run_dir, single_stage_r2=ss_r2)

        if ts_m:
            exps[eid]["best_score"]     = ts_m.get("two_stage_r2")
            exps[eid]["delta_r2"]       = ts_m.get("delta_r2")
            exps[eid]["classifier_f1"]  = ts_m.get("classifier_f1")
            exps[eid]["classifier_auc"] = ts_m.get("classifier_auc")
            delta_str = f"{ts_m.get('delta_r2'):+.4f}" if ts_m.get("delta_r2") is not None else "None"
            log_event(state, f"inject two-stage [{eid}]: "
                      f"r2={ts_m.get('two_stage_r2'):.4f}  "
                      f"ss_r2={ts_m.get('single_stage_r2')}  "
                      f"delta={delta_str}", "INFO")
        else:
            log_event(state, f"inject two-stage [{eid}]: metrics not found at {run_dir}", "WARN")

        state["two_stage_tested"] = True
        evaluate_preferred_arch(state)
        if not args.dry_run: save_state(state, state_path)
        return

    if args.force_phase:
        state["phase"] = args.force_phase
        log_event(state, f"Phase forced → {args.force_phase}", "ACT")
        if not args.dry_run:
            save_state(state, state_path)
        return

    # ---- mark flags ----
    if args.mark_quality_gate_done:
        mark_quality_gate(state)
        if not args.dry_run: save_state(state, state_path)
        return

    if args.mark_build_done:
        parts = args.mark_build_done.split("_")
        mark_build(state, int(parts[0]), int(parts[1]))
        if not args.dry_run: save_state(state, state_path)
        return

    if args.mark_inspect_done:
        mark_inspect(state, args.mark_inspect_done)
        if not args.dry_run: save_state(state, state_path)
        return

    if args.mark_training_done:
        eid = args.mark_training_done
        exp = next((e for e in state.get("dynamic_queue", []) if e["id"] == eid), None)
        mark_training(state, eid, exp)
        if not args.dry_run: save_state(state, state_path)
        return

    if args.mark_summarize_done:
        mark_summarize(state, args.mark_summarize_done)
        if not args.dry_run: save_state(state, state_path)
        return

    # ---- decide ----
    action, exp = decide_l3(state)
    print_decision(action, exp, state, args.state)

    if action in ("converged", "done"):
        state["status"] = action
        if not args.dry_run:
            save_state(state, state_path)


if __name__ == "__main__":
    main()
