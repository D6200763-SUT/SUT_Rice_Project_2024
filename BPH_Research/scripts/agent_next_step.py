#!/usr/bin/env python3
"""
agent_next_step.py  —  BPH Agentic Loop: Decision Engine (Layer 3)
=======================================================================
อ่าน agent_state.json → ตัดสินใจว่าจะรันอะไรต่อ → พิมพ์คำสั่ง bash ออกมา
รัน: python scripts/agent_next_step.py [--state results/agent_state.json] [--dry-run]
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime


# ============================================================
# Search Space — ชุดการทดลองทั้งหมดเรียงตาม priority
# Agent จะเลือกตาม priority ที่ต่ำที่สุดที่ยัง pending
# ============================================================
SEARCH_SPACE = [
    # priority, id,               window, horizon, feature_set, roll_days, preset
    (1,  "w30_h1_context",        30,  1,  "context", 7,  "balanced"),
    (2,  "w30_h7_context",        30,  7,  "context", 7,  "balanced"),
    (3,  "w60_h7_context",        60,  7,  "context", 7,  "balanced"),
    (4,  "w60_h7_full",           60,  7,  "full",    7,  "balanced"),
    (5,  "w90_h14_context",       90,  14, "context", 14, "balanced"),
    (6,  "w90_h14_full",          90,  14, "full",    14, "balanced"),
    (7,  "w30_h7_context_quality",30,  7,  "context", 7,  "quality"),
    (8,  "w60_h7_context_quality",60,  7,  "context", 7,  "quality"),
    (9,  "w30_h1_full",           30,  1,  "full",    7,  "balanced"),
    (10, "w45_h7_context",        45,  7,  "context", 7,  "balanced"),
]

# Training hyperparameters per preset
PRESETS = {
    "fast":     dict(lstm_ep=50,  lstm_pat=10,  cnn_ep=50,  cnn_pat=10,  tf_ep=50,  tf_pat=10,  bs=256, lr=0.001),
    "balanced": dict(lstm_ep=120, lstm_pat=20,  cnn_ep=150, cnn_pat=25,  tf_ep=200, tf_pat=30,  bs=128, lr=0.0005),
    "quality":  dict(lstm_ep=200, lstm_pat=35,  cnn_ep=260, cnn_pat=45,  tf_ep=260, tf_pat=45,  bs=128, lr=0.0003),
}

PYTHON = "/home/ai-station/my_project/.tf251p310/bin/python"
DEFAULT_STATE = "results/agent_state.json"


# ============================================================
# State helpers
# ============================================================

def load_state(state_path: Path) -> dict:
    """โหลด state หรือสร้างใหม่ถ้ายังไม่มี"""
    if state_path.exists():
        return json.loads(state_path.read_text())
    return {
        "created": datetime.now().isoformat(),
        "goal": {"metric": "r2_log1p", "target": 0.50, "min_acceptable": 0.40},
        "experiments": {},
        "current_best": None,
        "current_best_score": None,
        "iteration": 0,
        "status": "running",
        "log": []
    }


def save_state(state: dict, state_path: Path):
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False))


def log_event(state: dict, msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    state["log"].append(f"[{ts}] {msg}")
    print(f"  [AGENT] {msg}")


# ============================================================
# Read results from metrics.json
# ============================================================

def read_best_metric(run_dir: Path, metric: str = "r2_log1p") -> float | None:
    """ค้นหา metric จาก metrics.json ในโฟลเดอร์ย่อย"""
    best = None
    for mf in run_dir.glob("*/metrics.json"):
        try:
            data = json.loads(mf.read_text())
            val = data.get(metric) or data.get("test", {}).get(metric)
            if val is not None:
                best = max(best, float(val)) if best is not None else float(val)
        except Exception:
            pass
    return best


# ============================================================
# Build bash commands
# ============================================================

def make_build_cmd(exp: dict) -> str:
    w, h, feat, roll = exp["window"], exp["horizon"], exp["feature_set"], exp["roll_days"]
    npz_dir = f"results/out_feature_sets_w{w}_h{h}"
    return (
        f"{PYTHON} scripts/build_sequences.py \\\n"
        f"  --input_csv results/out_quality_gate/cleaned_raw.csv \\\n"
        f"  --out_dir {npz_dir} \\\n"
        f"  --window {w} --horizon {h} --roll_days {roll} \\\n"
        f"  --all_sets --require_consecutive"
    )


def make_inspect_cmd(exp: dict) -> str:
    w, h, feat = exp["window"], exp["horizon"], exp["feature_set"]
    npz = f"results/out_feature_sets_w{w}_h{h}/{feat}/sequences_window{w}_h{h}.npz"
    return f"{PYTHON} scripts/inspect_npz_for_nan.py --npz {npz}"


def make_train_cmd(exp: dict) -> str:
    w, h, feat, preset_name = exp["window"], exp["horizon"], exp["feature_set"], exp["preset"]
    p = PRESETS[preset_name]
    npz = f"results/out_feature_sets_w{w}_h{h}/{feat}/sequences_window{w}_h{h}.npz"
    out = f"results/out_train_{exp['id']}"
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
# Decision logic
# ============================================================

def decide(state: dict) -> tuple[str, dict | None]:
    """
    คืนค่า: (action, experiment_dict | None)
    action: 'quality_gate' | 'build' | 'inspect' | 'train' | 'summarize' | 'done' | 'converged'
    """
    experiments = state.get("experiments", {})
    goal_target = state["goal"]["target"]

    # ตรวจว่า quality_gate ผ่านแล้วไหม
    if not state.get("quality_gate_done"):
        return "quality_gate", None

    # ตรวจว่าถึงเป้าหมายแล้วไหม
    best_score = state.get("current_best_score")
    if best_score is not None and best_score >= goal_target:
        return "converged", None

    # หา experiment ถัดไปตาม priority
    for priority, exp_id, window, horizon, feature_set, roll_days, preset in SEARCH_SPACE:
        exp_state = experiments.get(exp_id, {})
        status = exp_state.get("status", "pending")

        if status == "pending":
            exp = dict(
                id=exp_id, priority=priority,
                window=window, horizon=horizon,
                feature_set=feature_set, roll_days=roll_days,
                preset=preset
            )
            # ตรวจว่า NPZ มีแล้วไหม
            npz_path = Path(f"results/out_feature_sets_w{window}_h{horizon}/{feature_set}/sequences_window{window}_h{horizon}.npz")
            if not npz_path.exists():
                # ต้อง build ก่อน
                build_done_key = f"build_w{window}_h{horizon}"
                if not state.get(build_done_key):
                    return "build", exp
                return "inspect", exp

            # NPZ มีแล้ว ตรวจ nan ก่อนไหม
            inspect_done = exp_state.get("inspect_done", False)
            if not inspect_done:
                return "inspect", exp

            # พร้อมเทรน
            return "train", exp

        elif status == "training_done":
            # ต้อง summarize
            summarize_done = exp_state.get("summarize_done", False)
            if not summarize_done:
                exp = dict(
                    id=exp_id, priority=priority,
                    window=window, horizon=horizon,
                    feature_set=feature_set, roll_days=roll_days,
                    preset=preset
                )
                return "summarize", exp

        # status == "done" → ข้ามไป next priority

    return "done", None


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser(description="BPH Agent: decide next step")
    ap.add_argument("--state", default=DEFAULT_STATE, help="Path to agent_state.json")
    ap.add_argument("--mark-quality-gate-done", action="store_true",
                    help="อัปเดต state ว่า quality_gate ผ่านแล้ว")
    ap.add_argument("--mark-build-done", metavar="W_H",
                    help="อัปเดต state ว่า build w{W}_h{H} เสร็จแล้ว เช่น 60_7")
    ap.add_argument("--mark-inspect-done", metavar="EXP_ID",
                    help="อัปเดต state ว่า inspect ของ exp_id เสร็จแล้ว")
    ap.add_argument("--mark-training-done", metavar="EXP_ID",
                    help="อัปเดต state ว่า train ของ exp_id เสร็จแล้ว")
    ap.add_argument("--mark-summarize-done", metavar="EXP_ID",
                    help="อัปเดต state ว่า summarize ของ exp_id เสร็จแล้ว พร้อม read metrics")
    ap.add_argument("--dry-run", action="store_true",
                    help="แสดงคำสั่งที่จะรัน แต่ไม่อัปเดต state")
    ap.add_argument("--show-state", action="store_true",
                    help="แสดง state ปัจจุบันแล้วออก")
    ap.add_argument("--reset-data", action="store_true",
                    help="รีเซ็ต pipeline เพื่อรับข้อมูลชุดใหม่: ล้าง quality_gate + build flags "
                         "แต่คง experiments เดิมไว้เป็น baseline")
    ap.add_argument("--reset-data-csv", metavar="CSV_PATH",
                    help="เหมือน --reset-data แต่ระบุ path ของ CSV ใหม่ด้วย (บันทึกใน state)")
    ap.add_argument("--reset-all", action="store_true",
                    help="รีเซ็ตทุกอย่างกลับเป็น fresh state (เริ่มใหม่ทั้งหมด)")
    args = ap.parse_args()

    state_path = Path(args.state)
    state = load_state(state_path)

    # ---- Show state ----
    if args.show_state:
        print(json.dumps(state, indent=2, ensure_ascii=False))
        return

    # ---- Reset flags ----
    if args.reset_all:
        fresh = {
            "created": state.get("created", datetime.now().isoformat()),
            "goal": state.get("goal", {"metric": "r2_log1p", "target": 0.5, "min_acceptable": 0.4}),
            "experiments": {},
            "current_best": None,
            "current_best_score": None,
            "iteration": 0,
            "status": "running",
            "log": state.get("log", []),
        }
        log_event(fresh, "RESET-ALL: full reset — experiments + pipeline flags cleared")
        if not args.dry_run:
            save_state(fresh, state_path)
            print("  [AGENT] ✅ Full reset complete — state is now fresh")
        else:
            print("  [AGENT] [DRY-RUN] Would reset all state")
        return

    if args.reset_data or args.reset_data_csv:
        csv_path = args.reset_data_csv or state.get("data_csv", "data/env_daily_with_rice_monthly_raw.csv")

        # สิ่งที่ต้องล้าง: quality_gate + build flags ทั้งหมด
        removed_keys = []
        keys_to_remove = ["quality_gate_done"] + [k for k in state if k.startswith("build_w")]
        for k in keys_to_remove:
            if k in state:
                del state[k]
                removed_keys.append(k)

        # reset inspect_done ของทุก experiment (เพราะ NPZ จะ rebuild ใหม่)
        for eid, exp_data in state.get("experiments", {}).items():
            exp_data.pop("inspect_done", None)
            # ถ้า status ยัง pending ก็ไม่ต้องทำอะไร
            # ถ้า done/training_done → เก็บ score ไว้เป็น baseline แต่ให้ re-train
            if exp_data.get("status") in ("done", "training_done"):
                exp_data["baseline_score"] = exp_data.get("best_score")
                exp_data["status"] = "pending"
                exp_data.pop("summarize_done", None)
                exp_data.pop("best_score", None)
                exp_data.pop("train_finished_at", None)

        # บันทึก CSV ใหม่ใน state
        state["data_csv"] = csv_path
        state["status"] = "running"
        state["data_reset_at"] = datetime.now().isoformat()

        msg = (f"RESET-DATA: ล้าง {removed_keys} | "
               f"experiments reset to pending (baseline scores kept) | "
               f"new CSV: {csv_path}")
        log_event(state, msg)

        if not args.dry_run:
            save_state(state, state_path)
            print()
            print("  [AGENT] ✅ Data reset complete:")
            print(f"    - ล้าง flags : {removed_keys}")
            print(f"    - CSV ใหม่  : {csv_path}")
            print(f"    - Experiments: reset to pending (baseline scores เก็บไว้)")
            print(f"    - Best เดิม : {state.get('current_best')} = {state.get('current_best_score')}")
            print()
            print("  ขั้นตอนถัดไป:")
            print(f"    python scripts/agent_next_step.py --state {state_path}")
        else:
            print("  [AGENT] [DRY-RUN] Would reset data pipeline")
            print(f"    Would remove: {removed_keys}")
            print(f"    Would set CSV: {csv_path}")
        return

    # ---- Mark flags ----
    if args.mark_quality_gate_done:
        state["quality_gate_done"] = True
        log_event(state, "quality_gate marked DONE")
        if not args.dry_run:
            save_state(state, state_path)
        return

    if args.mark_build_done:
        # input like "30_1" → key "build_w30_h1"
        parts = args.mark_build_done.replace("-","_").split("_")
        if len(parts) == 2:
            key = f"build_w{parts[0]}_h{parts[1]}"
        else:
            key = f"build_w{args.mark_build_done}"
        state[key] = True
        log_event(state, f"{key} marked DONE")
        if not args.dry_run:
            save_state(state, state_path)
        return

    if args.mark_inspect_done:
        eid = args.mark_inspect_done
        state.setdefault("experiments", {}).setdefault(eid, {})["inspect_done"] = True
        log_event(state, f"inspect[{eid}] marked DONE")
        if not args.dry_run:
            save_state(state, state_path)
        return

    if args.mark_training_done:
        eid = args.mark_training_done
        state.setdefault("experiments", {}).setdefault(eid, {})["status"] = "training_done"
        state["experiments"][eid]["train_finished_at"] = datetime.now().isoformat()
        log_event(state, f"train[{eid}] marked DONE")
        if not args.dry_run:
            save_state(state, state_path)
        return

    if args.mark_summarize_done:
        eid = args.mark_summarize_done
        exps = state.setdefault("experiments", {}).setdefault(eid, {})
        exps["summarize_done"] = True
        exps["status"] = "done"

        # อ่าน metrics จริง
        run_dir = Path(f"results/out_train_{eid}")
        score = read_best_metric(run_dir, state["goal"]["metric"])
        exps["best_score"] = score

        # อัปเดต global best
        if score is not None:
            old_best = state.get("current_best_score")
            if old_best is None or score > old_best:
                state["current_best"] = eid
                state["current_best_score"] = score
                log_event(state, f"NEW BEST: {eid} → {state['goal']['metric']}={score:.4f}")
            else:
                log_event(state, f"summarize[{eid}] done, score={score:.4f} (best still {state['current_best']}={old_best:.4f})")
        else:
            log_event(state, f"summarize[{eid}] done (metrics not found in {run_dir})")

        state["iteration"] = state.get("iteration", 0) + 1
        if not args.dry_run:
            save_state(state, state_path)
        return

    # ---- Decide next step ----
    action, exp = decide(state)

    print("\n" + "="*60)
    print(f"  AGENT DECISION  (iteration={state.get('iteration',0)})")
    print("="*60)
    print(f"  Action  : {action.upper()}")

    goal = state["goal"]
    best_score = state.get("current_best_score")
    best_exp   = state.get("current_best")
    if best_score is not None:
        gap = goal["target"] - best_score
        print(f"  Best    : {best_exp} → {goal['metric']}={best_score:.4f}  (gap to goal: {gap:+.4f})")
    else:
        print(f"  Best    : (none yet)")
    print(f"  Goal    : {goal['metric']} ≥ {goal['target']}")
    print()

    if action == "converged":
        print(f"  ✅ GOAL REACHED!  Best={best_score:.4f} ≥ target={goal['target']}")
        print(f"  Best experiment: {best_exp}")
        state["status"] = "converged"
        if not args.dry_run:
            save_state(state, state_path)
        return

    if action == "done":
        print("  ⚠️  Search space exhausted. No more experiments to run.")
        print(f"  Best achieved: {best_score}")
        state["status"] = "exhausted"
        if not args.dry_run:
            save_state(state, state_path)
        return

    if action == "quality_gate":
        cmd = (
            f"{PYTHON} scripts/quality_gate.py \\\n"
            f"  --input_csv data/env_daily_with_rice_monthly_raw.csv \\\n"
            f"  --out_dir results/out_quality_gate \\\n"
            f"  --mode soft --drop_rows_missing_mandatory"
        )
        print("  📋 COMMAND TO RUN:")
        print()
        print(cmd)
        print()
        print("  After success, run:")
        print(f"    python scripts/agent_next_step.py --state {args.state} --mark-quality-gate-done")
        return

    if exp:
        print(f"  Experiment : {exp['id']}")
        print(f"  Config     : W={exp['window']}, H={exp['horizon']}, feat={exp['feature_set']}, preset={exp['preset']}")
        print()

    if action == "build":
        w, h = exp["window"], exp["horizon"]
        cmd = make_build_cmd(exp)
        print("  📋 COMMAND TO RUN:")
        print()
        print(cmd)
        print()
        print("  After success, run:")
        print(f"    python scripts/agent_next_step.py --state {args.state} --mark-build-done {w}_{h}")

    elif action == "inspect":
        cmd = make_inspect_cmd(exp)
        print("  📋 COMMAND TO RUN:")
        print()
        print(cmd)
        print()
        print("  After success (nan=0, inf=0), run:")
        print(f"    python scripts/agent_next_step.py --state {args.state} --mark-inspect-done {exp['id']}")

    elif action == "train":
        cmd = make_train_cmd(exp)
        print("  📋 COMMAND TO RUN:")
        print()
        print(cmd)
        print()
        print("  After training finishes, run:")
        print(f"    python scripts/agent_next_step.py --state {args.state} --mark-training-done {exp['id']}")

    elif action == "summarize":
        cmd = make_summarize_cmd(exp)
        print("  📋 COMMAND TO RUN:")
        print()
        print(cmd)
        print()
        print("  After summarize finishes, run:")
        print(f"    python scripts/agent_next_step.py --state {args.state} --mark-summarize-done {exp['id']}")

    print("="*60 + "\n")


if __name__ == "__main__":
    main()
