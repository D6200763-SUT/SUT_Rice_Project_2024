#!/usr/bin/env python3
"""
agent_status.py  —  แสดงสถานะ Agentic Loop แบบ Dashboard
รัน: python scripts/agent_status.py [--state results/agent_state.json]
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

SEARCH_SPACE_IDS = [
    "w30_h1_context", "w30_h7_context", "w60_h7_context", "w60_h7_full",
    "w90_h14_context", "w90_h14_full", "w30_h7_context_quality",
    "w60_h7_context_quality", "w30_h1_full", "w45_h7_context",
]

STATUS_ICON = {
    "pending":       "⬜",
    "done":          "✅",
    "training_done": "🔄",
    "failed":        "❌",
    "running":       "▶️ ",
}

def fmt_score(v):
    return f"{v:.4f}" if v is not None else "—"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="results/agent_state.json")
    args = ap.parse_args()

    p = Path(args.state)
    if not p.exists():
        print("❌  State file not found:", p)
        print("    Run: python scripts/agent_next_step.py --state", args.state)
        return

    s = json.loads(p.read_text())
    exps = s.get("experiments", {})
    goal = s.get("goal", {})
    best = s.get("current_best")
    best_score = s.get("current_best_score")
    target = goal.get("target", 0.5)

    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║          🤖  BPH Agent Status Dashboard                  ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  State file : {p}")
    print(f"  Status     : {s.get('status','running')}")
    print(f"  Iteration  : {s.get('iteration', 0)}")
    print(f"  Goal       : {goal.get('metric','r2_log1p')} ≥ {target}")
    if best_score is not None:
        bar_len = 30
        filled = int((best_score / target) * bar_len) if target > 0 else 0
        filled = min(filled, bar_len)
        bar = "█" * filled + "░" * (bar_len - filled)
        pct = min(100, best_score / target * 100)
        print(f"  Progress   : [{bar}] {pct:.0f}%")
        print(f"  Best so far: {best}  →  {fmt_score(best_score)}")
    else:
        print("  Best so far: (none yet)")
    print()
    print("  Experiments:")
    print(f"  {'ID':<30} {'Status':<15} {'Score':>8}  {'Notes'}")
    print("  " + "─" * 65)

    for eid in SEARCH_SPACE_IDS:
        exp = exps.get(eid, {})
        status = exp.get("status", "pending")
        score = exp.get("best_score")
        icon = STATUS_ICON.get(status, "❓")
        flag = " ⭐" if eid == best else ""
        note = ""
        if status == "training_done":
            note = "(waiting summarize)"
        elif exp.get("inspect_done") and status == "pending":
            note = "(inspect done, ready to train)"
        print(f"  {icon} {eid:<28} {status:<15} {fmt_score(score):>8}{flag}  {note}")

    print()
    # Recent log
    logs = s.get("log", [])
    if logs:
        print("  Recent log (last 5):")
        for line in logs[-5:]:
            print(f"    {line}")
    print()
    print("  Commands:")
    print("    Next step : python scripts/agent_next_step.py")
    print("    Auto loop : ./scripts/agent_loop.sh")
    print("    Dashboard : python scripts/agent_status.py")
    print()

if __name__ == "__main__":
    main()
