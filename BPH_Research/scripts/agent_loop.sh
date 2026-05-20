#!/bin/bash
# =============================================================
# agent_loop.sh  —  BPH Agentic Loop Runner (Layer 3)
# =============================================================
# วนซ้ำอัตโนมัติ: decide → run → mark → repeat จนถึงเป้าหมาย
#
# Usage:
#   chmod +x scripts/agent_loop.sh
#   ./scripts/agent_loop.sh                          # รันจน converge
#   ./scripts/agent_loop.sh --max-iter 3             # จำกัด 3 รอบ
#   ./scripts/agent_loop.sh --state results/my.json  # custom state file
#   ./scripts/agent_loop.sh --dry-run                # แสดงแต่ไม่รันจริง
# =============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORK_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$WORK_DIR"

PYTHON="/home/ai-station/my_project/.tf251p310/bin/python"
STATE_FILE="results/agent_state.json"
MAX_ITER=20
DRY_RUN=false
LOG_FILE="results/agent_loop.log"

# ---- Parse args ----
while [[ $# -gt 0 ]]; do
  case $1 in
    --state)     STATE_FILE="$2"; shift 2 ;;
    --max-iter)  MAX_ITER="$2";   shift 2 ;;
    --dry-run)   DRY_RUN=true;    shift   ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

mkdir -p results
exec > >(tee -a "$LOG_FILE") 2>&1

# ---- Helpers ----
timestamp() { date "+%Y-%m-%d %H:%M:%S"; }
log()  { echo "[$(timestamp)] $*"; }
sep()  { echo ""; echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; }

agent_decide() {
  $PYTHON scripts/agent_next_step.py --state "$STATE_FILE"
}

# ---- Extract next action from agent output ----
get_action() {
  $PYTHON scripts/agent_next_step.py --state "$STATE_FILE" 2>/dev/null \
    | grep "Action  :" | awk '{print $NF}' | tr '[:upper:]' '[:lower:]'
}

get_exp_id() {
  $PYTHON scripts/agent_next_step.py --state "$STATE_FILE" 2>/dev/null \
    | grep "Experiment :" | awk '{print $NF}'
}

get_window_horizon() {
  $PYTHON scripts/agent_next_step.py --state "$STATE_FILE" 2>/dev/null \
    | grep "Config     :" | grep -oP 'W=\K[0-9]+' | head -1
}

# ---- Extract the actual bash command block from agent output ----
get_command() {
  $PYTHON scripts/agent_next_step.py --state "$STATE_FILE" 2>/dev/null \
    | awk '/COMMAND TO RUN/{found=1; next} found && /^$/{if(cmd!="") exit} found{cmd=cmd"\n"$0} END{print cmd}' \
    | sed '/^[[:space:]]*$/d'
}

# ============================================================
# Main Loop
# ============================================================
sep
log "🤖  BPH Agentic Loop START"
log "    State file : $STATE_FILE"
log "    Max iter   : $MAX_ITER"
log "    Dry run    : $DRY_RUN"
sep

ITER=0

while [ $ITER -lt $MAX_ITER ]; do
  ITER=$((ITER + 1))
  sep
  log "🔄  Iteration $ITER / $MAX_ITER"
  sep

  # ---- Get decision ----
  ACTION=$(get_action)
  EXP_ID=$(get_exp_id)
  log "    Action  : $ACTION"
  log "    Exp ID  : ${EXP_ID:-n/a}"

  # Show full decision
  echo ""
  agent_decide
  echo ""

  # ---- Handle terminal states ----
  if [[ "$ACTION" == "converged" ]]; then
    sep
    log "✅  CONVERGED! Goal reached. Loop complete."
    sep
    exit 0
  fi

  if [[ "$ACTION" == "done" ]]; then
    sep
    log "⚠️  Search space exhausted. Stopping."
    sep
    exit 0
  fi

  # ---- Extract command ----
  CMD=$(get_command)
  if [[ -z "$CMD" ]]; then
    log "❌  Could not extract command. Stopping."
    exit 1
  fi

  log "    Command preview:"
  echo "$CMD" | head -5

  if [[ "$DRY_RUN" == "true" ]]; then
    log "    [DRY RUN] Skipping execution."
    break
  fi

  # ============================================================
  # Execute by action type
  # ============================================================

  if [[ "$ACTION" == "quality_gate" ]]; then
    log "▶️  Running quality_gate..."
    eval "$CMD"
    log "✔️  quality_gate done"
    $PYTHON scripts/agent_next_step.py --state "$STATE_FILE" --mark-quality-gate-done

  elif [[ "$ACTION" == "build" ]]; then
    # หา W และ H จาก exp config
    WH=$($PYTHON scripts/agent_next_step.py --state "$STATE_FILE" 2>/dev/null \
      | grep "Config     :" \
      | grep -oP '(?<=W=)\d+|(?<=H=)\d+' \
      | paste -sd '_')
    log "▶️  Building NPZ sequences (W/H=$WH)..."
    eval "$CMD"
    log "✔️  build done"
    $PYTHON scripts/agent_next_step.py --state "$STATE_FILE" --mark-build-done "$WH"

  elif [[ "$ACTION" == "inspect" ]]; then
    log "▶️  Inspecting NPZ for NaN/Inf..."
    if eval "$CMD"; then
      log "✔️  inspect passed (no NaN/Inf)"
      $PYTHON scripts/agent_next_step.py --state "$STATE_FILE" --mark-inspect-done "$EXP_ID"
    else
      log "❌  NaN/Inf found in NPZ! Need to rebuild."
      # Force rebuild by clearing build_done flag
      $PYTHON - <<PYEOF
import json
from pathlib import Path
p = Path("$STATE_FILE")
s = json.loads(p.read_text())
# ดึง W,H จาก EXP_ID เช่น w60_h7_context → 60_7
parts = "$EXP_ID".split("_")
key = f"build_{parts[0]}_{parts[1]}"
s.pop(key, None)
p.write_text(json.dumps(s, indent=2, ensure_ascii=False))
print(f"Cleared {key} from state")
PYEOF
    fi

  elif [[ "$ACTION" == "train" ]]; then
    log "▶️  Training models (this may take a while)..."
    log "    Experiment: $EXP_ID"

    # รันแบบ foreground พร้อม log
    TRAIN_LOG="results/train_${EXP_ID}.log"
    if eval "$CMD" 2>&1 | tee "$TRAIN_LOG"; then
      log "✔️  training done → $TRAIN_LOG"
      $PYTHON scripts/agent_next_step.py --state "$STATE_FILE" --mark-training-done "$EXP_ID"
    else
      log "❌  training FAILED for $EXP_ID (see $TRAIN_LOG)"
      # Mark as failed แต่ไม่หยุด loop
      $PYTHON - <<PYEOF
import json
from pathlib import Path
p = Path("$STATE_FILE")
s = json.loads(p.read_text())
s.setdefault("experiments", {}).setdefault("$EXP_ID", {})["status"] = "failed"
p.write_text(json.dumps(s, indent=2, ensure_ascii=False))
print("Marked $EXP_ID as failed")
PYEOF
    fi

  elif [[ "$ACTION" == "summarize" ]]; then
    log "▶️  Summarizing results for $EXP_ID..."
    if eval "$CMD"; then
      log "✔️  summarize done"
      $PYTHON scripts/agent_next_step.py --state "$STATE_FILE" --mark-summarize-done "$EXP_ID"
    else
      log "❌  summarize FAILED"
    fi

  else
    log "❌  Unknown action: $ACTION"
    exit 1
  fi

  log "    State saved → $STATE_FILE"
  sleep 1  # เว้นช่องว่างระหว่าง iteration

done

sep
log "⏹️  Loop ended after $ITER iterations (max=$MAX_ITER)"
log "    Run again to continue: ./scripts/agent_loop.sh"
sep
