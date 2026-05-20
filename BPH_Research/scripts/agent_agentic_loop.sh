#!/bin/bash
# =============================================================
# agent_agentic_loop.sh  —  BPH Layer 3: Agentic Auto Loop
# =============================================================
# ต่างจาก agent_loop.sh (Layer 2) ตรงที่:
#   - ใช้ agent_agentic.py (Layer 3 decision engine)
#   - หลัง summarize ทุกครั้ง Agent จะ re-evaluate phase
#     และอาจสร้าง dynamic experiments ใหม่อัตโนมัติ
#   - รองรับ peak_focus mode
#
# Usage:
#   chmod +x scripts/agent_agentic_loop.sh
#   ./scripts/agent_agentic_loop.sh
#   ./scripts/agent_agentic_loop.sh --max-iter 5 --dry-run
# =============================================================

set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORK_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$WORK_DIR"

PYTHON="/home/ai-station/my_project/.tf251p310/bin/python"
AGENT="scripts/agent_agentic.py"
STATE_FILE="results/agent_state_l3.json"
MAX_ITER=30
DRY_RUN=false
LOG_FILE="results/agent_agentic_loop.log"

while [[ $# -gt 0 ]]; do
  case $1 in
    --state)    STATE_FILE="$2"; shift 2 ;;
    --max-iter) MAX_ITER="$2";   shift 2 ;;
    --dry-run)  DRY_RUN=true;    shift   ;;
    *) echo "Unknown: $1"; exit 1 ;;
  esac
done

mkdir -p results
exec > >(tee -a "$LOG_FILE") 2>&1

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(timestamp)] $*"; }
sep() { echo ""; echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; }

get_action() {
  $PYTHON $AGENT --state "$STATE_FILE" 2>/dev/null \
    | grep "Action  :" | awk '{print $NF}' | tr '[:upper:]' '[:lower:]'
}

get_exp_id() {
  $PYTHON $AGENT --state "$STATE_FILE" 2>/dev/null \
    | grep "Experiment :" | awk '{print $NF}'
}

get_command() {
  $PYTHON $AGENT --state "$STATE_FILE" 2>/dev/null \
    | awk '/COMMAND:/{found=1;next} found && /^$/{if(cmd!="")exit} found{cmd=cmd"\n"$0} END{print cmd}' \
    | sed '/^[[:space:]]*$/d'
}

sep
log "🤖 BPH AGENTIC LOOP (Layer 3) START"
log "   State : $STATE_FILE"
log "   MaxIter: $MAX_ITER | DryRun: $DRY_RUN"
sep

ITER=0
while [ $ITER -lt $MAX_ITER ]; do
  ITER=$((ITER+1))
  sep
  log "🔄 Iteration $ITER / $MAX_ITER"

  ACTION=$(get_action)
  EXP_ID=$(get_exp_id)
  log "   Action=$ACTION  Exp=${EXP_ID:-n/a}"

  echo ""
  $PYTHON $AGENT --state "$STATE_FILE"
  echo ""

  if [[ "$ACTION" == "converged" || "$ACTION" == "done" ]]; then
    sep; log "✅ Loop complete: $ACTION"; sep
    exit 0
  fi

  CMD=$(get_command)
  [[ -z "$CMD" ]] && { log "❌ No command"; exit 1; }

  [[ "$DRY_RUN" == "true" ]] && { log "[DRY-RUN]"; break; }

  TRAIN_LOG="results/agentic_${EXP_ID:-unknown}_$(date +%H%M%S).log"

  case "$ACTION" in
    quality_gate)
      log "▶ quality_gate"
      eval "$CMD"
      $PYTHON $AGENT --state "$STATE_FILE" --mark-quality-gate-done
      ;;
    build)
      WH=$($PYTHON $AGENT --state "$STATE_FILE" 2>/dev/null \
        | grep "Config     :" | grep -oP '(?<=W=)\d+|(?<=H=)\d+' | paste -sd '_')
      log "▶ build NPZ (W/H=$WH)"
      eval "$CMD"
      $PYTHON $AGENT --state "$STATE_FILE" --mark-build-done "$WH"
      ;;
    inspect)
      log "▶ inspect NaN"
      eval "$CMD" && $PYTHON $AGENT --state "$STATE_FILE" --mark-inspect-done "$EXP_ID" \
        || log "❌ NaN found — rebuild needed"
      ;;
    train|train_peak)
      log "▶ training: $EXP_ID (${ACTION})"
      set +e
      eval "$CMD" 2>&1 | tee "$TRAIN_LOG"
      TRAIN_EXIT=${PIPESTATUS[0]}
      set -e
      if [[ $TRAIN_EXIT -eq 0 ]]; then
        $PYTHON $AGENT --state "$STATE_FILE" --mark-training-done "$EXP_ID"
      else
        log "❌ training FAILED ($EXP_ID) exit=$TRAIN_EXIT"
        $PYTHON - <<PYEOF
import json; from pathlib import Path
p=Path("$STATE_FILE"); s=json.loads(p.read_text())
s.setdefault("experiments",{}).setdefault("$EXP_ID",{})["status"]="failed"
p.write_text(json.dumps(s,indent=2,ensure_ascii=False))
PYEOF
      fi
      ;;
    summarize)
      log "▶ summarize: $EXP_ID"
      eval "$CMD" || log "   ⚠ summarize_runs exited non-zero (no metrics yet)"
      $PYTHON $AGENT --state "$STATE_FILE" --mark-summarize-done "$EXP_ID"
      log "   → Agent will re-evaluate phase and may add dynamic experiments"
      ;;
    *)
      log "❌ Unknown action: $ACTION"; exit 1 ;;
  esac

  sleep 1
done

sep
log "⏹  Loop ended after $ITER iterations"
log "   Resume: ./scripts/agent_agentic_loop.sh --state $STATE_FILE"
sep
