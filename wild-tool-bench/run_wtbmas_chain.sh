#!/usr/bin/env bash
# run_wtbmas_chain.sh — Sequentially run WTB-MAS inference on multiple backbones,
#                       then evaluate each one. Designed for multi-day unattended runs.
#
# Usage:
#   bash run_wtbmas_chain.sh                     # default 4-model chain
#   bash run_wtbmas_chain.sh qwen3:8b qwen3:14b  # custom subset
#
# Behavior:
#   - Runs each backbone via run_local.sh in sequence (no parallel — one model at a time
#     keeps VRAM clean and the run reproducible)
#   - Logs each run to logs/wtbmas_<backbone>_<timestamp>.log
#   - Writes a STATUS file you can `tail -F` from elsewhere to watch overall progress
#   - SKIPS a backbone if its result file already has all 256 sessions (resume support)
#   - On per-backbone failure: records the failure, moves to the next backbone instead
#     of aborting the whole chain
#
# Recommended: run inside tmux/screen so terminal closure doesn't kill it.
#   tmux new -s wtbmas
#   bash run_wtbmas_chain.sh
#   # Ctrl-b d to detach;  tmux attach -t wtbmas to come back

set -uo pipefail
cd "$(dirname "$0")"

# ── Models to run (in order) ─────────────────────────────────────
if [ "$#" -gt 0 ]; then
    BACKBONES=("$@")
else
    BACKBONES=(qwen3:8b qwen3:14b qwen3:32b gemma4:31b)
fi

LOG_DIR="logs"
mkdir -p "$LOG_DIR"
STATUS_FILE="$LOG_DIR/wtbmas_chain_status.txt"

# Restore the original test_case_ids_to_generate.json if a sanity run modified it
ORIG_IDS_BACKUP="test_case_ids_to_generate.json.orig"
if [ -f "$ORIG_IDS_BACKUP" ]; then
    cp "$ORIG_IDS_BACKUP" test_case_ids_to_generate.json
fi

# ── Helpers ─────────────────────────────────────────────────────
EXPECTED_SESSIONS=256

count_sessions() {
    local model="$1"
    local rfile="result/${model}/Wild-Tool-Bench_result.jsonl"
    if [ -f "$rfile" ]; then
        wc -l < "$rfile" | tr -d ' '
    else
        echo 0
    fi
}

write_status() {
    local msg="$1"
    {
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] $msg"
    } | tee -a "$STATUS_FILE"
}

write_status "=== WTB-MAS chain starting; backbones: ${BACKBONES[*]} ==="
START_TIME=$(date +%s)

# ── Main loop ───────────────────────────────────────────────────
for backbone in "${BACKBONES[@]}"; do
    model="wtbmas:${backbone}"
    log_file="$LOG_DIR/wtbmas_${backbone//:/_}_$(date +%Y%m%d_%H%M%S).log"

    # Resume: skip if already complete
    have=$(count_sessions "$model")
    if [ "$have" -ge "$EXPECTED_SESSIONS" ]; then
        write_status "[$model] SKIP — already has $have/$EXPECTED_SESSIONS sessions"
        continue
    fi

    write_status "[$model] START — currently $have/$EXPECTED_SESSIONS sessions"
    bb_start=$(date +%s)

    if bash run_local.sh infer "$model" 2>&1 | tee "$log_file"; then
        infer_status="ok"
    else
        infer_status="failed (exit=$?)"
    fi

    have=$(count_sessions "$model")
    bb_elapsed=$(( $(date +%s) - bb_start ))
    bb_h=$(printf "%.1f" "$(echo "$bb_elapsed/3600" | bc -l)")
    write_status "[$model] inference $infer_status  ($have/$EXPECTED_SESSIONS sessions, ${bb_h}h elapsed)"

    # Always score whatever we have (even if partial) — gives us SOMETHING for the presentation
    if [ "$have" -gt 0 ]; then
        score_log="$LOG_DIR/wtbmas_${backbone//:/_}_score_$(date +%Y%m%d_%H%M%S).log"
        if bash run_local.sh eval "$model" 2>&1 | tee "$score_log"; then
            write_status "[$model] eval ok"
        else
            write_status "[$model] eval FAILED (exit=$?)"
        fi
    else
        write_status "[$model] eval SKIPPED — no result file"
    fi
done

ELAPSED=$(( $(date +%s) - START_TIME ))
H=$(printf "%.1f" "$(echo "$ELAPSED/3600" | bc -l)")
write_status "=== chain complete after ${H}h ==="
