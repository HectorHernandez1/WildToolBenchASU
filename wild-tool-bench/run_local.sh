#!/usr/bin/env bash
# run_local.sh — Run WildToolBench inference or evaluation with local Ollama models.
#
# Usage:
#   bash run_local.sh infer [MODEL] [EXTRA_ARGS...]
#   bash run_local.sh eval  [MODEL] [EXTRA_ARGS...]
#
# Examples:
#   bash run_local.sh infer qwen3:14b --run-ids          # inference on test subset
#   bash run_local.sh infer qwen3:32b                    # inference on full benchmark
#   bash run_local.sh eval  qwen3:14b                    # evaluate results

set -euo pipefail
cd "$(dirname "$0")"

# Use the wildtoolbench conda env if available, otherwise fall back to system python
PYTHON="python3"
CONDA_ENV_PYTHON="$HOME/miniconda3/envs/wildtoolbench/bin/python"
if [ -x "$CONDA_ENV_PYTHON" ]; then
    PYTHON="$CONDA_ENV_PYTHON"
fi

MODE="${1:?Usage: bash run_local.sh <infer|eval> [MODEL] [EXTRA_ARGS...]}"
MODEL="${2:-qwen3:14b}"
shift 2 2>/dev/null || shift 1 2>/dev/null || true

case "$MODE" in
  infer)
    echo "==> Running inference with model: $MODEL"
    $PYTHON -u -m wtb.openfunctions_evaluation --model="$MODEL" "$@"
    ;;
  eval)
    echo "==> Running evaluation for model: $MODEL"
    $PYTHON -u -m wtb.eval_runner --model="$MODEL" "$@"
    ;;
  *)
    echo "Unknown mode: $MODE"
    echo "Usage: bash run_local.sh <infer|eval> [MODEL] [EXTRA_ARGS...]"
    exit 1
    ;;
esac
