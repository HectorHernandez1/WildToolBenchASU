#!/bin/bash
# Run enhanced evaluation for all 4 models.
# Results go to result_v2/ and score_v2/ to keep baseline data untouched.
set -e

cd "$(dirname "$0")"

# Activate the wildtoolbench conda environment
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate wildtoolbench

MODELS=("qwen3:8b" "qwen3:14b" "qwen3:32b" "gemma4:31b")

echo "=== Enhanced WildToolBench Evaluation ==="
echo "Results will be written to result_v2/ and score_v2/"
echo ""

# Phase 1: Inference
for model in "${MODELS[@]}"; do
    echo "--- Running inference for $model ---"
    python -u -m wtb.openfunctions_evaluation \
        --model "$model" \
        --temperature 0.0 \
        --num-threads 1 \
        --result-dir result_v2 \
        --allow-overwrite
    echo "--- Inference complete for $model ---"
    echo ""
done

# Phase 2: Scoring
for model in "${MODELS[@]}"; do
    echo "--- Running scoring for $model ---"
    python -m wtb.eval_runner \
        --model "$model" \
        --result-dir result_v2 \
        --score-dir score_v2
    echo "--- Scoring complete for $model ---"
    echo ""
done

echo "=== All evaluations complete ==="
echo "Compare results:"
echo "  Baseline: score/{model}/Wild-Tool-Bench_metric.json"
echo "  Enhanced: score_v2/{model}/Wild-Tool-Bench_metric.json"
