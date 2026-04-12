# WildToolBench

Benchmark for evaluating LLM tool-use capabilities in real-world multi-turn scenarios. 256 sessions, 1024 tasks, 4 turns per session.

## Branch: eval-improvements

Three enhancements to improve model scores, targeting the top failure modes from baseline evaluation:

1. **Enhanced System Prompt** (`wtb/model_handler/base_handler.py`) -- Adds structured instructions for tool use vs. clarification vs. conversation, argument precision, multi-step planning, and context resolution. Targets Instruction Transition (22.5%) and Unnecessary Tool Call (15.6%) failures.

2. **Argument Tolerance Layer** (`wtb/checker_utils.py`) -- Tolerates extra optional parameters defined in schema, normalizes dates, and coerces numeric types. Targets Wrong Arguments (28.2%) failures, especially the 780 "args keys mismatch" cases.

3. **Coreference Resolution** (`wtb/model_handler/base_handler.py`) -- Injects entity context from prior turns to help models resolve references like "that location" and "the same date". Targets Coreference Failure (19.9%) and Long-Range Dependency degradation.

## Running Enhanced Evaluation

```bash
cd wild-tool-bench
bash run_enhanced_evaluation.sh
```

Results go to `result_v2/` and `score_v2/` (baseline stays in `result/` and `score/`).

## Key Files

- `wild-tool-bench/wtb/model_handler/base_handler.py` -- Inference message construction
- `wild-tool-bench/wtb/checker_utils.py` -- Argument validation and scoring
- `wild-tool-bench/wtb/eval_runner.py` -- Evaluation orchestrator
- `wild-tool-bench/wtb/_llm_response_generation.py` -- Model inference pipeline
- `wild-tool-bench/Enhancements/analysis_report.md` -- Baseline failure analysis
- `wild-tool-bench/Enhancements/enhancement_changes_report.md` -- Enhancement details and rationale
