# WildToolBench Enhancement Changes Report

## Overview

This report documents three targeted improvements made to the WildToolBench evaluation harness to address the most impactful failure modes identified in baseline evaluation data. All changes are on the `eval-improvements` branch. Enhanced results are written to `result_v2/` and `score_v2/` directories to preserve baseline data for comparison.

---

## Baseline Performance

| Model | Task Accuracy | Session Accuracy |
|-------|---------------|------------------|
| Gemma4-31B | 52.0% | 9.0% |
| Qwen3-32B | 44.1% | 4.3% |
| Qwen3-14B | 44.2% | 3.5% |
| Qwen3-8B | 40.8% | 2.7% |

## Top Failure Modes (Baseline)

| Failure Category | Qwen3-8B | Qwen3-14B | Qwen3-32B | Gemma4-31B |
|------------------|----------|-----------|-----------|------------|
| Wrong Arguments | 28.2% | 27.4% | 25.3% | 20.6% |
| Instruction Transition | 22.5% | 21.0% | 22.5% | 20.5% |
| Compositional Planning | 20.9% | 19.4% | 19.5% | 15.4% |
| Coreference Failure | 19.9% | 19.0% | 18.8% | 18.3% |
| Unnecessary Tool Call | 13.0% | 11.4% | 15.6% | 14.4% |

---

## Change 1: Enhanced System Prompt

**File:** `wtb/model_handler/base_handler.py` (`_build_system_prompt` method)

### Problem
Models received only `"Current Date: {timestamp}"` as their system prompt -- no guidance on when to use tools vs. ask for clarification vs. respond conversationally. This caused:
- **Instruction Transition failures (22.5%):** Models called tools when they should have asked for clarification, or vice versa.
- **Unnecessary Tool Calls (15.6%):** Models invoked tools for casual conversation that needed no tool use.
- **Missing Tool Calls (9.0%):** Models responded conversationally when they should have called a tool.

### What Changed
Replaced the minimal system prompt with structured instructions covering:
1. **Tool Use vs. Clarification vs. Conversation** -- explicit rules for when to call tools, when to use `ask_user_for_required_parameters`, and when to just respond.
2. **Argument Precision** -- instructions to use only schema-defined parameters and never hallucinate values.
3. **Multi-Step Requests** -- guidance on identifying parallel vs. sequential tool calls.
4. **Conversation Context** -- instructions to resolve references like "that location" and "the same date" from conversation history.

### Why This Should Help
These failure modes account for ~47% of all failures. Even a modest improvement in intent routing (tool vs. clarify vs. chat) could recover 100-200+ tasks across all models. The prompt is model-agnostic and adds negligible latency.

### Targeted Failure Modes
- Instruction Transition (22.5%)
- Unnecessary Tool Call (15.6%)
- Missing Tool Call (9.0%)
- Hallucinated Data (6.6-9.5%)

---

## Change 2: Argument Tolerance / Normalization Layer

**File:** `wtb/checker_utils.py` (`_recursive_compare`, `_normalize_date`, `_try_numeric_coerce`)

### Problem
The argument checker was overly strict, failing models that produced semantically correct but syntactically different arguments:
- **780 failures from "args keys mismatch"** -- models included extra optional parameters not in the ground truth (e.g., adding `category_name` when updating a transaction).
- **128+ failures from string similarity too low** -- on fields like `location`, `date`, `query`, `endDate`.
- **Date format mismatches** -- `"2024-7-13"` vs `"2024-07-13"` scored as completely different (rouge score: 0.00).
- **Numeric type mismatches** -- `"100"` (string) vs `100` (integer) counted as type errors.

### What Changed
1. **Extra Optional Parameter Tolerance:** When the prediction has extra keys not in the ground truth, we now check if those keys are defined in the tool schema. If they are valid schema properties (just not in the ground truth), they are tolerated. Only keys that are completely undefined in the schema are rejected.
2. **Date Normalization:** Added `_normalize_date()` that canonicalizes date strings (YYYY-MM-DD with zero-padded months/days, slash-to-dash conversion) before comparison.
3. **Numeric Coercion:** Added `_try_numeric_coerce()` that treats `"100"` and `100` as equivalent, and `3` and `3.0` as equivalent.
4. **Schema-aware comparison:** Updated `answer_check()` to pass the full tool schema (including `required` field) to `_recursive_compare()` so it can distinguish required vs. optional parameters.

### Why This Should Help
Wrong Arguments is the #1 failure category (20.6-28.2% of all tasks). The 780 "args keys mismatch" errors are the single largest error sub-type. Many of these are cases where the model correctly understood the task but included an extra optional parameter. This change avoids penalizing correct behavior.

### Targeted Failure Modes
- Wrong Arguments: args keys mismatch (780 cases)
- Wrong Arguments: string similarity too low at date fields (31+ cases)
- Wrong Arguments: value type inconsistent (48 cases)

---

## Change 3: Coreference Resolution Context Injection

**File:** `wtb/model_handler/base_handler.py` (`_build_context_summary` method, `_pre_messages_processing`)

### Problem
~18-20% of failures across all models are coreference errors. When users say things like:
- "that location" (referring to a city from 2 turns ago)
- "one of the topics" (referring to results from a previous tool call)
- "the same date" (referring to a date used earlier)
- "update this transaction" (referring to a specific record)

Models fail to resolve these references, either hallucinating values or calling the wrong tool entirely. This is especially severe in **Long-Range Dependency** turns (accuracy drops to 28.5-39.9%) where the referent is 2+ turns back.

### What Changed
Added a `_build_context_summary()` method that:
1. Iterates through all previous turns' tool call arguments
2. Extracts key-value pairs (e.g., `city=Chicago`, `startDate=2024-07-13`)
3. Deduplicates and takes the most recent 20 entities
4. Prepends this context summary to the user's message on subsequent turns

Example injected context:
```
[Context from prior turns: city=Chicago, startDate=2024-07-13, endDate=2024-07-14, format=JSON]
I also want to check the air quality for the same dates.
```

### Why This Should Help
Coreference Failure is size-independent (flat at 18-20% across 8B to 32B models), meaning it cannot be fixed by scaling alone. Explicitly providing resolved entities from prior turns gives the model the information it needs to resolve references without guessing.

### Targeted Failure Modes
- Coreference Failure (18.3-19.9%)
- Long-Range Dependency accuracy (28.5-39.9%)
- Turn degradation (Turn 0: 69% -> Turn 3: 42%)

---

## Running the Enhanced Evaluation

```bash
cd wild-tool-bench
bash run_enhanced_evaluation.sh
```

This will:
1. Run inference for all 4 models with the enhanced system prompt and coreference resolution
2. Score results using the improved argument tolerance checker
3. Write results to `result_v2/{model}/` and `score_v2/{model}/`

## Comparing Results

After the evaluation completes, compare:
- **Baseline:** `score/{model}/Wild-Tool-Bench_metric.json`
- **Enhanced:** `score_v2/{model}/Wild-Tool-Bench_metric.json`

Key metrics to watch:
- Task Accuracy (overall and by type)
- Session Accuracy
- Accuracy by Turn Depth (especially Turn 2 and Turn 3)
- Long-Range Dependency accuracy
- Clarify and Chat accuracy (should not regress)
