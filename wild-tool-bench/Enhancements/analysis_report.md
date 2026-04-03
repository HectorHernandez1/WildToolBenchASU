# WildToolBench Comprehensive Analysis Report

## 1. Executive Summary

- **Gemma4-31B is the clear winner** with 52.0% task accuracy and 9.0% session accuracy — significantly outperforming all three Qwen3 variants (40.8%–44.2% task, 2.7%–4.3% session).
- **No model exceeds 10% session accuracy**, confirming WildToolBench's difficulty. The benchmark requires perfection across all 4 turns, making session accuracy extremely punishing.
- **The most common failure mode is "Wrong Arguments"** (1040 occurrences across all models), followed by "Instruction Transition" (885).
- **Scaling within the Qwen3 family yields diminishing returns**: going from 8B→32B improves task accuracy by only 3.3 percentage points, while the architecture switch to Gemma4 gains 7.9pp over the best Qwen3.
- **394 out of 1024 tasks (38.5%) are failed by ALL models** — these represent universally hard tasks that likely require architectural interventions, not just scale.

## 2. Results Summary Table

### Overall Accuracy

| Metric | Qwen3-8B | Qwen3-14B | Qwen3-32B | Gemma4-31B |
|--------|----------|-----------|-----------|------------|
| Task Accuracy | 40.8% | 44.2% | 44.1% | 52.0% |
| Session Accuracy | 2.7% | 3.5% | 4.3% | 9.0% |

### Accuracy by Task Type

| Task Type | Qwen3-8B | Qwen3-14B | Qwen3-32B | Gemma4-31B |
|-----------|----------|-----------|-----------|------------|
| Single-Tool | 36.7% | 39.8% | 45.2% | 52.2% |
| Parallel Multi-Tool | 24.4% | 30.1% | 29.7% | 49.7% |
| Sequential Multi-Tool | 6.2% | 13.3% | 12.5% | 18.8% |
| Mixed Multi-Tool | 3.6% | 7.2% | 7.2% | 19.3% |
| Clarify | 23.0% | 27.5% | 31.1% | 37.3% |
| Chat | 87.1% | 88.1% | 78.5% | 80.5% |

### Accuracy by Turn Depth

| Turn | Qwen3-8B | Qwen3-14B | Qwen3-32B | Gemma4-31B |
|------|----------|-----------|-----------|------------|
| Turn 0 | 49.2% | 51.4% | 50.0% | 69.0% |
| Turn 1 | 45.7% | 46.2% | 50.0% | 52.5% |
| Turn 2 | 34.8% | 41.9% | 41.7% | 44.7% |
| Turn 3 | 33.6% | 37.2% | 34.6% | 41.6% |

### Accuracy by Turn Subtype (Implicit Intent)

| Subtype | Qwen3-8B | Qwen3-14B | Qwen3-32B | Gemma4-31B |
|---------|----------|-----------|-----------|------------|
| First Turn | 49.2% | 51.4% | 50.0% | 69.0% |
| Coreferential Reference | 41.5% | 43.1% | 44.7% | 46.1% |
| Partial Information | 40.0% | 47.1% | 47.5% | 51.2% |
| Long-Range Dependency | 28.5% | 32.0% | 29.9% | 39.9% |

## 3. Failure Mode Breakdown

Each failed task is classified into one or more of 9 failure categories. Categories overlap — a single failure can be both "Wrong Arguments" and "Coreference Failure" if it occurred on a coreferential reference turn.

### Primary Failure Categories (count and % of total tasks per model)

| Failure Category | Qwen3-8B | Qwen3-14B | Qwen3-32B | Gemma4-31B |
|------------------|----------|----------|----------|----------|
| Wrong Tool | 96 (9.4%) | 75 (7.3%) | 91 (8.9%) | 87 (8.5%) |
| Wrong Arguments | 289 (28.2%) | 281 (27.4%) | 259 (25.3%) | 211 (20.6%) |
| Missing Tool Call | 88 (8.6%) | 92 (9.0%) | 58 (5.7%) | 45 (4.4%) |
| Unnecessary Tool Call | 133 (13.0%) | 117 (11.4%) | 160 (15.6%) | 147 (14.4%) |
| Coreference Failure | 204 (19.9%) | 195 (19.0%) | 192 (18.8%) | 187 (18.3%) |
| Compositional Planning | 214 (20.9%) | 199 (19.4%) | 200 (19.5%) | 158 (15.4%) |
| Instruction Transition | 230 (22.5%) | 215 (21.0%) | 230 (22.5%) | 210 (20.5%) |
| Hallucinated Data | 68 (6.6%) | 68 (6.6%) | 83 (8.1%) | 97 (9.5%) |
| Format/Parse Error | 4 (0.4%) | 3 (0.3%) | 4 (0.4%) | 3 (0.3%) |
| Timeout | 0 (0.0%) | 12 (1.2%) | 8 (0.8%) | 4 (0.4%) |

### Argument Error Sub-types (aggregated across all models)

| Error Sub-type | Count |
|----------------|-------|
| error_match: args keys mismatch | 780 |
| error_schema: nested args type inconsistent | 48 |
| error_match: string similarity too low at 'location' | 33 |
| error_match: value mismatch at 'year' | 25 |
| error_schema: nested args not defined | 21 |
| error_match: string similarity too low at 'date' | 20 |
| error_match: string similarity too low at 'query' | 16 |
| error_match: string similarity too low at 'url' | 15 |
| error_schema: nested args value not in enum | 14 |
| error_match: string similarity too low at 'keywords' | 14 |
| error_match: string similarity too low at 'author' | 14 |
| error_match: string similarity too low at 'endDate' | 11 |
| error_match: string similarity too low at 'symbol_id' | 11 |
| error_match: string similarity too low at 'language' | 10 |
| error_match: string similarity too low at 'address' | 10 |

## 4. Example Failure Snippets

### Wrong Tool

**Example 1** — Gemma4-31B, wild_tool_bench_0 Turn 2 (Mixed Multi-Tool, Partial Information)

- **User message:** "Check the air quality as well. If it's good, then check what special performances are on those two days."
- **Expected:** `getCityAQIForecast, getCityShow, getShowDetail, getShowDetail, prepare_to_answer`
- **Predicted:** `getCityShow`
- **Error:** action name not in candidate_answer_function_list

**Example 2** — Gemma4-31B, wild_tool_bench_4 Turn 1 (Single-Tool, Partial Information)

- **User message:** "ok, then proceed with email verification"
- **Expected:** `emailLogin, prepare_to_answer`
- **Predicted:** `twoFactorAuthentication`
- **Error:** action name not in candidate_answer_function_list

### Wrong Arguments

**Example 1** — Gemma4-31B, wild_tool_bench_1 Turn 0 (Single-Tool, First Turn)

- **User message:** "I would like to know if there have been any debates on environmental protection in the Swiss parliament in the past two months. Could you help me check?"
- **Expected:** `getParliamentDebates, prepare_to_answer`
- **Predicted:** `getParliamentDebates`
- **Error:** error_match: string similarity too low at 'dateRange.endDate' (rouge score: 0.00)

**Example 2** — Gemma4-31B, wild_tool_bench_2 Turn 3 (Single-Tool, Coreferential Reference)

- **User message:** "Please help me update this transaction record. Change the amount to 250 yuan and keep the other information unchanged."
- **Expected:** `updateTransaction, prepare_to_answer`
- **Predicted:** `updateTransaction`
- **Error:** error_match: args keys mismatch (extra: ['category_name'])

### Missing Tool Call

**Example 1** — Gemma4-31B, wild_tool_bench_2 Turn 1 (Single-Tool, Partial Information)

- **User message:** "Create another new transaction record with an amount of 200 yuan, a date of 2024-08-01, and classify it as 'Entertainment Expenses'."
- **Expected:** `createTransaction, prepare_to_answer`
- **Predicted:** `Text: I'd be happy to help you create that transaction. However, I need the **Account ID** to know which account the transaction should be recorded in. Could you please provide the Account ID?...`
- **Error:** action name not in candidate_answer_function_list

**Example 2** — Gemma4-31B, wild_tool_bench_16 Turn 2 (Clarify, Partial Information)

- **User message:** "I also want to know if there is any mention of other content"
- **Expected:** `ask_user_for_required_parameters, ask_user_for_required_parameters, streamTweets, prepare_to_answer`
- **Predicted:** `Text: It looks like you've mentioned two of the biggest names in the world of Artificial Intelligence. Since your message was brief, I'm not sure exactly what you're looking for, but here is a quick overvie...`
- **Error:** action name not in candidate_answer_function_list

### Unnecessary Tool Call

**Example 1** — Gemma4-31B, wild_tool_bench_1 Turn 3 (Clarify, Coreferential Reference)

- **User message:** "I want to know what courses related to one of the topics are offered in schools?"
- **Expected:** `ask_user_for_required_parameters, ask_user_for_required_parameters, getSchoolsAndUniversitiesInfo, prepare_to_answer`
- **Predicted:** `getSchoolsAndUniversitiesInfo, getSchoolsAndUniversitiesInfo`
- **Error:** action name not in candidate_answer_function_list

**Example 2** — Gemma4-31B, wild_tool_bench_3 Turn 2 (Clarify, Coreferential Reference)

- **User message:** "I have a visitor who is from one of the above IP addresses. I want to know where he is from."
- **Expected:** `ask_user_for_required_parameters, getLocationByIP, prepare_to_answer`
- **Predicted:** `getLocationByIP, getLocationByIP, getLocationByIP`
- **Error:** action name not in candidate_answer_function_list

### Coreference Failure

**Example 1** — Gemma4-31B, wild_tool_bench_1 Turn 3 (Clarify, Coreferential Reference)

- **User message:** "I want to know what courses related to one of the topics are offered in schools?"
- **Expected:** `ask_user_for_required_parameters, ask_user_for_required_parameters, getSchoolsAndUniversitiesInfo, prepare_to_answer`
- **Predicted:** `getSchoolsAndUniversitiesInfo, getSchoolsAndUniversitiesInfo`
- **Error:** action name not in candidate_answer_function_list

**Example 2** — Gemma4-31B, wild_tool_bench_2 Turn 3 (Single-Tool, Coreferential Reference)

- **User message:** "Please help me update this transaction record. Change the amount to 250 yuan and keep the other information unchanged."
- **Expected:** `updateTransaction, prepare_to_answer`
- **Predicted:** `updateTransaction`
- **Error:** error_match: args keys mismatch (extra: ['category_name'])

### Compositional Planning

**Example 1** — Gemma4-31B, wild_tool_bench_0 Turn 2 (Mixed Multi-Tool, Partial Information)

- **User message:** "Check the air quality as well. If it's good, then check what special performances are on those two days."
- **Expected:** `getCityAQIForecast, getCityShow, getShowDetail, getShowDetail, prepare_to_answer`
- **Predicted:** `getCityShow`
- **Error:** action name not in candidate_answer_function_list

**Example 2** — Gemma4-31B, wild_tool_bench_7 Turn 3 (Parallel Multi-Tool, Long-Range Dependency)

- **User message:** "I also want to know the information about an even number of provinces in the results?"
- **Expected:** `getDivisionDetails, getDivisionDetails, getDivisionDetails, getDivisionDetails, getDivisionDetails, getDivisionDetails, prepare_to_answer`
- **Predicted:** `getDivisionDetails, getDivisionDetails, getDivisionDetails, getDivisionDetails, getDivisionDetails, getDivisionDetails`
- **Error:** error_match: string similarity too low at 'divisionCode' (edit score: 0.50); error_match: string similarity too low at 'divisionCode' (edit score: 0.50); error_match: string similarity too low at 'divisionCode' (edit score: 0.50)

## 5. Cross-Model Comparison

### Task Agreement Analysis

- **Tasks ALL models solve:** 295 (28.8%)
- **Tasks NO model solves:** 394 (38.5%)
- **Discriminating tasks** (some models pass, some fail): 335 (32.7%)

### Size-Scaling Analysis (Qwen3 Family: 8B → 14B → 32B)

| Failure Category | 8B Rate | 14B Rate | 32B Rate | Trend |
|------------------|---------|----------|----------|-------|
| Wrong Tool | 9.4% | 7.3% | 8.9% | non-monotonic |
| Wrong Arguments | 28.2% | 27.4% | 25.3% | size-dependent (decreases) |
| Missing Tool Call | 8.6% | 9.0% | 5.7% | non-monotonic |
| Unnecessary Tool Call | 13.0% | 11.4% | 15.6% | non-monotonic |
| Coreference Failure | 19.9% | 19.0% | 18.8% | flat (size-independent) |
| Compositional Planning | 20.9% | 19.4% | 19.5% | flat (size-independent) |
| Instruction Transition | 22.5% | 21.0% | 22.5% | flat (size-independent) |
| Hallucinated Data | 6.6% | 6.6% | 8.1% | flat (size-independent) |
| Format/Parse Error | 0.4% | 0.3% | 0.4% | flat (size-independent) |

### Architecture Comparison: Gemma4 vs Best Qwen3

| Failure Category | Gemma4 Rate | Best Qwen3 Rate | Delta |
|------------------|-------------|-----------------|-------|
| Wrong Tool | 8.5% | 7.3% | +1.2pp (Qwen better) |
| Wrong Arguments | 20.6% | 25.3% | -4.7pp (Gemma better) |
| Missing Tool Call | 4.4% | 5.7% | -1.3pp (Gemma better) |
| Unnecessary Tool Call | 14.4% | 11.4% | +2.9pp (Qwen better) |
| Coreference Failure | 18.3% | 18.8% | -0.5pp |
| Compositional Planning | 15.4% | 19.4% | -4.0pp (Gemma better) |
| Instruction Transition | 20.5% | 21.0% | -0.5pp |
| Hallucinated Data | 9.5% | 6.6% | +2.8pp (Qwen better) |
| Format/Parse Error | 0.3% | 0.3% | +0.0pp |

### Accuracy-to-Cost Ratio

Since all models run locally on the same hardware via Ollama, "cost" is primarily inference time.

| Model | Task Acc | Approx Runtime | Tasks/Hour |
|-------|----------|----------------|------------|
| Qwen3-8B | 40.8% | ~6.5h | 39.4 |
| Qwen3-14B | 44.2% | ~8.5h | 30.1 |
| Qwen3-32B | 44.1% | ~19h | 13.5 |
| Gemma4-31B | 52.0% | ~16h | 16.0 |

**Best accuracy-to-cost ratio: Gemma4-31B** — highest accuracy (52.0%) at moderate runtime (16h), yielding 16 tasks/hour. Qwen3-8B is the fastest (39.4 tasks/hour) but at significantly lower accuracy.

## 6. Ranked Harness Improvement Proposals

| Rank | Intervention | Failure Modes Addressed | Affected Tasks | Est. Fix Rate | Expected Impact | Difficulty |
|------|-------------|------------------------|----------------|---------------|-----------------|------------|
| 1 | **Intent Classifier (Clarify/Chat/Tool Router)** | Unnecessary Tool Call, Instruction Transition, Missing Tool Call | 1725 (76.6%) | 50% | 862 tasks | Medium |
| 2 | **Argument Tolerance Layer** | Wrong Arguments | 783 (34.8%) | 80% | 626 tasks | Low |
| 3 | **Coreference Resolution Prompt Injection** | Coreference Failure | 778 (34.5%) | 40% | 311 tasks | Medium |
| 4 | **Multi-Step Task Planner** | Compositional Planning | 771 (34.2%) | 35% | 270 tasks | High |
| 5 | **String Normalization Pre-processor** | Wrong Arguments, Hallucinated Data | 428 (19.0%) | 60% | 257 tasks | Low |
| 6 | **Structured Output Retry** | Format/Parse Error | 14 (0.6%) | 70% | 10 tasks | Low |

### 1. Intent Classifier (Clarify/Chat/Tool Router)

**What it does:** Add a lightweight classification layer before tool invocation that determines whether the user turn requires a tool call, a clarification request, or a conversational response. Can be a fine-tuned small model or prompt-based few-shot classifier.

**Model role:** Qwen3-8B as router, larger model as executor

### 2. Argument Tolerance Layer

**What it does:** Post-process tool calls to strip extra optional parameters that match the tool schema but aren't in the ground truth. Accept arguments that are semantically equivalent (e.g., different date formats, case variations).

**Model role:** N/A (rule-based post-processing)

### 3. Coreference Resolution Prompt Injection

**What it does:** Before each turn after the first, inject a prompt step that explicitly resolves all pronouns and references ("the third one", "that location") to concrete entities from the conversation history. Feed the resolved query to the tool-calling model.

**Model role:** Any model for resolution; Gemma4-31B recommended for best baseline

### 4. Multi-Step Task Planner

**What it does:** For compound requests, decompose into a dependency graph of sub-tasks before execution. Use the tool schema to identify which calls can be parallelized vs must be sequential. Execute the plan step by step.

**Model role:** Gemma4-31B as planner, any model as executor

### 5. String Normalization Pre-processor

**What it does:** Normalize argument values before comparison: standardize date formats, trim whitespace, lowercase where appropriate, expand abbreviations. Catches cases where the model produces semantically correct but syntactically different values.

**Model role:** N/A (rule-based pre-processing)

### 6. Structured Output Retry

**What it does:** When a tool call fails JSON schema validation, retry with explicit schema constraints in the prompt. Use constrained decoding or function calling mode if available.

**Model role:** Same model with stricter prompting

## 7. Surprising Findings

1. **Gemma4 dominates Parallel Multi-Tool despite similar size to Qwen3-32B.** Gemma4 achieves 49.7% on parallel tasks vs Qwen3-32B's 29.7% — a 20pp gap. This suggests Gemma4's architecture is fundamentally better at recognizing when multiple independent tool calls should be made simultaneously.

2. **Qwen3-32B has LOWER Chat accuracy (78.5%) than Qwen3-8B (87.1%).** Larger Qwen3 models are more likely to unnecessarily invoke tools when they should just respond conversationally. This is an over-eagerness problem that scales with model size in the Qwen3 family.

3. **Sequential Multi-Tool is the hardest category for all models** (6.3%–18.8% accuracy). Even Gemma4 only manages 18.8%. These tasks require strict dependency chains where each step's output feeds the next — a single error cascades and ruins the entire sequence.

4. **Long-Range Dependency is the hardest turn subtype** across all models (28.5%–39.9%). These turns require the model to recall and use information from 2+ turns back in the conversation. This drops accuracy by ~10-20pp compared to First Turn performance.

5. **394 tasks (38.5%) are universally failed** by all 4 models. These represent tasks where current open-weight models at the 8B-32B scale consistently break down — likely candidates for targeted data augmentation or specialized fine-tuning.

---

*Report generated by `Enhancements/analysis_report.py` from WildToolBench evaluation data.*