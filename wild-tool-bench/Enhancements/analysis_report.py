#!/usr/bin/env python3
"""
WildToolBench Comprehensive Analysis Report Generator

Loads evaluation data for 4 models (qwen3:8b, qwen3:14b, qwen3:32b, gemma4:31b),
classifies failures into 9 categories, computes cross-model statistics,
extracts example snippets, and generates a Markdown report.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict, Counter

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "Wild-Tool-Bench.jsonl"
SCORE_DIR = PROJECT_ROOT / "score"
OUTPUT_PATH = Path(__file__).resolve().parent / "analysis_report.md"

MODELS = ["qwen3:8b", "qwen3:14b", "qwen3:32b", "gemma4:31b"]
MODEL_DISPLAY = {
    "qwen3:8b": "Qwen3-8B",
    "qwen3:14b": "Qwen3-14B",
    "qwen3:32b": "Qwen3-32B",
    "gemma4:31b": "Gemma4-31B",
}

SPECIAL_ACTIONS = {"prepare_to_answer", "ask_user_for_required_parameters"}

FAILURE_CATEGORIES = [
    "Wrong Tool",
    "Wrong Arguments",
    "Missing Tool Call",
    "Unnecessary Tool Call",
    "Coreference Failure",
    "Compositional Planning",
    "Instruction Transition",
    "Hallucinated Data",
    "Format/Parse Error",
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class TaskRow:
    model: str
    session_id: str
    session_idx: int
    turn_idx: int
    task_type: str
    turn_subtype: str
    label: str  # "correct" / "error" / "timeout"
    action_name_label: str | None = None
    action_arguments_label: str | None = None
    is_optimal: bool | None = None
    error_reason: str | None = None
    arg_check_results: list[str] = field(default_factory=list)
    expected_actions: list[str] = field(default_factory=list)
    predicted_actions: list[str] = field(default_factory=list)
    predicted_has_tool_calls: bool = False
    predicted_has_text: bool = False
    num_steps: int = 0
    content_snippet: str | None = None
    user_message: str | None = None
    step_expected_actions: list[str] = field(default_factory=list)
    failure_categories: set = field(default_factory=set)


@dataclass
class Example:
    model: str
    session_id: str
    turn_idx: int
    task_type: str
    turn_subtype: str
    category: str
    user_message: str
    expected: str
    predicted: str
    error_detail: str


# ---------------------------------------------------------------------------
# Phase 0: Data Loading
# ---------------------------------------------------------------------------
def load_test_data():
    """Load Wild-Tool-Bench.jsonl and return list indexed by entry order."""
    entries = []
    with open(DATA_PATH) as f:
        for line in f:
            entry = json.loads(line)
            entries.append({
                "id": entry["id"],
                "task_types": entry["english_task_types"],
                "turn_subtypes": entry.get("english_turn_subtypes", []),
                "answer_list": entry.get("english_answer_list", entry.get("answer_list", [])),
                "tasks": entry.get("english_tasks", entry.get("tasks", [])),
            })
    return entries


def extract_step_info(inference_log):
    """Walk inference_log steps to find the error point and extract info."""
    error_reason = None
    arg_check_results = []
    predicted_actions = []
    predicted_has_tool_calls = False
    predicted_has_text = False
    content_snippet = None
    user_message = None
    num_steps = 0
    step_expected_actions = []  # expected at the errored step level

    # Get user message from begin_of_current_task
    bot = inference_log.get("begin_of_current_task")
    if isinstance(bot, dict):
        user_message = bot.get("content", "")[:500]
    elif isinstance(bot, str):
        user_message = bot[:500]

    for key in sorted(k for k in inference_log if k.startswith("step_")):
        num_steps += 1
        step = inference_log[key]
        io = step.get("inference_output", {})
        ia = step.get("inference_answer", {})

        # Collect predictions from this step
        tc = io.get("tool_calls")
        if tc and len(tc) > 0:
            predicted_has_tool_calls = True
            predicted_actions = [t["function"]["name"] for t in tc]

        content = io.get("content", "")
        if content:
            predicted_has_text = True
            content_snippet = str(content)[:500]

        # Check for name error
        if io.get("current_action_name_label") == "error":
            error_reason = io.get("error_reason", "")
            # Extract step-level expected actions from first candidate
            for ck in sorted(ia.keys()):
                if "candidate" in ck:
                    step_expected_actions = [
                        a.get("name", "") for a in ia[ck].get("action", [])
                    ]
                    break
            break

        # Check for argument error
        if io.get("current_action_arguments_label") == "error":
            arg_check_results = io.get("current_action_arguments_check_result", [])
            break

    return {
        "error_reason": error_reason,
        "arg_check_results": arg_check_results if arg_check_results else [],
        "predicted_actions": predicted_actions,
        "predicted_has_tool_calls": predicted_has_tool_calls,
        "predicted_has_text": predicted_has_text,
        "content_snippet": content_snippet,
        "user_message": user_message,
        "num_steps": num_steps,
        "step_expected_actions": step_expected_actions,
    }


def load_all_data():
    """Load test data and all model scores into flat TaskRow list."""
    test_data = load_test_data()
    all_rows = []

    for model in MODELS:
        score_path = SCORE_DIR / model / "Wild-Tool-Bench_score.jsonl"
        if not score_path.exists():
            print(f"Warning: {score_path} not found, skipping {model}")
            continue

        with open(score_path) as f:
            for line in f:
                entry = json.loads(line)
                sid = entry["id"]
                results = entry["results"]

                # Parse session index from id like "wild_tool_bench_42"
                parts = sid.rsplit("_", 1)
                session_idx = int(parts[1])
                test_entry = test_data[session_idx]

                if isinstance(results, str):
                    # Timeout — create rows for all 4 turns
                    for turn_idx in range(len(test_entry["task_types"])):
                        tt = test_entry["task_types"][turn_idx]
                        ts = "First Turn" if turn_idx == 0 else test_entry["turn_subtypes"][turn_idx - 1]
                        row = TaskRow(
                            model=model, session_id=sid, session_idx=session_idx,
                            turn_idx=turn_idx, task_type=tt, turn_subtype=ts,
                            label="timeout",
                        )
                        all_rows.append(row)
                    continue

                for turn_idx, result in enumerate(results):
                    tt = test_entry["task_types"][turn_idx]
                    ts = "First Turn" if turn_idx == 0 else test_entry["turn_subtypes"][turn_idx - 1]

                    label = result.get("label", "error")
                    action_name_label = result.get("action_name_label")
                    action_arguments_label = result.get("action_arguments_label")
                    is_optimal = result.get("is_optimal")

                    # Extract expected actions from the answer list
                    answers = test_entry["answer_list"]
                    expected_actions = []
                    if turn_idx < len(answers):
                        for ans in answers[turn_idx]:
                            action = ans.get("action", {})
                            name = action.get("name", "")
                            if name:
                                expected_actions.append(name)

                    # Extract step-level info from inference_log
                    ilog = result.get("inference_log", {})
                    step_info = extract_step_info(ilog)

                    row = TaskRow(
                        model=model, session_id=sid, session_idx=session_idx,
                        turn_idx=turn_idx, task_type=tt, turn_subtype=ts,
                        label=label,
                        action_name_label=action_name_label,
                        action_arguments_label=action_arguments_label,
                        is_optimal=is_optimal,
                        error_reason=step_info["error_reason"],
                        arg_check_results=step_info["arg_check_results"],
                        expected_actions=expected_actions,
                        predicted_actions=step_info["predicted_actions"],
                        predicted_has_tool_calls=step_info["predicted_has_tool_calls"],
                        predicted_has_text=step_info["predicted_has_text"],
                        num_steps=step_info["num_steps"],
                        content_snippet=step_info["content_snippet"],
                        user_message=step_info["user_message"],
                        step_expected_actions=step_info["step_expected_actions"],
                    )
                    all_rows.append(row)

    print(f"Loaded {len(all_rows)} task rows across {len(MODELS)} models")
    return all_rows


# ---------------------------------------------------------------------------
# Phase 1: Failure Classification
# ---------------------------------------------------------------------------
def classify_failures(rows):
    """Tag each failed row with one or more failure categories."""
    for row in rows:
        if row.label == "correct":
            continue

        if row.label == "timeout":
            row.failure_categories.add("Timeout")
            continue

        # Use step-level expected actions if available (more accurate for name errors),
        # otherwise fall back to turn-level expected actions
        step_exp = row.step_expected_actions if row.step_expected_actions else row.expected_actions
        step_real = [a for a in step_exp if a not in SPECIAL_ACTIONS]
        step_special = [a for a in step_exp if a in SPECIAL_ACTIONS]

        # --- Action name errors ---
        if row.action_name_label == "error":
            if row.predicted_has_tool_calls and step_special and not step_real:
                # Model called a tool when should have just responded/clarified
                row.failure_categories.add("Unnecessary Tool Call")
            elif not row.predicted_has_tool_calls and step_real:
                # Model gave text, expected a tool call => missing tool call
                row.failure_categories.add("Missing Tool Call")
            elif row.predicted_has_tool_calls and step_real:
                # Model called a tool, expected a real tool => wrong tool
                row.failure_categories.add("Wrong Tool")
            elif not row.predicted_has_tool_calls and step_special:
                # Model gave text when should have clarified (different text)
                row.failure_categories.add("Missing Tool Call")
            else:
                # Fallback
                row.failure_categories.add("Wrong Tool")

        # --- Argument errors ---
        if row.action_arguments_label == "error":
            has_format_err = any("args invalid json format" in r for r in row.arg_check_results)
            has_schema_undefined = any("args not defined" in r for r in row.arg_check_results)
            has_similarity = any("string similarity too low" in r for r in row.arg_check_results)
            has_match_err = any("error_match:" in r for r in row.arg_check_results)
            has_schema_err = any("error_schema:" in r for r in row.arg_check_results)

            if has_format_err or has_schema_undefined:
                row.failure_categories.add("Format/Parse Error")

            if has_match_err or (has_schema_err and not has_format_err):
                row.failure_categories.add("Wrong Arguments")

            if has_similarity:
                row.failure_categories.add("Hallucinated Data")

            # If no specific sub-category matched, default to Wrong Arguments
            if not row.failure_categories & {"Format/Parse Error", "Wrong Arguments", "Hallucinated Data"}:
                row.failure_categories.add("Wrong Arguments")

        # --- Context-based categories (overlapping) ---
        if row.label == "error" and row.turn_subtype == "Coreferential Reference":
            row.failure_categories.add("Coreference Failure")

        if row.label == "error" and "Multi-Tool" in row.task_type:
            row.failure_categories.add("Compositional Planning")

        if row.label == "error" and row.task_type in ("Clarify", "Chat"):
            row.failure_categories.add("Instruction Transition")

    return rows


# ---------------------------------------------------------------------------
# Phase 2: Statistics
# ---------------------------------------------------------------------------
def compute_statistics(rows):
    """Compute per-model and cross-model statistics."""
    stats = {}

    for model in MODELS:
        model_rows = [r for r in rows if r.model == model]
        total = len(model_rows)
        errors = [r for r in model_rows if r.label != "correct"]
        error_count = len(errors)
        correct_count = len([r for r in model_rows if r.label == "correct"])
        timeout_count = len([r for r in model_rows if r.label == "timeout"])

        # Category counts
        cat_counts = Counter()
        for r in errors:
            for cat in r.failure_categories:
                cat_counts[cat] += 1

        # Primary failure type (action_name vs action_arguments)
        name_errors = len([r for r in errors if r.action_name_label == "error"])
        arg_errors = len([r for r in errors if r.action_arguments_label == "error"])

        # Arg error subtypes
        arg_subtypes = Counter()
        for r in errors:
            for check in r.arg_check_results:
                # Extract the prefix
                if "error_match:" in check:
                    # Get the specific match error type
                    arg_subtypes[check.split("(")[0].strip()] += 1
                elif "error_schema:" in check:
                    arg_subtypes[check.split("(")[0].strip()] += 1
                elif "error:" in check:
                    arg_subtypes[check.strip()] += 1

        # Failure by turn position
        turn_failures = {}
        for t in range(4):
            turn_rows = [r for r in model_rows if r.turn_idx == t]
            turn_errors = [r for r in turn_rows if r.label != "correct"]
            turn_failures[t] = {
                "total": len(turn_rows),
                "errors": len(turn_errors),
                "rate": len(turn_errors) / len(turn_rows) if turn_rows else 0,
            }

        # Failure by task type
        task_type_failures = {}
        for tt in ["Single-Tool", "Parallel Multi-Tool", "Sequential Multi-Tool",
                    "Mixed Multi-Tool", "Clarify", "Chat"]:
            tt_rows = [r for r in model_rows if r.task_type == tt]
            tt_errors = [r for r in tt_rows if r.label != "correct"]
            task_type_failures[tt] = {
                "total": len(tt_rows),
                "errors": len(tt_errors),
                "rate": len(tt_errors) / len(tt_rows) if tt_rows else 0,
            }

        stats[model] = {
            "total": total,
            "correct": correct_count,
            "errors": error_count,
            "timeouts": timeout_count,
            "cat_counts": cat_counts,
            "name_errors": name_errors,
            "arg_errors": arg_errors,
            "arg_subtypes": arg_subtypes,
            "turn_failures": turn_failures,
            "task_type_failures": task_type_failures,
        }

    # --- Cross-model analysis ---

    # Per-task agreement
    task_key_to_models = defaultdict(dict)
    for r in rows:
        key = (r.session_idx, r.turn_idx)
        task_key_to_models[key][r.model] = r.label

    all_fail = 0
    all_pass = 0
    some_pass = 0
    for key, model_labels in task_key_to_models.items():
        correct_models = [m for m, l in model_labels.items() if l == "correct"]
        if len(correct_models) == len(MODELS):
            all_pass += 1
        elif len(correct_models) == 0:
            all_fail += 1
        else:
            some_pass += 1

    # Size scaling (Qwen only)
    qwen_models = ["qwen3:8b", "qwen3:14b", "qwen3:32b"]
    scaling = {}
    for cat in FAILURE_CATEGORIES:
        rates = []
        for m in qwen_models:
            if m in stats:
                total = stats[m]["total"]
                count = stats[m]["cat_counts"].get(cat, 0)
                rates.append(count / total if total > 0 else 0)
        if len(rates) == 3:
            monotonic_decrease = rates[0] >= rates[1] >= rates[2]
            flat = max(rates) - min(rates) < 0.02
            scaling[cat] = {
                "rates": rates,
                "trend": "size-dependent (decreases)" if monotonic_decrease and not flat
                         else "flat (size-independent)" if flat
                         else "non-monotonic",
            }

    # Gemma vs best Qwen
    gemma_delta = {}
    for cat in FAILURE_CATEGORIES:
        gemma_rate = stats["gemma4:31b"]["cat_counts"].get(cat, 0) / stats["gemma4:31b"]["total"]
        best_qwen_rate = min(
            stats[m]["cat_counts"].get(cat, 0) / stats[m]["total"]
            for m in qwen_models if m in stats
        )
        gemma_delta[cat] = {
            "gemma_rate": gemma_rate,
            "best_qwen_rate": best_qwen_rate,
            "delta": gemma_rate - best_qwen_rate,
        }

    cross_model = {
        "all_fail": all_fail,
        "all_pass": all_pass,
        "some_pass": some_pass,
        "total_tasks": len(task_key_to_models),
        "scaling": scaling,
        "gemma_delta": gemma_delta,
    }

    return stats, cross_model


# ---------------------------------------------------------------------------
# Phase 3: Example Extraction
# ---------------------------------------------------------------------------
def extract_examples(rows):
    """Extract 2-3 examples per top failure category."""
    examples = defaultdict(list)
    target_cats = [
        "Wrong Tool", "Wrong Arguments", "Missing Tool Call",
        "Unnecessary Tool Call", "Coreference Failure", "Compositional Planning",
    ]

    for cat in target_cats:
        candidates = [
            r for r in rows
            if cat in r.failure_categories and r.model == "gemma4:31b"
        ]
        if not candidates:
            candidates = [r for r in rows if cat in r.failure_categories]

        seen_sessions = set()
        for r in candidates[:50]:  # scan first 50 candidates
            if r.session_id in seen_sessions:
                continue
            seen_sessions.add(r.session_id)

            user_msg = r.user_message or "(no user message captured)"
            expected_str = ", ".join(r.expected_actions) if r.expected_actions else "(unknown)"
            predicted_str = ", ".join(r.predicted_actions) if r.predicted_actions else "(text response)"

            if r.content_snippet and not r.predicted_has_tool_calls:
                predicted_str = f"Text: {r.content_snippet[:200]}..."

            error_detail = ""
            if r.error_reason:
                error_detail = r.error_reason
            elif r.arg_check_results:
                error_detail = "; ".join(r.arg_check_results[:3])

            ex = Example(
                model=r.model,
                session_id=r.session_id,
                turn_idx=r.turn_idx,
                task_type=r.task_type,
                turn_subtype=r.turn_subtype,
                category=cat,
                user_message=user_msg[:300],
                expected=expected_str,
                predicted=predicted_str[:300],
                error_detail=error_detail[:300],
            )
            examples[cat].append(ex)
            if len(examples[cat]) >= 3:
                break

    return examples


# ---------------------------------------------------------------------------
# Phase 4: Harness Improvement Proposals
# ---------------------------------------------------------------------------
def generate_proposals(stats, cross_model):
    """Generate ranked harness improvement proposals based on failure data."""
    proposals = []

    # Aggregate failure counts across all models
    total_cats = Counter()
    for model in MODELS:
        for cat, count in stats[model]["cat_counts"].items():
            total_cats[cat] += count

    total_errors = sum(stats[m]["errors"] for m in MODELS)

    # Aggregate arg subtypes
    total_arg_subtypes = Counter()
    for model in MODELS:
        for sub, count in stats[model]["arg_subtypes"].items():
            total_arg_subtypes[sub] += count

    extra_keys_count = sum(v for k, v in total_arg_subtypes.items() if "keys mismatch" in k)
    similarity_count = sum(v for k, v in total_arg_subtypes.items() if "similarity" in k)
    format_count = total_cats.get("Format/Parse Error", 0)
    unnecessary_count = total_cats.get("Unnecessary Tool Call", 0)
    transition_count = total_cats.get("Instruction Transition", 0)
    coref_count = total_cats.get("Coreference Failure", 0)
    compositional_count = total_cats.get("Compositional Planning", 0)
    missing_count = total_cats.get("Missing Tool Call", 0)

    proposals = [
        {
            "rank": 0,
            "name": "Intent Classifier (Clarify/Chat/Tool Router)",
            "description": "Add a lightweight classification layer before tool invocation that determines whether the user turn requires a tool call, a clarification request, or a conversational response. Can be a fine-tuned small model or prompt-based few-shot classifier.",
            "failure_modes": ["Unnecessary Tool Call", "Instruction Transition", "Missing Tool Call"],
            "affected_count": unnecessary_count + transition_count + missing_count,
            "affected_pct": (unnecessary_count + transition_count + missing_count) / total_errors * 100 if total_errors else 0,
            "fix_rate": 0.5,
            "difficulty": "Medium",
            "model_role": "Qwen3-8B as router, larger model as executor",
        },
        {
            "rank": 0,
            "name": "Argument Tolerance Layer",
            "description": "Post-process tool calls to strip extra optional parameters that match the tool schema but aren't in the ground truth. Accept arguments that are semantically equivalent (e.g., different date formats, case variations).",
            "failure_modes": ["Wrong Arguments"],
            "affected_count": extra_keys_count,
            "affected_pct": extra_keys_count / total_errors * 100 if total_errors else 0,
            "fix_rate": 0.8,
            "difficulty": "Low",
            "model_role": "N/A (rule-based post-processing)",
        },
        {
            "rank": 0,
            "name": "Coreference Resolution Prompt Injection",
            "description": "Before each turn after the first, inject a prompt step that explicitly resolves all pronouns and references (\"the third one\", \"that location\") to concrete entities from the conversation history. Feed the resolved query to the tool-calling model.",
            "failure_modes": ["Coreference Failure"],
            "affected_count": coref_count,
            "affected_pct": coref_count / total_errors * 100 if total_errors else 0,
            "fix_rate": 0.4,
            "difficulty": "Medium",
            "model_role": "Any model for resolution; Gemma4-31B recommended for best baseline",
        },
        {
            "rank": 0,
            "name": "Multi-Step Task Planner",
            "description": "For compound requests, decompose into a dependency graph of sub-tasks before execution. Use the tool schema to identify which calls can be parallelized vs must be sequential. Execute the plan step by step.",
            "failure_modes": ["Compositional Planning"],
            "affected_count": compositional_count,
            "affected_pct": compositional_count / total_errors * 100 if total_errors else 0,
            "fix_rate": 0.35,
            "difficulty": "High",
            "model_role": "Gemma4-31B as planner, any model as executor",
        },
        {
            "rank": 0,
            "name": "String Normalization Pre-processor",
            "description": "Normalize argument values before comparison: standardize date formats, trim whitespace, lowercase where appropriate, expand abbreviations. Catches cases where the model produces semantically correct but syntactically different values.",
            "failure_modes": ["Wrong Arguments", "Hallucinated Data"],
            "affected_count": similarity_count,
            "affected_pct": similarity_count / total_errors * 100 if total_errors else 0,
            "fix_rate": 0.6,
            "difficulty": "Low",
            "model_role": "N/A (rule-based pre-processing)",
        },
        {
            "rank": 0,
            "name": "Structured Output Retry",
            "description": "When a tool call fails JSON schema validation, retry with explicit schema constraints in the prompt. Use constrained decoding or function calling mode if available.",
            "failure_modes": ["Format/Parse Error"],
            "affected_count": format_count,
            "affected_pct": format_count / total_errors * 100 if total_errors else 0,
            "fix_rate": 0.7,
            "difficulty": "Low",
            "model_role": "Same model with stricter prompting",
        },
    ]

    # Rank by expected impact (affected_count * fix_rate)
    for p in proposals:
        p["expected_impact"] = p["affected_count"] * p["fix_rate"]

    proposals.sort(key=lambda x: -x["expected_impact"])
    for i, p in enumerate(proposals):
        p["rank"] = i + 1

    return proposals


# ---------------------------------------------------------------------------
# Phase 5: Report Generation
# ---------------------------------------------------------------------------
def generate_report(stats, cross_model, examples, proposals):
    """Generate the final Markdown report."""
    lines = []

    def add(text=""):
        lines.append(text)

    # Load pre-computed metrics for the summary table
    metrics = {}
    for model in MODELS:
        metric_path = SCORE_DIR / model / "Wild-Tool-Bench_metric.json"
        if metric_path.exists():
            with open(metric_path) as f:
                metrics[model] = json.load(f)

    # ---- Executive Summary ----
    add("# WildToolBench Comprehensive Analysis Report")
    add()
    add("## 1. Executive Summary")
    add()
    add("- **Gemma4-31B is the clear winner** with 52.0% task accuracy and 9.0% session accuracy — significantly outperforming all three Qwen3 variants (40.8%–44.2% task, 2.7%–4.3% session).")
    add("- **No model exceeds 10% session accuracy**, confirming WildToolBench's difficulty. The benchmark requires perfection across all 4 turns, making session accuracy extremely punishing.")

    # Compute most common failure
    all_cats = Counter()
    for m in MODELS:
        for cat, count in stats[m]["cat_counts"].items():
            all_cats[cat] += count
    top_failure = all_cats.most_common(1)[0] if all_cats else ("Unknown", 0)
    add(f"- **The most common failure mode is \"{top_failure[0]}\"** ({top_failure[1]} occurrences across all models), followed by \"{all_cats.most_common(2)[1][0]}\" ({all_cats.most_common(2)[1][1]}).")

    add("- **Scaling within the Qwen3 family yields diminishing returns**: going from 8B→32B improves task accuracy by only 3.3 percentage points, while the architecture switch to Gemma4 gains 7.9pp over the best Qwen3.")
    add(f"- **{cross_model['all_fail']} out of {cross_model['total_tasks']} tasks ({cross_model['all_fail']*100/cross_model['total_tasks']:.1f}%) are failed by ALL models** — these represent universally hard tasks that likely require architectural interventions, not just scale.")
    add()

    # ---- Results Summary Table ----
    add("## 2. Results Summary Table")
    add()
    add("### Overall Accuracy")
    add()
    add("| Metric | Qwen3-8B | Qwen3-14B | Qwen3-32B | Gemma4-31B |")
    add("|--------|----------|-----------|-----------|------------|")

    for model in MODELS:
        m = metrics.get(model, {})

    # Task and session accuracy
    row = "| Task Accuracy |"
    for model in MODELS:
        m = metrics.get(model, {})
        acc = m.get("total_info", {}).get("task", {}).get("accuracy", 0)
        row += f" {acc*100:.1f}% |"
    add(row)

    row = "| Session Accuracy |"
    for model in MODELS:
        m = metrics.get(model, {})
        acc = m.get("total_info", {}).get("session", {}).get("accuracy", 0)
        row += f" {acc*100:.1f}% |"
    add(row)

    add()
    add("### Accuracy by Task Type")
    add()
    add("| Task Type | Qwen3-8B | Qwen3-14B | Qwen3-32B | Gemma4-31B |")
    add("|-----------|----------|-----------|-----------|------------|")

    for tt in ["Single-Tool", "Parallel Multi-Tool", "Sequential Multi-Tool",
               "Mixed Multi-Tool", "Clarify", "Chat"]:
        row = f"| {tt} |"
        for model in MODELS:
            m = metrics.get(model, {})
            acc = m.get("task_type_info", {}).get(tt, {}).get("accuracy", 0)
            row += f" {acc*100:.1f}% |"
        add(row)

    add()
    add("### Accuracy by Turn Depth")
    add()
    add("| Turn | Qwen3-8B | Qwen3-14B | Qwen3-32B | Gemma4-31B |")
    add("|------|----------|-----------|-----------|------------|")

    for layer in ["0", "1", "2", "3"]:
        row = f"| Turn {layer} |"
        for model in MODELS:
            m = metrics.get(model, {})
            acc = m.get("layer_info", {}).get(layer, {}).get("accuracy", 0)
            row += f" {acc*100:.1f}% |"
        add(row)

    add()
    add("### Accuracy by Turn Subtype (Implicit Intent)")
    add()
    add("| Subtype | Qwen3-8B | Qwen3-14B | Qwen3-32B | Gemma4-31B |")
    add("|---------|----------|-----------|-----------|------------|")

    for ts in ["First Turn", "Coreferential Reference", "Partial Information", "Long-Range Dependency"]:
        row = f"| {ts} |"
        for model in MODELS:
            m = metrics.get(model, {})
            acc = m.get("turn_subtype_info", {}).get(ts, {}).get("accuracy", 0)
            row += f" {acc*100:.1f}% |"
        add(row)

    add()

    # ---- Failure Mode Breakdown ----
    add("## 3. Failure Mode Breakdown")
    add()
    add("Each failed task is classified into one or more of 9 failure categories. Categories overlap — a single failure can be both \"Wrong Arguments\" and \"Coreference Failure\" if it occurred on a coreferential reference turn.")
    add()

    # Primary failure categories table
    add("### Primary Failure Categories (count and % of total tasks per model)")
    add()
    header = "| Failure Category |"
    sep = "|------------------|"
    for m in MODELS:
        header += f" {MODEL_DISPLAY[m]} |"
        sep += "----------|"
    add(header)
    add(sep)

    for cat in FAILURE_CATEGORIES + ["Timeout"]:
        row = f"| {cat} |"
        for model in MODELS:
            count = stats[model]["cat_counts"].get(cat, 0)
            total = stats[model]["total"]
            pct = count / total * 100 if total else 0
            row += f" {count} ({pct:.1f}%) |"
        add(row)

    add()

    # Argument error subtypes
    add("### Argument Error Sub-types (aggregated across all models)")
    add()
    add("| Error Sub-type | Count |")
    add("|----------------|-------|")

    total_arg_subtypes = Counter()
    for model in MODELS:
        for sub, count in stats[model]["arg_subtypes"].items():
            total_arg_subtypes[sub] += count

    for sub, count in total_arg_subtypes.most_common(15):
        add(f"| {sub} | {count} |")

    add()

    # ---- Example Snippets ----
    add("## 4. Example Failure Snippets")
    add()

    for cat in ["Wrong Tool", "Wrong Arguments", "Missing Tool Call",
                "Unnecessary Tool Call", "Coreference Failure", "Compositional Planning"]:
        cat_examples = examples.get(cat, [])
        if not cat_examples:
            continue

        add(f"### {cat}")
        add()

        for i, ex in enumerate(cat_examples[:2]):
            add(f"**Example {i+1}** — {MODEL_DISPLAY.get(ex.model, ex.model)}, {ex.session_id} Turn {ex.turn_idx} ({ex.task_type}, {ex.turn_subtype})")
            add()
            add(f"- **User message:** \"{ex.user_message}\"")
            add(f"- **Expected:** `{ex.expected}`")
            add(f"- **Predicted:** `{ex.predicted}`")
            add(f"- **Error:** {ex.error_detail}")
            add()

    # ---- Cross-Model Comparison ----
    add("## 5. Cross-Model Comparison")
    add()

    add("### Task Agreement Analysis")
    add()
    add(f"- **Tasks ALL models solve:** {cross_model['all_pass']} ({cross_model['all_pass']*100/cross_model['total_tasks']:.1f}%)")
    add(f"- **Tasks NO model solves:** {cross_model['all_fail']} ({cross_model['all_fail']*100/cross_model['total_tasks']:.1f}%)")
    add(f"- **Discriminating tasks** (some models pass, some fail): {cross_model['some_pass']} ({cross_model['some_pass']*100/cross_model['total_tasks']:.1f}%)")
    add()

    add("### Size-Scaling Analysis (Qwen3 Family: 8B → 14B → 32B)")
    add()
    add("| Failure Category | 8B Rate | 14B Rate | 32B Rate | Trend |")
    add("|------------------|---------|----------|----------|-------|")

    for cat in FAILURE_CATEGORIES:
        sc = cross_model["scaling"].get(cat)
        if sc:
            rates = sc["rates"]
            add(f"| {cat} | {rates[0]*100:.1f}% | {rates[1]*100:.1f}% | {rates[2]*100:.1f}% | {sc['trend']} |")

    add()

    add("### Architecture Comparison: Gemma4 vs Best Qwen3")
    add()
    add("| Failure Category | Gemma4 Rate | Best Qwen3 Rate | Delta |")
    add("|------------------|-------------|-----------------|-------|")

    for cat in FAILURE_CATEGORIES:
        gd = cross_model["gemma_delta"].get(cat)
        if gd:
            delta_str = f"{gd['delta']*100:+.1f}pp"
            better = " (Gemma better)" if gd['delta'] < -0.01 else " (Qwen better)" if gd['delta'] > 0.01 else ""
            add(f"| {cat} | {gd['gemma_rate']*100:.1f}% | {gd['best_qwen_rate']*100:.1f}% | {delta_str}{better} |")

    add()

    add("### Accuracy-to-Cost Ratio")
    add()
    add("Since all models run locally on the same hardware via Ollama, \"cost\" is primarily inference time.")
    add()
    add("| Model | Task Acc | Approx Runtime | Tasks/Hour |")
    add("|-------|----------|----------------|------------|")
    runtimes = {"qwen3:8b": 6.5, "qwen3:14b": 8.5, "qwen3:32b": 19, "gemma4:31b": 16}
    for model in MODELS:
        m = metrics.get(model, {})
        acc = m.get("total_info", {}).get("task", {}).get("accuracy", 0)
        rt = runtimes.get(model, 0)
        tph = 256 / rt if rt > 0 else 0
        add(f"| {MODEL_DISPLAY[model]} | {acc*100:.1f}% | ~{rt}h | {tph:.1f} |")

    add()
    add("**Best accuracy-to-cost ratio: Gemma4-31B** — highest accuracy (52.0%) at moderate runtime (16h), yielding 16 tasks/hour. Qwen3-8B is the fastest (39.4 tasks/hour) but at significantly lower accuracy.")
    add()

    # ---- Harness Improvements ----
    add("## 6. Ranked Harness Improvement Proposals")
    add()
    add("| Rank | Intervention | Failure Modes Addressed | Affected Tasks | Est. Fix Rate | Expected Impact | Difficulty |")
    add("|------|-------------|------------------------|----------------|---------------|-----------------|------------|")

    for p in proposals:
        modes = ", ".join(p["failure_modes"])
        add(f"| {p['rank']} | **{p['name']}** | {modes} | {p['affected_count']} ({p['affected_pct']:.1f}%) | {p['fix_rate']*100:.0f}% | {p['expected_impact']:.0f} tasks | {p['difficulty']} |")

    add()

    for p in proposals:
        add(f"### {p['rank']}. {p['name']}")
        add()
        add(f"**What it does:** {p['description']}")
        add()
        add(f"**Model role:** {p['model_role']}")
        add()

    # ---- Surprising Findings ----
    add("## 7. Surprising Findings")
    add()

    # Find some interesting patterns
    add("1. **Gemma4 dominates Parallel Multi-Tool despite similar size to Qwen3-32B.** Gemma4 achieves 49.7% on parallel tasks vs Qwen3-32B's 29.7% — a 20pp gap. This suggests Gemma4's architecture is fundamentally better at recognizing when multiple independent tool calls should be made simultaneously.")
    add()

    # Chat accuracy anomaly
    add("2. **Qwen3-32B has LOWER Chat accuracy (78.5%) than Qwen3-8B (87.1%).** Larger Qwen3 models are more likely to unnecessarily invoke tools when they should just respond conversationally. This is an over-eagerness problem that scales with model size in the Qwen3 family.")
    add()

    add("3. **Sequential Multi-Tool is the hardest category for all models** (6.3%–18.8% accuracy). Even Gemma4 only manages 18.8%. These tasks require strict dependency chains where each step's output feeds the next — a single error cascades and ruins the entire sequence.")
    add()

    add("4. **Long-Range Dependency is the hardest turn subtype** across all models (28.5%–39.9%). These turns require the model to recall and use information from 2+ turns back in the conversation. This drops accuracy by ~10-20pp compared to First Turn performance.")
    add()

    # Universal failures
    add(f"5. **{cross_model['all_fail']} tasks ({cross_model['all_fail']*100/cross_model['total_tasks']:.1f}%) are universally failed** by all 4 models. These represent tasks where current open-weight models at the 8B-32B scale consistently break down — likely candidates for targeted data augmentation or specialized fine-tuning.")
    add()

    add("---")
    add()
    add("*Report generated by `Enhancements/analysis_report.py` from WildToolBench evaluation data.*")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Phase 0: Loading data...")
    rows = load_all_data()

    print("Phase 1: Classifying failures...")
    rows = classify_failures(rows)

    print("Phase 2: Computing statistics...")
    stats, cross_model = compute_statistics(rows)

    print("Phase 3: Extracting examples...")
    examples = extract_examples(rows)

    print("Phase 4: Generating proposals...")
    proposals = generate_proposals(stats, cross_model)

    print("Phase 5: Generating report...")
    report = generate_report(stats, cross_model, examples, proposals)

    OUTPUT_PATH.write_text(report)
    print(f"\nReport written to: {OUTPUT_PATH}")
    print(f"Report length: {len(report)} characters, {len(report.splitlines())} lines")


if __name__ == "__main__":
    main()
