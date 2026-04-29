# WTB-MAS: Wild-Tool-Bench Multi-Agent System

Implementation of the multi-agent system proposed in
`wild-tool-bench/Phase2_Report/Phase2_Report.docx` (Section 5).

## Architecture

```
User Turn → IR → ┬→ TOOL:    PA (uses MCA-grounded query) → AV per call → CR → emit
                 ├→ CLARIFY: DA → ask_user_for_required_parameters
                 └→ CHAT:    direct response
```

| Code  | Agent                        | Targets failure category    | LLM-backed |
|-------|------------------------------|------------------------------|------------|
| IR    | Intent Router                | TDF (Tool Dispatch)          | yes        |
| DA    | Disambiguation Agent         | AHF (Ambiguity Handling)     | yes        |
| PA    | Planner Agent                | CPF (Compositional Planning) | yes        |
| MCA   | Memory / Coreference Agent   | CGF (Contextual Grounding)   | tracker = code, resolution = code (heuristic) |
| AV    | Argument Validator           | AFF (Argument Fidelity)      | NO (jsonschema + regex) |
| CR    | Critic Agent                 | residual cross-cutting       | yes        |

All LLM-backed agents share **one Ollama-served backbone** (Strategy A from the
design discussion); each agent uses a different system prompt. The backbone is
parsed from the model name: `wtbmas:qwen3:14b` → backbone `qwen3:14b`.

## Files

```
wtbmas/
├── __init__.py
├── handler.py             — WTBMASHandler (BaseHandler subclass)
├── orchestrator.py        — WTBMASOrchestrator (the WTB_MAS pipeline)
├── memory.py              — SharedMemoryStore (entity tracker)
├── agents/
│   ├── base.py            — LLMClient (Ollama OpenAI-compatible wrapper)
│   ├── intent_router.py   — IR
│   ├── disambiguation.py  — DA
│   ├── planner.py         — PA
│   ├── memory_agent.py    — MCA
│   ├── arg_validator.py   — AV
│   └── critic.py          — CR
├── smoke_test.py          — synthetic 3-turn test
└── README.md              — this file
```

## Prerequisites

```bash
# 1. Ollama running locally
ollama serve

# 2. Backbone model pulled (any of these)
ollama pull qwen3:8b      # ~5GB, fastest
ollama pull qwen3:14b     # ~9GB
ollama pull qwen3:32b     # ~20GB, slowest but strongest

# 3. Python env with this repo's deps
conda activate wildtoolbench
```

## Quickstart: smoke test

Run a 3-turn synthetic conversation through the full pipeline:

```bash
cd wild-tool-bench
python -m wtb.model_handler.wtbmas.smoke_test --backbone qwen3:8b
```

You'll see per-agent decisions and per-turn latencies (~2–8s/turn on Qwen3-8B).

## Running on the full WildToolBench benchmark

The handler is registered in `wtb/model_handler/handler_map.py` as
`wtbmas:<backbone>`. Use it the same way you use any other model:

```bash
cd wild-tool-bench
bash run_local.sh infer wtbmas:qwen3:8b   --run-ids   # 5-sample subset first!
bash run_local.sh eval  wtbmas:qwen3:8b               # score it
bash run_local.sh infer wtbmas:qwen3:14b              # full benchmark
```

Results land in `result/wtbmas:<backbone>/Wild-Tool-Bench_result.jsonl` and
`score/wtbmas:<backbone>/Wild-Tool-Bench_score.jsonl`.

## Expected timing on local 32 GB AMD GPU

| Backbone     | Smoke test | Full benchmark (1024 turns) |
|--------------|:----------:|:----------------------------:|
| qwen3:8b     | ~10–25s    | ~5–8 hours                   |
| qwen3:14b    | ~15–40s    | ~7–12 hours                  |
| qwen3:32b    | ~50–120s   | ~30–60 hours (not recommended locally) |
| gemma4:31b   | ~45–100s   | ~25–50 hours (not recommended locally) |

For 32B-class WTB-MAS runs, prefer SOL with A100s (~10–15h).

## Defaults (and how to change them)

These were chosen sensibly without explicit user input. Each lives in one place
in code so it's easy to toggle.

| Decision                       | Default                              | Where to change                       |
|--------------------------------|--------------------------------------|----------------------------------------|
| Backbone strategy              | Single backbone (Strategy A)         | `agents/base.py` — share `LLMClient`   |
| AV implementation              | Pure code                            | `agents/arg_validator.py`              |
| MCA tracker                    | Pure code                            | `memory.py` + `agents/memory_agent.py` |
| MCA reference resolution       | Heuristic (context summary prepend)  | `agents/memory_agent.py`               |
| Memory persistence scope       | Per session, reset on session change | `handler.py:_request_tool_call`        |
| Entity capacity                | 20 most-recent                       | `memory.py:DEFAULT_ENTITY_CAPACITY`    |
| Critic retry budget            | 1                                    | `orchestrator.py:MAX_CRITIC_RETRIES`   |
| Per-turn LLM call timeout      | 300s (`OLLAMA_REQUEST_TIMEOUT` env)  | `agents/base.py`                       |

## Ablations

Not yet wired up. To run an ablation removing one agent, the cleanest path is
to subclass `WTBMASOrchestrator` and override the agent in question to a
no-op. A `--disable-{IR,DA,PA,MCA,AV,CR}` CLI flag is left as future work.

## Status

This is an MVP scaffold:
- ✅ All six agents implemented
- ✅ End-to-end smoke test passes against Ollama (qwen3:8b verified)
- ✅ Registered in `HANDLER_MAP` so the existing eval pipeline can drive it
- ⏳ Full benchmark run not yet performed (see timing table for cost)
- ⏳ Prompt tuning to actually beat baseline is future work
- ⏳ Ablations not yet wired
