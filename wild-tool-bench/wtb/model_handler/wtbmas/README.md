# WTB-MAS: Wild-Tool-Bench Multi-Agent System

Implementation of the multi-agent system proposed in
`wild-tool-bench/Phase2_Report/Phase2_Report.docx` (Section 5).

## Architecture (v2 — wrapper style)

After the v1 design (which actively interfered with the LLM's natural
tool-completion behavior) regressed scores from 36% baseline → 2.8%, v2 was
redesigned as a **thin wrapper** around the same call the baseline makes.

```
Per-call flow (the eval harness drives multi-step on its own):

  Continuation step?
  (last message = tool observation)
  ├── YES → pass straight through to LLM, no agent wrappers
  └── NO  → fresh user turn:
              IR (Intent Router) classifies →
              ├── CHAT    → call LLM with NO tools, return content
              ├── CLARIFY → DA detects ambiguity → emit
              │              ask_user_for_required_parameters
              │            (or fall through to TOOL if DA disagrees)
              └── TOOL    → MCA prepends entity context →
                            call LLM with tools →
                            AV repairs tool-call arguments
```

Key v1 → v2 changes:
- **Removed the Planner agent's heavy system prompt.** The LLM's native
  function-calling now drives planning; we just route + validate.
- **Removed the Critic's retry loop.** Its retries injected "previous
  attempt was wrong" messages that confused the LLM further.
- **Continuation steps pass straight through** — IR/DA only fire on fresh
  user turns, not after tool observations.
- **Chat branch actually generates a response** (v1 returned empty).

| Agent | Role                          | LLM-backed | Active when                |
|-------|-------------------------------|------------|----------------------------|
| IR    | Classify Tool/Clarify/Chat    | yes        | fresh user turn only       |
| DA    | Detect under-specification    | yes        | intent=CLARIFY only        |
| MCA   | Prepend entity context        | NO (code)  | intent=TOOL only           |
| AV    | Validate + repair tool args   | NO (code)  | any tool call output       |

## Files

```
wtbmas/
├── __init__.py
├── handler.py             — WTBMASHandler (BaseHandler subclass)
├── orchestrator.py        — WTBMASOrchestrator (the per-call pipeline)
├── memory.py              — SharedMemoryStore (entity tracker)
├── sidecar_logger.py      — Per-turn agent telemetry → agent_logs.jsonl
├── agents/
│   ├── base.py            — LLMClient
│   ├── intent_router.py   — IR
│   ├── disambiguation.py  — DA
│   ├── memory_agent.py    — MCA
│   ├── arg_validator.py   — AV
│   ├── planner.py         — (legacy v1; not used by v2 orchestrator)
│   └── critic.py          — (legacy v1; not used by v2 orchestrator)
├── smoke_test.py          — synthetic 3-turn test
└── README.md              — this file
```

The legacy `planner.py` and `critic.py` are kept on disk for reference but
are no longer imported by the v2 orchestrator. Safe to delete in a future
cleanup if not wanted as historical context.

## 5-sample validation result (qwen3:8b)

On `wild_tool_bench_0…4`:

| Approach              | Turn accuracy | Δ vs baseline |
|-----------------------|--------------:|--------------:|
| Baseline qwen3:8b     | 4/20 (20%)    | —             |
| Eval-improvements v2  | 9/20 (45%)    | +25pp         |
| **WTB-MAS v2**        | **9/20 (45%)** | **+25pp**    |

Per-session breakdown shows WTB-MAS v2 actually beats eval-improvements
on `wild_tool_bench_2` (3/4 vs 2/4), and matches it overall.

## Running

### Smoke test (no benchmark data)
```bash
cd wild-tool-bench
python -m wtb.model_handler.wtbmas.smoke_test --backbone qwen3:8b
```

### 5-sample sanity (requires test_case_ids_to_generate.json with the 5 IDs)
```bash
bash run_local.sh infer wtbmas:qwen3:8b --run-ids
bash run_local.sh eval  wtbmas:qwen3:8b
```

### Full benchmark — sequential chain (recommended for cloud)
```bash
bash run_wtbmas_chain.sh                # all 4 backbones in sequence
bash run_wtbmas_chain.sh qwen3:8b       # subset
```

The chain script is resume-safe (skips backbones that already have 256/256
sessions) and writes a single status timeline to
`logs/wtbmas_chain_status.txt`.

## Cloud handoff

See `CLOUD_HANDOFF.md` in the repo root for explicit cloud-instance steps.
