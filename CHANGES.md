# Changes: Ollama Local Model Support

This document describes changes made to run WildToolBench with local Ollama models
(no cloud API keys required).

## Modified Files

### `wild-tool-bench/wtb/model_handler/handler_map.py`
Added `qwen3:14b` and `qwen3:32b` to the handler map, pointing to `OllamaHandler`.

### `wild-tool-bench/wtb/model_handler/api_inference/ollama.py` (new)
Lightweight subclass of `OpenAIHandler` that adds a per-request timeout to prevent
hangs when Ollama gets stuck on complex multi-step tasks. Default timeout is 300s
(5 minutes), configurable via `OLLAMA_REQUEST_TIMEOUT` env var. When a request times
out, the existing retry/error-handling in `_llm_response_generation.py` catches it and
moves to the next test case.

### `wild-tool-bench/.env` (new)
Environment config pointing to Ollama's local endpoint:
- `OPENAI_API_KEY=ollama`
- `OPENAI_BASE_URL=http://localhost:11434/v1`
- `OLLAMA_REQUEST_TIMEOUT=300` (per-request timeout in seconds)

### `wild-tool-bench/run_local.sh` (new)
Convenience script for running inference and evaluation:
```bash
bash run_local.sh infer qwen3:14b --run-ids   # inference on 5-sample subset
bash run_local.sh eval  qwen3:14b              # evaluate results
bash run_local.sh infer qwen3:32b              # full benchmark
```

### `wild-tool-bench/test_case_ids_to_generate.json`
Populated with the first 5 test case IDs for quick testing with `--run-ids`.

## What Was NOT Changed
- Evaluation/scoring logic (`eval_runner.py`, `checker_utils.py`, etc.)
- Base handler class or existing handler implementations
- Test data (`Wild-Tool-Bench.jsonl`)
- Any other pipeline logic

## Prerequisites
- [Ollama](https://ollama.ai) installed and running locally
- Models pulled: `ollama pull qwen3:14b` and/or `ollama pull qwen3:32b`
- Python dependencies: `pip install -r requirements.txt`
