# Handoff Guide for the Next Claude Instance

You are taking over a benchmark run that has been validated on a 5-sample
test and is ready to run on the full 256-session WildToolBench. The owner
is expecting clean per-model result files + a 3-way comparison report.

## TL;DR

1. Check out branch `wtb-mas-impl` (latest commit on remote should match
   what you see locally).
2. Run `bash wild-tool-bench/run_wtbmas_chain.sh` from the repo root.
3. After it finishes (3–10 days depending on hardware), run
   `python wild-tool-bench/scripts/aggregate_three_way.py` to produce the
   3-way comparison artifacts.
4. Push everything and notify the owner.

## What's been done (don't redo)

1. **Phase 2 deliverable** is complete on this branch:
   `wild-tool-bench/Phase2_Report/Phase2_Report.docx` — error analysis,
   5-category taxonomy, multi-agent proposal, references verified.

2. **Baseline (no changes)** results exist for all 4 models:
   `wild-tool-bench/score/{qwen3:8b, qwen3:14b, qwen3:32b, gemma4:31b}/`

3. **Eval-improvements (v2)** results exist for all 4 models:
   `wild-tool-bench/score_v2/{...}/` (system-prompt + arg-tolerance + coreference
   injection improvements; +4.8 to +12.5pp over baseline).

4. **WTB-MAS v2 implementation** is complete and validated on a 5-sample
   test (qwen3:8b: +25pp over baseline). Code in
   `wild-tool-bench/wtb/model_handler/wtbmas/`.
   See `wild-tool-bench/wtb/model_handler/wtbmas/README.md` for design.

5. **First WTB-MAS implementation (v1) was a regression** (-33pp) due to
   over-aggressive Planner system prompt and a broken Chat branch. v2
   redesigned as a thin wrapper around the LLM's native function-calling.

## What you need to do

### 1. Verify the branch is what you expect

```bash
git checkout wtb-mas-impl
git log --oneline -8       # should show the v2 orchestrator commit on top
git pull                   # if there's a remote you can pull from
```

### 2. Verify the environment

```bash
# Conda env
conda activate wildtoolbench

# Ollama running with the four backbones pulled
curl -s http://localhost:11434/api/tags | python -c "import json,sys; print([m['name'] for m in json.load(sys.stdin)['models']])"
# Must include: qwen3:8b, qwen3:14b, qwen3:32b, gemma4:31b

# .env file in wild-tool-bench/ exists and points at Ollama
cat wild-tool-bench/.env
```

If any are missing, see `wild-tool-bench/CHANGES.md` for setup steps.

### 3. Run the chain

The chain script runs the four backbones sequentially with per-backbone
resume support:

```bash
cd wild-tool-bench
bash run_wtbmas_chain.sh                   # all 4 backbones in sequence
# OR a subset:
bash run_wtbmas_chain.sh qwen3:8b qwen3:14b
```

Estimated wall-clock per backbone (on 32GB AMD GPU; A100 will be ~3× faster):

| Backbone     | Per-session | Full 1024-task benchmark |
|--------------|:-----------:|:-------------------------:|
| qwen3:8b     | ~70s        | ~5–8 hours                |
| qwen3:14b    | ~120s       | ~10–18 hours              |
| qwen3:32b    | ~250s       | ~30–60 hours              |
| gemma4:31b   | ~220s       | ~25–50 hours              |

**Total sequential**: 60–130 hours. Run inside `tmux` or `nohup` so
terminal closures don't kill it:

```bash
tmux new -s wtbmas
bash run_wtbmas_chain.sh
# Ctrl-b d to detach;  tmux attach -t wtbmas to come back
```

### 4. Watch progress

```bash
# Real-time status timeline (one line per model start/end/error)
tail -f logs/wtbmas_chain_status.txt

# Per-model inference log (the active one)
tail -f logs/wtbmas_qwen3_8b_*.log

# Session count for the active model
wc -l result/wtbmas:qwen3:8b/Wild-Tool-Bench_result.jsonl   # max 256
```

If a backbone hangs or fails, the chain script records it and moves on.
The next backbone starts automatically.

### 5. Sanity-check results vs baseline

After each backbone completes, eyeball it. The chain runs the eval
automatically, so:

```bash
python -c "
import json
for m in ['qwen3:8b']:    # update list after each completes
    for src in ['score', 'score_v2', f'score']:
        path = f'{src}/{m if src==\"score_v2\" else (\"wtbmas:\"+m if \"wtbmas\" not in src else m)}/Wild-Tool-Bench_metric.json'
        try:
            d = json.load(open(path))
            print(f'{path}: task_acc={d[\"total_info\"][\"task\"][\"accuracy\"]*100:.1f}%')
        except FileNotFoundError:
            print(f'{path}: not yet')
"
```

WTB-MAS v2 should be **at least matching baseline**, ideally ≥ +5pp. If
it's regressing significantly, STOP THE CHAIN and report back — we may
have a model-specific failure (prompt incompatibility, etc.).

### 6. Aggregate the three datasets

After all backbones complete:

```bash
python wild-tool-bench/scripts/aggregate_three_way.py
```

This writes to `wild-tool-bench/Phase3_Report/`:
- `three_way_comparison.json` — full structured data
- `three_way_comparison.md`   — slide-ready table
- `three_way_comparison.csv`  — long-form for plotting

### 7. Commit + push

```bash
git add wild-tool-bench/result/wtbmas:* wild-tool-bench/score/wtbmas:* wild-tool-bench/Phase3_Report/
git commit -m "WTB-MAS full benchmark results: 4 backbones + 3-way comparison"
git push
```

### 8. Notify the owner

Hand off the comparison numbers (deltas vs baseline) and link to the
generated artifacts.

## Things to NOT do

- Don't modify the v2 orchestrator without re-validating on the 5-sample
  test first (`bash run_local.sh infer wtbmas:qwen3:8b --run-ids`).
- Don't run multiple backbones in parallel — VRAM is tight on 32GB and
  the chain script is designed for sequential operation.
- Don't reset/clear the existing `score/` and `score_v2/` directories;
  those are the comparison anchors.

## Files you'll likely touch

| File | Purpose |
|------|---------|
| `wild-tool-bench/run_wtbmas_chain.sh` | Sequential chain runner |
| `wild-tool-bench/scripts/aggregate_three_way.py` | 3-way comparison generator |
| `wild-tool-bench/logs/wtbmas_chain_status.txt` | Status timeline (auto-written by chain) |
| `wild-tool-bench/Phase3_Report/three_way_comparison.{json,md,csv}` | Final outputs |

## If you hit a hang

There are 300s and 600s timeouts in place at the LLM-call and per-turn
levels respectively. If you see a session take >15 min that's
suspicious — check `ps aux | grep openfunctions_evaluation` and consider
killing the process. The chain script will continue with the next session.

The most likely cause of a hang is Ollama itself getting stuck. Restart
Ollama (`sudo systemctl restart ollama` or `killall ollama && ollama
serve &`) and re-run the chain — it'll resume from the last completed
session.

Good luck.
