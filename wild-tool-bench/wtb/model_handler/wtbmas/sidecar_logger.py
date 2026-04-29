"""Sidecar telemetry logger for WTB-MAS runs.

The standard WildToolBench result file captures the synthesized response
(content, tool_calls, latencies, tokens) but NOT the per-agent decisions
that justify each WTB-MAS turn. This module persists those decisions to a
separate JSONL file so they can be analyzed for the Phase 3 / presentation
write-up.

Output file: result/<model_name>/agent_logs.jsonl
One line per turn-step. Schema:

    {
      "session_id":  "wild_tool_bench_X",
      "turn_idx":    int,                      # task index in the session
      "wall_clock":  float,                    # epoch seconds when written
      "intent":      "TOOL"|"CLARIFY"|"CHAT"|"TIMEOUT"|"ERROR",
      "agent_log":   {...},                    # full per-agent log dict
      "synthesized_tool_calls": [{name, arguments}, ...] | None,
      "synthesized_content":    str | None,
      "total_latency":          float,
    }

The sidecar file lives in the same directory as the standard result file
so it ships with the rest of the run output. Append-only — each turn-step
adds one line.
"""
from __future__ import annotations

import json
import os
import threading
import time
from typing import Any


class SidecarLogger:
    """Thread-safe append-only JSONL logger for per-turn agent telemetry."""

    def __init__(self, model_name: str, base_dir: str | None = None):
        if base_dir is None:
            # Mirror the harness's result-dir layout: <repo>/result/<model>/
            # We resolve relative to wild-tool-bench/ which is the cwd when running.
            base_dir = os.path.join("result", model_name)
        os.makedirs(base_dir, exist_ok=True)
        self._path = os.path.join(base_dir, "agent_logs.jsonl")
        self._lock = threading.Lock()

    @property
    def path(self) -> str:
        return self._path

    def log_turn(
        self,
        *,
        session_id: str | None,
        turn_idx: int,
        result: dict,
    ) -> None:
        """Persist one turn-step's agent log. Safe to call from any thread."""
        record = {
            "session_id": session_id,
            "turn_idx": turn_idx,
            "wall_clock": time.time(),
            "intent": result.get("intent"),
            "agent_log": result.get("agent_log", {}),
            "synthesized_tool_calls": _compact_tool_calls(result.get("tool_calls")),
            "synthesized_content": result.get("content"),
            "total_latency": result.get("total_latency"),
        }
        line = json.dumps(record, ensure_ascii=False, default=str)
        with self._lock:
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(line + "\n")


def _compact_tool_calls(tcs):
    """Strip OpenAI envelope, keep just {name, arguments} per call."""
    if not tcs:
        return None
    out = []
    for tc in tcs:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function", {})
        if isinstance(fn, dict):
            out.append({"name": fn.get("name"), "arguments": fn.get("arguments")})
        else:
            out.append({"name": None, "arguments": None})
    return out
