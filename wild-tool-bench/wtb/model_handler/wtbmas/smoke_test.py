"""Smoke test for WTB-MAS — runs the orchestrator on a synthetic conversation.

Usage:
    cd wild-tool-bench
    python -m wtb.model_handler.wtbmas.smoke_test --backbone qwen3:8b

Prerequisites:
  - Ollama running at http://localhost:11434
  - The backbone model pulled: `ollama pull qwen3:8b`

This does NOT run against the benchmark; it just verifies the agents wire up
and produce sensible outputs end-to-end on one fake user turn.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from .orchestrator import WTBMASOrchestrator


# Tiny tool schema for the smoke test
SAMPLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "getWeather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "The city name."},
                    "units": {"type": "string", "enum": ["C", "F"], "description": "Temperature units."},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "getAirQuality",
            "description": "Get the current air quality index for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                },
                "required": ["city"],
            },
        },
    },
]


SCENARIOS = [
    # (turn 0) — should route to TOOL, planner emits getWeather
    (
        "Tool call",
        [
            {"role": "user", "content": "What's the weather in Chicago?"},
        ],
    ),
    # (turn 1) — Chat: model should NOT call tools
    (
        "Chat",
        [
            {"role": "user", "content": "Thanks, that was helpful!"},
        ],
    ),
    # (turn 2) — Clarify: ambiguous reference
    (
        "Clarify",
        [
            {"role": "user", "content": "Also check air quality for one of those cities."},
        ],
    ),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="qwen3:8b", help="Ollama model name")
    parser.add_argument("--scenario", default="all", choices=["all", "Tool call", "Chat", "Clarify"])
    args = parser.parse_args()

    # Ensure .env is loaded (sets OPENAI_BASE_URL=http://localhost:11434/v1)
    try:
        from wtb.constant import DOTENV_PATH
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=DOTENV_PATH, verbose=False, override=False)
    except Exception:
        pass
    os.environ.setdefault("OPENAI_BASE_URL", "http://localhost:11434/v1")
    os.environ.setdefault("OPENAI_API_KEY", "ollama")

    print(f"=== WTB-MAS smoke test  (backbone={args.backbone}) ===")
    orch = WTBMASOrchestrator(args.backbone, temperature=0.0)
    orch.reset_session()

    scenarios = SCENARIOS if args.scenario == "all" else [s for s in SCENARIOS if s[0] == args.scenario]

    for turn_idx, (label, messages) in enumerate(scenarios):
        print(f"\n── Turn {turn_idx} [{label}] ──")
        print(f"User: {messages[-1]['content']}")
        try:
            result = orch.handle_turn(messages, SAMPLE_TOOLS, turn_idx)
        except Exception as exc:
            print(f"  ✗ ERROR: {exc.__class__.__name__}: {exc}")
            return 1
        print(f"Intent  : {result['intent']}")
        if result["tool_calls"]:
            for tc in result["tool_calls"]:
                fn = tc.get("function", {})
                print(f"Tool    : {fn.get('name')}({fn.get('arguments')})")
        if result["content"]:
            print(f"Content : {result['content'][:200]}")
        # Pretty-print agent log
        print(f"Latency : {result['total_latency']:.2f}s")
        for agent, info in result["agent_log"].items():
            line = f"  {agent}: "
            if isinstance(info, dict):
                line += json.dumps({k: v for k, v in info.items() if k != "latency"}, ensure_ascii=False)[:200]
            else:
                line += str(info)[:200]
            print(line)

    print("\n=== smoke test complete ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
