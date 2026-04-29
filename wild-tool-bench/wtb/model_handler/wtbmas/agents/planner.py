"""PA — Planner Agent.

Decomposes the user request into one or more tool calls and emits them via
the model's native function calling. Targets CPF (Compositional Planning) —
the largest failure category overall (771 errors).

For multi-tool tasks the planner emits multiple tool_calls in a single
response; the executor handles dependency-aware sequencing afterwards (the
existing eval harness already handles tool_call ordering via the
benchmark's optimal-path graph).
"""
from __future__ import annotations

import json
import os
import threading
import time
from typing import Any

from .base import LLMClient


_DEFAULT_PLANNER_TIMEOUT = int(os.getenv("OLLAMA_REQUEST_TIMEOUT", "300"))


_SYSTEM_PROMPT = """You are a planner that calls tools to fulfill user requests.

Rules:
  1. Think briefly before calling tools, but do NOT include reasoning in user-visible output.
  2. If the request requires multiple INDEPENDENT tool calls, emit them in parallel
     (multiple tool_calls in one response).
  3. If the request requires SEQUENTIAL tool calls (one depends on another's output),
     emit only the FIRST step now; later steps will be planned after observing results.
  4. Use ONLY parameters defined in the tool schema. Do NOT invent parameters.
  5. Use entity values from the conversation history where applicable
     (e.g., reuse city, date, IDs already mentioned).
  6. If no tool is needed, respond with a brief text answer.

When you are done with all tool calls and have all needed information,
call the special tool `prepare_to_answer` to signal completion."""


class PlannerAgent:
    def __init__(self, llm: LLMClient):
        self._llm = llm

    def plan(
        self,
        grounded_user_message: str,
        tool_schemas: list[dict],
        prior_assistant_messages: list[dict] | None = None,
    ) -> tuple[list[dict], str | None, dict, float]:
        """
        Returns (tool_calls, content_text, raw_message, latency).
        tool_calls is a list of {name, arguments} dicts.
        """
        # Build messages: system prompt + (optional) prior turns + grounded user msg
        messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
        if prior_assistant_messages:
            messages.extend(prior_assistant_messages)
        messages.append({"role": "user", "content": grounded_user_message})

        # Use the existing OpenAI-compatible function-calling path with a hard
        # wall-clock timeout (the OpenAI SDK's default doesn't help when Ollama
        # streams tokens slowly or hangs mid-generation).
        result_box: list = [None]
        error_box: list = [None]

        def _call():
            try:
                start = time.time()
                resp = self._llm.client.chat.completions.create(
                    model=self._llm.model_name,
                    messages=messages,
                    temperature=self._llm.temperature,
                    tools=tool_schemas,
                    extra_body={"keep_alive": "30m"},
                )
                result_box[0] = (resp, time.time() - start)
            except Exception as exc:  # noqa: BLE001
                error_box[0] = exc

        t = threading.Thread(target=_call, daemon=True)
        t.start()
        t.join(timeout=_DEFAULT_PLANNER_TIMEOUT)
        if t.is_alive():
            return [], f"planner_error: timeout after {_DEFAULT_PLANNER_TIMEOUT}s", {}, float(_DEFAULT_PLANNER_TIMEOUT)
        if error_box[0] is not None:
            return [], f"planner_error: {error_box[0]}", {}, 0.0
        api_response, _wall = result_box[0]

        parsed = json.loads(api_response.json())
        choice = parsed["choices"][0]
        message = choice["message"]
        latency = float(parsed.get("usage", {}).get("total_time", _wall))

        tool_calls_raw = message.get("tool_calls") or []
        tool_calls: list[dict] = []
        for tc in tool_calls_raw:
            fn = tc.get("function", {})
            name = fn.get("name", "")
            args_text = fn.get("arguments", "{}")
            try:
                args = json.loads(args_text) if isinstance(args_text, str) else args_text
            except json.JSONDecodeError:
                args = {}
            tool_calls.append({"name": name, "arguments": args, "raw": tc})

        return tool_calls, message.get("content"), message, latency
