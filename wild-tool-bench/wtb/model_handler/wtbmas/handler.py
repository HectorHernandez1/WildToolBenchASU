"""WTBMASHandler — wraps the WTB-MAS orchestrator behind the BaseHandler API.

Plugs into the existing WildToolBench evaluation pipeline. Each per-turn call
to `_request_tool_call` runs the full 6-agent pipeline; the response is shaped
to look like a normal OpenAI ChatCompletion so the rest of the eval harness
treats it identically.

To register, append to wtb/model_handler/handler_map.py:

    from .wtbmas import WTBMASHandler
    HANDLER_MAP["wtbmas:qwen3:8b"]  = WTBMASHandler
    HANDLER_MAP["wtbmas:qwen3:14b"] = WTBMASHandler

The model name passed to WTBMASHandler.__init__ must use the prefix
"wtbmas:<backbone>" so the orchestrator knows which Ollama model to use.
"""
from __future__ import annotations

import os
import time
from typing import Any

from wtb.model_handler.base_handler import BaseHandler

from .orchestrator import WTBMASOrchestrator


_BACKBONE_PREFIX = "wtbmas:"


class WTBMASHandler(BaseHandler):
    """BaseHandler subclass that runs WTB-MAS instead of a single LLM call.

    The backbone Ollama model is parsed from the model_name:
        "wtbmas:qwen3:14b" -> backbone="qwen3:14b"
    """

    def __init__(self, model_name: str, temperature: float = 0.0):
        super().__init__(model_name, temperature)
        if not model_name.startswith(_BACKBONE_PREFIX):
            raise ValueError(
                f"WTBMASHandler model_name must start with '{_BACKBONE_PREFIX}', got '{model_name}'"
            )
        backbone = model_name[len(_BACKBONE_PREFIX):]
        self.backbone = backbone
        self.orchestrator = WTBMASOrchestrator(backbone, temperature=temperature)
        self._last_session_id: str | None = None

    # ─────────────────────────────────────────────────────
    # BaseHandler API
    # ─────────────────────────────────────────────────────
    def _request_tool_call(self, inference_data: dict) -> tuple[Any, float]:
        """One per-turn invocation. Returns (synthesized_response_dict, latency)."""
        # Reset memory at start of a new session
        session_id = inference_data.get("test_entry_id") or inference_data.get("id")
        if session_id and session_id != self._last_session_id:
            self.orchestrator.reset_session()
            self._last_session_id = session_id

        messages = inference_data.get("messages", [])
        tools = inference_data.get("tools", [])
        turn_idx = inference_data.get("task_idx", 0)

        start = time.time()
        result = self.orchestrator.handle_turn(messages, tools, turn_idx)
        latency = time.time() - start

        # Synthesize an OpenAI ChatCompletion-shaped response (dict).
        # _parse_api_response below consumes this dict (we override to skip .json()).
        synthesized = {
            "_wtbmas_synthetic": True,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": result.get("content"),
                        "tool_calls": result.get("tool_calls"),
                        "reasoning_content": None,
                    },
                    "finish_reason": "tool_calls" if result.get("tool_calls") else "stop",
                }
            ],
            "usage": {
                "prompt_tokens": result.get("input_token", 0),
                "completion_tokens": result.get("output_token", 0),
                "total_tokens": result.get("input_token", 0) + result.get("output_token", 0),
            },
            "_agent_log": result.get("agent_log", {}),
        }
        return synthesized, latency

    def _parse_api_response(self, api_response: Any) -> dict:
        """Mirror the OpenAIHandler parse, but accept either a dict (from us)
        or an OpenAI ChatCompletion (so this handler is safe to chain)."""
        if isinstance(api_response, dict):
            payload = api_response
        else:
            import json as _json
            payload = _json.loads(api_response.json())

        choice = payload["choices"][0]
        message = choice["message"]
        return {
            "reasoning_content": message.get("reasoning_content"),
            "content": message.get("content"),
            "tool_calls": message.get("tool_calls"),
            "input_token": payload.get("usage", {}).get("prompt_tokens", 0),
            "output_token": payload.get("usage", {}).get("completion_tokens", 0),
        }
