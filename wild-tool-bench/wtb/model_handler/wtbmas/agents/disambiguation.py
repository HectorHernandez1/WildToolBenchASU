"""DA — Disambiguation Agent.

Detects under-specification in the user's request given the available tools'
required parameters. Targets AHF (Ambiguity Handling) — the largest failure
category for smaller models (197 errors for Qwen3-8B).

When ambiguity is detected, the orchestrator emits an
`ask_user_for_required_parameters` action instead of attempting a tool call.
"""
from __future__ import annotations

import json
from typing import Any

from .base import LLMClient, safe_json_loads


_SYSTEM_PROMPT = """You are an ambiguity detector for a tool-using assistant.

You will be given:
  1. The user's latest message
  2. A list of available tools (with their REQUIRED parameters)
  3. Conversation history (entities seen so far)

Your job: decide whether the user's message is too ambiguous to call a tool RIGHT NOW.

Common ambiguity patterns:
  - References without antecedents: "one of the topics", "the above", "that one"
  - Missing required parameters that cannot be inferred from history
  - Plural/numeric references with no concrete count: "those items"

Respond ONLY with valid JSON in this exact form:
{
  "needs_clarification": true | false,
  "missing_required_parameters": [
    {"tool_name": "<name>", "missing_required_parameters": ["<param1>", ...]}
  ],
  "rationale": "<one short sentence>"
}

If clarification is NOT needed, set needs_clarification=false and missing_required_parameters=[].
Do not include any other text."""


class DisambiguationAgent:
    def __init__(self, llm: LLMClient):
        self._llm = llm

    def detect(
        self,
        user_message: str,
        tool_schemas: list[dict],
        history_excerpt: str,
    ) -> tuple[bool, list[dict], str, float]:
        # Compact tool list for the prompt
        compact_tools = []
        for t in tool_schemas:
            fn = t.get("function", t)
            params = fn.get("parameters", {}) or {}
            compact_tools.append({
                "name": fn.get("name", ""),
                "description": (fn.get("description", "") or "")[:200],
                "required": params.get("required", []),
            })

        user_prompt_parts = [
            f"Available tools (compact):\n{json.dumps(compact_tools, ensure_ascii=False)[:2500]}",
        ]
        if history_excerpt:
            user_prompt_parts.append(f"\nConversation history & known entities:\n{history_excerpt}")
        user_prompt_parts.append(f"\nUser message: {user_message}")

        try:
            parsed, latency = self._llm.chat(
                _SYSTEM_PROMPT, "\n".join(user_prompt_parts), json_mode=True
            )
        except Exception:  # noqa: BLE001
            return False, [], "fallback:no_clarify", 0.0

        text = parsed["choices"][0]["message"].get("content", "")
        obj = safe_json_loads(text, {"needs_clarification": False})
        needs = bool(obj.get("needs_clarification", False))
        missing = obj.get("missing_required_parameters", []) or []
        if not isinstance(missing, list):
            missing = []
        return needs, missing, str(obj.get("rationale", "")), latency

    @staticmethod
    def to_action(missing_required_parameters: list[dict]) -> dict[str, Any]:
        """Build the ask_user_for_required_parameters action dict."""
        return {
            "name": "ask_user_for_required_parameters",
            "arguments": {"tool_list": missing_required_parameters},
        }
