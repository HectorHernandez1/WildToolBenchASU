"""CR — Critic Agent.

Final validator that reviews the executed plan vs. the original user request
and signals whether to retry. Cross-cutting: catches residual errors that
slipped past the upstream agents.

The retry budget is hard-capped at 1 (in the orchestrator) to prevent
runaway latency.
"""
from __future__ import annotations

import json
from typing import Any

from .base import LLMClient, safe_json_loads


_SYSTEM_PROMPT = """You are a critic reviewing whether an assistant correctly handled a user turn.

Inputs:
  - user_message: the user's request
  - planned_actions: list of tool calls the assistant made (may be empty for chat/clarify)
  - had_text_response: bool — whether the assistant produced text instead of/in addition to tools

Decide: does this response COMPLETELY address the user's intent and use tools correctly?

Respond ONLY with valid JSON:
{
  "ok": true | false,
  "retry_recommended": true | false,
  "reason": "<one short sentence>"
}

Be lenient — only fail when the response is clearly wrong or incomplete.
Do not include any other text."""


class CriticAgent:
    def __init__(self, llm: LLMClient):
        self._llm = llm

    def review(
        self,
        user_message: str,
        planned_actions: list[dict],
        had_text_response: bool,
    ) -> tuple[bool, bool, str, float]:
        """Return (ok, retry_recommended, reason, latency)."""
        payload = {
            "user_message": user_message,
            "planned_actions": [{"name": a.get("name"), "args": a.get("arguments")} for a in planned_actions],
            "had_text_response": had_text_response,
        }
        try:
            parsed, latency = self._llm.chat(
                _SYSTEM_PROMPT, json.dumps(payload, ensure_ascii=False), json_mode=True
            )
        except Exception:  # noqa: BLE001
            return True, False, "fallback:critic_pass", 0.0

        text = parsed["choices"][0]["message"].get("content", "")
        obj = safe_json_loads(text, {"ok": True, "retry_recommended": False})
        return (
            bool(obj.get("ok", True)),
            bool(obj.get("retry_recommended", False)),
            str(obj.get("reason", "")),
            latency,
        )
