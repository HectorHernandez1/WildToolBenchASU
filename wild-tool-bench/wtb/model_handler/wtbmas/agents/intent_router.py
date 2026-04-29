"""IR — Intent Router.

Classifies each user turn as TOOL / CLARIFY / CHAT before any tool dispatch.
Targets TDF (Tool Dispatch) errors, especially over-eager tool calls on Chat
turns observed at scale (Qwen3-32B has the worst Chat accuracy).
"""
from __future__ import annotations

from typing import Literal

from .base import LLMClient, safe_json_loads


Intent = Literal["TOOL", "CLARIFY", "CHAT"]


_SYSTEM_PROMPT = """You are an intent classifier for a tool-using assistant.

Given the user's latest message and the conversation history, classify the user's
intent into exactly ONE of three labels:

- TOOL: The user is asking for information or an action that REQUIRES calling a tool.
- CLARIFY: The user's request is ambiguous or refers to entities not yet specified
  (e.g., "one of the topics", "the above"); you need MORE INFORMATION before acting.
- CHAT: The user is making conversation, expressing thanks, or asking something that
  does NOT need a tool call (e.g., commentary, opinion, simple acknowledgment).

Respond ONLY with valid JSON in this exact form:
{"intent": "TOOL" | "CLARIFY" | "CHAT", "rationale": "<one short sentence>"}

Do not include any other text."""


class IntentRouter:
    def __init__(self, llm: LLMClient):
        self._llm = llm

    def classify(self, user_message: str, history_excerpt: str = "") -> tuple[Intent, str, float]:
        prompt_parts = []
        if history_excerpt:
            prompt_parts.append(f"Conversation history:\n{history_excerpt}\n")
        prompt_parts.append(f"User message: {user_message}")
        user_prompt = "\n".join(prompt_parts)

        try:
            parsed, latency = self._llm.chat(
                _SYSTEM_PROMPT, user_prompt, json_mode=True
            )
        except Exception as exc:  # noqa: BLE001
            # On any failure, default to TOOL (the prior baseline behavior)
            return "TOOL", f"fallback:{exc.__class__.__name__}", 0.0

        text = parsed["choices"][0]["message"].get("content", "")
        obj = safe_json_loads(text, {"intent": "TOOL", "rationale": "parse_fail"})
        intent = str(obj.get("intent", "TOOL")).upper()
        if intent not in ("TOOL", "CLARIFY", "CHAT"):
            intent = "TOOL"
        return intent, str(obj.get("rationale", "")), latency  # type: ignore[return-value]
