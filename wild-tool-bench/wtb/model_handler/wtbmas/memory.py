"""Shared Memory Store for WTB-MAS.

Tracks entities (city=Chicago, endDate=2024-07-14, ...) extracted from prior
tool calls, plan state, and conversation history. Cleared at the start of each
new session.

Pure Python — no LLM. The MCA agent's *resolve_references* step can optionally
use an LLM, but the tracker itself is rule-based.
"""
from __future__ import annotations

from collections import deque
from typing import Any


class SharedMemoryStore:
    DEFAULT_ENTITY_CAPACITY = 20

    def __init__(self, entity_capacity: int = DEFAULT_ENTITY_CAPACITY):
        self._capacity = entity_capacity
        self._entities: deque[dict[str, Any]] = deque(maxlen=entity_capacity)
        self.tool_results: dict[str, Any] = {}
        self.plan_state: dict[str, Any] = {}
        self.turn_history: list[dict[str, Any]] = []

    def reset(self) -> None:
        self._entities.clear()
        self.tool_results.clear()
        self.plan_state.clear()
        self.turn_history.clear()

    def update_from_tool_call(self, tool_name: str, arguments: dict, observation: Any, turn_idx: int) -> None:
        if not isinstance(arguments, dict):
            return
        for key, value in arguments.items():
            if value is None or value == "" or isinstance(value, (list, dict)):
                continue
            self._entities.append({
                "name": key,
                "value": value,
                "turn": turn_idx,
                "tool": tool_name,
            })

    def record_observation(self, step_id: str, observation: Any) -> None:
        self.tool_results[step_id] = observation

    def append_turn(self, role: str, content: str) -> None:
        self.turn_history.append({"role": role, "content": content})

    @property
    def entities(self) -> list[dict[str, Any]]:
        return list(self._entities)

    def context_summary(self) -> str:
        if not self._entities:
            return ""
        seen: dict[str, str] = {}
        for ent in self._entities:
            seen[ent["name"]] = str(ent["value"])
        if not seen:
            return ""
        parts = ", ".join(f"{k}={v}" for k, v in seen.items())
        return f"[Context from prior turns: {parts}]"
