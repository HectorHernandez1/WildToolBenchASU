"""MCA — Memory / Coreference Agent.

Two responsibilities:
  1. Maintain the entity tracker (delegated to SharedMemoryStore — pure code).
  2. Resolve cross-turn references in the user's message before tool dispatch.

For the resolution step we use a heuristic-first approach: we prepend a
"[Context from prior turns: ...]" line constructed from the entity tracker.
This is the same intervention validated in the eval-improvements branch
which produced +6 to +12pp on Long-Range Dependency turns.

Targets CGF (Contextual Grounding) — 315 errors across all models.
"""
from __future__ import annotations

from ..memory import SharedMemoryStore


class MemoryAgent:
    def __init__(self, memory: SharedMemoryStore):
        self._memory = memory

    def resolve_references(self, user_message: str) -> str:
        """Return the user message prepended with a compact context summary."""
        ctx = self._memory.context_summary()
        if not ctx:
            return user_message
        return f"{ctx}\n{user_message}"

    def update_from_action(self, tool_name: str, arguments: dict, observation, turn_idx: int) -> None:
        self._memory.update_from_tool_call(tool_name, arguments, observation, turn_idx)

    @property
    def memory(self) -> SharedMemoryStore:
        return self._memory
