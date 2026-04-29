"""WTB-MAS orchestrator.

Coordinates the six agents per the pseudocode in the Phase 2 proposal:

    User Turn → IR → ┬→ Tool: PA (uses MCA-grounded query) → AV per call → emit
                     ├→ Clarify: DA → ask_user_for_required_parameters
                     └→ Chat: direct response
                                 ↓
                     All paths → CR → final action

The orchestrator outputs a synthesized OpenAI-style chat-completion *dict*
that the WTBMASHandler returns to the evaluation harness.
"""
from __future__ import annotations

import json
import time
from typing import Any

from .agents.base import LLMClient
from .agents import (
    ArgumentValidator,
    CriticAgent,
    DisambiguationAgent,
    IntentRouter,
    MemoryAgent,
    PlannerAgent,
)
from .memory import SharedMemoryStore


class WTBMASOrchestrator:
    """Single-backbone orchestrator (Strategy A).

    Each LLM-backed agent shares the same Ollama-served model with different
    system prompts. The shared memory store persists across turns within one
    session and is reset by the handler when a new session starts.
    """

    MAX_CRITIC_RETRIES = 1

    def __init__(self, model_name: str, temperature: float = 0.0):
        self.model_name = model_name
        self.temperature = temperature
        llm = LLMClient(model_name, temperature)
        self.memory = SharedMemoryStore()
        self._llm = llm
        self.ir = IntentRouter(llm)
        self.da = DisambiguationAgent(llm)
        self.pa = PlannerAgent(llm)
        self.mca = MemoryAgent(self.memory)
        self.av = ArgumentValidator()
        self.cr = CriticAgent(llm)

    def reset_session(self) -> None:
        self.memory.reset()

    def handle_turn(self, messages: list[dict], tools: list[dict], turn_idx: int) -> dict:
        """Run the 6-agent pipeline for one turn. Returns a dict with:

            {
              "tool_calls": [...openai_tool_call_dicts...] | None,
              "content": str | None,
              "intent": "TOOL"|"CLARIFY"|"CHAT",
              "agent_log": {...},
              "input_token": int,
              "output_token": int,
              "total_latency": float,
            }
        """
        agent_log: dict[str, Any] = {}
        total_latency = 0.0

        # Pull latest user message
        user_message = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                user_message = str(m.get("content", ""))
                break

        # Build a compact history excerpt for IR/DA prompts
        history_lines = []
        for m in messages[-6:-1]:  # last few before the current user turn
            role = m.get("role", "")
            c = m.get("content")
            if c is None:
                # tool/assistant w/ tool_calls
                tc = m.get("tool_calls") or []
                if tc:
                    names = [(t.get("function") or {}).get("name", "?") for t in tc]
                    c = f"<tool_calls: {names}>"
                else:
                    c = ""
            history_lines.append(f"{role}: {str(c)[:200]}")
        history_excerpt = "\n".join(history_lines)

        # ── 1. Intent Router (IR) ─────────────────────────
        intent, ir_reason, ir_lat = self.ir.classify(user_message, history_excerpt)
        total_latency += ir_lat
        agent_log["IR"] = {"intent": intent, "reason": ir_reason, "latency": ir_lat}

        if intent == "CHAT":
            # Direct response — no tool, no further agent calls beyond critic
            return self._finalize(
                user_message=user_message,
                tool_calls=None,
                content="",
                agent_log=agent_log,
                latency=total_latency,
                input_tokens=0,
                output_tokens=0,
            )

        # ── 2. Disambiguation (DA) — only if intent is CLARIFY OR proactively ──
        if intent == "CLARIFY":
            needs, missing, da_reason, da_lat = self.da.detect(user_message, tools, history_excerpt)
            total_latency += da_lat
            agent_log["DA"] = {
                "needs_clarification": needs,
                "missing": missing,
                "reason": da_reason,
                "latency": da_lat,
            }
            if needs:
                clarify_action = DisambiguationAgent.to_action(missing or [])
                fake_tc = self._fake_tool_call(clarify_action["name"], clarify_action["arguments"])
                return self._finalize(
                    user_message=user_message,
                    tool_calls=[fake_tc],
                    content=None,
                    agent_log=agent_log,
                    latency=total_latency,
                    input_tokens=0,
                    output_tokens=0,
                )
            # DA decided no clarification needed → fall through to tool branch

        # ── 3. Memory / Coreference (MCA) ─────────────────
        grounded_user = self.mca.resolve_references(user_message)
        agent_log["MCA"] = {"grounded_message_excerpt": grounded_user[:200]}

        # ── 4. Planner (PA) ───────────────────────────────
        # Pass prior assistant + tool messages so the planner has full conversation context
        prior = [m for m in messages if m.get("role") != "system"][:-1]
        tool_calls, content_text, raw_message, pa_lat = self.pa.plan(
            grounded_user_message=grounded_user,
            tool_schemas=tools,
            prior_assistant_messages=prior,
        )
        total_latency += pa_lat
        agent_log["PA"] = {
            "n_tool_calls": len(tool_calls),
            "tool_names": [tc.get("name") for tc in tool_calls],
            "latency": pa_lat,
        }

        # ── 5. Argument Validator (AV) ────────────────────
        repaired_calls = []
        av_log = []
        for tc in tool_calls:
            ok, errors, repaired = self.av.check(tc.get("name", ""), tc.get("arguments", {}), tools)
            av_log.append({"name": tc.get("name"), "ok": ok, "errors": errors[:5]})
            tc_out = dict(tc)
            tc_out["arguments"] = repaired
            repaired_calls.append(tc_out)
        agent_log["AV"] = av_log
        tool_calls = repaired_calls

        # Update memory with these tool calls (no observations available — eval harness handles them)
        for tc in tool_calls:
            self.mca.update_from_action(tc.get("name", ""), tc.get("arguments", {}), None, turn_idx)

        # Synthesize OpenAI-style tool_calls
        synthesized_tool_calls = [self._fake_tool_call(tc["name"], tc["arguments"]) for tc in tool_calls]

        # ── 6. Critic (CR) ────────────────────────────────
        ok, retry, cr_reason, cr_lat = self.cr.review(
            user_message=user_message,
            planned_actions=tool_calls,
            had_text_response=bool(content_text),
        )
        total_latency += cr_lat
        agent_log["CR"] = {"ok": ok, "retry": retry, "reason": cr_reason, "latency": cr_lat}

        # Single retry if critic asks (best-effort; we don't loop more than once)
        if retry and not ok:
            tool_calls2, content2, _, pa_lat2 = self.pa.plan(
                grounded_user_message=grounded_user + "\n[Critic retry: previous attempt was incomplete or wrong.]",
                tool_schemas=tools,
                prior_assistant_messages=prior,
            )
            total_latency += pa_lat2
            agent_log["PA_retry"] = {"n_tool_calls": len(tool_calls2), "latency": pa_lat2}
            if tool_calls2:
                tool_calls = tool_calls2
                synthesized_tool_calls = [self._fake_tool_call(tc["name"], tc["arguments"]) for tc in tool_calls]
                content_text = content2

        return self._finalize(
            user_message=user_message,
            tool_calls=synthesized_tool_calls if synthesized_tool_calls else None,
            content=content_text,
            agent_log=agent_log,
            latency=total_latency,
            input_tokens=0,
            output_tokens=0,
        )

    # ── helpers ─────────────────────────────────────────
    @staticmethod
    def _fake_tool_call(name: str, arguments: dict | str) -> dict:
        """Build an OpenAI-style tool_call dict that the eval harness will accept."""
        import secrets
        if isinstance(arguments, dict):
            arg_text = json.dumps(arguments, ensure_ascii=False)
        else:
            arg_text = str(arguments)
        return {
            "id": "call_" + secrets.token_hex(6),
            "type": "function",
            "index": 0,
            "function": {"name": name, "arguments": arg_text},
        }

    @staticmethod
    def _finalize(
        *,
        user_message: str,
        tool_calls: list | None,
        content: str | None,
        agent_log: dict,
        latency: float,
        input_tokens: int,
        output_tokens: int,
    ) -> dict:
        return {
            "tool_calls": tool_calls,
            "content": content,
            "intent": agent_log.get("IR", {}).get("intent"),
            "agent_log": agent_log,
            "total_latency": latency,
            "input_token": input_tokens,
            "output_token": output_tokens,
        }
