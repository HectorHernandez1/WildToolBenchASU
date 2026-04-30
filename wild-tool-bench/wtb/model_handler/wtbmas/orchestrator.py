"""WTB-MAS orchestrator (v2 — wrapper-style).

Lessons from the v1 experiment (qwen3:8b dropped from 36% baseline → 2.8%):

  - The aggressive Planner system prompt was OVERRIDING the LLM's natural
    "I'm done with tools, let me return text now" behavior. The harness
    interprets a content-only response as `prepare_to_answer({"answer_type":
    "tool"})` — but our PA kept re-emitting tool calls instead.
  - The Chat branch returned empty content+null tool_calls, which the
    harness rejects.
  - The Critic's retry was injecting "previous attempt was incomplete or
    wrong" prompts that further confused the LLM.

v2 design — WTB-MAS as a thin wrapper around the same call the baseline
makes, NOT a replacement for the LLM's planning:

  Per-call flow (the eval harness drives multi-step on its own):

  ┌─ Is this a continuation step? (last msg = tool observation) ─┐
  │  YES → pass straight through to LLM, no agent wrappers       │
  │  NO (fresh user turn) → run intent routing                   │
  └──────────────────────────────────────────────────────────────┘

  Fresh user turn → IR classifies:
    - CHAT    : call LLM with NO tools → return content
    - CLARIFY : DA detects ambiguity → emit ask_user_for_required_parameters
                (or fall through to TOOL if DA disagrees)
    - TOOL    : MCA prepends entity context → call LLM with tools
                AV repairs tool-call arguments before returning

The Critic and the heavy Planner system prompt are gone. The LLM IS the
planner now — we just route, validate, and (lightly) ground references.
"""
from __future__ import annotations

import json
import secrets
import threading
import time
from typing import Any

from .agents.base import LLMClient
from .agents import (
    ArgumentValidator,
    DisambiguationAgent,
    IntentRouter,
    MemoryAgent,
)
from .memory import SharedMemoryStore


_DEFAULT_LLM_TIMEOUT = 300


class WTBMASOrchestrator:
    """Wrapper-style orchestrator (v2)."""

    def __init__(self, model_name: str, temperature: float = 0.0):
        self.model_name = model_name
        self.temperature = temperature
        llm = LLMClient(model_name, temperature)
        self._llm = llm
        self.memory = SharedMemoryStore()
        self.ir = IntentRouter(llm)
        self.da = DisambiguationAgent(llm)
        self.mca = MemoryAgent(self.memory)
        self.av = ArgumentValidator()

    def reset_session(self) -> None:
        self.memory.reset()

    # ── public entry point ───────────────────────────────────────
    def handle_turn(self, messages: list[dict], tools: list[dict], turn_idx: int) -> dict:
        agent_log: dict[str, Any] = {}
        total_latency = 0.0

        last = messages[-1] if messages else {}
        last_role = last.get("role", "")

        # ─────────────────────────────────────────────────────────
        # CONTINUATION STEP — model has just received a tool result.
        # Pass straight to the LLM with the same tools; let it decide
        # whether to call another tool or return content text.
        # ─────────────────────────────────────────────────────────
        if last_role == "tool":
            agent_log["mode"] = "continuation"
            resp = self._llm_call(messages, tools=tools)
            agent_log["llm"] = {"latency": resp["latency"], "had_tool_calls": bool(resp["tool_calls"])}
            # AV repair on any tool calls
            if resp["tool_calls"]:
                resp["tool_calls"], av_log = self._repair_tool_calls(resp["tool_calls"], tools)
                agent_log["AV"] = av_log
                # Update memory tracker
                for tc in resp["tool_calls"]:
                    self.mca.update_from_action(
                        (tc.get("function") or {}).get("name", ""),
                        _decode_args((tc.get("function") or {}).get("arguments")),
                        None,
                        turn_idx,
                    )
            return self._finalize(
                tool_calls=resp["tool_calls"],
                content=resp["content"],
                intent="CONTINUE",
                agent_log=agent_log,
                latency=resp["latency"],
                input_tokens=resp["input_token"],
                output_tokens=resp["output_token"],
            )

        # ─────────────────────────────────────────────────────────
        # FRESH USER TURN — v3.1: trust the LLM to decide whether to
        # call a tool, return text, or do something else. Skip the
        # explicit IR/DA branching that caused 0% accuracy on Clarify
        # and a regression on Chat in v3.0.
        #
        # We keep MCA (entity-context grounding) and AV (arg repair)
        # because they were demonstrably net-positive on Single-Tool
        # turns (+14pp over baseline).
        # ─────────────────────────────────────────────────────────
        user_message = str(last.get("content", "")) if last_role == "user" else ""
        agent_log["mode"] = "uniform"
        agent_log["IR"] = {"intent": "SKIPPED", "reason": "v3.1: always pass through to LLM"}

        # MCA: prepend entity context to user message (if there are entities)
        grounded_user = self.mca.resolve_references(user_message)
        agent_log["MCA"] = {"context_injected": grounded_user != user_message}

        # Build the messages for the LLM call. Replace the last user message
        # with the grounded version (if changed).
        call_messages = list(messages)
        if grounded_user != user_message:
            call_messages = list(messages[:-1]) + [{"role": "user", "content": grounded_user}]

        resp = self._llm_call(call_messages, tools=tools)
        total_latency += resp["latency"]
        agent_log["LLM"] = {"latency": resp["latency"], "n_tool_calls": len(resp["tool_calls"] or [])}

        # AV repair tool-call arguments
        if resp["tool_calls"]:
            resp["tool_calls"], av_log = self._repair_tool_calls(resp["tool_calls"], tools)
            agent_log["AV"] = av_log
            # Update memory tracker
            for tc in resp["tool_calls"]:
                self.mca.update_from_action(
                    (tc.get("function") or {}).get("name", ""),
                    _decode_args((tc.get("function") or {}).get("arguments")),
                    None,
                    turn_idx,
                )

        return self._finalize(
            tool_calls=resp["tool_calls"],
            content=resp["content"],
            intent="UNIFORM",
            agent_log=agent_log,
            latency=total_latency,
            input_tokens=resp["input_token"],
            output_tokens=resp["output_token"],
        )

    # ── helpers ──────────────────────────────────────────────────
    def _llm_call(self, messages: list[dict], tools: list[dict] | None) -> dict:
        """Make a single OpenAI/Ollama chat-completion call with a wall-clock timeout.

        Returns dict with keys: content, tool_calls, input_token, output_token, latency.
        """
        result_box: list = [None]
        error_box: list = [None]

        def _call():
            try:
                start = time.time()
                kwargs: dict[str, Any] = {
                    "model": self._llm.model_name,
                    "messages": messages,
                    "temperature": self._llm.temperature,
                    "extra_body": {"keep_alive": "30m"},
                }
                if tools:
                    kwargs["tools"] = tools
                resp = self._llm.client.chat.completions.create(**kwargs)
                result_box[0] = (resp, time.time() - start)
            except Exception as exc:  # noqa: BLE001
                error_box[0] = exc

        t = threading.Thread(target=_call, daemon=True)
        t.start()
        t.join(timeout=_DEFAULT_LLM_TIMEOUT)
        if t.is_alive():
            return {
                "content": f"[wtbmas LLM call timeout after {_DEFAULT_LLM_TIMEOUT}s]",
                "tool_calls": None,
                "input_token": 0,
                "output_token": 0,
                "latency": float(_DEFAULT_LLM_TIMEOUT),
            }
        if error_box[0] is not None:
            return {
                "content": f"[wtbmas LLM error: {error_box[0]}]",
                "tool_calls": None,
                "input_token": 0,
                "output_token": 0,
                "latency": 0.0,
            }
        api_response, latency = result_box[0]
        parsed = json.loads(api_response.json())
        message = parsed["choices"][0]["message"]
        return {
            "content": message.get("content"),
            "tool_calls": message.get("tool_calls"),
            "input_token": parsed.get("usage", {}).get("prompt_tokens", 0),
            "output_token": parsed.get("usage", {}).get("completion_tokens", 0),
            "latency": latency,
        }

    def _repair_tool_calls(self, tool_calls: list[dict], tools: list[dict]) -> tuple[list[dict], list[dict]]:
        """Run the AV over each tool call. Return (possibly-repaired tool_calls, log)."""
        out: list[dict] = []
        log: list[dict] = []
        for tc in tool_calls:
            fn = tc.get("function") or {}
            name = fn.get("name", "")
            raw_args = fn.get("arguments")
            args_dict = _decode_args(raw_args)
            ok, errors, repaired = self.av.check(name, args_dict, tools)
            log.append({"name": name, "ok": ok, "errors": errors[:5]})
            if errors:
                # Re-encode repaired args back to JSON string (OpenAI tool_calls format)
                tc_out = dict(tc)
                tc_out["function"] = dict(fn)
                tc_out["function"]["arguments"] = json.dumps(repaired, ensure_ascii=False)
                out.append(tc_out)
            else:
                out.append(tc)
        return out, log

    @staticmethod
    def _build_history_excerpt(prior_messages: list[dict]) -> str:
        lines = []
        for m in prior_messages[-6:]:
            role = m.get("role", "")
            c = m.get("content")
            if c is None:
                tc = m.get("tool_calls") or []
                if tc:
                    names = [(t.get("function") or {}).get("name", "?") for t in tc]
                    c = f"<tool_calls: {names}>"
                else:
                    c = ""
            lines.append(f"{role}: {str(c)[:200]}")
        return "\n".join(lines)

    @staticmethod
    def _fake_tool_call(name: str, arguments: dict | str) -> dict:
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
        tool_calls,
        content,
        intent: str,
        agent_log: dict,
        latency: float,
        input_tokens: int,
        output_tokens: int,
    ) -> dict:
        return {
            "tool_calls": tool_calls,
            "content": content,
            "intent": intent,
            "agent_log": agent_log,
            "total_latency": latency,
            "input_token": input_tokens,
            "output_token": output_tokens,
        }


def _decode_args(args_text):
    if isinstance(args_text, dict):
        return args_text
    if isinstance(args_text, str):
        try:
            return json.loads(args_text)
        except (json.JSONDecodeError, TypeError):
            return {}
    return {}
