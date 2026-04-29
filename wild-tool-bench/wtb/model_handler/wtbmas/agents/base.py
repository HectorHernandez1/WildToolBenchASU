"""Shared LLM-call helper for all LLM-backed agents.

All agents in WTB-MAS that need an LLM call into a single Ollama-served
backbone via the OpenAI-compatible client. Each agent supplies its own
system prompt; the model is shared.
"""
from __future__ import annotations

import json
import os
import threading
import time
from typing import Any

from openai import OpenAI


_DEFAULT_TIMEOUT = int(os.getenv("OLLAMA_REQUEST_TIMEOUT", "300"))


class LLMClient:
    """Thin wrapper around the OpenAI/Ollama client with a wall-clock timeout."""

    def __init__(self, model_name: str, temperature: float = 0.0):
        self.model_name = model_name
        self.temperature = temperature
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY", "ollama"),
            base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1"),
        )

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        json_mode: bool = False,
        tools: list | None = None,
        timeout: int = _DEFAULT_TIMEOUT,
        keep_alive: str = "30m",
    ) -> tuple[dict, float]:
        """One synchronous call. Returns (parsed_response_dict, latency_seconds)."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if tools:
            kwargs["tools"] = tools
        kwargs["extra_body"] = {"keep_alive": keep_alive}

        result: list = [None]
        error: list = [None]

        def target():
            try:
                start = time.time()
                resp = self.client.chat.completions.create(**kwargs)
                end = time.time()
                result[0] = (resp, end - start)
            except Exception as exc:  # noqa: BLE001
                error[0] = exc

        t = threading.Thread(target=target, daemon=True)
        t.start()
        t.join(timeout=timeout)
        if t.is_alive():
            raise TimeoutError(f"LLM call exceeded {timeout}s")
        if error[0] is not None:
            raise error[0]

        api_response, latency = result[0]
        parsed = json.loads(api_response.json())
        return parsed, latency


def safe_json_loads(text: str, default: Any = None) -> Any:
    """Robust JSON parse — strips code fences and falls back to a default."""
    if not text:
        return default
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find the first {...} or [...]
        for opener, closer in [("{", "}"), ("[", "]")]:
            i = text.find(opener)
            j = text.rfind(closer)
            if i != -1 and j != -1 and j > i:
                try:
                    return json.loads(text[i:j + 1])
                except json.JSONDecodeError:
                    pass
        return default
