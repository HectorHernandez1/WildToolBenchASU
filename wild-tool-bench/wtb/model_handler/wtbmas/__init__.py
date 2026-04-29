"""WTB-MAS: Wild-Tool-Bench Multi-Agent System.

Six-agent pipeline that wraps Ollama-served LLMs with specialized roles for
intent routing, disambiguation, planning, memory/coreference, argument
validation, and critic-based retry.

See README.md for architecture details.
"""
from .handler import WTBMASHandler

__all__ = ["WTBMASHandler"]
