from .api_inference.oai import OpenAIHandler
from .api_inference.deepseek import DeepSeekAPIHandler
from .api_inference.hunyuan import HunYuanAPIHandler
from .api_inference.ollama import OllamaHandler
from .wtbmas import WTBMASHandler


api_inference_handler_map = {
    "gpt-4o-2024-11-20": OpenAIHandler,
    "deepseek-chat": DeepSeekAPIHandler,
    "hunyuan-2.0-thinking-20251109": HunYuanAPIHandler,
    "hunyuan-2.0-instruct-20251111": HunYuanAPIHandler,
    # Ollama local models (OpenAI-compatible endpoint, with per-request timeout)
    "qwen3:8b": OllamaHandler,
    "qwen3:14b": OllamaHandler,
    "qwen3:32b": OllamaHandler,
    "gemma4:31b": OllamaHandler,
    # WTB-MAS multi-agent system. Backbone is parsed from the prefix.
    "wtbmas:qwen3:8b":  WTBMASHandler,
    "wtbmas:qwen3:14b": WTBMASHandler,
    "wtbmas:qwen3:32b": WTBMASHandler,
    "wtbmas:gemma4:31b": WTBMASHandler,
}

HANDLER_MAP = {**api_inference_handler_map}
