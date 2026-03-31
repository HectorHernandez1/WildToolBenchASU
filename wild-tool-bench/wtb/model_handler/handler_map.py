from .api_inference.oai import OpenAIHandler
from .api_inference.deepseek import DeepSeekAPIHandler
from .api_inference.hunyuan import HunYuanAPIHandler
from .api_inference.ollama import OllamaHandler


api_inference_handler_map = {
    "gpt-4o-2024-11-20": OpenAIHandler,
    "deepseek-chat": DeepSeekAPIHandler,
    "hunyuan-2.0-thinking-20251109": HunYuanAPIHandler,
    "hunyuan-2.0-instruct-20251111": HunYuanAPIHandler,
    # Ollama local models (OpenAI-compatible endpoint, with per-request timeout)
    "qwen3:8b": OllamaHandler,
    "qwen3:14b": OllamaHandler,
    "qwen3:32b": OllamaHandler,
}

HANDLER_MAP = {**api_inference_handler_map}
