import os
import time

from wtb.model_handler.api_inference.oai import OpenAIHandler
from wtb.model_handler.utils import retry_with_backoff
from openai import OpenAI, RateLimitError


# Default per-request timeout in seconds (5 minutes).
# Override with OLLAMA_REQUEST_TIMEOUT env var.
DEFAULT_REQUEST_TIMEOUT = 300


class OllamaHandler(OpenAIHandler):
    def __init__(self, model_name, temperature):
        super().__init__(model_name, temperature)
        timeout = int(os.getenv("OLLAMA_REQUEST_TIMEOUT", DEFAULT_REQUEST_TIMEOUT))
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY", "ollama"),
            base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1"),
            timeout=timeout,
        )
