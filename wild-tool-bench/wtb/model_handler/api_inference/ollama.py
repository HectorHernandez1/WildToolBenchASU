import os
import threading

from wtb.model_handler.api_inference.oai import OpenAIHandler
from openai import OpenAI


# Default per-request wall-clock timeout in seconds (5 minutes).
# Override with OLLAMA_REQUEST_TIMEOUT env var.
DEFAULT_REQUEST_TIMEOUT = 300


class OllamaHandler(OpenAIHandler):
    def __init__(self, model_name, temperature):
        super().__init__(model_name, temperature)
        self.request_timeout = int(os.getenv("OLLAMA_REQUEST_TIMEOUT", DEFAULT_REQUEST_TIMEOUT))
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY", "ollama"),
            base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1"),
        )

    def _request_tool_call(self, inference_data):
        result = [None]
        error = [None]

        def target():
            try:
                result[0] = super(OllamaHandler, self)._request_tool_call(inference_data)
            except Exception as e:
                error[0] = e

        t = threading.Thread(target=target, daemon=True)
        t.start()
        t.join(timeout=self.request_timeout)

        if t.is_alive():
            # Thread still running — it's a daemon so it won't block process exit.
            raise TimeoutError(
                f"Ollama request exceeded {self.request_timeout}s wall-clock timeout"
            )
        if error[0] is not None:
            raise error[0]
        return result[0]
