"""Local LLM backends and the note-taking prompts."""

from lectern.config.models import OllamaConfig
from lectern.llm.base import (
    LLMBackend,
    LLMError,
    LLMHealth,
    LLMModel,
    LLMUnavailableError,
)
from lectern.llm.ollama import OllamaBackend

__all__ = [
    "LLMBackend",
    "LLMError",
    "LLMHealth",
    "LLMModel",
    "LLMUnavailableError",
    "OllamaBackend",
    "OllamaConfig",
    "build_backend",
]


def build_backend(config: OllamaConfig) -> LLMBackend:
    """Instantiate the configured local LLM backend."""
    return OllamaBackend(
        config.host,
        timeout=config.request_timeout_seconds,
        keep_alive=config.keep_alive,
    )
