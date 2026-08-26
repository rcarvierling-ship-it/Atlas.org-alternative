"""LLM backend interface.

Only local backends are permitted. Nothing in Lectern may fall back to a hosted
API: if the local model is unavailable the correct behaviour is to keep
transcribing and tell the user that note generation is paused.
"""

from __future__ import annotations

import abc
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class LLMModel:
    """An installed model as reported by the backend."""

    name: str
    size_bytes: int | None = None
    family: str = ""
    parameter_size: str = ""
    quantization: str = ""

    @property
    def size_label(self) -> str:
        if not self.size_bytes:
            return "—"
        gigabytes = self.size_bytes / 1_000_000_000
        if gigabytes >= 1:
            return f"{gigabytes:.1f} GB"
        return f"{self.size_bytes / 1_000_000:.0f} MB"

    @property
    def detail(self) -> str:
        bits = [bit for bit in (self.parameter_size, self.quantization) if bit]
        return " · ".join(bits)


@dataclass(slots=True)
class LLMHealth:
    """Backend availability, as shown on the Home screen and in doctor."""

    available: bool
    detail: str = ""
    version: str = ""
    models: list[LLMModel] = field(default_factory=list)


class LLMError(RuntimeError):
    """The backend was unreachable or returned an unusable response."""


class LLMUnavailableError(LLMError):
    """The local model server is not running.

    Distinguished from a generic error because it is *expected* and recoverable:
    the note worker keeps retrying, and the transcript keeps flowing meanwhile.
    """


class LLMBackend(abc.ABC):
    """Generates text from a locally hosted model."""

    name: str = "base"

    @abc.abstractmethod
    async def health(self) -> LLMHealth:
        """Check whether the backend is reachable and list its models."""

    @abc.abstractmethod
    async def list_models(self) -> list[LLMModel]:
        """Installed models, ordered for presentation."""

    @abc.abstractmethod
    async def generate(
        self,
        prompt: str,
        *,
        model: str,
        system: str | None = None,
        json_schema: dict[str, Any] | None = None,
        temperature: float = 0.2,
        num_ctx: int | None = None,
        on_token: Callable[[str], None] | None = None,
    ) -> str:
        """Generate a completion.

        ``on_token`` receives incremental text so the UI can show progress;
        the return value is the complete response. Callers must only commit
        state once this returns — a partially streamed response is never valid.
        """

    @abc.abstractmethod
    async def close(self) -> None:
        """Release network resources."""
