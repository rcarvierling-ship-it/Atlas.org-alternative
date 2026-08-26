"""Transcription backend interface and the transcript segment schema.

``TranscriptSegment`` is the unit of currency for the whole application: audio
produces segments, persistence appends them, the note scheduler batches them,
and the UI renders them. Only *final* segments are ever persisted or sent to
the LLM; partial hypotheses are display-only and are replaced in place.
"""

from __future__ import annotations

import abc
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

from lectern.utils.timefmt import utcnow


@dataclass(slots=True)
class TranscriptSegment:
    """One chunk of recognised speech, positioned relative to session start."""

    id: int
    start_time: float
    end_time: float
    text: str
    is_final: bool = True
    confidence: float | None = None
    created_at: datetime = field(default_factory=utcnow)

    @property
    def duration(self) -> float:
        return max(0.0, self.end_time - self.start_time)

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "start_time": round(self.start_time, 3),
            "end_time": round(self.end_time, 3),
            "text": self.text,
            "is_final": self.is_final,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> TranscriptSegment:
        created = data.get("created_at")
        return cls(
            id=int(data["id"]),
            start_time=float(data["start_time"]),
            end_time=float(data["end_time"]),
            text=str(data["text"]),
            is_final=bool(data.get("is_final", True)),
            confidence=data.get("confidence"),
            created_at=datetime.fromisoformat(created) if created else utcnow(),
        )


@dataclass(slots=True)
class TranscriptionHealth:
    """Snapshot of backend state for the status bar and error modals."""

    ready: bool
    detail: str = ""
    model: str = ""
    last_latency_ms: float | None = None


class TranscriptionError(RuntimeError):
    """Backend failed in a way the UI should surface to the user."""


class TranscriptionBackend(abc.ABC):
    """Turns 16 kHz mono float32 audio into transcript segments.

    Implementations must keep their model loaded for the lifetime of
    ``start()``/``stop()`` — reloading weights per chunk would blow the latency
    budget many times over.
    """

    name: str = "base"

    @abc.abstractmethod
    async def start(self) -> None:
        """Load the model / spawn the server. Raises ``TranscriptionError``."""

    @abc.abstractmethod
    async def stop(self) -> None:
        """Release the model and any child process. Must be idempotent."""

    @abc.abstractmethod
    async def transcribe(self, audio: np.ndarray, *, sample_rate: int = 16_000) -> str:
        """Transcribe one utterance of mono float32 audio and return its text."""

    @abc.abstractmethod
    def health(self) -> TranscriptionHealth:
        """Cheap, non-blocking status snapshot."""

    async def stream(self, chunks: AsyncIterator[np.ndarray]) -> AsyncIterator[str]:
        """Convenience adapter: transcribe an async stream of utterances."""
        async for chunk in chunks:
            yield await self.transcribe(chunk)
