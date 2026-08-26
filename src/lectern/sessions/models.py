"""Session metadata and markers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from lectern.utils.timefmt import format_clock, utcnow


class SessionStatus(StrEnum):
    RECORDING = "recording"
    COMPLETE = "complete"
    #: Written to disk but never finalized — the app died mid-session.
    INCOMPLETE = "incomplete"
    #: Transcript and live notes exist, but final synthesis failed or was skipped.
    NEEDS_FINALIZATION = "needs_finalization"


class MarkerKind(StrEnum):
    IMPORTANT = "important"
    NOTE = "note"


@dataclass(slots=True)
class Marker:
    """A moment the student flagged, or a note they typed, while recording."""

    time: float
    kind: MarkerKind = MarkerKind.IMPORTANT
    text: str = ""
    created_at: datetime = field(default_factory=utcnow)

    @property
    def clock(self) -> str:
        return format_clock(self.time)

    @property
    def label(self) -> str:
        if self.kind is MarkerKind.NOTE:
            return self.text or "Note"
        return self.text or "Important"

    def to_json(self) -> dict[str, Any]:
        return {
            "time": round(self.time, 3),
            "kind": str(self.kind),
            "text": self.text,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Marker:
        created = data.get("created_at")
        return cls(
            time=float(data.get("time", 0.0) or 0.0),
            kind=MarkerKind(data.get("kind", "important")),
            text=str(data.get("text", "")),
            created_at=datetime.fromisoformat(created) if created else utcnow(),
        )


@dataclass
class SessionMeta:
    """Everything about a session except its transcript and notes."""

    id: str
    title: str
    course: str = ""
    created_at: datetime = field(default_factory=utcnow)
    ended_at: datetime | None = None
    duration_seconds: float = 0.0
    word_count: int = 0
    segment_count: int = 0
    whisper_model: str = ""
    ollama_model: str = ""
    audio_source: str = "microphone"
    status: SessionStatus = SessionStatus.RECORDING
    folder_path: str = ""
    has_audio: bool = False
    has_final_notes: bool = False
    final_title: str = ""

    @property
    def folder(self) -> Path:
        return Path(self.folder_path)

    @property
    def display_title(self) -> str:
        return self.final_title or self.title

    @property
    def is_active(self) -> bool:
        return self.status is SessionStatus.RECORDING

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["created_at"] = self.created_at.isoformat()
        payload["ended_at"] = self.ended_at.isoformat() if self.ended_at else None
        payload["status"] = str(self.status)
        return payload

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> SessionMeta:
        ended = data.get("ended_at")
        return cls(
            id=str(data["id"]),
            title=str(data.get("title", "Untitled session")),
            course=str(data.get("course", "")),
            created_at=datetime.fromisoformat(data["created_at"])
            if data.get("created_at")
            else utcnow(),
            ended_at=datetime.fromisoformat(ended) if ended else None,
            duration_seconds=float(data.get("duration_seconds", 0.0) or 0.0),
            word_count=int(data.get("word_count", 0) or 0),
            segment_count=int(data.get("segment_count", 0) or 0),
            whisper_model=str(data.get("whisper_model", "")),
            ollama_model=str(data.get("ollama_model", "")),
            audio_source=str(data.get("audio_source", "microphone")),
            status=SessionStatus(data.get("status", "complete")),
            folder_path=str(data.get("folder_path", "")),
            has_audio=bool(data.get("has_audio", False)),
            has_final_notes=bool(data.get("has_final_notes", False)),
            final_title=str(data.get("final_title", "")),
        )
