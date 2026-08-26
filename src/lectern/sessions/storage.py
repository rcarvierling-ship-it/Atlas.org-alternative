"""On-disk session folder.

Layout, all plain files so a session survives Lectern itself:

    2026-08-26-bio-113-cell-structure/
        session.json        metadata
        transcript.jsonl    one finalized segment per line, appended live
        transcript.md       readable transcript, rewritten on save
        notes-live.json     serialized NoteState (the recovery source of truth)
        notes-live.md       readable live notes
        notes-final.md      final study guide
        markers.json        markers and manual notes
        audio.wav           optional recording

**Durability is the point of this module.** The transcript is the one stream
that must never be lost, so every finalized segment is appended and flushed to
the OS immediately, with an ``fsync`` at intervals — a crash costs at most the
last few seconds, never the lecture. Everything else (notes, metadata) is
written atomically via a temp file and ``os.replace``, so a crash mid-write
leaves the *previous* good version in place rather than a truncated file.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import TextIO

from lectern.logging_setup import get_logger
from lectern.notes.models import NoteState
from lectern.sessions.models import Marker, SessionMeta, SessionStatus
from lectern.transcription.base import TranscriptSegment
from lectern.utils.audio_utils import TARGET_SAMPLE_RATE
from lectern.utils.text import word_count
from lectern.utils.timefmt import format_clock, utcnow

log = get_logger("sessions.storage")

SESSION_FILE = "session.json"
TRANSCRIPT_JSONL = "transcript.jsonl"
TRANSCRIPT_MD = "transcript.md"
NOTES_LIVE_JSON = "notes-live.json"
NOTES_LIVE_MD = "notes-live.md"
NOTES_FINAL_MD = "notes-final.md"
MARKERS_JSON = "markers.json"
AUDIO_WAV = "audio.wav"

#: Force data to the platter every N appended segments.
FSYNC_EVERY_N_SEGMENTS = 5


def write_atomic(path: Path, text: str) -> None:
    """Write a file such that readers only ever see a complete version."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


class SessionStore:
    """Reads and writes one session folder."""

    def __init__(self, folder: Path) -> None:
        self.folder = Path(folder)
        self._transcript_handle: TextIO | None = None
        self._appends_since_sync = 0

    # -- paths -------------------------------------------------------------
    @property
    def session_file(self) -> Path:
        return self.folder / SESSION_FILE

    @property
    def transcript_jsonl(self) -> Path:
        return self.folder / TRANSCRIPT_JSONL

    @property
    def transcript_md(self) -> Path:
        return self.folder / TRANSCRIPT_MD

    @property
    def notes_live_json(self) -> Path:
        return self.folder / NOTES_LIVE_JSON

    @property
    def notes_live_md(self) -> Path:
        return self.folder / NOTES_LIVE_MD

    @property
    def notes_final_md(self) -> Path:
        return self.folder / NOTES_FINAL_MD

    @property
    def markers_json(self) -> Path:
        return self.folder / MARKERS_JSON

    @property
    def audio_wav(self) -> Path:
        return self.folder / AUDIO_WAV

    def ensure(self) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)

    # -- metadata ----------------------------------------------------------
    def save_meta(self, meta: SessionMeta) -> None:
        self.ensure()
        meta.folder_path = str(self.folder)
        write_atomic(self.session_file, json.dumps(meta.to_json(), indent=2))

    def load_meta(self) -> SessionMeta:
        data = json.loads(self.session_file.read_text(encoding="utf-8"))
        meta = SessionMeta.from_json(data)
        # Trust the folder we actually found over a stale recorded path.
        meta.folder_path = str(self.folder)
        return meta

    def has_meta(self) -> bool:
        return self.session_file.exists()

    # -- transcript (append-only, durability critical) ---------------------
    def open_transcript(self) -> None:
        """Open the transcript for appending. Safe to call repeatedly."""
        if self._transcript_handle is not None:
            return
        self.ensure()
        self._transcript_handle = self.transcript_jsonl.open("a", encoding="utf-8")

    def append_segment(self, segment: TranscriptSegment) -> None:
        """Append one finalized segment and push it out of Python's buffers."""
        if self._transcript_handle is None:
            self.open_transcript()
        assert self._transcript_handle is not None
        self._transcript_handle.write(json.dumps(segment.to_json(), ensure_ascii=False) + "\n")
        self._transcript_handle.flush()
        self._appends_since_sync += 1
        if self._appends_since_sync >= FSYNC_EVERY_N_SEGMENTS:
            self.sync_transcript()

    def sync_transcript(self) -> None:
        """Force the OS to commit appended segments to disk."""
        if self._transcript_handle is None:
            return
        try:
            self._transcript_handle.flush()
            os.fsync(self._transcript_handle.fileno())
            self._appends_since_sync = 0
        except OSError as exc:  # pragma: no cover - filesystem dependent
            log.warning("fsync of transcript failed: %s", exc)

    def close_transcript(self) -> None:
        if self._transcript_handle is None:
            return
        self.sync_transcript()
        try:
            self._transcript_handle.close()
        finally:
            self._transcript_handle = None

    def load_segments(self) -> list[TranscriptSegment]:
        """Read every persisted segment, tolerating a torn final line."""
        if not self.transcript_jsonl.exists():
            return []
        segments: list[TranscriptSegment] = []
        with self.transcript_jsonl.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    segments.append(TranscriptSegment.from_json(json.loads(line)))
                except (json.JSONDecodeError, KeyError, ValueError) as exc:
                    # A crash during the final append can leave a half-written
                    # line. Drop it and keep every complete segment.
                    log.warning("skipping malformed transcript line %d: %s", line_number, exc)
        return segments

    def transcript_text(self) -> str:
        return " ".join(segment.text.strip() for segment in self.load_segments()).strip()

    def write_transcript_markdown(self, meta: SessionMeta, segments: list[TranscriptSegment]) -> None:
        """Render the human-readable transcript."""
        lines = [f"# {meta.display_title} — Transcript", ""]
        if meta.course:
            lines += [f"*{meta.course}*", ""]
        lines += [
            f"*Recorded {meta.created_at.astimezone():%Y-%m-%d %H:%M} · "
            f"{len(segments)} segments · {sum(word_count(s.text) for s in segments):,} words*",
            "",
        ]
        for segment in segments:
            lines.append(f"`{format_clock(segment.start_time)}`  {segment.text.strip()}")
            lines.append("")
        write_atomic(self.transcript_md, "\n".join(lines).rstrip() + "\n")

    # -- notes -------------------------------------------------------------
    def save_note_state(self, state: NoteState, *, title: str = "") -> None:
        """Persist the live note state in both machine and human form."""
        self.ensure()
        write_atomic(self.notes_live_json, json.dumps(state.to_json(), indent=2, ensure_ascii=False))
        markdown = state.to_markdown(title=title)
        if markdown.strip():
            write_atomic(self.notes_live_md, markdown)

    def load_note_state(self) -> NoteState:
        if not self.notes_live_json.exists():
            return NoteState()
        try:
            return NoteState.from_json(json.loads(self.notes_live_json.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError, KeyError, ValueError) as exc:
            log.error("could not read live notes (%s); starting from empty notes", exc)
            return NoteState()

    def save_final_notes(self, markdown: str) -> None:
        write_atomic(self.notes_final_md, markdown.rstrip() + "\n")

    def load_final_notes(self) -> str:
        if not self.notes_final_md.exists():
            return ""
        return self.notes_final_md.read_text(encoding="utf-8")

    def load_live_notes_markdown(self) -> str:
        if not self.notes_live_md.exists():
            return ""
        return self.notes_live_md.read_text(encoding="utf-8")

    # -- markers -----------------------------------------------------------
    def save_markers(self, markers: list[Marker]) -> None:
        self.ensure()
        write_atomic(
            self.markers_json,
            json.dumps([marker.to_json() for marker in markers], indent=2, ensure_ascii=False),
        )

    def load_markers(self) -> list[Marker]:
        if not self.markers_json.exists():
            return []
        try:
            raw = json.loads(self.markers_json.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            log.error("could not read markers: %s", exc)
            return []
        return [Marker.from_json(entry) for entry in raw]

    # -- audio -------------------------------------------------------------
    def open_audio_writer(self) -> AudioRecorder:
        return AudioRecorder(self.audio_wav)

    def audio_exists(self) -> bool:
        return self.audio_wav.exists() and self.audio_wav.stat().st_size > 44

    def close(self) -> None:
        self.close_transcript()


class AudioRecorder:
    """Streams captured audio to a WAV file while the session runs.

    Frames are written as they arrive rather than buffered in memory, so a
    three-hour lecture costs a constant handful of kilobytes of RAM and the
    recording survives a crash up to the last flush.
    """

    def __init__(self, path: Path, sample_rate: int = TARGET_SAMPLE_RATE) -> None:
        self.path = path
        self.sample_rate = sample_rate
        self._handle = None
        self._frames_written = 0

    def open(self) -> None:
        import wave

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = wave.open(str(self.path), "wb")
        self._handle.setnchannels(1)
        self._handle.setsampwidth(2)
        self._handle.setframerate(self.sample_rate)

    def write(self, block) -> None:  # noqa: ANN001 - numpy array
        from lectern.utils.audio_utils import float_to_pcm16

        if self._handle is None:
            return
        self._handle.writeframes(float_to_pcm16(block))
        self._frames_written += len(block)

    @property
    def seconds_written(self) -> float:
        return self._frames_written / self.sample_rate

    def close(self) -> None:
        handle, self._handle = self._handle, None
        if handle is not None:
            try:
                handle.close()
            except Exception as exc:  # noqa: BLE001 pragma: no cover
                log.warning("error closing audio recorder: %s", exc)


def session_folder_name(created_at: datetime, title: str) -> str:
    """``2026-08-26-bio-113-cell-structure`` — sortable and readable."""
    from lectern.utils.text import slugify

    return f"{created_at.astimezone():%Y-%m-%d}-{slugify(title)}"


def unique_folder(root: Path, name: str) -> Path:
    """Avoid colliding with an existing session recorded the same day."""
    candidate = root / name
    counter = 2
    while candidate.exists():
        candidate = root / f"{name}-{counter}"
        counter += 1
    return candidate


def mark_ended(meta: SessionMeta, *, status: SessionStatus) -> SessionMeta:
    """Stamp the end time and duration on a session that has stopped."""
    meta.ended_at = meta.ended_at or utcnow()
    if not meta.duration_seconds:
        meta.duration_seconds = max(0.0, (meta.ended_at - meta.created_at).total_seconds())
    meta.status = status
    return meta
