"""Crash recovery.

A session folder whose ``session.json`` still says ``recording`` was interrupted:
the terminal was closed, the Mac slept, the app crashed, Ctrl+C was pressed.
Because every finalized segment was appended and flushed as it happened, the
transcript on disk is intact up to the last few seconds regardless of how the
process died.

On startup Lectern finds those sessions and offers four choices:

* **Resume** — keep recording into the same session (append to the transcript).
* **Recover** — close it out and keep transcript + live notes as they are.
* **Finalize** — close it out and run the final synthesis over what was captured.
* **Discard** — delete the folder.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from lectern.logging_setup import get_logger
from lectern.sessions.manager import SessionManager
from lectern.sessions.models import SessionMeta, SessionStatus
from lectern.sessions.storage import SessionStore, mark_ended
from lectern.utils.text import word_count

log = get_logger("sessions.recovery")


class RecoveryAction(StrEnum):
    RESUME = "resume"
    RECOVER = "recover"
    FINALIZE = "finalize"
    DISCARD = "discard"


@dataclass(slots=True)
class RecoverableSession:
    """An interrupted session, described well enough to choose an action."""

    meta: SessionMeta
    segment_count: int
    word_count: int
    has_notes: bool
    has_audio: bool

    @property
    def is_empty(self) -> bool:
        """Nothing was captured, so there is nothing worth recovering."""
        return self.segment_count == 0 and not self.has_notes

    @property
    def summary(self) -> str:
        if self.is_empty:
            return "no transcript was captured"
        parts = [f"{self.word_count:,} words", f"{self.segment_count} segments"]
        if self.has_notes:
            parts.append("live notes")
        if self.has_audio:
            parts.append("audio")
        return " · ".join(parts)


def find_recoverable(manager: SessionManager) -> list[RecoverableSession]:
    """Inspect interrupted sessions on disk.

    Reads the folders rather than trusting the index, so a session that was
    written while the index was unavailable is still recoverable.
    """
    recoverable: list[RecoverableSession] = []
    for meta in manager.incomplete_sessions():
        store = SessionStore(meta.folder)
        if not store.has_meta():
            log.warning("dropping index entry for missing session %s", meta.id)
            manager.index.remove(meta.id)
            continue
        segments = store.load_segments()
        state = store.load_note_state()
        recoverable.append(
            RecoverableSession(
                meta=store.load_meta(),
                segment_count=len(segments),
                word_count=sum(word_count(segment.text) for segment in segments),
                has_notes=not state.is_empty,
                has_audio=store.audio_exists(),
            )
        )
    return recoverable


def recover(manager: SessionManager, meta: SessionMeta) -> SessionMeta:
    """Close an interrupted session, keeping everything captured so far.

    Metadata that only exists in memory during recording (duration, word count)
    is recomputed from the files on disk.
    """
    store = SessionStore(meta.folder)
    segments = store.load_segments()
    meta = store.load_meta()
    meta.segment_count = len(segments)
    meta.word_count = sum(word_count(segment.text) for segment in segments)
    meta.has_audio = store.audio_exists()
    meta.has_final_notes = bool(store.load_final_notes())

    # The last segment's end time is a better duration estimate than wall clock,
    # which would include however long the machine sat asleep.
    if segments:
        meta.duration_seconds = max(meta.duration_seconds, segments[-1].end_time)

    status = SessionStatus.COMPLETE if meta.has_final_notes else SessionStatus.NEEDS_FINALIZATION
    mark_ended(meta, status=status)

    store.save_meta(meta)
    store.write_transcript_markdown(meta, segments)
    manager.update_index_entry(meta, store=store)
    log.info("recovered session %s (%d segments)", meta.id, len(segments))
    return meta


def discard(manager: SessionManager, meta: SessionMeta) -> bool:
    log.info("discarding interrupted session %s", meta.id)
    return manager.delete(meta.id)


def prepare_resume(manager: SessionManager, meta: SessionMeta) -> tuple[SessionMeta, SessionStore]:
    """Reopen an interrupted session so recording can continue into it.

    New segments continue the existing id sequence and are timestamped after
    the last one, so the resumed portion lands in the right place in the
    transcript and the timeline.
    """
    store = SessionStore(meta.folder)
    meta = store.load_meta()
    meta.status = SessionStatus.RECORDING
    meta.ended_at = None
    store.save_meta(meta)
    store.open_transcript()
    manager.index.upsert(meta)
    log.info("resuming session %s", meta.id)
    return meta, store


def resume_offsets(store: SessionStore) -> tuple[int, float]:
    """Return the next segment id and the time offset to continue from."""
    segments = store.load_segments()
    if not segments:
        return 1, 0.0
    return segments[-1].id + 1, segments[-1].end_time
