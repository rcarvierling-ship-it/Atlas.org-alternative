"""Session lifecycle: create, list, open, delete, reindex.

The sessions directory on disk is authoritative. The SQLite index is a cache
over it, so ``reindex`` can always rebuild the database by walking the folders —
which is what makes it safe to delete ``index.sqlite3`` if it ever misbehaves.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

from lectern.config.models import LecternConfig
from lectern.logging_setup import get_logger
from lectern.notes.models import NoteState
from lectern.sessions.index import SessionIndex
from lectern.sessions.models import Marker, SessionMeta, SessionStatus
from lectern.sessions.storage import SessionStore, session_folder_name, unique_folder
from lectern.transcription.base import TranscriptSegment
from lectern.utils import paths
from lectern.utils.text import word_count
from lectern.utils.timefmt import utcnow

log = get_logger("sessions.manager")


@dataclass
class LoadedSession:
    """A session read back from disk in full."""

    meta: SessionMeta
    store: SessionStore
    segments: list[TranscriptSegment] = field(default_factory=list)
    notes: NoteState = field(default_factory=NoteState)
    markers: list[Marker] = field(default_factory=list)
    final_notes: str = ""

    @property
    def transcript_text(self) -> str:
        return " ".join(segment.text.strip() for segment in self.segments).strip()

    @property
    def notes_markdown(self) -> str:
        """Final notes if they exist, otherwise the live notes."""
        return self.final_notes or self.notes.to_markdown(title=self.meta.display_title)


class SessionManager:
    """Owns the sessions directory and its index."""

    def __init__(
        self,
        config: LecternConfig | None = None,
        *,
        root: Path | None = None,
        index: SessionIndex | None = None,
    ) -> None:
        self.config = config or LecternConfig()
        configured = self.config.storage.resolved_output_dir()
        self.root = Path(root) if root else (configured or paths.sessions_dir())
        self.root.mkdir(parents=True, exist_ok=True)
        self._index = index or SessionIndex()

    @property
    def index(self) -> SessionIndex:
        return self._index

    # -- creation ----------------------------------------------------------
    def create(
        self,
        *,
        title: str,
        course: str = "",
        whisper_model: str = "",
        ollama_model: str = "",
        audio_source: str = "microphone",
        save_audio: bool = True,
    ) -> tuple[SessionMeta, SessionStore]:
        """Create a new session folder and register it as recording."""
        created = utcnow()
        folder = unique_folder(self.root, session_folder_name(created, title))
        meta = SessionMeta(
            id=folder.name,
            title=title.strip() or "Untitled session",
            course=course.strip(),
            created_at=created,
            whisper_model=whisper_model,
            ollama_model=ollama_model,
            audio_source=audio_source,
            status=SessionStatus.RECORDING,
            folder_path=str(folder),
            has_audio=False,
        )
        store = SessionStore(folder)
        store.ensure()
        store.save_meta(meta)
        store.save_markers([])
        store.open_transcript()
        self._index.upsert(meta)
        log.info("created session %s at %s (audio=%s)", meta.id, folder, save_audio)
        return meta, store

    # -- reading -----------------------------------------------------------
    def list_recent(self, limit: int = 10) -> list[SessionMeta]:
        return self._index.list_sessions(limit=limit)

    def all_sessions(self) -> list[SessionMeta]:
        return self._index.list_sessions()

    def get(self, session_id: str) -> SessionMeta | None:
        return self._index.get(session_id)

    def find(self, needle: str) -> SessionMeta | None:
        return self._index.find(needle)

    def open(self, session_id: str) -> LoadedSession | None:
        """Load a session's metadata, transcript, notes and markers."""
        meta = self.find(session_id)
        if meta is None:
            folder = self.root / session_id
            if not (folder / "session.json").exists():
                return None
            meta = SessionStore(folder).load_meta()
        store = SessionStore(meta.folder)
        if not store.has_meta():
            log.warning("session %s is indexed but its folder is gone", session_id)
            return None
        return LoadedSession(
            meta=store.load_meta(),
            store=store,
            segments=store.load_segments(),
            notes=store.load_note_state(),
            markers=store.load_markers(),
            final_notes=store.load_final_notes(),
        )

    def search(self, query: str, *, limit: int = 20):
        return self._index.search(query, limit=limit)

    # -- maintenance -------------------------------------------------------
    def reindex(self) -> int:
        """Rebuild the index from the session folders. Returns sessions found.

        The folders are authoritative, so this also drops rows for sessions
        that no longer exist on disk. Without that, a folder deleted outside
        ``delete()`` leaves a ghost row that lists but cannot be opened.
        """
        found = 0
        seen: set[str] = set()
        for folder in sorted(self.root.iterdir()) if self.root.exists() else []:
            if not folder.is_dir() or not (folder / "session.json").exists():
                continue
            store = SessionStore(folder)
            try:
                meta = store.load_meta()
            except Exception as exc:  # noqa: BLE001
                log.warning("skipping unreadable session at %s: %s", folder, exc)
                continue
            segments = store.load_segments()
            meta.segment_count = len(segments)
            meta.word_count = sum(word_count(segment.text) for segment in segments)
            meta.has_audio = store.audio_exists()
            final = store.load_final_notes()
            meta.has_final_notes = bool(final)
            store.save_meta(meta)
            self._index.upsert(
                meta,
                transcript=" ".join(segment.text for segment in segments),
                notes=final or store.load_live_notes_markdown(),
            )
            seen.add(meta.id)
            found += 1

        for stale in [meta.id for meta in self._index.list_sessions() if meta.id not in seen]:
            log.info("dropping index row for missing session %s", stale)
            self._index.remove(stale)

        log.info("reindexed %d sessions", found)
        return found

    def update_index_entry(self, meta: SessionMeta, *, store: SessionStore | None = None) -> None:
        """Refresh one session's index row and search document."""
        store = store or SessionStore(meta.folder)
        final = store.load_final_notes()
        self._index.upsert(
            meta,
            transcript=store.transcript_text(),
            notes=final or store.load_live_notes_markdown(),
        )

    def delete(self, session_id: str) -> bool:
        """Delete a session's folder and index row."""
        meta = self.find(session_id)
        if meta is None:
            return False
        folder = meta.folder
        # Refuse to remove anything that is not recognisably a session folder.
        if folder.exists() and (folder / "session.json").exists():
            shutil.rmtree(folder)
        self._index.remove(meta.id)
        log.info("deleted session %s", meta.id)
        return True

    def incomplete_sessions(self) -> list[SessionMeta]:
        """Sessions that were recording when Lectern last exited."""
        stale: list[SessionMeta] = []
        for meta in self._index.list_sessions():
            if meta.status in (SessionStatus.RECORDING, SessionStatus.INCOMPLETE):
                stale.append(meta)
        return stale

    def prune_recordings(self, *, days: int | None = None) -> int:
        """Delete audio files older than the retention window. Returns count."""
        days = self.config.storage.recording_retention_days if days is None else days
        if days <= 0:
            return 0
        cutoff = utcnow() - timedelta(days=days)
        removed = 0
        for meta in self._index.list_sessions():
            if not meta.has_audio or meta.created_at >= cutoff:
                continue
            store = SessionStore(meta.folder)
            if store.audio_wav.exists():
                store.audio_wav.unlink()
                meta.has_audio = False
                store.save_meta(meta)
                self._index.upsert(meta)
                removed += 1
        if removed:
            log.info("pruned %d recording(s) older than %d days", removed, days)
        return removed

    def close(self) -> None:
        self._index.close()
