"""SQLite session index and full-text search.

The database holds *metadata and a search index only*. Transcripts and notes
live in their session folders as plain files, which keeps the source of truth
human-readable and means a corrupt database can always be rebuilt by rescanning
the sessions directory (``rebuild``).

Search uses FTS5 when the local SQLite build provides it, and degrades to
``LIKE`` matching otherwise — a slower search is much better than a crash on a
Python build without the extension.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path

from lectern.logging_setup import get_logger
from lectern.sessions.models import SessionMeta, SessionStatus
from lectern.utils import paths
from lectern.utils.timefmt import utcnow

log = get_logger("sessions.index")

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id     TEXT PRIMARY KEY,
    title          TEXT NOT NULL,
    course         TEXT NOT NULL DEFAULT '',
    created_at     TEXT NOT NULL,
    ended_at       TEXT,
    duration       REAL NOT NULL DEFAULT 0,
    word_count     INTEGER NOT NULL DEFAULT 0,
    whisper_model  TEXT NOT NULL DEFAULT '',
    ollama_model   TEXT NOT NULL DEFAULT '',
    audio_source   TEXT NOT NULL DEFAULT '',
    status         TEXT NOT NULL DEFAULT 'complete',
    folder_path    TEXT NOT NULL,
    has_audio      INTEGER NOT NULL DEFAULT 0,
    has_final      INTEGER NOT NULL DEFAULT 0,
    final_title    TEXT NOT NULL DEFAULT '',
    indexed_at     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS sessions_created_idx ON sessions(created_at DESC);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""

FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS session_fts USING fts5(
    session_id UNINDEXED,
    title,
    course,
    transcript,
    notes,
    tokenize = 'porter unicode61'
);
"""


@dataclass(slots=True)
class SearchHit:
    """One matching session with the snippets that matched."""

    session_id: str
    title: str
    course: str
    created_at: datetime
    snippets: list[str]
    field: str = "transcript"


@lru_cache(maxsize=1)
def fts5_available() -> bool:
    """Probe once whether this SQLite build has FTS5 compiled in."""
    try:
        with closing(sqlite3.connect(":memory:")) as connection:
            connection.execute("CREATE VIRTUAL TABLE probe USING fts5(x)")
        return True
    except sqlite3.Error:
        log.warning("SQLite FTS5 unavailable; search falls back to LIKE matching")
        return False


class SessionIndex:
    """Metadata index over the sessions directory."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or paths.index_db()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        # WAL keeps a reader (the UI) from blocking the writer (the recorder).
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=NORMAL")
        self._create_schema()

    def _create_schema(self) -> None:
        with self._connection:
            self._connection.executescript(SCHEMA)
            if fts5_available():
                self._connection.executescript(FTS_SCHEMA)
            self._connection.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )

    # -- writes ------------------------------------------------------------
    def upsert(self, meta: SessionMeta, *, transcript: str = "", notes: str = "") -> None:
        """Insert or update a session row and its search document."""
        with self._connection:
            self._connection.execute(
                """
                INSERT INTO sessions (
                    session_id, title, course, created_at, ended_at, duration, word_count,
                    whisper_model, ollama_model, audio_source, status, folder_path,
                    has_audio, has_final, final_title, indexed_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(session_id) DO UPDATE SET
                    title=excluded.title, course=excluded.course, ended_at=excluded.ended_at,
                    duration=excluded.duration, word_count=excluded.word_count,
                    whisper_model=excluded.whisper_model, ollama_model=excluded.ollama_model,
                    audio_source=excluded.audio_source, status=excluded.status,
                    folder_path=excluded.folder_path, has_audio=excluded.has_audio,
                    has_final=excluded.has_final, final_title=excluded.final_title,
                    indexed_at=excluded.indexed_at
                """,
                (
                    meta.id,
                    meta.title,
                    meta.course,
                    meta.created_at.isoformat(),
                    meta.ended_at.isoformat() if meta.ended_at else None,
                    meta.duration_seconds,
                    meta.word_count,
                    meta.whisper_model,
                    meta.ollama_model,
                    meta.audio_source,
                    str(meta.status),
                    str(meta.folder_path),
                    int(meta.has_audio),
                    int(meta.has_final_notes),
                    meta.final_title,
                    utcnow().isoformat(),
                ),
            )
            # Always refresh, even with empty content: the row also carries the
            # title and course, and a stale row would otherwise survive.
            if fts5_available():
                self._connection.execute(
                    "DELETE FROM session_fts WHERE session_id = ?", (meta.id,)
                )
                self._connection.execute(
                    "INSERT INTO session_fts (session_id, title, course, transcript, notes)"
                    " VALUES (?,?,?,?,?)",
                    (meta.id, meta.display_title, meta.course, transcript, notes),
                )

    def remove(self, session_id: str) -> None:
        with self._connection:
            self._connection.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            if fts5_available():
                self._connection.execute(
                    "DELETE FROM session_fts WHERE session_id = ?", (session_id,)
                )

    # -- reads -------------------------------------------------------------
    def list_sessions(self, *, limit: int | None = None, status: SessionStatus | None = None) -> list[SessionMeta]:
        query = "SELECT * FROM sessions"
        params: list[object] = []
        if status is not None:
            query += " WHERE status = ?"
            params.append(str(status))
        query += " ORDER BY created_at DESC"
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        rows = self._connection.execute(query, params).fetchall()
        return [_row_to_meta(row) for row in rows]

    def get(self, session_id: str) -> SessionMeta | None:
        row = self._connection.execute(
            "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        return _row_to_meta(row) if row else None

    def find(self, needle: str) -> SessionMeta | None:
        """Resolve a session by exact id, id prefix, or title substring."""
        exact = self.get(needle)
        if exact:
            return exact
        row = self._connection.execute(
            "SELECT * FROM sessions WHERE session_id LIKE ? OR title LIKE ? OR final_title LIKE ?"
            " ORDER BY created_at DESC LIMIT 1",
            (f"{needle}%", f"%{needle}%", f"%{needle}%"),
        ).fetchone()
        return _row_to_meta(row) if row else None

    def count(self) -> int:
        return int(self._connection.execute("SELECT COUNT(*) FROM sessions").fetchone()[0])

    def search(self, query: str, *, limit: int = 20) -> list[SearchHit]:
        """Full-text search across transcripts and notes."""
        query = query.strip()
        if not query:
            return []
        if fts5_available():
            try:
                return self._search_fts(query, limit)
            except sqlite3.OperationalError as exc:
                # A bare FTS syntax error (unbalanced quote, stray operator)
                # should degrade to a literal search, not surface as a crash.
                log.info("FTS query rejected (%s); falling back to LIKE", exc)
        return self._search_like(query, limit)

    def _search_fts(self, query: str, limit: int) -> list[SearchHit]:
        rows = self._connection.execute(
            """
            SELECT f.session_id AS session_id,
                   snippet(session_fts, 3, '[', ']', ' … ', 12) AS transcript_snippet,
                   snippet(session_fts, 4, '[', ']', ' … ', 12) AS notes_snippet,
                   s.title AS title, s.final_title AS final_title,
                   s.course AS course, s.created_at AS created_at
            FROM session_fts f
            JOIN sessions s ON s.session_id = f.session_id
            WHERE session_fts MATCH ?
            ORDER BY bm25(session_fts) LIMIT ?
            """,
            (query, limit),
        ).fetchall()

        hits: list[SearchHit] = []
        for row in rows:
            snippets = [
                text
                for text in (row["transcript_snippet"], row["notes_snippet"])
                if text and "[" in text
            ]
            hits.append(
                SearchHit(
                    session_id=row["session_id"],
                    title=row["final_title"] or row["title"],
                    course=row["course"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                    snippets=snippets or ["(matched session metadata)"],
                )
            )
        return hits

    def _search_like(self, query: str, limit: int) -> list[SearchHit]:
        """Fallback search that reads the session files directly."""
        hits: list[SearchHit] = []
        needle = query.lower()
        for meta in self.list_sessions():
            from lectern.sessions.storage import SessionStore

            store = SessionStore(meta.folder)
            haystacks = {
                "transcript": store.transcript_text(),
                "notes": store.load_final_notes() or store.load_live_notes_markdown(),
            }
            snippets: list[str] = []
            for text in haystacks.values():
                lowered = text.lower()
                position = lowered.find(needle)
                if position >= 0:
                    start = max(0, position - 60)
                    snippets.append("… " + text[start : position + len(needle) + 60].strip() + " …")
            if not snippets and needle in f"{meta.display_title} {meta.course}".lower():
                snippets = ["(matched session title)"]
            if snippets:
                hits.append(
                    SearchHit(
                        session_id=meta.id,
                        title=meta.display_title,
                        course=meta.course,
                        created_at=meta.created_at,
                        snippets=snippets[:2],
                    )
                )
            if len(hits) >= limit:
                break
        return hits

    def close(self) -> None:
        self._connection.close()


def _row_to_meta(row: sqlite3.Row) -> SessionMeta:
    return SessionMeta(
        id=row["session_id"],
        title=row["title"],
        course=row["course"],
        created_at=datetime.fromisoformat(row["created_at"]),
        ended_at=datetime.fromisoformat(row["ended_at"]) if row["ended_at"] else None,
        duration_seconds=float(row["duration"] or 0.0),
        word_count=int(row["word_count"] or 0),
        whisper_model=row["whisper_model"],
        ollama_model=row["ollama_model"],
        audio_source=row["audio_source"],
        status=SessionStatus(row["status"]),
        folder_path=row["folder_path"],
        has_audio=bool(row["has_audio"]),
        has_final_notes=bool(row["has_final"]),
        final_title=row["final_title"],
    )
