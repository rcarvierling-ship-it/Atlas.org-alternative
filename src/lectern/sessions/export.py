"""Session exporters.

Markdown is the primary human-readable format; plain text and JSON exist for
piping into other tools. New formats (HTML, PDF) plug in by subclassing
``Exporter`` and registering it in ``EXPORTERS`` — nothing else needs to change.
"""

from __future__ import annotations

import abc
import json
from datetime import datetime
from pathlib import Path

from lectern.sessions.manager import LoadedSession
from lectern.utils.timefmt import format_clock, format_duration


class Exporter(abc.ABC):
    """Renders a loaded session into one file format."""

    format_id: str = ""
    extension: str = ""
    label: str = ""

    @abc.abstractmethod
    def render(self, session: LoadedSession) -> str:
        """Produce the file's contents."""

    def write(self, session: LoadedSession, path: Path) -> Path:
        from lectern.sessions.storage import write_atomic

        write_atomic(path, self.render(session))
        return path

    def default_filename(self, session: LoadedSession) -> str:
        return f"{session.meta.id}{self.extension}"


def _meta_lines(session: LoadedSession) -> list[str]:
    meta = session.meta
    created: datetime = meta.created_at.astimezone()
    lines = [
        f"- **Recorded:** {created:%A, %B %d, %Y at %H:%M}",
        f"- **Duration:** {format_duration(meta.duration_seconds)}",
        f"- **Words:** {meta.word_count:,}",
    ]
    if meta.course:
        lines.insert(0, f"- **Course:** {meta.course}")
    if meta.whisper_model:
        lines.append(f"- **Transcription:** whisper.cpp · {meta.whisper_model}")
    if meta.ollama_model:
        lines.append(f"- **Notes:** Ollama · {meta.ollama_model}")
    lines.append(f"- **Audio source:** {meta.audio_source}")
    return lines


class MarkdownExporter(Exporter):
    """The document a student actually reads: notes, markers, then transcript."""

    format_id = "markdown"
    extension = ".md"
    label = "Markdown"

    def render(self, session: LoadedSession) -> str:
        meta = session.meta
        parts: list[str] = [f"# {meta.display_title}", ""]
        parts += _meta_lines(session)
        parts.append("")

        notes = session.final_notes.strip()
        if notes:
            parts += ["---", "", "## Study Notes", "", _demote_headings(notes), ""]
        else:
            live = session.notes.to_markdown(title="")
            if live.strip():
                parts += [
                    "---",
                    "",
                    "## Live Notes",
                    "",
                    "*Final synthesis has not been run for this session.*",
                    "",
                    _demote_headings(live),
                    "",
                ]

        if session.markers:
            parts += ["---", "", "## Markers", ""]
            for marker in session.markers:
                icon = "★" if marker.kind == "important" else "✎"
                parts.append(f"- `{marker.clock}` {icon} {marker.label}")
            parts.append("")

        if session.segments:
            parts += ["---", "", "## Transcript", ""]
            for segment in session.segments:
                parts.append(f"`{format_clock(segment.start_time)}`  {segment.text.strip()}")
                parts.append("")

        return "\n".join(parts).rstrip() + "\n"


class TextExporter(Exporter):
    """Plain text for pasting anywhere."""

    format_id = "text"
    extension = ".txt"
    label = "Plain text"

    def render(self, session: LoadedSession) -> str:
        meta = session.meta
        lines = [
            meta.display_title.upper(),
            "=" * len(meta.display_title),
            "",
            f"Recorded: {meta.created_at.astimezone():%Y-%m-%d %H:%M}",
            f"Duration: {format_duration(meta.duration_seconds)}",
            f"Words:    {meta.word_count:,}",
            "",
        ]
        if meta.course:
            lines.insert(3, f"Course:   {meta.course}")

        notes = session.final_notes or session.notes.to_markdown(title="")
        if notes.strip():
            lines += ["NOTES", "-----", "", _strip_markdown(notes), ""]

        if session.markers:
            lines += ["MARKERS", "-------", ""]
            lines += [f"[{marker.clock}] {marker.label}" for marker in session.markers]
            lines.append("")

        if session.segments:
            lines += ["TRANSCRIPT", "----------", ""]
            lines += [
                f"[{format_clock(segment.start_time)}] {segment.text.strip()}"
                for segment in session.segments
            ]
        return "\n".join(lines).rstrip() + "\n"


class JSONExporter(Exporter):
    """Machine-readable export containing every field Lectern stores."""

    format_id = "json"
    extension = ".json"
    label = "JSON"

    def render(self, session: LoadedSession) -> str:
        payload = {
            "session": session.meta.to_json(),
            "notes": session.notes.to_json(),
            "final_notes_markdown": session.final_notes,
            "markers": [marker.to_json() for marker in session.markers],
            "transcript": [segment.to_json() for segment in session.segments],
        }
        return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"


EXPORTERS: dict[str, Exporter] = {
    exporter.format_id: exporter
    for exporter in (MarkdownExporter(), TextExporter(), JSONExporter())
}


def get_exporter(format_id: str) -> Exporter:
    try:
        return EXPORTERS[format_id.lower()]
    except KeyError:
        raise ValueError(
            f"unknown export format {format_id!r} (available: {', '.join(sorted(EXPORTERS))})"
        ) from None


def export_session(session: LoadedSession, *, format_id: str = "markdown", destination: Path | None = None) -> Path:
    """Export a session, defaulting to a file inside its own folder."""
    exporter = get_exporter(format_id)
    if destination is None:
        target = session.meta.folder / f"export{exporter.extension}"
    elif destination.is_dir():
        target = destination / exporter.default_filename(session)
    else:
        target = destination
    return exporter.write(session, target)


def _demote_headings(markdown: str) -> str:
    """Shift headings one level down so an embedded document nests correctly."""
    lines = []
    for line in markdown.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#") and len(stripped) - len(stripped.lstrip("#")) <= 5:
            lines.append("#" + line)
        else:
            lines.append(line)
    return "\n".join(lines)


def _strip_markdown(markdown: str) -> str:
    """Flatten Markdown for the plain-text export."""
    import re

    text = re.sub(r"^#{1,6}\s*", "", markdown, flags=re.MULTILINE)
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"\1", text)
    text = re.sub(r"`(.+?)`", r"\1", text)
    return text
