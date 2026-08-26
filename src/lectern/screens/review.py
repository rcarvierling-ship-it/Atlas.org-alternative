"""Session review.

Everything captured for one session, in tabs: the study notes, the full
transcript, a navigable timeline, extracted vocabulary, review questions and
the technical metadata. Also the place to export, and to retry a final
synthesis that failed or was never run.
"""

from __future__ import annotations

from rich.text import Text
from textual import work
from textual.app import ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.screen import Screen
from textual.widgets import Footer, Markdown, Static, TabbedContent, TabPane

from lectern.llm.base import LLMError
from lectern.logging_setup import get_logger
from lectern.notes.finalizer import FinalizationProgress, NoteFinalizer
from lectern.screens.modals import ConfirmModal, ExportModal, MessageModal
from lectern.sessions.manager import LoadedSession
from lectern.sessions.models import SessionStatus
from lectern.sessions.storage import mark_ended
from lectern.theme import ICONS
from lectern.utils.timefmt import format_clock, format_duration

log = get_logger("screens.review")


class ReviewScreen(Screen):
    """Read a finished session."""

    BINDINGS = [
        ("escape", "back", "Back"),
        ("e", "export", "Export"),
        ("r", "retry_final", "Retry final notes"),
        ("t", "show_transcript", "Transcript"),
        ("o", "show_notes", "Notes"),
        ("delete", "delete_session", "Delete"),
    ]

    def __init__(self, session: LoadedSession) -> None:
        super().__init__()
        self.session = session

    def compose(self) -> ComposeResult:
        meta = self.session.meta
        with Horizontal(id="review-header"):
            yield Static(meta.display_title, id="review-title")
            yield Static(self._meta_line(), id="review-meta")

        with TabbedContent(id="review-tabs"):
            with TabPane("Notes", id="tab-notes"):
                yield VerticalScroll(Markdown(self._notes_markdown(), id="notes-md"))
            with TabPane("Transcript", id="tab-transcript"):
                yield VerticalScroll(Static(self._transcript_text(), id="transcript-text"))
            with TabPane("Timeline", id="tab-timeline"):
                yield VerticalScroll(Static(self._timeline_text()))
            with TabPane("Key Terms", id="tab-terms"):
                yield VerticalScroll(Static(self._terms_text()))
            with TabPane("Questions", id="tab-questions"):
                yield VerticalScroll(Static(self._questions_text()))
            with TabPane("Metadata", id="tab-metadata"):
                yield VerticalScroll(Static(self._metadata_text()))
        yield Footer()

    def on_mount(self) -> None:
        if self.session.meta.status is SessionStatus.NEEDS_FINALIZATION:
            self.app.notify(
                "This session has no final study guide yet. Press r to generate it.",
                title="Live notes only",
                timeout=6,
            )

    # -- rendering ---------------------------------------------------------
    def _meta_line(self) -> str:
        meta = self.session.meta
        parts = [
            f"{meta.created_at.astimezone():%b %d, %Y %H:%M}",
            format_duration(meta.duration_seconds),
            f"{meta.word_count:,} words",
        ]
        if meta.course:
            parts.insert(0, meta.course)
        return f"   {ICONS.dot}   ".join(parts)

    def _notes_markdown(self) -> str:
        if self.session.final_notes.strip():
            return self.session.final_notes
        live = self.session.notes.to_markdown(title=self.session.meta.display_title)
        if live.strip():
            return (
                "> These are the live notes taken during the lecture. "
                "Press **r** to generate the final study guide.\n\n" + live
            )
        return "*No notes were generated for this session.*"

    def _transcript_text(self) -> Text:
        if not self.session.segments:
            return Text("No transcript was captured.", style="#5f6672")
        rendered = Text()
        for segment in self.session.segments:
            rendered.append(f"{format_clock(segment.start_time)}  ", style="#5f6672")
            rendered.append(f"{segment.text.strip()}\n\n", style="#c3c9d4")
        return rendered

    def _timeline_text(self) -> Text:
        entries = list(self.session.notes.timeline)
        for marker in self.session.markers:
            if not any(abs(entry.time - marker.time) < 0.01 for entry in entries):
                from lectern.notes.models import TimelineEntry

                entries.append(
                    TimelineEntry(time=marker.time, label=marker.label, kind=str(marker.kind))
                )
        entries.sort(key=lambda entry: entry.time)

        if not entries:
            return Text("Nothing was marked and no topic changes were detected.", style="#5f6672")

        rendered = Text()
        for entry in entries:
            rendered.append(f"{entry.clock}  ", style="#5f6672")
            if entry.kind == "topic":
                rendered.append(f"{ICONS.topic} ", style="#7c7cff")
                rendered.append(f"{entry.label}\n", style="bold #e6e8ec")
            elif entry.kind == "note":
                rendered.append(f"{ICONS.note} ", style="#56d4dd")
                rendered.append(f"{entry.label}\n", style="#c3c9d4")
            else:
                rendered.append(f"{ICONS.star} ", style="#fbbf24")
                rendered.append(f"{entry.label}\n", style="#c3c9d4")
        return rendered

    def _terms_text(self) -> Text:
        notes = self.session.notes
        entries = list(notes.definitions) + list(notes.key_terms)
        if not entries:
            return Text("No key terms were extracted.", style="#5f6672")
        rendered = Text()
        seen: set[str] = set()
        for entry in entries:
            if entry.key in seen:
                continue
            seen.add(entry.key)
            rendered.append(f"{entry.term}\n", style="bold #e6e8ec")
            if entry.definition:
                rendered.append(f"  {entry.definition}\n", style="#c3c9d4")
            rendered.append("\n")
        return rendered

    def _questions_text(self) -> Text:
        notes = self.session.notes
        rendered = Text()
        if notes.questions:
            rendered.append("QUESTIONS TO REVIEW\n", style="bold #8b919e")
            for item in notes.questions:
                rendered.append(f"  {ICONS.bullet} {item.text}\n", style="#c3c9d4")
            rendered.append("\n")
        if notes.unclear_points:
            rendered.append("NEEDS CLARIFICATION\n", style="bold #8b919e")
            for item in notes.unclear_points:
                rendered.append(f"  {ICONS.bullet} {item.text}\n", style="#fbbf24")
            rendered.append("\n")
        starred = [
            item
            for name in ("key_points", "important_details")
            for item in getattr(notes, name)
            if item.starred
        ]
        if starred:
            rendered.append("FLAGGED AS EXAM-WORTHY\n", style="bold #8b919e")
            for item in starred:
                rendered.append(f"  {ICONS.star} {item.text}\n", style="#e6e8ec")
        if not rendered.plain:
            return Text("No review questions were captured.", style="#5f6672")
        return rendered

    def _metadata_text(self) -> Text:
        meta = self.session.meta
        rows = [
            ("Session ID", meta.id),
            ("Title", meta.title),
            ("Final title", meta.final_title or "—"),
            ("Course", meta.course or "—"),
            ("Status", str(meta.status)),
            ("Created", f"{meta.created_at.astimezone():%Y-%m-%d %H:%M:%S}"),
            ("Ended", f"{meta.ended_at.astimezone():%Y-%m-%d %H:%M:%S}" if meta.ended_at else "—"),
            ("Duration", format_duration(meta.duration_seconds)),
            ("Words", f"{meta.word_count:,}"),
            ("Segments", str(meta.segment_count)),
            ("Markers", str(len(self.session.markers))),
            ("Whisper model", meta.whisper_model or "—"),
            ("Ollama model", meta.ollama_model or "—"),
            ("Audio source", meta.audio_source),
            ("Audio saved", "yes" if meta.has_audio else "no"),
            ("Folder", str(meta.folder)),
        ]
        rendered = Text()
        for label, value in rows:
            rendered.append(f"{label:<16}", style="#8b919e")
            rendered.append(f"{value}\n", style="#c3c9d4")
        return rendered

    # -- actions -----------------------------------------------------------
    def action_back(self) -> None:
        self.app.pop_screen()

    def action_show_notes(self) -> None:
        self.query_one("#review-tabs", TabbedContent).active = "tab-notes"

    def action_show_transcript(self) -> None:
        self.query_one("#review-tabs", TabbedContent).active = "tab-transcript"

    def action_export(self) -> None:
        self.app.push_screen(ExportModal(), callback=self._do_export)

    def _do_export(self, format_id: str | None) -> None:
        if not format_id:
            return
        from lectern.sessions.export import export_session

        try:
            path = export_session(self.session, format_id=format_id)
        except Exception as exc:  # noqa: BLE001
            log.exception("export failed")
            self.app.push_screen(
                MessageModal(f"Export failed: {exc}", title="Export", severity="error")
            )
            return
        self.app.notify(f"Exported to {path}", title="Export complete", timeout=8)

    def action_delete_session(self) -> None:
        self.app.push_screen(
            ConfirmModal(
                f"Permanently delete “{self.session.meta.display_title}” and everything in its "
                "folder — transcript, notes and audio?",
                title="Delete session?",
                confirm_label="Delete",
                danger=True,
            ),
            callback=self._confirm_delete,
        )

    def _confirm_delete(self, confirmed: bool) -> None:
        if not confirmed:
            return
        self.app.services.sessions.delete(self.session.meta.id)
        self.app.notify("Session deleted.", timeout=4)
        self.app.pop_screen()

    def action_retry_final(self) -> None:
        if self.session.meta.has_final_notes:
            self.app.push_screen(
                ConfirmModal(
                    "This session already has a final study guide. Generate it again?",
                    title="Regenerate final notes?",
                    confirm_label="Regenerate",
                ),
                callback=lambda confirmed: self.run_finalization() if confirmed else None,
            )
        else:
            self.run_finalization()

    @work(exclusive=True, group="review-finalize")
    async def run_finalization(self) -> None:
        """Run (or re-run) the final synthesis for this session."""
        transcript = self.session.transcript_text
        if not transcript:
            self.app.push_screen(
                MessageModal(
                    "This session has no transcript, so there is nothing to synthesise.",
                    title="Nothing to finalize",
                    severity="warning",
                )
            )
            return

        services = self.app.services
        config = services.config
        model = config.ollama.final_model or config.ollama.notes_model
        if not model:
            self.app.push_screen(
                MessageModal(
                    "No Ollama model is configured. Choose one in Settings first.",
                    title="No model selected",
                    severity="warning",
                )
            )
            return

        health = await services.refresh_llm_health()
        if not health.available:
            self.app.push_screen(
                MessageModal(
                    "Ollama is not responding. Start it with 'ollama serve' and try again — "
                    "your transcript and live notes are safe.",
                    title="Ollama unavailable",
                    severity="warning",
                )
            )
            return

        self.app.notify("Generating the final study guide…", title="Working", timeout=120)

        finalizer = NoteFinalizer(services.llm, model=model, num_ctx=config.ollama.num_ctx)
        markers = "\n".join(f"- {marker.clock} {marker.label}" for marker in self.session.markers)

        def on_progress(progress: FinalizationProgress) -> None:
            if progress.total:
                self.app.notify(progress.detail, title="Working", timeout=30)

        try:
            result = await finalizer.finalize(
                state=self.session.notes,
                transcript=transcript,
                session_title=self.session.meta.title,
                course=self.session.meta.course,
                duration=format_duration(self.session.meta.duration_seconds),
                markers=markers,
                on_progress=on_progress,
            )
        except LLMError as exc:
            log.warning("final synthesis failed: %s", exc)
            self.app.push_screen(
                MessageModal(
                    f"The final study guide could not be generated: {exc}\n\n"
                    "Your transcript and live notes are unchanged.",
                    title="Synthesis failed",
                    severity="error",
                )
            )
            return

        if not result.ok:
            self.app.push_screen(
                MessageModal(
                    f"The final study guide could not be generated: {result.error}\n\n"
                    "Your transcript and live notes are unchanged.",
                    title="Synthesis failed",
                    severity="error",
                )
            )
            return

        meta = self.session.meta
        self.session.store.save_final_notes(result.markdown)
        self.session.final_notes = result.markdown
        meta.final_title = result.title
        meta.has_final_notes = True
        mark_ended(meta, status=SessionStatus.COMPLETE)
        meta.status = SessionStatus.COMPLETE
        self.session.store.save_meta(meta)
        services.sessions.update_index_entry(meta, store=self.session.store)

        self.query_one("#notes-md", Markdown).update(result.markdown)
        self.query_one("#review-title", Static).update(meta.display_title)
        self.action_show_notes()
        self.app.notify("Final study guide ready.", title="Done", timeout=6)
