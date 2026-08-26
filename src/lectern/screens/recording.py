"""The recording screen.

Owns a ``RecordingPipeline`` for the lifetime of one session and renders its
events. The screen never does any heavy work itself: audio, transcription and
note generation all happen in pipeline tasks, and this class only reacts to
callbacks, which is what keeps keystrokes instant while whisper.cpp and Ollama
are busy.

Finishing is treated as a first-class flow rather than a teardown: the pipeline
stops, the modal shows each step completing, the final synthesis runs, and the
screen hands off to Session Review. If synthesis fails, the transcript and live
notes are still saved and the review screen offers a retry.
"""

from __future__ import annotations

import asyncio
import contextlib

from textual import work
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Footer, Label, Static

from lectern.audio import build_source
from lectern.audio.base import AudioError, PermissionDeniedError
from lectern.config.models import AudioSourceKind
from lectern.llm.base import LLMError
from lectern.logging_setup import get_logger
from lectern.notes.finalizer import FinalizationProgress, NoteFinalizer
from lectern.notes.models import NoteState
from lectern.pipeline import PipelineCallbacks, PipelineState, PipelineStatus, RecordingPipeline
from lectern.screens.modals import (
    ConfirmModal,
    FinalizingModal,
    MessageModal,
    PermissionModal,
    TextPromptModal,
)
from lectern.services import SessionRequest
from lectern.sessions.models import Marker, MarkerKind, SessionMeta, SessionStatus
from lectern.sessions.recovery import prepare_resume, resume_offsets
from lectern.sessions.storage import SessionStore, mark_ended
from lectern.theme import ICONS
from lectern.transcription import build_backend as build_transcriber
from lectern.transcription.base import TranscriptionError, TranscriptSegment
from lectern.utils.timefmt import format_clock, format_duration
from lectern.widgets.notes import NotesView
from lectern.widgets.recorder import RecorderHeader
from lectern.widgets.status import StatusBar
from lectern.widgets.topics import TopicsPanel
from lectern.widgets.transcript import TranscriptView

log = get_logger("screens.recording")

FINALIZE_STEPS = [
    "Transcript saved",
    "Recording saved",
    "Consolidating notes",
    "Creating final study guide",
]


class RecordingScreen(Screen):
    """Live transcript and live notes for one session."""

    BINDINGS = [
        ("space", "toggle_pause", "Pause"),
        ("m", "add_marker", "Marker"),
        ("n", "add_note", "Note"),
        ("t", "focus_transcript", "Transcript"),
        ("o", "focus_notes", "Notes"),
        ("q", "finish", "Finish"),
        ("escape", "finish", "Finish"),
        ("question_mark", "help", "Help"),
    ]

    def __init__(self, request: SessionRequest) -> None:
        super().__init__()
        self.request = request
        self.pipeline: RecordingPipeline | None = None
        self.meta: SessionMeta | None = None
        self.store: SessionStore | None = None
        self._timer = None
        self._finishing = False
        self._start_failed = False

    # -- layout ------------------------------------------------------------
    def compose(self) -> ComposeResult:
        yield RecorderHeader(
            title=self.request.title,
            course=self.request.course,
            whisper_model=self.request.whisper_model,
            ollama_model=self.request.notes_model,
            id="rec-header",
        )
        with Horizontal(id="rec-top"):
            with Vertical(id="topics-panel"):
                yield Label("TOPICS", classes="panel-title")
                yield TopicsPanel(id="topics")
            with Vertical(id="transcript-panel"):
                yield Label("LIVE TRANSCRIPT", classes="panel-title")
                yield TranscriptView(id="transcript")
                yield Static("", id="transcript-jump")
        with Vertical(id="notes-panel"):
            yield Label("LIVE NOTES", classes="panel-title")
            yield NotesView(id="notes")
        yield StatusBar(id="status-bar")
        yield Footer()

    def on_mount(self) -> None:
        label = " · ".join(
            part
            for part in (self.request.whisper_model, self.request.notes_model or "no notes model")
            if part
        )
        self.query_one("#status-bar", StatusBar).set_model_label(label)
        self.start_session()

    # -- session startup ---------------------------------------------------
    @work(exclusive=True, group="recording-start")
    async def start_session(self) -> None:
        """Create (or resume) the session and start the pipeline."""
        services = self.app.services
        config = services.config
        manager = services.sessions

        initial_notes = NoteState()
        initial_markers: list[Marker] = []
        initial_segments: list[TranscriptSegment] = []
        start_id, time_offset = 1, 0.0

        try:
            if self.request.resume_session_id:
                existing = manager.get(self.request.resume_session_id)
                if existing is None:
                    raise AudioError(f"session {self.request.resume_session_id} no longer exists")
                self.meta, self.store = prepare_resume(manager, existing)
                start_id, time_offset = resume_offsets(self.store)
                initial_notes = self.store.load_note_state()
                initial_markers = self.store.load_markers()
                initial_segments = self.store.load_segments()
                self.query_one("#transcript", TranscriptView).load_segments(initial_segments)
            else:
                self.meta, self.store = manager.create(
                    title=self.request.title,
                    course=self.request.course,
                    whisper_model=self.request.whisper_model,
                    ollama_model=self.request.notes_model,
                    audio_source="file" if self.request.is_file_mode else self.request.audio_source,
                    save_audio=self.request.save_audio,
                )
        except Exception as exc:  # noqa: BLE001
            log.exception("could not create session")
            await self._fail_start("Session", f"Could not create the session folder: {exc}")
            return

        audio_config = config.audio.model_copy()
        if self.request.input_device:
            audio_config.input_device = self.request.input_device

        try:
            source = build_source(
                audio_config,
                kind=AudioSourceKind(self.request.audio_source),
                file_path=self.request.file_path,
                speed=self.request.file_speed,
            )
        except (AudioError, ValueError) as exc:
            await self._fail_start("Audio", str(exc))
            return

        transcription_config = config.transcription.model_copy()
        if self.request.whisper_model:
            transcription_config.model = self.request.whisper_model
        transcriber = build_transcriber(transcription_config)

        llm = services.llm if self.request.notes_model else None

        self.pipeline = RecordingPipeline(
            config=config,
            source=source,
            transcriber=transcriber,
            llm=llm,
            store=self.store,
            meta=self.meta,
            callbacks=PipelineCallbacks(
                on_segment=self._on_segment,
                on_partial=self._on_partial,
                on_notes=self._on_notes,
                on_status=self._on_status,
                on_marker=self._on_marker,
                on_error=self._on_error,
            ),
            save_audio=self.request.save_audio,
            notes_model=self.request.notes_model,
            start_segment_id=start_id,
            time_offset=time_offset,
            initial_notes=initial_notes,
            initial_markers=initial_markers,
            initial_segments=initial_segments,
        )

        if not initial_notes.is_empty:
            self._on_notes(initial_notes)

        try:
            await self.pipeline.start()
        except PermissionDeniedError as exc:
            await self._fail_start_permission(exc)
            return
        except (AudioError, TranscriptionError) as exc:
            await self._fail_start(
                "Audio" if isinstance(exc, AudioError) else "Transcription", str(exc)
            )
            return
        except Exception as exc:  # noqa: BLE001
            log.exception("pipeline failed to start")
            await self._fail_start("Lectern", f"Could not start recording: {exc}")
            return

        self._timer = self.set_interval(1.0, self._tick)
        self.app.notify(
            f"Recording {self.meta.title}. Press q to finish.",
            title="Session started",
            timeout=4,
        )

    async def _fail_start(self, category: str, message: str) -> None:
        """Startup failed: clean up the empty session and return Home."""
        self._start_failed = True
        await self._cleanup_failed_session()
        self.app.push_screen(
            MessageModal(message, title=f"{category} problem", severity="error"),
            callback=lambda _: self._leave(),
        )

    async def _fail_start_permission(self, exc: PermissionDeniedError) -> None:
        self._start_failed = True
        await self._cleanup_failed_session()
        self.app.push_screen(
            PermissionModal(
                permission=exc.permission, message=str(exc), remediation=exc.remediation
            ),
            callback=lambda _: self._leave(),
        )

    async def _cleanup_failed_session(self) -> None:
        """Remove the folder for a session that never captured anything."""
        if self.meta is None or self.store is None:
            return
        with contextlib.suppress(Exception):
            self.store.close()
        if self.request.resume_session_id:
            return
        if not self.store.load_segments():
            with contextlib.suppress(Exception):
                self.app.services.sessions.delete(self.meta.id)
        self.meta = self.store = None

    def _leave(self) -> None:
        if self.is_running:
            self.app.pop_screen()

    # -- pipeline callbacks ------------------------------------------------
    def _on_segment(self, segment: TranscriptSegment) -> None:
        transcript = self.query_one("#transcript", TranscriptView)
        transcript.add_segment(segment)
        self._refresh_jump_pill()

    def _on_partial(self, text: str) -> None:
        self.query_one("#transcript", TranscriptView).set_partial(text)

    def _on_notes(self, state: NoteState) -> None:
        self.query_one("#notes", NotesView).update_notes(state)
        self.query_one("#topics", TopicsPanel).update_notes(state)

    def _on_status(self, status: PipelineStatus) -> None:
        self.query_one("#status-bar", StatusBar).refresh_status(status)
        self.query_one("#rec-header", RecorderHeader).refresh_status(status)

    def _on_marker(self, marker: Marker) -> None:
        self.query_one("#transcript", TranscriptView).add_marker(marker.time, marker.label)

    def _on_error(self, category: str, message: str) -> None:
        """Expected failures become notifications, never tracebacks."""
        severity = "information" if "resumed" in message else "warning"
        self.app.notify(message, title=category, severity=severity, timeout=8)

    def _tick(self) -> None:
        if self.pipeline is None:
            return
        self._on_status(self.pipeline.status())
        self._refresh_jump_pill()

    def _refresh_jump_pill(self) -> None:
        transcript = self.query_one("#transcript", TranscriptView)
        pill = self.query_one("#transcript-jump", Static)
        if not transcript.following and transcript.pending_count > 0:
            plural = "s" if transcript.pending_count != 1 else ""
            pill.update(f"{ICONS.down} {transcript.pending_count} newer segment{plural} — press f")
            pill.add_class("visible")
        else:
            pill.remove_class("visible")

    # -- actions -----------------------------------------------------------
    def action_toggle_pause(self) -> None:
        if self.pipeline is None:
            return
        if self.pipeline.is_paused:
            self.pipeline.resume()
            self.app.notify("Recording resumed.", timeout=2)
        else:
            self.pipeline.pause()
            self.app.notify("Recording paused. Press space to resume.", timeout=3)

    def action_add_marker(self) -> None:
        if self.pipeline is None:
            return
        marker = self.pipeline.add_marker(kind=MarkerKind.IMPORTANT)
        self.app.notify(f"{ICONS.star} Marked {marker.clock}", timeout=2)

    def action_add_note(self) -> None:
        if self.pipeline is None:
            return
        self.app.push_screen(
            TextPromptModal(
                title="Quick note",
                placeholder="Professor said this is on Exam 1",
                hint="Saved with the current timestamp and included in your final notes.",
            ),
            callback=self._save_note,
        )

    def _save_note(self, text: str | None) -> None:
        if not text or self.pipeline is None:
            return
        marker = self.pipeline.add_marker(text=text, kind=MarkerKind.NOTE)
        self.app.notify(f"Note saved at {marker.clock}", timeout=2)

    def action_focus_transcript(self) -> None:
        self.query_one("#transcript", TranscriptView).focus()

    def action_focus_notes(self) -> None:
        self.query_one("#notes", NotesView).focus()

    def action_help(self) -> None:
        from lectern.screens.modals import HelpModal

        self.app.push_screen(HelpModal())

    def action_finish(self) -> None:
        """Never end a live session on a single keypress."""
        if self._finishing or self._start_failed:
            return
        if self.pipeline is None or self.pipeline.state in (
            PipelineState.STOPPED,
            PipelineState.FAILED,
        ):
            self._leave()
            return

        elapsed = format_clock(self.pipeline.elapsed)
        words = self.pipeline.status().word_count
        self.app.push_screen(
            ConfirmModal(
                f"Stop recording after {elapsed} ({words:,} words) and generate the final notes?",
                title="Finish session?",
                confirm_label="Finish",
                cancel_label="Keep recording",
            ),
            callback=lambda confirmed: self.finish_session() if confirmed else None,
        )

    # -- finishing ---------------------------------------------------------
    @work(exclusive=True, group="recording-finish")
    async def finish_session(self) -> None:
        """Stop the pipeline, run the final synthesis, then open Session Review."""
        if self.pipeline is None or self.meta is None or self.store is None or self._finishing:
            return
        self._finishing = True
        if self._timer is not None:
            self._timer.stop()
            self._timer = None

        modal = FinalizingModal(FINALIZE_STEPS)
        self.app.push_screen(modal)
        modal.set_step(FINALIZE_STEPS[0], "active")

        pipeline = self.pipeline
        await pipeline.stop()

        segments = pipeline.segments
        meta = self.meta
        meta.word_count = pipeline.status().word_count
        meta.segment_count = len(segments)
        meta.duration_seconds = pipeline.elapsed
        self.store.write_transcript_markdown(meta, segments)
        modal.set_step(FINALIZE_STEPS[0], "done", f"{meta.word_count:,} words")

        modal.set_step(
            FINALIZE_STEPS[1],
            "done" if self.store.audio_exists() else "pending",
            "" if self.store.audio_exists() else "audio recording disabled",
        )

        notes = pipeline.notes
        final_markdown = ""
        error = ""

        # Every transcribed word is in `segments`, including anything the
        # scheduler had buffered for a note update that never ran, so this is
        # the whole lecture exactly once.
        transcript_text = " ".join(segment.text.strip() for segment in segments).strip()

        if not self.request.notes_model:
            modal.set_step(FINALIZE_STEPS[2], "failed", "no notes model selected")
            modal.set_step(FINALIZE_STEPS[3], "failed", "final notes need a local model")
            error = "no Ollama model was selected for this session"
        elif not transcript_text:
            modal.set_step(FINALIZE_STEPS[2], "failed", "nothing was transcribed")
            modal.set_step(FINALIZE_STEPS[3], "failed")
            error = "no speech was transcribed"
        else:
            modal.set_step(FINALIZE_STEPS[2], "done")
            modal.set_step(FINALIZE_STEPS[3], "active")
            markers = "\n".join(
                f"- {marker.clock} {marker.label}" for marker in pipeline.markers
            )
            config = self.app.services.config
            finalizer = NoteFinalizer(
                self.app.services.llm,
                model=config.ollama.final_model or self.request.notes_model,
                num_ctx=config.ollama.num_ctx,
            )

            def on_progress(progress: FinalizationProgress) -> None:
                # The finalizer runs on this event loop, so the modal can be
                # updated directly — no thread hand-off needed.
                modal.set_step(FINALIZE_STEPS[3], "active", progress.detail)

            try:
                result = await finalizer.finalize(
                    state=notes,
                    transcript=transcript_text,
                    session_title=meta.title,
                    course=meta.course,
                    duration=format_duration(meta.duration_seconds),
                    markers=markers,
                    on_progress=on_progress,
                )
            except LLMError as exc:
                result = None
                error = str(exc)

            if result is not None and result.ok:
                final_markdown = result.markdown
                meta.final_title = result.title
                meta.has_final_notes = True
                self.store.save_final_notes(final_markdown)
                modal.set_step(FINALIZE_STEPS[3], "done")
            else:
                error = error or (result.error if result else "final synthesis failed")
                modal.set_step(FINALIZE_STEPS[3], "failed", error[:60])

        mark_ended(
            meta,
            status=SessionStatus.COMPLETE if final_markdown else SessionStatus.NEEDS_FINALIZATION,
        )
        self.store.save_meta(meta)
        self.store.save_note_state(notes, title=meta.title)
        self.app.services.sessions.update_index_entry(meta, store=self.store)
        self.store.close()

        # Let the completed steps stay on screen briefly so finishing reads as
        # a sequence of results rather than a flash.
        await asyncio.sleep(0.6 if final_markdown else 1.6)
        with contextlib.suppress(Exception):
            modal.dismiss(None)

        if error:
            self.app.notify(
                f"Final notes were not generated ({error}). "
                "Your transcript and live notes are saved — you can retry from the review screen.",
                title="Finalization incomplete",
                severity="warning",
                timeout=10,
            )
        self.app.open_session(meta.id, replace_current=True)

    # -- teardown ----------------------------------------------------------
    async def shutdown(self) -> None:
        """Stop cleanly if the app exits while this session is recording."""
        if self._timer is not None:
            self._timer.stop()
            self._timer = None
        if self.pipeline is not None and self.pipeline.state not in (
            PipelineState.STOPPED,
            PipelineState.FAILED,
        ):
            await self.pipeline.stop()
            if self.meta is not None and self.store is not None:
                # Leave the session marked INCOMPLETE so the next launch offers
                # to recover it rather than silently losing the recording.
                self.meta.status = SessionStatus.INCOMPLETE
                self.meta.word_count = self.pipeline.status().word_count
                self.meta.segment_count = len(self.pipeline.segments)
                self.meta.duration_seconds = self.pipeline.elapsed
                self.store.save_meta(self.meta)
                self.store.write_transcript_markdown(self.meta, self.pipeline.segments)
                with contextlib.suppress(Exception):
                    self.app.services.sessions.update_index_entry(self.meta, store=self.store)
                self.store.close()
