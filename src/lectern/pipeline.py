"""The recording pipeline.

Everything that happens during a live session is orchestrated here, as a set of
independent asyncio tasks joined by bounded queues:

    AudioSource.frames()
        └─ audio task ── WAV recorder
                      └─ VAD segmenter ── utterance queue
                                              └─ transcription task ── whisper.cpp
                                                       ├─ persistence (append + flush)
                                                       ├─ UI callback
                                                       └─ NoteScheduler
                                                                └─ notes task ── Ollama
                                                                         ├─ persistence
                                                                         └─ UI callback

Design rules that the implementation enforces:

* **The transcript is the priority stream.** If Ollama dies, notes pause and
  everything else keeps running. If the note worker falls behind, transcription
  is unaffected — they share nothing but a queue.
* **No blocking work on the event loop.** whisper.cpp and Ollama are reached
  over HTTP with async clients; audio arrives via a callback that only enqueues.
* **The session clock is the audio clock.** Elapsed time is derived from
  captured samples, not wall time, so pausing stops the clock and a file-fed
  demo session produces exactly the timestamps the recording implies.
* **Bounded memory.** Nothing accumulates without a ceiling: audio goes to disk
  as it arrives, transcript segments are appended and the in-memory list is the
  only copy the UI needs, and prompts are capped by the scheduler.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np

from lectern.audio.base import AudioError, AudioSource
from lectern.audio.vad import VADConfig, VoiceSegmenter
from lectern.config.models import LecternConfig
from lectern.llm.base import LLMBackend, LLMError, LLMUnavailableError
from lectern.logging_setup import get_logger
from lectern.notes.consolidator import NoteConsolidator
from lectern.notes.models import NoteItem, NoteState, TimelineEntry
from lectern.notes.scheduler import NoteScheduler
from lectern.notes.updater import NoteUpdater
from lectern.sessions.models import Marker, MarkerKind, SessionMeta
from lectern.sessions.storage import AudioRecorder, SessionStore
from lectern.transcription.base import (
    TranscriptionBackend,
    TranscriptionError,
    TranscriptSegment,
)
from lectern.utils.audio_utils import TARGET_SAMPLE_RATE
from lectern.utils.text import word_count
from lectern.utils.timefmt import format_clock, utcnow

log = get_logger("pipeline")

#: Utterances waiting for whisper.cpp. Large enough to absorb a slow model for
#: minutes; if it ever fills, speech is being produced faster than the machine
#: can transcribe and the user needs to know.
UTTERANCE_QUEUE_SIZE = 400

#: How often the notes task wakes to check the scheduler.
NOTE_TICK_SECONDS = 1.0

#: How often session metadata and note state are flushed while recording.
PERSIST_INTERVAL_SECONDS = 20.0

#: Minimum speech before a partial hypothesis is worth decoding.
MIN_PARTIAL_SECONDS = 1.0


class PipelineState(StrEnum):
    IDLE = "idle"
    STARTING = "starting"
    RECORDING = "recording"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass(slots=True)
class PipelineStatus:
    """Snapshot rendered by the status bar. Cheap to build, no polling."""

    state: PipelineState = PipelineState.IDLE
    elapsed: float = 0.0
    word_count: int = 0
    segment_count: int = 0
    audio_level: float = 0.0
    stt_latency_ms: float | None = None
    stt_ready: bool = False
    stt_backlog: int = 0
    notes_updating: bool = False
    notes_last_update: float | None = None
    notes_available: bool = True
    notes_detail: str = ""
    dropped_utterances: int = 0

    @property
    def clock(self) -> str:
        return format_clock(self.elapsed)


@dataclass
class PipelineCallbacks:
    """UI hooks. Every callback is invoked on the event loop thread.

    Callbacks are wrapped so that an exception raised while rendering can never
    take down a recording that is in progress.
    """

    on_segment: Callable[[TranscriptSegment], None] | None = None
    on_partial: Callable[[str], None] | None = None
    on_notes: Callable[[NoteState], None] | None = None
    on_status: Callable[[PipelineStatus], None] | None = None
    on_marker: Callable[[Marker], None] | None = None
    on_error: Callable[[str, str], None] | None = None
    on_notes_activity: Callable[[bool], None] | None = None


class RecordingPipeline:
    """Owns a live recording session end to end."""

    def __init__(
        self,
        *,
        config: LecternConfig,
        source: AudioSource,
        transcriber: TranscriptionBackend,
        llm: LLMBackend | None,
        store: SessionStore,
        meta: SessionMeta,
        callbacks: PipelineCallbacks | None = None,
        save_audio: bool = True,
        notes_model: str = "",
        start_segment_id: int = 1,
        time_offset: float = 0.0,
        initial_notes: NoteState | None = None,
        initial_markers: list[Marker] | None = None,
        initial_segments: list[TranscriptSegment] | None = None,
    ) -> None:
        self.config = config
        self.source = source
        self.transcriber = transcriber
        self.llm = llm
        self.store = store
        self.meta = meta
        self.callbacks = callbacks or PipelineCallbacks()
        self.save_audio = save_audio

        self.state = PipelineState.IDLE
        self.notes = initial_notes.copy() if initial_notes else NoteState()
        self.markers: list[Marker] = list(initial_markers or [])
        # A resumed session starts holding everything captured before the
        # interruption. Finalization rewrites transcript.md and recomputes the
        # word and segment counts from this list, so starting it empty would
        # quietly reduce a recovered lecture to only its resumed half. These
        # segments are deliberately *not* given to the scheduler: they were
        # already turned into notes before the interruption.
        self.segments: list[TranscriptSegment] = list(initial_segments or [])

        self._segmenter = VoiceSegmenter(
            config=VADConfig(sample_rate=TARGET_SAMPLE_RATE),
            enabled=config.transcription.vad,
        )
        self._scheduler = NoteScheduler(config.notes)
        self._updater: NoteUpdater | None = None
        self._consolidator: NoteConsolidator | None = None
        if llm is not None:
            model = notes_model or config.ollama.notes_model
            self._updater = NoteUpdater(llm, model=model, num_ctx=config.ollama.num_ctx)
            self._consolidator = NoteConsolidator(llm, model=model, num_ctx=config.ollama.num_ctx)

        self._utterances: asyncio.Queue[object] = asyncio.Queue(maxsize=UTTERANCE_QUEUE_SIZE)
        self._tasks: list[asyncio.Task] = []
        self._recorder: AudioRecorder | None = None

        self._time_offset = time_offset
        self._audio_seconds = 0.0
        self._next_segment_id = start_segment_id
        self._paused = False
        self._stopping = False
        self._last_partial_at = 0.0
        self._partial_task: asyncio.Task[None] | None = None
        self._notes_updating = False
        self._notes_last_update: float | None = None
        self._notes_available = llm is not None
        self._notes_detail = "" if llm is not None else "no local model configured"
        self._dropped_utterances = 0
        self._transcript_words = sum(word_count(segment.text) for segment in self.segments)

    # -- properties --------------------------------------------------------
    @property
    def elapsed(self) -> float:
        """Session time derived from captured audio, not wall clock."""
        return self._time_offset + self._audio_seconds

    @property
    def is_recording(self) -> bool:
        return self.state is PipelineState.RECORDING

    @property
    def is_paused(self) -> bool:
        return self._paused

    def status(self) -> PipelineStatus:
        health = self.transcriber.health()
        return PipelineStatus(
            state=self.state,
            elapsed=self.elapsed,
            word_count=self._transcript_words,
            segment_count=len(self.segments),
            audio_level=self.source.level if not self._paused else 0.0,
            stt_latency_ms=health.last_latency_ms,
            stt_ready=health.ready,
            stt_backlog=self._utterances.qsize(),
            notes_updating=self._notes_updating,
            notes_last_update=self._notes_last_update,
            notes_available=self._notes_available,
            notes_detail=self._notes_detail,
            dropped_utterances=self._dropped_utterances,
        )

    # -- lifecycle ---------------------------------------------------------
    async def start(self) -> None:
        """Start capture, transcription and note generation.

        Raises ``AudioError`` / ``TranscriptionError`` if the session cannot
        begin; in that case nothing is left running.
        """
        self.state = PipelineState.STARTING
        started_transcriber = False
        try:
            await self.transcriber.start()
            started_transcriber = True
            await self.source.start()
        except (AudioError, TranscriptionError):
            if started_transcriber:
                with contextlib.suppress(Exception):
                    await self.transcriber.stop()
            self.state = PipelineState.FAILED
            raise

        if self.save_audio:
            self._recorder = self.store.open_audio_writer()
            try:
                self._recorder.open()
            except OSError as exc:
                log.warning("could not open audio recorder: %s", exc)
                self._recorder = None
            else:
                self.meta.has_audio = True

        self.store.open_transcript()
        self.state = PipelineState.RECORDING

        self._tasks = [
            asyncio.create_task(self._audio_task(), name="audio"),
            asyncio.create_task(self._transcribe_task(), name="transcribe"),
            asyncio.create_task(self._notes_task(), name="notes"),
            asyncio.create_task(self._persist_task(), name="persist"),
        ]
        log.info("pipeline started for session %s", self.meta.id)
        self._emit_status()

    def pause(self) -> None:
        """Stop consuming audio without tearing anything down."""
        if self.state is not PipelineState.RECORDING:
            return
        self._paused = True
        self.state = PipelineState.PAUSED
        log.info("session %s paused at %s", self.meta.id, format_clock(self.elapsed))
        self._emit_status()

    def resume(self) -> None:
        if self.state is not PipelineState.PAUSED:
            return
        self._paused = False
        self.state = PipelineState.RECORDING
        log.info("session %s resumed", self.meta.id)
        self._emit_status()

    async def stop(self) -> None:
        """Stop capture and drain in-flight work. Safe to call twice."""
        if self.state in (PipelineState.STOPPED, PipelineState.STOPPING):
            return
        self._stopping = True
        self.state = PipelineState.STOPPING
        self._emit_status()
        log.info("stopping pipeline for session %s", self.meta.id)

        with contextlib.suppress(Exception):
            await self.source.stop()

        # Let the trailing utterance through before shutting transcription down.
        trailing = self._segmenter.flush()
        if trailing is not None and trailing.audio.size:
            await self._enqueue_blocking(trailing, timeout=10.0)

        audio_task = self._task_named("audio")
        if audio_task is not None:
            audio_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await audio_task

        if self._partial_task is not None:
            self._partial_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._partial_task
            self._partial_task = None

        # Sentinel tells the transcription task to finish the queue and exit.
        # It must actually land: dropping it on a full queue leaves the worker
        # blocked on an empty queue once it has drained, so finishing appears
        # to hang until the timeout below cancels it.
        transcribe_task = self._task_named("transcribe")
        delivered = await self._enqueue_blocking(None, timeout=120.0)
        if transcribe_task is not None:
            if not delivered:
                log.error("could not deliver the transcription sentinel; cancelling the worker")
                transcribe_task.cancel()
            with contextlib.suppress(Exception):
                await asyncio.wait_for(transcribe_task, timeout=60.0)

        for name in ("notes", "persist"):
            task = self._task_named(name)
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
        self._tasks = []

        if self._recorder is not None:
            self._recorder.close()
            self.meta.has_audio = self.store.audio_exists()
            self._recorder = None

        with contextlib.suppress(Exception):
            await self.transcriber.stop()

        self._flush_state()
        self.store.close_transcript()
        self.state = PipelineState.STOPPED
        self._emit_status()
        log.info(
            "pipeline stopped: %d segments, %d words, %s",
            len(self.segments),
            self._transcript_words,
            format_clock(self.elapsed),
        )

    def _task_named(self, name: str) -> asyncio.Task | None:
        for task in self._tasks:
            if task.get_name() == name:
                return task
        return None

    # -- audio -------------------------------------------------------------
    async def _audio_task(self) -> None:
        """Consume captured blocks: record them, segment them, queue utterances."""
        try:
            async for block in self.source.frames():
                if self._paused:
                    continue
                self._audio_seconds += block.size / TARGET_SAMPLE_RATE

                if self._recorder is not None:
                    try:
                        self._recorder.write(block)
                    except Exception as exc:  # noqa: BLE001
                        log.error("audio recording failed, continuing without it: %s", exc)
                        self._recorder = None
                        self._report_error(
                            "Recording",
                            "Saving the audio file failed. Transcription continues normally.",
                        )

                for utterance in self._segmenter.feed(block):
                    self._enqueue_utterance(utterance)

                self._maybe_start_partial()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            log.exception("audio task failed")
            self._report_error("Audio", f"Audio capture stopped unexpectedly: {exc}")

    def _enqueue_utterance(self, utterance) -> None:  # noqa: ANN001
        try:
            self._utterances.put_nowait(utterance)
        except asyncio.QueueFull:
            # Transcription is minutes behind. Drop the oldest so the newest
            # speech still gets transcribed, and tell the user once.
            with contextlib.suppress(asyncio.QueueEmpty):
                self._utterances.get_nowait()
            self._dropped_utterances += 1
            if self._dropped_utterances == 1:
                self._report_error(
                    "Transcription",
                    "Speech is arriving faster than it can be transcribed. "
                    "Consider a smaller Whisper model (Settings → Whisper model).",
                )
            with contextlib.suppress(asyncio.QueueFull):
                self._utterances.put_nowait(utterance)

    async def _enqueue_blocking(self, item: object, *, timeout: float) -> bool:
        """Put an item on the utterance queue, waiting for space if needed.

        Used only on the shutdown path, where the consumer is still draining
        and dropping the item would strand the worker.
        """
        try:
            await asyncio.wait_for(self._utterances.put(item), timeout=timeout)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            return False
        return True

    def _maybe_start_partial(self) -> None:
        """Kick off a preview decode for the utterance being spoken.

        Partials are display-only: they are never persisted and never sent to
        the note model, because unstable text would poison the notes. They are
        skipped whenever whisper.cpp already has finalized work queued —
        finished speech always outranks a preview.

        The decode runs in its own task rather than inline. Awaiting it here
        would stop this coroutine consuming ``source.frames()`` for the length
        of a decode, and the source drops its oldest blocks when its queue
        fills — losing captured audio for the sake of a preview inverts the
        transcript-priority rule.
        """
        if not self.config.transcription.partials or not self.config.ui.show_partials:
            return
        if self.callbacks.on_partial is None or not self._segmenter.in_speech:
            return
        if self._utterances.qsize() > 0:
            return
        if self._partial_task is not None and not self._partial_task.done():
            return

        now = time.monotonic()
        if now - self._last_partial_at < self.config.transcription.partial_interval_seconds:
            return
        pending = self._segmenter.pending_audio
        if pending.size < TARGET_SAMPLE_RATE * MIN_PARTIAL_SECONDS:
            return

        self._last_partial_at = now
        self._partial_task = asyncio.create_task(self._decode_partial(pending), name="partial")

    async def _decode_partial(self, pending: np.ndarray) -> None:
        try:
            text = await self.transcriber.transcribe(pending)
        except (TranscriptionError, asyncio.CancelledError) as exc:
            log.debug("partial decode failed (ignored): %s", exc)
            return
        except Exception as exc:  # noqa: BLE001 - a preview must never break a session
            log.debug("partial decode error (ignored): %s", exc)
            return
        if text and self._segmenter.in_speech:
            self._safe(self.callbacks.on_partial, text)

    # -- transcription -----------------------------------------------------
    async def _transcribe_task(self) -> None:
        """Turn queued utterances into persisted transcript segments."""
        consecutive_failures = 0
        while True:
            item = await self._utterances.get()
            if item is None:
                return
            try:
                text = await self.transcriber.transcribe(item.audio)
                consecutive_failures = 0
            except TranscriptionError as exc:
                consecutive_failures += 1
                log.error("transcription failed: %s", exc)
                if consecutive_failures == 3:
                    self._report_error(
                        "Transcription",
                        "Whisper stopped responding. Your transcript so far is saved. "
                        "Restart transcription from the recording screen.",
                    )
                continue
            except Exception as exc:  # noqa: BLE001
                log.exception("unexpected transcription error")
                self._report_error("Transcription", f"Transcription error: {exc}")
                continue

            if not text.strip():
                continue

            segment = TranscriptSegment(
                id=self._next_segment_id,
                start_time=self._time_offset + item.start_time,
                end_time=self._time_offset + item.end_time,
                text=text.strip(),
                is_final=True,
                created_at=utcnow(),
            )
            self._next_segment_id += 1
            self.segments.append(segment)
            self._transcript_words += word_count(segment.text)

            try:
                self.store.append_segment(segment)
            except OSError as exc:
                log.exception("could not append transcript segment")
                self._report_error(
                    "Storage",
                    f"Writing the transcript failed: {exc}. Free some disk space — "
                    "the session is still running.",
                )

            self._scheduler.add_segment(segment)
            self._safe(self.callbacks.on_segment, segment)
            self._emit_status()

    # -- notes -------------------------------------------------------------
    async def _notes_task(self) -> None:
        """Run note updates and consolidation when the scheduler says so."""
        if self._updater is None:
            return
        try:
            while True:
                await asyncio.sleep(NOTE_TICK_SECONDS)
                if self._paused:
                    continue
                if self._scheduler.should_update():
                    await self._run_update()
                elif self._consolidator is not None and self._scheduler.should_consolidate():
                    await self._run_consolidation()
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            log.exception("notes task crashed; transcript continues")
            self._notes_available = False
            self._notes_detail = "note generation stopped"
            self._report_error(
                "Notes",
                "Live note generation stopped unexpectedly. Your transcript is still being saved.",
            )

    async def _run_update(self) -> None:
        assert self._updater is not None
        batch = self._scheduler.take_batch()
        if batch is None:
            return

        self._notes_updating = True
        self._safe(self.callbacks.on_notes_activity, True)
        self._emit_status()
        try:
            result = await self._updater.update(
                self.notes,
                batch.text,
                session_title=self.meta.title,
                course=self.meta.course,
                markers=batch.markers,
                elapsed=format_clock(self.elapsed),
                timestamp=batch.start_time,
                max_context_words=self.config.notes.max_context_words,
            )
        except LLMUnavailableError as exc:
            self._on_llm_unavailable(str(exc))
            self._scheduler.finish_update(success=False)
            return
        except LLMError as exc:
            log.warning("note update error: %s", exc)
            self._scheduler.finish_update(success=False)
            return
        finally:
            self._notes_updating = False
            self._safe(self.callbacks.on_notes_activity, False)

        if result.error:
            if not self._notes_available:
                self._notes_available = True
            self._scheduler.finish_update(success=False)
            self._emit_status()
            return

        if not self._notes_available:
            self._notes_available = True
            self._notes_detail = ""
            self._report_error("Notes", "Ollama is responding again — live notes have resumed.")

        self.notes = result.state
        self._notes_last_update = time.monotonic()
        self._scheduler.finish_update(success=True)
        if result.changed:
            self._safe(self.callbacks.on_notes, self.notes)
            self._save_notes()
        self._emit_status()

    async def _run_consolidation(self) -> None:
        assert self._consolidator is not None
        if not self._scheduler.begin_consolidation():
            return
        self._notes_updating = True
        self._safe(self.callbacks.on_notes_activity, True)
        try:
            result = await self._consolidator.consolidate(
                self.notes, session_title=self.meta.title
            )
        except LLMUnavailableError as exc:
            self._on_llm_unavailable(str(exc))
            self._scheduler.finish_consolidation()
            return
        except LLMError as exc:
            log.warning("consolidation error: %s", exc)
            self._scheduler.finish_consolidation()
            return
        finally:
            self._notes_updating = False
            self._safe(self.callbacks.on_notes_activity, False)

        if result.applied:
            self.notes = result.state
            self._safe(self.callbacks.on_notes, self.notes)
            self._save_notes()
        self._scheduler.finish_consolidation()
        self._emit_status()

    def _on_llm_unavailable(self, detail: str) -> None:
        """Ollama went away: pause notes, keep transcribing, tell the user once."""
        if self._notes_available:
            self._notes_available = False
            self._notes_detail = "Ollama unavailable"
            log.warning("Ollama unavailable: %s", detail)
            self._report_error(
                "Notes",
                "Ollama stopped responding. Your transcript is still being saved. "
                "Live notes will resume automatically when Ollama returns.",
            )
        self._emit_status()

    # -- markers and manual notes -----------------------------------------
    def add_marker(self, *, text: str = "", kind: MarkerKind = MarkerKind.IMPORTANT) -> Marker:
        """Record a flagged moment; it reaches the notes, timeline and final synthesis."""
        marker = Marker(time=self.elapsed, kind=kind, text=text.strip())
        self.markers.append(marker)
        try:
            self.store.save_markers(self.markers)
        except OSError as exc:
            log.error("could not save markers: %s", exc)
            self._report_error(
                "Storage",
                f"Saving your marker failed: {exc}. The recording is still running.",
            )

        self.notes.add_timeline_entry(
            TimelineEntry(
                time=marker.time,
                label=marker.label,
                kind="marker" if kind is MarkerKind.IMPORTANT else "note",
            )
        )
        if kind is MarkerKind.NOTE and marker.text:
            # A note the student typed is theirs: it is starred, attributed, and
            # protected from being dropped by consolidation.
            self.notes.add_bullets(
                "key_points",
                [
                    NoteItem(
                        text=marker.text,
                        topic=self.notes.current_topic,
                        starred=True,
                        timestamp=marker.time,
                        source="user",
                    )
                ],
            )
        self.notes.revision += 1
        self._scheduler.add_marker(marker.label, marker.time)
        self._safe(self.callbacks.on_marker, marker)
        self._safe(self.callbacks.on_notes, self.notes)
        self._save_notes()
        log.info("marker added at %s: %s", marker.clock, marker.label)
        return marker

    # -- persistence -------------------------------------------------------
    async def _persist_task(self) -> None:
        """Periodically flush metadata and note state so a crash costs little."""
        try:
            while True:
                await asyncio.sleep(PERSIST_INTERVAL_SECONDS)
                self._flush_state()
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            log.exception("persistence task failed")

    def _flush_state(self) -> None:
        self.meta.word_count = self._transcript_words
        self.meta.segment_count = len(self.segments)
        self.meta.duration_seconds = self.elapsed
        try:
            self.store.save_meta(self.meta)
            self._save_notes()
            self.store.sync_transcript()
        except OSError as exc:
            log.error("failed to persist session state: %s", exc)

    def _save_notes(self) -> None:
        try:
            self.store.save_note_state(self.notes, title=self.meta.title)
        except OSError as exc:
            log.error("failed to save notes: %s", exc)

    # -- callback plumbing -------------------------------------------------
    def _emit_status(self) -> None:
        self._safe(self.callbacks.on_status, self.status())

    def _report_error(self, category: str, message: str) -> None:
        self._safe(self.callbacks.on_error, category, message)

    @staticmethod
    def _safe(callback: Callable | None, *args) -> None:
        """Invoke a UI callback; a rendering bug must not stop a recording."""
        if callback is None:
            return
        try:
            callback(*args)
        except Exception:  # noqa: BLE001
            log.exception("UI callback raised")


@dataclass
class FinalizationOutcome:
    """Result of stopping and finalizing a session."""

    meta: SessionMeta
    final_markdown: str = ""
    error: str = ""
    steps: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.error
