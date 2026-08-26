"""Decides *when* to spend a note update, and on what text.

This is the backpressure valve between a transcript that arrives continuously
and a local model that takes several seconds per update. It is deliberately a
pure, clock-injected object with no async in it, so its timing rules can be
tested exactly rather than by sleeping.

Rules:

* Never run two updates at once. New speech that arrives during a run is
  buffered for the next cycle.
* Run when the interval has elapsed *and* enough new words have accumulated —
  waking a model to process "um, okay, so" is wasted latency.
* Run early when a lot of speech has piled up, so a fast talker does not sit
  behind the interval.
* Never hand the model more than ``max_context_words`` at once, and cut only
  on a segment boundary so the model is never handed half an utterance. The
  remainder stays buffered — the transcript is never dropped, only deferred.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

from lectern.config.models import NotesConfig
from lectern.transcription.base import TranscriptSegment
from lectern.utils.text import word_count

#: Multiple of ``min_new_words`` that triggers an update before the interval.
EARLY_TRIGGER_MULTIPLE = 6


@dataclass(slots=True)
class PendingBatch:
    """Text handed to one update, plus where it sits in the session."""

    text: str
    start_time: float
    end_time: float
    markers: str = ""

    @property
    def words(self) -> int:
        return word_count(self.text)


@dataclass
class NoteScheduler:
    """Buffers finalized transcript and decides when an update is due."""

    config: NotesConfig
    clock: Callable[[], float] = time.monotonic
    _pending: list[TranscriptSegment] = field(default_factory=list, init=False)
    _markers: list[str] = field(default_factory=list, init=False)
    #: The batch handed to the update currently running. Held so a failed
    #: update can put it back rather than losing the speech entirely.
    _inflight: list[TranscriptSegment] = field(default_factory=list, init=False)
    _inflight_markers: list[str] = field(default_factory=list, init=False)
    _last_update: float = field(default=0.0, init=False)
    _last_consolidate: float = field(default=0.0, init=False)
    _running: bool = field(default=False, init=False)
    _forced: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        now = self.clock()
        self._last_update = now
        self._last_consolidate = now

    # -- input -------------------------------------------------------------
    def add_segment(self, segment: TranscriptSegment) -> None:
        """Buffer a finalized transcript segment."""
        if not segment.is_final or not segment.text.strip():
            return
        self._pending.append(segment)

    def add_marker(self, label: str, timestamp: float) -> None:
        """Attach a marker so the next update knows the student flagged this moment."""
        from lectern.utils.timefmt import format_clock

        self._markers.append(f"- {format_clock(timestamp)} {label}")
        # A marker means "pay attention here", so bring the next update forward.
        self._forced = True

    def force_next(self) -> None:
        """Request an update as soon as one can run (used when stopping)."""
        self._forced = True

    # -- state -------------------------------------------------------------
    @property
    def running(self) -> bool:
        return self._running

    @property
    def pending_words(self) -> int:
        return sum(word_count(segment.text) for segment in self._pending)

    @property
    def has_pending(self) -> bool:
        return bool(self._pending)

    def seconds_since_update(self, now: float | None = None) -> float:
        return (now if now is not None else self.clock()) - self._last_update

    # -- decisions ---------------------------------------------------------
    def should_update(self, now: float | None = None) -> bool:
        """True when an update should start right now."""
        if self._running or not self._pending:
            return False
        now = now if now is not None else self.clock()
        words = self.pending_words

        if self._forced:
            return True
        if words >= self.config.min_new_words * EARLY_TRIGGER_MULTIPLE:
            return True
        if now - self._last_update < self.config.update_interval_seconds:
            return False
        return words >= self.config.min_new_words

    def should_consolidate(self, now: float | None = None) -> bool:
        """True when the periodic clean-up pass is due."""
        if self._running:
            return False
        now = now if now is not None else self.clock()
        return (now - self._last_consolidate) >= self.config.consolidate_interval_seconds

    # -- taking work -------------------------------------------------------
    def take_batch(self) -> PendingBatch | None:
        """Claim buffered transcript for an update and mark the scheduler busy."""
        if self._running or not self._pending:
            return None

        limit = self.config.max_context_words
        taken: list[TranscriptSegment] = []
        used = 0
        for segment in self._pending:
            words = word_count(segment.text)
            if taken and used + words > limit:
                break
            taken.append(segment)
            used += words

        self._pending = self._pending[len(taken) :]
        taken_markers, self._markers = self._markers, []
        markers = "\n".join(taken_markers)
        self._inflight = taken
        self._inflight_markers = taken_markers
        self._running = True
        self._forced = False

        return PendingBatch(
            text=" ".join(segment.text.strip() for segment in taken).strip(),
            start_time=taken[0].start_time,
            end_time=taken[-1].end_time,
            markers=markers,
        )

    def finish_update(self, *, success: bool = True, now: float | None = None) -> None:
        """Release the busy flag after an update completes (or fails).

        A failed update returns its batch to the front of the queue. Without
        that, an Ollama outage would consume a batch per cycle and erase that
        speech from the notes permanently — the retry would only ever see
        newly arriving words.
        """
        self._running = False
        now = now if now is not None else self.clock()
        if success:
            self._last_update = now
        else:
            self._pending = self._inflight + self._pending
            self._markers = self._inflight_markers + self._markers
            # Back off a little on failure rather than hammering a sick backend,
            # but stay well under the normal interval so recovery is quick.
            self._last_update = now - self.config.update_interval_seconds / 2
        self._inflight = []
        self._inflight_markers = []

    def finish_consolidation(self, now: float | None = None) -> None:
        self._running = False
        self._last_consolidate = now if now is not None else self.clock()

    def begin_consolidation(self) -> bool:
        """Claim the scheduler for a consolidation pass."""
        if self._running:
            return False
        self._running = True
        return True

    def drain(self) -> str:
        """Return every buffered word without scheduling anything.

        Includes a batch whose update never completed, so stopping mid-update
        does not strand it.
        """
        segments = self._inflight + self._pending
        text = " ".join(segment.text.strip() for segment in segments).strip()
        self._pending = []
        self._inflight = []
        return text
