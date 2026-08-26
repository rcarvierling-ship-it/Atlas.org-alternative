"""Note scheduler timing, debounce and backpressure.

The clock is injected so these assertions are exact rather than timing-dependent.
"""

from __future__ import annotations

from lectern.config.models import NotesConfig
from lectern.notes.scheduler import NoteScheduler
from lectern.transcription.base import TranscriptSegment


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def make_segment(text: str, index: int = 1, start: float = 0.0) -> TranscriptSegment:
    return TranscriptSegment(id=index, start_time=start, end_time=start + 3.0, text=text)


def words(count: int) -> str:
    return " ".join(f"word{index}" for index in range(count))


def test_no_update_without_transcript():
    clock = FakeClock()
    scheduler = NoteScheduler(NotesConfig(), clock=clock)
    clock.advance(120)
    assert not scheduler.should_update()


def test_no_update_before_the_interval_elapses():
    clock = FakeClock()
    scheduler = NoteScheduler(NotesConfig(update_interval_seconds=15), clock=clock)
    scheduler.add_segment(make_segment(words(50)))
    clock.advance(10)
    assert not scheduler.should_update()
    clock.advance(6)
    assert scheduler.should_update()


def test_no_update_for_a_trickle_of_words():
    clock = FakeClock()
    scheduler = NoteScheduler(NotesConfig(update_interval_seconds=15, min_new_words=25), clock=clock)
    scheduler.add_segment(make_segment("um, okay, so"))
    clock.advance(60)
    assert not scheduler.should_update()


def test_large_backlog_triggers_early():
    """A fast talker should not sit behind the interval."""
    clock = FakeClock()
    scheduler = NoteScheduler(NotesConfig(update_interval_seconds=15, min_new_words=25), clock=clock)
    scheduler.add_segment(make_segment(words(200)))
    clock.advance(2)
    assert scheduler.should_update()


def test_marker_forces_the_next_update():
    clock = FakeClock()
    scheduler = NoteScheduler(NotesConfig(update_interval_seconds=15), clock=clock)
    scheduler.add_segment(make_segment(words(5)))
    scheduler.add_marker("Important", 42.0)
    assert scheduler.should_update()
    batch = scheduler.take_batch()
    assert "00:00:42" in batch.markers


def test_no_concurrent_updates_and_new_speech_is_buffered():
    clock = FakeClock()
    scheduler = NoteScheduler(NotesConfig(update_interval_seconds=15), clock=clock)
    scheduler.add_segment(make_segment(words(40), index=1))
    clock.advance(20)

    batch = scheduler.take_batch()
    assert batch is not None
    assert scheduler.running
    # More speech arrives mid-update.
    scheduler.add_segment(make_segment(words(40), index=2, start=10))
    assert not scheduler.should_update()
    assert scheduler.take_batch() is None

    scheduler.finish_update(success=True)
    clock.advance(20)
    assert scheduler.should_update()
    second = scheduler.take_batch()
    assert second.words == 40


def test_batch_is_capped_but_nothing_is_dropped():
    """Overflow is deferred to the next cycle, never discarded."""
    clock = FakeClock()
    scheduler = NoteScheduler(
        NotesConfig(update_interval_seconds=15, max_context_words=100), clock=clock
    )
    for index in range(4):
        scheduler.add_segment(make_segment(words(40), index=index + 1, start=index * 4))
    clock.advance(20)

    first = scheduler.take_batch()
    assert first.words == 80  # two segments fit; a third would exceed the cap
    assert scheduler.pending_words == 80
    scheduler.finish_update()

    clock.advance(20)
    second = scheduler.take_batch()
    assert second.words == 80
    assert not scheduler.has_pending


def test_failed_update_backs_off_but_retries_soon():
    clock = FakeClock()
    scheduler = NoteScheduler(NotesConfig(update_interval_seconds=20), clock=clock)
    scheduler.add_segment(make_segment(words(40)))
    clock.advance(25)
    scheduler.take_batch()
    scheduler.finish_update(success=False)

    scheduler.add_segment(make_segment(words(40), index=2))
    clock.advance(5)
    assert not scheduler.should_update()
    clock.advance(6)
    assert scheduler.should_update()


def test_consolidation_is_due_on_its_own_schedule():
    clock = FakeClock()
    scheduler = NoteScheduler(NotesConfig(consolidate_interval_seconds=180), clock=clock)
    clock.advance(100)
    assert not scheduler.should_consolidate()
    clock.advance(100)
    assert scheduler.should_consolidate()
    assert scheduler.begin_consolidation()
    assert not scheduler.begin_consolidation()
    scheduler.finish_consolidation()
    assert not scheduler.should_consolidate()


def test_drain_returns_everything_buffered():
    scheduler = NoteScheduler(NotesConfig())
    scheduler.add_segment(make_segment("first part"))
    scheduler.add_segment(make_segment("second part", index=2))
    assert scheduler.drain() == "first part second part"
    assert not scheduler.has_pending


def test_non_final_segments_are_ignored():
    scheduler = NoteScheduler(NotesConfig())
    segment = make_segment("unstable partial")
    segment.is_final = False
    scheduler.add_segment(segment)
    assert not scheduler.has_pending
