"""Regression tests for the defects found in review of PR #1.

Each test here failed before its fix. They are grouped in one module because
they share a theme: data that was silently dropped on a path the happy-case
tests never took — resuming a session, an LLM outage, a long transcript, or a
backlogged shutdown.
"""

from __future__ import annotations

import asyncio
import wave

import numpy as np
import pytest

from lectern.config.models import LecternConfig, NotesConfig
from lectern.notes.models import NoteItem, NoteState
from lectern.notes.scheduler import NoteScheduler
from lectern.sessions.storage import AudioRecorder
from lectern.transcription.base import TranscriptSegment
from lectern.utils.audio_utils import TARGET_SAMPLE_RATE, read_wav


def segment(index: int, text: str, start: float = 0.0) -> TranscriptSegment:
    return TranscriptSegment(id=index, start_time=start, end_time=start + 4.0, text=text)


def words(count: int) -> str:
    return " ".join(f"word{index}" for index in range(count))


# --- P1: a resumed session must carry its earlier transcript ---------------


async def test_resumed_pipeline_starts_from_the_persisted_transcript(
    manager, fake_whisper, fake_ollama
):
    """Finishing a resumed session must not discard the pre-interruption half.

    ``finish_session`` rewrites transcript.md, recomputes word/segment counts
    and builds the finalizer input from ``pipeline.segments``. If the pipeline
    starts with an empty list on resume, all of that silently covers only the
    resumed portion.
    """
    from lectern.audio.file_source import FileAudioSource
    from lectern.pipeline import RecordingPipeline
    from lectern.transcription.whisper_cpp import WhisperCppBackend

    meta, store = manager.create(title="Interrupted", whisper_model="small.en")
    earlier = [segment(1, "First half of the lecture.", 0.0), segment(2, "Still the first half.", 5.0)]
    for item in earlier:
        store.append_segment(item)
    store.close()

    config = LecternConfig()
    config.transcription.server_url = fake_whisper.url

    pipeline = RecordingPipeline(
        config=config,
        source=FileAudioSource("/dev/null"),
        transcriber=WhisperCppBackend(server_url=fake_whisper.url),
        llm=None,
        store=store,
        meta=meta,
        save_audio=False,
        start_segment_id=3,
        time_offset=9.0,
        initial_segments=store.load_segments(),
    )

    assert len(pipeline.segments) == 2
    assert pipeline.status().word_count == 9
    assert pipeline.status().segment_count == 2
    # The earlier speech must not be re-sent to the note model: it was already
    # turned into notes before the interruption.
    assert not pipeline._scheduler.has_pending  # noqa: SLF001


# --- P1: long-transcript finalization must keep notes and markers ----------


async def test_long_transcript_finalization_keeps_markers_and_user_notes(fake_ollama):
    """A quick note typed during the lecture is not in the transcript at all.

    The single-pass path passed the note state and markers to the model; the
    map/reduce path dropped both, so any lecture long enough to be chunked lost
    explicitly flagged material from its study guide.
    """
    from lectern.llm.ollama import OllamaBackend
    from lectern.notes.finalizer import NoteFinalizer

    state = NoteState(summary="A long lecture.")
    state.add_bullets(
        "key_points",
        [NoteItem(text="Professor said this is on Exam 1", starred=True, source="user")],
    )

    llm = OllamaBackend(fake_ollama.url)
    finalizer = NoteFinalizer(llm, model="qwen3:8b", num_ctx=4096)
    result = await finalizer.finalize(
        state=state,
        transcript=" ".join(f"word{index}" for index in range(6000)),
        session_title="Long Lecture",
        markers="- 00:12:30 Important",
    )
    await llm.close()

    assert result.ok
    assert result.chunks_used > 1, "this transcript should have been chunked"

    reduce_prompts = [
        request["prompt"]
        for request in fake_ollama.prompts
        if "Combine them into one coherent study guide" in request["prompt"]
    ]
    assert reduce_prompts, "the reduce step never ran"
    final_prompt = reduce_prompts[-1]
    assert "Professor said this is on Exam 1" in final_prompt
    assert "00:12:30" in final_prompt


# --- P1: resuming must not truncate the existing recording -----------------


async def test_resuming_appends_to_the_existing_recording(tmp_path):
    """Reopening audio.wav with 'wb' destroyed everything recorded before."""
    path = tmp_path / "audio.wav"

    first = AudioRecorder(path)
    first.open()
    first.write(np.full(TARGET_SAMPLE_RATE, 0.5, dtype=np.float32))
    first.close()

    original, rate = read_wav(path)
    assert original.size == TARGET_SAMPLE_RATE

    second = AudioRecorder(path)
    second.open()
    second.write(np.full(TARGET_SAMPLE_RATE // 2, -0.25, dtype=np.float32))
    second.close()

    combined, rate = read_wav(path)
    assert rate == TARGET_SAMPLE_RATE
    assert combined.size == TARGET_SAMPLE_RATE + TARGET_SAMPLE_RATE // 2
    # The original audio is intact and the new audio follows it.
    assert combined[:TARGET_SAMPLE_RATE] == pytest.approx(original, abs=1e-4)
    assert combined[TARGET_SAMPLE_RATE] == pytest.approx(-0.25, abs=1e-3)

    # The header must still describe the file correctly.
    with wave.open(str(path), "rb") as handle:
        assert handle.getnframes() == combined.size
        assert handle.getnchannels() == 1
        assert handle.getsampwidth() == 2


async def test_incompatible_existing_recording_starts_a_fresh_file(tmp_path):
    """A file we cannot safely append to is replaced, not corrupted."""
    path = tmp_path / "audio.wav"
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)  # stereo: not our format
        handle.setsampwidth(2)
        handle.setframerate(44_100)
        handle.writeframes(b"\x00" * 400)

    recorder = AudioRecorder(path)
    recorder.open()
    recorder.write(np.full(1000, 0.1, dtype=np.float32))
    recorder.close()

    audio, rate = read_wav(path)
    assert rate == TARGET_SAMPLE_RATE
    assert audio.size == 1000


# --- P2: a failed note update must not lose its transcript ----------------


def test_failed_update_returns_its_batch_to_the_queue():
    """An Ollama outage must not erase the speech from the live notes."""
    scheduler = NoteScheduler(NotesConfig(update_interval_seconds=15, min_new_words=5))
    scheduler.add_segment(segment(1, words(40)))
    scheduler.add_marker("Important", 12.0)

    batch = scheduler.take_batch()
    assert batch is not None
    assert batch.words == 40
    assert "00:00:12" in batch.markers

    scheduler.finish_update(success=False)

    # The same speech, and the marker, are available to the retry.
    assert scheduler.pending_words == 40
    retry = scheduler.take_batch()
    assert retry is not None
    assert retry.text == batch.text
    assert "00:00:12" in retry.markers


def test_successful_update_does_not_replay_its_batch():
    scheduler = NoteScheduler(NotesConfig(update_interval_seconds=15, min_new_words=5))
    scheduler.add_segment(segment(1, words(40)))
    scheduler.take_batch()
    scheduler.finish_update(success=True)

    assert scheduler.pending_words == 0
    assert scheduler.take_batch() is None


def test_drain_includes_a_batch_whose_update_never_finished():
    """Stopping mid-update must not strand the in-flight text."""
    scheduler = NoteScheduler(NotesConfig())
    scheduler.add_segment(segment(1, "in flight text"))
    scheduler.add_segment(segment(2, "still pending text", 10.0))
    scheduler.take_batch()

    drained = scheduler.drain()
    assert "in flight text" in drained
    assert "still pending text" in drained


# --- P2: shutdown must not hang on a full utterance queue -----------------


async def test_stop_completes_promptly_with_a_full_utterance_queue(
    manager, fake_whisper, fake_ollama
):
    """The sentinel must land even when the queue is at capacity.

    Dropping it left the transcription task blocked on an empty queue after
    draining, so finishing hung until the 60-second timeout cancelled it.
    """
    from lectern.audio.vad import Utterance
    from lectern.pipeline import RecordingPipeline
    from lectern.transcription.whisper_cpp import WhisperCppBackend

    meta, store = manager.create(title="Backlogged", whisper_model="small.en")
    config = LecternConfig()
    config.transcription.server_url = fake_whisper.url

    from lectern.audio.file_source import FileAudioSource

    pipeline = RecordingPipeline(
        config=config,
        source=FileAudioSource("/dev/null"),
        transcriber=WhisperCppBackend(server_url=fake_whisper.url),
        llm=None,
        store=store,
        meta=meta,
        save_audio=False,
    )

    await pipeline.transcriber.start()
    pipeline.store.open_transcript()
    pipeline.state = pipeline.state.__class__.RECORDING
    task = asyncio.create_task(pipeline._transcribe_task(), name="transcribe")  # noqa: SLF001
    pipeline._tasks = [task]  # noqa: SLF001

    # Fill the queue to capacity with real utterances.
    audio = np.full(int(TARGET_SAMPLE_RATE * 0.5), 0.2, dtype=np.float32)
    for index in range(pipeline._utterances.maxsize):  # noqa: SLF001
        pipeline._enqueue_utterance(  # noqa: SLF001
            Utterance(audio=audio, start_time=index * 0.5, end_time=index * 0.5 + 0.5)
        )
    assert pipeline._utterances.full()  # noqa: SLF001

    started = asyncio.get_running_loop().time()
    await asyncio.wait_for(pipeline.stop(), timeout=45.0)
    elapsed = asyncio.get_running_loop().time() - started

    assert task.done()
    # Well under the 60s timeout that the dropped sentinel used to force.
    assert elapsed < 40.0
    assert pipeline.segments, "queued utterances should still have been transcribed"
