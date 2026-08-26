"""End-to-end pipeline tests.

These drive the *production* code path — real VAD, real whisper.cpp HTTP client,
real Ollama HTTP client, real persistence — with the two servers replaced by
in-process fakes that speak the same protocols. A WAV file stands in for the
microphone, exactly as ``lectern record --file`` does.
"""

from __future__ import annotations

import asyncio

import pytest

from lectern.audio.file_source import FileAudioSource
from lectern.config.models import LecternConfig
from lectern.llm.ollama import OllamaBackend
from lectern.notes.models import NoteState
from lectern.pipeline import PipelineCallbacks, PipelineState, RecordingPipeline
from lectern.sessions.models import MarkerKind, SessionStatus
from lectern.sessions.storage import SessionStore
from lectern.transcription.whisper_cpp import WhisperCppBackend

pytestmark = pytest.mark.asyncio


def build_config(fake_ollama, **overrides) -> LecternConfig:
    config = LecternConfig()
    config.ollama.host = fake_ollama.url
    config.ollama.notes_model = "qwen3:8b"
    config.ollama.final_model = "qwen3:8b"
    # Update aggressively so a 15-second fixture produces several note cycles.
    config.notes.update_interval_seconds = 5.0
    config.notes.min_new_words = 5
    config.transcription.partials = False
    for key, value in overrides.items():
        section, field = key.split(".")
        setattr(getattr(config, section), field, value)
    return config


async def run_session(
    manager,
    fixture_wav,
    fake_whisper,
    fake_ollama,
    *,
    speed: float = 8.0,
    title: str = "Test Lecture",
    on_started=None,
    config: LecternConfig | None = None,
):
    """Run a complete session over the fixture WAV and return the pipeline."""
    config = config or build_config(fake_ollama)
    meta, store = manager.create(
        title=title, course="BIO 113", whisper_model="small.en", ollama_model="qwen3:8b"
    )

    source = FileAudioSource(fixture_wav, speed=speed)
    transcriber = WhisperCppBackend(server_url=fake_whisper.url)
    llm = OllamaBackend(fake_ollama.url)

    events: dict[str, list] = {"segments": [], "notes": [], "errors": []}
    pipeline = RecordingPipeline(
        config=config,
        source=source,
        transcriber=transcriber,
        llm=llm,
        store=store,
        meta=meta,
        callbacks=PipelineCallbacks(
            on_segment=lambda segment: events["segments"].append(segment),
            on_notes=lambda state: events["notes"].append(state.copy()),
            on_error=lambda category, message: events["errors"].append((category, message)),
        ),
        save_audio=True,
        notes_model="qwen3:8b",
    )

    await pipeline.start()
    if on_started is not None:
        await on_started(pipeline)

    # Wait for the file to finish replaying, then let the workers drain.
    deadline = asyncio.get_running_loop().time() + 30
    while source.running and asyncio.get_running_loop().time() < deadline:
        await asyncio.sleep(0.05)
    await asyncio.sleep(1.5)
    await pipeline.stop()
    await llm.close()

    pipeline.events = events  # type: ignore[attr-defined]
    return pipeline, meta, store


async def test_audio_to_transcript_to_notes(manager, fixture_wav, fake_whisper, fake_ollama):
    """The headline path: speech in, transcript and notes out, all persisted."""
    pipeline, meta, store = await run_session(manager, fixture_wav, fake_whisper, fake_ollama)

    # Transcription happened, one segment per detected utterance.
    assert len(pipeline.segments) == 5
    assert fake_whisper.call_count == 5
    assert "phospholipid bilayer" in pipeline.segments[1].text

    # Segments carry sane, increasing timestamps.
    starts = [segment.start_time for segment in pipeline.segments]
    assert starts == sorted(starts)
    assert pipeline.segments[0].start_time < 1.0

    # Notes were generated *during* the session, not after.
    assert fake_ollama.prompts, "the note model was never called"
    assert not pipeline.notes.is_empty
    assert pipeline.notes.summary
    assert pipeline.notes.topics

    # The transcript actually reached the model.
    update_prompts = [
        request["prompt"]
        for request in fake_ollama.prompts
        if "NEW TRANSCRIPT SINCE THE LAST UPDATE" in request["prompt"]
    ]
    assert update_prompts
    assert any("peptidoglycan" in prompt.lower() for prompt in update_prompts)

    # Everything is on disk.
    reloaded = SessionStore(meta.folder)
    assert len(reloaded.load_segments()) == 5
    assert not reloaded.load_note_state().is_empty
    assert reloaded.notes_live_md.exists()
    assert reloaded.audio_exists()


async def test_notes_prompt_never_carries_the_whole_transcript(
    manager, fixture_wav, fake_whisper, fake_ollama
):
    """Rolling updates send the delta, not the accumulated lecture."""
    await run_session(manager, fixture_wav, fake_whisper, fake_ollama)
    update_prompts = [
        request["prompt"]
        for request in fake_ollama.prompts
        if "NEW TRANSCRIPT SINCE THE LAST UPDATE" in request["prompt"]
    ]
    assert len(update_prompts) >= 2
    for prompt in update_prompts:
        transcript_block = prompt.split("NEW TRANSCRIPT SINCE THE LAST UPDATE")[1]
        assert len(transcript_block.split()) < 400


async def test_markers_reach_notes_transcript_and_prompt(
    manager, fixture_wav, fake_whisper, fake_ollama
):
    async def add_markers(pipeline):
        await asyncio.sleep(0.4)
        pipeline.add_marker(kind=MarkerKind.IMPORTANT)
        pipeline.add_marker(text="Professor said this is on Exam 1", kind=MarkerKind.NOTE)

    pipeline, meta, store = await run_session(
        manager, fixture_wav, fake_whisper, fake_ollama, on_started=add_markers
    )

    assert len(pipeline.markers) == 2
    stored = SessionStore(meta.folder).load_markers()
    assert [marker.text for marker in stored][1] == "Professor said this is on Exam 1"

    # The typed note is a first-class, attributed, starred note.
    user_items = [item for item in pipeline.notes.key_points if item.source == "user"]
    assert user_items and user_items[0].starred
    assert any(entry.kind in {"marker", "note"} for entry in pipeline.notes.timeline)

    # And the model was told about the flagged moment.
    assert any("Exam 1" in request["prompt"] for request in fake_ollama.prompts)


async def test_pause_stops_the_clock_and_the_transcript(
    manager, fixture_wav, fake_whisper, fake_ollama
):
    async def pause_immediately(pipeline):
        pipeline.pause()
        assert pipeline.state is PipelineState.PAUSED
        await asyncio.sleep(1.0)

    pipeline, _, _ = await run_session(
        manager, fixture_wav, fake_whisper, fake_ollama, speed=4.0, on_started=pause_immediately
    )
    # Paused audio is discarded, so the whole 15 s file cannot have been consumed.
    assert pipeline.elapsed < 14.0


async def test_pause_then_resume_keeps_recording(manager, fixture_wav, fake_whisper, fake_ollama):
    async def pause_and_resume(pipeline):
        await asyncio.sleep(0.3)
        pipeline.pause()
        await asyncio.sleep(0.3)
        pipeline.resume()
        assert pipeline.state is PipelineState.RECORDING

    pipeline, _, _ = await run_session(
        manager, fixture_wav, fake_whisper, fake_ollama, on_started=pause_and_resume
    )
    assert pipeline.segments
    assert pipeline.state is PipelineState.STOPPED


async def test_transcription_continues_when_ollama_dies(
    manager, fixture_wav, fake_whisper, fake_ollama
):
    """The transcript is the priority stream: an LLM outage must not touch it."""

    async def kill_ollama(pipeline):
        await asyncio.sleep(0.3)
        fake_ollama.available = False

    pipeline, meta, _ = await run_session(
        manager, fixture_wav, fake_whisper, fake_ollama, on_started=kill_ollama
    )

    assert len(pipeline.segments) == 5
    assert len(SessionStore(meta.folder).load_segments()) == 5
    assert pipeline.state is PipelineState.STOPPED
    # The user was told, in plain language, without the app falling over.
    messages = [message for _, message in pipeline.events["errors"]]
    assert any("transcript is still being saved" in message for message in messages)


async def test_unparseable_model_output_keeps_previous_notes(
    manager, fixture_wav, fake_whisper, fake_ollama
):
    from fakes import unavailable_responder

    fake_ollama.responder = unavailable_responder
    pipeline, _, _ = await run_session(manager, fixture_wav, fake_whisper, fake_ollama)

    assert len(pipeline.segments) == 5
    assert pipeline.notes.is_empty  # nothing valid was ever committed
    assert pipeline.state is PipelineState.STOPPED


async def test_whisper_failure_does_not_stop_the_session(
    manager, fixture_wav, fake_whisper, fake_ollama
):
    fake_whisper.fail_after = 2
    pipeline, meta, _ = await run_session(manager, fixture_wav, fake_whisper, fake_ollama)

    assert len(pipeline.segments) == 2  # the two that succeeded are kept
    assert len(SessionStore(meta.folder).load_segments()) == 2
    assert pipeline.state is PipelineState.STOPPED


async def test_crash_midway_leaves_a_recoverable_session(
    manager, fixture_wav, fake_whisper, fake_ollama
):
    """Simulate the process dying: no stop(), no finalization."""
    from lectern.sessions import recovery

    config = build_config(fake_ollama)
    meta, store = manager.create(title="Crashed Lecture", whisper_model="small.en")
    source = FileAudioSource(fixture_wav, speed=8.0)
    transcriber = WhisperCppBackend(server_url=fake_whisper.url)
    llm = OllamaBackend(fake_ollama.url)
    pipeline = RecordingPipeline(
        config=config,
        source=source,
        transcriber=transcriber,
        llm=llm,
        store=store,
        meta=meta,
        save_audio=False,
        notes_model="qwen3:8b",
    )
    await pipeline.start()
    while len(pipeline.segments) < 3:
        await asyncio.sleep(0.05)

    # Hard stop: cancel every task without any graceful shutdown.
    for task in pipeline._tasks:  # noqa: SLF001 - deliberately simulating a crash
        task.cancel()
    await asyncio.sleep(0.1)
    await source.stop()
    await transcriber.stop()
    await llm.close()

    recoverable = recovery.find_recoverable(manager)
    assert len(recoverable) == 1
    assert recoverable[0].segment_count >= 3

    recovered = recovery.recover(manager, recoverable[0].meta)
    assert recovered.status is SessionStatus.NEEDS_FINALIZATION
    assert recovered.word_count > 0
    assert len(manager.open(recovered.id).segments) >= 3


async def test_final_synthesis_produces_a_study_guide(
    manager, fixture_wav, fake_whisper, fake_ollama
):
    from lectern.notes.finalizer import NoteFinalizer

    pipeline, meta, store = await run_session(manager, fixture_wav, fake_whisper, fake_ollama)

    finalizer = NoteFinalizer(OllamaBackend(fake_ollama.url), model="qwen3:8b")
    result = await finalizer.finalize(
        state=pipeline.notes,
        transcript=" ".join(segment.text for segment in pipeline.segments),
        session_title=meta.title,
        course=meta.course,
        duration="15s",
        markers="",
    )
    assert result.ok
    assert result.title == "Bacterial Cell Structure and Gram Staining"
    assert "## Executive Summary" in result.markdown
    assert "Peptidoglycan" in result.markdown

    store.save_final_notes(result.markdown)
    assert "Executive Summary" in SessionStore(meta.folder).load_final_notes()


async def test_long_transcript_is_chunked_not_truncated(fake_ollama):
    """A lecture longer than the context window keeps its beginning."""
    from lectern.notes.finalizer import NoteFinalizer, chunk_transcript

    transcript = " ".join(f"word{index}" for index in range(6000))
    chunks = chunk_transcript(transcript, max_words=1000)
    assert len(chunks) > 1
    assert chunks[0].startswith("word0 ")
    assert chunks[-1].endswith("word5999")
    # Overlap means consecutive chunks share text rather than cutting mid-thought.
    assert set(chunks[0].split()) & set(chunks[1].split())

    llm = OllamaBackend(fake_ollama.url)
    finalizer = NoteFinalizer(llm, model="qwen3:8b", num_ctx=4096)
    progress: list[str] = []
    result = await finalizer.finalize(
        state=NoteState(),
        transcript=transcript,
        session_title="Long Lecture",
        on_progress=lambda event: progress.append(event.step),
    )
    await llm.close()

    assert result.ok
    assert result.chunks_used > 1
    assert "reduce" in progress  # the map step ran
    # Every chunk was summarised: no part of the lecture was skipped.
    chunk_calls = [
        request for request in fake_ollama.prompts if "This is part" in request["prompt"]
    ]
    assert len(chunk_calls) == result.chunks_used
