"""The acceptance workflow, driven through the real UI.

This is the product spec's manual checklist, automated: start the app, record a
session, watch transcript and notes populate live, add a marker, pause, resume,
finish, get final notes, quit, reopen, confirm everything is still there, export
Markdown and check the file contains real session data.

Audio comes from the WAV fixture (the same path ``lectern record --file`` uses)
and the two model servers are the protocol-level fakes, so every other layer —
VAD, the whisper client, the note scheduler, persistence, the screens — is the
production code.
"""

from __future__ import annotations

import pytest

from lectern.app import LecternApp
from lectern.config.models import LecternConfig
from lectern.services import AppServices, SessionRequest

pytestmark = pytest.mark.asyncio

TITLE = "Test Lecture"


def build_config(fake_whisper, fake_ollama) -> LecternConfig:
    config = LecternConfig()
    config.transcription.server_url = fake_whisper.url
    config.transcription.model = "small.en"
    config.transcription.partials = False
    config.ollama.host = fake_ollama.url
    config.ollama.notes_model = "qwen3:8b"
    config.ollama.final_model = "qwen3:8b"
    config.notes.update_interval_seconds = 5.0
    config.notes.min_new_words = 5
    config.audio.save_recording = True
    return config


async def wait_for(pilot, predicate, *, timeout: float = 25.0, message: str = "condition") -> None:
    """Poll the UI until ``predicate`` holds, keeping the event loop running."""
    elapsed = 0.0
    step = 0.05
    while elapsed < timeout:
        if predicate():
            return
        await pilot.pause(step)
        elapsed += step
    raise AssertionError(f"timed out waiting for {message}")


async def test_full_acceptance_workflow(manager, fixture_wav, fake_whisper, fake_ollama, tmp_path):
    from lectern.screens.recording import RecordingScreen
    from lectern.screens.review import ReviewScreen
    from lectern.sessions.models import SessionStatus
    from lectern.widgets.notes import NotesView
    from lectern.widgets.transcript import SegmentLine, TranscriptView

    config = build_config(fake_whisper, fake_ollama)
    services = AppServices(config=config, _manager=manager)

    # Steps 1-6: open Lectern and start "Test Lecture" from a (file) audio source.
    request = SessionRequest(
        title=TITLE,
        course="BIO 113",
        whisper_model="small.en",
        notes_model="qwen3:8b",
        file_path=fixture_wav,
        file_speed=6.0,
    )
    app = LecternApp(services=services, start_request=request)

    async with app.run_test(size=(120, 40)) as pilot:
        await wait_for(
            pilot,
            lambda: isinstance(app.screen, RecordingScreen)
            and app.screen.pipeline is not None
            and app.screen.pipeline.is_recording,
            message="the recording screen to start",
        )
        screen = app.screen
        pipeline = screen.pipeline

        # Step 7: words appear incrementally in the live transcript.
        await wait_for(
            pilot,
            lambda: len(screen.query_one("#transcript", TranscriptView).query(SegmentLine)) >= 2,
            message="transcript segments to appear",
        )
        assert pipeline.is_recording, "transcription must not stop the recording"

        # Steps 8-10: notes populate while recording continues, and follow topics.
        await wait_for(
            pilot,
            lambda: not pipeline.notes.is_empty,
            message="live notes to be generated",
        )
        assert pipeline.is_recording
        assert pipeline.notes.topics
        notes_view = screen.query_one("#notes", NotesView)
        assert "Key Points" in str(notes_view._body.content) or pipeline.notes.key_points

        # Step 11: add an important marker.
        await pilot.press("m")
        await pilot.pause()
        assert len(pipeline.markers) == 1
        marker_time = pipeline.markers[0].time

        # Step 12: pause.
        await pilot.press("space")
        await pilot.pause()
        assert pipeline.is_paused

        # Step 13: resume.
        await pilot.press("space")
        await pilot.pause()
        assert pipeline.is_recording

        segments_before_stop = len(pipeline.segments)
        assert segments_before_stop >= 2

        # Steps 14-15: stop the session; confirm; final notes are generated.
        await pilot.press("q")
        await pilot.pause()
        await pilot.press("enter")  # confirm "Finish"

        await wait_for(
            pilot,
            lambda: isinstance(app.screen, ReviewScreen),
            timeout=60.0,
            message="session review to open after finalization",
        )
        review = app.screen
        session_id = review.session.meta.id
        finished_segment_count = len(review.session.segments)
        assert review.session.meta.has_final_notes
        assert "Executive Summary" in review.session.final_notes

        # Step 16: exit Lectern.
        await app.action_quit()

    # Steps 17-21: reopen Lectern. Quitting closed the previous services, so this
    # builds a fresh session manager exactly as a new launch would.
    from lectern.sessions.manager import SessionManager

    reopened_services = AppServices(config=config, _manager=SessionManager(config))
    app2 = LecternApp(services=reopened_services, open_session_id=session_id)
    async with app2.run_test(size=(120, 40)) as pilot:
        await wait_for(
            pilot, lambda: isinstance(app2.screen, ReviewScreen), message="review screen to reopen"
        )
        loaded = app2.screen.session

        assert loaded.meta.display_title
        assert loaded.meta.status is SessionStatus.COMPLETE
        # Speech that finished between the snapshot above and pressing q is
        # still captured, so this only grows.
        assert finished_segment_count >= segments_before_stop
        assert len(loaded.segments) == finished_segment_count
        assert "phospholipid bilayer" in loaded.transcript_text
        assert not loaded.notes.is_empty
        assert loaded.markers and loaded.markers[0].time == pytest.approx(marker_time)
        assert "Executive Summary" in loaded.final_notes

        # Step 22: export Markdown from the TUI.
        app2.screen._do_export("markdown")
        await pilot.pause()

    # Step 23: the exported file contains real session data.
    export_path = loaded.meta.folder / "export.md"
    assert export_path.exists()
    content = export_path.read_text(encoding="utf-8")
    assert TITLE in content or loaded.meta.final_title in content
    assert "BIO 113" in content
    assert "phospholipid bilayer" in content  # transcript
    assert "Executive Summary" in content  # final notes
    assert "## Markers" in content  # the marker survived
    assert "`00:00:" in content  # timestamps


async def test_quitting_mid_recording_leaves_a_recoverable_session(
    manager, fixture_wav, fake_whisper, fake_ollama
):
    """Ctrl+C during a lecture saves everything and offers recovery next launch."""
    from lectern.screens.recording import RecordingScreen
    from lectern.sessions.recovery import find_recoverable

    config = build_config(fake_whisper, fake_ollama)
    services = AppServices(config=config, _manager=manager)
    app = LecternApp(
        services=services,
        start_request=SessionRequest(
            title="Interrupted Lecture",
            whisper_model="small.en",
            notes_model="qwen3:8b",
            file_path=fixture_wav,
            file_speed=6.0,
        ),
    )

    async with app.run_test(size=(120, 40)) as pilot:
        await wait_for(
            pilot,
            lambda: isinstance(app.screen, RecordingScreen)
            and app.screen.pipeline is not None
            and len(app.screen.pipeline.segments) >= 2,
            message="a couple of segments to be transcribed",
        )
        screen = app.screen
        captured = len(screen.pipeline.segments)

        # Quit while recording: the confirmation appears, then the session is saved.
        quit_task = pilot.app.run_worker(app.action_quit(), name="quit")
        await pilot.pause()
        await pilot.press("enter")  # confirm "Save and quit"
        await wait_for(pilot, lambda: quit_task.is_finished, message="quit to complete")

    from lectern.sessions.manager import SessionManager

    fresh_manager = SessionManager(config)
    recoverable = find_recoverable(fresh_manager)
    assert len(recoverable) == 1
    assert recoverable[0].segment_count >= captured
    assert not recoverable[0].is_empty
    fresh_manager.close()
