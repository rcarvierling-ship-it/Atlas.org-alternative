"""TUI smoke tests driven headlessly through Textual's Pilot."""

from __future__ import annotations

import pytest

from lectern.app import LecternApp
from lectern.config.models import LecternConfig
from lectern.notes.models import NoteItem, NoteState
from lectern.services import AppServices
from lectern.sessions.models import SessionStatus
from lectern.transcription.base import TranscriptSegment

pytestmark = pytest.mark.asyncio


def finish(manager, meta, store) -> None:
    """Close a test session so it is not treated as interrupted."""
    meta.status = SessionStatus.COMPLETE
    store.save_meta(meta)
    manager.index.upsert(meta)

TERMINAL_SIZES = [(100, 30), (120, 40), (150, 45)]


def make_app(config: LecternConfig | None = None, **kwargs) -> LecternApp:
    services = AppServices(config=config or LecternConfig())
    return LecternApp(services=services, **kwargs)


async def test_home_screen_renders():
    from lectern.screens.home import HomeScreen

    app = make_app()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        assert isinstance(app.screen, HomeScreen)
        assert app.theme == "lectern-dark"
        assert "New Session" in str(app.screen.query_one("#new-session-card").content)


@pytest.mark.parametrize("size", TERMINAL_SIZES)
async def test_home_screen_is_responsive(size):
    """The layout must survive the terminal sizes people actually use."""
    app = make_app()
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        # An exception during layout would surface here.
        assert app.screen.query_one("#home-body").size.width > 0


async def test_navigate_to_new_session_and_back():
    from lectern.screens.home import HomeScreen
    from lectern.screens.new_session import NewSessionScreen

    app = make_app()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.press("n")
        await pilot.pause()
        assert isinstance(app.screen, NewSessionScreen)

        await pilot.press("escape")
        await pilot.pause()
        assert isinstance(app.screen, HomeScreen)


async def test_new_session_requires_a_title():
    from lectern.screens.new_session import NewSessionScreen

    app = make_app()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.press("n")
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, NewSessionScreen)
        screen.action_start()
        await pilot.pause()
        # Still on the form: a session was never created.
        assert isinstance(app.screen, NewSessionScreen)


async def test_settings_screen_saves_config(tmp_path):
    from lectern.config import manager as config_manager
    from lectern.screens.settings import SettingsScreen
    from lectern.utils import paths

    app = make_app()
    async with app.run_test(size=(120, 45)) as pilot:
        await pilot.press("comma")
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, SettingsScreen)

        screen.query_one("#language").value = "de"
        screen.query_one("#update-interval").value = "25"
        screen.action_save()
        await pilot.pause()

    assert paths.config_file().exists()
    saved = config_manager.load()
    assert saved.transcription.language == "de"
    assert saved.notes.update_interval_seconds == 25


async def test_help_and_command_palette_open():
    from lectern.screens.modals import HelpModal

    app = make_app()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.press("question_mark")
        await pilot.pause()
        assert isinstance(app.screen, HelpModal)
        await pilot.press("escape")
        await pilot.pause()

        commands = list(app.get_system_commands(app.screen))
        titles = [command.title for command in commands]
        assert "New session" in titles
        assert "Search" in titles


async def test_doctor_screen_runs_checks():
    from lectern.screens.setup_wizard import CheckLine, SetupWizardScreen

    app = make_app()
    async with app.run_test(size=(120, 45)) as pilot:
        await pilot.press("d")
        await pilot.pause()
        screen = app.screen
        assert isinstance(screen, SetupWizardScreen)

        for _ in range(80):
            await pilot.pause(0.05)
            if screen.query(CheckLine):
                break
        assert screen.query(CheckLine)


async def test_transcript_follow_live_behaviour():
    """Scrolling up stops following; the pending count then accumulates."""
    from lectern.widgets.transcript import TranscriptView

    from textual.app import App, ComposeResult

    class Harness(App[None]):
        def compose(self) -> ComposeResult:
            yield TranscriptView(id="transcript")

    app = Harness()
    async with app.run_test(size=(80, 12)) as pilot:
        view = app.query_one("#transcript", TranscriptView)
        for index in range(1, 30):
            view.add_segment(
                TranscriptSegment(
                    id=index,
                    start_time=index * 5.0,
                    end_time=index * 5.0 + 4,
                    text=f"Segment number {index} with some words in it.",
                )
            )
        await pilot.pause()
        assert view.following
        assert view.pending_count == 0

        view.action_scroll_up()
        await pilot.pause()
        assert not view.following

        view.add_segment(TranscriptSegment(id=99, start_time=500.0, end_time=504.0, text="New one"))
        await pilot.pause()
        assert view.pending_count == 1

        view.action_follow_live()
        await pilot.pause()
        assert view.following
        assert view.pending_count == 0


async def test_transcript_view_bounds_rendered_segments():
    from lectern.widgets.transcript import MAX_RENDERED_SEGMENTS, TranscriptView

    from textual.app import App, ComposeResult

    class Harness(App[None]):
        def compose(self) -> ComposeResult:
            yield TranscriptView(id="transcript")

    app = Harness()
    async with app.run_test(size=(80, 12)) as pilot:
        view = app.query_one("#transcript", TranscriptView)
        for index in range(MAX_RENDERED_SEGMENTS + 50):
            view.add_segment(
                TranscriptSegment(id=index, start_time=index, end_time=index + 1, text="word")
            )
        await pilot.pause()
        assert len(view._segment_lines) <= MAX_RENDERED_SEGMENTS


async def test_notes_view_renders_state_and_skips_unchanged_revisions():
    from lectern.widgets.notes import NotesView

    from textual.app import App, ComposeResult

    class Harness(App[None]):
        def compose(self) -> ComposeResult:
            yield NotesView(id="notes")

    app = Harness()
    async with app.run_test(size=(100, 20)) as pilot:
        view = app.query_one("#notes", NotesView)
        state = NoteState(summary="A summary of the lecture.", current_topic="Cells")
        state.add_bullets("key_points", [NoteItem(text="Membranes are bilayers", starred=True)])
        state.revision = 1
        view.update_notes(state)
        await pilot.pause()
        assert view._revision == 1

        state.summary = "Changed but same revision"
        view.update_notes(state)
        assert view._revision == 1


async def test_review_screen_shows_stored_session(manager):
    from lectern.screens.review import ReviewScreen
    from lectern.sessions.models import Marker

    meta, store = manager.create(title="Reviewable", course="BIO 113")
    store.append_segment(
        TranscriptSegment(id=1, start_time=0.0, end_time=4.0, text="Cells have membranes.")
    )
    store.save_markers([Marker(time=2.0)])
    store.save_final_notes("# Final Guide\n\n## Executive Summary\n\nAll about cells.")
    finish(manager, meta, store)
    store.close()
    manager.reindex()

    services = AppServices(config=LecternConfig(), _manager=manager)
    app = LecternApp(services=services, open_session_id=meta.id)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        await pilot.pause()
        assert isinstance(app.screen, ReviewScreen)
        assert "Reviewable" in str(app.screen.query_one("#review-title").content)

        await pilot.press("t")
        await pilot.pause()
        assert app.screen.query_one("#review-tabs").active == "tab-transcript"


async def test_search_screen_finds_a_session(manager):
    from lectern.screens.search import HitRow, SearchScreen

    meta, store = manager.create(title="Searchable")
    store.append_segment(
        TranscriptSegment(
            id=1, start_time=0.0, end_time=4.0, text="Mitochondria are the powerhouse of the cell."
        )
    )
    finish(manager, meta, store)
    store.close()
    manager.reindex()

    services = AppServices(config=LecternConfig(), _manager=manager)
    app = LecternApp(services=services)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.press("slash")
        await pilot.pause()
        assert isinstance(app.screen, SearchScreen)

        app.screen.query_one("#search-input").value = "mitochondria"
        await pilot.press("enter")
        for _ in range(40):
            await pilot.pause(0.05)
            if app.screen.query(HitRow):
                break
        rows = app.screen.query(HitRow)
        assert rows
        assert rows.first().hit.session_id == meta.id
