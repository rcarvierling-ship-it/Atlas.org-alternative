"""Home screen: start a session, reopen a recent one, see local AI status."""

from __future__ import annotations

from rich.text import Text
from textual import on, work
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import Footer, Label, ListItem, ListView, Static

from lectern.logging_setup import get_logger
from lectern.sessions.models import SessionMeta, SessionStatus
from lectern.theme import ICONS
from lectern.utils.text import truncate
from lectern.utils.timefmt import format_duration, format_relative

log = get_logger("screens.home")

RECENT_LIMIT = 8


class SessionRow(ListItem):
    """One recent session: when, what, how long."""

    def __init__(self, meta: SessionMeta) -> None:
        super().__init__()
        self.meta = meta

    def compose(self) -> ComposeResult:
        yield Static(self._render_row())

    def _render_row(self) -> Text:
        meta = self.meta
        row = Text()
        row.append(f"{format_relative(meta.created_at):<10}", style="#5f6672")
        title = meta.display_title
        if meta.course:
            title = f"{meta.course} {ICONS.dash} {title}"
        row.append(f"{truncate(title, 46):<48}", style="#e6e8ec")
        row.append(f"{format_duration(meta.duration_seconds):>8}", style="#8b919e")
        if meta.status in (SessionStatus.RECORDING, SessionStatus.INCOMPLETE):
            row.append("  unfinished", style="#fbbf24")
        elif meta.status is SessionStatus.NEEDS_FINALIZATION:
            row.append("  no final notes", style="#8b919e")
        return row


class StatusRow(Static):
    """One line of the LOCAL AI panel."""

    def __init__(self, label: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self._label = label
        self._value = "checking…"
        self._state = "pending"

    def on_mount(self) -> None:
        self._refresh_row()

    def set_value(self, value: str, state: str = "ok") -> None:
        self._value = value
        self._state = state
        self._refresh_row()

    def _refresh_row(self) -> None:
        icon, colour = {
            "ok": (ICONS.live, "#4ade80"),
            "warn": (ICONS.live, "#fbbf24"),
            "bad": (ICONS.live, "#f87171"),
            "pending": (ICONS.dot, "#5f6672"),
        }[self._state]
        row = Text()
        row.append(f"{self._label:<13}", style="#8b919e")
        row.append(f"{icon} ", style=colour)
        row.append(self._value[:22], style="#c3c9d4")
        self.update(row)


class HomeScreen(Screen):
    """The landing screen."""

    BINDINGS = [
        ("n", "new_session", "New session"),
        ("s", "browse_sessions", "Sessions"),
        ("slash", "search", "Search"),
        ("comma", "settings", "Settings"),
        ("d", "doctor", "Doctor"),
        ("question_mark", "help", "Help"),
        ("q", "quit_app", "Quit"),
    ]

    def compose(self) -> ComposeResult:
        yield Static("LECTERN", id="home-title")
        yield Static("Local AI lecture intelligence", id="home-subtitle")
        with Horizontal(id="home-body"):
            with Vertical(id="home-main"):
                yield Static(self._new_session_card(), id="new-session-card")
                yield Label("RECENT", classes="section-title")
                yield ListView(id="recent-list")
            with Vertical(id="home-side"):
                yield Label("LOCAL AI", classes="section-title")
                with Vertical(id="status-panel"):
                    yield StatusRow("Whisper", id="status-whisper")
                    yield StatusRow("Ollama", id="status-ollama")
                    yield StatusRow("Notes model", id="status-model")
                    yield StatusRow("Microphone", id="status-mic")
                    yield StatusRow("Storage", id="status-storage")
        yield Footer()

    def _new_session_card(self) -> Text:
        card = Text()
        card.append("New Session", style="bold #e6e8ec")
        card.append(f"        {ICONS.enter}\n", style="#7c7cff")
        card.append("Start recording and live note-taking", style="#8b919e")
        return card

    def on_mount(self) -> None:
        card = self.query_one("#new-session-card", Static)
        card.can_focus = True
        card.focus()
        self.refresh_sessions()
        self.check_environment()

    def on_screen_resume(self) -> None:
        """Recent sessions may have changed while we were on another screen."""
        self.refresh_sessions()

    # -- data --------------------------------------------------------------
    def refresh_sessions(self) -> None:
        listing = self.query_one("#recent-list", ListView)
        listing.clear()
        try:
            sessions = self.app.services.sessions.list_recent(RECENT_LIMIT)
        except Exception as exc:  # noqa: BLE001
            log.exception("could not list sessions")
            listing.append(ListItem(Static(f"Could not read sessions: {exc}", classes="bad")))
            return

        if not sessions:
            listing.append(
                ListItem(
                    Static(
                        "No sessions yet — press n to record your first lecture.",
                        classes="empty-state",
                    )
                )
            )
            return
        for meta in sessions:
            listing.append(SessionRow(meta))

    @work(exclusive=True, group="home-status")
    async def check_environment(self) -> None:
        """Fill in the LOCAL AI panel without blocking the first paint."""
        import asyncio

        from lectern.audio.devices import default_input_device, sounddevice_available
        from lectern.doctor import check_storage
        from lectern.transcription.models import find_model, find_whisper_server

        services = self.app.services
        config = services.config

        binary = await asyncio.to_thread(find_whisper_server, config.transcription.whisper_server_binary)
        model_path = await asyncio.to_thread(find_model, config.transcription.model)
        whisper_row = self.query_one("#status-whisper", StatusRow)
        if binary is None and not config.transcription.server_url:
            whisper_row.set_value("not installed", "bad")
        elif model_path is None:
            whisper_row.set_value(f"{config.transcription.model} missing", "bad")
        else:
            whisper_row.set_value(f"Ready {ICONS.dot} {config.transcription.model}", "ok")

        ollama_row = self.query_one("#status-ollama", StatusRow)
        model_row = self.query_one("#status-model", StatusRow)
        health = await services.refresh_llm_health()
        if health.available:
            ollama_row.set_value("Running", "ok")
            configured = config.ollama.notes_model
            installed = {entry.name for entry in health.models}
            if not configured:
                model_row.set_value("none selected", "warn")
            elif configured in installed:
                model_row.set_value(f"{configured} ready", "ok")
            else:
                model_row.set_value(f"{configured} not pulled", "bad")
        else:
            ollama_row.set_value("Not running", "bad")
            model_row.set_value("unavailable", "bad")

        mic_row = self.query_one("#status-mic", StatusRow)
        if not await asyncio.to_thread(sounddevice_available):
            mic_row.set_value("PortAudio missing", "bad")
        else:
            device = await asyncio.to_thread(default_input_device)
            if device is None:
                mic_row.set_value("no input device", "bad")
            else:
                mic_row.set_value(device.name, "ok")

        storage_check = await asyncio.to_thread(check_storage)
        storage_row = self.query_one("#status-storage", StatusRow)
        state = {"ok": "ok", "warn": "warn", "fail": "bad", "unknown": "pending"}[
            str(storage_check.status)
        ]
        storage_row.set_value(storage_check.detail.split(" at ")[0], state)

    # -- actions -----------------------------------------------------------
    def action_new_session(self) -> None:
        from lectern.screens.new_session import NewSessionScreen

        self.app.push_screen(NewSessionScreen())

    def action_browse_sessions(self) -> None:
        from lectern.screens.sessions import SessionsScreen

        self.app.push_screen(SessionsScreen())

    def action_search(self) -> None:
        from lectern.screens.search import SearchScreen

        self.app.push_screen(SearchScreen())

    def action_settings(self) -> None:
        from lectern.screens.settings import SettingsScreen

        self.app.push_screen(SettingsScreen())

    def action_doctor(self) -> None:
        from lectern.screens.setup_wizard import SetupWizardScreen

        self.app.push_screen(SetupWizardScreen(mode="doctor"))

    def action_help(self) -> None:
        from lectern.screens.modals import HelpModal

        self.app.push_screen(HelpModal())

    def action_quit_app(self) -> None:
        self.app.exit()

    @on(ListView.Selected, "#recent-list")
    def _open_session(self, event: ListView.Selected) -> None:
        if isinstance(event.item, SessionRow):
            self.app.open_session(event.item.meta.id)

    def on_key(self, event) -> None:  # noqa: ANN001
        """Enter on the New Session card starts a session."""
        if event.key == "enter" and self.focused is not None and self.focused.id == "new-session-card":
            event.stop()
            self.action_new_session()

    def on_click(self, event) -> None:  # noqa: ANN001
        widget = getattr(event, "widget", None)
        if widget is not None and widget.id == "new-session-card":
            self.action_new_session()
