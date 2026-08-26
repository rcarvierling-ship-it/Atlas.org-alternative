"""Browse every recorded session, with a live title filter."""

from __future__ import annotations

from textual import on
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import Footer, Input, Label, ListItem, ListView, Static

from lectern.screens.home import SessionRow
from lectern.sessions.models import SessionMeta


class SessionsScreen(Screen):
    """All sessions, newest first."""

    BINDINGS = [
        ("escape", "back", "Back"),
        ("slash", "focus_filter", "Filter"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._sessions: list[SessionMeta] = []

    def compose(self) -> ComposeResult:
        with Vertical(id="settings-body"):
            yield Label("SESSIONS", classes="section-title")
            yield Input(placeholder="Filter by title or course…", id="search-input")
            yield ListView(id="search-results")
        yield Footer()

    def on_mount(self) -> None:
        self._sessions = self.app.services.sessions.all_sessions()
        self._render_rows(self._sessions)
        self.query_one("#search-results", ListView).focus()

    def _render_rows(self, sessions: list[SessionMeta]) -> None:
        listing = self.query_one("#search-results", ListView)
        listing.clear()
        if not sessions:
            listing.append(
                ListItem(Static("No sessions match.", classes="empty-state"))
            )
            return
        for meta in sessions:
            listing.append(SessionRow(meta))

    @on(Input.Changed, "#search-input")
    def _filter(self, event: Input.Changed) -> None:
        needle = event.value.strip().lower()
        if not needle:
            self._render_rows(self._sessions)
            return
        self._render_rows(
            [
                meta
                for meta in self._sessions
                if needle in f"{meta.display_title} {meta.course}".lower()
            ]
        )

    @on(Input.Submitted, "#search-input")
    def _submitted(self) -> None:
        self.query_one("#search-results", ListView).focus()

    @on(ListView.Selected, "#search-results")
    def _open(self, event: ListView.Selected) -> None:
        if isinstance(event.item, SessionRow):
            self.app.open_session(event.item.meta.id)

    def action_focus_filter(self) -> None:
        self.query_one("#search-input", Input).focus()

    def action_back(self) -> None:
        self.app.pop_screen()
