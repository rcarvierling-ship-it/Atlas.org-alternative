"""Full-text search across every session's transcript and notes."""

from __future__ import annotations

from rich.text import Text
from textual import on, work
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import Footer, Input, Label, ListItem, ListView, Static

from lectern.sessions.index import SearchHit
from lectern.theme import ICONS
from lectern.utils.timefmt import format_relative


class HitRow(ListItem):
    """A matching session with its highlighted snippets."""

    def __init__(self, hit: SearchHit) -> None:
        super().__init__()
        self.hit = hit

    def compose(self) -> ComposeResult:
        rendered = Text()
        rendered.append(f"{self.hit.title}", style="bold #e6e8ec")
        if self.hit.course:
            rendered.append(f"  {ICONS.dot}  {self.hit.course}", style="#8b919e")
        rendered.append(f"  {format_relative(self.hit.created_at)}\n", style="#5f6672")
        for snippet in self.hit.snippets[:2]:
            rendered.append("  ", style="#a5abb8")
            highlighted = _highlight(snippet)
            highlighted.stylize("#a5abb8", 0, len(highlighted))
            rendered.append_text(highlighted)
            rendered.append("\n")
        yield Static(rendered)


def _highlight(snippet: str) -> Text:
    """Turn FTS5's ``[...]`` markers into accent-coloured spans."""
    rendered = Text()
    remainder = snippet
    while "[" in remainder and "]" in remainder:
        before, _, rest = remainder.partition("[")
        match, _, remainder = rest.partition("]")
        rendered.append(before)
        rendered.append(match, style="bold #7c7cff")
    rendered.append(remainder)
    return rendered


class SearchScreen(Screen):
    """Search transcripts and notes."""

    BINDINGS = [("escape", "back", "Back")]

    def compose(self) -> ComposeResult:
        with Vertical(id="settings-body"):
            yield Label("SEARCH", classes="section-title")
            yield Input(placeholder="gram positive", id="search-input")
            yield ListView(id="search-results")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#search-input", Input).focus()
        self._show_message("Type a phrase to search every session you have recorded.")

    def _show_message(self, message: str) -> None:
        listing = self.query_one("#search-results", ListView)
        listing.clear()
        listing.append(ListItem(Static(message, classes="empty-state")))

    @on(Input.Submitted, "#search-input")
    def _submitted(self, event: Input.Submitted) -> None:
        query = event.value.strip()
        if not query:
            return
        self._show_message("Searching…")
        self.run_search(query)

    @work(exclusive=True, group="search")
    async def run_search(self, query: str) -> None:
        import asyncio

        hits = await asyncio.to_thread(self.app.services.sessions.search, query)
        listing = self.query_one("#search-results", ListView)
        listing.clear()
        if not hits:
            listing.append(
                ListItem(Static(f"Nothing matched “{query}”.", classes="empty-state"))
            )
            return
        for hit in hits:
            listing.append(HitRow(hit))

    @on(ListView.Selected, "#search-results")
    def _open(self, event: ListView.Selected) -> None:
        if isinstance(event.item, HitRow):
            self.app.open_session(event.item.hit.session_id)

    def action_back(self) -> None:
        self.app.pop_screen()
