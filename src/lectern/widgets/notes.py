"""Live notes view.

The note state is rendered as a single Rich renderable rather than a tree of
widgets. Notes are rewritten wholesale on every update, and swapping one
renderable is both faster and visually calmer than mounting and unmounting
dozens of widgets — no flicker, no scroll jump, no layout thrash.

Section headings are small-caps-ish uppercase in muted text; emphasis (the
model's "this is test-worthy" signal) is a single star in amber. Nothing else
is coloured.
"""

from __future__ import annotations

from rich.console import Group, RenderableType
from rich.table import Table
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.reactive import reactive
from textual.widgets import Static

from lectern.notes.models import BULLET_FIELDS, SECTION_TITLES, TERM_FIELDS, NoteState
from lectern.theme import ICONS

#: Order sections appear in the live pane — most useful during a lecture first.
DISPLAY_ORDER: tuple[str, ...] = (
    "key_points",
    "definitions",
    "key_terms",
    "important_details",
    "formulas",
    "examples",
    "questions",
    "unclear_points",
)

EMPTY_MESSAGE = (
    "Notes will appear here.\n\n"
    "Lectern reads the transcript as it arrives and writes structured notes\n"
    "every few seconds — you don't need to stop recording."
)


class NotesBody(Static):
    """Renders a ``NoteState`` as Rich content."""

    DEFAULT_CSS = """
    NotesBody { height: auto; padding: 0 1 1 0; }
    """

    def __init__(self) -> None:
        super().__init__()
        self._state: NoteState | None = None

    def set_state(self, state: NoteState) -> None:
        self._state = state
        self.update(self._render_state())

    def _render_state(self) -> RenderableType:
        state = self._state
        if state is None or state.is_empty:
            return Text(EMPTY_MESSAGE, style="#5f6672")

        blocks: list[RenderableType] = []

        if state.summary:
            blocks.append(Text(state.summary + "\n", style="#c3c9d4"))

        for name in DISPLAY_ORDER:
            entries = getattr(state, name, [])
            if not entries:
                continue
            blocks.append(Text(SECTION_TITLES[name].upper(), style="bold #8b919e"))
            # A two-column grid gives bullets a hanging indent: wrapped lines
            # align under the text, not under the marker.
            grid = Table.grid(padding=(0, 1))
            grid.add_column(width=3, no_wrap=True)
            grid.add_column(ratio=1, overflow="fold")

            for entry in entries:
                starred = getattr(entry, "starred", False)
                marker = Text(f"  {ICONS.star}" if starred else f"  {ICONS.bullet}")
                marker.stylize("#fbbf24" if starred else "#5f6672")

                body = Text()
                if name in TERM_FIELDS:
                    body.append(entry.term, style="bold #e6e8ec")
                    if entry.definition:
                        body.append(f" {ICONS.dash} ", style="#5f6672")
                        body.append(entry.definition, style="#c3c9d4")
                else:
                    body.append(entry.text, style="#e6e8ec" if starred else "#c3c9d4")
                    if entry.source == "user":
                        body.append("  (your note)", style="italic #5f6672")
                grid.add_row(marker, body)

            blocks.append(grid)
            blocks.append(Text(""))

        return Group(*blocks)


class NotesView(VerticalScroll):
    """Scrollable notes pane with the same follow-live contract as the transcript."""

    BINDINGS = [("f", "follow_live", "Follow")]

    following: reactive[bool] = reactive(True)
    updating: reactive[bool] = reactive(False)

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.can_focus = True
        self._body = NotesBody()
        self._revision = -1

    def compose(self) -> ComposeResult:
        yield self._body

    def on_mount(self) -> None:
        self._body.set_state(NoteState())

    def update_notes(self, state: NoteState) -> None:
        """Re-render if the state actually changed."""
        if state.revision == self._revision:
            return
        self._revision = state.revision
        was_at_bottom = self.following and self.scroll_offset.y >= self.max_scroll_y - 1
        self._body.set_state(state)
        if was_at_bottom:
            self.call_after_refresh(lambda: self.scroll_end(animate=False))

    def action_follow_live(self) -> None:
        self.following = True
        self.scroll_end(animate=False)

    def action_scroll_up(self) -> None:
        self.following = False
        super().action_scroll_up()

    def action_page_up(self) -> None:
        self.following = False
        super().action_page_up()

    def action_scroll_end(self) -> None:
        super().action_scroll_end()
        self.following = True
