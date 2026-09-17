"""Detected topics side panel.

Topics arrive from the note model as the lecture moves on. The current one is
marked with a caret and brightened; the rest are quiet. Timestamps are shown
only when the panel is wide enough to carry them without wrapping.
"""

from __future__ import annotations

from rich.console import RenderableType
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Static

from lectern.notes.models import NoteState
from lectern.theme import ICONS
from lectern.utils.timefmt import format_clock

NARROW_WIDTH = 24


class TopicsBody(Static):
    DEFAULT_CSS = "TopicsBody { height: auto; }"

    def __init__(self) -> None:
        super().__init__()
        self._topics: list[str] = []
        self._current = ""
        self._times: dict[str, float] = {}

    def set_state(self, state: NoteState) -> None:
        self._topics = list(state.topics)
        self._current = state.current_topic
        self._times = {
            entry.label: entry.time for entry in state.timeline if entry.kind == "topic"
        }
        self.update(self._render_topics())

    def _render_topics(self) -> RenderableType:
        if not self._topics:
            return Text("No topics detected yet.", style="#5f6672")

        show_times = self.size.width >= NARROW_WIDTH
        rendered = Text()
        for topic in self._topics:
            is_current = topic == self._current
            rendered.append(
                f"{ICONS.topic} " if is_current else "  ",
                style="#7c7cff" if is_current else "",
            )
            rendered.append(
                topic,
                style="bold #e6e8ec" if is_current else "#a5abb8",
            )
            timestamp = self._times.get(topic)
            if show_times and timestamp is not None:
                rendered.append(f"\n    {format_clock(timestamp)}", style="#5f6672")
            rendered.append("\n")
        return rendered


class TopicsPanel(VerticalScroll):
    """Scrollable list of detected topics."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.can_focus = True
        self._body = TopicsBody()

    def compose(self) -> ComposeResult:
        yield self._body

    def update_notes(self, state: NoteState) -> None:
        self._body.set_state(state)
