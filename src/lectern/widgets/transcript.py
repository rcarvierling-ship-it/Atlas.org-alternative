"""Live transcript view.

Two behaviours make this readable during a lecture rather than a wall of text:

* **Follow-live with an escape hatch.** New segments scroll into view only
  while the reader is already at the bottom. The moment they scroll up to
  re-read something, following stops and a "↓ N newer segments" pill appears;
  pressing ``f`` (or clicking the pill) returns to live.
* **The newest segment is subtly brighter** with an accent rule down its left
  edge, so the eye finds the live edge without any animation.

Timestamps are deliberately quiet: dim, fixed width, never competing with the
words.
"""

from __future__ import annotations

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.reactive import reactive
from textual.widgets import Static

from lectern.theme import ICONS
from lectern.transcription.base import TranscriptSegment
from lectern.utils.timefmt import format_clock

#: Cap on rendered segments. The full transcript is always on disk; keeping the
#: widget tree bounded is what stops a three-hour lecture from slowing the UI.
MAX_RENDERED_SEGMENTS = 400


class SegmentLine(Static):
    """One transcript segment."""

    DEFAULT_CSS = """
    SegmentLine {
        height: auto;
        padding: 0 0 1 1;
        border-left: blank;
        color: #c3c9d4;
    }
    SegmentLine.-newest {
        color: $foreground;
        border-left: outer $primary;
    }
    SegmentLine.-partial {
        color: $text-muted;
        text-style: italic;
        border-left: outer $secondary;
    }
    SegmentLine.-marker {
        color: $warning;
        border-left: outer $warning;
    }
    """

    def __init__(self, timestamp: float, text: str, *, kind: str = "segment") -> None:
        super().__init__(id=None)
        self._timestamp = timestamp
        self._text = text
        self._kind = kind
        if kind == "partial":
            self.add_class("-partial")
        elif kind == "marker":
            self.add_class("-marker")

    def render(self) -> Text:
        rendered = Text()
        rendered.append(f"{format_clock(self._timestamp)}  ", style="#5f6672")
        rendered.append(self._text)
        return rendered

    def update_text(self, text: str) -> None:
        self._text = text
        self.refresh(layout=True)


class TranscriptView(VerticalScroll):
    """Scrollable transcript with follow-live behaviour."""

    BINDINGS = [
        ("f", "follow_live", "Follow live"),
        ("g", "scroll_home", "Top"),
        ("G", "follow_live", "Live"),
    ]

    following: reactive[bool] = reactive(True)
    pending_count: reactive[int] = reactive(0)

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.can_focus = True
        self._partial: SegmentLine | None = None
        self._segment_lines: list[SegmentLine] = []
        self._empty: Static | None = None

    def compose(self) -> ComposeResult:
        self._empty = Static(
            "Waiting for speech…\nTranscription begins as soon as someone starts talking.",
            classes="empty-state",
        )
        yield self._empty

    # -- content -----------------------------------------------------------
    def add_segment(self, segment: TranscriptSegment) -> None:
        """Append a finalized segment, replacing any partial hypothesis."""
        self._clear_empty()
        self.clear_partial()

        for line in self._segment_lines[-1:]:
            line.remove_class("-newest")

        line = SegmentLine(segment.start_time, segment.text)
        line.add_class("-newest")
        self.mount(line)
        self._segment_lines.append(line)
        self._trim()

        if self.following:
            self.call_after_refresh(self._scroll_to_live)
        else:
            self.pending_count += 1

    def set_partial(self, text: str) -> None:
        """Show the unstable hypothesis for speech in progress."""
        if not text.strip():
            return
        self._clear_empty()
        if self._partial is None:
            self._partial = SegmentLine(0.0, text, kind="partial")
            self.mount(self._partial)
        else:
            self._partial.update_text(text)
        if self.following:
            self.call_after_refresh(self._scroll_to_live)

    def clear_partial(self) -> None:
        if self._partial is not None:
            self._partial.remove()
            self._partial = None

    def add_marker(self, timestamp: float, label: str) -> None:
        """Show a flagged moment inline in the transcript."""
        self._clear_empty()
        line = SegmentLine(timestamp, f"{ICONS.star} {label}", kind="marker")
        self.mount(line, before=self._partial if self._partial is not None else None)
        self._segment_lines.append(line)
        if self.following:
            self.call_after_refresh(self._scroll_to_live)

    def load_segments(self, segments: list[TranscriptSegment]) -> None:
        """Populate from a stored session (used when resuming)."""
        for segment in segments[-MAX_RENDERED_SEGMENTS:]:
            self._clear_empty()
            line = SegmentLine(segment.start_time, segment.text)
            self.mount(line)
            self._segment_lines.append(line)
        if self._segment_lines:
            self._segment_lines[-1].add_class("-newest")
            self.call_after_refresh(self._scroll_to_live)

    def _clear_empty(self) -> None:
        if self._empty is not None:
            self._empty.remove()
            self._empty = None

    def _trim(self) -> None:
        while len(self._segment_lines) > MAX_RENDERED_SEGMENTS:
            self._segment_lines.pop(0).remove()

    # -- follow behaviour --------------------------------------------------
    def _scroll_to_live(self) -> None:
        self.scroll_end(animate=False)
        self.pending_count = 0

    def action_follow_live(self) -> None:
        self.following = True
        self._scroll_to_live()

    def _stop_following(self) -> None:
        if self.following:
            self.following = False

    # Any deliberate upward movement drops out of follow mode.
    def action_scroll_up(self) -> None:
        self._stop_following()
        super().action_scroll_up()

    def action_page_up(self) -> None:
        self._stop_following()
        super().action_page_up()

    def action_scroll_home(self) -> None:
        self._stop_following()
        super().action_scroll_home()

    def action_scroll_down(self) -> None:
        super().action_scroll_down()
        self._resume_if_at_bottom()

    def action_page_down(self) -> None:
        super().action_page_down()
        self._resume_if_at_bottom()

    def action_scroll_end(self) -> None:
        super().action_scroll_end()
        self.following = True
        self.pending_count = 0

    def on_mouse_scroll_up(self, event) -> None:  # noqa: ANN001, ARG002
        """Mouse wheel up is a deliberate scroll back through the transcript."""
        self._stop_following()

    def on_mouse_scroll_down(self, event) -> None:  # noqa: ANN001, ARG002
        self.call_after_refresh(self._resume_if_at_bottom)

    def _resume_if_at_bottom(self) -> None:
        if self.scroll_offset.y >= self.max_scroll_y - 1:
            self.following = True
            self.pending_count = 0
