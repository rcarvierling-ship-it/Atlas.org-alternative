"""The status bar.

Everything shown here is already known to the pipeline — it is pushed on state
changes and refreshed once a second for the clock. Nothing polls the operating
system: sampling CPU and memory every second would cost more than it tells the
user, so those are deliberately absent rather than faked.
"""

from __future__ import annotations

import time

from rich.text import Text
from textual.widgets import Static

from lectern.pipeline import PipelineState, PipelineStatus
from lectern.theme import ICONS

LEVEL_BLOCKS = "▁▂▃▄▅▆▇█"

STATE_LABELS: dict[PipelineState, tuple[str, str]] = {
    PipelineState.IDLE: ("Idle", "#8b919e"),
    PipelineState.STARTING: ("Starting", "#fbbf24"),
    PipelineState.RECORDING: ("Listening", "#4ade80"),
    PipelineState.PAUSED: ("Paused", "#fbbf24"),
    PipelineState.STOPPING: ("Finishing", "#fbbf24"),
    PipelineState.STOPPED: ("Stopped", "#8b919e"),
    PipelineState.FAILED: ("Failed", "#f87171"),
}


class StatusBar(Static):
    """One-line summary of the pipeline's health."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._status = PipelineStatus()
        self._model_label = ""

    def set_model_label(self, label: str) -> None:
        self._model_label = label
        self.refresh_status(self._status)

    def refresh_status(self, status: PipelineStatus) -> None:
        self._status = status
        self.update(self._render_status())

    def _render_status(self) -> Text:
        status = self._status
        line = Text()

        label, colour = STATE_LABELS.get(status.state, ("Unknown", "#8b919e"))
        icon = ICONS.paused if status.state is PipelineState.PAUSED else ICONS.record
        line.append(f"{icon} ", style=colour)
        line.append(label, style=colour)

        if status.state is PipelineState.RECORDING:
            line.append("  ")
            line.append(self._level_meter(status.audio_level), style="#4ade80")

        self._separator(line)
        if status.stt_latency_ms is not None:
            line.append("STT ", style="#5f6672")
            line.append(f"{status.stt_latency_ms:.0f}ms")
        else:
            line.append("STT ", style="#5f6672")
            line.append("ready" if status.stt_ready else "starting…", style="#8b919e")

        if status.stt_backlog > 1:
            line.append(f" (+{status.stt_backlog} queued)", style="#fbbf24")

        self._separator(line)
        line.append("Notes ", style="#5f6672")
        if not status.notes_available:
            line.append(status.notes_detail or "unavailable", style="#fbbf24")
        elif status.notes_updating:
            line.append(f"{ICONS.spinner} updating…", style="#56d4dd")
        elif status.notes_last_update is not None:
            seconds = int(time.monotonic() - status.notes_last_update)
            line.append(f"{seconds}s ago" if seconds < 90 else f"{seconds // 60}m ago")
        else:
            line.append("waiting for speech", style="#8b919e")

        self._separator(line)
        line.append(f"{status.word_count:,}", style="#e6e8ec")
        line.append(" words", style="#5f6672")

        if status.dropped_utterances:
            self._separator(line)
            line.append(f"{status.dropped_utterances} dropped", style="#f87171")

        if self._model_label:
            self._separator(line)
            line.append(self._model_label, style="#8b919e")

        return line

    @staticmethod
    def _separator(line: Text) -> None:
        line.append("  │  ", style="#2b303a")

    @staticmethod
    def _level_meter(level: float, width: int = 8) -> str:
        """Render the input level as a compact bar of block glyphs."""
        if ICONS.ascii_only:
            filled = int(min(1.0, level * 6) * width)
            return "[" + "=" * filled + " " * (width - filled) + "]"
        # Speech sits around 0.05-0.3 RMS, so scale for that range.
        scaled = min(1.0, level * 6.0)
        cells = []
        for index in range(width):
            threshold = (index + 1) / width
            if scaled >= threshold:
                cells.append(LEVEL_BLOCKS[-1])
            elif scaled > threshold - (1 / width):
                fraction = (scaled - (threshold - 1 / width)) * width
                cells.append(LEVEL_BLOCKS[max(0, min(len(LEVEL_BLOCKS) - 1, int(fraction * 8)))])
            else:
                cells.append(" ")
        return "".join(cells)
