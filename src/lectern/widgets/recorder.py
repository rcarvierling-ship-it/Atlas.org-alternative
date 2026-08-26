"""Recording screen header: identity, state, timer and the models in use."""

from __future__ import annotations

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widgets import Static

from lectern.pipeline import PipelineState, PipelineStatus
from lectern.theme import ICONS
from lectern.utils.timefmt import format_clock


class RecorderHeader(Static):
    """Two-line header: what is being recorded, and how it is going."""

    def __init__(
        self,
        *,
        title: str,
        course: str = "",
        whisper_model: str = "",
        ollama_model: str = "",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._title = title
        self._course = course
        self._whisper_model = whisper_model
        self._ollama_model = ollama_model
        self._status = PipelineStatus()

    def compose(self) -> ComposeResult:
        with Horizontal(id="rec-header-top"):
            yield Static("LECTERN", id="rec-title")
            yield Static(self._course_label(), id="rec-course")
        with Horizontal(id="rec-header-bottom"):
            yield Static("", id="rec-state")
            yield Static("00:00:00", id="rec-timer")
            yield Static(self._models_label(), id="rec-models")

    def on_mount(self) -> None:
        self.refresh_status(self._status)

    def _course_label(self) -> str:
        if self._course:
            return f"{self._title}  {ICONS.dot}  {self._course}"
        return self._title

    def _models_label(self) -> str:
        parts = []
        if self._whisper_model:
            parts.append(f"Whisper: {self._whisper_model}")
        if self._ollama_model:
            parts.append(f"Ollama: {self._ollama_model}")
        return f"   {ICONS.dot}   ".join(parts)

    def refresh_status(self, status: PipelineStatus) -> None:
        self._status = status
        try:
            state_widget = self.query_one("#rec-state", Static)
            timer_widget = self.query_one("#rec-timer", Static)
        except Exception:  # noqa: BLE001 - header not composed yet
            return
        state_widget.update(self._state_text(status))
        timer_widget.update(format_clock(status.elapsed))

    @staticmethod
    def _state_text(status: PipelineStatus) -> Text:
        mapping = {
            PipelineState.RECORDING: (f"{ICONS.record} RECORDING", "#f87171"),
            PipelineState.PAUSED: (f"{ICONS.paused} PAUSED", "#fbbf24"),
            PipelineState.STARTING: (f"{ICONS.spinner} STARTING", "#fbbf24"),
            PipelineState.STOPPING: (f"{ICONS.spinner} FINISHING", "#fbbf24"),
            PipelineState.STOPPED: (f"{ICONS.stopped} STOPPED", "#8b919e"),
            PipelineState.FAILED: (f"{ICONS.cross} FAILED", "#f87171"),
            PipelineState.IDLE: (f"{ICONS.stopped} IDLE", "#8b919e"),
        }
        label, colour = mapping.get(status.state, ("UNKNOWN", "#8b919e"))
        return Text(label, style=f"bold {colour}")
