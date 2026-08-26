"""New session form.

Selects are populated from what is actually installed — Whisper weights on
disk, models Ollama reports, input devices PortAudio can see — so the form can
never offer a choice that will fail at start. Anything unavailable is disabled
with the reason shown, rather than silently missing.
"""

from __future__ import annotations

from textual import on, work
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, Footer, Input, Label, Select, Static, Switch

from lectern.config.models import AudioSourceKind
from lectern.logging_setup import get_logger
from lectern.services import SessionRequest
from lectern.theme import ICONS

log = get_logger("screens.new_session")

AUDIO_SOURCE_OPTIONS = [
    ("Microphone", AudioSourceKind.MICROPHONE.value),
    ("System audio", AudioSourceKind.SYSTEM.value),
    ("Microphone + system audio", AudioSourceKind.BOTH.value),
]


class NewSessionScreen(Screen):
    """Collects the details for a recording, then starts it."""

    BINDINGS = [
        ("escape", "cancel", "Back"),
        ("ctrl+s", "start", "Start recording"),
    ]

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="form-wrap"):
            with Vertical(id="form"):
                yield Label("New session", id="form-title")

                yield Label("Session title", classes="field-label")
                yield Input(placeholder="BIO 113 — Chapter 4", id="title")

                yield Label("Course or category", classes="field-label")
                yield Input(placeholder="Microbiology (optional)", id="course")

                yield Label("Audio source", classes="field-label")
                yield Select(AUDIO_SOURCE_OPTIONS, id="audio-source", allow_blank=False)

                yield Label("Input device", classes="field-label")
                yield Select([("Loading…", "")], id="device", allow_blank=False)

                yield Label("Whisper model", classes="field-label")
                yield Select([("Loading…", "")], id="whisper-model", allow_blank=False)

                yield Label("Notes model (Ollama)", classes="field-label")
                yield Select([("Loading…", "")], id="notes-model", allow_blank=False)

                with Horizontal(classes="settings-row"):
                    yield Label("Save the audio recording", classes="field-label")
                    yield Switch(value=True, id="save-audio")

                yield Static("", id="form-warning", classes="warn")
                with Horizontal(id="form-buttons"):
                    yield Button("Cancel", id="cancel")
                    yield Button("Start Recording", id="start", classes="-primary")
                yield Static(
                    f"ctrl+s starts recording {ICONS.dot} esc goes back",
                    classes="hint",
                )
        yield Footer()

    def on_mount(self) -> None:
        config = self.app.services.config
        self.query_one("#audio-source", Select).value = config.audio.source.value
        self.query_one("#save-audio", Switch).value = config.audio.save_recording
        self.query_one("#title", Input).focus()
        self.populate_choices()

    @work(exclusive=True, group="new-session-options")
    async def populate_choices(self) -> None:
        """Load device and model options in the background."""
        import asyncio

        from lectern.audio.devices import list_input_devices
        from lectern.transcription.models import installed_models

        config = self.app.services.config
        warnings: list[str] = []

        devices = await asyncio.to_thread(list_input_devices)
        device_select = self.query_one("#device", Select)
        if devices:
            device_select.set_options(
                [("System default", "")] + [(device.label, device.name) for device in devices]
            )
            # A device saved on another day may be unplugged now; assigning a
            # value outside the options raises and aborts this worker.
            available = {device.name for device in devices}
            device_select.value = (
                config.audio.input_device if config.audio.input_device in available else ""
            )
        else:
            device_select.set_options([("No input devices found", "")])
            device_select.disabled = True
            warnings.append("No microphone was detected.")

        whisper_select = self.query_one("#whisper-model", Select)
        models = await asyncio.to_thread(installed_models)
        if models:
            whisper_select.set_options(
                [(f"{model.name}  ({model.size_label})", model.name) for model in models]
            )
            names = {model.name for model in models}
            whisper_select.value = (
                config.transcription.model if config.transcription.model in names else models[0].name
            )
        else:
            whisper_select.set_options([("No Whisper models installed", "")])
            whisper_select.disabled = True
            warnings.append(
                f"No Whisper models installed — run "
                f"'lectern models whisper --download {config.transcription.model}'."
            )

        notes_select = self.query_one("#notes-model", Select)
        health = await self.app.services.refresh_llm_health()
        if health.available and health.models:
            notes_select.set_options(
                [
                    (f"{model.name}  ({model.size_label})" if model.size_bytes else model.name, model.name)
                    for model in health.models
                ]
            )
            names = {model.name for model in health.models}
            configured = config.ollama.notes_model
            notes_select.value = configured if configured in names else health.models[0].name
        else:
            reason = "Ollama is not running" if not health.available else "no models installed"
            notes_select.set_options([(f"Unavailable — {reason}", "")])
            notes_select.disabled = True
            warnings.append(
                f"Live notes are unavailable ({reason}). Recording and transcription still work."
            )

        if warnings:
            self.query_one("#form-warning", Static).update("\n".join(warnings))

    # -- actions -----------------------------------------------------------
    @on(Button.Pressed, "#start")
    def action_start(self) -> None:
        title = self.query_one("#title", Input).value.strip()
        if not title:
            self.app.notify("Give the session a title first.", severity="warning")
            self.query_one("#title", Input).focus()
            return

        whisper_model = _select_value(self.query_one("#whisper-model", Select))
        if not whisper_model:
            self.app.notify(
                "No Whisper model is installed, so there is nothing to transcribe with.",
                severity="error",
            )
            return

        request = SessionRequest(
            title=title,
            course=self.query_one("#course", Input).value.strip(),
            audio_source=_select_value(self.query_one("#audio-source", Select)) or "microphone",
            input_device=_select_value(self.query_one("#device", Select)),
            whisper_model=whisper_model,
            notes_model=_select_value(self.query_one("#notes-model", Select)),
            save_audio=self.query_one("#save-audio", Switch).value,
        )
        log.info("starting session %r (%s)", request.title, request.audio_source)
        self.app.start_recording(request)

    @on(Button.Pressed, "#cancel")
    def action_cancel(self) -> None:
        self.app.pop_screen()

    @on(Input.Submitted, "#title")
    def _title_submitted(self) -> None:
        self.query_one("#course", Input).focus()

    @on(Input.Submitted, "#course")
    def _course_submitted(self) -> None:
        self.action_start()


def _select_value(select: Select) -> str:
    """Read a Select, treating the blank sentinel as an empty string."""
    value = select.value
    if value is None or value is Select.BLANK:
        return ""
    return str(value)
