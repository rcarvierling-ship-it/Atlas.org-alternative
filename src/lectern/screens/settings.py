"""Settings.

Edits the same ``config.toml`` a user could edit by hand. Model pickers are
populated from what is installed rather than free text, so a typo cannot leave
the app configured for a model that does not exist.
"""

from __future__ import annotations

from textual import on, work
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, Footer, Input, Label, Select, Static, Switch

from lectern.config.models import AudioSourceKind
from lectern.logging_setup import get_logger
from lectern.screens.new_session import AUDIO_SOURCE_OPTIONS, _select_value
from lectern.theme import ICONS

log = get_logger("screens.settings")

THEME_OPTIONS = [("Lectern Dark", "lectern-dark"), ("Lectern Light", "lectern-light")]


class SettingsScreen(Screen):
    """Edit configuration."""

    BINDINGS = [
        ("escape", "back", "Back"),
        ("ctrl+s", "save", "Save"),
    ]

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="settings-body"):
            yield Label("SETTINGS", classes="section-title")

            with Vertical(classes="settings-group"):
                yield Label("Transcription", classes="settings-group-title")
                with Horizontal(classes="settings-row"):
                    yield Label("Whisper model", classes="field-label")
                    yield Select([("Loading…", "")], id="whisper-model", allow_blank=False)
                with Horizontal(classes="settings-row"):
                    yield Label("Language", classes="field-label")
                    yield Input(id="language")
                with Horizontal(classes="settings-row"):
                    yield Label("Voice activity detection", classes="field-label")
                    yield Switch(id="vad")
                with Horizontal(classes="settings-row"):
                    yield Label("whisper-server binary", classes="field-label")
                    yield Input(placeholder="auto-detect", id="whisper-binary")

            with Vertical(classes="settings-group"):
                yield Label("Local AI (Ollama)", classes="settings-group-title")
                with Horizontal(classes="settings-row"):
                    yield Label("Ollama URL", classes="field-label")
                    yield Input(id="ollama-host")
                with Horizontal(classes="settings-row"):
                    yield Label("Live notes model", classes="field-label")
                    yield Select([("Loading…", "")], id="notes-model", allow_blank=False)
                with Horizontal(classes="settings-row"):
                    yield Label("Final notes model", classes="field-label")
                    yield Select([("Loading…", "")], id="final-model", allow_blank=False)
                yield Static("", id="ollama-note", classes="hint")

            with Vertical(classes="settings-group"):
                yield Label("Notes", classes="settings-group-title")
                with Horizontal(classes="settings-row"):
                    yield Label("Update every (seconds)", classes="field-label")
                    yield Input(id="update-interval", type="number")
                with Horizontal(classes="settings-row"):
                    yield Label("Consolidate every (seconds)", classes="field-label")
                    yield Input(id="consolidate-interval", type="number")
                with Horizontal(classes="settings-row"):
                    yield Label("Mark exam material", classes="field-label")
                    yield Switch(id="mark-exam")

            with Vertical(classes="settings-group"):
                yield Label("Audio", classes="settings-group-title")
                with Horizontal(classes="settings-row"):
                    yield Label("Default source", classes="field-label")
                    yield Select(AUDIO_SOURCE_OPTIONS, id="audio-source", allow_blank=False)
                with Horizontal(classes="settings-row"):
                    yield Label("Input device", classes="field-label")
                    yield Select([("System default", "")], id="device", allow_blank=False)
                with Horizontal(classes="settings-row"):
                    yield Label("Save recordings", classes="field-label")
                    yield Switch(id="save-audio")
                with Horizontal(classes="settings-row"):
                    yield Label("Delete recordings after (days)", classes="field-label")
                    yield Input(id="retention", type="number")

            with Vertical(classes="settings-group"):
                yield Label("Storage & appearance", classes="settings-group-title")
                with Horizontal(classes="settings-row"):
                    yield Label("Output directory", classes="field-label")
                    yield Input(placeholder="default: ~/.local/share/lectern/sessions", id="output-dir")
                with Horizontal(classes="settings-row"):
                    yield Label("Theme", classes="field-label")
                    yield Select(THEME_OPTIONS, id="theme", allow_blank=False)
                with Horizontal(classes="settings-row"):
                    yield Label("ASCII icons", classes="field-label")
                    yield Switch(id="ascii-icons")

            with Horizontal(id="form-buttons"):
                yield Button("Back", id="cancel")
                yield Button("Save", id="save", classes="-primary")
            yield Static(f"ctrl+s saves {ICONS.dot} esc discards", classes="hint")
        yield Footer()

    def on_mount(self) -> None:
        config = self.app.services.config
        self.query_one("#language", Input).value = config.transcription.language
        self.query_one("#vad", Switch).value = config.transcription.vad
        self.query_one("#whisper-binary", Input).value = config.transcription.whisper_server_binary
        self.query_one("#ollama-host", Input).value = config.ollama.host
        self.query_one("#update-interval", Input).value = str(int(config.notes.update_interval_seconds))
        self.query_one("#consolidate-interval", Input).value = str(
            int(config.notes.consolidate_interval_seconds)
        )
        self.query_one("#mark-exam", Switch).value = config.notes.mark_exam_material
        self.query_one("#audio-source", Select).value = config.audio.source.value
        self.query_one("#save-audio", Switch).value = config.audio.save_recording
        self.query_one("#retention", Input).value = str(config.storage.recording_retention_days)
        self.query_one("#output-dir", Input).value = config.storage.output_dir
        self.query_one("#theme", Select).value = (
            config.ui.theme if config.ui.theme in {value for _, value in THEME_OPTIONS} else "lectern-dark"
        )
        self.query_one("#ascii-icons", Switch).value = config.ui.ascii_icons
        self.populate_choices()

    @work(exclusive=True, group="settings-options")
    async def populate_choices(self) -> None:
        import asyncio

        from lectern.audio.devices import list_input_devices
        from lectern.transcription.models import installed_models

        config = self.app.services.config

        models = await asyncio.to_thread(installed_models)
        whisper_select = self.query_one("#whisper-model", Select)
        if models:
            whisper_select.set_options(
                [(f"{model.name}  ({model.size_label})", model.name) for model in models]
            )
            names = {model.name for model in models}
            whisper_select.value = (
                config.transcription.model if config.transcription.model in names else models[0].name
            )
        else:
            whisper_select.set_options([("No models installed", "")])
            whisper_select.disabled = True

        devices = await asyncio.to_thread(list_input_devices)
        device_select = self.query_one("#device", Select)
        device_select.set_options(
            [("System default", "")] + [(device.label, device.name) for device in devices]
        )
        device_select.value = config.audio.input_device or ""

        health = await self.app.services.refresh_llm_health()
        options = [(model.name, model.name) for model in health.models]
        note = ""
        if not options:
            options = [("Unavailable", "")]
            note = (
                "Ollama is not running or has no models. "
                "Start it with 'ollama serve' and pull a model, e.g. 'ollama pull qwen3:8b'."
            )
        for select_id, configured in (
            ("#notes-model", config.ollama.notes_model),
            ("#final-model", config.ollama.final_model or config.ollama.notes_model),
        ):
            select = self.query_one(select_id, Select)
            select.set_options(options)
            names = {name for _, name in options}
            select.value = configured if configured in names else options[0][1]
            select.disabled = not health.models
        self.query_one("#ollama-note", Static).update(note)

    # -- actions -----------------------------------------------------------
    @on(Button.Pressed, "#save")
    def action_save(self) -> None:
        services = self.app.services
        config = services.config
        try:
            config.transcription.model = (
                _select_value(self.query_one("#whisper-model", Select)) or config.transcription.model
            )
            config.transcription.language = self.query_one("#language", Input).value.strip() or "en"
            config.transcription.vad = self.query_one("#vad", Switch).value
            config.transcription.whisper_server_binary = self.query_one(
                "#whisper-binary", Input
            ).value.strip()

            config.ollama.host = self.query_one("#ollama-host", Input).value.strip() or config.ollama.host
            config.ollama.notes_model = _select_value(self.query_one("#notes-model", Select))
            config.ollama.final_model = _select_value(self.query_one("#final-model", Select))

            config.notes.update_interval_seconds = float(
                self.query_one("#update-interval", Input).value or 15
            )
            config.notes.consolidate_interval_seconds = float(
                self.query_one("#consolidate-interval", Input).value or 180
            )
            config.notes.mark_exam_material = self.query_one("#mark-exam", Switch).value

            config.audio.source = AudioSourceKind(
                _select_value(self.query_one("#audio-source", Select)) or "microphone"
            )
            config.audio.input_device = _select_value(self.query_one("#device", Select))
            config.audio.save_recording = self.query_one("#save-audio", Switch).value
            config.storage.recording_retention_days = int(
                self.query_one("#retention", Input).value or 0
            )
            config.storage.output_dir = self.query_one("#output-dir", Input).value.strip()
            config.ui.theme = _select_value(self.query_one("#theme", Select)) or "lectern-dark"
            config.ui.ascii_icons = self.query_one("#ascii-icons", Switch).value
        except (ValueError, TypeError) as exc:
            self.app.notify(f"Could not save: {exc}", severity="error", timeout=8)
            return

        services.save_config()
        self.app.theme = config.ui.theme
        self.app.notify("Settings saved.", timeout=3)
        log.info("settings saved")
        self.app.pop_screen()

    @on(Button.Pressed, "#cancel")
    def action_back(self) -> None:
        # Discard in-memory edits by reloading from disk.
        self.app.services.reload_config()
        self.app.pop_screen()
