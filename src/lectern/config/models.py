"""Typed configuration schema.

Everything the user can tune lives here as a pydantic model, which gives us
validation, defaults and TOML round-tripping for free. Unknown keys in an older
or newer config file are ignored rather than fatal, so upgrading Lectern never
strands a user with a config the app refuses to load.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator


class AudioSourceKind(StrEnum):
    MICROPHONE = "microphone"
    SYSTEM = "system"
    BOTH = "both"
    FILE = "file"


class _Base(BaseModel):
    model_config = ConfigDict(extra="ignore", validate_assignment=True)


class TranscriptionConfig(_Base):
    backend: str = "whisper_cpp"
    model: str = "small.en"
    language: str = "en"
    vad: bool = True
    #: Path to the whisper.cpp ``whisper-server`` binary. Empty means "search PATH".
    whisper_server_binary: str = ""
    #: Attach to an already-running whisper.cpp server instead of spawning one.
    server_url: str = ""
    #: Threads handed to whisper.cpp. 0 lets whisper.cpp decide.
    threads: int = 0
    #: Emit unstable partial hypotheses for the utterance currently being spoken.
    partials: bool = True
    partial_interval_seconds: float = Field(default=1.2, ge=0.3, le=5.0)

    @field_validator("server_url")
    @classmethod
    def _strip_slash(cls, value: str) -> str:
        return value.rstrip("/")


class OllamaConfig(_Base):
    host: str = "http://localhost:11434"
    notes_model: str = ""
    final_model: str = ""
    request_timeout_seconds: float = Field(default=180.0, gt=0)
    #: Context window requested from Ollama for note updates.
    num_ctx: int = Field(default=8192, ge=2048)
    keep_alive: str = "10m"

    @field_validator("host")
    @classmethod
    def _strip_slash(cls, value: str) -> str:
        return value.rstrip("/")


class NotesConfig(_Base):
    update_interval_seconds: float = Field(default=15.0, ge=5.0, le=120.0)
    consolidate_interval_seconds: float = Field(default=180.0, ge=30.0)
    #: Don't bother the model until this much new speech has accumulated.
    min_new_words: int = Field(default=25, ge=5)
    #: Hard ceiling on transcript words sent in a single update prompt.
    max_context_words: int = Field(default=900, ge=100)
    mark_exam_material: bool = True


class AudioConfig(_Base):
    source: AudioSourceKind = AudioSourceKind.MICROPHONE
    save_recording: bool = True
    #: Substring match against device names; empty means system default input.
    input_device: str = ""
    #: Gain applied to the microphone leg when mixing mic + system audio.
    mic_gain: float = Field(default=1.0, ge=0.0, le=4.0)
    system_gain: float = Field(default=1.0, ge=0.0, le=4.0)
    native_helper_path: str = ""


class StorageConfig(_Base):
    output_dir: str = ""
    #: Delete recorded WAVs older than N days. 0 disables retention pruning.
    recording_retention_days: int = Field(default=0, ge=0)

    def resolved_output_dir(self) -> Path | None:
        return Path(self.output_dir).expanduser() if self.output_dir else None


class UIConfig(_Base):
    theme: str = "lectern-dark"
    #: Fall back to ASCII glyphs for terminals without good Unicode coverage.
    ascii_icons: bool = False
    show_partials: bool = True


class LecternConfig(_Base):
    """Root configuration object, serialized to ``~/.config/lectern/config.toml``."""

    transcription: TranscriptionConfig = Field(default_factory=TranscriptionConfig)
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    notes: NotesConfig = Field(default_factory=NotesConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    ui: UIConfig = Field(default_factory=UIConfig)

    def to_toml_dict(self) -> dict:
        """Plain dict with enums flattened, ready for ``tomlkit.dumps``."""
        return self.model_dump(mode="json")
