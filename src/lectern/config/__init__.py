"""Configuration loading and the typed config schema."""

from lectern.config.manager import ConfigError, load, load_or_create, save, set_value
from lectern.config.models import (
    AudioConfig,
    AudioSourceKind,
    LecternConfig,
    NotesConfig,
    OllamaConfig,
    StorageConfig,
    TranscriptionConfig,
    UIConfig,
)

__all__ = [
    "AudioConfig",
    "AudioSourceKind",
    "ConfigError",
    "LecternConfig",
    "NotesConfig",
    "OllamaConfig",
    "StorageConfig",
    "TranscriptionConfig",
    "UIConfig",
    "load",
    "load_or_create",
    "save",
    "set_value",
]
