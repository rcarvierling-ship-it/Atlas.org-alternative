"""Filesystem locations.

Lectern follows the XDG layout on every platform (macOS included) so that the
config file lives somewhere a developer can edit by hand:

    ~/.config/lectern/config.toml       configuration
    ~/.local/share/lectern/sessions/    session folders + index.sqlite3
    ~/.local/state/lectern/lectern.log  application log

Every path can be redirected with ``LECTERN_HOME``, which is what the test
suite uses to keep runs hermetic.
"""

from __future__ import annotations

import os
from pathlib import Path


def _root_override() -> Path | None:
    raw = os.environ.get("LECTERN_HOME")
    return Path(raw).expanduser() if raw else None


def config_dir() -> Path:
    override = _root_override()
    if override:
        return override / "config"
    return Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "lectern"


def data_dir() -> Path:
    override = _root_override()
    if override:
        return override / "data"
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share")) / "lectern"


def state_dir() -> Path:
    override = _root_override()
    if override:
        return override / "state"
    return Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local/state")) / "lectern"


def cache_dir() -> Path:
    override = _root_override()
    if override:
        return override / "cache"
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "lectern"


def config_file() -> Path:
    return config_dir() / "config.toml"


def sessions_dir() -> Path:
    return data_dir() / "sessions"


def index_db() -> Path:
    return data_dir() / "index.sqlite3"


def log_file() -> Path:
    return state_dir() / "lectern.log"


def whisper_models_dir() -> Path:
    """Where Lectern keeps whisper.cpp ``ggml-*.bin`` weights it downloaded."""
    return data_dir() / "whisper-models"


def native_helper_path() -> Path:
    """Bundled Swift ScreenCaptureKit helper, once built by ``scripts/build-native.sh``."""
    return Path(__file__).resolve().parents[3] / "native" / "audio-capture" / "build" / "lectern-audio-capture"


def ensure_dirs() -> None:
    """Create every directory Lectern writes to."""
    for path in (config_dir(), data_dir(), state_dir(), sessions_dir(), whisper_models_dir()):
        path.mkdir(parents=True, exist_ok=True)
