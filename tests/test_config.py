"""Configuration loading, saving and mutation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from lectern.config import manager
from lectern.config.models import AudioSourceKind, LecternConfig
from lectern.utils import paths


def test_defaults_are_local_only():
    config = LecternConfig()
    assert config.ollama.host.startswith("http://localhost")
    assert config.transcription.backend == "whisper_cpp"
    assert config.audio.source is AudioSourceKind.MICROPHONE


def test_save_then_load_round_trips():
    config = LecternConfig()
    config.ollama.notes_model = "qwen3:8b"
    config.notes.update_interval_seconds = 22
    config.audio.source = AudioSourceKind.BOTH
    path = manager.save(config)

    assert path.exists()
    reloaded = manager.load(path)
    assert reloaded.ollama.notes_model == "qwen3:8b"
    assert reloaded.notes.update_interval_seconds == 22
    assert reloaded.audio.source is AudioSourceKind.BOTH


def test_load_missing_file_returns_defaults():
    assert not paths.config_file().exists()
    config = manager.load()
    assert config.transcription.model == "small.en"


def test_unknown_keys_are_ignored():
    path = paths.config_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "[ollama]\nnotes_model = 'x'\nfuture_option = 42\n\n[nonexistent]\nfoo = 1\n",
        encoding="utf-8",
    )
    config = manager.load(path)
    assert config.ollama.notes_model == "x"


def test_invalid_toml_raises_config_error():
    path = paths.config_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("this is not = = toml", encoding="utf-8")
    with pytest.raises(manager.ConfigError):
        manager.load(path)


def test_trailing_slash_stripped_from_hosts():
    config = LecternConfig.model_validate({"ollama": {"host": "http://localhost:11434/"}})
    assert config.ollama.host == "http://localhost:11434"


@pytest.mark.parametrize(
    ("key", "value", "expected"),
    [
        ("ollama.notes_model", "llama3.2", "llama3.2"),
        ("notes.update_interval_seconds", "30", 30.0),
        ("audio.save_recording", "false", False),
        ("transcription.threads", "8", 8),
    ],
)
def test_set_value_coerces_types(key, value, expected):
    config = LecternConfig()
    manager.set_value(config, key, value)
    section, field = key.split(".")
    assert getattr(getattr(config, section), field) == expected


def test_set_value_rejects_unknown_key():
    with pytest.raises(KeyError):
        manager.set_value(LecternConfig(), "ollama.nope", "1")


def test_set_value_rejects_out_of_range():
    config = LecternConfig()
    with pytest.raises(ValidationError):
        manager.set_value(config, "notes.update_interval_seconds", "1")


def test_set_value_rejects_an_invalid_boolean():
    """A typo must not silently read as False and disable a feature."""
    config = LecternConfig()
    with pytest.raises(ValueError, match="invalid boolean"):
        manager.set_value(config, "audio.save_recording", "treu")
    assert config.audio.save_recording is True


def test_saving_preserves_hand_written_comments():
    path = paths.config_file()
    manager.save(LecternConfig(), path)
    path.write_text(path.read_text(encoding="utf-8") + "\n# my own note\n", encoding="utf-8")

    config = manager.load(path)
    config.ollama.notes_model = "qwen3:8b"
    manager.save(config, path)

    saved = path.read_text(encoding="utf-8")
    assert "# my own note" in saved
    assert manager.load(path).ollama.notes_model == "qwen3:8b"


def test_save_is_atomic_leaves_no_temp_file():
    manager.save(LecternConfig())
    leftovers = list(paths.config_dir().glob("*.tmp"))
    assert leftovers == []
