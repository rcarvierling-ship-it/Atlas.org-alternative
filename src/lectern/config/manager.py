"""Load and save ``config.toml``.

Writes are atomic (temp file + ``os.replace``) so a crash mid-save can never
leave the user with a truncated config, and comments in a hand-edited file are
preserved by round-tripping through tomlkit.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

import tomlkit

from lectern.config.models import LecternConfig
from lectern.logging_setup import get_logger
from lectern.utils import paths

log = get_logger("config")

HEADER_COMMENT = """\
# Lectern configuration.
# Everything here runs locally: whisper.cpp for speech, Ollama for notes.
# Docs: https://github.com/rcarvierling-ship-it/Atlas.org-alternative
"""


class ConfigError(RuntimeError):
    """Raised when a config file exists but cannot be parsed."""


def load(path: Path | None = None) -> LecternConfig:
    """Load configuration, falling back to defaults when the file is absent."""
    path = path or paths.config_file()
    if not path.exists():
        return LecternConfig()
    try:
        raw: dict[str, Any] = tomllib.loads(path.read_text(encoding="utf-8"))
    except (tomllib.TOMLDecodeError, OSError) as exc:
        raise ConfigError(f"could not read {path}: {exc}") from exc
    return LecternConfig.model_validate(raw)


def save(config: LecternConfig, path: Path | None = None) -> Path:
    """Persist configuration atomically and return the path written."""
    path = path or paths.config_file()
    path.parent.mkdir(parents=True, exist_ok=True)

    # Round-trip an existing file so hand-written comments survive a save.
    document = None
    if path.exists():
        try:
            document = tomlkit.parse(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 - a corrupt file is rewritten
            log.warning("could not parse %s (%s); rewriting it", path, exc)
    if document is None:
        document = tomlkit.document()
        for line in HEADER_COMMENT.splitlines():
            document.add(tomlkit.comment(line.lstrip("#").strip()))
        document.add(tomlkit.nl())

    for section, values in config.to_toml_dict().items():
        table = document.get(section)
        if table is None:
            table = tomlkit.table()
            document[section] = table
        for key, value in values.items():
            table[key] = value

    tmp = path.with_suffix(".toml.tmp")
    tmp.write_text(tomlkit.dumps(document), encoding="utf-8")
    os.replace(tmp, path)
    return path


def load_or_create(path: Path | None = None) -> LecternConfig:
    """Load config, materialising a default file on first run."""
    path = path or paths.config_file()
    config = load(path)
    if not path.exists():
        save(config, path)
    return config


def exists(path: Path | None = None) -> bool:
    return (path or paths.config_file()).exists()


def set_value(config: LecternConfig, dotted_key: str, value: str) -> LecternConfig:
    """Apply a ``section.key=value`` assignment, coercing via the pydantic schema.

    Used by ``lectern config set``. Validation errors propagate so the CLI can
    show the user exactly which field rejected the value.
    """
    if "." not in dotted_key:
        raise KeyError(f"expected 'section.key', got {dotted_key!r}")
    section_name, key = dotted_key.split(".", 1)
    section = getattr(config, section_name, None)
    if section is None or not hasattr(section, key):
        raise KeyError(f"unknown setting {dotted_key!r}")

    current = getattr(section, key)
    coerced: Any = value
    if isinstance(current, bool):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            coerced = True
        elif normalized in {"0", "false", "no", "off"}:
            coerced = False
        else:
            # Silently reading a typo as False would, for example, turn off
            # recording without saying so.
            raise ValueError(f"invalid boolean for {dotted_key!r}: {value!r}")
    elif isinstance(current, int) and not isinstance(current, bool):
        coerced = int(value)
    elif isinstance(current, float):
        coerced = float(value)

    setattr(section, key, coerced)
    return config
