"""whisper.cpp model discovery and download.

Models are plain ``ggml-<name>.bin`` files. Lectern looks for them in its own
data directory first, then in the places Homebrew and a source checkout of
whisper.cpp typically leave them, so an existing install is reused rather than
re-downloaded.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import httpx

from lectern.logging_setup import get_logger
from lectern.utils import paths

log = get_logger("transcription.models")

HF_BASE_URL = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main"

#: Approximate on-disk sizes, shown before a download so nothing surprises the user.
KNOWN_MODELS: dict[str, str] = {
    "tiny.en": "75 MB",
    "tiny": "75 MB",
    "base.en": "142 MB",
    "base": "142 MB",
    "small.en": "466 MB",
    "small": "466 MB",
    "medium.en": "1.5 GB",
    "medium": "1.5 GB",
    "large-v3-turbo": "1.6 GB",
    "large-v3": "2.9 GB",
}

RECOMMENDED = "small.en"


@dataclass(slots=True)
class WhisperModel:
    name: str
    path: Path | None
    size_bytes: int | None = None
    approx_size: str = ""

    @property
    def installed(self) -> bool:
        return self.path is not None and self.path.exists()

    @property
    def size_label(self) -> str:
        if self.size_bytes:
            return f"{self.size_bytes / 1_000_000:.0f} MB"
        return self.approx_size or "—"


def search_dirs() -> list[Path]:
    """Directories scanned for ``ggml-*.bin`` weights, most specific first."""
    candidates = [
        paths.whisper_models_dir(),
        Path.home() / ".cache/whisper.cpp",
        Path.home() / "whisper.cpp/models",
        Path("/opt/homebrew/share/whisper-cpp"),
        Path("/opt/homebrew/share/whisper.cpp/models"),
        Path("/usr/local/share/whisper-cpp"),
        Path("/usr/share/whisper.cpp/models"),
    ]
    env_dir = os.environ.get("WHISPER_MODEL_DIR")
    if env_dir:
        candidates.insert(0, Path(env_dir).expanduser())
    return [path for path in candidates if path.is_dir()]


def find_model(name: str) -> Path | None:
    """Resolve a model name (or an explicit path) to a ``ggml-*.bin`` file."""
    if not name:
        return None
    direct = Path(name).expanduser()
    if direct.suffix == ".bin" and direct.exists():
        return direct
    filename = name if name.startswith("ggml-") else f"ggml-{name}.bin"
    for directory in search_dirs():
        candidate = directory / filename
        if candidate.exists():
            return candidate
    return None


def installed_models() -> list[WhisperModel]:
    """Every model Lectern can see on disk, de-duplicated by name."""
    found: dict[str, WhisperModel] = {}
    for directory in search_dirs():
        for path in sorted(directory.glob("ggml-*.bin")):
            name = path.stem.removeprefix("ggml-")
            if name in found:
                continue
            found[name] = WhisperModel(
                name=name,
                path=path,
                size_bytes=path.stat().st_size,
                approx_size=KNOWN_MODELS.get(name, ""),
            )
    return sorted(found.values(), key=lambda model: model.name)


def available_models() -> list[WhisperModel]:
    """Installed models plus the well-known ones that could be downloaded."""
    models = {model.name: model for model in installed_models()}
    for name, approx in KNOWN_MODELS.items():
        models.setdefault(name, WhisperModel(name=name, path=None, approx_size=approx))
    return sorted(models.values(), key=lambda model: (not model.installed, model.name))


def download_model(
    name: str,
    *,
    destination: Path | None = None,
    progress: Callable[[int, int | None], None] | None = None,
) -> Path:
    """Download a ggml model, skipping the transfer if it already exists.

    The file lands via a ``.part`` temp path so an interrupted download can
    never be mistaken for a complete model on the next run.
    """
    existing = find_model(name)
    if existing is not None:
        log.info("model %s already present at %s", name, existing)
        return existing

    if name not in KNOWN_MODELS:
        raise ValueError(f"unknown whisper model {name!r}; known: {', '.join(sorted(KNOWN_MODELS))}")

    target_dir = destination or paths.whisper_models_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"ggml-{name}.bin"
    partial = target.with_suffix(".bin.part")
    url = f"{HF_BASE_URL}/ggml-{name}.bin"

    log.info("downloading whisper model %s from %s", name, url)
    with httpx.stream("GET", url, follow_redirects=True, timeout=60.0) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length") or 0) or None
        downloaded = 0
        with partial.open("wb") as handle:
            for chunk in response.iter_bytes(chunk_size=1 << 20):
                handle.write(chunk)
                downloaded += len(chunk)
                if progress:
                    progress(downloaded, total)
    shutil.move(str(partial), str(target))
    log.info("model %s downloaded to %s", name, target)
    return target


def find_whisper_server(configured: str = "") -> Path | None:
    """Locate the ``whisper-server`` binary from config, PATH, or a checkout."""
    if configured:
        candidate = Path(configured).expanduser()
        return candidate if candidate.exists() else None
    for binary_name in ("whisper-server", "whisper-cpp-server"):
        found = shutil.which(binary_name)
        if found:
            return Path(found)
    fallbacks: Iterable[Path] = (
        Path.home() / "whisper.cpp/build/bin/whisper-server",
        Path("/opt/homebrew/bin/whisper-server"),
        Path("/usr/local/bin/whisper-server"),
    )
    for candidate in fallbacks:
        if candidate.exists():
            return candidate
    return None
