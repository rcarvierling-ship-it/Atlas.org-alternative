"""Environment diagnostics shared by ``lectern doctor`` and the setup wizard.

Each check answers three questions: is it OK, what did we find, and — when it
isn't OK — exactly what should the user run to fix it. Checks never raise; a
diagnostic tool that crashes on a broken environment is useless.
"""

from __future__ import annotations

import asyncio
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from enum import StrEnum

from lectern.config.models import LecternConfig
from lectern.logging_setup import get_logger
from lectern.utils import paths

log = get_logger("doctor")


class CheckStatus(StrEnum):
    OK = "ok"
    WARN = "warn"
    FAIL = "fail"
    UNKNOWN = "unknown"


@dataclass(slots=True)
class Check:
    name: str
    status: CheckStatus
    detail: str = ""
    remedy: str = ""
    #: A check the app cannot run without (as opposed to a nice-to-have).
    required: bool = True

    @property
    def icon(self) -> str:
        return {
            CheckStatus.OK: "✓",
            CheckStatus.WARN: "!",
            CheckStatus.FAIL: "✗",
            CheckStatus.UNKNOWN: "?",
        }[self.status]


@dataclass
class DoctorReport:
    checks: list[Check] = field(default_factory=list)

    @property
    def failures(self) -> list[Check]:
        return [check for check in self.checks if check.status is CheckStatus.FAIL and check.required]

    @property
    def warnings(self) -> list[Check]:
        return [
            check
            for check in self.checks
            if check.status is CheckStatus.WARN
            or (check.status is CheckStatus.FAIL and not check.required)
        ]

    @property
    def healthy(self) -> bool:
        return not self.failures

    @property
    def can_record(self) -> bool:
        """True when transcription is possible: a whisper server and a model."""
        needed = {"whisper.cpp", "Whisper model"}
        return all(
            check.status is CheckStatus.OK for check in self.checks if check.name in needed
        )

    def get(self, name: str) -> Check | None:
        for check in self.checks:
            if check.name == name:
                return check
        return None

    def summary(self) -> str:
        if self.failures:
            return f"{len(self.failures)} problem(s) need attention before Lectern can record."
        if self.warnings:
            return "Lectern is ready. Some optional features are unavailable."
        return "Everything looks good."


def _is_macos() -> bool:
    return sys.platform == "darwin"


def check_python() -> Check:
    version = platform.python_version()
    ok = sys.version_info >= (3, 12)
    return Check(
        name="Python",
        status=CheckStatus.OK if ok else CheckStatus.FAIL,
        detail=version,
        remedy="" if ok else "Lectern needs Python 3.12 or newer.",
    )


def check_platform() -> Check:
    system = platform.system()
    if _is_macos():
        return Check(name="macOS", status=CheckStatus.OK, detail=platform.mac_ver()[0] or "unknown")
    return Check(
        name="macOS",
        status=CheckStatus.WARN,
        detail=f"running on {system}",
        remedy=(
            "Lectern targets macOS. The TUI, whisper.cpp and Ollama work here, but "
            "system-audio capture needs macOS 13+ with ScreenCaptureKit."
        ),
        required=False,
    )


def check_architecture() -> Check:
    machine = platform.machine()
    if machine == "arm64":
        return Check(name="Apple Silicon", status=CheckStatus.OK, detail="arm64")
    return Check(
        name="Apple Silicon",
        status=CheckStatus.WARN,
        detail=machine,
        remedy="Whisper runs without Metal acceleration on this architecture and will be slower.",
        required=False,
    )


def check_whisper_binary(config: LecternConfig) -> Check:
    from lectern.transcription.models import find_whisper_server

    binary = find_whisper_server(config.transcription.whisper_server_binary)
    if binary is not None:
        return Check(name="whisper.cpp", status=CheckStatus.OK, detail=str(binary))
    if config.transcription.server_url:
        return Check(
            name="whisper.cpp",
            status=CheckStatus.OK,
            detail=f"using external server {config.transcription.server_url}",
        )
    return Check(
        name="whisper.cpp",
        status=CheckStatus.FAIL,
        detail="whisper-server not found",
        remedy="brew install whisper-cpp",
    )


def check_whisper_model(config: LecternConfig) -> Check:
    from lectern.transcription.models import find_model, installed_models

    model = config.transcription.model
    path = find_model(model)
    if path is not None:
        size = path.stat().st_size / 1_000_000
        return Check(name="Whisper model", status=CheckStatus.OK, detail=f"{model} ({size:.0f} MB)")

    installed = installed_models()
    if installed:
        names = ", ".join(entry.name for entry in installed[:4])
        return Check(
            name="Whisper model",
            status=CheckStatus.FAIL,
            detail=f"{model} is not installed (found: {names})",
            remedy=f"lectern models whisper --download {model}",
        )
    return Check(
        name="Whisper model",
        status=CheckStatus.FAIL,
        detail="no models installed",
        remedy=f"lectern models whisper --download {model}",
    )


def check_metal() -> Check:
    """Report Metal availability, which is what makes whisper.cpp fast on Macs."""
    if not _is_macos():
        return Check(
            name="Metal", status=CheckStatus.WARN, detail="not applicable", required=False
        )
    if platform.machine() == "arm64":
        return Check(name="Metal", status=CheckStatus.OK, detail="available")
    return Check(
        name="Metal",
        status=CheckStatus.WARN,
        detail="Intel Mac — CPU inference",
        required=False,
    )


async def check_ollama(config: LecternConfig) -> tuple[Check, Check]:
    """Check the daemon and the configured note model together."""
    from lectern.llm.ollama import OllamaBackend

    backend = OllamaBackend(config.ollama.host)
    try:
        health = await backend.health()
    finally:
        await backend.close()

    if not health.available:
        daemon = Check(
            name="Ollama",
            status=CheckStatus.FAIL,
            detail=f"not responding at {config.ollama.host}",
            remedy="ollama serve",
        )
        model = Check(
            name="Ollama model",
            status=CheckStatus.UNKNOWN,
            detail="cannot check while Ollama is down",
        )
        return daemon, model

    daemon = Check(
        name="Ollama",
        status=CheckStatus.OK,
        detail=f"running{f' (v{health.version})' if health.version else ''} · {health.detail}",
    )

    installed = [entry.name for entry in health.models]
    configured = config.ollama.notes_model
    if not installed:
        model = Check(
            name="Ollama model",
            status=CheckStatus.FAIL,
            detail="no models installed",
            remedy="ollama pull qwen3:8b",
        )
    elif not configured:
        model = Check(
            name="Ollama model",
            status=CheckStatus.WARN,
            detail=f"none selected (installed: {', '.join(installed[:4])})",
            remedy="Choose one in Settings, or set ollama.notes_model in your config.",
            required=False,
        )
    elif configured in installed or any(name.startswith(f"{configured}:") for name in installed):
        model = Check(name="Ollama model", status=CheckStatus.OK, detail=configured)
    else:
        model = Check(
            name="Ollama model",
            status=CheckStatus.FAIL,
            detail=f"{configured} is not installed (have: {', '.join(installed[:4])})",
            remedy=f"ollama pull {configured}",
        )
    return daemon, model


def check_microphone() -> Check:
    from lectern.audio.devices import list_input_devices, sounddevice_available

    if not sounddevice_available():
        return Check(
            name="Microphone",
            status=CheckStatus.FAIL,
            detail="PortAudio bindings unavailable",
            remedy="uv sync --extra audio   (and 'brew install portaudio' if that fails)",
        )
    devices = list_input_devices()
    if not devices:
        return Check(
            name="Microphone",
            status=CheckStatus.FAIL,
            detail="no input devices found",
            remedy=(
                "System Settings → Privacy & Security → Microphone, enable your terminal, "
                "then restart it."
            ),
        )
    default = next((device for device in devices if device.is_default), devices[0])
    return Check(
        name="Microphone",
        status=CheckStatus.OK,
        detail=f"{default.name} ({len(devices)} input device(s))",
    )


def check_screen_recording(config: LecternConfig) -> Check:
    """Check the native helper for system-audio capture (optional feature)."""
    from lectern.audio.system import helper_binary

    if not _is_macos():
        return Check(
            name="System audio",
            status=CheckStatus.WARN,
            detail="macOS only",
            required=False,
        )
    binary = helper_binary(config.audio.native_helper_path)
    if binary is None:
        return Check(
            name="System audio",
            status=CheckStatus.WARN,
            detail="native helper not built",
            remedy="scripts/build-native.sh   (needs Xcode command line tools)",
            required=False,
        )
    return Check(
        name="System audio",
        status=CheckStatus.OK,
        detail=f"helper at {binary}",
        required=False,
    )


def check_storage() -> Check:
    """Report free space where sessions are written."""
    target = paths.data_dir()
    probe = target if target.exists() else target.parent
    try:
        usage = shutil.disk_usage(probe)
    except OSError as exc:
        return Check(name="Storage", status=CheckStatus.UNKNOWN, detail=str(exc), required=False)

    free_gb = usage.free / 1_000_000_000
    if free_gb < 1:
        status, remedy = CheckStatus.FAIL, "Free up disk space before recording."
    elif free_gb < 5:
        status, remedy = CheckStatus.WARN, "Recordings may fill the remaining space."
    else:
        status, remedy = CheckStatus.OK, ""
    return Check(
        name="Storage",
        status=status,
        detail=f"{free_gb:.0f} GB free at {target}",
        remedy=remedy,
        required=status is CheckStatus.FAIL,
    )


def check_ffmpeg() -> Check:
    """ffmpeg is optional — Lectern reads and writes WAV itself."""
    binary = shutil.which("ffmpeg")
    return Check(
        name="ffmpeg",
        status=CheckStatus.OK if binary else CheckStatus.WARN,
        detail=binary or "not installed (optional)",
        remedy="" if binary else "brew install ffmpeg   (only needed to import non-WAV audio)",
        required=False,
    )


def check_build_tools() -> Check:
    """Xcode command line tools, needed only to build the Swift helper."""
    if not _is_macos():
        return Check(name="Build tools", status=CheckStatus.WARN, detail="macOS only", required=False)
    try:
        result = subprocess.run(
            ["xcode-select", "-p"], capture_output=True, text=True, timeout=5, check=False
        )
    except (OSError, subprocess.SubprocessError):
        result = None
    if result is not None and result.returncode == 0 and result.stdout.strip():
        return Check(
            name="Build tools",
            status=CheckStatus.OK,
            detail=result.stdout.strip(),
            required=False,
        )
    return Check(
        name="Build tools",
        status=CheckStatus.WARN,
        detail="Xcode command line tools not found",
        remedy="xcode-select --install   (only needed for system-audio capture)",
        required=False,
    )


def check_config() -> Check:
    from lectern.config import manager

    path = paths.config_file()
    if not path.exists():
        return Check(
            name="Config",
            status=CheckStatus.WARN,
            detail="not created yet (defaults in use)",
            remedy="It will be written on first run, or run 'lectern config init'.",
            required=False,
        )
    try:
        manager.load(path)
    except Exception as exc:  # noqa: BLE001
        return Check(
            name="Config",
            status=CheckStatus.FAIL,
            detail=str(exc),
            remedy=f"Fix or delete {path}",
        )
    return Check(name="Config", status=CheckStatus.OK, detail=str(path), required=False)


async def run_all(config: LecternConfig | None = None) -> DoctorReport:
    """Run every check. Blocking probes run in threads to keep the TUI responsive."""
    config = config or LecternConfig()
    report = DoctorReport()

    report.checks.append(check_python())
    report.checks.append(check_platform())
    report.checks.append(check_architecture())
    report.checks.append(await asyncio.to_thread(check_whisper_binary, config))
    report.checks.append(await asyncio.to_thread(check_whisper_model, config))
    report.checks.append(check_metal())

    daemon, model = await check_ollama(config)
    report.checks.append(daemon)
    report.checks.append(model)

    report.checks.append(await asyncio.to_thread(check_microphone))
    report.checks.append(check_screen_recording(config))
    report.checks.append(await asyncio.to_thread(check_ffmpeg))
    report.checks.append(await asyncio.to_thread(check_build_tools))
    report.checks.append(check_config())
    report.checks.append(await asyncio.to_thread(check_storage))
    return report


def first_run_needed(config: LecternConfig | None = None) -> bool:
    """True when the setup wizard should be shown instead of the Home screen."""
    if os.environ.get("LECTERN_SKIP_WIZARD"):
        return False
    return not paths.config_file().exists()
