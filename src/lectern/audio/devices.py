"""Input device discovery.

``sounddevice`` (PortAudio) is imported lazily and defensively: it needs a
native library that is absent on plenty of machines, and a missing microphone
must degrade to a clear message rather than an ImportError traceback at startup.
"""

from __future__ import annotations

from functools import lru_cache

from lectern.audio.base import AudioDevice
from lectern.logging_setup import get_logger

log = get_logger("audio.devices")


class SoundDeviceUnavailable(RuntimeError):
    """PortAudio bindings are missing or the native library failed to load."""


@lru_cache(maxsize=1)
def _import_sounddevice():  # pragma: no cover - environment dependent
    try:
        import sounddevice
    except (ImportError, OSError) as exc:
        raise SoundDeviceUnavailable(
            "sounddevice/PortAudio is not available. Install it with "
            "'uv sync --extra audio' (and 'brew install portaudio' if needed)."
        ) from exc
    return sounddevice


def sounddevice_available() -> bool:
    try:
        _import_sounddevice()
    except SoundDeviceUnavailable:
        return False
    return True


def list_input_devices() -> list[AudioDevice]:
    """Enumerate input-capable devices. Returns ``[]`` when PortAudio is absent."""
    try:
        sd = _import_sounddevice()
    except SoundDeviceUnavailable as exc:
        log.warning("device enumeration unavailable: %s", exc)
        return []

    try:
        raw_devices = sd.query_devices()
        default_input = sd.default.device[0] if sd.default.device else None
    except Exception as exc:  # pragma: no cover - PortAudio runtime failure
        log.warning("PortAudio device query failed: %s", exc)
        return []

    devices: list[AudioDevice] = []
    for index, info in enumerate(raw_devices):
        if int(info.get("max_input_channels", 0)) <= 0:
            continue
        devices.append(
            AudioDevice(
                index=index,
                name=str(info.get("name", f"device {index}")),
                channels=int(info["max_input_channels"]),
                default_sample_rate=float(info.get("default_samplerate", 48_000.0)),
                is_default=(index == default_input),
            )
        )
    return devices


def default_input_device() -> AudioDevice | None:
    for device in list_input_devices():
        if device.is_default:
            return device
    devices = list_input_devices()
    return devices[0] if devices else None


def resolve_device(name_fragment: str) -> AudioDevice | None:
    """Find a device by case-insensitive substring, or the default if unset."""
    if not name_fragment:
        return default_input_device()
    fragment = name_fragment.lower()
    for device in list_input_devices():
        if fragment in device.name.lower():
            return device
    return None
