"""Audio capture: microphone, macOS system audio, both mixed, or a WAV file."""

from __future__ import annotations

from pathlib import Path

from lectern.audio.base import (
    AudioDevice,
    AudioError,
    AudioSource,
    PermissionDeniedError,
)
from lectern.audio.combined import CombinedAudioSource
from lectern.audio.devices import (
    default_input_device,
    list_input_devices,
    resolve_device,
    sounddevice_available,
)
from lectern.audio.file_source import FileAudioSource
from lectern.audio.microphone import MicrophoneSource
from lectern.audio.system import SystemAudioSource, helper_binary
from lectern.audio.vad import Utterance, VADConfig, VoiceSegmenter
from lectern.config.models import AudioConfig, AudioSourceKind

__all__ = [
    "AudioConfig",
    "AudioDevice",
    "AudioError",
    "AudioSource",
    "AudioSourceKind",
    "CombinedAudioSource",
    "FileAudioSource",
    "MicrophoneSource",
    "PermissionDeniedError",
    "SystemAudioSource",
    "Utterance",
    "VADConfig",
    "VoiceSegmenter",
    "build_source",
    "default_input_device",
    "helper_binary",
    "list_input_devices",
    "resolve_device",
    "sounddevice_available",
]


def build_source(
    config: AudioConfig,
    *,
    kind: AudioSourceKind | None = None,
    file_path: Path | str | None = None,
    speed: float = 1.0,
) -> AudioSource:
    """Construct the audio source for a session from configuration.

    This is the single place that maps the user's choice onto a concrete
    implementation; screens and the CLI never instantiate sources directly.
    """
    kind = kind or config.source
    if file_path is not None:
        return FileAudioSource(file_path, speed=speed)

    if kind is AudioSourceKind.MICROPHONE:
        return MicrophoneSource(device_name=config.input_device, gain=config.mic_gain)
    if kind is AudioSourceKind.SYSTEM:
        return SystemAudioSource(helper_path=config.native_helper_path, gain=config.system_gain)
    if kind is AudioSourceKind.BOTH:
        return CombinedAudioSource(
            MicrophoneSource(device_name=config.input_device),
            SystemAudioSource(helper_path=config.native_helper_path),
            primary_gain=config.mic_gain,
            secondary_gain=config.system_gain,
        )
    if kind is AudioSourceKind.FILE:
        raise AudioError("audio source 'file' requires --file PATH")
    raise AudioError(f"unknown audio source: {kind}")
