"""Microphone capture through PortAudio / CoreAudio.

PortAudio delivers blocks on its own high-priority thread. That callback must
never block, so it does the minimum possible work — convert to mono, resample
if the device would not give us 16 kHz — and hands the block to the asyncio
queue via ``loop.call_soon_threadsafe``.
"""

from __future__ import annotations

import asyncio

import numpy as np

from lectern.audio.base import AudioDevice, AudioError, AudioSource, PermissionDeniedError
from lectern.audio.devices import SoundDeviceUnavailable, _import_sounddevice, resolve_device
from lectern.logging_setup import get_logger
from lectern.utils.audio_utils import TARGET_SAMPLE_RATE, resample, to_mono

log = get_logger("audio.microphone")

MIC_PERMISSION_REMEDIATION = (
    "Open System Settings → Privacy & Security → Microphone and enable access "
    "for your terminal app (Terminal, iTerm2 or Warp), then restart it."
)


class MicrophoneSource(AudioSource):
    """Capture from a Mac input device at 16 kHz mono."""

    kind = "microphone"

    def __init__(
        self,
        *,
        device: AudioDevice | None = None,
        device_name: str = "",
        block_ms: int = 100,
        gain: float = 1.0,
    ) -> None:
        super().__init__()
        self._device = device or resolve_device(device_name)
        # Falling back to the default here would record a different microphone
        # than the one the user selected, without saying so.
        self._missing_device = device is None and bool(device_name) and self._device is None
        self._block_ms = block_ms
        self._gain = gain
        self._stream = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._capture_rate = TARGET_SAMPLE_RATE

    @property
    def device(self) -> AudioDevice | None:
        return self._device

    async def start(self) -> None:
        if self._missing_device:
            raise AudioError(
                f"the selected microphone is not available. "
                f"Choose another input device in Settings."
            )
        try:
            sd = _import_sounddevice()
        except SoundDeviceUnavailable as exc:
            raise AudioError(str(exc)) from exc

        self._loop = asyncio.get_running_loop()
        device_index = self._device.index if self._device else None

        # Ask CoreAudio for 16 kHz directly; it resamples in the HAL, which is
        # cheaper and higher quality than doing it ourselves. Fall back to the
        # device's native rate if the driver refuses.
        self._capture_rate = TARGET_SAMPLE_RATE
        try:
            self._stream = sd.InputStream(
                device=device_index,
                channels=1,
                samplerate=TARGET_SAMPLE_RATE,
                dtype="float32",
                blocksize=int(TARGET_SAMPLE_RATE * self._block_ms / 1000),
                callback=self._callback,
            )
            self._stream.start()
        except Exception as exc:  # noqa: BLE001 - PortAudio raises many types
            if self._is_permission_error(exc):
                raise PermissionDeniedError(
                    "Lectern does not have microphone access.",
                    permission="Microphone",
                    remediation=MIC_PERMISSION_REMEDIATION,
                ) from exc
            log.info("16 kHz capture rejected (%s); retrying at the device rate", exc)
            try:
                native_rate = int(self._device.default_sample_rate) if self._device else 48_000
                self._capture_rate = native_rate
                self._stream = sd.InputStream(
                    device=device_index,
                    channels=1,
                    samplerate=native_rate,
                    dtype="float32",
                    blocksize=int(native_rate * self._block_ms / 1000),
                    callback=self._callback,
                )
                self._stream.start()
            except Exception as retry_exc:  # noqa: BLE001
                if self._is_permission_error(retry_exc):
                    raise PermissionDeniedError(
                        "Lectern does not have microphone access.",
                        permission="Microphone",
                        remediation=MIC_PERMISSION_REMEDIATION,
                    ) from retry_exc
                raise AudioError(f"could not open microphone: {retry_exc}") from retry_exc

        self._running = True
        log.info(
            "microphone capture started (device=%s, rate=%d)",
            self._device.name if self._device else "default",
            self._capture_rate,
        )

    @staticmethod
    def _is_permission_error(exc: Exception) -> bool:
        text = str(exc).lower()
        return any(
            marker in text
            for marker in ("permission", "not authorized", "unauthorized", "-10851", "access denied")
        )

    def _callback(self, indata, frames, time_info, status) -> None:  # noqa: ANN001, ARG002
        """PortAudio callback — runs on the audio thread, must stay fast."""
        if status:  # overflow/underflow flags
            log.debug("PortAudio status: %s", status)
        block = to_mono(np.array(indata, dtype=np.float32, copy=True))
        if self._capture_rate != TARGET_SAMPLE_RATE:
            block = resample(block, self._capture_rate, TARGET_SAMPLE_RATE)
        if self._gain != 1.0:
            block = np.clip(block * self._gain, -1.0, 1.0)
        loop = self._loop
        if loop is not None and not loop.is_closed():
            loop.call_soon_threadsafe(self._publish, block)

    async def stop(self) -> None:
        self._running = False
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception as exc:  # noqa: BLE001 pragma: no cover
                log.warning("error closing microphone stream: %s", exc)
        self._close_stream()
        log.info("microphone capture stopped")
