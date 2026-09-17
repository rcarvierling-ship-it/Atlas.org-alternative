"""Microphone + system audio, mixed into a single 16 kHz mono stream.

Two independent capture devices never produce blocks in lock-step, so the mixer
keeps a small per-leg buffer and emits a mixed block whenever *both* legs have
enough samples. If one leg stalls (helper crash, device unplugged), the mixer
falls back to passing the surviving leg through after a short grace period
rather than going silent — a lecture recording with only half the audio still
beats no recording at all.
"""

from __future__ import annotations

import asyncio
import contextlib

import numpy as np

from lectern.audio.base import AudioError, AudioSource
from lectern.logging_setup import get_logger
from lectern.utils.audio_utils import TARGET_SAMPLE_RATE, mix

log = get_logger("audio.combined")

#: How long one leg may lag before we emit the other leg on its own.
STALL_GRACE_SECONDS = 0.75


class CombinedAudioSource(AudioSource):
    """Mix two sources sample-for-sample at a fixed block size."""

    kind = "both"

    def __init__(
        self,
        primary: AudioSource,
        secondary: AudioSource,
        *,
        primary_gain: float = 1.0,
        secondary_gain: float = 1.0,
        block_ms: int = 100,
    ) -> None:
        super().__init__()
        self._primary = primary
        self._secondary = secondary
        self._gains = (primary_gain, secondary_gain)
        self._block_samples = int(TARGET_SAMPLE_RATE * block_ms / 1000)
        self._buffers: list[np.ndarray] = [
            np.zeros(0, dtype=np.float32),
            np.zeros(0, dtype=np.float32),
        ]
        self._last_seen = [0.0, 0.0]
        self._finished = [False, False]
        self._lock = asyncio.Lock()
        self._tasks: list[asyncio.Task[None]] = []

    async def start(self) -> None:
        started: list[AudioSource] = []
        try:
            for source in (self._primary, self._secondary):
                await source.start()
                started.append(source)
        except AudioError:
            for source in started:
                with contextlib.suppress(Exception):
                    await source.stop()
            raise

        now = asyncio.get_running_loop().time()
        self._last_seen = [now, now]
        self._running = True
        self._tasks = [
            asyncio.create_task(self._pump(self._primary, 0), name="mix-primary"),
            asyncio.create_task(self._pump(self._secondary, 1), name="mix-secondary"),
        ]
        log.info("combined capture started (%s + %s)", self._primary.kind, self._secondary.kind)

    async def _pump(self, source: AudioSource, index: int) -> None:
        loop = asyncio.get_running_loop()
        try:
            async for block in source.frames():
                async with self._lock:
                    self._buffers[index] = np.concatenate([self._buffers[index], block])
                    self._last_seen[index] = loop.time()
                    self._drain_locked(loop)
        except asyncio.CancelledError:  # pragma: no cover - shutdown
            raise
        except Exception as exc:  # noqa: BLE001
            log.error("mixer leg %s failed: %s", source.kind, exc)
        finally:
            async with self._lock:
                self._finished[index] = True
                self._drain_locked(loop)
            if all(self._finished):
                self._close_stream()

    def _drain_locked(self, loop: asyncio.AbstractEventLoop) -> None:
        """Emit every complete mixed block available. Caller holds the lock."""
        while True:
            have = [buf.size >= self._block_samples for buf in self._buffers]
            now = loop.time()
            stalled = [
                self._finished[i] or (now - self._last_seen[i]) > STALL_GRACE_SECONDS
                for i in range(2)
            ]

            if all(have):
                ready = [0, 1]
            elif have[0] and stalled[1]:
                ready = [0]
            elif have[1] and stalled[0]:
                ready = [1]
            else:
                return

            tracks = []
            gains = []
            for index in ready:
                tracks.append(self._buffers[index][: self._block_samples])
                self._buffers[index] = self._buffers[index][self._block_samples :]
                gains.append(self._gains[index])
            self._publish(mix(*tracks, gains=tuple(gains)))

    async def stop(self) -> None:
        self._running = False
        for source in (self._primary, self._secondary):
            with contextlib.suppress(Exception):
                await source.stop()
        for task in self._tasks:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        self._tasks = []
        self._close_stream()
        log.info("combined capture stopped")

    @property
    def level(self) -> float:
        return max(self._primary.level, self._secondary.level)
