"""WAV file playback as an audio source — the demo / development mode.

``lectern record --file lecture.wav`` runs the *entire* production pipeline
(VAD, whisper.cpp, note scheduler, persistence, TUI) against a recording
instead of a microphone. Blocks are emitted in real time by default so timing
behaviour matches a live lecture; ``speed`` fast-forwards that for tests.
"""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path

import numpy as np

from lectern.audio.base import AudioError, AudioSource
from lectern.logging_setup import get_logger
from lectern.utils.audio_utils import TARGET_SAMPLE_RATE, read_wav, resample

log = get_logger("audio.file")


class FileAudioSource(AudioSource):
    """Replay a WAV file as though it were arriving from a microphone."""

    kind = "file"

    def __init__(self, path: Path | str, *, block_ms: int = 100, speed: float = 1.0) -> None:
        super().__init__()
        self.path = Path(path).expanduser()
        self._block_ms = block_ms
        self._speed = max(0.0, speed)
        self._task: asyncio.Task[None] | None = None
        self._audio: np.ndarray = np.zeros(0, dtype=np.float32)

    @property
    def duration(self) -> float:
        return self._audio.size / TARGET_SAMPLE_RATE

    async def start(self) -> None:
        if not self.path.exists():
            raise AudioError(f"audio file not found: {self.path}")
        try:
            samples, rate = await asyncio.to_thread(read_wav, self.path)
        except Exception as exc:  # noqa: BLE001
            raise AudioError(f"could not read {self.path.name}: {exc}") from exc

        self._audio = resample(samples, rate, TARGET_SAMPLE_RATE) if rate != TARGET_SAMPLE_RATE else samples
        self._running = True
        self._task = asyncio.create_task(self._replay(), name="file-audio-replay")
        log.info("file source started: %s (%.1fs, %.1fx)", self.path.name, self.duration, self._speed)

    async def _replay(self) -> None:
        block_samples = int(TARGET_SAMPLE_RATE * self._block_ms / 1000)
        block_seconds = self._block_ms / 1000.0
        try:
            for offset in range(0, self._audio.size, block_samples):
                if not self._running:
                    break
                self._publish(self._audio[offset : offset + block_samples].copy())
                if self._speed > 0:
                    await asyncio.sleep(block_seconds / self._speed)
                else:
                    await asyncio.sleep(0)  # yield without pacing
        except asyncio.CancelledError:  # pragma: no cover - shutdown
            raise
        finally:
            self._running = False
            self._close_stream()
            log.info("file source finished: %s", self.path.name)

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._task
            self._task = None
        self._close_stream()
