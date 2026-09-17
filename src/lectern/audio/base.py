"""Audio source interface.

Every source produces the same thing regardless of where the sound came from:
blocks of 16 kHz mono float32 samples, delivered through a bounded async queue.

Capture happens on a real-time-ish callback (PortAudio thread) or a subprocess
reader task; both hand blocks to the queue without ever blocking on the
consumer. If the consumer falls behind, the *oldest* audio is dropped and
counted — losing a little old audio is always better than stalling the capture
callback, which would glitch the recording.
"""

from __future__ import annotations

import abc
import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass

import numpy as np

from lectern.logging_setup import get_logger
from lectern.utils.audio_utils import TARGET_SAMPLE_RATE, rms_level

log = get_logger("audio")


class AudioError(RuntimeError):
    """Capture could not start or died mid-session."""


class PermissionDeniedError(AudioError):
    """macOS refused microphone or screen-recording access.

    Carries the human-readable remediation shown by the permissions modal
    rather than a raw OSStatus code.
    """

    def __init__(self, message: str, *, permission: str, remediation: str) -> None:
        super().__init__(message)
        self.permission = permission
        self.remediation = remediation


@dataclass(slots=True)
class AudioDevice:
    """An input device as reported by the platform."""

    index: int
    name: str
    channels: int
    default_sample_rate: float
    is_default: bool = False

    @property
    def label(self) -> str:
        return f"{self.name}{' (default)' if self.is_default else ''}"


class AudioSource(abc.ABC):
    """Base class for microphone / system / combined / file capture."""

    kind: str = "base"
    sample_rate: int = TARGET_SAMPLE_RATE

    def __init__(self, *, queue_blocks: int = 200) -> None:
        self._queue: asyncio.Queue[np.ndarray | None] = asyncio.Queue(maxsize=queue_blocks)
        self._running = False
        self._level = 0.0
        self.dropped_blocks = 0

    # -- lifecycle ---------------------------------------------------------
    @abc.abstractmethod
    async def start(self) -> None:
        """Begin capture. Raises ``AudioError`` / ``PermissionDeniedError``."""

    @abc.abstractmethod
    async def stop(self) -> None:
        """Stop capture and unblock any pending ``frames()`` iteration."""

    @property
    def running(self) -> bool:
        return self._running

    @property
    def level(self) -> float:
        """Most recent RMS level, for the audio meter in the status bar."""
        return self._level

    # -- data flow ---------------------------------------------------------
    def _publish(self, block: np.ndarray) -> None:
        """Push a captured block. Safe to call from a non-async thread context."""
        self._level = rms_level(block)
        try:
            self._queue.put_nowait(block)
        except asyncio.QueueFull:
            # Drop the oldest block so the newest audio still gets through.
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except asyncio.QueueEmpty:  # pragma: no cover - race, harmless
                pass
            self.dropped_blocks += 1
            if self.dropped_blocks % 50 == 1:
                log.warning("%s source dropped %d blocks (consumer behind)", self.kind, self.dropped_blocks)
            try:
                self._queue.put_nowait(block)
            except asyncio.QueueFull:  # pragma: no cover
                pass

    def _close_stream(self) -> None:
        """Send the sentinel that terminates ``frames()``.

        The sentinel must land even on a full queue. Dropping it would leave
        every ``frames()`` consumer — the pipeline's audio task, or a mixer leg —
        waiting forever on a stream that has already stopped, so old blocks are
        evicted until there is room.
        """
        while True:
            try:
                self._queue.put_nowait(None)
                return
            except asyncio.QueueFull:
                try:
                    self._queue.get_nowait()
                except asyncio.QueueEmpty:  # pragma: no cover - racing consumer
                    continue

    async def frames(self) -> AsyncIterator[np.ndarray]:
        """Yield captured blocks until the source stops."""
        while True:
            block = await self._queue.get()
            if block is None:
                return
            yield block
