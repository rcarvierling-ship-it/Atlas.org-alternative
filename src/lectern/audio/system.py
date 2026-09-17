"""System audio capture via the bundled Swift ScreenCaptureKit helper.

macOS has no public API for tapping system output from Python, and Lectern
refuses to make the user install BlackHole or another virtual audio driver. So
a tiny Swift helper (``native/audio-capture``) uses ScreenCaptureKit's
``SCStream`` audio capture — available since macOS 13 — and streams the result
to Lectern over a pipe.

Wire protocol (deliberately trivial so it is easy to fake in tests):

* **stdout** — raw little-endian float32 mono PCM at 16 kHz, nothing else.
* **stderr** — one JSON object per line: ``{"event": "...", "message": "..."}``.
* **exit 13** — screen-recording permission was denied.

The helper is fully isolated behind this class: nothing else in Lectern knows
that system audio involves a subprocess at all.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import shutil
from pathlib import Path

import numpy as np

from lectern.audio.base import AudioError, AudioSource, PermissionDeniedError
from lectern.logging_setup import get_logger
from lectern.utils import paths
from lectern.utils.audio_utils import TARGET_SAMPLE_RATE

log = get_logger("audio.system")

PERMISSION_EXIT_CODE = 13
BYTES_PER_SAMPLE = 4

SCREEN_PERMISSION_REMEDIATION = (
    "Open System Settings → Privacy & Security → Screen & System Audio Recording "
    "and enable access for your terminal app, then restart it. macOS only applies "
    "this permission to newly launched processes."
)

HELPER_MISSING_MESSAGE = (
    "The native system-audio helper has not been built yet.\n"
    "Run scripts/build-native.sh (requires Xcode command line tools and macOS 13+)."
)


def helper_binary(configured_path: str = "") -> Path | None:
    """Locate the helper: explicit config, then repo build dir, then PATH."""
    if configured_path:
        candidate = Path(configured_path).expanduser()
        return candidate if candidate.exists() else None
    bundled = paths.native_helper_path()
    if bundled.exists():
        return bundled
    found = shutil.which("lectern-audio-capture")
    return Path(found) if found else None


class SystemAudioSource(AudioSource):
    """Capture everything the Mac is playing, at 16 kHz mono."""

    kind = "system"

    def __init__(self, *, helper_path: str = "", block_ms: int = 100, gain: float = 1.0) -> None:
        super().__init__()
        self._helper_path = helper_binary(helper_path)
        self._block_ms = block_ms
        self._gain = gain
        self._process: asyncio.subprocess.Process | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._last_error: str = ""
        #: Bytes of an incomplete float32 sample carried to the next read. A
        #: pipe read can end mid-sample, and discarding the tail would shift
        #: every following sample by 1-3 bytes — permanent misalignment, and
        #: noise rather than speech from there on.
        self._pcm_remainder = b""

    async def start(self) -> None:
        if self._helper_path is None:
            raise AudioError(HELPER_MISSING_MESSAGE)

        try:
            self._process = await asyncio.create_subprocess_exec(
                str(self._helper_path),
                "--sample-rate",
                str(TARGET_SAMPLE_RATE),
                "--format",
                "f32",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError as exc:
            raise AudioError(f"could not launch system audio helper: {exc}") from exc

        self._running = True
        self._reader_task = asyncio.create_task(self._read_audio(), name="system-audio-reader")
        self._stderr_task = asyncio.create_task(self._read_stderr(), name="system-audio-stderr")

        # Give the helper a moment to fail fast on a permission denial so the
        # caller sees a permissions modal instead of a silent, empty stream.
        await asyncio.sleep(0.4)
        if self._process.returncode is not None:
            await self._raise_for_exit(self._process.returncode)
        log.info("system audio capture started via %s", self._helper_path)

    async def _raise_for_exit(self, code: int) -> None:
        self._running = False
        if code == PERMISSION_EXIT_CODE:
            raise PermissionDeniedError(
                "Lectern does not have Screen & System Audio Recording access.",
                permission="Screen & System Audio Recording",
                remediation=SCREEN_PERMISSION_REMEDIATION,
            )
        detail = self._last_error or f"helper exited with code {code}"
        raise AudioError(f"system audio capture failed: {detail}")

    async def _read_audio(self) -> None:
        assert self._process is not None and self._process.stdout is not None
        block_bytes = int(TARGET_SAMPLE_RATE * self._block_ms / 1000) * BYTES_PER_SAMPLE
        stream = self._process.stdout
        try:
            while self._running:
                chunk = await stream.read(block_bytes)
                if not chunk:
                    break
                data = self._pcm_remainder + chunk
                # Keep any partial sample at the tail for the next read rather
                # than discarding it, which would misalign the whole stream.
                usable = len(data) - (len(data) % BYTES_PER_SAMPLE)
                self._pcm_remainder = data[usable:]
                block = np.frombuffer(data[:usable], dtype="<f4").astype(np.float32)
                if self._gain != 1.0:
                    block = np.clip(block * self._gain, -1.0, 1.0)
                if block.size:
                    self._publish(block)
        except asyncio.CancelledError:  # pragma: no cover - shutdown path
            raise
        except Exception as exc:  # noqa: BLE001
            log.error("system audio reader failed: %s", exc)
        finally:
            self._close_stream()

    async def _read_stderr(self) -> None:
        assert self._process is not None and self._process.stderr is not None
        stream = self._process.stderr
        while True:
            line = await stream.readline()
            if not line:
                return
            text = line.decode("utf-8", "replace").strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
                event = payload.get("event", "log")
                message = payload.get("message", "")
            except json.JSONDecodeError:
                event, message = "log", text
            if event == "error":
                self._last_error = message
                log.error("system audio helper: %s", message)
            else:
                log.debug("system audio helper [%s]: %s", event, message)

    async def stop(self) -> None:
        self._running = False
        process, self._process = self._process, None
        if process is not None and process.returncode is None:
            process.terminate()
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(process.wait(), timeout=3.0)
            if process.returncode is None:  # pragma: no cover - stubborn child
                process.kill()
        for task in (self._reader_task, self._stderr_task):
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
        self._reader_task = self._stderr_task = None
        self._close_stream()
        log.info("system audio capture stopped")
