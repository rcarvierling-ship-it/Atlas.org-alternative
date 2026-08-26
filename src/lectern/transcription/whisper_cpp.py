"""whisper.cpp transcription backend.

Lectern drives whisper.cpp through its bundled HTTP server (``whisper-server``)
rather than the one-shot ``whisper-cli`` binary. That choice is the whole
latency story: the server loads the ggml weights **once**, keeps them resident
(and Metal-warm on Apple Silicon), and answers each utterance in the time it
takes to decode a few seconds of audio. Spawning ``whisper-cli`` per chunk
would re-read and re-upload hundreds of megabytes of weights every few seconds.

The server is a child process owned by this backend: started in ``start()``,
health-checked before the session is allowed to begin, and torn down in
``stop()``. Users who already run their own whisper.cpp server can point
``transcription.server_url`` at it, in which case Lectern attaches instead of
spawning and never touches the process lifecycle.
"""

from __future__ import annotations

import asyncio
import contextlib
import socket
import time
from collections import deque
from pathlib import Path

import httpx
import numpy as np

from lectern.logging_setup import get_logger
from lectern.transcription.base import (
    TranscriptionBackend,
    TranscriptionError,
    TranscriptionHealth,
)
from lectern.transcription.models import find_model, find_whisper_server
from lectern.utils.audio_utils import TARGET_SAMPLE_RATE, wav_bytes
from lectern.utils.text import looks_like_hallucination

log = get_logger("transcription.whisper")

STARTUP_TIMEOUT_SECONDS = 120.0
MODEL_MISSING_HINT = (
    "Download it with 'lectern models whisper --download {model}' "
    "or point transcription.model at an existing ggml-*.bin file."
)
SERVER_MISSING_HINT = (
    "whisper.cpp's 'whisper-server' binary was not found.\n"
    "Install it with 'brew install whisper-cpp', or build whisper.cpp and set "
    "transcription.whisper_server_binary to the built binary."
)


def _free_port() -> int:
    """Reserve an ephemeral port for the child server."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def build_server_command(
    binary: Path,
    model_path: Path,
    *,
    port: int,
    language: str = "en",
    threads: int = 0,
) -> list[str]:
    """Build the whisper-server command for the installed CLI contract.

    Boolean flags in whisper.cpp are switches, not key/value options. In
    particular, ``--print-progress`` must never be followed by ``false``;
    progress is already disabled by default, so Lectern simply omits it.
    """
    command = [
        str(binary),
        "--model",
        str(model_path),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--language",
        language or "en",
        # Each utterance is decoded independently: carrying decoder state
        # across chunks is whisper.cpp's main source of runaway repetition.
        "--no-context",
    ]
    if threads > 0:
        command += ["--threads", str(threads)]
    return command


class WhisperCppBackend(TranscriptionBackend):
    """Persistent whisper.cpp server driving utterance-level transcription."""

    name = "whisper_cpp"

    def __init__(
        self,
        *,
        model: str = "small.en",
        language: str = "en",
        threads: int = 0,
        server_binary: str = "",
        server_url: str = "",
        startup_timeout: float = STARTUP_TIMEOUT_SECONDS,
    ) -> None:
        self.model = model
        self.language = language
        self.threads = threads
        self._server_binary = server_binary
        self._external_url = server_url.rstrip("/")
        self._startup_timeout = startup_timeout

        self._process: asyncio.subprocess.Process | None = None
        self._client: httpx.AsyncClient | None = None
        self._url = self._external_url
        self._ready = False
        self._detail = "not started"
        self._last_latency_ms: float | None = None
        self._model_path: Path | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._stderr_lines: deque[str] = deque(maxlen=50)
        self._last_command: list[str] = []
        # whisper.cpp's server handles one decode at a time; serialise so a
        # partial decode never races an utterance decode.
        self._lock = asyncio.Lock()

    # -- lifecycle ---------------------------------------------------------
    async def start(self) -> None:
        if self._external_url:
            self._url = self._external_url
            self._client = httpx.AsyncClient(base_url=self._url, timeout=120.0)
            if not await self._probe():
                await self.stop()
                raise TranscriptionError(
                    f"no whisper.cpp server responded at {self._url}. "
                    "Clear transcription.server_url to let Lectern manage its own server."
                )
            self._ready = True
            self._detail = f"attached to {self._url}"
            log.info("attached to external whisper server at %s", self._url)
            return

        binary = find_whisper_server(self._server_binary)
        if binary is None:
            raise TranscriptionError(SERVER_MISSING_HINT)

        self._model_path = find_model(self.model)
        if self._model_path is None:
            raise TranscriptionError(
                f"whisper model {self.model!r} is not installed.\n"
                + MODEL_MISSING_HINT.format(model=self.model)
            )

        port = _free_port()
        self._url = f"http://127.0.0.1:{port}"
        command = build_server_command(
            binary,
            self._model_path,
            port=port,
            language=self.language,
            threads=self.threads,
        )
        self._last_command = command.copy()
        self._stderr_lines.clear()

        log.info("starting whisper server: %s", " ".join(command))
        try:
            self._process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError as exc:
            raise TranscriptionError(f"could not launch whisper-server: {exc}") from exc

        self._stderr_task = asyncio.create_task(self._drain_stderr(), name="whisper-stderr")
        self._client = httpx.AsyncClient(base_url=self._url, timeout=120.0)

        if not await self._wait_until_ready():
            code = self._process.returncode if self._process else None
            # If the process already exited, give the stderr reader a brief
            # chance to consume the final usage/error lines before reporting.
            if code is not None and self._stderr_task is not None:
                with contextlib.suppress(asyncio.TimeoutError, Exception):
                    await asyncio.wait_for(asyncio.shield(self._stderr_task), timeout=0.5)

            stderr_tail = "\n".join(self._stderr_lines).strip()
            command_text = " ".join(self._last_command)
            await self.stop()

            message = "whisper-server did not become ready"
            if code is not None:
                message += f" (exited with code {code})"
            else:
                message += " within the timeout"
            if stderr_tail:
                message += f".\n\nLast whisper-server output:\n{stderr_tail}"
            message += f"\n\nCommand:\n{command_text}"
            raise TranscriptionError(message)

        self._ready = True
        self._detail = f"{self.model} on {self._url}"
        log.info("whisper server ready (model=%s)", self.model)

    async def _drain_stderr(self) -> None:
        """Retain and log whisper.cpp stderr without polluting the TUI."""
        if self._process is None or self._process.stderr is None:
            return
        while True:
            line = await self._process.stderr.readline()
            if not line:
                return
            text = line.decode("utf-8", "replace").rstrip()
            self._stderr_lines.append(text)
            log.debug("whisper-server: %s", text)

    async def _wait_until_ready(self) -> bool:
        """Poll until the server answers, or its process dies."""
        deadline = time.monotonic() + self._startup_timeout
        while time.monotonic() < deadline:
            if self._process is not None and self._process.returncode is not None:
                return False
            if await self._probe():
                return True
            await asyncio.sleep(0.25)
        return False

    async def _probe(self) -> bool:
        if self._client is None:
            return False
        try:
            response = await self._client.get("/", timeout=2.0)
        except httpx.HTTPError:
            return False
        # Any HTTP answer means the listener is up; whisper.cpp's index route
        # returns 200 with its demo page, older builds return 404.
        return response.status_code < 500

    async def stop(self) -> None:
        self._ready = False
        self._detail = "stopped"
        if self._client is not None:
            with contextlib.suppress(Exception):
                await self._client.aclose()
            self._client = None

        process, self._process = self._process, None
        if process is not None and process.returncode is None:
            process.terminate()
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(process.wait(), timeout=5.0)
            if process.returncode is None:  # pragma: no cover
                process.kill()
                with contextlib.suppress(Exception):
                    await process.wait()

        if self._stderr_task is not None:
            self._stderr_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._stderr_task
            self._stderr_task = None
        log.info("whisper backend stopped")

    # -- inference ---------------------------------------------------------
    async def transcribe(self, audio: np.ndarray, *, sample_rate: int = TARGET_SAMPLE_RATE) -> str:
        """Transcribe a single utterance. Returns ``\"\"`` for non-speech."""
        if self._client is None or not self._ready:
            raise TranscriptionError("whisper backend is not running")
        audio = np.asarray(audio, dtype=np.float32)
        if audio.size < sample_rate * 0.15:
            return ""

        payload = wav_bytes(audio, sample_rate)
        started = time.monotonic()
        async with self._lock:
            try:
                response = await self._client.post(
                    "/inference",
                    files={"file": ("utterance.wav", payload, "audio/wav")},
                    data={
                        "temperature": "0.0",
                        "temperature_inc": "0.2",
                        "response_format": "json",
                        "language": self.language or "en",
                        "no_context": "true",
                    },
                )
            except httpx.HTTPError as exc:
                self._ready = False
                self._detail = f"connection lost: {exc}"
                raise TranscriptionError(f"whisper.cpp request failed: {exc}") from exc

        if response.status_code >= 400:
            raise TranscriptionError(
                f"whisper.cpp returned HTTP {response.status_code}: {response.text[:200]}"
            )

        self._last_latency_ms = (time.monotonic() - started) * 1000.0
        return self._clean(parse_whisper_response(response.text))

    @staticmethod
    def _clean(text: str) -> str:
        """Drop whisper's silence hallucinations before they reach the transcript."""
        text = text.strip()
        if not text or looks_like_hallucination(text):
            if text:
                log.debug("discarded likely hallucination: %r", text)
            return ""
        return text

    def health(self) -> TranscriptionHealth:
        return TranscriptionHealth(
            ready=self._ready,
            detail=self._detail,
            model=self.model,
            last_latency_ms=self._last_latency_ms,
        )

    @property
    def url(self) -> str:
        return self._url


def parse_whisper_response(body: str) -> str:
    """Extract transcript text from a whisper.cpp server response.

    Handles the shapes different whisper.cpp builds return: ``{\"text\": ...}``,
    verbose JSON with a ``segments`` array, and plain text.
    """
    import json

    body = body.strip()
    if not body:
        return ""
    if not body.startswith(("{", "[")):
        return body

    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return body

    if isinstance(payload, dict):
        text = payload.get("text")
        if isinstance(text, str) and text.strip():
            return text.strip()
        segments = payload.get("segments") or payload.get("transcription")
        if isinstance(segments, list):
            parts = [
                str(segment.get("text", "")).strip()
                for segment in segments
                if isinstance(segment, dict)
            ]
            return " ".join(part for part in parts if part).strip()
        if "error" in payload:
            raise TranscriptionError(str(payload["error"]))
    return ""
