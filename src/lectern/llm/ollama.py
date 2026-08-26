"""Ollama HTTP backend.

Talks to a local Ollama daemon (``http://localhost:11434`` by default) over its
native ``/api/generate`` endpoint with streaming enabled, so the UI can show
note generation happening rather than freezing on a long request.

Two details matter for note quality:

* ``format`` is set to a JSON schema when the caller wants structured output.
  Ollama constrains sampling to that grammar, which is what makes small local
  models reliable enough to drive a typed ``NoteState``.
* ``keep_alive`` holds the weights in memory between updates. Without it Ollama
  unloads the model after a few idle seconds and every note update pays a
  multi-second reload.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Any

import httpx

from lectern.llm.base import LLMBackend, LLMError, LLMHealth, LLMModel, LLMUnavailableError
from lectern.logging_setup import get_logger

log = get_logger("llm.ollama")

OLLAMA_MISSING_HINT = (
    "Ollama is not responding. Start it with 'ollama serve' (or launch the Ollama app), "
    "then pull a model with e.g. 'ollama pull qwen3:8b'."
)


class OllamaBackend(LLMBackend):
    """Local Ollama daemon."""

    name = "ollama"

    def __init__(
        self,
        host: str = "http://localhost:11434",
        *,
        timeout: float = 180.0,
        keep_alive: str = "10m",
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.host = host.rstrip("/")
        self.keep_alive = keep_alive
        self._timeout = timeout
        self._client = client
        self._owns_client = client is None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(base_url=self.host, timeout=self._timeout)
        return self._client

    # -- introspection -----------------------------------------------------
    async def health(self) -> LLMHealth:
        try:
            # Client construction itself raises InvalidURL for a malformed host,
            # and InvalidURL is not an HTTPError — so it has to be built in here.
            client = self._get_client()
            response = await client.get("/api/version", timeout=3.0)
            response.raise_for_status()
            version = str(response.json().get("version", ""))
        except (httpx.HTTPError, httpx.InvalidURL) as exc:
            log.info("ollama health check failed: %s", exc)
            return LLMHealth(available=False, detail=OLLAMA_MISSING_HINT)
        except (ValueError, KeyError):
            version = ""

        try:
            models = await self.list_models()
        except LLMError:
            models = []
        detail = f"{len(models)} model{'s' if len(models) != 1 else ''} installed"
        if not models:
            detail = "running, but no models installed — try 'ollama pull qwen3:8b'"
        return LLMHealth(available=True, detail=detail, version=version, models=models)

    async def list_models(self) -> list[LLMModel]:
        try:
            client = self._get_client()
            response = await client.get("/api/tags", timeout=10.0)
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, httpx.InvalidURL) as exc:
            raise LLMUnavailableError(OLLAMA_MISSING_HINT) from exc
        except ValueError as exc:
            raise LLMError(f"Ollama returned an unreadable model list: {exc}") from exc

        if not isinstance(payload, dict):
            raise LLMError("Ollama returned an unreadable model list: expected a JSON object")

        # "models" can be absent, null, or the wrong type. A bare .get(key, [])
        # only covers the first of those, and the other two reach the loop as a
        # TypeError, which is not an LLMError and so escapes health().
        entries = payload.get("models") or []
        if not isinstance(entries, list):
            raise LLMError("Ollama returned an unreadable model list: 'models' is not a list")

        models: list[LLMModel] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            details = entry.get("details") if isinstance(entry.get("details"), dict) else {}
            # size feeds arithmetic in LLMModel.size_label, so a string would
            # only fail later, at render time.
            size = entry.get("size")
            models.append(
                LLMModel(
                    name=str(entry.get("name", "")),
                    size_bytes=size if isinstance(size, int) and not isinstance(size, bool) else None,
                    family=str(details.get("family", "")),
                    parameter_size=str(details.get("parameter_size", "")),
                    quantization=str(details.get("quantization_level", "")),
                )
            )
        return sorted((model for model in models if model.name), key=lambda model: model.name)

    # -- generation --------------------------------------------------------
    async def generate(
        self,
        prompt: str,
        *,
        model: str,
        system: str | None = None,
        json_schema: dict[str, Any] | None = None,
        temperature: float = 0.2,
        num_ctx: int | None = None,
        on_token: Callable[[str], None] | None = None,
    ) -> str:
        if not model:
            raise LLMError("no Ollama model selected (set ollama.notes_model in your config)")

        options: dict[str, Any] = {"temperature": temperature}
        if num_ctx:
            options["num_ctx"] = num_ctx

        body: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": options,
            "keep_alive": self.keep_alive,
        }
        if system:
            body["system"] = system
        if json_schema is not None:
            body["format"] = json_schema

        client = self._get_client()
        chunks: list[str] = []
        try:
            async with client.stream("POST", "/api/generate", json=body) as response:
                if response.status_code >= 400:
                    detail = (await response.aread()).decode("utf-8", "replace")[:300]
                    message = f"Ollama returned HTTP {response.status_code}: {detail}"
                    # 5xx means the daemon is sick or restarting: recoverable, so
                    # callers should pause notes and retry rather than give up.
                    # 4xx is a request problem (missing model, bad options) that
                    # retrying will not fix.
                    if response.status_code >= 500:
                        raise LLMUnavailableError(message)
                    raise LLMError(message)
                async for line in response.aiter_lines():
                    piece = _parse_stream_line(line)
                    if piece is None:
                        continue
                    chunks.append(piece)
                    if on_token and piece:
                        on_token(piece)
        except httpx.HTTPError as exc:
            raise LLMUnavailableError(f"Ollama request failed: {exc}") from exc

        return "".join(chunks).strip()

    async def close(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
        self._client = None


def is_local_host(host: str) -> bool:
    """True when ``host`` points at this machine.

    Auto-starting a daemon only ever makes sense locally: if the user pointed
    Lectern at another machine, that machine's daemon is not ours to launch.
    """
    from urllib.parse import urlparse

    hostname = (urlparse(host).hostname or "").lower()
    return hostname in {"localhost", "127.0.0.1", "::1", "0.0.0.0"}


async def ensure_ollama_running(
    host: str = "http://localhost:11434", *, timeout: float = 25.0
) -> bool:
    """Start a local Ollama daemon if one is not already answering.

    This is what lets ``lectern`` be the only command a user types: whisper.cpp
    is already spawned per session by the transcription backend, and this does
    the same for the note model's daemon.

    Returns True if Ollama is running by the time this returns. Never raises —
    a machine without Ollama installed simply keeps note generation disabled,
    and the UI explains that.
    """
    import shutil
    import time

    backend = OllamaBackend(host)
    try:
        if (await backend.health()).available:
            return True

        if not is_local_host(host):
            log.info("not starting Ollama: %s is not a local host", host)
            return False

        binary = shutil.which("ollama")
        if binary is None:
            log.info("not starting Ollama: the 'ollama' binary is not installed")
            return False

        log.info("starting Ollama via %s", binary)
        try:
            process = await asyncio.create_subprocess_exec(
                binary,
                "serve",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
                stdin=asyncio.subprocess.DEVNULL,
                # Detach from Lectern's process group so quitting the TUI does
                # not take the daemon down with it — the user may well have
                # other things using Ollama.
                start_new_session=True,
            )
        except OSError as exc:
            log.warning("could not launch Ollama: %s", exc)
            return False

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if process.returncode is not None:
                # A daemon that exits immediately usually means one is already
                # bound to the port; re-probe before giving up.
                if (await backend.health()).available:
                    return True
                log.warning("Ollama exited immediately with code %s", process.returncode)
                return False
            await asyncio.sleep(0.4)
            if (await backend.health()).available:
                log.info("Ollama is up")
                return True

        log.warning("Ollama did not become ready within %.0fs", timeout)
        return False
    finally:
        await backend.close()


def _parse_stream_line(line: str) -> str | None:
    """Pull the text fragment out of one NDJSON line of an Ollama stream."""
    line = line.strip()
    if not line:
        return None
    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        log.debug("ignoring non-JSON stream line: %r", line[:120])
        return None
    if isinstance(payload, dict):
        if payload.get("error"):
            raise LLMError(str(payload["error"]))
        response = payload.get("response")
        if isinstance(response, str):
            return response
    return None
