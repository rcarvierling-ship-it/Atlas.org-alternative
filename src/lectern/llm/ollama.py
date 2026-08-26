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
        client = self._get_client()
        try:
            response = await client.get("/api/version", timeout=3.0)
            response.raise_for_status()
            version = str(response.json().get("version", ""))
        except httpx.HTTPError as exc:
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
        client = self._get_client()
        try:
            response = await client.get("/api/tags", timeout=10.0)
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPError as exc:
            raise LLMUnavailableError(OLLAMA_MISSING_HINT) from exc
        except ValueError as exc:
            raise LLMError(f"Ollama returned an unreadable model list: {exc}") from exc

        models: list[LLMModel] = []
        for entry in payload.get("models", []):
            details = entry.get("details") or {}
            models.append(
                LLMModel(
                    name=str(entry.get("name", "")),
                    size_bytes=entry.get("size"),
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
