"""Shared application services.

One object holding the long-lived collaborators (config, session index, LLM
client) that screens need. Keeping it separate from ``LecternApp`` means
screens import services, not the app class, which keeps the import graph acyclic
and lets tests construct a service bundle without starting a TUI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from lectern.config import manager as config_manager
from lectern.config.models import LecternConfig
from lectern.llm.base import LLMBackend, LLMHealth
from lectern.llm.ollama import OllamaBackend
from lectern.logging_setup import get_logger
from lectern.sessions.manager import SessionManager
from lectern.theme import configure_icons

log = get_logger("services")


@dataclass(slots=True)
class SessionRequest:
    """Everything needed to start a recording.

    Produced by the New Session screen or by ``lectern record`` flags, and
    consumed by the recording screen — so both entry points converge on one
    description of what to record and with what.
    """

    title: str
    course: str = ""
    audio_source: str = "microphone"
    input_device: str = ""
    whisper_model: str = ""
    notes_model: str = ""
    save_audio: bool = True
    #: Development mode: feed a WAV file through the real pipeline.
    file_path: Path | None = None
    file_speed: float = 1.0
    #: Set when continuing an interrupted session rather than starting fresh.
    resume_session_id: str = ""

    @property
    def is_file_mode(self) -> bool:
        return self.file_path is not None


@dataclass
class AppServices:
    """Everything a screen might need, constructed once at startup."""

    config: LecternConfig = field(default_factory=LecternConfig)
    _manager: SessionManager | None = field(default=None, repr=False)
    _llm: LLMBackend | None = field(default=None, repr=False)
    _llm_health: LLMHealth | None = field(default=None, repr=False)

    @classmethod
    def create(cls, *, config_path: Path | None = None) -> AppServices:
        config = config_manager.load(config_path)
        configure_icons(config.ui.ascii_icons)
        return cls(config=config)

    @property
    def sessions(self) -> SessionManager:
        """Session manager, opened lazily so the CLI can skip the database."""
        if self._manager is None:
            self._manager = SessionManager(self.config)
        return self._manager

    @property
    def llm(self) -> LLMBackend:
        if self._llm is None:
            self._llm = OllamaBackend(
                self.config.ollama.host,
                timeout=self.config.ollama.request_timeout_seconds,
                keep_alive=self.config.ollama.keep_alive,
            )
        return self._llm

    async def refresh_llm_health(self) -> LLMHealth:
        """Re-probe Ollama and cache the result for the Home screen."""
        self._llm_health = await self.llm.health()
        return self._llm_health

    @property
    def llm_health(self) -> LLMHealth | None:
        return self._llm_health

    def reload_config(self) -> LecternConfig:
        self.config = config_manager.load()
        configure_icons(self.config.ui.ascii_icons)
        if self._manager is not None:
            self._manager.config = self.config
        # The host may have changed, so drop the cached client.
        self._llm = None
        self._llm_health = None
        return self.config

    def save_config(self) -> None:
        config_manager.save(self.config)
        configure_icons(self.config.ui.ascii_icons)

    async def aclose(self) -> None:
        if self._llm is not None:
            try:
                await self._llm.close()
            except Exception as exc:  # noqa: BLE001
                log.debug("error closing LLM client: %s", exc)
            self._llm = None
        if self._manager is not None:
            self._manager.close()
            self._manager = None
