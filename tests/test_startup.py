"""Starting Lectern with a single command.

Typing ``lectern`` has to be enough: whisper.cpp is spawned per session by the
transcription backend, and the note model's daemon is brought up by the
services layer. These tests cover the daemon side and its refusals.
"""

from __future__ import annotations

import pytest

from lectern.config.models import LecternConfig
from lectern.llm.ollama import ensure_ollama_running, is_local_host
from lectern.services import AppServices


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("http://localhost:11434", True),
        ("http://127.0.0.1:11434", True),
        ("http://[::1]:11434", True),
        ("http://studio.local:11434", False),
        ("https://ollama.example.com", False),
        ("http://192.168.1.50:11434", False),
    ],
)
def test_local_host_detection(host, expected):
    assert is_local_host(host) is expected


async def test_already_running_daemon_is_left_alone(fake_ollama):
    assert await ensure_ollama_running(fake_ollama.url) is True


async def test_remote_host_is_never_started(monkeypatch):
    """Another machine's daemon is not ours to launch."""
    called = False

    def fake_which(name):  # noqa: ANN001, ARG001
        nonlocal called
        called = True
        return "/usr/local/bin/ollama"

    monkeypatch.setattr("shutil.which", fake_which)
    assert await ensure_ollama_running("http://studio.local:11434", timeout=1.0) is False
    assert not called, "a remote host should not even look for a local binary"


async def test_missing_binary_degrades_quietly(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: None)  # noqa: ARG005
    assert await ensure_ollama_running("http://localhost:11499", timeout=1.0) is False


async def test_services_autostart_recovers_a_down_daemon(fake_ollama, monkeypatch):
    """A daemon that comes up mid-probe is picked up by the health refresh."""
    fake_ollama.available = False

    config = LecternConfig()
    config.ollama.host = fake_ollama.url
    services = AppServices(config=config)

    async def fake_start(host, **kwargs):  # noqa: ANN001, ARG001
        fake_ollama.available = True
        return True

    monkeypatch.setattr("lectern.llm.ollama.ensure_ollama_running", fake_start)

    health = await services.refresh_llm_health()
    assert health.available
    await services.aclose()


async def test_services_autostart_is_attempted_only_once(fake_ollama, monkeypatch):
    fake_ollama.available = False
    config = LecternConfig()
    config.ollama.host = fake_ollama.url
    services = AppServices(config=config)

    attempts = 0

    async def fake_start(host, **kwargs):  # noqa: ANN001, ARG001
        nonlocal attempts
        attempts += 1
        return False

    monkeypatch.setattr("lectern.llm.ollama.ensure_ollama_running", fake_start)

    for _ in range(3):
        assert not (await services.refresh_llm_health()).available
    assert attempts == 1
    await services.aclose()


async def test_autostart_can_be_disabled(fake_ollama, monkeypatch):
    fake_ollama.available = False
    config = LecternConfig()
    config.ollama.host = fake_ollama.url
    config.ollama.autostart = False
    services = AppServices(config=config)

    async def fail(host, **kwargs):  # noqa: ANN001, ARG001
        raise AssertionError("autostart is disabled and must not run")

    monkeypatch.setattr("lectern.llm.ollama.ensure_ollama_running", fail)

    assert not (await services.refresh_llm_health()).available
    await services.aclose()
