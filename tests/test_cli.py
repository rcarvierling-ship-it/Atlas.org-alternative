"""CLI commands, exercised through Typer's runner."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from lectern.cli import app as cli_app
from lectern.transcription.base import TranscriptSegment

runner = CliRunner()


@pytest.fixture
def stored_session(manager):
    meta, store = manager.create(title="Cell Structure", course="BIO 113")
    store.append_segment(
        TranscriptSegment(
            id=1, start_time=0.0, end_time=5.0, text="Mitochondria are the powerhouse of the cell."
        )
    )
    store.save_final_notes("# Cell Structure\n\n## Executive Summary\n\nAbout cells.")
    store.close()
    manager.reindex()
    manager.close()
    return meta


def test_version():
    result = runner.invoke(cli_app, ["--version"])
    assert result.exit_code == 0
    assert "lectern" in result.stdout


def test_sessions_lists_nothing_gracefully():
    result = runner.invoke(cli_app, ["sessions"])
    assert result.exit_code == 0
    assert "No sessions yet" in result.stdout


def test_sessions_lists_a_session(stored_session):
    result = runner.invoke(cli_app, ["sessions"])
    assert result.exit_code == 0
    assert "Cell Structure" in result.stdout
    assert "BIO 113" in result.stdout


def test_search_finds_content(stored_session):
    result = runner.invoke(cli_app, ["search", "mitochondria"])
    assert result.exit_code == 0
    assert "Cell Structure" in result.stdout


def test_search_reports_no_match(stored_session):
    result = runner.invoke(cli_app, ["search", "quantum chromodynamics"])
    assert result.exit_code == 0
    assert "Nothing matched" in result.stdout


def test_export_markdown_writes_a_real_file(stored_session, tmp_path):
    destination = tmp_path / "out"
    destination.mkdir()
    result = runner.invoke(
        cli_app, ["export", stored_session.id, "--format", "markdown", "--out", str(destination)]
    )
    assert result.exit_code == 0

    written = list(destination.glob("*.md"))
    assert len(written) == 1
    content = written[0].read_text(encoding="utf-8")
    assert "Cell Structure" in content
    assert "powerhouse of the cell" in content


def test_export_json(stored_session, tmp_path):
    destination = tmp_path / "session.json"
    result = runner.invoke(
        cli_app, ["export", stored_session.id, "--format", "json", "--out", str(destination)]
    )
    assert result.exit_code == 0
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["session"]["course"] == "BIO 113"


def test_export_rejects_unknown_format(stored_session):
    result = runner.invoke(cli_app, ["export", stored_session.id, "--format", "pdf"])
    assert result.exit_code == 1


def test_export_reports_missing_session():
    result = runner.invoke(cli_app, ["export", "no-such-session"])
    assert result.exit_code == 1


def test_session_can_be_found_by_title_fragment(stored_session, tmp_path):
    result = runner.invoke(cli_app, ["export", "Cell", "--out", str(tmp_path / "x.md")])
    assert result.exit_code == 0


def _doctor_report(*checks):
    """Build a doctor Report without touching the host environment."""
    from lectern.doctor import DoctorReport

    return DoctorReport(checks=list(checks))


def test_doctor_exits_nonzero_when_a_required_check_fails(monkeypatch):
    """Stubbed so the result does not depend on what this machine has installed.

    Asserting on the real environment made the outcome an accident of the
    runner: green here only because whisper.cpp and Ollama are absent, and red
    on a properly set up Mac, which is exactly backwards.
    """
    from lectern.doctor import Check, CheckStatus

    async def fake_run_all(config):  # noqa: ANN001, ARG001
        return _doctor_report(
            Check(name="Python", status=CheckStatus.OK, detail="3.12"),
            Check(
                name="whisper.cpp",
                status=CheckStatus.FAIL,
                detail="whisper-server not found",
                remedy="brew install whisper-cpp",
            ),
        )

    monkeypatch.setattr("lectern.doctor.run_all", fake_run_all)
    result = runner.invoke(cli_app, ["doctor"])
    assert result.exit_code == 1
    assert "LECTERN DOCTOR" in result.stdout
    assert "whisper-server not found" in result.stdout


def test_doctor_exits_zero_when_every_check_passes(monkeypatch):
    """The companion case, which the host-dependent test could never reach."""
    from lectern.doctor import Check, CheckStatus

    async def fake_run_all(config):  # noqa: ANN001, ARG001
        return _doctor_report(
            Check(name="Python", status=CheckStatus.OK, detail="3.12"),
            Check(name="whisper.cpp", status=CheckStatus.OK, detail="installed"),
        )

    monkeypatch.setattr("lectern.doctor.run_all", fake_run_all)
    result = runner.invoke(cli_app, ["doctor"])
    assert result.exit_code == 0
    assert "LECTERN DOCTOR" in result.stdout


def test_config_init_and_set():
    from lectern.config import manager as config_manager
    from lectern.utils import paths

    assert runner.invoke(cli_app, ["config", "init"]).exit_code == 0
    assert paths.config_file().exists()

    result = runner.invoke(cli_app, ["config", "set", "ollama.notes_model=llama3.2:3b"])
    assert result.exit_code == 0
    assert config_manager.load().ollama.notes_model == "llama3.2:3b"


def test_config_set_rejects_bad_input():
    assert runner.invoke(cli_app, ["config", "set", "not-an-assignment"]).exit_code == 1
    assert runner.invoke(cli_app, ["config", "set", "ollama.nope=1"]).exit_code == 1


def test_config_show_without_a_file():
    result = runner.invoke(cli_app, ["config"])
    assert result.exit_code == 0
    assert "defaults are in use" in result.stdout


def test_models_whisper_lists_availability(monkeypatch, tmp_path):
    """Both states are asserted, rather than whichever this machine happens to be in.

    The installed model needs a path that really exists, because
    ``WhisperModel.installed`` stats it. Note the assertions distinguish the two
    states positively: "installed" alone would also match "not installed".
    """
    from lectern.transcription.models import WhisperModel

    present = tmp_path / "ggml-base.en.bin"
    present.write_bytes(b"weights")

    monkeypatch.setattr(
        "lectern.transcription.models.available_models",
        lambda: [
            WhisperModel(name="small.en", path=None, approx_size="488 MB"),
            WhisperModel(name="base.en", path=present, approx_size="148 MB"),
        ],
    )
    result = runner.invoke(cli_app, ["models", "whisper"])
    assert result.exit_code == 0

    lines = {
        line.split()[0]: line
        for line in result.stdout.splitlines()
        if line.strip().startswith(("small.en", "base.en"))
    }
    assert "not installed" in lines["small.en"]
    assert "not installed" not in lines["base.en"]
    assert "installed" in lines["base.en"]


def test_models_ollama_reports_a_missing_daemon(monkeypatch):
    """Stubbed: the old version probed the real localhost:11434 and required it
    to be down, so it failed on any machine actually running Ollama."""
    from lectern.llm.base import LLMHealth

    async def unavailable(self):  # noqa: ANN001, ARG001
        return LLMHealth(available=False, detail="not running")

    monkeypatch.setattr("lectern.llm.ollama.OllamaBackend.health", unavailable)
    result = runner.invoke(cli_app, ["models", "ollama"])
    assert result.exit_code == 0
    assert "not responding" in result.output


def test_models_ollama_lists_installed_models(monkeypatch):
    from lectern.llm.base import LLMHealth, LLMModel

    async def available(self):  # noqa: ANN001, ARG001
        return LLMHealth(
            available=True,
            detail="1 model installed",
            version="0.1.0",
            models=[LLMModel(name="qwen3:8b", size_bytes=5_200_000_000)],
        )

    monkeypatch.setattr("lectern.llm.ollama.OllamaBackend.health", available)
    result = runner.invoke(cli_app, ["models", "ollama"])
    assert result.exit_code == 0
    assert "qwen3:8b" in result.output


def test_logs_reports_path():
    result = runner.invoke(cli_app, ["logs", "--path"])
    assert result.exit_code == 0
    assert "lectern.log" in result.stdout


def test_reindex(stored_session):
    result = runner.invoke(cli_app, ["reindex"])
    assert result.exit_code == 0
    assert "Reindexed 1 session" in result.stdout


def test_record_rejects_a_missing_file():
    result = runner.invoke(cli_app, ["record", "--file", "/nonexistent/lecture.wav"])
    assert result.exit_code == 1
    assert "No such file" in result.output


def test_config_show_keeps_section_headers():
    """Rich markup used to swallow every [section] header in the output."""
    runner.invoke(cli_app, ["config", "init"])
    result = runner.invoke(cli_app, ["config"])
    assert result.exit_code == 0
    for section in ("[transcription]", "[ollama]", "[notes]", "[audio]", "[ui]"):
        assert section in result.stdout


def test_session_titles_with_brackets_survive_listing(manager):
    """A title is user data, not markup: it must appear verbatim."""
    _meta, store = manager.create(title="MATH 200 [Section 3] Vectors")
    store.close()
    manager.reindex()
    manager.close()

    result = runner.invoke(cli_app, ["sessions"])
    assert result.exit_code == 0
    assert "Section 3" in result.stdout
