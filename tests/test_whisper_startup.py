"""Regression tests for whisper.cpp server startup command construction."""

from pathlib import Path

from lectern.transcription.whisper_cpp import build_server_command


def test_build_server_command_omits_print_progress_false():
    command = build_server_command(
        Path("/opt/homebrew/bin/whisper-server"),
        Path("/tmp/ggml-small.en.bin"),
        port=61804,
        language="en",
    )

    assert "--print-progress" not in command
    assert "false" not in command
    assert command == [
        "/opt/homebrew/bin/whisper-server",
        "--model",
        "/tmp/ggml-small.en.bin",
        "--host",
        "127.0.0.1",
        "--port",
        "61804",
        "--language",
        "en",
        "--no-context",
    ]


def test_build_server_command_adds_threads_only_when_requested():
    command = build_server_command(
        Path("/usr/local/bin/whisper-server"),
        Path("/tmp/model.bin"),
        port=8080,
        threads=6,
    )

    assert command[-2:] == ["--threads", "6"]
