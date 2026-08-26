#!/usr/bin/env python3
"""Run the real Lectern TUI against stub model servers.

Useful for working on the interface without whisper.cpp or Ollama installed:
the transcript and notes are produced by the same fakes the test suite uses, so
every other layer — VAD, the HTTP clients, the scheduler, persistence, the
screens — is the production code.

    python scripts/demo.py [--file tests/fixtures/lecture.wav] [--speed 2.0]

Sessions are written to a temporary directory and deleted on exit, so this never
touches your real notes.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--file", type=Path, default=ROOT / "tests" / "fixtures" / "lecture.wav",
        help="WAV file to feed through the pipeline.",
    )
    parser.add_argument("--speed", type=float, default=2.0, help="Playback speed (1.0 = real time).")
    parser.add_argument(
        "--keep", action="store_true", help="Keep the temporary session directory on exit."
    )
    args = parser.parse_args()

    if not args.file.exists():
        parser.error(f"no such file: {args.file}\nRun scripts/make_fixture_audio.py first.")

    from fakes import FakeOllamaServer, FakeWhisperServer

    whisper = FakeWhisperServer().start()
    ollama = FakeOllamaServer().start()

    home = tempfile.mkdtemp(prefix="lectern-demo-")
    os.environ["LECTERN_HOME"] = home
    os.environ["LECTERN_SKIP_WIZARD"] = "1"

    from lectern.app import LecternApp
    from lectern.config.models import LecternConfig
    from lectern.services import AppServices, SessionRequest

    config = LecternConfig()
    config.transcription.server_url = whisper.url
    config.transcription.partials = False
    config.ollama.host = ollama.url
    config.ollama.notes_model = "qwen3:8b"
    config.ollama.final_model = "qwen3:8b"
    config.notes.update_interval_seconds = 8.0
    config.notes.min_new_words = 5

    print(f"Demo stack: whisper={whisper.url} ollama={ollama.url}")
    print(f"Sessions:   {home}")
    print("Starting Lectern…")

    services = AppServices(config=config)
    app = LecternApp(
        services=services,
        start_request=SessionRequest(
            title="Demo Lecture",
            course="Microbiology",
            whisper_model="small.en",
            notes_model="qwen3:8b",
            file_path=args.file,
            file_speed=args.speed,
        ),
    )
    try:
        app.run()
    finally:
        whisper.stop()
        ollama.stop()
        if args.keep:
            print(f"Session data kept at {home}")
        else:
            import shutil

            shutil.rmtree(home, ignore_errors=True)


if __name__ == "__main__":
    main()
