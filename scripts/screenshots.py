#!/usr/bin/env python3
"""Capture SVG screenshots of the TUI for the README.

Runs the real app headlessly against the fake whisper.cpp and Ollama servers
from the test suite, walks it through a short session, and exports each screen
as an SVG that renders anywhere.

    python scripts/screenshots.py [--out docs/screenshots]
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

SIZE = (120, 38)


async def capture(out_dir: Path) -> list[Path]:
    from fakes import FakeOllamaServer, FakeWhisperServer

    from lectern.app import LecternApp
    from lectern.config.models import LecternConfig
    from lectern.services import AppServices, SessionRequest

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    whisper = FakeWhisperServer().start()
    ollama = FakeOllamaServer().start()
    try:
        config = LecternConfig()
        config.transcription.server_url = whisper.url
        config.transcription.partials = False
        config.ollama.host = ollama.url
        config.ollama.notes_model = "qwen3:8b"
        config.ollama.final_model = "qwen3:8b"
        config.notes.update_interval_seconds = 5.0
        config.notes.min_new_words = 5

        services = AppServices(config=config)
        request = SessionRequest(
            title="BIO 113 — Cell Structure",
            course="Microbiology",
            whisper_model="small.en",
            notes_model="qwen3:8b",
            file_path=ROOT / "tests" / "fixtures" / "lecture.wav",
            file_speed=4.0,
        )
        app = LecternApp(services=services, start_request=request)

        def save(name: str) -> None:
            path = out_dir / f"{name}.svg"
            path.write_text(app.export_screenshot(title=f"Lectern — {name}"), encoding="utf-8")
            written.append(path)
            print(f"  {path.relative_to(ROOT)}")

        async with app.run_test(size=SIZE) as pilot:
            from lectern.screens.recording import RecordingScreen
            from lectern.screens.review import ReviewScreen

            # Recording, once transcript and notes have both populated.
            for _ in range(600):
                await pilot.pause(0.05)
                screen = app.screen
                if (
                    isinstance(screen, RecordingScreen)
                    and screen.pipeline is not None
                    and len(screen.pipeline.segments) >= 3
                    and not screen.pipeline.notes.is_empty
                ):
                    break
            await pilot.press("m")
            await pilot.pause(0.2)
            save("recording")

            # Finish, and capture the review screen.
            await pilot.press("q")
            await pilot.pause(0.2)
            save("finish-confirm")
            await pilot.press("enter")
            for _ in range(600):
                await pilot.pause(0.05)
                if isinstance(app.screen, ReviewScreen):
                    break
            await pilot.pause(0.4)
            save("review")

            await pilot.press("t")
            await pilot.pause(0.2)
            save("review-transcript")

            # Back to Home, which now has a session in Recent.
            app.pop_screen()
            await pilot.pause(0.5)
            save("home")

            await pilot.press("question_mark")
            await pilot.pause(0.3)
            save("help")
            await pilot.press("escape")

            await pilot.press("d")
            await pilot.pause(2.5)
            save("doctor")

            await app.action_quit()
    finally:
        whisper.stop()
        ollama.stop()
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "screenshots")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        # Never touch the developer's real sessions or config.
        os.environ["LECTERN_HOME"] = tmp
        os.environ["LECTERN_SKIP_WIZARD"] = "1"
        print("Capturing screenshots…")
        written = asyncio.run(capture(args.out))
    print(f"wrote {len(written)} screenshot(s)")


if __name__ == "__main__":
    main()
