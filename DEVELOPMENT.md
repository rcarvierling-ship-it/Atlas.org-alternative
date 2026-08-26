# Development

## Setup

```bash
uv sync --extra audio          # omit --extra audio if PortAudio isn't available
uv run lectern doctor          # see what your machine is missing
```

`uv run lectern` launches the app from the checkout. `uv run pytest` runs the suite.

To get a `lectern` command on your PATH that tracks this checkout, run
`./scripts/install.sh` (or `uv tool install --editable . --with sounddevice`).
It installs in editable mode, so `git pull` updates the installed command.

Set `LECTERN_HOME=/tmp/lectern-dev` to redirect config, sessions and logs
somewhere disposable — the test suite does exactly this.

---

## Working without whisper.cpp or Ollama

You do not need either installed to work on Lectern. `tests/fakes.py` provides
in-process HTTP servers that speak the real protocols, and two entry points use
them:

```bash
# The real TUI, driven by a WAV file and stub models.
uv run python scripts/demo.py --speed 2.0

# Regenerate the README screenshots (headless).
uv run python scripts/screenshots.py
```

Both write to a temporary directory, so your real sessions are untouched.

To run the pipeline against a real recording with real models:

```bash
uv run lectern record --file path/to/lecture.wav
```

This uses the full production pipeline — VAD, whisper.cpp, the note scheduler,
Ollama, persistence — with the file standing in for a microphone.

---

## Tests

```bash
uv run pytest                          # everything (~1 minute)
uv run pytest tests/test_notes.py      # one file
uv run pytest -k acceptance            # the end-to-end workflow
uv run pytest -q --timeout 120         # if you install pytest-timeout
```

| File | Covers |
|---|---|
| `test_config.py` | config load/save/coercion, atomic writes |
| `test_notes.py` | `NoteState` merge rules, consolidation safety |
| `test_scheduler.py` | update timing, debounce, backpressure (injected clock) |
| `test_parsing.py` | Ollama and whisper.cpp response shapes |
| `test_audio.py` | VAD segmentation, PCM helpers, sources, backpressure |
| `test_sessions.py` | persistence, torn writes, index, search, recovery, export |
| `test_pipeline.py` | end-to-end audio → transcript → notes, and failure paths |
| `test_tui.py` | screens and widgets driven by Textual's Pilot |
| `test_acceptance.py` | the full product workflow through the UI |
| `test_cli.py` | every CLI command |

The pipeline and acceptance tests drive the production code path; only the two
model servers are fakes. When you change the pipeline, those are the tests that
will actually tell you.

### Regenerating the audio fixture

```bash
uv run python scripts/make_fixture_audio.py
```

Produces `tests/fixtures/lecture.wav`: speech-shaped bursts separated by
silence, enough structure for the VAD to segment without shipping a real
recording of someone's voice.

---

## Building the native helper

```bash
./scripts/build-native.sh
```

Requires macOS 13+ and the Xcode command line tools. Produces
`native/audio-capture/build/lectern-audio-capture`.

The helper's contract with Python is deliberately trivial so it can be faked:

- **stdout** — raw little-endian float32 mono PCM at the requested rate
- **stderr** — one JSON object per line: `{"event": ..., "message": ...}`
- **exit 13** — screen-recording permission denied

Everything about it is isolated in `lectern/audio/system.py`; nothing else in
Lectern knows system audio involves a subprocess.

---

## Conventions

- **Typed Python.** Annotations everywhere; `from __future__ import annotations`
  at the top of each module.
- **Docstrings explain architecture, not syntax.** Module docstrings say why the
  module is shaped the way it is. Inline comments are for constraints the code
  cannot show — not narration.
- **Async, not threads.** Threads only where a library forces one (PortAudio's
  callback) or for blocking syscalls via `asyncio.to_thread`.
- **No cloud fallbacks, ever.** `LLMBackend` implementations must be local. If
  the local model is unavailable, keep transcribing and tell the user.
- **`lectern` is the only command a user should need.** whisper.cpp is spawned
  per session; Ollama is started by `ensure_ollama_running` when it is installed
  but not answering. Never auto-start anything for a non-local host, and never
  install software without explicit confirmation.
- **Expected errors become dialogs.** A missing permission, a dead daemon or a
  full disk is a message the user can act on, never a traceback over the TUI.
- **The transcript outranks everything.** Any change that risks transcript
  durability for the sake of notes, UI or performance is the wrong change.

---

## Adding things

**A transcription backend** — subclass `TranscriptionBackend`
(`transcription/base.py`), keep the model resident between `start()` and
`stop()`, and register it in `transcription/__init__.py:build_backend`.

**An LLM backend** — subclass `LLMBackend` (`llm/base.py`). It must support
streaming and schema-constrained output for note quality to hold up on small
models. Register it in `llm/__init__.py:build_backend`.

**An export format** — subclass `Exporter` (`sessions/export.py`), implement
`render()`, add it to `EXPORTERS`. The modal and CLI pick it up automatically.

**An audio source** — subclass `AudioSource` (`audio/base.py`), publish 16 kHz
mono float32 blocks with `self._publish()`, and add it to
`audio/__init__.py:build_source`.

**A screen** — add to `lectern/screens/`, style it in `lectern/lectern.tcss`,
and add a command to `LecternApp.get_system_commands` so it is reachable from
the palette.

---

## Debugging

Logs go to `~/.local/state/lectern/lectern.log` (never to the TUI):

```bash
lectern logs -n 200
tail -f "$(lectern logs --path)"
lectern --verbose            # DEBUG level, includes whisper-server output
```

For Textual layout problems:

```bash
uv run textual console        # in one terminal
uv run textual run --dev -c lectern   # in another
```
