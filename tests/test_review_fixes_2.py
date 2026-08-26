"""Regression tests for the second review round on PR #1.

The headline one is the whisper-server command line: two of the flags Lectern
passed do not exist in whisper.cpp's *server* (they are whisper-cli flags), and
an unrecognised argument makes the server print usage and exit — so recording
could never have started on a real machine.
"""

from __future__ import annotations


import numpy as np
import pytest

from lectern.config import manager as config_manager
from lectern.config.models import LecternConfig
from lectern.notes.models import NoteState
from lectern.transcription.whisper_cpp import WhisperCppBackend
from lectern.utils.text import looks_like_hallucination

# Every flag whisper.cpp's examples/server accepts. Anything outside this set
# hits its "error: unknown argument" path and exits before serving.
WHISPER_SERVER_FLAGS = {
    "-h", "--help", "-t", "--threads", "-p", "--processors", "-ot", "--offset-t",
    "-on", "--offset-n", "-d", "--duration", "-mc", "--max-context", "-ml", "--max-len",
    "-bo", "--best-of", "-bs", "--beam-size", "-ac", "--audio-ctx", "-wt", "--word-thold",
    "-et", "--entropy-thold", "-lpt", "--logprob-thold", "-debug", "--debug-mode",
    "-tr", "--translate", "-di", "--diarize", "-tdrz", "--tinydiarize",
    "-sow", "--split-on-word", "-nf", "--no-fallback", "-fp", "--font-path",
    "-ps", "--print-special", "-pc", "--print-colors", "-pr", "--print-realtime",
    "-pp", "--print-progress", "-nt", "--no-timestamps", "-l", "--language",
    "-dl", "--detect-language", "--prompt", "--carry-initial-prompt", "-m", "--model",
    "-oved", "--ov-e-device", "-dtw", "--dtw", "-ng", "--no-gpu", "-dev", "--device",
    "-fa", "--flash-attn", "-nfa", "--no-flash-attn", "-sns", "--suppress-nst",
    "-nth", "--no-speech-thold", "-nlp", "--no-language-probabilities",
    "--port", "--host", "--public", "--request-path", "--inference-path",
    "--convert", "--tmp-dir", "--vad", "-vm", "--vad-model", "-vt", "--vad-threshold",
    "-vspd", "--vad-min-speech-duration-ms", "-vsd", "--vad-min-silence-duration-ms",
    "-vmsd", "--vad-max-speech-duration-s", "-vp", "--vad-speech-pad-ms",
    "-vo", "--vad-samples-overlap",
}


def build_command(monkeypatch, tmp_path, threads: int = 0) -> list[str]:
    """Capture the argv Lectern would launch whisper-server with."""
    model = tmp_path / "ggml-small.en.bin"
    model.write_bytes(b"weights")
    binary = tmp_path / "whisper-server"
    binary.write_text("#!/bin/sh\n")

    captured: list[str] = []

    async def fake_exec(*args, **kwargs):  # noqa: ANN001, ARG001
        captured.extend(args)
        raise OSError("not actually launching")

    monkeypatch.setattr("lectern.transcription.whisper_cpp.find_model", lambda name: model)  # noqa: ARG005
    monkeypatch.setattr(
        "lectern.transcription.whisper_cpp.find_whisper_server", lambda configured="": binary  # noqa: ARG005
    )
    monkeypatch.setattr("asyncio.create_subprocess_exec", fake_exec)

    backend = WhisperCppBackend(model="small.en", threads=threads)
    return captured, backend


async def test_every_whisper_server_flag_is_one_the_server_accepts(monkeypatch, tmp_path):
    from lectern.transcription.base import TranscriptionError

    captured, backend = build_command(monkeypatch, tmp_path, threads=4)
    with pytest.raises(TranscriptionError):
        await backend.start()

    flags = [arg for arg in captured[1:] if arg.startswith("-")]
    assert flags, "no flags were captured"
    unknown = [flag for flag in flags if flag not in WHISPER_SERVER_FLAGS]
    assert not unknown, f"whisper-server would exit on: {unknown}"


async def test_no_bare_value_follows_a_valueless_flag(monkeypatch, tmp_path):
    """`--print-progress false` made "false" a stray positional argument."""
    from lectern.transcription.base import TranscriptionError

    captured, backend = build_command(monkeypatch, tmp_path)
    with pytest.raises(TranscriptionError):
        await backend.start()

    assert "--print-progress" not in captured
    assert "false" not in captured
    # whisper-cli's flag, not the server's.
    assert "--no-context" not in captured


# --- the sessions screen shadowed Textual's internal Widget._render ---------


async def test_sessions_screen_renders(manager):
    """`_render` on a Screen breaks layout with a NoneType error."""
    from lectern.app import LecternApp
    from lectern.screens.home import SessionRow
    from lectern.screens.sessions import SessionsScreen
    from lectern.services import AppServices
    from lectern.sessions.models import SessionStatus

    meta, store = manager.create(title="Browsable Lecture", course="BIO 113")
    meta.status = SessionStatus.COMPLETE
    store.save_meta(meta)
    manager.index.upsert(meta)
    store.close()

    app = LecternApp(services=AppServices(config=LecternConfig(), _manager=manager))
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.press("s")
        await pilot.pause()
        assert isinstance(app.screen, SessionsScreen)
        # The regression is a _render defined *on the screen itself*, shadowing
        # Widget._render. Checking hasattr/callable would pass either way,
        # because the inherited Widget._render always exists and is callable.
        for klass in type(app.screen).__mro__:
            if klass.__module__.startswith("lectern."):
                assert "_render" not in vars(klass), (
                    f"{klass.__name__} defines _render, which shadows Widget._render"
                )
        rows = app.screen.query(SessionRow)
        assert rows and rows.first().meta.id == meta.id

        app.screen.query_one("#search-input").value = "nonexistent"
        await pilot.pause()
        assert not app.screen.query(SessionRow)


# --- a device that vanished must not crash the form ------------------------


async def test_new_session_form_survives_a_missing_saved_device(manager, monkeypatch):
    """Assigning a Select a value outside its options raises in Textual."""
    from lectern.app import LecternApp
    from lectern.screens.new_session import NewSessionScreen
    from lectern.services import AppServices

    monkeypatch.setattr("lectern.audio.devices.list_input_devices", lambda: [])
    config = LecternConfig()
    config.audio.input_device = "AirPods That Are Not Here"

    app = LecternApp(services=AppServices(config=config, _manager=manager))
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.press("n")
        await pilot.pause()
        assert isinstance(app.screen, NewSessionScreen)
        for _ in range(40):
            await pilot.pause(0.05)
        # The worker ran to completion: the later fields were populated.
        assert app.screen.query_one("#whisper-model") is not None


# --- hallucination patterns must survive normalization ---------------------


@pytest.mark.parametrize(
    "text",
    [
        "Amara.org",
        "Subtitles by the Amara.org community",
        "www.mooji.org",
        "You're watching a video",
        "Thanks for watching!",
        "[ Silence ]",
        "you you you you",
        "...",
    ],
)
def test_known_whisper_hallucinations_are_filtered(text):
    assert looks_like_hallucination(text)


@pytest.mark.parametrize(
    "text",
    [
        "The cell membrane is a phospholipid bilayer.",
        "Gram-positive bacteria have thick walls.",
        "Watch the sign carefully when you integrate.",
    ],
)
def test_real_speech_is_not_filtered(text):
    assert not looks_like_hallucination(text)


# --- config: typos must not silently disable features ----------------------


def test_invalid_boolean_is_rejected():
    config = LecternConfig()
    with pytest.raises(ValueError, match="invalid boolean"):
        config_manager.set_value(config, "audio.save_recording", "treu")
    assert config.audio.save_recording is True


def test_false_spellings_are_accepted():
    for value in ("false", "0", "no", "off"):
        config = LecternConfig()
        config_manager.set_value(config, "audio.save_recording", value)
        assert config.audio.save_recording is False


# --- exports are written atomically ----------------------------------------


def test_export_leaves_no_temp_file_and_replaces_atomically(manager):
    from lectern.sessions.export import export_session
    from lectern.transcription.base import TranscriptSegment

    meta, store = manager.create(title="Atomic Export")
    store.append_segment(TranscriptSegment(id=1, start_time=0, end_time=2, text="Hello."))
    store.close()
    manager.reindex()

    session = manager.open(meta.id)
    first = export_session(session, format_id="markdown")
    assert first.exists()
    assert not list(first.parent.glob("*.tmp"))

    # Re-exporting replaces the file rather than truncating it in place.
    second = export_session(session, format_id="markdown")
    assert second == first
    assert "Hello." in second.read_text(encoding="utf-8")


# --- a new session is searchable by title before it has any transcript -----


def test_new_session_is_searchable_by_title(manager):
    meta, store = manager.create(title="Quantum Chromodynamics", course="PHYS 400")
    store.close()

    hits = manager.search("Quantum Chromodynamics")
    assert hits and hits[0].session_id == meta.id


# --- repeating the current topic must not churn the revision counter -------


def test_repeating_the_current_topic_is_not_a_change():
    from lectern.notes.updater import apply_update_payload

    state = NoteState()
    apply_update_payload(state, {"current_topic": "Cells", "summary": "About cells."})
    revision = state.revision

    result = apply_update_payload(state, {"current_topic": "Cells", "summary": "About cells."})
    assert not result.changed
    assert state.revision == revision


# --- system audio: a partial sample must not misalign the stream -----------


def test_partial_pcm_sample_is_carried_to_the_next_read():
    """A pipe read can end mid-sample; dropping the tail corrupts everything after."""
    import asyncio

    from lectern.audio.system import SystemAudioSource

    source = SystemAudioSource()
    samples = np.arange(8, dtype=np.float32)
    payload = samples.tobytes()

    class FakeStream:
        def __init__(self, chunks):
            self._chunks = list(chunks)

        async def read(self, _n):
            return self._chunks.pop(0) if self._chunks else b""

    class FakeProcess:
        def __init__(self, stream):
            self.stdout = stream

    # Split so the first read ends 2 bytes into a sample.
    source._process = FakeProcess(FakeStream([payload[:18], payload[18:]]))
    source._running = True

    received: list[np.ndarray] = []
    source._publish = lambda block: received.append(block)  # type: ignore[method-assign]

    asyncio.run(source._read_audio())

    assert received
    combined = np.concatenate(received)
    assert combined.size == 8
    assert combined == pytest.approx(samples)


# --- an audio source must always terminate frames() ------------------------


async def test_close_stream_delivers_the_sentinel_on_a_full_queue():
    """Dropping it leaves every frames() consumer waiting on a dead stream."""
    import asyncio

    from lectern.audio.base import AudioSource

    class Dummy(AudioSource):
        kind = "dummy"

        async def start(self) -> None:
            self._running = True

        async def stop(self) -> None:
            self._running = False

    source = Dummy()
    source._queue = asyncio.Queue(maxsize=3)
    for value in range(3):
        source._queue.put_nowait(np.full(2, float(value), dtype=np.float32))
    assert source._queue.full()

    source._close_stream()

    async def drain() -> list[np.ndarray]:
        return [block async for block in source.frames()]

    # A dropped sentinel makes frames() wait forever, which would hang the
    # suite rather than fail it — so bound the wait and assert what survived.
    received = await asyncio.wait_for(drain(), timeout=5.0)
    assert len(received) == 2, "the sentinel should evict exactly one buffered block"


# --- third review round ----------------------------------------------------
#
# Each test below fails against the code as it was before the fix it names.


def test_malformed_ollama_host_reports_unavailable_instead_of_raising():
    """httpx.InvalidURL is not an HTTPError, so it escaped the health guard.

    AppServices.refresh_llm_health calls health() during startup; an
    unhandled exception there took the whole app down instead of degrading
    to "notes paused".
    """
    import asyncio

    from lectern.llm.ollama import OllamaBackend

    # An unterminated IPv6 bracket — a plausible typo for someone pointing
    # ollama.host at ::1 — is one of the few inputs httpx refuses outright
    # rather than percent-encoding.
    backend = OllamaBackend("http://[::1")
    health = asyncio.run(backend.health())
    assert health.available is False


def test_null_models_list_does_not_raise_type_error():
    """`{"models": null}` reached the for-loop as None via .get(key, default)."""
    import asyncio

    import httpx

    from lectern.llm.base import LLMError
    from lectern.llm.ollama import OllamaBackend

    async def run(payload: object) -> list:
        backend = OllamaBackend("http://127.0.0.1:1")

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=payload)

        backend._client = httpx.AsyncClient(
            base_url="http://127.0.0.1:1", transport=httpx.MockTransport(handler)
        )
        try:
            return await backend.list_models()
        finally:
            await backend.close()

    assert asyncio.run(run({"models": None})) == []
    assert asyncio.run(run({})) == []

    # A non-list is a protocol error, not an empty list.
    with pytest.raises(LLMError):
        asyncio.run(run({"models": {"name": "qwen3:8b"}}))


def test_non_integer_model_size_does_not_break_rendering():
    """size_bytes feeds arithmetic in size_label, so a string failed at render."""
    import asyncio

    import httpx

    from lectern.llm.ollama import OllamaBackend

    async def run() -> list:
        backend = OllamaBackend("http://127.0.0.1:1")

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"models": [{"name": "m", "size": "4.7GB"}]})

        backend._client = httpx.AsyncClient(
            base_url="http://127.0.0.1:1", transport=httpx.MockTransport(handler)
        )
        try:
            return await backend.list_models()
        finally:
            await backend.close()

    models = asyncio.run(run())
    assert models[0].size_bytes is None
    assert models[0].size_label == "—"


def test_starring_an_existing_bullet_bumps_the_revision():
    """A starred upgrade adds no item, so `changed` stayed false and the
    notes pane skipped the render that would have shown the star."""
    from lectern.notes.models import NoteItem
    from lectern.notes.updater import apply_update_payload

    state = NoteState()
    state.add_bullets("key_points", [NoteItem(text="Membranes are bilayers")])
    before = state.revision

    result = apply_update_payload(
        state,
        {"key_points": [{"text": "Membranes are bilayers", "starred": True}]},
        timestamp=10.0,
    )

    assert result.state.key_points[0].starred is True
    assert result.changed is True
    assert result.state.revision == before + 1


def test_filling_in_a_missing_definition_bumps_the_revision():
    from lectern.notes.models import TermEntry
    from lectern.notes.updater import apply_update_payload

    state = NoteState()
    state.add_terms("definitions", [TermEntry(term="Enzyme", definition="")])
    before = state.revision

    result = apply_update_payload(
        state,
        {"definitions": [{"term": "Enzyme", "definition": "A biological catalyst."}]},
        timestamp=10.0,
    )

    assert result.state.definitions[0].definition == "A biological catalyst."
    assert result.changed is True
    assert result.state.revision == before + 1


def test_repeating_an_identical_item_does_not_bump_the_revision():
    """The counters must not treat a pure duplicate as a change."""
    from lectern.notes.models import NoteItem
    from lectern.notes.updater import apply_update_payload

    state = NoteState()
    state.add_bullets("key_points", [NoteItem(text="Membranes are bilayers", starred=True)])
    before = state.revision

    result = apply_update_payload(
        state,
        {"key_points": [{"text": "Membranes are bilayers", "starred": True}]},
        timestamp=10.0,
    )

    assert result.changed is False
    assert result.state.revision == before


def test_saving_settings_rebuilds_the_llm_backend_for_the_new_host():
    """Settings edits ollama.host in place and calls save_config; the cached
    backend kept talking to the old host."""
    from lectern.services import AppServices

    config = LecternConfig()
    config.ollama.host = "http://127.0.0.1:11434"
    services = AppServices(config=config)

    first = services.llm
    assert first.host.startswith("http://127.0.0.1:11434")

    services.config.ollama.host = "http://127.0.0.1:99"
    services.save_config()

    second = services.llm
    assert second is not first
    assert second.host.startswith("http://127.0.0.1:99")
    # The replaced backend is kept for closing, not dropped on the floor.
    assert first in services._retired_llms


def test_reindex_drops_rows_for_folders_that_no_longer_exist():
    """A folder removed outside delete() left a row that lists but won't open."""
    import shutil

    from lectern.sessions.manager import SessionManager

    config = LecternConfig()
    manager = SessionManager(config)
    try:
        meta, store = manager.create(title="Deleted Later", course="BIO 113")
        store.close()
        assert manager.reindex() == 1
        assert any(row.id == meta.id for row in manager.all_sessions())

        shutil.rmtree(meta.folder)

        assert manager.reindex() == 0
        assert not any(row.id == meta.id for row in manager.all_sessions())
    finally:
        manager.close()


def test_mix_keeps_each_gain_with_its_own_track():
    """Filtering empty tracks while leaving gains in place shifted every
    later gain onto the wrong track."""
    from lectern.utils.audio_utils import mix

    empty = np.zeros(0, dtype=np.float32)
    loud = np.ones(4, dtype=np.float32)

    # The first track is dropped; the surviving track must keep *its* gain
    # (0.5), not inherit the dropped track's (0.0).
    mixed = mix(empty, loud, gains=(0.0, 0.5))
    assert mixed == pytest.approx(np.full(4, 0.5, dtype=np.float32))
