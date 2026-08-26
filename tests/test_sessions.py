"""Session persistence, indexing, search, recovery and export."""

from __future__ import annotations

import json

from lectern.notes.models import NoteItem, NoteState
from lectern.sessions import recovery
from lectern.sessions.export import export_session, get_exporter
from lectern.sessions.models import Marker, MarkerKind, SessionStatus
from lectern.sessions.storage import SessionStore, session_folder_name
from lectern.transcription.base import TranscriptSegment
from lectern.utils.timefmt import utcnow


def segment(index: int, text: str) -> TranscriptSegment:
    return TranscriptSegment(
        id=index, start_time=index * 5.0, end_time=index * 5.0 + 4.0, text=text
    )


def test_folder_name_is_dated_and_slugged():
    name = session_folder_name(utcnow(), "BIO 113 — Cell Structure!")
    assert name.endswith("-bio-113-cell-structure")


def test_create_writes_the_folder_and_indexes_it(manager):
    meta, store = manager.create(title="Test Lecture", course="BIO 113", whisper_model="small.en")
    assert store.session_file.exists()
    assert meta.status is SessionStatus.RECORDING
    assert manager.get(meta.id) is not None
    store.close()


def test_segments_are_appended_and_reloaded(manager):
    meta, store = manager.create(title="Test Lecture")
    for index in range(1, 4):
        store.append_segment(segment(index, f"Sentence number {index}."))
    store.close()

    reloaded = SessionStore(meta.folder).load_segments()
    assert [item.id for item in reloaded] == [1, 2, 3]
    assert reloaded[2].text == "Sentence number 3."


def test_a_torn_final_line_does_not_lose_earlier_segments(manager):
    """Simulates a crash mid-append: the last line is incomplete."""
    meta, store = manager.create(title="Crash Test")
    store.append_segment(segment(1, "First complete segment."))
    store.append_segment(segment(2, "Second complete segment."))
    store.close()

    with store.transcript_jsonl.open("a", encoding="utf-8") as handle:
        handle.write('{"id": 3, "start_time": 10.0, "end_ti')

    reloaded = SessionStore(meta.folder).load_segments()
    assert len(reloaded) == 2
    assert reloaded[-1].text == "Second complete segment."


def test_note_state_persists_across_reload(manager):
    meta, store = manager.create(title="Notes Test")
    state = NoteState(summary="A summary.", current_topic="Topic A")
    state.add_topic("Topic A")
    state.add_bullets("key_points", [NoteItem(text="A point", starred=True)])
    store.save_note_state(state, title="Notes Test")
    store.close()

    reloaded = SessionStore(meta.folder).load_note_state()
    assert reloaded.summary == "A summary."
    assert reloaded.key_points[0].starred
    assert SessionStore(meta.folder).notes_live_md.exists()


def test_corrupt_note_state_degrades_to_empty_not_crash(manager):
    meta, store = manager.create(title="Corrupt Notes")
    store.notes_live_json.write_text("{not json", encoding="utf-8")
    store.close()
    assert SessionStore(meta.folder).load_note_state().is_empty


def test_markers_round_trip(manager):
    meta, store = manager.create(title="Marker Test")
    markers = [
        Marker(time=12.0, kind=MarkerKind.IMPORTANT),
        Marker(time=30.0, kind=MarkerKind.NOTE, text="On Exam 1"),
    ]
    store.save_markers(markers)
    store.close()

    reloaded = SessionStore(meta.folder).load_markers()
    assert reloaded[0].label == "Important"
    assert reloaded[1].text == "On Exam 1"
    assert reloaded[1].clock == "00:00:30"


def test_index_search_finds_transcript_text(manager):
    meta, store = manager.create(title="Microbiology")
    for index, text in enumerate(
        ["Gram positive bacteria have thick walls.", "Mitochondria make ATP."], start=1
    ):
        store.append_segment(segment(index, text))
    store.close()
    manager.reindex()

    hits = manager.search("mitochondria")
    assert hits and hits[0].session_id == meta.id

    assert manager.search("gram positive")
    assert manager.search("nonexistentterm") == []


def test_reindex_rebuilds_from_disk(manager, config):
    meta, store = manager.create(title="Rebuildable")
    store.append_segment(segment(1, "Some content here for counting words."))
    store.close()

    # Wipe the database entirely; the folders are the source of truth.
    manager.index.remove(meta.id)
    assert manager.get(meta.id) is None

    found = manager.reindex()
    assert found == 1
    restored = manager.get(meta.id)
    assert restored is not None
    assert restored.word_count == 6


def test_open_returns_everything(manager):
    meta, store = manager.create(title="Full Session", course="BIO")
    store.append_segment(segment(1, "Hello world."))
    store.save_markers([Marker(time=5.0)])
    state = NoteState(summary="Summary here")
    store.save_note_state(state)
    store.save_final_notes("# Final\n\nBody.")
    store.close()
    manager.reindex()

    loaded = manager.open(meta.id)
    assert loaded is not None
    assert loaded.transcript_text == "Hello world."
    assert loaded.markers[0].time == 5.0
    assert loaded.notes.summary == "Summary here"
    assert "# Final" in loaded.final_notes


def test_delete_removes_folder_and_index_row(manager):
    meta, store = manager.create(title="Delete Me")
    store.close()
    folder = meta.folder
    assert manager.delete(meta.id)
    assert not folder.exists()
    assert manager.get(meta.id) is None


# -- recovery ------------------------------------------------------------


def test_interrupted_session_is_detected(manager):
    meta, store = manager.create(title="Interrupted Lecture")
    store.append_segment(segment(1, "Words captured before the crash."))
    store.close()  # no finalization: status stays RECORDING

    recoverable = recovery.find_recoverable(manager)
    assert len(recoverable) == 1
    assert recoverable[0].meta.id == meta.id
    assert recoverable[0].segment_count == 1
    assert not recoverable[0].is_empty


def test_recover_closes_the_session_and_keeps_content(manager):
    meta, store = manager.create(title="Interrupted Lecture")
    for index in range(1, 4):
        store.append_segment(segment(index, f"Segment {index} content words here."))
    store.close()

    recovered = recovery.recover(manager, meta)
    assert recovered.status is SessionStatus.NEEDS_FINALIZATION
    assert recovered.segment_count == 3
    assert recovered.word_count > 0
    assert recovered.duration_seconds >= 14.0
    assert recovery.find_recoverable(manager) == []

    loaded = manager.open(meta.id)
    assert len(loaded.segments) == 3
    assert loaded.store.transcript_md.exists()


def test_resume_continues_ids_and_timestamps(manager):
    meta, store = manager.create(title="Resumable")
    store.append_segment(segment(1, "First half."))
    store.append_segment(segment(2, "Still first half."))
    store.close()

    resumed_meta, resumed_store = recovery.prepare_resume(manager, meta)
    next_id, offset = recovery.resume_offsets(resumed_store)
    assert resumed_meta.status is SessionStatus.RECORDING
    assert next_id == 3
    assert offset == 14.0
    resumed_store.close()


def test_discard_deletes_an_interrupted_session(manager):
    meta, store = manager.create(title="Throwaway")
    store.close()
    assert recovery.discard(manager, meta)
    assert manager.get(meta.id) is None


def test_recovery_drops_index_rows_for_missing_folders(manager):
    import shutil

    meta, store = manager.create(title="Vanished")
    store.close()
    shutil.rmtree(meta.folder)
    assert recovery.find_recoverable(manager) == []
    assert manager.get(meta.id) is None


# -- export --------------------------------------------------------------


def _finished_session(manager):
    meta, store = manager.create(title="Export Session", course="BIO 113")
    store.append_segment(segment(1, "The cell membrane is a phospholipid bilayer."))
    store.append_segment(segment(2, "Gram positive bacteria have thick walls."))
    store.save_markers([Marker(time=8.0, kind=MarkerKind.NOTE, text="On Exam 1")])
    state = NoteState(summary="Bacterial structure.")
    state.add_bullets("key_points", [NoteItem(text="Membranes are bilayers", starred=True)])
    store.save_note_state(state)
    store.save_final_notes("# Cell Structure\n\n## Executive Summary\n\nAll about cells.")
    store.close()
    manager.reindex()
    return manager.open(meta.id)


def test_markdown_export_contains_real_session_data(manager):
    session = _finished_session(manager)
    path = export_session(session, format_id="markdown")
    content = path.read_text(encoding="utf-8")

    assert path.suffix == ".md"
    assert "# Export Session" in content
    assert "BIO 113" in content
    assert "Executive Summary" in content
    assert "phospholipid bilayer" in content
    assert "On Exam 1" in content
    assert "`00:00:05`" in content


def test_markdown_export_falls_back_to_live_notes(manager):
    meta, store = manager.create(title="No Final Notes")
    store.append_segment(segment(1, "Only live notes exist."))
    state = NoteState(summary="Live only.")
    state.add_bullets("key_points", [NoteItem(text="A live point")])
    store.save_note_state(state)
    store.close()
    manager.reindex()

    content = export_session(manager.open(meta.id), format_id="markdown").read_text()
    assert "Live Notes" in content
    assert "A live point" in content


def test_text_and_json_exports(manager):
    session = _finished_session(manager)

    text = get_exporter("text").render(session)
    assert "EXPORT SESSION" in text
    assert "phospholipid bilayer" in text
    assert "#" not in text.split("TRANSCRIPT")[0].replace("----", "")

    payload = json.loads(get_exporter("json").render(session))
    assert payload["session"]["title"] == "Export Session"
    assert len(payload["transcript"]) == 2
    assert payload["markers"][0]["text"] == "On Exam 1"
    assert payload["notes"]["summary"] == "Bacterial structure."


def test_export_to_explicit_directory(manager, tmp_path):
    session = _finished_session(manager)
    destination = tmp_path / "out"
    destination.mkdir()
    path = export_session(session, format_id="markdown", destination=destination)
    assert path.parent == destination
    assert path.name.endswith(".md")


def test_retention_prunes_old_recordings(manager, config):
    from datetime import timedelta

    meta, store = manager.create(title="Old Recording")
    store.audio_wav.write_bytes(b"\x00" * 100)
    meta.has_audio = True
    meta.created_at = utcnow() - timedelta(days=40)
    store.save_meta(meta)
    # created_at is immutable once indexed, so re-insert the row as if the
    # session had genuinely been recorded 40 days ago.
    manager.index.remove(meta.id)
    manager.index.upsert(meta)
    store.close()

    assert manager.prune_recordings(days=0) == 0  # disabled
    assert manager.prune_recordings(days=30) == 1
    assert not store.audio_wav.exists()
