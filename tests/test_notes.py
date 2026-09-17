"""NoteState merge semantics, the update path and consolidation safety."""

from __future__ import annotations

from lectern.notes.consolidator import build_consolidated_state, is_safe_consolidation
from lectern.notes.models import NoteItem, NoteState, TermEntry
from lectern.notes.updater import apply_update_payload


def test_apply_update_adds_items_and_topic():
    state = NoteState()
    result = apply_update_payload(
        state,
        {
            "current_topic": "Cell Structure",
            "summary": "About cells.",
            "key_points": [{"text": "Membranes are bilayers", "starred": True}],
            "definitions": [{"term": "Peptidoglycan", "definition": "Wall polymer"}],
        },
        timestamp=12.0,
    )
    assert result.added_items == 2
    assert result.changed
    assert state.current_topic == "Cell Structure"
    assert state.key_points[0].starred
    assert state.key_points[0].timestamp == 12.0
    assert state.revision == 1
    assert any(entry.kind == "topic" for entry in state.timeline)


def test_duplicate_bullets_are_not_added_twice():
    state = NoteState()
    payload = {"current_topic": "T", "summary": "s", "key_points": [{"text": "The cell wall is thick."}]}
    apply_update_payload(state, payload)
    result = apply_update_payload(
        state,
        {"current_topic": "T", "summary": "s", "key_points": [{"text": "the cell wall is thick"}]},
    )
    assert result.added_items == 0
    assert len(state.key_points) == 1


def test_duplicate_can_upgrade_an_item_to_starred():
    """A later "this is on the exam" attaches to a point already recorded."""
    state = NoteState()
    apply_update_payload(state, {"current_topic": "T", "summary": "", "key_points": [{"text": "Gram stain"}]})
    apply_update_payload(
        state,
        {"current_topic": "T", "summary": "", "key_points": [{"text": "gram stain", "starred": True}]},
    )
    assert len(state.key_points) == 1
    assert state.key_points[0].starred


def test_existing_information_is_never_removed_by_an_update():
    state = NoteState()
    apply_update_payload(state, {"current_topic": "A", "summary": "x", "key_points": [{"text": "First fact"}]})
    apply_update_payload(state, {"current_topic": "B", "summary": "y", "key_points": [{"text": "Second fact"}]})
    texts = [item.text for item in state.key_points]
    assert texts == ["First fact", "Second fact"]
    assert state.topics == ["A", "B"]


def test_definition_gloss_is_filled_in_when_longer():
    state = NoteState()
    state.add_terms("definitions", [TermEntry(term="Enzyme", definition="")])
    state.add_terms("definitions", [TermEntry(term="enzyme", definition="A biological catalyst.")])
    assert len(state.definitions) == 1
    assert state.definitions[0].definition == "A biological catalyst."


def test_empty_payload_marks_no_change():
    state = NoteState(summary="unchanged")
    result = apply_update_payload(state, {"current_topic": "", "summary": ""})
    assert not result.changed
    assert state.revision == 0


def test_string_items_are_promoted_not_dropped():
    """Small models sometimes emit bare strings where objects were requested."""
    state = NoteState()
    result = apply_update_payload(
        state, {"current_topic": "T", "summary": "", "key_points": ["A bare string bullet"]}
    )
    assert result.added_items == 1
    assert state.key_points[0].text == "A bare string bullet"


def test_json_round_trip_preserves_everything(sample_notes):
    restored = NoteState.from_json(sample_notes.to_json())
    assert restored.summary == sample_notes.summary
    assert [item.text for item in restored.key_points] == [
        item.text for item in sample_notes.key_points
    ]
    assert restored.key_points[1].starred
    assert restored.definitions[0].term == "Peptidoglycan"
    assert restored.timeline[0].label == "Cell Structure"


def test_markdown_rendering_marks_emphasis(sample_notes):
    markdown = sample_notes.to_markdown(title="BIO 113")
    assert markdown.startswith("# BIO 113")
    assert "★ Gram-positive walls are thick." in markdown
    assert "**Peptidoglycan** — Polymer in bacterial cell walls." in markdown


def test_context_digest_is_bounded_and_flags_truncation():
    state = NoteState(summary="A long lecture.")
    state.add_bullets(
        "key_points",
        [NoteItem(text=f"Point number {index} with some detail attached") for index in range(80)],
    )
    digest = state.context_digest(max_words=200)
    assert len(digest.split()) < 400
    assert "already recorded" in digest
    # The most recent material must survive, since that is what continues.
    assert "Point number 79" in digest


def test_consolidation_preserves_user_notes_and_timestamps():
    previous = NoteState(summary="old")
    previous.add_bullets(
        "key_points",
        [
            NoteItem(text="Model bullet", timestamp=10.0),
            NoteItem(text="My own note", timestamp=20.0, source="user", starred=True),
        ],
    )
    payload = {
        "summary": "new",
        "key_points": [{"text": "Model bullet"}, {"text": "My own note"}],
    }
    candidate = build_consolidated_state(previous, payload)
    user_item = next(item for item in candidate.key_points if item.text == "My own note")
    assert user_item.source == "user"
    assert user_item.starred
    assert user_item.timestamp == 20.0


def test_consolidation_rejected_when_it_drops_material():
    previous = NoteState()
    previous.add_bullets("key_points", [NoteItem(text=f"Fact {index}") for index in range(10)])
    candidate = NoteState()
    candidate.add_bullets("key_points", [NoteItem(text="Fact 1")])
    safe, reason = is_safe_consolidation(previous, candidate)
    assert not safe
    assert "drop" in reason


def test_consolidation_rejected_when_it_drops_starred_items():
    previous = NoteState()
    previous.add_bullets(
        "key_points",
        [NoteItem(text=f"Fact {index}") for index in range(9)]
        + [NoteItem(text="Exam material", starred=True)],
    )
    candidate = NoteState()
    candidate.add_bullets("key_points", [NoteItem(text=f"Fact {index}") for index in range(9)])
    safe, reason = is_safe_consolidation(previous, candidate)
    assert not safe
    assert "emphasized" in reason


def test_consolidation_accepted_when_it_merges_duplicates():
    previous = NoteState()
    previous.add_bullets("key_points", [NoteItem(text=f"Fact {index}") for index in range(10)])
    candidate = NoteState()
    candidate.add_bullets("key_points", [NoteItem(text=f"Fact {index}") for index in range(8)])
    safe, reason = is_safe_consolidation(previous, candidate)
    assert safe, reason
