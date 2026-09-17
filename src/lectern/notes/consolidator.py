"""Periodic note consolidation.

Live updates optimise for latency, so over an hour the notes accumulate
near-duplicates and drift out of order. Every few minutes a background pass
asks the model to reorganise the *notes* — never the raw transcript, which
stays on disk as the permanent record — and the result replaces the state.

Consolidation is strictly a safety-checked operation: if the rewritten notes
have lost material, the original is kept. A tidier set of notes is worth
nothing if it silently dropped a formula.
"""

from __future__ import annotations

from dataclasses import dataclass

from lectern.llm.base import LLMBackend, LLMError, LLMUnavailableError
from lectern.llm.parsing import ResponseParseError, as_dict_list, as_str_list, extract_json_object
from lectern.llm.prompts import NOTE_CONSOLIDATE_SCHEMA, NOTE_SYSTEM_PROMPT, build_consolidate_prompt
from lectern.logging_setup import get_logger
from lectern.notes.models import BULLET_FIELDS, TERM_FIELDS, NoteItem, NoteState, TermEntry

log = get_logger("notes.consolidator")

#: Reject a consolidation that drops more than this fraction of the items, or
#: that loses any starred item. Small local models occasionally answer a
#: reorganisation request with a two-line summary.
MIN_RETAINED_FRACTION = 0.55


@dataclass(slots=True)
class ConsolidationResult:
    state: NoteState
    applied: bool = False
    reason: str = ""
    before_items: int = 0
    after_items: int = 0


def build_consolidated_state(previous: NoteState, payload: dict) -> NoteState:
    """Construct a fresh ``NoteState`` from a consolidation response.

    Timestamps and manual (user-authored) items are carried over from the
    previous state where the text still matches, so consolidating never loses
    the timeline or demotes a note the student wrote themselves.
    """
    previous_items: dict[str, NoteItem] = {}
    for name in BULLET_FIELDS:
        for item in getattr(previous, name):
            previous_items.setdefault(item.key, item)

    state = NoteState(
        current_topic=str(payload.get("current_topic", "") or previous.current_topic).strip(),
        summary=str(payload.get("summary", "") or previous.summary).strip(),
        revision=previous.revision + 1,
    )

    topics = as_str_list(payload.get("topics")) or list(previous.topics)
    for topic in topics:
        state.add_topic(topic)

    for name in BULLET_FIELDS:
        items: list[NoteItem] = []
        for raw in as_dict_list(payload.get(name)):
            text = str(raw.get("text", "")).strip()
            if not text:
                continue
            item = NoteItem(
                text=text,
                topic=str(raw.get("topic", "")).strip(),
                starred=bool(raw.get("starred", False)),
            )
            original = previous_items.get(item.key)
            if original is not None:
                item.timestamp = original.timestamp
                item.source = original.source
                item.starred = item.starred or original.starred
            items.append(item)
        state.add_bullets(name, items)

    previous_terms: dict[str, TermEntry] = {}
    for name in TERM_FIELDS:
        for entry in getattr(previous, name):
            previous_terms.setdefault(entry.key, entry)

    for name in TERM_FIELDS:
        entries: list[TermEntry] = []
        for raw in as_dict_list(payload.get(name)):
            term = str(raw.get("term", "")).strip()
            if not term:
                continue
            entry = TermEntry(
                term=term,
                definition=str(raw.get("definition", "")).strip(),
                starred=bool(raw.get("starred", False)),
            )
            original = previous_terms.get(entry.key)
            if original is not None:
                entry.timestamp = original.timestamp
                entry.starred = entry.starred or original.starred
                entry.definition = entry.definition or original.definition
            entries.append(entry)
        state.add_terms(name, entries)

    # Markers and manual notes live in the timeline, which the model never sees
    # and therefore cannot damage.
    state.timeline = list(previous.timeline)
    return state


def is_safe_consolidation(previous: NoteState, candidate: NoteState) -> tuple[bool, str]:
    """Guard against a consolidation that threw information away."""
    before, after = previous.item_count, candidate.item_count
    if before and after < before * MIN_RETAINED_FRACTION:
        return False, f"would drop {before - after} of {before} items"

    def starred_keys(state: NoteState) -> set[str]:
        keys: set[str] = set()
        for name in BULLET_FIELDS:
            keys |= {item.key for item in getattr(state, name) if item.starred}
        for name in TERM_FIELDS:
            keys |= {entry.key for entry in getattr(state, name) if entry.starred}
        return keys

    lost_starred = starred_keys(previous) - starred_keys(candidate)
    if lost_starred:
        return False, f"would drop {len(lost_starred)} emphasized item(s)"

    user_items = {
        item.key
        for name in BULLET_FIELDS
        for item in getattr(previous, name)
        if item.source == "user"
    }
    surviving = {
        item.key for name in BULLET_FIELDS for item in getattr(candidate, name)
    }
    if user_items - surviving:
        return False, "would drop one of your own notes"

    return True, ""


class NoteConsolidator:
    """Runs the periodic clean-up pass."""

    def __init__(self, backend: LLMBackend, *, model: str, num_ctx: int = 8192) -> None:
        self.backend = backend
        self.model = model
        self.num_ctx = num_ctx

    async def consolidate(
        self,
        state: NoteState,
        *,
        session_title: str,
        max_context_words: int = 1600,
    ) -> ConsolidationResult:
        if state.item_count < 8:
            return ConsolidationResult(state=state, reason="not enough material yet")

        prompt = build_consolidate_prompt(
            note_digest=state.context_digest(max_words=max_context_words),
            session_title=session_title,
        )
        try:
            raw = await self.backend.generate(
                prompt,
                model=self.model,
                system=NOTE_SYSTEM_PROMPT,
                json_schema=NOTE_CONSOLIDATE_SCHEMA,
                temperature=0.1,
                num_ctx=self.num_ctx,
            )
        except LLMUnavailableError:
            raise
        except LLMError as exc:
            log.warning("consolidation failed: %s", exc)
            return ConsolidationResult(state=state, reason=str(exc))

        try:
            payload = extract_json_object(raw)
        except ResponseParseError as exc:
            log.warning("unparseable consolidation (%s); keeping live notes", exc)
            return ConsolidationResult(state=state, reason="model returned unusable output")

        candidate = build_consolidated_state(state, payload)
        safe, reason = is_safe_consolidation(state, candidate)
        if not safe:
            log.warning("rejected consolidation: %s", reason)
            return ConsolidationResult(
                state=state,
                reason=reason,
                before_items=state.item_count,
                after_items=candidate.item_count,
            )

        log.info("consolidated notes: %d -> %d items", state.item_count, candidate.item_count)
        return ConsolidationResult(
            state=candidate,
            applied=True,
            before_items=state.item_count,
            after_items=candidate.item_count,
        )
