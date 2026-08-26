"""Incremental note updates.

One cycle is: take the current ``NoteState`` plus the transcript that arrived
since the last cycle, ask the model for a *delta*, and merge that delta in
Python. The state is only ever replaced after a complete, parseable response —
a truncated stream or an Ollama outage leaves the previous notes untouched.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from lectern.llm.base import LLMBackend, LLMError, LLMUnavailableError
from lectern.llm.parsing import ResponseParseError, as_dict_list, as_str_list, extract_json_object
from lectern.llm.prompts import NOTE_SYSTEM_PROMPT, NOTE_UPDATE_SCHEMA, build_update_prompt
from lectern.logging_setup import get_logger
from lectern.notes.models import BULLET_FIELDS, TERM_FIELDS, NoteItem, NoteState, TermEntry

log = get_logger("notes.updater")


@dataclass(slots=True)
class UpdateResult:
    """Outcome of one note update cycle."""

    state: NoteState
    added_items: int = 0
    new_topics: list[str] = field(default_factory=list)
    changed: bool = False
    error: str = ""
    duration_ms: float = 0.0

    @property
    def ok(self) -> bool:
        return not self.error


def apply_update_payload(
    state: NoteState,
    payload: dict,
    *,
    timestamp: float = 0.0,
    source: str = "llm",
) -> UpdateResult:
    """Merge a parsed delta into ``state`` in place.

    Pure and synchronous, which is what makes the merge rules directly testable
    without a model in the loop.
    """
    added = 0
    new_topics: list[str] = []
    changed = False

    summary = str(payload.get("summary", "")).strip()
    if summary and summary != state.summary:
        state.summary = summary
        changed = True

    current_topic = str(payload.get("current_topic", "")).strip()

    for topic in as_str_list(payload.get("new_topics")):
        if state.add_topic(topic, timestamp=timestamp):
            new_topics.append(topic)
            changed = True

    if current_topic:
        if state.add_topic(current_topic, timestamp=timestamp):
            new_topics.append(current_topic)
        if state.current_topic != current_topic:
            state.current_topic = current_topic
        changed = True

    for name in BULLET_FIELDS:
        items = [
            NoteItem(
                text=str(raw.get("text", "")).strip(),
                topic=str(raw.get("topic", "") or state.current_topic).strip(),
                starred=bool(raw.get("starred", False)),
                timestamp=timestamp,
                source=source,
            )
            for raw in as_dict_list(payload.get(name))
            if str(raw.get("text", "")).strip()
        ]
        count = state.add_bullets(name, items)
        added += count
        changed = changed or count > 0

    for name in TERM_FIELDS:
        entries = [
            TermEntry(
                term=str(raw.get("term", "")).strip(),
                definition=str(raw.get("definition", "")).strip(),
                starred=bool(raw.get("starred", False)),
                timestamp=timestamp,
            )
            for raw in as_dict_list(payload.get(name))
            if str(raw.get("term", "")).strip()
        ]
        count = state.add_terms(name, entries)
        added += count
        changed = changed or count > 0

    if changed:
        state.revision += 1

    return UpdateResult(state=state, added_items=added, new_topics=new_topics, changed=changed)


class NoteUpdater:
    """Drives one incremental update against a local LLM backend."""

    def __init__(
        self,
        backend: LLMBackend,
        *,
        model: str,
        num_ctx: int = 8192,
        temperature: float = 0.2,
    ) -> None:
        self.backend = backend
        self.model = model
        self.num_ctx = num_ctx
        self.temperature = temperature

    async def update(
        self,
        state: NoteState,
        new_transcript: str,
        *,
        session_title: str,
        course: str = "",
        markers: str = "",
        elapsed: str = "",
        timestamp: float = 0.0,
        max_context_words: int = 900,
        on_token=None,
    ) -> UpdateResult:
        """Run one update cycle and return the merged result.

        On any failure the returned result carries ``error`` and the *original*
        state object, unmodified.
        """
        import time

        transcript = new_transcript.strip()
        if not transcript:
            return UpdateResult(state=state, changed=False)

        prompt = build_update_prompt(
            note_digest=state.context_digest(max_words=max_context_words),
            new_transcript=transcript,
            session_title=session_title,
            course=course,
            markers=markers,
            elapsed=elapsed,
        )

        started = time.monotonic()
        try:
            raw = await self.backend.generate(
                prompt,
                model=self.model,
                system=NOTE_SYSTEM_PROMPT,
                json_schema=NOTE_UPDATE_SCHEMA,
                temperature=self.temperature,
                num_ctx=self.num_ctx,
                on_token=on_token,
            )
        except LLMUnavailableError:
            # Availability is the pipeline's concern: it pauses note generation
            # and tells the user, so this must not be flattened into a result.
            raise
        except LLMError as exc:
            log.warning("note update failed: %s", exc)
            return UpdateResult(state=state, error=str(exc))

        elapsed_ms = (time.monotonic() - started) * 1000.0
        try:
            payload = extract_json_object(raw)
        except ResponseParseError as exc:
            log.warning("unparseable note update (%s); keeping previous notes", exc)
            return UpdateResult(state=state, error="model returned unusable output", duration_ms=elapsed_ms)

        # Merge into a copy so a mid-merge exception cannot leave the live state
        # half-updated; the copy is only published once the merge succeeds.
        working = state.copy()
        result = apply_update_payload(working, payload, timestamp=timestamp)
        result.duration_ms = elapsed_ms
        log.info(
            "note update applied in %.0f ms (+%d items, %d new topics)",
            elapsed_ms,
            result.added_items,
            len(result.new_topics),
        )
        return result
