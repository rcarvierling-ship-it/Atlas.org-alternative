"""Prompts and JSON schemas for note generation.

All three note tasks (incremental update, consolidation, final synthesis) share
one set of rules, defined once in ``NOTE_RULES``. The rules exist to keep a
small local model honest: its job is to *organise what was said*, never to fill
gaps with plausible-sounding lecture material.

Schemas are passed to Ollama's ``format`` parameter, which constrains decoding
to valid JSON of that shape. That is what makes structured notes workable on an
8B model — without it, small models drift out of JSON several times an hour and
every one of those updates would be a dropped cycle.
"""

from __future__ import annotations

from typing import Any

NOTE_RULES = """\
RULES — follow all of them:
- Only record information that is actually present in the transcript. Never add
  outside knowledge, examples or explanations of your own.
- Preserve exact numbers, dates, units, formulas, names and technical vocabulary.
- If the speaker signals emphasis ("this will be on the exam", "remember this",
  "the key point is"), set "starred": true on that item.
- Record a definition when the speaker defines a term. Record an example
  separately from a definition — an illustration is not a definition.
- Group related information under the topic it belongs to.
- Remove filler, false starts and side chatter. Do not over-summarize technical
  material: detail that a student would be tested on must survive.
- The transcript comes from speech recognition and contains errors. Fix obvious
  formatting slips (capitalisation, homophones that context makes certain).
- If a passage seems garbled or a term looks mis-transcribed, put it under
  "unclear_points" describing what was unclear. NEVER invent a replacement and
  never present a guess as fact.
- Write bullets as compact, complete statements a student can revise from.
"""

_ITEM_SCHEMA: dict[str, Any] = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "topic": {"type": "string"},
            "starred": {"type": "boolean"},
        },
        "required": ["text"],
    },
}

_TERM_SCHEMA: dict[str, Any] = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "term": {"type": "string"},
            "definition": {"type": "string"},
            "starred": {"type": "boolean"},
        },
        "required": ["term"],
    },
}

#: Delta returned by an incremental update. Every list is *new material only*.
NOTE_UPDATE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "current_topic": {"type": "string"},
        "summary": {"type": "string"},
        "new_topics": {"type": "array", "items": {"type": "string"}},
        "key_points": _ITEM_SCHEMA,
        "important_details": _ITEM_SCHEMA,
        "examples": _ITEM_SCHEMA,
        "formulas": _ITEM_SCHEMA,
        "questions": _ITEM_SCHEMA,
        "unclear_points": _ITEM_SCHEMA,
        "definitions": _TERM_SCHEMA,
        "key_terms": _TERM_SCHEMA,
    },
    "required": ["current_topic", "summary"],
}

#: Consolidation returns a complete replacement state, not a delta.
NOTE_CONSOLIDATE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "current_topic": {"type": "string"},
        "summary": {"type": "string"},
        "topics": {"type": "array", "items": {"type": "string"}},
        "key_points": _ITEM_SCHEMA,
        "important_details": _ITEM_SCHEMA,
        "examples": _ITEM_SCHEMA,
        "formulas": _ITEM_SCHEMA,
        "questions": _ITEM_SCHEMA,
        "unclear_points": _ITEM_SCHEMA,
        "definitions": _TERM_SCHEMA,
        "key_terms": _TERM_SCHEMA,
    },
    "required": ["summary"],
}

NOTE_SYSTEM_PROMPT = f"""\
You are the note-taking engine inside Lectern, a local lecture assistant. You \
turn live speech-to-text into structured study notes for the student who is \
listening.

{NOTE_RULES}
Respond with JSON only. No prose, no markdown fences, no commentary."""

FINAL_SYSTEM_PROMPT = f"""\
You are writing the final study guide for a lecture that has just finished. \
This is the artefact the student will revise from, so it must be more careful, \
better organised and more complete than the live notes were.

{NOTE_RULES}
Write clean GitHub-flavoured Markdown. Do not wrap the document in a code fence."""


def build_update_prompt(
    *,
    note_digest: str,
    new_transcript: str,
    session_title: str,
    course: str = "",
    markers: str = "",
    elapsed: str = "",
) -> str:
    """Prompt for one incremental note update.

    The model sees a *digest* of existing notes (so it can avoid repeating
    itself and can continue the current topic) plus only the speech that has
    arrived since the last update — never the whole transcript again.
    """
    header = f"LECTURE: {session_title}"
    if course:
        header += f"  ({course})"
    if elapsed:
        header += f"\nELAPSED: {elapsed}"

    marker_block = ""
    if markers:
        marker_block = (
            "\nTHE STUDENT FLAGGED THESE MOMENTS AS IMPORTANT — make sure the related "
            f"material is captured and starred:\n{markers}\n"
        )

    return f"""\
{header}

NOTES SO FAR (already recorded — do NOT repeat any of it):
{note_digest}
{marker_block}
NEW TRANSCRIPT SINCE THE LAST UPDATE:
\"\"\"
{new_transcript}
\"\"\"

Extract ONLY the new information contained in the new transcript above.
Return a JSON object with these keys:
- "current_topic": what is being discussed right now (short phrase).
- "summary": a rewritten 1-3 sentence summary of the lecture so far.
- "new_topics": topics that started in this new transcript and are not in the notes yet.
- "key_points", "important_details", "examples", "formulas", "questions",
  "unclear_points": arrays of NEW items only, each {{"text": ..., "topic": ..., "starred": bool}}.
- "definitions", "key_terms": arrays of NEW entries, each {{"term": ..., "definition": ..., "starred": bool}}.

Use empty arrays for categories with nothing new. If the new transcript contains
no substantive content (filler, small talk, silence), return empty arrays and
leave "summary" unchanged."""


def build_consolidate_prompt(*, note_digest: str, session_title: str) -> str:
    """Prompt for periodic clean-up of accumulated notes.

    Deliberately transcript-free: the note state is the working memory, and
    re-sending the raw transcript every few minutes is exactly the unbounded
    context growth this design avoids.
    """
    return f"""\
LECTURE: {session_title}

These are the accumulated live notes. Reorganise them into a cleaner set.

{note_digest}

Rewrite the notes as a single JSON object with the same categories
("current_topic", "summary", "topics", "key_points", "important_details",
"examples", "formulas", "questions", "unclear_points", "definitions", "key_terms").

Requirements:
- Merge bullets that say the same thing; keep the clearest wording.
- Preserve every unique fact, number, formula, name and definition. Losing
  information is the one unacceptable outcome.
- Keep every item that is starred, and keep it starred.
- Group items under the topic they belong to and order topics as they occurred.
- Do not add anything that is not already in the notes above."""


def build_final_prompt(
    *,
    note_digest: str,
    transcript_digest: str,
    session_title: str,
    course: str = "",
    duration: str = "",
    markers: str = "",
) -> str:
    """Prompt for the final study guide."""
    meta = f"LECTURE: {session_title}"
    if course:
        meta += f"\nCOURSE: {course}"
    if duration:
        meta += f"\nDURATION: {duration}"
    marker_block = f"\n\nMOMENTS THE STUDENT FLAGGED AS IMPORTANT:\n{markers}" if markers else ""

    return f"""\
{meta}{marker_block}

STRUCTURED NOTES TAKEN DURING THE LECTURE:
{note_digest}

TRANSCRIPT:
\"\"\"
{transcript_digest}
\"\"\"

Write the final study guide in Markdown with exactly these sections, in this order:

# <a specific, descriptive session title>

## Executive Summary
## Main Topics
(one "### <topic>" subsection per topic, each with detailed organised notes)
## Key Concepts
## Definitions
## Important Details
## Examples
## Formulas / Equations
## Questions to Review
## Needs Clarification
## Exam / Quiz Worthy Material
## Key Takeaways

Omit a section entirely if the lecture genuinely contained nothing for it.
Be thorough: this replaces the student's own notes."""


def build_chunk_summary_prompt(*, chunk_text: str, index: int, total: int, session_title: str) -> str:
    """Prompt for one chunk in the hierarchical reduction of a long lecture.

    Long lectures exceed any local context window, so the transcript is reduced
    chunk-by-chunk and the summaries are then synthesised. Nothing is truncated
    away — the beginning of the lecture matters as much as the end.
    """
    return f"""\
LECTURE: {session_title}
This is part {index} of {total} of a lecture transcript.

\"\"\"
{chunk_text}
\"\"\"

Write detailed Markdown notes covering ONLY this part. Preserve all numbers,
dates, formulas, definitions, examples and technical terms. Mark anything the
speaker emphasised with a leading "★". Do not add information that is not in
this text. Do not write an introduction or conclusion — just the notes."""


def build_reduce_prompt(
    *,
    summaries: str,
    session_title: str,
    course: str = "",
    duration: str = "",
    note_digest: str = "",
    markers: str = "",
) -> str:
    """Prompt that merges per-chunk notes into one final study guide.

    The chunk summaries are derived from the transcript only, so the notes
    taken during the lecture are supplied alongside them: a note the student
    typed was never spoken, and would otherwise not exist anywhere in this
    prompt.
    """
    meta = f"LECTURE: {session_title}"
    if course:
        meta += f"\nCOURSE: {course}"
    if duration:
        meta += f"\nDURATION: {duration}"

    extra = ""
    if note_digest:
        extra += (
            "\n\nNOTES TAKEN DURING THE LECTURE (including notes the student typed "
            "themselves — these may not appear in the transcript, and must not be "
            f"dropped):\n{note_digest}"
        )
    if markers:
        extra += (
            "\n\nMOMENTS THE STUDENT FLAGGED AS IMPORTANT — make sure the related "
            f"material appears and is marked as significant:\n{markers}"
        )

    return f"""\
{meta}{extra}

These are sequential notes covering the whole lecture, part by part:

{summaries}

Combine them into one coherent study guide using exactly these sections:

# <a specific, descriptive session title>

## Executive Summary
## Main Topics
(one "### <topic>" subsection per topic, each with detailed organised notes)
## Key Concepts
## Definitions
## Important Details
## Examples
## Formulas / Equations
## Questions to Review
## Needs Clarification
## Exam / Quiz Worthy Material
## Key Takeaways

Merge duplicates across parts, keep every unique fact, and preserve the order in
which topics were taught. Omit a section only if nothing in the lecture fits it."""
