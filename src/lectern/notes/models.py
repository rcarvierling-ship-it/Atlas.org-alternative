"""The rolling note state.

``NoteState`` is Lectern's working memory. It is *not* regenerated from the
transcript every cycle — the model receives the current state plus only the
speech that arrived since the last update, and returns a delta which is merged
here, deterministically, in Python.

That split is deliberate:

* Merging in code (not in the prompt) means a bad model response can add
  nothing, but can never silently delete a fact the speaker already said.
* De-duplication is loose-matched on normalized text, so the model restating an
  existing bullet in slightly different words does not accumulate noise.
* Every item keeps the session timestamp at which it appeared, which is what
  makes the timeline and "jump to this moment" possible later.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from lectern.utils.text import normalize_for_compare, word_count
from lectern.utils.timefmt import format_clock

#: Categories that hold free-text bullets. Kept as data so merging, rendering
#: and prompt-building never drift out of sync with each other.
BULLET_FIELDS: tuple[str, ...] = (
    "key_points",
    "important_details",
    "examples",
    "formulas",
    "questions",
    "unclear_points",
)

#: Categories holding term/definition pairs.
TERM_FIELDS: tuple[str, ...] = ("definitions", "key_terms")

SECTION_TITLES: dict[str, str] = {
    "key_points": "Key Points",
    "important_details": "Important Details",
    "definitions": "Definitions",
    "key_terms": "Key Terms",
    "examples": "Examples",
    "formulas": "Formulas & Equations",
    "questions": "Questions to Review",
    "unclear_points": "Needs Clarification",
}


@dataclass(slots=True)
class NoteItem:
    """A single bullet, tagged with where and when it came from."""

    text: str
    topic: str = ""
    starred: bool = False
    timestamp: float = 0.0
    source: str = "llm"

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict[str, Any] | str) -> NoteItem:
        if isinstance(data, str):
            return cls(text=data)
        return cls(
            text=str(data.get("text", "")).strip(),
            topic=str(data.get("topic", "")).strip(),
            starred=bool(data.get("starred", False)),
            timestamp=float(data.get("timestamp", 0.0) or 0.0),
            source=str(data.get("source", "llm")),
        )

    @property
    def key(self) -> str:
        return normalize_for_compare(self.text)


@dataclass(slots=True)
class TermEntry:
    """A vocabulary term with its gloss."""

    term: str
    definition: str = ""
    timestamp: float = 0.0
    starred: bool = False

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict[str, Any] | str) -> TermEntry:
        if isinstance(data, str):
            term, _, definition = data.partition("—")
            return cls(term=term.strip(), definition=definition.strip())
        return cls(
            term=str(data.get("term", "")).strip(),
            definition=str(data.get("definition", "")).strip(),
            timestamp=float(data.get("timestamp", 0.0) or 0.0),
            starred=bool(data.get("starred", False)),
        )

    @property
    def key(self) -> str:
        return normalize_for_compare(self.term)


@dataclass(slots=True)
class TimelineEntry:
    """A moment worth navigating to: a topic change, a marker, a manual note."""

    time: float
    label: str
    kind: str = "topic"  # topic | marker | note

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> TimelineEntry:
        return cls(
            time=float(data.get("time", 0.0) or 0.0),
            label=str(data.get("label", "")).strip(),
            kind=str(data.get("kind", "topic")),
        )

    @property
    def clock(self) -> str:
        return format_clock(self.time)


@dataclass
class NoteState:
    """Structured, incrementally updated notes for one session."""

    current_topic: str = ""
    summary: str = ""
    topics: list[str] = field(default_factory=list)
    key_points: list[NoteItem] = field(default_factory=list)
    important_details: list[NoteItem] = field(default_factory=list)
    examples: list[NoteItem] = field(default_factory=list)
    formulas: list[NoteItem] = field(default_factory=list)
    questions: list[NoteItem] = field(default_factory=list)
    unclear_points: list[NoteItem] = field(default_factory=list)
    definitions: list[TermEntry] = field(default_factory=list)
    key_terms: list[TermEntry] = field(default_factory=list)
    timeline: list[TimelineEntry] = field(default_factory=list)
    #: Monotonic counter of applied updates; lets the UI detect real changes.
    revision: int = 0

    # -- derived -----------------------------------------------------------
    @property
    def is_empty(self) -> bool:
        return not any(
            (
                self.summary,
                self.topics,
                *[getattr(self, name) for name in BULLET_FIELDS],
                *[getattr(self, name) for name in TERM_FIELDS],
            )
        )

    @property
    def item_count(self) -> int:
        bullets = sum(len(getattr(self, name)) for name in BULLET_FIELDS)
        terms = sum(len(getattr(self, name)) for name in TERM_FIELDS)
        return bullets + terms

    def approx_words(self) -> int:
        return word_count(self.to_markdown(include_timeline=False))

    # -- mutation ----------------------------------------------------------
    def add_bullets(self, field_name: str, items: list[NoteItem]) -> int:
        """Merge bullets into a category, skipping near-duplicates.

        Returns how many were genuinely new. An incoming duplicate can still
        *upgrade* an existing bullet to starred — that is how a later "this is
        on the exam" remark attaches to a point made earlier.
        """
        target: list[NoteItem] = getattr(self, field_name)
        existing = {item.key: item for item in target}
        added = 0
        for item in items:
            text = item.text.strip()
            if not text:
                continue
            key = normalize_for_compare(text)
            if not key:
                continue
            match = existing.get(key)
            if match is not None:
                if item.starred and not match.starred:
                    match.starred = True
                continue
            target.append(item)
            existing[key] = item
            added += 1
        return added

    def add_terms(self, field_name: str, entries: list[TermEntry]) -> int:
        """Merge term/definition pairs, filling in a gloss that was missing."""
        target: list[TermEntry] = getattr(self, field_name)
        existing = {entry.key: entry for entry in target}
        added = 0
        for entry in entries:
            term = entry.term.strip()
            if not term:
                continue
            key = normalize_for_compare(term)
            match = existing.get(key)
            if match is not None:
                if entry.definition and len(entry.definition) > len(match.definition):
                    match.definition = entry.definition
                if entry.starred:
                    match.starred = True
                continue
            target.append(entry)
            existing[key] = entry
            added += 1
        return added

    def add_topic(self, topic: str, *, timestamp: float = 0.0) -> bool:
        """Record a topic and, if it is new, drop a timeline entry for it."""
        topic = topic.strip()
        if not topic:
            return False
        known = {normalize_for_compare(existing) for existing in self.topics}
        if normalize_for_compare(topic) in known:
            self.current_topic = topic
            return False
        self.topics.append(topic)
        self.current_topic = topic
        self.timeline.append(TimelineEntry(time=timestamp, label=topic, kind="topic"))
        return True

    def add_timeline_entry(self, entry: TimelineEntry) -> None:
        self.timeline.append(entry)
        self.timeline.sort(key=lambda item: item.time)

    # -- serialization -----------------------------------------------------
    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "current_topic": self.current_topic,
            "summary": self.summary,
            "topics": list(self.topics),
            "timeline": [entry.to_json() for entry in self.timeline],
            "revision": self.revision,
        }
        for name in BULLET_FIELDS:
            payload[name] = [item.to_json() for item in getattr(self, name)]
        for name in TERM_FIELDS:
            payload[name] = [entry.to_json() for entry in getattr(self, name)]
        return payload

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> NoteState:
        state = cls(
            current_topic=str(data.get("current_topic", "")),
            summary=str(data.get("summary", "")),
            topics=[str(topic) for topic in data.get("topics", []) if str(topic).strip()],
            revision=int(data.get("revision", 0) or 0),
        )
        for name in BULLET_FIELDS:
            setattr(state, name, [NoteItem.from_json(raw) for raw in data.get(name, []) or []])
        for name in TERM_FIELDS:
            setattr(state, name, [TermEntry.from_json(raw) for raw in data.get(name, []) or []])
        state.timeline = [TimelineEntry.from_json(raw) for raw in data.get("timeline", []) or []]
        return state

    def copy(self) -> NoteState:
        return NoteState.from_json(self.to_json())

    # -- rendering ---------------------------------------------------------
    def to_markdown(self, *, title: str = "", include_timeline: bool = True) -> str:
        """Render the notes as the Markdown a human reads and exports."""
        lines: list[str] = []
        if title:
            lines += [f"# {title}", ""]
        if self.summary:
            lines += ["## Summary", "", self.summary, ""]
        if self.topics:
            lines += ["## Topics", ""]
            lines += [
                f"- {topic}" + ("  *(current)*" if topic == self.current_topic else "")
                for topic in self.topics
            ]
            lines.append("")

        for name in ("key_points", "important_details"):
            items = getattr(self, name)
            if items:
                lines += [f"## {SECTION_TITLES[name]}", ""]
                lines += [_bullet_markdown(item) for item in items]
                lines.append("")

        for name in TERM_FIELDS:
            entries: list[TermEntry] = getattr(self, name)
            if entries:
                lines += [f"## {SECTION_TITLES[name]}", ""]
                for entry in entries:
                    star = "★ " if entry.starred else ""
                    if entry.definition:
                        lines.append(f"- {star}**{entry.term}** — {entry.definition}")
                    else:
                        lines.append(f"- {star}**{entry.term}**")
                lines.append("")

        for name in ("examples", "formulas", "questions", "unclear_points"):
            items = getattr(self, name)
            if items:
                lines += [f"## {SECTION_TITLES[name]}", ""]
                lines += [_bullet_markdown(item) for item in items]
                lines.append("")

        if include_timeline and self.timeline:
            lines += ["## Timeline", ""]
            lines += [f"- `{entry.clock}` {entry.label}" for entry in self.timeline]
            lines.append("")

        return "\n".join(lines).strip() + "\n" if lines else ""

    def context_digest(self, *, max_words: int = 700) -> str:
        """Compact rendering of the state for the *next* update prompt.

        Long lectures would otherwise grow the prompt without bound. The digest
        keeps every heading and the most recent items in each category, and
        signals the truncation so the model knows earlier material exists and
        must not be "helpfully" re-derived.
        """
        lines: list[str] = []
        if self.current_topic:
            lines.append(f"CURRENT TOPIC: {self.current_topic}")
        if self.summary:
            lines.append(f"SUMMARY: {self.summary}")
        if self.topics:
            lines.append("TOPICS SO FAR: " + "; ".join(self.topics[-12:]))

        budget = max_words - word_count("\n".join(lines))
        sections = list(BULLET_FIELDS) + list(TERM_FIELDS)
        # Newest material is the most likely to be continued, so it is kept.
        per_section = max(3, budget // max(1, len(sections) * 12))

        for name in sections:
            entries = getattr(self, name)
            if not entries:
                continue
            shown = entries[-per_section:]
            hidden = len(entries) - len(shown)
            lines.append("")
            lines.append(f"{SECTION_TITLES[name].upper()}:")
            for entry in shown:
                if isinstance(entry, TermEntry):
                    gloss = f" — {entry.definition}" if entry.definition else ""
                    lines.append(f"- {entry.term}{gloss}")
                else:
                    star = "[EMPHASIZED] " if entry.starred else ""
                    lines.append(f"- {star}{entry.text}")
            if hidden > 0:
                lines.append(f"- (+{hidden} earlier item(s) already recorded — do not repeat them)")

        return "\n".join(lines).strip() or "(no notes yet)"


def _bullet_markdown(item: NoteItem) -> str:
    prefix = "★ " if item.starred else ""
    suffix = ""
    if item.source == "user":
        suffix = "  *(your note)*"
    return f"- {prefix}{item.text}{suffix}"
