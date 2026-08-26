"""Text helpers for slugs, word counts and de-duplication."""

from __future__ import annotations

import re
import unicodedata

_SLUG_STRIP = re.compile(r"[^a-z0-9]+")
_WORD = re.compile(r"[\w'’-]+", re.UNICODE)

# Whisper emits these when fed silence or music. They are filtered before a
# segment is ever considered "speech".
HALLUCINATION_PATTERNS: tuple[str, ...] = (
    "thanks for watching",
    "thank you for watching",
    "subscribe to my channel",
    "please subscribe",
    "like and subscribe",
    "see you in the next video",
    "you're watching",
    "transcription by",
    "subtitles by",
    "amara.org",
    "www.",
    "beadaz",
    "sous-titres",
)


def slugify(value: str, *, max_length: int = 60) -> str:
    """Convert arbitrary text to a filesystem-safe slug."""
    normalized = unicodedata.normalize("NFKD", value)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii").lower()
    slug = _SLUG_STRIP.sub("-", ascii_text).strip("-")
    if len(slug) > max_length:
        slug = slug[:max_length].rstrip("-")
    return slug or "session"


def word_count(text: str) -> int:
    """Count words the way a human would skim them."""
    return len(_WORD.findall(text))


def truncate(text: str, limit: int, *, suffix: str = "…") -> str:
    """Truncate on a word boundary when possible."""
    text = text.strip()
    if len(text) <= limit:
        return text
    cut = text[: max(0, limit - len(suffix))]
    if " " in cut:
        cut = cut[: cut.rfind(" ")]
    return cut.rstrip() + suffix


def normalize_for_compare(text: str) -> str:
    """Aggressively normalize a line so near-duplicate bullets collapse."""
    lowered = unicodedata.normalize("NFKD", text).lower()
    stripped = re.sub(r"[^a-z0-9 ]+", " ", lowered)
    return " ".join(stripped.split())


def looks_like_hallucination(text: str) -> bool:
    """Detect the canned phrases whisper.cpp emits for silence and music.

    Whisper's training data is full of YouTube captions, so near-silent audio
    reliably decodes to "Thanks for watching!" and friends. Those segments must
    never reach the transcript, because from there they would poison the notes.
    """
    cleaned = normalize_for_compare(text)
    if not cleaned:
        return True
    # Bare punctuation / musical cues.
    if re.fullmatch(r"[\W_]+", text.strip()):
        return True
    if re.fullmatch(r"\[?\(?\s*(music|silence|applause|blank[ _]audio|inaudible)\s*\)?\]?", cleaned):
        return True
    for pattern in HALLUCINATION_PATTERNS:
        if pattern in cleaned:
            return True
    # A single repeated token ("you you you you") is a classic decode loop.
    tokens = cleaned.split()
    if len(tokens) >= 4 and len(set(tokens)) == 1:
        return True
    return False


def dedupe_preserve_order(items: list[str]) -> list[str]:
    """Drop later duplicates (compared loosely) while keeping the first form."""
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        key = normalize_for_compare(item)
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(item.strip())
    return result
