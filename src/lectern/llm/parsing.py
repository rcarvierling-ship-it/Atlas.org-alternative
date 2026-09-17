"""Recovering structured data from local-model output.

Even with grammar-constrained decoding, local models wrap their answers in
things the schema does not describe: reasoning models emit ``<think>`` blocks,
instruct models add markdown fences or a friendly sentence before the JSON.
Rather than discard an otherwise good update, we extract the first balanced
JSON object from the response.
"""

from __future__ import annotations

import json
import re
from typing import Any

_THINK_BLOCK = re.compile(r"<(think|thinking|reasoning)>.*?</\1>", re.DOTALL | re.IGNORECASE)
_UNCLOSED_THINK = re.compile(r"^\s*<(think|thinking|reasoning)>.*", re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)


class ResponseParseError(ValueError):
    """The model's response contained no usable JSON object."""


def strip_reasoning(text: str) -> str:
    """Remove ``<think>`` blocks emitted by reasoning models such as qwen3."""
    cleaned = _THINK_BLOCK.sub("", text)
    # A stream cut off mid-thought leaves an unclosed tag and no answer at all.
    if _UNCLOSED_THINK.match(cleaned) and "</" not in cleaned:
        return ""
    return cleaned.strip()


def extract_json_object(text: str) -> dict[str, Any]:
    """Return the first JSON object in ``text``.

    Raises ``ResponseParseError`` when nothing parseable is present, so callers
    can keep the previous valid state instead of committing garbage.
    """
    cleaned = strip_reasoning(text or "")
    if not cleaned:
        raise ResponseParseError("empty response")

    for candidate in _candidates(cleaned):
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
        if isinstance(payload, list) and payload and isinstance(payload[0], dict):
            return payload[0]
    raise ResponseParseError(f"no JSON object found in response: {cleaned[:200]!r}")


def _candidates(text: str) -> list[str]:
    """Progressively less literal interpretations of the response."""
    options = [text.strip()]
    fenced = _FENCE.search(text)
    if fenced:
        options.append(fenced.group(1).strip())
    balanced = _first_balanced_object(text)
    if balanced:
        options.append(balanced)
    return options


def _first_balanced_object(text: str) -> str | None:
    """Scan for the first ``{...}`` with balanced braces, ignoring string bodies."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def strip_markdown_fence(text: str) -> str:
    """Unwrap a whole-document markdown fence, used by the final synthesis."""
    cleaned = strip_reasoning(text or "").strip()
    match = re.fullmatch(r"```(?:markdown|md)?\s*(.*?)\s*```", cleaned, re.DOTALL)
    return match.group(1).strip() if match else cleaned


def as_str_list(value: Any) -> list[str]:
    """Coerce a schema field that should be a list of strings."""
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def as_dict_list(value: Any) -> list[dict[str, Any]]:
    """Coerce a schema field that should be a list of objects.

    Models sometimes emit bare strings where an object was requested; those are
    promoted rather than dropped.
    """
    if not isinstance(value, list):
        return []
    result: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, dict):
            result.append(item)
        elif isinstance(item, str) and item.strip():
            result.append({"text": item.strip(), "term": item.strip()})
    return result
