"""Parsing whisper.cpp and Ollama responses, including the messy real-world shapes."""

from __future__ import annotations

import json

import pytest

from lectern.llm.parsing import (
    ResponseParseError,
    as_dict_list,
    extract_json_object,
    strip_markdown_fence,
    strip_reasoning,
)
from lectern.transcription.base import TranscriptionError
from lectern.transcription.whisper_cpp import parse_whisper_response


def test_plain_json_object():
    assert extract_json_object('{"a": 1}') == {"a": 1}


def test_json_inside_a_markdown_fence():
    assert extract_json_object('```json\n{"a": 1}\n```') == {"a": 1}


def test_json_after_reasoning_block():
    raw = '<think>Let me consider the transcript…</think>\n{"summary": "ok"}'
    assert extract_json_object(raw) == {"summary": "ok"}


def test_json_after_a_chatty_preamble():
    raw = 'Sure! Here are the notes:\n{"summary": "ok"}\nLet me know if you need more.'
    assert extract_json_object(raw) == {"summary": "ok"}


def test_braces_inside_strings_do_not_confuse_the_scanner():
    payload = {"text": "a } brace and a \\\" quote"}
    raw = f"noise {json.dumps(payload)} trailing"
    assert extract_json_object(raw)["text"] == payload["text"]


def test_truncated_response_raises_rather_than_returning_partial():
    with pytest.raises(ResponseParseError):
        extract_json_object('{"summary": "half a resp')


def test_empty_response_raises():
    with pytest.raises(ResponseParseError):
        extract_json_object("")


def test_unclosed_reasoning_block_yields_nothing():
    assert strip_reasoning("<think>still thinking when the stream was cut") == ""


def test_strip_markdown_fence_unwraps_documents():
    assert strip_markdown_fence("```markdown\n# Title\n\nBody\n```") == "# Title\n\nBody"
    assert strip_markdown_fence("# Already plain") == "# Already plain"


def test_as_dict_list_promotes_strings():
    assert as_dict_list(["hello"])[0]["text"] == "hello"
    assert as_dict_list("not a list") == []


def test_whisper_json_shape():
    assert parse_whisper_response('{"text": " Hello there. "}') == "Hello there."


def test_whisper_verbose_json_shape():
    body = json.dumps({"segments": [{"text": " Part one."}, {"text": " Part two."}]})
    assert parse_whisper_response(body) == "Part one. Part two."


def test_whisper_plain_text_shape():
    assert parse_whisper_response("  Just text  ") == "Just text"


def test_whisper_error_payload_raises():
    with pytest.raises(TranscriptionError):
        parse_whisper_response('{"error": "model not loaded"}')


def test_whisper_empty_response():
    assert parse_whisper_response("") == ""
