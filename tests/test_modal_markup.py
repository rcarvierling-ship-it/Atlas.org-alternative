"""Regression tests for modal rendering of runtime diagnostics."""

from lectern.screens.modals import _literal


def test_literal_runtime_text_preserves_bracketed_paths_without_markup_parsing():
    message = (
        "Whisper failed. model = "
        "[/Users/reesevierling/.local/share/lectern/whisper-models/ggml-small.en.bin]"
    )

    rendered = _literal(message)

    assert rendered.plain == message
