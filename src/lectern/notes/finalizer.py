"""Final study-guide synthesis.

Run once when recording stops. Unlike the live updates — which trade depth for
latency — this pass is allowed to take a minute and is expected to produce the
document the student actually revises from.

A three-hour lecture does not fit in any local context window, so long
transcripts go through hierarchical map/reduce: the transcript is split into
overlapping chunks, each is summarised in full detail, and the summaries are
merged (recursively, if there are enough of them). Nothing is truncated — the
first ten minutes of a lecture survive to the final document exactly like the
last ten.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from lectern.llm.base import LLMBackend, LLMError
from lectern.llm.parsing import strip_markdown_fence
from lectern.llm.prompts import (
    FINAL_SYSTEM_PROMPT,
    build_chunk_summary_prompt,
    build_final_prompt,
    build_reduce_prompt,
)
from lectern.logging_setup import get_logger
from lectern.notes.models import NoteState
from lectern.utils.text import word_count

log = get_logger("notes.finalizer")

#: Words of transcript that comfortably fit alongside the prompt scaffolding.
#: Roughly 0.75 words per token, kept conservative so a small model's context
#: is never the thing that fails a 90-minute lecture.
WORDS_PER_CONTEXT_TOKEN = 0.5
CHUNK_OVERLAP_WORDS = 120
MIN_CHUNK_WORDS = 400


@dataclass(slots=True)
class FinalizationProgress:
    """Progress event for the finishing screen."""

    step: str
    detail: str = ""
    index: int = 0
    total: int = 0


@dataclass(slots=True)
class FinalNotesResult:
    markdown: str = ""
    title: str = ""
    error: str = ""
    chunks_used: int = 1
    partial_sections: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.markdown) and not self.error


ProgressCallback = Callable[[FinalizationProgress], None]


def chunk_transcript(text: str, *, max_words: int, overlap: int = CHUNK_OVERLAP_WORDS) -> list[str]:
    """Split a transcript into overlapping word windows.

    The overlap keeps a sentence that straddles a boundary intelligible in both
    chunks, so a definition split across the seam is not lost from both halves.
    """
    words = text.split()
    max_words = max(MIN_CHUNK_WORDS, max_words)
    if len(words) <= max_words:
        return [text.strip()] if text.strip() else []

    overlap = max(0, min(overlap, max_words // 4))
    step = max_words - overlap
    chunks: list[str] = []
    for start in range(0, len(words), step):
        window = words[start : start + max_words]
        if not window:
            break
        chunks.append(" ".join(window))
        if start + max_words >= len(words):
            break
    return chunks


def extract_title(markdown: str, fallback: str) -> str:
    """Pull the ``# Heading`` the model chose as the session title."""
    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            title = stripped[2:].strip()
            if title:
                return title
        if stripped and not stripped.startswith("#"):
            break
    return fallback


class NoteFinalizer:
    """Produces the final Markdown study guide for a completed session."""

    def __init__(self, backend: LLMBackend, *, model: str, num_ctx: int = 8192) -> None:
        self.backend = backend
        self.model = model
        self.num_ctx = num_ctx

    @property
    def chunk_budget_words(self) -> int:
        """Transcript words allowed in a single prompt at this context size."""
        return max(MIN_CHUNK_WORDS, int(self.num_ctx * WORDS_PER_CONTEXT_TOKEN) - 500)

    async def finalize(
        self,
        *,
        state: NoteState,
        transcript: str,
        session_title: str,
        course: str = "",
        duration: str = "",
        markers: str = "",
        on_progress: ProgressCallback | None = None,
    ) -> FinalNotesResult:
        transcript = transcript.strip()
        if not transcript and state.is_empty:
            return FinalNotesResult(error="there is nothing to synthesise — the session has no content")

        budget = self.chunk_budget_words
        try:
            if word_count(transcript) <= budget:
                markdown = await self._single_pass(
                    state=state,
                    transcript=transcript,
                    session_title=session_title,
                    course=course,
                    duration=duration,
                    markers=markers,
                    on_progress=on_progress,
                )
                chunks_used = 1
            else:
                markdown, chunks_used = await self._map_reduce(
                    transcript=transcript,
                    session_title=session_title,
                    course=course,
                    duration=duration,
                    on_progress=on_progress,
                )
        except LLMError as exc:
            log.error("final synthesis failed: %s", exc)
            return FinalNotesResult(error=str(exc))

        markdown = strip_markdown_fence(markdown)
        if not markdown:
            return FinalNotesResult(error="the model returned an empty study guide")

        return FinalNotesResult(
            markdown=markdown,
            title=extract_title(markdown, session_title),
            chunks_used=chunks_used,
        )

    async def _single_pass(
        self,
        *,
        state: NoteState,
        transcript: str,
        session_title: str,
        course: str,
        duration: str,
        markers: str,
        on_progress: ProgressCallback | None,
    ) -> str:
        _emit(on_progress, FinalizationProgress(step="synthesize", detail="Creating final study guide"))
        prompt = build_final_prompt(
            note_digest=state.context_digest(max_words=1200),
            transcript_digest=transcript,
            session_title=session_title,
            course=course,
            duration=duration,
            markers=markers,
        )
        return await self.backend.generate(
            prompt,
            model=self.model,
            system=FINAL_SYSTEM_PROMPT,
            temperature=0.3,
            num_ctx=self.num_ctx,
        )

    async def _map_reduce(
        self,
        *,
        transcript: str,
        session_title: str,
        course: str,
        duration: str,
        on_progress: ProgressCallback | None,
    ) -> tuple[str, int]:
        chunks = chunk_transcript(transcript, max_words=self.chunk_budget_words)
        total = len(chunks)
        log.info("long transcript: reducing %d chunks", total)

        summaries: list[str] = []
        for index, chunk in enumerate(chunks, start=1):
            _emit(
                on_progress,
                FinalizationProgress(
                    step="reduce",
                    detail=f"Summarising part {index} of {total}",
                    index=index,
                    total=total,
                ),
            )
            text = await self.backend.generate(
                build_chunk_summary_prompt(
                    chunk_text=chunk, index=index, total=total, session_title=session_title
                ),
                model=self.model,
                system=FINAL_SYSTEM_PROMPT,
                temperature=0.2,
                num_ctx=self.num_ctx,
            )
            summaries.append(f"### Part {index} of {total}\n\n{strip_markdown_fence(text)}")

        combined = "\n\n".join(summaries)
        # If the summaries themselves overflow, reduce them pairwise until they fit.
        depth = 0
        while word_count(combined) > self.chunk_budget_words and depth < 3:
            depth += 1
            _emit(
                on_progress,
                FinalizationProgress(step="reduce", detail=f"Merging notes (pass {depth + 1})"),
            )
            groups = chunk_transcript(combined, max_words=self.chunk_budget_words, overlap=0)
            merged: list[str] = []
            for index, group in enumerate(groups, start=1):
                text = await self.backend.generate(
                    build_chunk_summary_prompt(
                        chunk_text=group, index=index, total=len(groups), session_title=session_title
                    ),
                    model=self.model,
                    system=FINAL_SYSTEM_PROMPT,
                    temperature=0.2,
                    num_ctx=self.num_ctx,
                )
                merged.append(strip_markdown_fence(text))
            combined = "\n\n".join(merged)

        _emit(on_progress, FinalizationProgress(step="synthesize", detail="Creating final study guide"))
        final = await self.backend.generate(
            build_reduce_prompt(
                summaries=combined,
                session_title=session_title,
                course=course,
                duration=duration,
            ),
            model=self.model,
            system=FINAL_SYSTEM_PROMPT,
            temperature=0.3,
            num_ctx=self.num_ctx,
        )
        return final, total


def _emit(callback: ProgressCallback | None, progress: FinalizationProgress) -> None:
    if callback is not None:
        try:
            callback(progress)
        except Exception as exc:  # noqa: BLE001 - a UI callback must never abort synthesis
            log.debug("progress callback raised: %s", exc)
