"""Rolling note state, the update scheduler, and the three LLM note passes."""

from lectern.notes.consolidator import (
    ConsolidationResult,
    NoteConsolidator,
    build_consolidated_state,
    is_safe_consolidation,
)
from lectern.notes.finalizer import (
    FinalizationProgress,
    FinalNotesResult,
    NoteFinalizer,
    chunk_transcript,
    extract_title,
)
from lectern.notes.models import (
    BULLET_FIELDS,
    SECTION_TITLES,
    TERM_FIELDS,
    NoteItem,
    NoteState,
    TermEntry,
    TimelineEntry,
)
from lectern.notes.scheduler import NoteScheduler, PendingBatch
from lectern.notes.updater import NoteUpdater, UpdateResult, apply_update_payload

__all__ = [
    "BULLET_FIELDS",
    "SECTION_TITLES",
    "TERM_FIELDS",
    "ConsolidationResult",
    "FinalNotesResult",
    "FinalizationProgress",
    "NoteConsolidator",
    "NoteFinalizer",
    "NoteItem",
    "NoteScheduler",
    "NoteState",
    "NoteUpdater",
    "PendingBatch",
    "TermEntry",
    "TimelineEntry",
    "UpdateResult",
    "apply_update_payload",
    "build_consolidated_state",
    "chunk_transcript",
    "extract_title",
    "is_safe_consolidation",
]
