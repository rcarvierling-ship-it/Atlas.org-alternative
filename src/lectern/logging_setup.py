"""Application logging.

A full-screen TUI owns the terminal, so nothing may be written to stdout or
stderr while the app is running. Every log record goes to a rotating file at
``~/.local/state/lectern/lectern.log`` instead, viewable with ``lectern logs``.
"""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

from lectern.utils import paths

_CONFIGURED = False
_FORMAT = "%(asctime)s %(levelname)-7s %(name)-28s %(message)s"


def setup_logging(*, verbose: bool = False, path: Path | None = None) -> Path:
    """Install the rotating file handler exactly once. Returns the log path."""
    global _CONFIGURED
    log_path = path or paths.log_file()
    if _CONFIGURED:
        return log_path

    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=2_000_000, backupCount=3, encoding="utf-8"
    )
    handler.setFormatter(logging.Formatter(_FORMAT))

    root = logging.getLogger()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    root.addHandler(handler)

    # These libraries are chatty at DEBUG and would drown out our own records.
    for noisy in ("httpx", "httpcore", "asyncio", "markdown_it"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True
    logging.getLogger("lectern").info("logging initialised (verbose=%s)", verbose)
    return log_path


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"lectern.{name}")
