"""Time formatting helpers used across the TUI, exports and CLI."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone


def format_clock(seconds: float) -> str:
    """Format an offset from session start as ``HH:MM:SS``."""
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_duration(seconds: float | None) -> str:
    """Format a duration in a compact human form (``1h 12m``, ``48m``, ``9s``)."""
    if seconds is None:
        return "—"
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m"
    return f"{secs}s"


def format_relative(moment: datetime, *, now: datetime | None = None) -> str:
    """Format a timestamp as ``Today`` / ``Yesterday`` / ``Aug 24``."""
    now = now or datetime.now(timezone.utc)
    moment = _as_aware(moment)
    now = _as_aware(now)
    today = now.astimezone().date()
    day = moment.astimezone().date()
    delta = (today - day).days
    if delta == 0:
        return "Today"
    if delta == 1:
        return "Yesterday"
    if 0 < delta < 7:
        return moment.astimezone().strftime("%a")
    return moment.astimezone().strftime("%b %-d")


def format_ago(moment: datetime | None, *, now: datetime | None = None) -> str:
    """Format ``moment`` as an elapsed interval such as ``7s ago``."""
    if moment is None:
        return "never"
    now = _as_aware(now or datetime.now(timezone.utc))
    delta: timedelta = now - _as_aware(moment)
    seconds = int(delta.total_seconds())
    if seconds < 0:
        seconds = 0
    if seconds < 60:
        return f"{seconds}s ago"
    if seconds < 3600:
        return f"{seconds // 60}m ago"
    return f"{seconds // 3600}h ago"


def _as_aware(moment: datetime) -> datetime:
    """Treat naive datetimes as UTC so comparisons never raise."""
    if moment.tzinfo is None:
        return moment.replace(tzinfo=timezone.utc)
    return moment


def utcnow() -> datetime:
    """Timezone-aware UTC now (``datetime.utcnow`` is deprecated)."""
    return datetime.now(timezone.utc)
