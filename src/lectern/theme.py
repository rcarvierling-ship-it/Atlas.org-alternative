"""Visual language.

A dark, low-chroma palette: charcoal surfaces, one indigo accent for focus and
identity, cyan for live activity, and colour reserved for meaning — green is
healthy, amber is working, red is only ever a real error.

Icons are defined here as (unicode, ascii) pairs so a terminal with poor glyph
coverage can fall back without touching any widget code
(``ui.ascii_icons = true``).
"""

from __future__ import annotations

from textual.theme import Theme

BACKGROUND = "#0e1013"
SURFACE = "#16181d"
PANEL = "#1c1f26"
BORDER = "#2b303a"
BORDER_FOCUS = "#7c7cff"
TEXT = "#e6e8ec"
TEXT_MUTED = "#8b919e"
TEXT_FAINT = "#5f6672"
ACCENT = "#7c7cff"
CYAN = "#56d4dd"
GREEN = "#4ade80"
AMBER = "#fbbf24"
RED = "#f87171"

LECTERN_DARK = Theme(
    name="lectern-dark",
    primary=ACCENT,
    secondary=CYAN,
    accent=CYAN,
    foreground=TEXT,
    background=BACKGROUND,
    surface=SURFACE,
    panel=PANEL,
    success=GREEN,
    warning=AMBER,
    error=RED,
    dark=True,
    variables={
        "border": BORDER,
        "border-blurred": BORDER,
        "text-muted": TEXT_MUTED,
        "text-faint": TEXT_FAINT,
        "block-cursor-background": ACCENT,
        "block-cursor-foreground": BACKGROUND,
        "block-cursor-text-style": "none",
        "footer-key-foreground": ACCENT,
        "footer-description-foreground": TEXT_MUTED,
        "footer-background": BACKGROUND,
        "footer-key-background": BACKGROUND,
        "footer-item-background": BACKGROUND,
        "input-selection-background": f"{ACCENT} 35%",
        "scrollbar": PANEL,
        "scrollbar-hover": BORDER,
        "scrollbar-active": ACCENT,
        "markdown-h1-color": TEXT,
        "markdown-h2-color": ACCENT,
    },
)

#: A lighter alternative for bright terminals and projectors.
LECTERN_LIGHT = Theme(
    name="lectern-light",
    primary="#5b5bd6",
    secondary="#0e7490",
    accent="#0e7490",
    foreground="#1f2329",
    background="#fbfbfd",
    surface="#f2f3f7",
    panel="#e9ebf1",
    success="#15803d",
    warning="#b45309",
    error="#b91c1c",
    dark=False,
    variables={
        "border": "#d3d7e0",
        "text-muted": "#5b6270",
        "text-faint": "#8b919e",
        "footer-key-foreground": "#5b5bd6",
    },
)

THEMES = (LECTERN_DARK, LECTERN_LIGHT)


class Icons:
    """Glyphs with ASCII fallbacks, selected once at startup."""

    _PAIRS: dict[str, tuple[str, str]] = {
        "record": ("●", "*"),
        "paused": ("❙❙", "||"),
        "stopped": ("■", "#"),
        "star": ("★", "*"),
        "note": ("✎", "~"),
        "topic": ("▸", ">"),
        "check": ("✓", "+"),
        "cross": ("✗", "x"),
        "warn": ("!", "!"),
        "unknown": ("?", "?"),
        "spinner": ("◌", "o"),
        "down": ("↓", "v"),
        "bullet": ("•", "-"),
        "dot": ("·", "."),
        "dash": ("—", "--"),
        "arrow": ("→", "->"),
        "enter": ("⏎", "<-"),
        "live": ("◉", "@"),
    }

    def __init__(self, ascii_only: bool = False) -> None:
        self.ascii_only = ascii_only

    def __getattr__(self, name: str) -> str:
        try:
            unicode_glyph, ascii_glyph = self._PAIRS[name]
        except KeyError:
            raise AttributeError(name) from None
        return ascii_glyph if self.ascii_only else unicode_glyph


ICONS = Icons()


def configure_icons(ascii_only: bool) -> Icons:
    """Switch the shared icon set. Called once when config loads."""
    ICONS.ascii_only = ascii_only
    return ICONS
