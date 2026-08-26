"""Modal dialogs.

Every expected failure in Lectern surfaces through one of these rather than a
traceback: a missing permission, an LLM that went away, an accidental keypress
during a recording. They are small, keyboard-first, and always say what will
happen next.
"""

from __future__ import annotations

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Static

from lectern.theme import ICONS


def _literal(value: str) -> Text:
    """Render arbitrary runtime text without interpreting Rich markup tags."""
    return Text(value)


class ConfirmModal(ModalScreen[bool]):
    """Yes/no confirmation. Escape always means no."""

    BINDINGS = [("escape", "dismiss_false", "Cancel")]

    def __init__(
        self,
        message: str,
        *,
        title: str = "Are you sure?",
        confirm_label: str = "Confirm",
        cancel_label: str = "Cancel",
        danger: bool = False,
    ) -> None:
        super().__init__()
        self._message = message
        self._title = title
        self._confirm_label = confirm_label
        self._cancel_label = cancel_label
        self._danger = danger

    def compose(self) -> ComposeResult:
        classes = "modal -danger" if self._danger else "modal"
        with Vertical(classes=classes):
            yield Label(_literal(self._title), classes="modal-title")
            yield Static(_literal(self._message), classes="modal-body")
            with Horizontal(classes="modal-buttons"):
                yield Button(self._cancel_label, id="cancel")
                yield Button(self._confirm_label, id="confirm", classes="-primary")

    def on_mount(self) -> None:
        self.query_one("#confirm", Button).focus()

    @on(Button.Pressed, "#confirm")
    def _confirm(self) -> None:
        self.dismiss(True)

    @on(Button.Pressed, "#cancel")
    def _cancel(self) -> None:
        self.dismiss(False)

    def action_dismiss_false(self) -> None:
        self.dismiss(False)


class TextPromptModal(ModalScreen[str | None]):
    """Single-line input, used for quick notes while recording."""

    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(
        self,
        *,
        title: str = "Add a note",
        placeholder: str = "",
        hint: str = "",
        initial: str = "",
    ) -> None:
        super().__init__()
        self._title = title
        self._placeholder = placeholder
        self._hint = hint
        self._initial = initial

    def compose(self) -> ComposeResult:
        with Vertical(classes="modal"):
            yield Label(_literal(self._title), classes="modal-title")
            yield Input(value=self._initial, placeholder=self._placeholder, id="prompt-input")
            if self._hint:
                yield Static(_literal(self._hint), classes="hint")
            with Horizontal(classes="modal-buttons"):
                yield Button("Cancel", id="cancel")
                yield Button("Save", id="save", classes="-primary")

    def on_mount(self) -> None:
        self.query_one("#prompt-input", Input).focus()

    @on(Input.Submitted, "#prompt-input")
    def _submitted(self, event: Input.Submitted) -> None:
        self._save(event.value)

    @on(Button.Pressed, "#save")
    def _save_pressed(self) -> None:
        self._save(self.query_one("#prompt-input", Input).value)

    def _save(self, value: str) -> None:
        value = value.strip()
        self.dismiss(value or None)

    @on(Button.Pressed, "#cancel")
    def action_cancel(self) -> None:
        self.dismiss(None)


class MessageModal(ModalScreen[None]):
    """Informational dialog with a single dismiss button."""

    BINDINGS = [("escape", "dismiss_modal", "Close"), ("enter", "dismiss_modal", "Close")]

    def __init__(self, message: str, *, title: str = "Notice", severity: str = "information") -> None:
        super().__init__()
        self._message = message
        self._title = title
        self._severity = severity

    def compose(self) -> ComposeResult:
        classes = "modal"
        if self._severity == "error":
            classes += " -danger"
        elif self._severity == "warning":
            classes += " -warning"
        with Vertical(classes=classes):
            yield Label(_literal(self._title), classes="modal-title")
            yield Static(_literal(self._message), classes="modal-body")
            with Horizontal(classes="modal-buttons"):
                yield Button("Close", id="close", classes="-primary")

    def on_mount(self) -> None:
        self.query_one("#close", Button).focus()

    @on(Button.Pressed, "#close")
    def action_dismiss_modal(self) -> None:
        self.dismiss(None)


class PermissionModal(ModalScreen[None]):
    """Explains exactly which macOS permission is missing and how to grant it.

    macOS only applies a newly granted permission to processes launched
afterwards, which is the step users most often miss — so the dialog says so
    explicitly rather than leaving them to rediscover it.
    """

    BINDINGS = [("escape", "dismiss_modal", "Close")]

    def __init__(self, *, permission: str, message: str, remediation: str) -> None:
        super().__init__()
        self._permission = permission
        self._message = message
        self._remediation = remediation

    def compose(self) -> ComposeResult:
        with Vertical(classes="modal -warning"):
            yield Label(_literal(f"{self._permission} permission needed"), classes="modal-title")
            yield Static(_literal(self._message), classes="modal-body")
            yield Static(_literal(self._remediation), id="permission-steps")
            yield Static(
                "macOS only applies a new permission to apps started afterwards, "
                "so quit and reopen your terminal once you've granted it.",
                classes="hint",
            )
            with Horizontal(classes="modal-buttons"):
                yield Button("Close", id="close", classes="-primary")

    def on_mount(self) -> None:
        self.query_one("#close", Button).focus()

    @on(Button.Pressed, "#close")
    def action_dismiss_modal(self) -> None:
        self.dismiss(None)


class RecoveryModal(ModalScreen[str | None]):
    """Offers Resume / Recover / Finalize / Discard for an interrupted session."""

    BINDINGS = [("escape", "later", "Decide later")]

    def __init__(self, session) -> None:  # noqa: ANN001 - RecoverableSession
        super().__init__()
        self._session = session

    def compose(self) -> ComposeResult:
        meta = self._session.meta
        with Vertical(classes="modal -warning"):
            yield Label("Unfinished session found", classes="modal-title")
            yield Static(
                Text.assemble(
                    ("Lectern was interrupted while recording ", ""),
                    (f"{meta.display_title}", "bold"),
                    (f"\nRecorded {meta.created_at.astimezone():%b %d at %H:%M} · "
                     f"{self._session.summary}\n\n", "#8b919e"),
                    ("Everything captured before the interruption is safe on disk.", ""),
                ),
                classes="modal-body",
            )
            with Horizontal(classes="modal-buttons"):
                yield Button("Discard", id="discard")
                yield Button("Keep as-is", id="recover")
                yield Button("Finalize", id="finalize")
                yield Button("Resume", id="resume", classes="-primary")

    def on_mount(self) -> None:
        button_id = "recover" if self._session.is_empty else "resume"
        self.query_one(f"#{button_id}", Button).focus()

    @on(Button.Pressed)
    def _chosen(self, event: Button.Pressed) -> None:
        if event.button.id == "discard":
            self.app.push_screen(
                ConfirmModal(
                    "This permanently deletes the session folder, including its transcript.",
                    title="Discard session?",
                    confirm_label="Delete",
                    danger=True,
                ),
                callback=lambda confirmed: self.dismiss("discard") if confirmed else None,
            )
            return
        self.dismiss(event.button.id)

    def action_later(self) -> None:
        self.dismiss(None)


class FinalizingModal(ModalScreen[None]):
    """Progress dialog shown while a session is being finished.

    Steps are ticked off as they complete so a long final synthesis never looks
    like a hang.
    """

    def __init__(self, steps: list[str]) -> None:
        super().__init__()
        self._labels = steps
        self._states: dict[str, str] = {step: "pending" for step in steps}
        self._detail = ""

    def compose(self) -> ComposeResult:
        with Vertical(classes="modal"):
            yield Label("Finishing session", classes="modal-title")
            with Vertical(id="finalize-steps"):
                for index, step in enumerate(self._labels):
                    yield Static(self._step_text(step), id=f"step-{index}", classes="finalize-step")
            yield Static("", id="finalize-detail", classes="hint")

    def _step_text(self, step: str) -> Text:
        state = self._states[step]
        icon, style = {
            "pending": (ICONS.dot, "#5f6672"),
            "active": (ICONS.spinner, "#56d4dd"),
            "done": (ICONS.check, "#4ade80"),
            "failed": (ICONS.cross, "#f87171"),
        }[state]
        return Text.assemble((f"{icon} ", style), (step, "#e6e8ec" if state != "pending" else "#8b919e"))

    def set_step(self, step: str, state: str, detail: str = "") -> None:
        """Mark a step active / done / failed and optionally show a sub-detail."""
        if step not in self._states:
            return
        self._states[step] = state
        index = self._labels.index(step)
        try:
            self.query_one(f"#step-{index}", Static).update(self._step_text(step))
            if detail:
                self.query_one("#finalize-detail", Static).update(_literal(detail))
        except Exception:  # noqa: BLE001 - modal already dismissed
            pass


class ExportModal(ModalScreen[str | None]):
    """Pick an export format."""

    BINDINGS = [("escape", "cancel", "Cancel")]

    def compose(self) -> ComposeResult:
        from lectern.sessions.export import EXPORTERS

        with Vertical(classes="modal"):
            yield Label("Export session", classes="modal-title")
            yield Static(
                "The file is written into the session folder.",
                classes="modal-body",
            )
            with Horizontal(classes="modal-buttons"):
                yield Button("Cancel", id="cancel")
                for exporter in EXPORTERS.values():
                    classes = "-primary" if exporter.format_id == "markdown" else ""
                    yield Button(exporter.label, id=f"fmt-{exporter.format_id}", classes=classes)

    def on_mount(self) -> None:
        self.query_one("#fmt-markdown", Button).focus()

    @on(Button.Pressed)
    def _pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ""
        if button_id.startswith("fmt-"):
            self.dismiss(button_id.removeprefix("fmt-"))
        else:
            self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)


class HelpModal(ModalScreen[None]):
    """Keyboard reference."""

    BINDINGS = [("escape", "dismiss_modal", "Close"), ("question_mark", "dismiss_modal", "Close")]

    SHORTCUTS: tuple[tuple[str, str, str], ...] = (
        ("Global", "ctrl+p  /", "Command palette"),
        ("Global", "?", "This help"),
        ("Global", "ctrl+c", "Quit (safely, mid-recording too)"),
        ("Home", "n  enter", "New session"),
        ("Home", "s", "Browse all sessions"),
        ("Home", "/", "Search transcripts and notes"),
        ("Home", ",", "Settings"),
        ("Home", "d", "Run environment checks"),
        ("Recording", "space", "Pause / resume"),
        ("Recording", "m", "Add an important marker"),
        ("Recording", "n", "Write a quick note"),
        ("Recording", "t", "Focus the transcript"),
        ("Recording", "o", "Focus the notes"),
        ("Recording", "f", "Return to follow-live"),
        ("Recording", "q", "Finish and generate final notes"),
        ("Review", "e", "Export"),
        ("Review", "r", "Retry final synthesis"),
        ("Review", "escape", "Back"),
    )

    def compose(self) -> ComposeResult:
        with Vertical(classes="modal"):
            yield Label("Keyboard shortcuts", classes="modal-title")
            with VerticalScroll():
                yield Static(self._render_shortcuts())
            with Horizontal(classes="modal-buttons"):
                yield Button("Close", id="close", classes="-primary")

    def _render_shortcuts(self) -> Text:
        rendered = Text()
        current_group = ""
        for group, keys, description in self.SHORTCUTS:
            if group != current_group:
                if current_group:
                    rendered.append("\n")
                rendered.append(f"{group.upper()}\n", style="bold #8b919e")
                current_group = group
            rendered.append(f"  {keys:<12}", style="#7c7cff")
            rendered.append(f"{description}\n", style="#c3c9d4")
        return rendered

    def on_mount(self) -> None:
        self.query_one("#close", Button).focus()

    @on(Button.Pressed, "#close")
    def action_dismiss_modal(self) -> None:
        self.dismiss(None)
