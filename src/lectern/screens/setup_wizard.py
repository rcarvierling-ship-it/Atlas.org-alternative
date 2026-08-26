"""First-run setup and the in-app doctor.

The same screen serves both: on first launch it explains what is missing and how
to fix it before the user hits a wall mid-lecture; later, ``d`` from Home
re-runs the same checks.

Nothing is installed automatically. Remediation commands are shown for the user
to run themselves — installing Homebrew packages behind someone's back is not
something a note-taking app gets to do.
"""

from __future__ import annotations

from rich.console import RenderableType
from rich.table import Table
from rich.text import Text
from textual import on, work
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, Footer, Label, LoadingIndicator, Static

from lectern.config import manager as config_manager
from lectern.doctor import CheckStatus, DoctorReport
from lectern.logging_setup import get_logger
from lectern.theme import ICONS

log = get_logger("screens.setup")


class CheckLine(Static):
    """One diagnostic result plus its remedy.

    Laid out as a grid so a long detail or remedy wraps under its own column
    instead of running back to the left margin.
    """

    def __init__(self, check) -> None:  # noqa: ANN001 - doctor.Check
        super().__init__()
        self._check = check

    def render(self) -> RenderableType:
        check = self._check
        colour = {
            CheckStatus.OK: "#4ade80",
            CheckStatus.WARN: "#fbbf24",
            CheckStatus.FAIL: "#f87171",
            CheckStatus.UNKNOWN: "#8b919e",
        }[check.status]

        grid = Table.grid(padding=(0, 1))
        grid.add_column(width=20, no_wrap=True)
        grid.add_column(width=1, no_wrap=True)
        grid.add_column(ratio=1, overflow="fold")

        grid.add_row(
            Text(check.name, style="#c3c9d4"),
            Text(check.icon, style=colour),
            Text(check.detail, style="#8b919e"),
        )
        if check.remedy:
            grid.add_row(
                Text(""),
                Text(ICONS.arrow, style="#5f6672"),
                Text(check.remedy, style="#56d4dd"),
            )
        return grid


class SetupWizardScreen(Screen):
    """Environment checks, shown on first run and on demand."""

    BINDINGS = [
        ("escape", "leave", "Back"),
        ("r", "recheck", "Re-run checks"),
    ]

    def __init__(self, *, mode: str = "doctor") -> None:
        super().__init__()
        #: ``setup`` on first run (writes the config on continue), ``doctor`` otherwise.
        self.mode = mode
        self._report: DoctorReport | None = None

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="wizard-body"):
            if self.mode == "setup":
                yield Label("WELCOME TO LECTERN", classes="section-title")
                yield Static(
                    "Lectern records lectures and takes notes entirely on this machine.\n"
                    "Nothing is sent to a cloud service. Here's what it found:",
                    classes="muted",
                )
            else:
                yield Label("SYSTEM CHECK", classes="section-title")
            yield LoadingIndicator(id="wizard-loading")
            yield Vertical(id="check-list")
            yield Static("", id="wizard-summary", classes="muted")
            with Horizontal(id="form-buttons"):
                yield Button("Re-run checks", id="recheck")
                yield Button(
                    "Continue" if self.mode == "setup" else "Back",
                    id="continue",
                    classes="-primary",
                )
        yield Footer()

    def on_mount(self) -> None:
        self.run_checks()

    @work(exclusive=True, group="doctor")
    async def run_checks(self) -> None:
        from lectern.doctor import run_all

        loading = self.query_one("#wizard-loading", LoadingIndicator)
        loading.display = True
        container = self.query_one("#check-list", Vertical)
        await container.remove_children()

        report = await run_all(self.app.services.config)
        self._report = report

        loading.display = False
        for check in report.checks:
            await container.mount(CheckLine(check))

        summary = self.query_one("#wizard-summary", Static)
        summary.update(self._summary_text(report))

    def _summary_text(self, report: DoctorReport) -> Text:
        rendered = Text()
        rendered.append("\n")
        if report.healthy and not report.warnings:
            rendered.append(f"{ICONS.check} Everything looks good.", style="bold #4ade80")
            return rendered
        if report.failures:
            rendered.append(f"{report.summary()}\n", style="bold #fbbf24")
            if report.can_record:
                rendered.append(
                    "You can still record and transcribe; the items above limit some features.\n",
                    style="#8b919e",
                )
            else:
                rendered.append(
                    "Transcription needs whisper.cpp and a Whisper model before you can record.\n",
                    style="#8b919e",
                )
        else:
            rendered.append(f"{ICONS.check} Ready to record.", style="bold #4ade80")
            rendered.append(
                "  Some optional features are unavailable — see the warnings above.\n",
                style="#8b919e",
            )
        return rendered

    @on(Button.Pressed, "#recheck")
    def action_recheck(self) -> None:
        self.run_checks()

    @on(Button.Pressed, "#continue")
    def action_leave(self) -> None:
        if self.mode == "setup":
            # Writing the config marks first-run as done, so the wizard does
            # not reappear on every launch.
            config_manager.save(self.app.services.config)
            log.info("first-run setup completed")
            self.app.switch_to_home()
            return
        self.app.pop_screen()
