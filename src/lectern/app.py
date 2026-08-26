"""The Textual application.

``LecternApp`` owns the services bundle, the screen stack and the few
cross-screen operations (start a recording, open a session, quit safely). It
deliberately contains no audio, transcription or note logic — those live in the
pipeline, which the recording screen drives.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from textual.app import App, ComposeResult, SystemCommand
from textual.binding import Binding
from textual.screen import Screen

from lectern import __version__
from lectern.logging_setup import get_logger, setup_logging
from lectern.services import AppServices, SessionRequest
from lectern.theme import THEMES

log = get_logger("app")


class LecternApp(App[None]):
    """Local AI lecture intelligence, in the terminal."""

    TITLE = "Lectern"
    CSS_PATH = "lectern.tcss"
    ENABLE_COMMAND_PALETTE = True

    BINDINGS = [
        Binding("ctrl+p", "command_palette", "Commands", show=False, priority=True),
        Binding("ctrl+c", "quit", "Quit", show=False, priority=True),
    ]

    def __init__(
        self,
        *,
        services: AppServices | None = None,
        start_request: SessionRequest | None = None,
        open_session_id: str | None = None,
        force_wizard: bool = False,
    ) -> None:
        super().__init__()
        self.services = services or AppServices.create()
        self._start_request = start_request
        self._open_session_id = open_session_id
        self._force_wizard = force_wizard
        self._recovery_checked = False

    # -- lifecycle ---------------------------------------------------------
    def on_mount(self) -> None:
        for theme in THEMES:
            self.register_theme(theme)
        self.theme = (
            self.services.config.ui.theme
            if self.services.config.ui.theme in {theme.name for theme in THEMES}
            else "lectern-dark"
        )
        log.info("Lectern %s starting", __version__)

        from lectern.doctor import first_run_needed
        from lectern.screens.home import HomeScreen
        from lectern.screens.setup_wizard import SetupWizardScreen

        if self._force_wizard or (
            first_run_needed(self.services.config)
            and self._start_request is None
            and self._open_session_id is None
        ):
            self.push_screen(SetupWizardScreen(mode="setup"))
            return

        self.push_screen(HomeScreen())

        if self._start_request is not None:
            self.start_recording(self._start_request)
        elif self._open_session_id is not None:
            self.open_session(self._open_session_id)
        else:
            self.check_for_recovery()

    def switch_to_home(self) -> None:
        """Replace whatever is on screen with a fresh Home screen."""
        from lectern.screens.home import HomeScreen

        self.pop_screen()
        self.push_screen(HomeScreen())
        self.check_for_recovery()

    # -- cross-screen operations ------------------------------------------
    def start_recording(self, request: SessionRequest) -> None:
        """Open the recording screen for ``request``."""
        from lectern.screens.new_session import NewSessionScreen
        from lectern.screens.recording import RecordingScreen

        if isinstance(self.screen, NewSessionScreen):
            self.pop_screen()
        self.push_screen(RecordingScreen(request))

    def open_session(self, session_id: str, *, replace_current: bool = False) -> None:
        """Open Session Review for a stored session."""
        from lectern.screens.modals import MessageModal
        from lectern.screens.review import ReviewScreen

        try:
            session = self.services.sessions.open(session_id)
        except Exception as exc:  # noqa: BLE001
            log.exception("could not open session %s", session_id)
            self.push_screen(
                MessageModal(f"Could not open that session: {exc}", title="Error", severity="error")
            )
            return

        if session is None:
            self.push_screen(
                MessageModal(
                    f"Session “{session_id}” could not be found on disk.",
                    title="Not found",
                    severity="warning",
                )
            )
            return

        if replace_current:
            self.pop_screen()
        self.push_screen(ReviewScreen(session))

    # -- crash recovery ----------------------------------------------------
    def check_for_recovery(self) -> None:
        """Offer to recover any session that was interrupted."""
        if self._recovery_checked:
            return
        self._recovery_checked = True

        from lectern.screens.modals import RecoveryModal
        from lectern.sessions.recovery import find_recoverable

        try:
            recoverable = find_recoverable(self.services.sessions)
        except Exception:  # noqa: BLE001
            log.exception("recovery scan failed")
            return
        if not recoverable:
            return

        session = recoverable[0]
        log.info("found interrupted session %s", session.meta.id)
        self.push_screen(
            RecoveryModal(session),
            callback=lambda action: self._handle_recovery(session, action),
        )

    def _handle_recovery(self, session, action: str | None) -> None:  # noqa: ANN001
        from lectern.sessions import recovery

        if action is None:
            return
        manager = self.services.sessions

        if action == "discard":
            recovery.discard(manager, session.meta)
            self.notify("Interrupted session discarded.", timeout=4)
            self._refresh_home()
            return

        if action == "resume":
            self.start_recording(
                SessionRequest(
                    title=session.meta.title,
                    course=session.meta.course,
                    audio_source=session.meta.audio_source,
                    whisper_model=session.meta.whisper_model
                    or self.services.config.transcription.model,
                    notes_model=session.meta.ollama_model or self.services.config.ollama.notes_model,
                    save_audio=self.services.config.audio.save_recording,
                    resume_session_id=session.meta.id,
                )
            )
            return

        meta = recovery.recover(manager, session.meta)
        self._refresh_home()
        if action == "finalize":
            self.open_session(meta.id)
            self.call_after_refresh(self._trigger_finalize)
        else:
            self.notify(
                f"Recovered {meta.word_count:,} words. The session is saved.",
                title="Session recovered",
                timeout=6,
            )

    def _trigger_finalize(self) -> None:
        from lectern.screens.review import ReviewScreen

        if isinstance(self.screen, ReviewScreen):
            self.screen.run_finalization()

    def _refresh_home(self) -> None:
        from lectern.screens.home import HomeScreen

        if isinstance(self.screen, HomeScreen):
            self.screen.refresh_sessions()

    # -- quitting ----------------------------------------------------------
    async def action_quit(self) -> None:
        """Quit safely, confirming first if a recording is in progress.

        Ctrl+C during a lecture must never mean "throw away the last 40
        minutes", so an active session is stopped and flushed to disk before
        the app exits, and its status is left recoverable.
        """
        from lectern.pipeline import PipelineState
        from lectern.screens.modals import ConfirmModal
        from lectern.screens.recording import RecordingScreen

        screen = self.screen
        if isinstance(screen, RecordingScreen) and screen.pipeline is not None:
            if screen.pipeline.state not in (PipelineState.STOPPED, PipelineState.FAILED):
                confirmed = await self.push_screen_wait(
                    ConfirmModal(
                        "A recording is in progress. Quitting saves the transcript and notes so "
                        "far — Lectern will offer to finish the session next time you open it.",
                        title="Quit while recording?",
                        confirm_label="Save and quit",
                        cancel_label="Keep recording",
                    )
                )
                if not confirmed:
                    return
                await screen.shutdown()

        await self.services.aclose()
        self.exit()

    async def shutdown_active_recording(self) -> None:
        """Used by the CLI on an abrupt exit path."""
        from lectern.screens.recording import RecordingScreen

        for screen in self.screen_stack:
            if isinstance(screen, RecordingScreen):
                await screen.shutdown()

    # -- command palette ---------------------------------------------------
    def get_system_commands(self, screen: Screen) -> Iterable[SystemCommand]:
        yield SystemCommand("New session", "Start recording a lecture", self._cmd_new_session)
        yield SystemCommand("Sessions", "Browse every recorded session", self._cmd_sessions)
        yield SystemCommand("Search", "Search transcripts and notes", self._cmd_search)
        yield SystemCommand("Settings", "Models, audio and storage", self._cmd_settings)
        yield SystemCommand("System check", "Verify the local setup", self._cmd_doctor)
        yield SystemCommand("Keyboard shortcuts", "Show the key bindings", self._cmd_help)
        yield SystemCommand("Open log file", "Show where logs are written", self._cmd_log_path)
        yield from super().get_system_commands(screen)

    def _cmd_new_session(self) -> None:
        from lectern.screens.new_session import NewSessionScreen

        self.push_screen(NewSessionScreen())

    def _cmd_sessions(self) -> None:
        from lectern.screens.sessions import SessionsScreen

        self.push_screen(SessionsScreen())

    def _cmd_search(self) -> None:
        from lectern.screens.search import SearchScreen

        self.push_screen(SearchScreen())

    def _cmd_settings(self) -> None:
        from lectern.screens.settings import SettingsScreen

        self.push_screen(SettingsScreen())

    def _cmd_doctor(self) -> None:
        from lectern.screens.setup_wizard import SetupWizardScreen

        self.push_screen(SetupWizardScreen(mode="doctor"))

    def _cmd_help(self) -> None:
        from lectern.screens.modals import HelpModal

        self.push_screen(HelpModal())

    def _cmd_log_path(self) -> None:
        from lectern.utils import paths

        self.notify(str(paths.log_file()), title="Log file", timeout=10)


def run_app(
    *,
    start_request: SessionRequest | None = None,
    open_session_id: str | None = None,
    force_wizard: bool = False,
    verbose: bool = False,
    config_path: Path | None = None,
) -> None:
    """Entry point used by the CLI."""
    setup_logging(verbose=verbose)
    services = AppServices.create(config_path=config_path)
    app = LecternApp(
        services=services,
        start_request=start_request,
        open_session_id=open_session_id,
        force_wizard=force_wizard,
    )
    try:
        app.run()
    finally:
        import asyncio

        with_cleanup = services.aclose()
        try:
            asyncio.run(with_cleanup)
        except RuntimeError:  # pragma: no cover - loop already closed
            with_cleanup.close()
        log.info("Lectern exited")
