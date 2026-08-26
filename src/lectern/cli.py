"""Command line interface.

The TUI is the product; these commands exist for the things a terminal is
better at — scripting an export, checking the environment, tailing the log.
Running ``lectern`` with no arguments launches the full-screen app.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from lectern import __version__
from lectern.config import manager as config_manager
from lectern.logging_setup import setup_logging
from lectern.utils import paths
from lectern.utils.timefmt import format_duration, format_relative

console = Console()
error_console = Console(stderr=True)

app = typer.Typer(
    name="lectern",
    help="Local AI lecture note-taking. Your lectures stay on your Mac.",
    no_args_is_help=False,
    add_completion=False,
)
models_app = typer.Typer(help="Manage local Whisper and Ollama models.", no_args_is_help=False)
config_app = typer.Typer(help="Inspect and edit configuration.", no_args_is_help=False)
app.add_typer(models_app, name="models")
app.add_typer(config_app, name="config")


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"lectern {__version__}")
        raise typer.Exit()


@app.callback(invoke_without_command=True)
def main_callback(
    ctx: typer.Context,
    version: Annotated[
        bool, typer.Option("--version", callback=_version_callback, is_eager=True, help="Show the version.")
    ] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Verbose logging.")] = False,
) -> None:
    """Launch the Lectern TUI when no subcommand is given."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    if ctx.invoked_subcommand is None:
        from lectern.app import run_app

        run_app(verbose=verbose)


@app.command()
def record(
    title: Annotated[Optional[str], typer.Option("--title", "-t", help="Session title.")] = None,
    course: Annotated[str, typer.Option("--course", "-c", help="Course or category.")] = "",
    source: Annotated[
        str, typer.Option("--source", "-s", help="microphone | system | both.")
    ] = "",
    file: Annotated[
        Optional[Path],
        typer.Option("--file", "-f", help="Transcribe a WAV file through the full pipeline."),
    ] = None,
    speed: Annotated[
        float, typer.Option("--speed", help="Playback speed for --file (1.0 = real time).")
    ] = 1.0,
    no_audio: Annotated[bool, typer.Option("--no-audio", help="Do not save the recording.")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Start recording immediately, skipping the new-session form."""
    from lectern.app import run_app
    from lectern.services import SessionRequest

    config = config_manager.load()

    if file is not None and not file.exists():
        error_console.print(f"[red]No such file:[/] {file}")
        raise typer.Exit(1)

    default_title = file.stem.replace("-", " ").title() if file else "Quick session"
    request = SessionRequest(
        title=title or default_title,
        course=course,
        audio_source=source or config.audio.source.value,
        whisper_model=config.transcription.model,
        notes_model=config.ollama.notes_model,
        save_audio=not no_audio and config.audio.save_recording,
        file_path=file,
        file_speed=speed,
    )
    run_app(start_request=request, verbose=verbose)


@app.command(name="sessions")
def list_sessions(
    limit: Annotated[int, typer.Option("--limit", "-n", help="How many to show.")] = 20,
) -> None:
    """List recorded sessions."""
    manager = _session_manager()
    try:
        sessions = manager.list_recent(limit)
        if not sessions:
            console.print("[dim]No sessions yet. Run 'lectern' to record one.[/]")
            return

        table = Table(box=None, pad_edge=False, header_style="dim")
        table.add_column("When", style="dim", width=10)
        table.add_column("Session")
        table.add_column("Course", style="dim")
        table.add_column("Length", justify="right", style="dim")
        table.add_column("Words", justify="right", style="dim")
        table.add_column("ID", style="dim")
        for meta in sessions:
            table.add_row(
                format_relative(meta.created_at),
                meta.display_title,
                meta.course or "—",
                format_duration(meta.duration_seconds),
                f"{meta.word_count:,}",
                meta.id,
            )
        console.print(table)
    finally:
        manager.close()


@app.command(name="open")
def open_session(
    session: Annotated[str, typer.Argument(help="Session id, id prefix or title fragment.")],
) -> None:
    """Open a session in the TUI."""
    from lectern.app import run_app

    manager = _session_manager()
    try:
        meta = manager.find(session)
    finally:
        manager.close()
    if meta is None:
        error_console.print(f"[red]No session matches[/] {session!r}")
        raise typer.Exit(1)
    run_app(open_session_id=meta.id)


@app.command()
def search(
    query: Annotated[str, typer.Argument(help="Text to search for.")],
    limit: Annotated[int, typer.Option("--limit", "-n")] = 10,
) -> None:
    """Search transcripts and notes across all sessions."""
    manager = _session_manager()
    try:
        hits = manager.search(query, limit=limit)
        if not hits:
            console.print(f"[dim]Nothing matched[/] {query!r}")
            return
        for hit in hits:
            header = Text()
            header.append(hit.title, style="bold")
            if hit.course:
                header.append(f"  {hit.course}", style="dim")
            header.append(f"  {format_relative(hit.created_at)}  {hit.session_id}", style="dim")
            console.print(header)
            for snippet in hit.snippets:
                console.print(Text(f"  {snippet}", style="dim"))
            console.print()
    finally:
        manager.close()


@app.command()
def export(
    session: Annotated[str, typer.Argument(help="Session id, id prefix or title fragment.")],
    format: Annotated[str, typer.Option("--format", "-f", help="markdown | text | json.")] = "markdown",
    out: Annotated[Optional[Path], typer.Option("--out", "-o", help="Output file or directory.")] = None,
) -> None:
    """Export a session."""
    from lectern.sessions.export import export_session, get_exporter

    manager = _session_manager()
    try:
        try:
            get_exporter(format)
        except ValueError as exc:
            error_console.print(f"[red]{exc}[/]")
            raise typer.Exit(1) from exc

        meta = manager.find(session)
        if meta is None:
            error_console.print(f"[red]No session matches[/] {session!r}")
            raise typer.Exit(1)
        loaded = manager.open(meta.id)
        if loaded is None:
            error_console.print(f"[red]Session folder missing for[/] {meta.id}")
            raise typer.Exit(1)
        path = export_session(loaded, format_id=format, destination=out)
        console.print(f"[green]Exported[/] {meta.display_title} → {path}", soft_wrap=True)
    finally:
        manager.close()


@models_app.callback(invoke_without_command=True)
def models_callback(ctx: typer.Context) -> None:
    """Show installed Whisper and Ollama models."""
    if ctx.invoked_subcommand is not None:
        return
    _print_whisper_models()
    console.print()
    _print_ollama_models()


@models_app.command("whisper")
def models_whisper(
    download: Annotated[
        Optional[str], typer.Option("--download", "-d", help="Download a model, e.g. small.en.")
    ] = None,
) -> None:
    """List or download whisper.cpp models."""
    from lectern.transcription.models import download_model, find_model

    if download:
        existing = find_model(download)
        if existing is not None:
            console.print(f"[green]Already installed:[/] {existing}")
            return
        console.print(f"Downloading [bold]{download}[/] …")
        try:
            with console.status("Downloading…") as status:

                def progress(done: int, total: int | None) -> None:
                    if total:
                        status.update(f"Downloading… {done / total:.0%} ({done / 1e6:.0f} MB)")
                    else:
                        status.update(f"Downloading… {done / 1e6:.0f} MB")

                path = download_model(download, progress=progress)
        except Exception as exc:  # noqa: BLE001
            error_console.print(f"[red]Download failed:[/] {exc}")
            raise typer.Exit(1) from exc
        console.print(f"[green]Installed[/] {path}", soft_wrap=True)
        return
    _print_whisper_models()


@models_app.command("ollama")
def models_ollama() -> None:
    """List models installed in Ollama."""
    _print_ollama_models()


def _print_whisper_models() -> None:
    from lectern.transcription.models import available_models

    config = config_manager.load()
    table = Table(title="Whisper models", box=None, title_justify="left", header_style="dim")
    table.add_column("Model")
    table.add_column("Size", justify="right", style="dim")
    table.add_column("Status")
    for model in available_models():
        if model.installed:
            status = Text("installed", style="green")
            if model.name == config.transcription.model:
                status.append("  (selected)", style="dim")
        else:
            status = Text("not installed", style="dim")
        table.add_row(model.name, model.size_label, status)
    console.print(table)
    console.print(f"[dim]Downloaded models live in {paths.whisper_models_dir()}[/]", soft_wrap=True)


def _print_ollama_models() -> None:
    from lectern.llm.ollama import OllamaBackend

    config = config_manager.load()

    async def fetch():
        backend = OllamaBackend(config.ollama.host)
        try:
            return await backend.health()
        finally:
            await backend.close()

    health = asyncio.run(fetch())
    if not health.available:
        error_console.print(f"[yellow]Ollama is not responding at {config.ollama.host}[/]")
        error_console.print("[dim]Start it with 'ollama serve', then 'ollama pull qwen3:8b'.[/]")
        return

    table = Table(title="Ollama models", box=None, title_justify="left", header_style="dim")
    table.add_column("Model")
    table.add_column("Size", justify="right", style="dim")
    table.add_column("Details", style="dim")
    table.add_column("Used for")
    for model in health.models:
        roles = []
        if model.name == config.ollama.notes_model:
            roles.append("live notes")
        if model.name == (config.ollama.final_model or config.ollama.notes_model):
            roles.append("final notes")
        table.add_row(model.name, model.size_label, model.detail, ", ".join(roles) or "—")
    console.print(table)


@app.command()
def doctor() -> None:
    """Check that everything Lectern needs is present."""
    from lectern.doctor import CheckStatus, run_all

    config = config_manager.load()
    report = asyncio.run(run_all(config))

    console.print()
    console.print("[bold]LECTERN DOCTOR[/]")
    console.print()
    styles = {
        CheckStatus.OK: "green",
        CheckStatus.WARN: "yellow",
        CheckStatus.FAIL: "red",
        CheckStatus.UNKNOWN: "dim",
    }
    for check in report.checks:
        style = styles[check.status]
        line = Text()
        line.append(f"{check.name:<22}")
        line.append(f"{check.icon} ", style=style)
        line.append(check.detail, style="dim")
        console.print(line)
        if check.remedy:
            console.print(Text(f"{' ' * 24}→ {check.remedy}", style="cyan"))
    console.print()
    console.print(report.summary(), style="green" if report.healthy else "yellow")
    console.print()
    if not report.healthy:
        raise typer.Exit(1)


@config_app.callback(invoke_without_command=True)
def config_callback(ctx: typer.Context) -> None:
    """Show the current configuration."""
    if ctx.invoked_subcommand is not None:
        return
    path = paths.config_file()
    if path.exists():
        console.print(f"[dim]{path}[/]\n", soft_wrap=True)
        console.print(path.read_text(encoding="utf-8"))
    else:
        console.print("[dim]No config file yet — defaults are in use.[/]")
        console.print(f"[dim]It will be created at {path}[/]", soft_wrap=True)


@config_app.command("path")
def config_path() -> None:
    """Print the config file path."""
    # soft_wrap keeps the path on one line so it stays copy-pasteable and pipeable.
    console.print(str(paths.config_file()), soft_wrap=True)


@config_app.command("init")
def config_init() -> None:
    """Write a config file containing the current defaults."""
    path = config_manager.save(config_manager.load())
    console.print(f"[green]Wrote[/] {path}")


@config_app.command("set")
def config_set(
    assignment: Annotated[
        str, typer.Argument(help="section.key=value, e.g. ollama.notes_model=qwen3:8b")
    ],
) -> None:
    """Change one setting."""
    if "=" not in assignment:
        error_console.print("[red]Expected section.key=value[/]")
        raise typer.Exit(1)
    key, _, value = assignment.partition("=")
    config = config_manager.load()
    try:
        config_manager.set_value(config, key.strip(), value.strip())
    except (KeyError, ValueError) as exc:
        error_console.print(f"[red]{exc}[/]")
        raise typer.Exit(1) from exc
    config_manager.save(config)
    console.print(f"[green]Set[/] {key.strip()} = {value.strip()}")


@app.command()
def logs(
    lines: Annotated[int, typer.Option("--lines", "-n", help="How many lines to show.")] = 60,
    path_only: Annotated[bool, typer.Option("--path", help="Print the log path and exit.")] = False,
) -> None:
    """Show the application log."""
    log_path = paths.log_file()
    if path_only:
        console.print(str(log_path), soft_wrap=True)
        return
    if not log_path.exists():
        console.print(f"[dim]No log yet at {log_path}[/]", soft_wrap=True)
        return
    content = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    for line in content[-lines:]:
        console.print(Text(line, style="dim" if " DEBUG " in line else ""))


@app.command()
def reindex() -> None:
    """Rebuild the session index from the sessions folder."""
    manager = _session_manager()
    try:
        count = manager.reindex()
        console.print(f"[green]Reindexed[/] {count} session(s)")
    finally:
        manager.close()


def _session_manager():
    from lectern.sessions.manager import SessionManager

    paths.ensure_dirs()
    return SessionManager(config_manager.load())


def main() -> None:
    """Console-script entry point."""
    setup_logging()
    paths.ensure_dirs()
    try:
        app()
    except KeyboardInterrupt:  # pragma: no cover - interactive
        error_console.print("\n[dim]Interrupted.[/]")
        sys.exit(130)


if __name__ == "__main__":  # pragma: no cover
    main()
