"""Lectern — local, privacy-first AI lecture note-taking for the terminal.

The whole pipeline runs on the local machine:

    Audio -> whisper.cpp -> transcript -> Ollama -> rolling notes -> TUI

No audio, transcript or note text ever leaves the machine.
"""

__version__ = "0.1.0"
__all__ = ["__version__"]
