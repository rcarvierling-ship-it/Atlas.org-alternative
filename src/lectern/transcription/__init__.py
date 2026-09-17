"""Speech-to-text backends. whisper.cpp is the only one shipped today."""

from lectern.config.models import TranscriptionConfig
from lectern.transcription.base import (
    TranscriptionBackend,
    TranscriptionError,
    TranscriptionHealth,
    TranscriptSegment,
)
from lectern.transcription.models import (
    KNOWN_MODELS,
    RECOMMENDED,
    WhisperModel,
    available_models,
    download_model,
    find_model,
    find_whisper_server,
    installed_models,
)
from lectern.transcription.whisper_cpp import WhisperCppBackend, parse_whisper_response

__all__ = [
    "KNOWN_MODELS",
    "RECOMMENDED",
    "TranscriptSegment",
    "TranscriptionBackend",
    "TranscriptionConfig",
    "TranscriptionError",
    "TranscriptionHealth",
    "WhisperCppBackend",
    "WhisperModel",
    "available_models",
    "build_backend",
    "download_model",
    "find_model",
    "find_whisper_server",
    "installed_models",
    "parse_whisper_response",
]


def build_backend(config: TranscriptionConfig) -> TranscriptionBackend:
    """Instantiate the configured transcription backend."""
    if config.backend != "whisper_cpp":
        raise TranscriptionError(
            f"unknown transcription backend {config.backend!r} (only 'whisper_cpp' is available)"
        )
    return WhisperCppBackend(
        model=config.model,
        language=config.language,
        threads=config.threads,
        server_binary=config.whisper_server_binary,
        server_url=config.server_url,
    )
