"""PCM helpers: resampling, mixing and WAV writing.

Whisper wants 16 kHz mono float32 in ``[-1, 1]``. Everything upstream of the
transcription backend converts into that shape as early as possible so the rest
of the pipeline only ever deals with one audio format.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

TARGET_SAMPLE_RATE = 16_000


def to_mono(audio: np.ndarray) -> np.ndarray:
    """Collapse an ``(n, channels)`` block to mono, preserving 1-D input."""
    if audio.ndim == 1:
        return audio.astype(np.float32, copy=False)
    return audio.astype(np.float32, copy=False).mean(axis=1)


def resample(audio: np.ndarray, source_rate: int, target_rate: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    """Resample mono float32 audio.

    Downsampling first applies a short windowed-sinc low-pass so that energy
    above the new Nyquist frequency does not alias down into the speech band —
    aliased hiss measurably degrades whisper.cpp accuracy.
    """
    audio = np.asarray(audio, dtype=np.float32)
    if source_rate == target_rate or audio.size == 0:
        return audio

    if target_rate < source_rate:
        audio = _lowpass(audio, cutoff=target_rate / 2.0, sample_rate=source_rate)

    duration = audio.size / source_rate
    out_samples = int(round(duration * target_rate))
    if out_samples <= 0:
        return np.zeros(0, dtype=np.float32)

    source_positions = np.arange(audio.size, dtype=np.float64)
    target_positions = np.linspace(0.0, audio.size - 1, out_samples, dtype=np.float64)
    return np.interp(target_positions, source_positions, audio).astype(np.float32)


def _lowpass(audio: np.ndarray, *, cutoff: float, sample_rate: int, taps: int = 63) -> np.ndarray:
    """Zero-phase-ish FIR low-pass using a Hamming-windowed sinc kernel."""
    if audio.size < taps:
        return audio
    normalized = min(0.99, cutoff / (sample_rate / 2.0)) / 2.0 * 2.0
    n = np.arange(taps, dtype=np.float64) - (taps - 1) / 2.0
    kernel = np.sinc(normalized * n) * np.hamming(taps)
    kernel /= kernel.sum()
    return np.convolve(audio, kernel, mode="same").astype(np.float32)


def mix(*tracks: np.ndarray, gains: tuple[float, ...] | None = None) -> np.ndarray:
    """Sum equal-rate mono tracks, zero-padding to the longest, with soft clipping."""
    tracks = tuple(np.asarray(t, dtype=np.float32) for t in tracks if t is not None and t.size)
    if not tracks:
        return np.zeros(0, dtype=np.float32)
    if gains is None:
        gains = tuple(1.0 for _ in tracks)

    length = max(t.size for t in tracks)
    total = np.zeros(length, dtype=np.float32)
    for track, gain in zip(tracks, gains, strict=False):
        padded = np.zeros(length, dtype=np.float32)
        padded[: track.size] = track
        total += padded * float(gain)
    return np.clip(total, -1.0, 1.0)


def rms_level(audio: np.ndarray) -> float:
    """Root-mean-square level of a block, in ``[0, 1]``."""
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))


def float_to_pcm16(audio: np.ndarray) -> bytes:
    """Convert float32 ``[-1, 1]`` to little-endian signed 16-bit PCM bytes."""
    clipped = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


def pcm16_to_float(data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0


def write_wav(path: Path, audio: np.ndarray, sample_rate: int = TARGET_SAMPLE_RATE) -> None:
    """Write mono float32 audio to a 16-bit PCM WAV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(float_to_pcm16(audio))


def wav_bytes(audio: np.ndarray, sample_rate: int = TARGET_SAMPLE_RATE) -> bytes:
    """Serialize mono float32 audio into an in-memory WAV container.

    whisper.cpp's HTTP server takes a multipart file upload, so utterances are
    wrapped as WAV in memory rather than staged through the filesystem.
    """
    import io

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(float_to_pcm16(audio))
    return buffer.getvalue()


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    """Read a PCM WAV file as mono float32 plus its sample rate."""
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())

    if width == 2:
        samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 4:
        samples = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
    elif width == 1:
        samples = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"unsupported WAV sample width: {width} bytes")

    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return samples.astype(np.float32), rate
