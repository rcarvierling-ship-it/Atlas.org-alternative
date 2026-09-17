#!/usr/bin/env python3
"""Generate the demo/test WAV fixture.

The file contains speech-shaped bursts separated by silence: enough structure
for the VAD to segment it into utterances, without shipping a real recording
(and someone's voice) in the repository. The transcript itself comes from the
fake whisper server in tests, or from real whisper.cpp when one is installed —
in which case this audio produces no meaningful words, which is fine, because
its job is to exercise timing and segmentation.

    python scripts/make_fixture_audio.py [--out tests/fixtures/lecture.wav]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000
UTTERANCE_SECONDS = 1.8
GAP_SECONDS = 1.1
UTTERANCES = 5


def speech_burst(seconds: float, rng: np.random.Generator) -> np.ndarray:
    """A syllable-rate amplitude-modulated tone stack, roughly speech-shaped."""
    t = np.linspace(0, seconds, int(SAMPLE_RATE * seconds), endpoint=False)
    # Formant-ish carriers in the voice band.
    signal = sum(
        amplitude * np.sin(2 * np.pi * frequency * t)
        for frequency, amplitude in ((140.0, 0.5), (420.0, 0.3), (900.0, 0.15), (1700.0, 0.08))
    )
    # Syllable envelope at ~4 Hz, plus a slow breath envelope.
    syllables = 0.5 * (1 + np.sin(2 * np.pi * 4.0 * t - np.pi / 2))
    breath = np.hanning(len(t))
    noise = rng.normal(0, 0.02, len(t))
    burst = (signal * syllables * breath + noise) * 0.35
    return burst.astype(np.float32)


def build(utterances: int = UTTERANCES, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    silence = (rng.normal(0, 0.0008, int(SAMPLE_RATE * GAP_SECONDS))).astype(np.float32)
    lead_in = (rng.normal(0, 0.0008, int(SAMPLE_RATE * 0.5))).astype(np.float32)

    parts: list[np.ndarray] = [lead_in]
    for _ in range(utterances):
        parts.append(speech_burst(UTTERANCE_SECONDS, rng))
        parts.append(silence.copy())
    return np.concatenate(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "lecture.wav",
    )
    parser.add_argument("--utterances", type=int, default=UTTERANCES)
    args = parser.parse_args()

    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from lectern.utils.audio_utils import write_wav

    audio = build(args.utterances)
    write_wav(args.out, audio, SAMPLE_RATE)
    print(f"wrote {args.out} ({audio.size / SAMPLE_RATE:.1f}s, {args.out.stat().st_size / 1000:.0f} kB)")


if __name__ == "__main__":
    main()
