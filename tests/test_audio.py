"""VAD segmentation, PCM helpers and the audio sources."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from lectern.audio.base import AudioSource
from lectern.audio.combined import CombinedAudioSource
from lectern.audio.file_source import FileAudioSource
from lectern.audio.vad import VADConfig, VoiceSegmenter
from lectern.utils.audio_utils import (
    TARGET_SAMPLE_RATE,
    float_to_pcm16,
    mix,
    pcm16_to_float,
    read_wav,
    resample,
    rms_level,
    wav_bytes,
    write_wav,
)

SR = TARGET_SAMPLE_RATE


def tone(seconds: float, *, amplitude: float = 0.3, frequency: float = 300.0) -> np.ndarray:
    t = np.linspace(0, seconds, int(SR * seconds), endpoint=False)
    return (np.sin(2 * np.pi * frequency * t) * amplitude).astype(np.float32)


def silence(seconds: float, *, noise: float = 0.0005) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.normal(0, noise, int(SR * seconds)).astype(np.float32)


def feed_all(segmenter: VoiceSegmenter, audio: np.ndarray, block: int = 1600):
    found = []
    for offset in range(0, audio.size, block):
        found.extend(segmenter.feed(audio[offset : offset + block]))
    return found


# -- PCM helpers ---------------------------------------------------------


def test_pcm_round_trip_preserves_signal():
    original = tone(0.1)
    restored = pcm16_to_float(float_to_pcm16(original))
    assert restored.shape == original.shape
    assert np.max(np.abs(restored - original)) < 1e-3


def test_resample_changes_length_and_keeps_energy():
    original = tone(1.0, frequency=200.0)
    upsampled = resample(original, SR, 48_000)
    downsampled = resample(upsampled, 48_000, SR)
    assert upsampled.size == 48_000
    assert abs(downsampled.size - SR) <= 1
    assert rms_level(downsampled) == pytest.approx(rms_level(original), abs=0.05)


def test_resample_is_a_noop_at_the_same_rate():
    original = tone(0.1)
    assert resample(original, SR, SR) is original


def test_mix_sums_and_clips():
    mixed = mix(np.full(10, 0.6, dtype=np.float32), np.full(10, 0.6, dtype=np.float32))
    assert np.all(mixed <= 1.0)
    assert mixed[0] == pytest.approx(1.0)


def test_mix_pads_to_the_longest_track():
    mixed = mix(np.ones(5, dtype=np.float32), np.ones(10, dtype=np.float32))
    assert mixed.size == 10


def test_wav_round_trip(tmp_path):
    original = tone(0.5)
    path = tmp_path / "sample.wav"
    write_wav(path, original)
    restored, rate = read_wav(path)
    assert rate == SR
    assert restored.size == original.size
    assert np.max(np.abs(restored - original)) < 1e-3


def test_wav_bytes_is_a_valid_container():
    data = wav_bytes(tone(0.2))
    assert data[:4] == b"RIFF"
    assert data[8:12] == b"WAVE"


# -- VAD -----------------------------------------------------------------


def test_speech_between_silence_is_one_utterance():
    audio = np.concatenate([silence(1.0), tone(2.0), silence(1.5)])
    segmenter = VoiceSegmenter()
    found = feed_all(segmenter, audio)
    assert len(found) == 1
    utterance = found[0]
    # Pre-roll means the utterance starts slightly before the speech.
    assert 0.6 < utterance.start_time < 1.0
    assert utterance.duration > 2.0


def test_two_phrases_become_two_utterances():
    audio = np.concatenate([silence(0.6), tone(1.5), silence(1.5), tone(1.5), silence(1.0)])
    found = feed_all(VoiceSegmenter(), audio)
    assert len(found) == 2
    assert found[1].start_time > found[0].end_time


def test_short_pause_does_not_split_a_sentence():
    """A 300 ms breath is well inside the hangover window."""
    audio = np.concatenate([silence(0.6), tone(1.2), silence(0.3), tone(1.2), silence(1.5)])
    found = feed_all(VoiceSegmenter(), audio)
    assert len(found) == 1


def test_pure_silence_produces_nothing():
    """The main defence against Whisper hallucinating over an empty room."""
    found = feed_all(VoiceSegmenter(), silence(8.0))
    assert found == []


def test_a_click_is_rejected_as_too_short():
    audio = np.concatenate([silence(1.0), tone(0.05, amplitude=0.9), silence(1.5)])
    assert feed_all(VoiceSegmenter(), audio) == []


def test_continuous_speech_is_cut_at_the_ceiling():
    """A monologue must still yield material to the note model."""
    config = VADConfig(max_utterance_ms=3000)
    found = feed_all(VoiceSegmenter(config=config), np.concatenate([silence(0.5), tone(10.0)]))
    assert len(found) >= 3
    assert all(utterance.duration <= 3.2 for utterance in found)


def test_flush_emits_speech_still_in_progress():
    segmenter = VoiceSegmenter()
    feed_all(segmenter, np.concatenate([silence(0.6), tone(2.0)]))
    trailing = segmenter.flush()
    assert trailing is not None
    assert trailing.duration > 1.5


def test_vad_disabled_still_segments_on_length():
    segmenter = VoiceSegmenter(config=VADConfig(max_utterance_ms=2000), enabled=False)
    found = feed_all(segmenter, tone(5.0))
    assert len(found) >= 2


def test_noise_floor_adapts_to_a_loud_room():
    """Speech is detected relative to the room, not an absolute threshold."""
    rng = np.random.default_rng(3)
    room = rng.normal(0, 0.02, int(SR * 2.0)).astype(np.float32)
    speech = tone(2.0, amplitude=0.35) + rng.normal(0, 0.02, int(SR * 2.0)).astype(np.float32)
    audio = np.concatenate([room, speech, room])
    found = feed_all(VoiceSegmenter(), audio)
    assert len(found) == 1


def test_pending_audio_exposes_the_utterance_in_progress():
    segmenter = VoiceSegmenter()
    feed_all(segmenter, np.concatenate([silence(0.6), tone(1.0)]))
    assert segmenter.in_speech
    assert segmenter.pending_audio.size > SR * 0.8


# -- sources -------------------------------------------------------------


@pytest.mark.asyncio
async def test_file_source_streams_blocks_and_stops(fixture_wav):
    source = FileAudioSource(fixture_wav, speed=50.0)
    await source.start()
    total = 0
    async for block in source.frames():
        total += block.size
    assert total == pytest.approx(15 * SR, rel=0.05)
    assert not source.running


@pytest.mark.asyncio
async def test_file_source_reports_a_missing_file():
    from lectern.audio.base import AudioError

    source = FileAudioSource("/nonexistent/path.wav")
    with pytest.raises(AudioError, match="not found"):
        await source.start()


class ScriptedSource(AudioSource):
    """Emits a fixed list of blocks, then finishes."""

    kind = "scripted"

    def __init__(self, blocks: list[np.ndarray], *, delay: float = 0.0) -> None:
        super().__init__()
        self._blocks = blocks
        self._delay = delay
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._emit())

    async def _emit(self) -> None:
        for block in self._blocks:
            if self._delay:
                await asyncio.sleep(self._delay)
            self._publish(block)
        self._close_stream()

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
        self._close_stream()


@pytest.mark.asyncio
async def test_combined_source_mixes_both_legs():
    block = np.full(1600, 0.2, dtype=np.float32)
    left = ScriptedSource([block.copy() for _ in range(4)])
    right = ScriptedSource([block.copy() for _ in range(4)])
    combined = CombinedAudioSource(left, right, block_ms=100)

    await combined.start()
    received = [chunk async for chunk in combined.frames()]
    await combined.stop()

    assert received
    # Both legs contributed, so the mixed level is roughly double one leg.
    assert float(received[0][0]) == pytest.approx(0.4, abs=0.01)


@pytest.mark.asyncio
async def test_combined_source_survives_one_leg_stalling():
    """Half a recording beats none if the helper dies mid-lecture."""
    block = np.full(1600, 0.25, dtype=np.float32)
    healthy = ScriptedSource([block.copy() for _ in range(6)], delay=0.05)
    dead = ScriptedSource([])
    combined = CombinedAudioSource(healthy, dead, block_ms=100)

    await combined.start()
    received = [chunk async for chunk in combined.frames()]
    await combined.stop()

    assert len(received) >= 4
    assert float(received[-1][0]) == pytest.approx(0.25, abs=0.01)


@pytest.mark.asyncio
async def test_queue_backpressure_drops_oldest_not_newest():
    source = ScriptedSource([])
    source._queue = asyncio.Queue(maxsize=2)
    for value in (1.0, 2.0, 3.0, 4.0):
        source._publish(np.full(4, value, dtype=np.float32))

    assert source.dropped_blocks == 2
    remaining = [source._queue.get_nowait()[0] for _ in range(2)]
    assert remaining == [3.0, 4.0]
