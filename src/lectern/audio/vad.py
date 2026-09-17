"""Voice activity detection and utterance segmentation.

Rather than transcribing the microphone continuously — which wastes compute and
reliably makes Whisper hallucinate YouTube outros over silence — audio is cut
into *utterances* at natural pauses and only those are sent to whisper.cpp.

The detector is a pure-numpy adaptive energy gate:

* A rolling percentile of recent frame energies estimates the noise floor, so
  the gate adapts to a quiet study room or a noisy lecture hall on its own.
* Speech starts when energy exceeds the floor by ``start_margin`` dB and stays
  there for a few frames (rejects key clicks and coughs).
* Speech ends after ``hangover_ms`` of sub-threshold audio, so a speaker's
  natural mid-sentence pause does not split an utterance.
* ``pre_roll_ms`` of audio *before* the trigger is prepended, which is what
  stops the first syllable from being clipped.

There is no C extension: one less native dependency to build, and the behaviour
is deterministic enough to unit test with synthetic audio.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

FRAME_MS = 20


@dataclass(slots=True)
class VADConfig:
    sample_rate: int = 16_000
    frame_ms: int = FRAME_MS
    start_margin_db: float = 9.0
    stop_margin_db: float = 5.0
    hangover_ms: int = 700
    pre_roll_ms: int = 300
    min_utterance_ms: int = 600
    #: Force a cut in continuous speech so notes are not starved by a monologue.
    max_utterance_ms: int = 14_000
    #: Absolute floor: audio quieter than this is never speech, however adaptive
    #: the noise estimate becomes in a silent room.
    absolute_floor_db: float = -62.0


@dataclass(slots=True)
class Utterance:
    """A contiguous stretch of speech, timestamped from session start."""

    audio: np.ndarray
    start_time: float
    end_time: float

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


def _db(value: float) -> float:
    return 20.0 * np.log10(max(value, 1e-9))


@dataclass
class VoiceSegmenter:
    """Streaming VAD segmenter. Feed blocks, receive complete utterances."""

    config: VADConfig = field(default_factory=VADConfig)
    enabled: bool = True

    def __post_init__(self) -> None:
        cfg = self.config
        self._frame_len = int(cfg.sample_rate * cfg.frame_ms / 1000)
        self._hangover_frames = max(1, cfg.hangover_ms // cfg.frame_ms)
        self._pre_roll_frames = max(1, cfg.pre_roll_ms // cfg.frame_ms)
        self._min_frames = max(1, cfg.min_utterance_ms // cfg.frame_ms)
        self._max_frames = max(self._min_frames + 1, cfg.max_utterance_ms // cfg.frame_ms)

        self._residual = np.zeros(0, dtype=np.float32)
        self._pre_roll: deque[np.ndarray] = deque(maxlen=self._pre_roll_frames)
        self._noise_history: deque[float] = deque(maxlen=150)  # ~3 s of frames
        self._voiced: list[np.ndarray] = []
        self._silence_run = 0
        self._trigger_run = 0
        self._in_speech = False
        self._frames_seen = 0
        self._utterance_start_frame = 0
        # Frames that were actually above threshold. The minimum-length test
        # uses this, not the buffer length: a door slam followed by hangover
        # silence would otherwise look like a second of speech and get sent to
        # Whisper, which is exactly where hallucinations come from.
        self._loud_frames = 0
        self._silence_run_at_finish = 0
        self._tail_pad_frames = max(1, 250 // cfg.frame_ms)

    # -- introspection for the status bar ---------------------------------
    @property
    def in_speech(self) -> bool:
        return self._in_speech

    @property
    def noise_floor_db(self) -> float:
        if not self._noise_history:
            return self.config.absolute_floor_db
        return float(np.percentile(np.array(self._noise_history), 25))

    @property
    def pending_audio(self) -> np.ndarray:
        """Audio of the utterance currently in progress (for partial decodes)."""
        if not self._voiced:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self._voiced)

    @property
    def pending_start_time(self) -> float:
        return self._utterance_start_frame * self.config.frame_ms / 1000.0

    # -- streaming API -----------------------------------------------------
    def feed(self, block: np.ndarray) -> list[Utterance]:
        """Consume one captured block, returning any utterances it completed."""
        block = np.asarray(block, dtype=np.float32).ravel()
        if block.size == 0:
            return []

        data = np.concatenate([self._residual, block]) if self._residual.size else block
        usable = (data.size // self._frame_len) * self._frame_len
        self._residual = data[usable:].copy()

        finished: list[Utterance] = []
        for offset in range(0, usable, self._frame_len):
            frame = data[offset : offset + self._frame_len]
            utterance = self._feed_frame(frame)
            if utterance is not None:
                finished.append(utterance)
        return finished

    def _feed_frame(self, frame: np.ndarray) -> Utterance | None:
        cfg = self.config
        self._frames_seen += 1
        energy_db = _db(float(np.sqrt(np.mean(np.square(frame, dtype=np.float64)))))

        if not self.enabled:
            # VAD disabled: accumulate everything and cut on the length ceiling.
            self._voiced.append(frame.copy())
            self._loud_frames += 1
            if not self._in_speech:
                self._in_speech = True
                self._utterance_start_frame = self._frames_seen - 1
            if len(self._voiced) >= self._max_frames:
                self._silence_run_at_finish = 0
                return self._finish_utterance()
            return None

        floor = max(self.noise_floor_db, cfg.absolute_floor_db)
        is_loud = energy_db > floor + cfg.start_margin_db
        still_loud = energy_db > floor + cfg.stop_margin_db

        if not self._in_speech:
            # Only quiet frames update the noise estimate, so a long sentence
            # cannot drag the floor up and gate itself out.
            self._noise_history.append(energy_db)
            self._pre_roll.append(frame.copy())
            self._trigger_run = self._trigger_run + 1 if is_loud else 0
            if self._trigger_run >= 2:
                self._in_speech = True
                self._voiced = list(self._pre_roll)
                self._utterance_start_frame = self._frames_seen - len(self._voiced)
                self._pre_roll.clear()
                self._silence_run = 0
                self._trigger_run = 0
                self._loud_frames = 2  # the two frames that triggered the start
            return None

        self._voiced.append(frame.copy())
        if still_loud:
            self._silence_run = 0
            self._loud_frames += 1
        else:
            self._silence_run += 1
            self._noise_history.append(energy_db)

        if self._silence_run >= self._hangover_frames:
            self._silence_run_at_finish = self._silence_run
            return self._finish_utterance()
        if len(self._voiced) >= self._max_frames:
            self._silence_run_at_finish = self._silence_run
            return self._finish_utterance()
        return None

    def _finish_utterance(self) -> Utterance | None:
        frames = self._voiced
        loud_frames = self._loud_frames
        self._voiced = []
        self._loud_frames = 0
        self._in_speech = False
        self._silence_run = 0
        self._pre_roll.clear()

        if loud_frames < self._min_frames:
            return None

        # Trim the hangover tail, keeping a short pad so the final consonant
        # is not clipped. Less silence in means fewer hallucinations out.
        keep_tail = max(0, self._silence_run_at_finish - self._tail_pad_frames)
        if keep_tail:
            frames = frames[:-keep_tail] or frames

        audio = np.concatenate(frames)
        start = self._utterance_start_frame * self.config.frame_ms / 1000.0
        end = start + audio.size / self.config.sample_rate
        return Utterance(audio=audio, start_time=start, end_time=end)

    def flush(self) -> Utterance | None:
        """Emit whatever speech is buffered. Called when recording stops."""
        if self._residual.size and self._in_speech:
            self._voiced.append(self._residual.copy())
            self._residual = np.zeros(0, dtype=np.float32)
        if not self._voiced:
            return None
        # A trailing fragment is worth keeping even below the minimum length.
        audio = np.concatenate(self._voiced)
        start = self._utterance_start_frame * self.config.frame_ms / 1000.0
        end = start + audio.size / self.config.sample_rate
        self._voiced = []
        self._loud_frames = 0
        self._in_speech = False
        if audio.size < self.config.sample_rate * 0.2:
            return None
        return Utterance(audio=audio, start_time=start, end_time=end)
