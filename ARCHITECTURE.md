# Architecture

Lectern turns a live lecture into structured notes, entirely on one machine.
This document explains how the pieces fit together and — more usefully — *why*
they are arranged this way.

```
Audio → STT → Transcript → Notes → Storage → UI
```

---

## The pipeline

Everything during a live session is orchestrated by `RecordingPipeline`
(`src/lectern/pipeline.py`) as independent asyncio tasks joined by bounded
queues:

```
                    AudioSource.frames()          bounded queue, drop-oldest
                            │
                    ┌───────┴────────┐
                    │   audio task   │
                    └───┬────────┬───┘
                        │        │
             AudioRecorder   VoiceSegmenter (VAD)
             (WAV on disk)         │
                                   │  utterances
                        ┌──────────┴──────────┐
                        │  transcription task │──→ whisper.cpp (HTTP, persistent)
                        └──────────┬──────────┘
                                   │  TranscriptSegment
             ┌─────────────────────┼──────────────────────┐
             │                     │                      │
    append + flush to        UI callback            NoteScheduler
    transcript.jsonl        (transcript pane)    (debounce/backpressure)
                                                          │
                                              ┌───────────┴───────────┐
                                              │      notes task       │──→ Ollama (HTTP, streaming)
                                              └───────────┬───────────┘
                                                          │  NoteState
                                             ┌────────────┴────────────┐
                                             │                         │
                                   notes-live.json/md            UI callback
                                                                (notes + topics)
```

A fifth task (`persist`) flushes metadata and note state every 20 seconds so a
crash costs seconds, not a lecture.

### Design rules the code enforces

**The transcript is the priority stream.** If Ollama dies, note generation
pauses and everything else continues; the user is told once, in plain language,
and notes resume automatically when it returns. If note generation falls behind,
transcription is unaffected — they share nothing but a queue. If disk writes
fail, the session keeps running and says so.

**Nothing blocks the event loop.** whisper.cpp and Ollama are reached over HTTP
with async clients. Audio arrives on PortAudio's real-time callback, which does
the minimum possible work and hands the block to the loop via
`call_soon_threadsafe`. Blocking probes (device enumeration, filesystem scans)
run in threads. Keystrokes stay instant while both models are busy.

**The session clock is the audio clock.** Elapsed time is derived from captured
samples, not wall time. Pausing stops the clock; a file-fed demo session
produces exactly the timestamps the recording implies; a sleeping Mac does not
add an hour to the duration.

**Bounded memory.** Audio goes to disk as it arrives rather than accumulating.
The transcript widget renders at most 400 segments (the full transcript is
always on disk). Prompt size is capped by the scheduler and by the note
digest. A three-hour lecture has no unbounded growth anywhere.

**Backpressure, not stalling.** Every queue is bounded. The audio queue drops
the *oldest* block if a consumer falls behind — a capture callback must never
block. The utterance queue holds several minutes of speech and, if it ever
fills, tells the user their Whisper model is too slow for the machine.

---

## Layers

### Audio (`lectern/audio/`)

`AudioSource` is the interface: start, stop, and an async iterator of 16 kHz
mono float32 blocks. Four implementations:

| Class | Source |
|---|---|
| `MicrophoneSource` | PortAudio/CoreAudio input device |
| `SystemAudioSource` | the Swift ScreenCaptureKit helper, over a pipe |
| `CombinedAudioSource` | both, mixed sample-for-sample |
| `FileAudioSource` | a WAV file replayed in real time (demo/tests) |

Format normalisation happens as early as possible, so nothing downstream ever
sees anything but 16 kHz mono float32.

`CombinedAudioSource` buffers each leg and emits a mixed block when both have
enough samples; if one leg stalls for 750 ms it passes the survivor through
rather than going silent. Half a recording beats none.

### Voice activity detection (`lectern/audio/vad.py`)

Continuous transcription wastes compute and makes Whisper hallucinate YouTube
outros over silence. Instead, `VoiceSegmenter` cuts audio into utterances with a
pure-numpy adaptive energy gate:

- A rolling percentile of *quiet* frames estimates the noise floor, so the gate
  adapts to a silent study room or a noisy lecture hall by itself. Only quiet
  frames update the estimate, so a long sentence cannot drag the floor up and
  gate itself out.
- Speech starts when energy exceeds the floor by a margin for two consecutive
  frames; 300 ms of pre-roll is prepended so the first syllable is not clipped.
- Speech ends after 700 ms below threshold, so a mid-sentence breath does not
  split an utterance. The hangover tail is trimmed before transcription.
- The minimum-length test counts *voiced* frames, not buffered frames — so a
  door slam followed by hangover silence is discarded rather than sent to
  Whisper as "a second of speech".
- Continuous speech is force-cut at 14 seconds so a monologue still feeds notes.

No C extension: one fewer native dependency, and behaviour deterministic enough
to unit test with synthetic audio.

### Transcription (`lectern/transcription/`)

`WhisperCppBackend` drives whisper.cpp through its bundled **HTTP server**
rather than the one-shot `whisper-cli`. That is the entire latency story: the
server loads the ggml weights once, keeps them resident and Metal-warm, and
answers each utterance in the time it takes to decode a few seconds of audio.
Spawning `whisper-cli` per chunk would re-read hundreds of megabytes of weights
every few seconds.

The server is a child process owned by the backend — started, health-checked
and torn down with the session. Users who run their own can point
`transcription.server_url` at it, in which case Lectern attaches and never
touches the process lifecycle.

Each `/inference` request sets `no_context`, so decoder state never carries
across utterances (whisper.cpp's main source of runaway repetition). It is sent
per request rather than as a startup flag: `whisper-server` does not accept
`--no-context`, and exits on an argument it does not recognise, so
`build_server_command` stays minimal on purpose. Known silence hallucinations
are filtered before a segment can reach the transcript.

`TranscriptSegment` is the unit of currency for the whole application. Only
*final* segments are persisted or sent to the LLM; partial hypotheses are
display-only, are never written to disk, and are skipped whenever finished
speech is already queued.

### Notes (`lectern/notes/`)

This is the part most worth understanding.

**`NoteState` is the working memory.** It is never regenerated from the
transcript. Each cycle the model receives the current state (as a bounded
digest) plus only the speech since the last cycle, and returns a *delta*. The
delta is merged in Python — not in the prompt.

That split is deliberate: merging in code means a bad model response can add
nothing useful, but it can never silently delete a fact the lecturer already
said. De-duplication is loose-matched on normalised text, so restating a bullet
in different words does not accumulate noise, while a later "this is on the
exam" *can* upgrade an existing bullet to starred.

**`NoteScheduler` decides when to spend an update.** It is a pure,
clock-injected object with no async in it, so its timing rules are tested
exactly rather than by sleeping:

- never two updates at once; speech arriving mid-update is buffered for next time
- run when the interval has elapsed *and* enough new words accumulated — waking
  a model to process "um, okay, so" is wasted latency
- run early when a lot of speech has piled up, so a fast talker is not stuck
  behind the interval
- a marker forces the next update, because the student just said "pay attention here"
- never hand the model more than `max_context_words`; the remainder stays
  buffered — the transcript is deferred, never dropped

**Consolidation** runs every few minutes over the *notes*, never the raw
transcript (which stays on disk as the permanent record). Re-sending the
transcript periodically is exactly the unbounded context growth this design
avoids. The result is safety-checked before it is accepted: a consolidation that
drops more than 45% of items, loses any starred item, or loses one of the
student's own notes is rejected and the original kept. Tidier notes are worth
nothing if they silently dropped a formula.

**Finalization** is a separate, slower pass allowed to take a minute. If the
transcript exceeds the context window it goes through hierarchical map/reduce:
overlapping chunks are each summarised in full detail, then merged (recursively
if needed). Nothing is truncated — the first ten minutes survive exactly like
the last ten.

**Structured output.** Prompts pass a JSON schema to Ollama's `format`
parameter, which constrains sampling to that grammar. That is what makes typed
notes workable on an 8B model; without it, small models drift out of JSON
several times an hour and every one of those updates is a dropped cycle. On top
of that, `llm/parsing.py` recovers the first balanced JSON object from responses
wrapped in `<think>` blocks, markdown fences, or a friendly preamble. A response
that cannot be parsed leaves the previous notes untouched.

### Storage (`lectern/sessions/`)

Sessions are plain files in a folder; SQLite is only an index over them and can
be rebuilt by rescanning (`lectern reindex`). This keeps the source of truth
human-readable and means a corrupt database is an inconvenience, not a loss.

Durability is the point of `storage.py`:

- Every finalized segment is **appended and flushed** immediately, with an
  `fsync` every five segments. A crash costs seconds.
- A torn final line (a crash mid-append) is skipped on load; every complete
  segment is kept.
- Everything else — metadata, note state, markers — is written **atomically**
  via a temp file and `os.replace`, so a crash mid-write leaves the previous
  good version rather than a truncated file.
- Audio is streamed to the WAV file as it arrives, so a multi-hour lecture costs
  constant memory and survives a crash up to the last flush.

**Recovery** (`recovery.py`): a session whose `session.json` still says
`recording` was interrupted. On startup Lectern finds it and offers Resume
(continue recording into it, continuing segment ids and timestamps), Recover
(close it out, keeping everything), Finalize (close it out and synthesise), or
Discard.

**Search** uses SQLite FTS5 with BM25 ranking and snippet extraction, degrading
to `LIKE` matching over the session files when the local SQLite lacks FTS5.

### UI (`lectern/screens/`, `lectern/widgets/`)

Textual, with a `Screen` per destination and reactive widgets. The recording
screen owns a pipeline and does no heavy work itself — it only reacts to
callbacks, which is what keeps the interface responsive under load. Every UI
callback is wrapped so a rendering bug cannot take down a recording in progress.

Notes render as a single Rich renderable rather than a widget tree: notes are
rewritten wholesale each update, and swapping one renderable is both faster and
visually calmer than mounting and unmounting dozens of widgets.

---

## Replaceable interfaces

Each external dependency sits behind an abstract base class, so a second
implementation is additive:

| Interface | Today | Where |
|---|---|---|
| `TranscriptionBackend` | `WhisperCppBackend` | `transcription/base.py` |
| `LLMBackend` | `OllamaBackend` | `llm/base.py` |
| `AudioSource` | mic / system / combined / file | `audio/base.py` |
| `Exporter` | Markdown, text, JSON | `sessions/export.py` |

`LLMBackend` implementations must be local. Nothing in Lectern may fall back to
a hosted API: if the local model is unavailable, the correct behaviour is to
keep transcribing and tell the user notes are paused.

---

## Testing strategy

The interesting layers are tested against the *real* code paths, with the two
external servers replaced by in-process fakes that speak the actual wire
protocols — multipart `/inference` for whisper.cpp, NDJSON-streamed
`/api/generate` for Ollama (`tests/fakes.py`). A WAV file stands in for the
microphone, exactly as `lectern record --file` does.

That means `tests/test_pipeline.py` and `tests/test_acceptance.py` exercise VAD,
the HTTP clients, the scheduler, the merge logic, persistence and the screens as
shipped — including the failure paths (Ollama dying mid-lecture, whisper failing,
unparseable model output, a hard crash mid-session).

Pure logic — the merge rules, the scheduler's timing, response parsing, VAD
segmentation — is tested directly, without a model in the loop.
