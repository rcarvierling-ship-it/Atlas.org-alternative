"""In-process stand-ins for whisper.cpp and Ollama.

Both speak the real wire protocols — multipart ``/inference`` for whisper.cpp,
NDJSON-streamed ``/api/generate`` for Ollama — so the code under test is the
production code path, not a mock of it. That is what lets the full pipeline be
exercised on a machine that has neither binary installed.

They are also used by ``scripts/demo.py`` to drive the real TUI without any
local models.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Iterable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DEFAULT_TRANSCRIPT_LINES = [
    "Today we are looking at bacterial cell structure, starting with the cell membrane.",
    "The cell membrane consists primarily of a phospholipid bilayer, and it controls what "
    "moves into and out of the cell.",
    "Now, gram-positive bacteria have a thick peptidoglycan layer. This distinction is going "
    "to be on the exam, so make sure you remember it.",
    "Gram-negative bacteria, by contrast, have a much thinner peptidoglycan layer plus an "
    "outer membrane containing lipopolysaccharide.",
    "For example, Escherichia coli is gram-negative, while Staphylococcus aureus is "
    "gram-positive.",
]


class _Handler(BaseHTTPRequestHandler):
    """Routes requests to the owning server's handler table."""

    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        """Silence the default stderr logging."""

    def do_GET(self) -> None:  # noqa: N802
        route = self.server.routes.get(("GET", self.path.split("?")[0]))
        if route is None:
            self._send(404, b"not found", "text/plain")
            return
        status, body, content_type = route(self, b"")
        self._send(status, body, content_type)

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        payload = self.rfile.read(length) if length else b""
        route = self.server.routes.get(("POST", self.path.split("?")[0]))
        if route is None:
            self._send(404, b"not found", "text/plain")
            return
        result = route(self, payload)
        if isinstance(result, tuple):
            status, body, content_type = result
            self._send(status, body, content_type)

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_stream(self, chunks: Iterable[bytes]) -> None:
        """Stream NDJSON with chunked transfer encoding, like Ollama does."""
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        for chunk in chunks:
            self.wfile.write(f"{len(chunk):X}\r\n".encode())
            self.wfile.write(chunk)
            self.wfile.write(b"\r\n")
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, routes: dict) -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        self.routes = routes


class FakeServer:
    """Base class: a threaded HTTP server on an ephemeral port."""

    def __init__(self) -> None:
        self._server = _Server(self._routes())
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self.requests: list[str] = []

    def _routes(self) -> dict:
        raise NotImplementedError

    @property
    def url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def __enter__(self) -> FakeServer:
        self._thread.start()
        return self

    def __exit__(self, *exc_info) -> None:
        self.stop()

    def start(self) -> FakeServer:
        self._thread.start()
        return self

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


class FakeWhisperServer(FakeServer):
    """Speaks whisper.cpp's HTTP API.

    Each ``/inference`` call returns the next line of ``lines``, cycling once
    exhausted, so a WAV of N utterances yields a deterministic transcript.
    """

    def __init__(self, lines: list[str] | None = None, *, fail_after: int | None = None) -> None:
        self.lines = list(lines or DEFAULT_TRANSCRIPT_LINES)
        self.fail_after = fail_after
        self.call_count = 0
        self._lock = threading.Lock()
        super().__init__()

    def _routes(self) -> dict:
        return {
            ("GET", "/"): self._index,
            ("POST", "/inference"): self._inference,
        }

    def _index(self, handler, body):  # noqa: ANN001, ARG002
        return 200, b"whisper.cpp server", "text/plain"

    def _inference(self, handler, body: bytes):  # noqa: ANN001
        with self._lock:
            self.call_count += 1
            index = self.call_count - 1
            if self.fail_after is not None and self.call_count > self.fail_after:
                return 500, b'{"error": "simulated whisper failure"}', "application/json"
            text = self.lines[index % len(self.lines)] if self.lines else ""
        self.requests.append(f"inference:{len(body)}")
        return 200, json.dumps({"text": text}).encode(), "application/json"


class FakeOllamaServer(FakeServer):
    """Speaks Ollama's ``/api/version``, ``/api/tags`` and ``/api/generate``.

    ``responder`` receives the request body and returns the text to stream back,
    which lets a test decide what the "model" says for a given prompt. The
    default responder produces a valid note delta derived from the transcript it
    was given, so notes genuinely reflect the speech that reached it.
    """

    def __init__(
        self,
        *,
        models: list[str] | None = None,
        responder: Callable[[dict], str] | None = None,
        available: bool = True,
    ) -> None:
        self.models = models or ["qwen3:8b", "llama3.2:3b"]
        self.responder = responder or default_note_responder
        self.available = available
        self.prompts: list[dict] = []
        super().__init__()

    def _routes(self) -> dict:
        return {
            ("GET", "/api/version"): self._version,
            ("GET", "/api/tags"): self._tags,
            ("POST", "/api/generate"): self._generate,
        }

    def _version(self, handler, body):  # noqa: ANN001, ARG002
        if not self.available:
            return 503, b'{"error":"down"}', "application/json"
        return 200, json.dumps({"version": "0.5.0-fake"}).encode(), "application/json"

    def _tags(self, handler, body):  # noqa: ANN001, ARG002
        if not self.available:
            return 503, b'{"error":"down"}', "application/json"
        payload = {
            "models": [
                {
                    "name": name,
                    "size": 4_800_000_000,
                    "details": {"parameter_size": "8B", "quantization_level": "Q4_K_M"},
                }
                for name in self.models
            ]
        }
        return 200, json.dumps(payload).encode(), "application/json"

    def _generate(self, handler, body: bytes):  # noqa: ANN001
        if not self.available:
            return 503, b'{"error":"down"}', "application/json"
        request = json.loads(body or b"{}")
        self.prompts.append(request)
        text = self.responder(request)

        # Stream in small pieces so the streaming parser is genuinely exercised.
        chunks = []
        step = max(1, len(text) // 8)
        for start in range(0, len(text), step):
            chunks.append(
                json.dumps({"response": text[start : start + step], "done": False}).encode() + b"\n"
            )
        chunks.append(json.dumps({"response": "", "done": True}).encode() + b"\n")
        handler.send_stream(chunks)
        return None


def default_note_responder(request: dict) -> str:
    """Produce a plausible note delta (or final guide) for a prompt.

    Deliberately derived from the prompt's transcript rather than canned, so a
    test that asserts "the notes mention peptidoglycan" is really asserting that
    the transcript reached the model.
    """
    prompt = request.get("prompt", "")
    if "final study guide" in prompt or "Executive Summary" in prompt:
        return _final_markdown(prompt)

    transcript = _extract_between(prompt, 'NEW TRANSCRIPT SINCE THE LAST UPDATE:\n"""', '"""')
    if not transcript:
        # Consolidation prompt: echo a compacted structure.
        return json.dumps(
            {
                "current_topic": "Bacterial Cell Structure",
                "summary": "Consolidated notes on bacterial cell structure.",
                "topics": ["Bacterial Cell Structure"],
                "key_points": [{"text": "Consolidated key point", "starred": True}],
                "definitions": [],
                "key_terms": [],
                "examples": [],
                "formulas": [],
                "questions": [],
                "unclear_points": [],
                "important_details": [],
            }
        )

    sentences = [part.strip() for part in transcript.split(".") if part.strip()]
    key_points = [
        {"text": sentence + ".", "starred": "exam" in sentence.lower()}
        for sentence in sentences[:3]
    ]
    definitions = []
    if "peptidoglycan" in transcript.lower():
        definitions.append(
            {"term": "Peptidoglycan", "definition": "Structural polymer in bacterial cell walls."}
        )
    topic = "Gram Staining" if "gram" in transcript.lower() else "Cell Structure"

    return json.dumps(
        {
            "current_topic": topic,
            "summary": "The lecture covers bacterial cell structure and gram staining.",
            "new_topics": [topic],
            "key_points": key_points,
            "important_details": [],
            "examples": (
                [{"text": "E. coli is gram-negative; S. aureus is gram-positive."}]
                if "coli" in transcript.lower()
                else []
            ),
            "formulas": [],
            "questions": [],
            "unclear_points": [],
            "definitions": definitions,
            "key_terms": [{"term": "Bilayer", "definition": "Two layers of phospholipids."}]
            if "bilayer" in transcript.lower()
            else [],
        }
    )


def _final_markdown(prompt: str) -> str:
    mentions_gram = "gram" in prompt.lower()
    return f"""# Bacterial Cell Structure and Gram Staining

## Executive Summary
The lecture introduced bacterial cell structure, focusing on the membrane and the
cell wall differences revealed by gram staining.

## Main Topics

### Cell Membrane
- Composed primarily of a phospholipid bilayer.
- Controls movement into and out of the cell.

### Gram Staining
- Gram-positive bacteria have a thick peptidoglycan layer.
- Gram-negative bacteria have a thin layer plus an outer membrane.

## Key Concepts
- The cell wall determines the gram stain result.

## Definitions
- **Peptidoglycan** — structural polymer found in bacterial cell walls.

## Important Details
- {"The gram distinction was flagged as exam material." if mentions_gram else "See transcript."}

## Examples
- E. coli (gram-negative), S. aureus (gram-positive).

## Questions to Review
- Why does peptidoglycan thickness change the stain result?

## Exam / Quiz Worthy Material
- The gram-positive versus gram-negative distinction.

## Key Takeaways
- Cell wall structure is the basis of gram classification.
"""


def _extract_between(text: str, start_marker: str, end_marker: str) -> str:
    start = text.find(start_marker)
    if start == -1:
        return ""
    start += len(start_marker)
    end = text.find(end_marker, start)
    return text[start:end].strip() if end != -1 else text[start:].strip()


def unavailable_responder(request: dict) -> str:  # noqa: ARG001
    """Responder that returns text the JSON parser cannot use."""
    return "I'm sorry, I can't help with that."
