"""Small HTTP wrapper around the real Olimpia engine in agent.py: FEN in, move out.

Standard library only, to keep the container's memory low. Each request builds a fresh
Engine, so nothing about one visitor's game carries over to the next request.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import chess

PORT = int(os.environ.get("PORT", "8080"))
THINK_MS = 500  # fixed per move, visitors cannot change it
MAX_BODY_BYTES = 512
MAX_FEN_LENGTH = 100
PER_IP_PER_MINUTE = 40
GLOBAL_PER_MINUTE = 300
MAX_WAITING_SEARCHES = 4
MAX_OPEN_REQUESTS = 32
SOCKET_TIMEOUT_S = 10
ALLOWED_ORIGINS = {
    origin.strip()
    for origin in os.environ.get("ALLOWED_ORIGINS", "http://localhost:5173").split(",")
    if origin.strip()
}

FEN_SHAPE = re.compile(
    r"^[pnbrqkPNBRQK1-8/]{15,71} [wb] (-|[KQkqA-Ha-h]{1,4}) (-|[a-h][36]) \d{1,3} \d{1,4}$"
)

agent = None
engine_state = "warming"
engine_error = ""
search_lock = threading.Lock()
open_requests = threading.BoundedSemaphore(MAX_OPEN_REQUESTS)
waiting = 0
waiting_lock = threading.Lock()


def load_engine() -> None:
    """Importing agent.py compiles and warms the search, which takes a while on a cold start."""
    global agent, engine_state, engine_error
    started = time.perf_counter()
    try:
        import agent as loaded

        agent = loaded
        engine_state = "ready"
        print(f"engine ready in {time.perf_counter() - started:.1f}s", flush=True)
    except Exception as exc:  # reported through /healthz, the page says the engine is down
        engine_state = "failed"
        engine_error = type(exc).__name__
        print(f"engine failed to load: {exc!r}", flush=True)


class RateLimiter:
    """Sliding one-minute window, per client and overall. Memory stays bounded."""

    MAX_TRACKED = 5000

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.clients: dict[str, deque[float]] = {}
        self.overall: deque[float] = deque()

    def allow(self, client: str) -> bool:
        now = time.monotonic()
        cutoff = now - 60.0
        with self.lock:
            while self.overall and self.overall[0] < cutoff:
                self.overall.popleft()
            if len(self.overall) >= GLOBAL_PER_MINUTE:
                return False
            hits = self.clients.get(client)
            if hits is None:
                if len(self.clients) >= self.MAX_TRACKED:
                    self.clients = {k: v for k, v in self.clients.items() if v and v[-1] >= cutoff}
                    if len(self.clients) >= self.MAX_TRACKED:
                        return False
                hits = self.clients[client] = deque()
            while hits and hits[0] < cutoff:
                hits.popleft()
            if len(hits) >= PER_IP_PER_MINUTE:
                return False
            hits.append(now)
            self.overall.append(now)
            return True


limiter = RateLimiter()


def parse_position(raw: bytes) -> chess.Board | str:
    """Return a playable board, or a short message saying why the request was refused."""
    try:
        payload = json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        return "body must be JSON"
    if not isinstance(payload, dict) or set(payload) != {"fen"}:
        return 'body must be {"fen": "<position>"}'
    fen = payload["fen"]
    if not isinstance(fen, str) or len(fen) > MAX_FEN_LENGTH or not FEN_SHAPE.match(fen):
        return "fen is not in the expected format"
    try:
        board = chess.Board(fen)
    except ValueError:
        return "fen could not be parsed"
    if not board.is_valid():
        return "fen is not a legal position"
    if board.is_game_over(claim_draw=False):
        return "the game is already over in this position"
    return board


def search(board: chess.Board) -> dict:
    """Run the engine's own search for the fixed think time and report what it says it did."""
    engine = agent.Engine()
    started = time.perf_counter()
    deadline = started + THINK_MS / 1000.0
    source = "search"
    move = None
    try:
        engine.remember(board)
        move = engine.search(board, deadline)
        if not board.is_legal(move):
            move = None
    except Exception as exc:
        print(f"compiled search failed: {exc!r}", flush=True)
        move = None
    if move is None:
        # the same safety net get_move has in agent.py
        source = "fallback"
        move = agent.fallback_move(board, deadline)
    elapsed = time.perf_counter() - started
    nodes = int(engine.nodes)
    score = int(engine.st[agent.ROOT_SCORE])
    return {
        "move": move.uci(),
        "san": board.san(move),
        "stats": {
            "source": source,
            "nodes": nodes,
            "depth": int(engine.depth),
            "time_ms": round(elapsed * 1000.0, 1),
            "nps": round(nodes / elapsed) if elapsed > 0 else 0,
            "score_cp": score if abs(score) < agent.MATE_THRESHOLD else None,
            # positive when Olimpia is the side delivering mate
            "mate_in_plies": (
                (agent.MATE - abs(score)) * (1 if score > 0 else -1)
                if abs(score) >= agent.MATE_THRESHOLD
                else None
            ),
        },
        "think_ms": THINK_MS,
    }


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "olimpia"
    sys_version = ""
    timeout = SOCKET_TIMEOUT_S

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass  # no request logs, so no visitor addresses are kept

    def send_json(self, status: int, body: dict, extra: dict[str, str] | None = None) -> None:
        data = json.dumps(body, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        origin = self.headers.get("Origin", "")
        if origin in ALLOWED_ORIGINS:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(data)

    def client_key(self) -> str:
        # Railway's edge sets X-Real-IP; the global limit covers anyone who manages to vary it
        return self.headers.get("X-Real-IP") or self.client_address[0]

    def do_OPTIONS(self) -> None:  # noqa: N802
        origin = self.headers.get("Origin", "")
        self.send_response(204)
        if origin in ALLOWED_ORIGINS:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Max-Age", "600")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/healthz":
            self.send_json(
                200 if engine_state != "failed" else 500,
                {"engine": engine_state, "think_ms": THINK_MS},
            )
        else:
            self.send_json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        self.close_connection = True
        if self.path != "/move":
            self.send_json(404, {"error": "not found"})
            return
        if self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json":
            self.send_json(415, {"error": "content type must be application/json"})
            return
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            self.send_json(411, {"error": "content length required"})
            return
        if length < 0 or length > MAX_BODY_BYTES:
            self.send_json(413, {"error": f"body must be at most {MAX_BODY_BYTES} bytes"})
            return
        if not limiter.allow(self.client_key()):
            self.send_json(429, {"error": "too many requests"}, {"Retry-After": "30"})
            return
        raw = self.rfile.read(length)
        if len(raw) != length:
            self.send_json(400, {"error": "incomplete body"})
            return
        board = parse_position(raw)
        if isinstance(board, str):
            self.send_json(422, {"error": board})
            return
        if engine_state != "ready":
            status = 503 if engine_state == "warming" else 500
            self.send_json(status, {"error": f"engine {engine_state}"}, {"Retry-After": "3"})
            return

        global waiting
        with waiting_lock:
            if waiting >= MAX_WAITING_SEARCHES:
                busy = True
            else:
                busy = False
                waiting += 1
        if busy:
            self.send_json(503, {"error": "engine busy"}, {"Retry-After": "2"})
            return
        try:
            with search_lock:  # one search at a time, the service has one small core to spare
                result = search(board)
        finally:
            with waiting_lock:
                waiting -= 1
        self.send_json(200, result)


class Server(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 32

    def process_request(self, request, client_address) -> None:  # type: ignore[no-untyped-def]
        if not open_requests.acquire(blocking=False):
            self.shutdown_request(request)
            return
        super().process_request(request, client_address)

    def process_request_thread(self, request, client_address) -> None:  # type: ignore[no-untyped-def]
        try:
            super().process_request_thread(request, client_address)
        finally:
            open_requests.release()


if __name__ == "__main__":
    threading.Thread(target=load_engine, daemon=True).start()
    print(f"listening on {PORT}", flush=True)
    Server(("0.0.0.0", PORT), Handler).serve_forever()
