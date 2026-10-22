"""One long-lived Node scoring worker per Python process.

`score_worker.cjs` loads the flatppl-js engine once and answers JSON-line
requests, replacing one Node process per scored case (each paid the engine load
and a macOS code-signing check). It runs the same cores as the one-shot scripts
(`score_core.cjs`), so a request scores as its script would.

The worker is a lazy per-process singleton (`shared()`), so each pytest-xdist
worker owns its own. A request that hangs, or a worker that dies, never takes the
run with it: the worker is killed and the next request starts a fresh one.
Environment knobs: FLATPPL_SCORE_TIMEOUT (seconds per request, default 300) and
FLATPPL_SCORE_RSS_LIMIT (bytes; the worker restarts after a reply once its
resident set passes it, default 3 GiB).
"""
from __future__ import annotations

import atexit
import json
import os
import select
import subprocess
import tempfile
import threading
import time
from pathlib import Path

from flatppl_testsuite.config import CONFIG

_TIMEOUT = float(os.environ.get("FLATPPL_SCORE_TIMEOUT", "300"))
_RSS_LIMIT = int(os.environ.get("FLATPPL_SCORE_RSS_LIMIT", str(3 << 30)))
# Engine load is slow on a loaded machine; the first reply is a ready line.
_START_TIMEOUT = 120.0


class ScoreWorkerFailed(RuntimeError):
    """The scoring process or its reply stream failed, not an individual model."""


def engine_dir() -> str:
    return str(CONFIG.flatppl_js_dir / "packages" / "engine")


class ScoreWorker:
    """A Node worker plus the pipe protocol to it. One request in flight at a time."""

    def __init__(self, node_bin: str | None = None, script: Path | None = None,
                 engine: str | None = None) -> None:
        self.node_bin = node_bin or CONFIG.node_bin
        self.script = str(script or CONFIG.score_worker)
        self.engine = engine or engine_dir()
        self._lock = threading.Lock()
        self._process: subprocess.Popen | None = None
        self._errors = None
        self._buffer = b""
        self._next_id = 0
        self._pid = os.getpid()
        self.started = 0  # Node processes this worker has started
        self.loaded_engine: str | None = None  # engine directory the live worker reported loading

    # --- lifecycle -------------------------------------------------------
    def _spawn(self, label: str) -> None:
        self._discard()
        self._errors = tempfile.TemporaryFile(mode="w+")
        # The child stays in the caller's process group, so a timeout that
        # kills the run also stops Node.
        self._process = subprocess.Popen(
            [self.node_bin, self.script, "--engine", self.engine],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._errors,
        )
        self._buffer = b""
        self.started += 1
        try:
            ready = self._read_line(time.monotonic() + _START_TIMEOUT, label, "start")
        except ScoreWorkerFailed:
            self._discard()
            raise
        if not ready.get("ready"):
            self._discard()
            raise ScoreWorkerFailed(f"{label} failed: score worker sent no ready line: {ready}")
        self.loaded_engine = ready.get("engine")

    def _stderr_tail(self) -> str:
        if self._errors is None:
            return ""
        try:
            self._errors.seek(0)
            return self._errors.read().strip()[-4000:]
        except (OSError, ValueError):
            return ""

    def _discard(self) -> None:
        process, self._process = self._process, None
        if process is not None:
            try:
                process.stdin.close()
            except OSError:
                pass
            if process.poll() is None:
                process.kill()
            process.wait()
            process.stdout.close()
        if self._errors is not None:
            self._errors.close()
            self._errors = None

    def close(self) -> None:
        with self._lock:
            process = self._process
            if process is not None:
                try:
                    process.stdin.close()
                except OSError:
                    pass
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    pass
            self._discard()

    # --- protocol --------------------------------------------------------
    def _read_line(self, deadline: float, label: str, what: str) -> dict:
        fd = self._process.stdout.fileno()
        while b"\n" not in self._buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._discard()
                raise ScoreWorkerFailed(f"{label} failed: score worker timed out ({what})")
            ready, _, _ = select.select([fd], [], [], min(remaining, 1.0))
            if not ready:
                if self._process.poll() is not None and not select.select([fd], [], [], 0)[0]:
                    break
                continue
            chunk = os.read(fd, 1 << 20)
            if not chunk:
                break
            self._buffer += chunk
        if b"\n" not in self._buffer:
            tail = self._stderr_tail()
            code = self._process.poll() if self._process is not None else None
            self._discard()
            raise ScoreWorkerFailed(
                f"{label} failed: score worker exited"
                + (f" with code {code}" if code is not None else "") + (f": {tail}" if tail else ""))
        line, _, self._buffer = self._buffer.partition(b"\n")
        try:
            return json.loads(line)
        except ValueError as error:
            self._discard()
            raise ScoreWorkerFailed(
                f"{label} failed: invalid score worker reply: {line[:200]!r}") from error

    def request(self, op: str, label: str, timeout: float | None = None, **payload) -> dict:
        """Send one request and return its reply dict (`ok` says whether it scored)."""
        with self._lock:
            if self._process is None or self._process.poll() is not None:
                self._spawn(label)
            self._next_id += 1
            request = {"id": self._next_id, "op": op, **payload}
            data = (json.dumps(request) + "\n").encode()
            try:
                self._process.stdin.write(data)
                self._process.stdin.flush()
            except OSError:
                # The worker died between requests; nothing was scored. Start a
                # fresh one and send the request once more.
                self._spawn(label)
                self._process.stdin.write(data)
                self._process.stdin.flush()
            reply = self._read_line(time.monotonic() + (timeout or _TIMEOUT), label, op)
            if reply.get("id") != request["id"]:
                self._discard()
                raise ScoreWorkerFailed(f"{label} failed: score worker reply out of sequence")
            if reply.get("rss", 0) > _RSS_LIMIT:
                # Recycle a bloated worker; this reply is already in hand.
                try:
                    self._process.stdin.close()
                except OSError:
                    pass
                self._discard()
            return reply


_SHARED: ScoreWorker | None = None
_SHARED_KEY: tuple | None = None


def shared() -> ScoreWorker:
    """The process-wide worker for the configured Node, script and engine."""
    global _SHARED, _SHARED_KEY
    key = (os.getpid(), CONFIG.node_bin, str(CONFIG.score_worker), engine_dir())
    if _SHARED is None or _SHARED_KEY != key:
        if _SHARED is not None and _SHARED._pid == os.getpid():
            _SHARED.close()
        _SHARED, _SHARED_KEY = ScoreWorker(), key
    return _SHARED


@atexit.register
def _close_shared() -> None:
    if _SHARED is not None and _SHARED._pid == os.getpid():
        _SHARED.close()


# --- typed requests, each mirroring one one-shot script ----------------------

def score_binding_file(path: Path, binding: str, worker: ScoreWorker | None = None) -> float:
    """`score_flatpdl.cjs <path> <binding>`: one binding of a FlatPDL file."""
    reply = (worker or shared()).request(
        "binding", "score_flatpdl", model=os.path.abspath(path), binding=binding)
    if not reply["ok"]:
        raise RuntimeError(f"score_flatpdl failed: {reply['stderr'].strip()}")
    return float(reply["value"])


def score_js_file(path: Path, binding: str, theta: str, worker: ScoreWorker | None = None) -> float:
    """`score_js.cjs <path> <binding> <theta>`: a measure binding's log-density at a point."""
    reply = (worker or shared()).request(
        "js", "score_js", model=os.path.abspath(path), binding=binding, theta=theta)
    if not reply["ok"]:
        raise RuntimeError(f"score_js failed: {reply['stderr'].strip()}")
    return float(reply["value"])


def score_batch_rows(sources: list[str], binding: str, vector_size: int | None = None,
                     worker: ScoreWorker | None = None) -> list[dict]:
    """`score_flatpdl_batch.cjs`: one result row per source (or vector element)."""
    reply = (worker or shared()).request(
        "batch", "score_flatpdl_batch", sources=sources, binding=binding, vectorSize=vector_size)
    if not reply["ok"]:
        raise RuntimeError(f"score_flatpdl_batch failed: {reply['stderr'].strip()}")
    return reply["rows"]


def sample_sweep_rows(path: Path, n: int, bindings: list[str], base: int = 0,
                      worker: ScoreWorker | None = None) -> list[dict]:
    """`sample_sweep.cjs <path> <n> <bindings> --base <base>`."""
    reply = (worker or shared()).request(
        "sample_sweep", "sample_sweep", model=os.path.abspath(path), n=n,
        bindings=bindings, base=base)
    if not reply["ok"]:
        raise RuntimeError(f"sample_sweep failed: {reply['stderr'].strip()}")
    return reply["rows"]
