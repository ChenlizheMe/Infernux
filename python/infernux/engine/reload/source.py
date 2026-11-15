"""Snapshot and compile work. Only the coordinator decides runtime availability."""

from dataclasses import dataclass
from pathlib import Path
from threading import Condition
from types import CodeType


@dataclass(frozen=True, slots=True)
class SourceResult:
    path: str
    generation: int
    source: bytes | None
    code: CodeType | None = None
    error: Exception | None = None


class SourceFrontend:
    def __init__(self):
        self._lock = Condition()
        self._generation = 0
        self._pending = {}
        self._ready = {}
        self._latest = {}
        self._running = {}

    def submit(self, path, source=None):
        with self._lock:
            self._generation += 1
            request = (
                SourceResult(path, self._generation, None, error=source)
                if isinstance(source, Exception)
                else SourceResult(path, self._generation, source)
            )
            self._latest[path] = request.generation
            self._pending[path] = request
            self._ready.pop(path, None)
        return request

    def compile_pending(self):
        with self._lock:
            pending, self._pending = self._pending, {}
            self._running.update({path: request.generation for path, request in pending.items()})
        for path, request in pending.items():
            source, code, error = request.source, None, request.error
            try:
                if error is None:
                    if source is None:
                        source = Path(path).read_bytes()
                    code = compile(source, path, "exec", dont_inherit=True)
            except Exception as exc:
                error = exc
            result = SourceResult(path, request.generation, source, code, error)
            with self._lock:
                if self._latest.get(path) == result.generation:
                    self._ready[path] = result
                if self._running.get(path) == result.generation:
                    self._running.pop(path)
                self._lock.notify_all()
        return len(pending)

    def take_ready(self):
        with self._lock:
            ready, self._ready = self._ready, {}
        return ready

    def cancel(self, path):
        with self._lock:
            self._latest.pop(path, None)
            self._pending.pop(path, None)
            self._ready.pop(path, None)
            self._lock.notify_all()

    def wait_for(self, requests):
        """An explicit Play/load barrier waits for work already being compiled."""
        with self._lock:
            self._lock.wait_for(
                lambda: all(
                    self._latest.get(request.path) != request.generation
                    or (result := self._ready.get(request.path)) is not None
                    and result.generation == request.generation
                    for request in requests
                )
            )

    @property
    def pending_count(self):
        with self._lock:
            return len(self._pending)

    def clear(self):
        with self._lock:
            self._pending.clear()
            self._ready.clear()
            self._latest.clear()
            self._lock.notify_all()
