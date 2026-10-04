"""Thread-safe status feed shared by editor background operations.

Status producers use a stable ``source`` name. Concurrent producers no longer
erase each other: the highest-priority active source is shown in the native
status bar, and a lower-priority status becomes visible again when it finishes.
"""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time


@dataclass(frozen=True, slots=True)
class _StatusEntry:
    text: str
    progress: float
    kind: str
    expire_at: float
    priority: int
    sequence: int


class EngineStatus:
    """Global, source-aware engine activity indicator."""

    _lock = threading.RLock()
    _entries: dict[str, _StatusEntry] = {}
    _sequence: int = 0

    @classmethod
    def _store(
        cls,
        text: str,
        progress: float,
        kind: str,
        expire_at: float,
        source: str,
        priority: int,
    ) -> None:
        source = source or "default"
        with cls._lock:
            cls._sequence += 1
            cls._entries[source] = _StatusEntry(
                text=str(text),
                progress=float(progress),
                kind=str(kind),
                expire_at=float(expire_at),
                priority=int(priority),
                sequence=cls._sequence,
            )

    @classmethod
    def set(
        cls,
        text: str,
        progress: float = -1.0,
        kind: str | None = None,
        *,
        source: str = "default",
        priority: int = 0,
    ) -> None:
        """Set a persistent status for one producer."""
        resolved_kind = kind or ("progress" if progress >= 0.0 else "activity")
        cls._store(text, progress, resolved_kind, 0.0, source, priority)

    @classmethod
    def flash(
        cls,
        text: str,
        progress: float = -1.0,
        duration: float = 1.5,
        kind: str | None = None,
        *,
        source: str = "default",
        priority: int = 0,
    ) -> None:
        """Set a status for one producer that expires after ``duration``."""
        if kind is not None:
            resolved_kind = kind
        elif progress >= 1.0:
            resolved_kind = "success"
        elif progress == 0.0:
            resolved_kind = "error"
        else:
            resolved_kind = "activity"
        cls._store(
            text,
            progress,
            resolved_kind,
            time.monotonic() + max(0.0, float(duration)),
            source,
            priority,
        )

    @classmethod
    def clear(cls, *, source: str | None = None) -> None:
        """Clear one producer, or every producer when ``source`` is omitted."""
        with cls._lock:
            if source is None:
                cls._entries.clear()
            else:
                cls._entries.pop(source or "default", None)

    @classmethod
    def _current_locked(cls) -> _StatusEntry | None:
        now = time.monotonic()
        expired = [
            source
            for source, entry in cls._entries.items()
            if entry.expire_at > 0.0 and now >= entry.expire_at
        ]
        for source in expired:
            cls._entries.pop(source, None)
        if not cls._entries:
            return None
        return max(cls._entries.values(), key=lambda entry: (entry.priority, entry.sequence))

    @classmethod
    def get(cls) -> tuple[str, float, str]:
        """Return the highest-priority active ``(text, progress, kind)``."""
        with cls._lock:
            current = cls._current_locked()
            if current is None:
                return "", -1.0, "idle"
            return current.text, current.progress, current.kind

    @classmethod
    def is_active(cls) -> bool:
        with cls._lock:
            return cls._current_locked() is not None
