"""Owner-thread callbacks that must run outside native GUI drawing."""

from __future__ import annotations

from collections import deque
from contextlib import nullcontext
from typing import Any, Callable


_pending: deque[tuple[Callable[[], Any], Any]] = deque()


def enqueue(callback: Callable[[], Any]) -> None:
    if not callable(callback):
        raise TypeError("runtime callback must be callable")
    from infernux.engine.reload.owners import owner_for
    _pending.append((callback, owner_for(callback)))


def drain() -> int:
    """Run callbacks queued during the previous native frame."""
    count = 0
    while _pending:
        callback, owner = _pending.popleft()
        if owner is not None and not owner.active:
            continue
        try:
            with owner.enter() if owner is not None else nullcontext():
                callback()
        except Exception as exc:
            from infernux.debug import Debug
            Debug.log_exception(exc)
        count += 1
    return count


def clear() -> None:
    _pending.clear()
