"""Preload ownership shared by editor registrations, not input focus scope."""

from contextlib import contextmanager
from contextvars import ContextVar
from collections.abc import Iterator


current_owner: ContextVar[str] = ContextVar("infernux_editor_contribution_owner", default="")


@contextmanager
def contribution_scope(owner: str) -> Iterator[None]:
    token = current_owner.set(str(owner or ""))
    try:
        yield
    finally:
        current_owner.reset(token)
