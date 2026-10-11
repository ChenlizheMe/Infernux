from __future__ import annotations

from collections.abc import Callable
from os import PathLike
from typing import Any

from .engine import Engine

HeadlessUpdate = Callable[[Engine, int], Any]

def run_headless(
    project_path: str | PathLike[str],
    update: HeadlessUpdate,
    *,
    fixed_delta: float = ...,
    max_frames: int | None = ...,
    engine_log_level: Any = ...,
) -> int: ...
