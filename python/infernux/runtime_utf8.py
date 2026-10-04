"""UTF-8 defaults for Infernux Python processes (especially Windows).

Sets PEP 540-style UTF-8 mode for child processes and best-effort UTF-8
stdio so Unicode paths and logs behave consistently with the C++ engine
(ToFsPath / UTF-8 strings).
"""

from __future__ import annotations

import os
import sys


def configure_process_utf8() -> None:
    """Apply UTF-8 environment defaults and reconfigure stdio when possible."""
    if sys.platform == "win32":
        os.environ.setdefault("PYTHONUTF8", "1")
        os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def merge_child_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    """Build an environment dict for subprocesses (inherits UTF-8 mode on Windows)."""
    merged = {**os.environ, **(extra or {})}
    if sys.platform == "win32":
        merged.setdefault("PYTHONUTF8", "1")
        merged.setdefault("PYTHONIOENCODING", "utf-8")
    return merged


def apply_utf8_defaults(env: dict[str, str]) -> dict[str, str]:
    """Apply UTF-8 defaults to an existing environment mapping (mutates a copy)."""
    out = dict(env)
    if sys.platform == "win32":
        out.setdefault("PYTHONUTF8", "1")
        out.setdefault("PYTHONIOENCODING", "utf-8")
    return out
