"""project_utils — file-system helpers for the Project panel.

Usage::

    from Infernux.engine.ui.project_utils import (
        open_in_vscode, get_file_type, reveal_in_file_explorer,
    )
"""

from __future__ import annotations

from typing import Set


# ── Filter sets ─────────────────────────────────────────────────────

HIDDEN_EXTENSIONS: Set[str]
"""Extensions hidden in the Project panel (e.g. ``".meta"``, ``".pyc"``)."""
HIDDEN_PREFIXES: Set[str]
HIDDEN_FILES: Set[str]


# ── Public API ──────────────────────────────────────────────────────

def should_show(name: str) -> bool:
    """Return ``True`` if *name* should be visible in the Project panel."""
    ...

def open_in_vscode(
    file_path: str, line: int = 0, project_root: str = "",
) -> bool:
    """Open *file_path* in VS Code. Returns ``True`` on success."""
    ...

def open_in_pycharm(
    file_path: str, line: int = 0, project_root: str = "",
) -> bool:
    """Open *file_path* in Pycharm. Returns ``True`` on success."""
    ...

def open_file_with_system(
    file_path: str, project_root: str = "",
) -> None:
    """Open *file_path* with the OS default application."""
    ...

def get_file_type(filename: str) -> str:
    """Return a category string (``"script"``, ``"texture"``, …) for *filename*."""
    ...

def update_material_name_in_file(mat_path: str, new_name: str) -> None:
    """Patch the ``name`` field inside a ``.mat`` JSON file."""
    ...

def reveal_in_file_explorer(path: str) -> None:
    """Open the OS file explorer and highlight *path*."""
    ...
