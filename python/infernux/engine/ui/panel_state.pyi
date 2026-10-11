"""Panel state persistence — per-panel settings stored as JSON.

Example::

    from infernux.engine.ui import panel_state
    panel_state.init("/path/to/layout/dir")
    panel_state.put("console", {"filter": "errors"})
    data = panel_state.get("console")
    panel_state.save()
"""

from __future__ import annotations


def init(layout_dir: str) -> None:
    """Set the directory for ``panel_state.json`` and load existing state.

    Args:
        layout_dir: Absolute path to the layout directory.
    """
    ...

def get(panel_id: str) -> dict:
    """Return the saved state dict for *panel_id*, or empty dict."""
    ...

def put(panel_id: str, data: dict) -> None:
    """Update the state for *panel_id*.

    Args:
        panel_id: The panel's unique identifier.
        data: Arbitrary dict to persist.
    """
    ...

def save() -> None:
    """Write the current state to disk."""
    ...
