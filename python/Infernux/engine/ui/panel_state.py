"""
Panel state persistence — saves / loads per-panel settings to a JSON file
inside the project's layout directory (Documents/Infernux/{project}/).
"""
import json
import os
import threading
from Infernux.debug import Debug

_state_path: str = ""
_state: dict = {}
_lock = threading.Lock()


def init(layout_dir: str) -> None:
    """Set the directory for panel_state.json and load existing state."""
    global _state_path, _state
    _state_path = os.path.join(layout_dir, "panel_state.json")
    if os.path.isfile(_state_path):
        try:
            with open(_state_path, "r", encoding="utf-8") as f:
                _state = json.load(f)
        except (OSError, json.JSONDecodeError):
            _state = {}
    else:
        _state = {}


def get(panel_id: str) -> dict:
    """Return the saved state dict for a panel, or empty dict."""
    with _lock:
        return dict(_state.get(panel_id, {}))


def put(panel_id: str, data: dict) -> None:
    """Update the state for a panel (merged)."""
    with _lock:
        _state[panel_id] = data


def delete(panel_id: str) -> None:
    """Remove a panel state entry if it exists."""
    with _lock:
        _state.pop(panel_id, None)


def save() -> None:
    """Write the current state to disk."""
    if not _state_path:
        return
    with _lock:
        snapshot = dict(_state)
    try:
        os.makedirs(os.path.dirname(_state_path), exist_ok=True)
        with open(_state_path, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, indent=2, ensure_ascii=False)
    except OSError as _exc:
        Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
        pass
