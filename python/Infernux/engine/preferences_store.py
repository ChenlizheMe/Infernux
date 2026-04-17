"""
Shared preferences storage for the Infernux editor.

This module provides a minimal JSON-backed preference store used by
different preference classes. It preserves the original persistence logic:

- preferences file: Documents/Infernux/preferences.json
- load the whole JSON object
- update only owned fields
- keep unrelated fields intact
"""

from __future__ import annotations

import json
import os
import pathlib
from Infernux.debug import Debug

_PREFS_FILE = "preferences.json"


def _prefs_path() -> str:
    """Return the path to the global preferences file."""
    if os.name == "nt":
        docs = pathlib.Path.home() / "Documents"
        try:
            import ctypes.wintypes
            buf = ctypes.create_unicode_buffer(ctypes.wintypes.MAX_PATH)
            ctypes.windll.shell32.SHGetFolderPathW(None, 5, None, 0, buf)
            if buf.value:
                docs = pathlib.Path(buf.value)
        except (OSError, ValueError) as _exc:
            Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
            pass
    else:
        docs = pathlib.Path.home() / "Documents"

    prefs_dir = docs / "Infernux"
    os.makedirs(prefs_dir, exist_ok=True)
    return str(prefs_dir / _PREFS_FILE)


class PreferencesStore:
    """Minimal JSON-backed preferences storage."""

    def __init__(self) -> None:
        self._path = _prefs_path()

    def load(self) -> dict:
        """Load and return the full preferences dictionary."""
        if not os.path.isfile(self._path):
            return {}

        try:
            with open(self._path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, OSError) as _exc:
            Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
            return {}

    def save(self, data: dict) -> None:
        """Save the full preferences dictionary."""
        try:
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except OSError as _exc:
            Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
            pass

    def get(self, key: str, default=None):
        """Return a single preference value."""
        data = self.load()
        return data.get(key, default)

    def set(self, key: str, value) -> None:
        """Update one preference key without overwriting unrelated keys."""
        data = self.load()
        data[key] = value
        self.save(data)
