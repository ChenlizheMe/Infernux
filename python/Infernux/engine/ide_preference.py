"""
IDE preference for the Infernux editor.

Stores the user's preferred external IDE in the shared preferences file.

Supported IDEs:
- "vscode"
- "pycharm"

Default:
- "vscode"
"""

from __future__ import annotations

from Infernux.engine.preferences_store import PreferencesStore

_IDES = {"vscode", "pycharm"}
_current_ide: str = "vscode"
_store = PreferencesStore()


def get_ide() -> str:
    """Return the current preferred IDE."""
    return _current_ide


def set_ide(ide: str) -> None:
    """Set the preferred IDE and persist it."""
    global _current_ide
    if ide not in _IDES:
        return
    _current_ide = ide
    _store.set("preferred_ide", _current_ide)


def _load_preference() -> None:
    """Load the preferred IDE from the shared preferences file."""
    global _current_ide
    ide = _store.get("preferred_ide", "vscode")
    if ide in _IDES:
        _current_ide = ide


_load_preference()
