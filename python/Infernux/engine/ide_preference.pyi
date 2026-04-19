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

_IDES: set[str]
_current_ide: str
_store: PreferencesStore

def get_ide() -> str: ...
def set_ide(ide: str) -> None: ...
def _load_preference() -> None: ...
