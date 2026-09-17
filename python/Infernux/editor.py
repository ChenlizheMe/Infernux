"""Editor-only authoring API, available as ``inx.editor``.

Tools use the same command and shortcut services as the built-in Editor.
Register contributions from an editor preload, not from gameplay callbacks.
"""

from .engine.interaction.commands import (
    CommandContext, CommandResult, CommandSource, CommandStatus,
    EditorCommand, EditorCommandRegistry,
)
from .engine.interaction.shortcuts import (
    KeyChord, ShortcutBinding, ShortcutModifier, ShortcutPhase,
    ShortcutRouter, ShortcutScope,
)

__all__ = (
    "CommandContext", "CommandResult", "CommandSource", "CommandStatus",
    "EditorCommand", "EditorCommandRegistry", "KeyChord", "ShortcutBinding",
    "ShortcutModifier", "ShortcutPhase", "ShortcutRouter", "ShortcutScope",
)
