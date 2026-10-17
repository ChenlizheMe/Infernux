"""Editor preload contributions share one scope, independently of input focus."""

import pytest

from infernux.engine.interaction.commands import EditorCommand, EditorCommandRegistry
from infernux.engine.interaction.shortcuts import KeyChord, ShortcutBinding, ShortcutRouter
from infernux.engine.ui.panel_registry import PanelRegistry
from infernux.plugins.preload import _remove_editor_contribution_owner


@pytest.fixture
def registries(monkeypatch):
    monkeypatch.setattr(EditorCommandRegistry, "_instance", None)
    monkeypatch.setattr(ShortcutRouter, "_instance", None)
    return EditorCommandRegistry.instance(), ShortcutRouter.instance()


def register_pair(registries, name, *, replace=False):
    commands, shortcuts = registries
    commands.register(EditorCommand(name, lambda ctx: name), replace=replace)
    shortcuts.register(ShortcutBinding(name, KeyChord.parse("F9"), binding_id=name), replace=replace)


def test_nested_scope_exception_restores_owner_and_preserves_core(registries):
    register_pair(registries, "core")
    with PanelRegistry.contribution_scope("outer"):
        register_pair(registries, "outer.a")
        with pytest.raises(RuntimeError):
            with PanelRegistry.contribution_scope("inner"):
                register_pair(registries, "inner")
                raise RuntimeError("author import failed")
        register_pair(registries, "outer.b")
    register_pair(registries, "core.after")
    assert _remove_editor_contribution_owner("inner", runtime=False)
    assert _remove_editor_contribution_owner("outer", runtime=False)
    commands, shortcuts = registries
    assert {item.command_id for item in commands.commands} == {"core", "core.after"}
    assert {item.command_id for item in shortcuts.bindings} == {"core", "core.after"}


@pytest.mark.parametrize("index", [0, 1])
def test_cross_owner_replacement_is_rejected_and_same_owner_can_replace(registries, index):
    with PanelRegistry.contribution_scope("first"):
        register_pair(registries, "tool")
        register_pair(registries, "tool", replace=True)
    registry = registries[index]
    value = (
        EditorCommand("tool", lambda ctx: None)
        if index == 0
        else ShortcutBinding("tool", KeyChord.parse("F10"), binding_id="tool")
    )
    before = registry.revision
    with PanelRegistry.contribution_scope("second"):
        with pytest.raises(ValueError, match="another contributor"):
            registry.register(value, replace=True)
    assert registry.revision == before
    assert registry.unregister_owner("second") == 0
    assert registry.unregister_owner("first") == 1
    assert registry.unregister_owner("first") == 0


def test_refused_panel_close_keeps_commands_and_shortcuts(registries, monkeypatch):
    with PanelRegistry.contribution_scope("author"):
        register_pair(registries, "tool")
    monkeypatch.setattr(PanelRegistry, "remove_owner", lambda owner: False)
    assert not _remove_editor_contribution_owner("author", runtime=False)
    assert registries[0].get("tool") is not None
    assert len(registries[1].bindings) == 1
