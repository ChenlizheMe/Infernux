"""Editor preload contributions share one scope, independently of input focus."""

import pytest

from Infernux.engine.interaction.commands import EditorCommand, EditorCommandRegistry
from Infernux.engine.interaction.shortcuts import KeyChord, ShortcutBinding, ShortcutRouter
from Infernux.engine.ui.panel_registry import PanelRegistry
from Infernux.plugins.preload import _remove_editor_contribution_owner


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
    value = (EditorCommand("tool", lambda ctx: None) if index == 0 else
             ShortcutBinding("tool", KeyChord.parse("F10"), binding_id="tool"))
    before = registry.revision
    with PanelRegistry.contribution_scope("second"):
        with pytest.raises(ValueError, match="another contributor"):
            registry.register(value, replace=True)
    assert registry.revision == before
    assert registry.unregister_owner("second") == 0
    assert registry.unregister_owner("first") == 1
    assert registry.unregister_owner("first") == 0


@pytest.mark.parametrize("index", [0, 1])
def test_manual_unregister_and_clear_retire_owner_metadata(registries, index):
    with PanelRegistry.contribution_scope("author"):
        register_pair(registries, "a")
        register_pair(registries, "b")
    registry = registries[index]
    assert registry.unregister("a")
    assert not registry.unregister("a")
    registry.clear()
    assert registry.unregister_owner("author") == 0
    with pytest.raises(ValueError, match="must not be empty"):
        registry.unregister_owner("")


def test_refused_panel_close_keeps_commands_and_shortcuts(registries, monkeypatch):
    with PanelRegistry.contribution_scope("author"):
        register_pair(registries, "tool")
    monkeypatch.setattr(PanelRegistry, "remove_owner", lambda owner: False)
    assert not _remove_editor_contribution_owner("author", runtime=False)
    assert registries[0].get("tool") is not None
    assert len(registries[1].bindings) == 1


def test_shortcut_profiles_preserve_preload_bindings(registries):
    from Infernux.engine.interaction.preferences import PreferencesCommandService

    _, router = registries
    preferences = PreferencesCommandService()
    with PanelRegistry.contribution_scope("author"):
        register_pair(registries, "tool")
    preferences.bind_shortcuts(
        (ShortcutBinding("core.save", KeyChord.parse("Ctrl+S"), binding_id="core.save"),),
        router, load=lambda: None, save=lambda data: None,
    )
    assert {binding.binding_id for binding in router.bindings} == {"core.save", "tool"}
    preferences.create_shortcut_profile("Custom", profile_id="custom")
    preferences.activate_shortcut_profile("custom")
    preferences.assign_shortcut("core.save", None)
    assert {binding.binding_id for binding in router.bindings} == {"tool"}
    preferences.reset_shortcut_profile()
    assert {binding.binding_id for binding in router.bindings} == {"core.save", "tool"}
    assert router.unregister_owner("author") == 1
    assert {binding.binding_id for binding in router.bindings} == {"core.save"}
    preferences.shutdown()
