from types import SimpleNamespace

import pytest

from Infernux.engine.model_import import toolchain


def test_blender_preference_updates_live_database_and_does_not_touch_project(monkeypatch, tmp_path):
    from Infernux.core.assets import AssetManager
    state, calls = {}, []
    store = SimpleNamespace(get=lambda key, default="": state.get(key, default),
                            set=lambda key, value: state.__setitem__(key, value))
    monkeypatch.setattr(toolchain, "PreferencesStore", lambda: store)
    database = SimpleNamespace(configure_blender_import=lambda *args: calls.append(args))
    monkeypatch.setattr(AssetManager, "require_asset_database", lambda: database)
    executable = tmp_path / "工具 路径" / "blender.exe"
    executable.parent.mkdir()
    executable.touch()
    toolchain.set_blender_executable(str(executable))
    assert toolchain.get_blender_executable() == str(executable.resolve())
    assert calls[-1] == (str(executable.resolve()), toolchain.export_script())
    toolchain.configure_database(database)
    assert len(calls) == 2
    toolchain.set_blender_executable("")
    assert calls[-1] == ("", "") and toolchain.get_blender_executable() == ""
    toolchain.configure_database(database)
    assert len(calls) == 3
    with pytest.raises(ValueError, match="existing"):
        toolchain.set_blender_executable(str(tmp_path / "missing"))
    assert len(calls) == 3 and state["blender_executable"] == ""


def test_blender_preference_is_a_shared_undo_command(monkeypatch):
    from Infernux.engine.interaction import CommandSource, EditorInteractionCore
    from Infernux.engine.undo import UndoManager
    state = {"path": "old"}
    monkeypatch.setattr(toolchain, "get_blender_executable", lambda: state["path"])
    monkeypatch.setattr(toolchain, "set_blender_executable", lambda value: state.__setitem__("path", value))
    previous = UndoManager.instance()
    core = EditorInteractionCore()
    manager = UndoManager(core.action_journal)
    try:
        result = core.commands.execute("preferences.set_blender_executable", source=CommandSource.API,
                                       payload={"value": "new"})
        assert result.accepted and state["path"] == "new"
        manager.undo()
        assert state["path"] == "old"
        manager.redo()
        assert state["path"] == "new"
        assert not manager.action_journal.applied_entries()[-1].action.marks_dirty
    finally:
        core.shutdown()
        UndoManager._instance = previous
