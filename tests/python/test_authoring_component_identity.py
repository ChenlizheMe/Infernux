"""Component history addresses stable instances across batch edits and reloads."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


def _run(project, operation, order="01", reload_world=False):
    child = subprocess.run(
        [sys.executable, "-X", "faulthandler", str(Path(__file__).resolve()),
         str(project), operation, order, str(int(reload_world))],
        env=dict(os.environ), capture_output=True, text=True, encoding="utf-8",
        errors="replace", timeout=60,
    )
    assert child.returncode == 0, child.stdout + child.stderr
    assert "COMPONENT_IDENTITY_OK" in child.stdout


@pytest.mark.parametrize("order", ["01", "10", "02", "20", "12", "21"])
@pytest.mark.parametrize("reload_world", [False, True])
def test_batch_remove_and_history_preserve_exact_components(tmp_path, order, reload_world):
    _run(tmp_path, "batch", order, reload_world)


@pytest.mark.parametrize("order", ["01", "10", "02", "20", "12", "21"])
@pytest.mark.parametrize("reload_world", [False, True])
def test_native_batch_remove_history_preserves_component_order(tmp_path, order, reload_world):
    _run(tmp_path, "native_batch", order, reload_world)


@pytest.mark.parametrize("operation", ["add", "remove"])
def test_component_history_follows_identity_after_reorder(tmp_path, operation):
    _run(tmp_path, "reorder_" + operation)


@pytest.mark.parametrize("operation", ["add", "remove"])
def test_missing_component_history_does_not_select_same_type_peer(tmp_path, operation):
    _run(tmp_path, "missing_" + operation)


@pytest.mark.parametrize("operation", ["add", "remove"])
def test_unbound_component_cannot_address_first_live_component(tmp_path, operation):
    _run(tmp_path, "unbound_" + operation)


@pytest.mark.parametrize("operation", ["add", "remove"])
def test_component_history_rejects_a_different_owner(tmp_path, operation):
    _run(tmp_path, "wrong_owner_" + operation)


@pytest.mark.parametrize("operation", ["add", "remove"])
def test_snapshot_failure_does_not_remove_component(tmp_path, operation):
    _run(tmp_path, "snapshot_" + operation)


def _exercise(project, operation, order, reload_world):
    from infernux.components import InxComponent, serialized_field
    from infernux.engine.engine import Engine
    from infernux.engine.preferences_store import PreferencesStore
    from infernux.engine.interaction import (
        ComponentCommandService, EditorInteractionCore, SelectionDomain,
        SelectionService, SelectionSnapshot, SelectionTarget,
    )
    from infernux.engine.component_restore import deserialize_scene_document_transactionally
    from infernux.engine.undo import AddPyComponentCommand, RemovePyComponentCommand, UndoManager
    from infernux.lib import LogLevel, RuntimeMode, SceneManager, Vector3

    class PuzzleIdentityStep(InxComponent):
        marker: int = serialized_field(default=0)

        def _serialize_fields(self):
            if getattr(self, "_refuse_snapshot", False):
                raise RuntimeError("snapshot refused")
            return super()._serialize_fields()

    (project / "Assets").mkdir()
    (project / "ProjectSettings").mkdir()
    PreferencesStore()._path = str(project / "preferences.json")
    engine = Engine(LogLevel.Warn, RuntimeMode.Headless)
    try:
        engine.init_headless(str(project))
        scene = SceneManager.instance().create_scene("IdentityHistory")
        owner = scene.create_game_object("PuzzleController")
        components = []
        native_components = operation == "native_batch"
        for marker in range(3):
            if native_components:
                component = owner.add_component("BoxCollider")
                component.center = Vector3(marker + 10, 0, 0)
            else:
                component = owner.add_py_component(PuzzleIdentityStep())
                component.marker = marker + 10
            components.append(component)
            if marker == 0:
                if native_components:
                    owner.add_py_component(PuzzleIdentityStep())
                else:
                    owner.add_component("BoxCollider")
        owner_id = owner.id
        ids = tuple(int(component.component_id) for component in components)
        before_order = tuple(owner.get_component_order())

        def snapshot():
            if native_components:
                return (tuple(owner.get_component_order()),
                        {int(component.component_id): component.center.x
                         for component in owner.get_components("BoxCollider")})
            return (tuple(owner.get_component_order()),
                    {int(component.component_id): component.marker
                     for component in owner.get_py_components()})

        original = snapshot()
        if operation in ("batch", "native_batch"):
            EditorInteractionCore.instance().panels.register_selection_authority(
                "identity_probe", (SelectionDomain.COMPONENT, SelectionDomain.SCENE_OBJECT),
            )
            selected = tuple(int(index) for index in order)
            targets = tuple(SelectionTarget.component(owner.id, ids[index]) for index in selected)
            before = SelectionSnapshot.create(targets, owner_id="identity_probe")
            after = SelectionSnapshot.create(
                (SelectionTarget.scene_object(owner.id),), owner_id="identity_probe")
            SelectionService.instance().apply_snapshot(before, record_history=False)
            assert ComponentCommandService.require().remove_many(
                [(owner, components[index]) for index in selected],
                before_selection=before, after_selection=after,
            )
            removed = set(ids[index] for index in selected)
            expected = (tuple(value for value in before_order if value not in removed),
                        {value: index + 10 for index, value in enumerate(ids) if value not in removed})
            assert snapshot() == expected
            if reload_world:
                assert deserialize_scene_document_transactionally(scene, scene.serialize_document())
                owner = scene.find_by_id(owner_id)
            for _ in range(3):
                UndoManager.instance().undo()
                assert snapshot() == original
                UndoManager.instance().redo()
                assert snapshot() == expected
        else:
            command_type = AddPyComponentCommand if operation.endswith("add") else RemovePyComponentCommand
            if operation.startswith("unbound_"):
                with pytest.raises((ValueError, RuntimeError), match="identity|bound|live"):
                    command_type(owner.id, PuzzleIdentityStep())
                assert snapshot() == original
            elif operation.startswith("wrong_owner_"):
                other = scene.create_game_object("OtherController")
                with pytest.raises(ValueError, match="bound.*GameObject"):
                    command_type(other.id, components[1])
                assert snapshot() == original
                assert not other.get_py_components()
            else:
                command = command_type(owner.id, components[1])
                invoke = command.undo if operation.endswith("add") else command.execute
                if operation.startswith("snapshot_"):
                    components[1]._refuse_snapshot = True
                    with pytest.raises(RuntimeError, match="snapshot refused"):
                        invoke()
                    assert snapshot() == original
                elif operation.startswith("missing_"):
                    assert owner.remove_py_component(components[1])
                    replacement = owner.add_py_component(PuzzleIdentityStep())
                    replacement.marker = 99
                    before_rejection = snapshot()
                    with pytest.raises(RuntimeError, match="not found|missing"):
                        invoke()
                    assert snapshot() == before_rejection
                else:
                    reordered = before_order[2:] + before_order[:2]
                    assert owner.set_component_order(reordered)
                    invoke()
                    assert set(snapshot()[1]) == {ids[0], ids[2]}
                    assert tuple(owner.get_component_order()) == tuple(
                        value for value in reordered if value != ids[1])
                    restore = command.redo if operation.endswith("add") else command.undo
                    restore()
                    assert snapshot() == (reordered, original[1])
        print("COMPONENT_IDENTITY_OK " + json.dumps(dict(operation=operation, order=order)), flush=True)
    finally:
        engine.exit()


if __name__ == "__main__":
    _exercise(Path(sys.argv[1]), sys.argv[2], sys.argv[3], bool(int(sys.argv[4])))
