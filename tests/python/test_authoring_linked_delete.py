"""Structural history restores a linked object set before script lifecycle runs."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("reference", ["object", "component", "native"])
@pytest.mark.parametrize("scenario", ["same_world", "cross_world"])
def test_linked_delete_undo_publishes_complete_graph(tmp_path, nested, reference, scenario):
    _run(tmp_path, nested, reference, scenario)


@pytest.mark.parametrize("scenario", ["missing_parent", "bad_descriptor", "bad_native", "after_failure"])
@pytest.mark.parametrize("multi_world", [False, True])
def test_failed_linked_restore_cleans_only_its_owned_objects(tmp_path, scenario, multi_world):
    _run(tmp_path, True, "component", scenario + ("_cross_world" if multi_world else ""))


def _run(project, nested, reference, scenario):
    child = subprocess.run(
        [sys.executable, "-X", "faulthandler", str(Path(__file__).resolve()),
         str(project), str(int(nested)), reference, scenario],
        env=dict(os.environ), capture_output=True, text=True, encoding="utf-8",
        errors="replace", timeout=60,
    )
    assert child.returncode == 0, child.stdout + child.stderr
    assert "LINKED_DELETE_OK" in child.stdout


def _exercise(project, nested, reference, scenario):
    from infernux.components import FieldType, InxComponent, serialized_field
    from infernux.components.ref_wrappers import ComponentRef, GameObjectRef
    from infernux.engine.component_restore import preflight_scene_python_components
    from infernux.engine.engine import Engine
    from infernux.engine.preferences_store import PreferencesStore
    from infernux.engine.undo import DeleteGameObjectsCommand
    from infernux.lib import LogLevel, RuntimeMode, SceneManager, Vector3

    class LinkedPuzzleNode(InxComponent):
        target = serialized_field(default=None, field_type=FieldType.GAME_OBJECT)
        peer = serialized_field(default=None, field_type=FieldType.COMPONENT)
        marker: int = serialized_field(default=0)
        _events = []
        _fail_after = False

        def _observe(self, phase):
            target = self.target if reference == "object" else self.peer
            identifier = (target.id if reference == "object" else target.component_id) if target else 0
            type(self)._events.append((phase, self.marker, int(identifier)))

        def on_after_deserialize(self):
            self._observe("after")
            if type(self)._fail_after and self.marker == 2:
                raise RuntimeError("intentional linked restore hook failure")

        def awake(self):
            self._observe("awake")

        def on_enable(self):
            self._observe("enable")

    (project / "Assets").mkdir()
    (project / "ProjectSettings").mkdir()
    PreferencesStore()._path = str(project / "preferences.json")
    engine = Engine(LogLevel.Warn, RuntimeMode.Headless)
    try:
        engine.init_headless(str(project))
        scene = SceneManager.instance().create_scene("LinkedPuzzle")
        second_scene = (SceneManager.instance().create_scene("LinkedPuzzleSecond")
                        if "cross_world" in scenario else scene)
        survivor = scene.create_game_object("Unchanged")
        second_survivor = second_scene.create_game_object("OtherUnchanged")
        parent = scene.create_game_object("Room")
        parent.transform.local_position = Vector3(50, 20, 10)
        parent.transform.local_scale = Vector3(2, 3, 4)
        parent.transform.local_euler_angles = Vector3(15, 35, 20)
        second_parent = parent
        if second_scene is not scene:
            second_parent = second_scene.create_game_object("OtherRoom")
            second_parent.transform.local_position = Vector3(-20, 40, 30)
            second_parent.transform.local_scale = Vector3(3, 4, 2)
            second_parent.transform.local_euler_angles = Vector3(25, -45, 10)
        first = scene.create_game_object("Switch")
        second = second_scene.create_game_object("Door")
        if nested:
            first.set_parent(parent, False)
            second.set_parent(second_parent, False)
        first.transform.local_position = Vector3(1, 2, 3)
        second.transform.local_position = Vector3(4, 5, 6)
        first_component = first.add_py_component(LinkedPuzzleNode())
        second_component = second.add_py_component(LinkedPuzzleNode())
        first_component.marker, second_component.marker = 1, 2
        if reference == "object":
            first_component.target = GameObjectRef(second)
            second_component.target = GameObjectRef(first)
            expected = {1: int(second.id), 2: int(first.id)}
        else:
            first_peer, second_peer = first_component, second_component
            if reference == "native":
                first_peer = first.add_component("BoxCollider")
                second_peer = second.add_component("BoxCollider")
            first_component.peer = ComponentRef(second_peer)
            second_component.peer = ComponentRef(first_peer)
            expected = {1: int(second_peer.component_id), 2: int(first_peer.component_id)}
        ids = (int(first.id), int(second.id))
        documents = (first.serialize_document(), second.serialize_document())
        for owner in dict.fromkeys((scene, second_scene)):
            prepared = preflight_scene_python_components(owner.serialize_document(), prefer_loaded_types=True)
            prepared.discard()
        command = DeleteGameObjectsCommand(list(ids))
        owners = (scene, second_scene)
        # Cross-World references currently have a separate authoring-file
        # limitation. Their live Undo contract is tested here independently.
        authored_documents = (tuple(owner.serialize_asset_document() for owner in owners)
                              if scene is second_scene else None)
        author_identities = tuple(owner.serialize_document()["authoring_identity"] for owner in owners)
        if second_scene is not scene:
            try:
                second_scene._remove_game_object_immediately(first)
            except ValueError as exc:
                assert "owned by this Scene" in str(exc)
            else:
                raise AssertionError("another World removed the restoration target")
        fault = scenario.removesuffix("_cross_world")
        if fault in {"missing_parent", "bad_descriptor", "bad_native", "after_failure"}:
            import copy

            command.execute()
            original_entries = copy.deepcopy(command._entries)
            if fault == "missing_parent":
                command._entries[1]["parent_id"] = 999999999
            elif fault == "bad_descriptor":
                record = next(record for record in command._entries[1]["document"]["components"]
                              if record["type_id"].startswith("python:"))
                record["component_id"] = 0
            elif fault == "bad_native":
                command._entries[1]["document"]["prefab_source_id"] = -1
            else:
                LinkedPuzzleNode._fail_after = True
            pending = scene.create_game_object("PreviouslyQueuedForDestruction")
            scene.destroy_game_object(pending)
            # Allocation counters and identity tombstones remain monotonic;
            # only authored files and live unrelated objects must be identical.
            before = tuple(owner.serialize_asset_document() for owner in owners)
            LinkedPuzzleNode._events.clear()
            try:
                command.undo()
            except RuntimeError:
                pass
            else:
                raise AssertionError(f"{fault} did not reject the restore")
            assert all(owner.find_by_id(value) is None for owner, value in zip(owners, ids))
            after = tuple(owner.serialize_asset_document() for owner in owners)
            if after != before:
                import difflib
                raise AssertionError("\n".join(difflib.unified_diff(
                    json.dumps(before, indent=2).splitlines(),
                    json.dumps(after, indent=2).splitlines(),
                    fromfile="before", tofile="after",
                )))
            assert scene.find("Unchanged") is survivor
            assert second_scene.find("OtherUnchanged") is second_survivor
            assert all(not owner.has_pending_py_components() for owner in owners)
            assert not any(phase == "awake" for phase, _marker, _target in LinkedPuzzleNode._events)
            assert pending, "rollback consumed an unrelated destruction queue"
            scene.process_pending_destroys()
            assert not pending
            command._entries = original_entries
            LinkedPuzzleNode._fail_after = False
            command.undo()
        for _ in range(3):
            command.execute()
            assert all(owner.find_by_id(value) is None for owner, value in zip(owners, ids))
            LinkedPuzzleNode._events.clear()
            command.undo()
            restored = tuple(owner.find_by_id(value) for owner, value in zip(owners, ids))
            assert all(restored)
            assert tuple(obj.serialize_document() for obj in restored) == documents
            if authored_documents is not None:
                assert tuple(owner.serialize_asset_document() for owner in owners) == authored_documents
            for owner, original in zip(owners, author_identities):
                current = owner.serialize_document()["authoring_identity"]
                for table in ("objects", "components"):
                    assert all(current[table][key] == value for key, value in original[table].items())
            assert scene.find("Unchanged") is survivor
            assert scene.find("Room") is parent
            assert second_scene.find("OtherUnchanged") is second_survivor
            events = LinkedPuzzleNode._events
            assert len(events) == 6, events
            assert all(target == expected[marker] for _phase, marker, target in events), events
            assert [phase for phase, _marker, _target in events] == [
                "after", "after", "awake", "enable", "awake", "enable",
            ]
        print("LINKED_DELETE_OK " + json.dumps(dict(nested=nested, reference=reference, scenario=scenario)), flush=True)
    finally:
        engine.exit()


if __name__ == "__main__":
    _exercise(Path(sys.argv[1]), bool(int(sys.argv[2])), sys.argv[3], sys.argv[4])
