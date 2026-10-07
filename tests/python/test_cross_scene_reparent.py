"""Native scene ownership, deferred lifecycle and camera migration contracts."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


def _run(project, *scenario):
    import infernux

    result = subprocess.run(
        [sys.executable, "-X", "utf8", "-B", str(Path(__file__).resolve()), str(project), *scenario],
        env={**os.environ, "PYTHONPATH": str(Path(infernux.__file__).resolve().parent.parent)},
        capture_output=True, text=True, encoding="utf-8", timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads((project / "migration-evidence.json").read_text(encoding="utf-8"))["status"] == "passed"


@pytest.mark.parametrize("layout", ["root", "child"])
@pytest.mark.parametrize("position", ["world", "local"])
@pytest.mark.parametrize("camera", ["vacant", "existing"])
def test_cross_scene_reparent_moves_complete_live_state(tmp_path, layout, position, camera):
    _run(tmp_path, "live", layout, position, camera)


@pytest.mark.parametrize("layout", ["root", "child"])
@pytest.mark.parametrize("pending", ["root", "descendant", "destination"])
def test_cross_scene_reparent_rejects_pending_destruction_without_mutation(tmp_path, layout, pending):
    _run(tmp_path, "pending", layout, pending)


@pytest.mark.parametrize("camera", ["vacant", "existing"])
def test_root_scene_move_keeps_cached_builtin_wrappers(tmp_path, camera):
    _run(tmp_path, "root_move", "root", "world", camera)


@pytest.mark.parametrize("layout", ["root", "child"])
def test_cross_scene_reparent_defers_start_under_inactive_parent(tmp_path, layout):
    _run(tmp_path, "live_inactive", layout, "world", "vacant")


@pytest.mark.parametrize("operation", ["conflict_reparent", "conflict_move"])
def test_cross_scene_identity_conflict_keeps_camera_and_lifecycle_ownership(tmp_path, operation):
    _run(tmp_path, operation, "root")


def exercise(project, scenario, layout, *options):
    from infernux.components import InxComponent
    from infernux.engine.engine import Engine
    from infernux.engine.preferences_store import PreferencesStore
    from infernux.lib import LogLevel, RuntimeMode, SceneManager, Vector3

    (project / "Assets").mkdir(parents=True)
    (project / "ProjectSettings").mkdir()
    PreferencesStore()._path = str(project / "preferences.json")
    engine = Engine(LogLevel.Warn, RuntimeMode.Headless)
    events = []
    failures = []

    def require(condition, label):
        if not condition:
            failures.append(label)

    def xyz(vector):
        return (vector.x, vector.y, vector.z)

    class MigrationProbe(InxComponent):
        def awake(self):
            events.append(["awake", self.game_object.name])

        def start(self):
            events.append(["start", self.game_object.name])

        def on_enable(self):
            events.append(["enable", self.game_object.name])

        def on_disable(self):
            events.append(["disable", self.game_object.name])

        def on_destroy(self):
            events.append(["destroy", self.game_object.name])

    try:
        engine.init_headless(str(project))
        manager = SceneManager.instance()
        source = manager.create_scene("Source")
        destination = manager.create_scene("Destination")
        manager.set_active_scene(source)
        manager.play()
        manager.pause()
        source_parent = source.create_game_object("OldParent")
        source_parent.transform.local_position = Vector3(3, 0, 0)
        new_parent = destination.create_game_object("NewParent")
        new_parent.transform.local_position = Vector3(10, 2, 0)
        root = source.create_game_object("MovingRoot")
        if layout == "child":
            root.set_parent(source_parent, False)
        root.transform.local_position = Vector3(1, 2, 3)
        child = source.create_game_object("MovingChild")
        child.set_parent(root, False)
        grandchild = source.create_game_object("MovingGrandchild")
        grandchild.set_parent(child, False)
        moving = (root, child, grandchild)
        light = child.add_component("Light")
        camera = grandchild.add_component("Camera")
        native_components = (light._cpp_component, camera._cpp_component)
        source.main_camera = camera
        components = [item.add_component(MigrationProbe) for item in moving]
        old_handles = [item.handle for item in moving]
        old_component_handles = [item._cpp_component.handle for item in components]
        old_native_handles = [light.handle, camera.handle]
        original_identity = source.serialize_document()["authoring_identity"]
        if scenario == "pending" or scenario.startswith("conflict_"):
            if scenario == "pending":
                pending = {"root": root, "descendant": grandchild, "destination": new_parent}[options[0]]
                pending.scene.destroy_game_object(pending)
            else:
                conflicted = destination.serialize_document()
                conflicted["authoring_identity"]["objects"][str(new_parent.id)] = original_identity["objects"][str(root.id)]
                assert destination._commit_document(conflicted)
                new_parent = destination.find("NewParent")
            before_source = source.serialize_document()
            before_destination = destination.serialize_document()
            before_events = list(events)
            rejected = False
            try:
                if scenario == "conflict_move":
                    manager.move_game_object_to_scene(root, destination)
                else:
                    root.set_parent(new_parent)
            except (RuntimeError, ValueError):
                rejected = True
            require(rejected, "invalid migration was accepted")
            require(source.serialize_document() == before_source, "source changed after rejection")
            require(destination.serialize_document() == before_destination, "destination changed after rejection")
            require(events == before_events, "rejection ran lifecycle callbacks")
        else:
            position, camera_policy = options
            existing_camera = new_parent.add_component("Camera") if camera_policy == "existing" else None
            destination.main_camera = existing_camera
            expected_camera_id = (existing_camera or camera).component_id
            old_world_position = xyz(root.transform.position)
            old_local_position = xyz(root.transform.local_position)
            assert not any(event[0] == "start" for event in events), events
            if scenario == "live_inactive":
                new_parent.active = False
            if scenario == "root_move":
                manager.move_game_object_to_scene(root, destination)
            else:
                root.set_parent(new_parent, position == "world")
            for obj, old_handle, component, old_component_handle in zip(
                moving, old_handles, components, old_component_handles,
            ):
                # These must hold before unloading the source, independently
                # of the camera/deferred-start assertions below.
                assert obj.scene is destination
                assert source.find_by_id(obj.id) is None
                assert destination.find_by_id(obj.id) is obj
                assert source.resolve_game_object(old_handle) is None
                assert destination.resolve_game_object(old_handle) is None
                assert destination.resolve_game_object(obj.handle) is obj
                assert source.resolve_component(old_component_handle) is None
                assert destination.resolve_component(component._cpp_component.handle) is not None
                assert component.game_object is obj
            for native, old_handle in zip(native_components, old_native_handles):
                assert source.resolve_component(old_handle) is None
                assert destination.resolve_component(native.handle).handle == native.handle
            after_identity = destination.serialize_document()["authoring_identity"]
            for kind in ("objects", "components"):
                # Include only moved members; the source's parent is retained.
                moved_ids = ([obj.id for obj in moving] if kind == "objects" else
                             [obj.transform.component_id for obj in moving] +
                             [comp.component_id for comp in components] + [handle.id for handle in old_native_handles])
                for identity in moved_ids:
                    assert after_identity[kind][str(identity)] == original_identity[kind][str(identity)]
            require(light.is_valid and camera.is_valid, "retained builtin wrapper lost its live component")
            require(source.main_camera is None, "source retained foreign main camera")
            selected = destination.main_camera
            require(selected is not None and selected.component_id == expected_camera_id,
                    "destination main camera policy was lost")
            require(xyz(root.transform.position if position == "world" else root.transform.local_position) ==
                    (old_world_position if position == "world" else old_local_position), "transform changed")
            # Destroying the old Scene before its pending Start queue runs must
            # not lose the moved components' deferred lifecycle work.
            manager.set_active_scene(destination)
            manager.unload_scene(source)
            if scenario == "live_inactive":
                manager.step()
                require(not any(event[0] == "start" for event in events), "inactive subtree started")
                new_parent.active = True
            manager.step()
            manager.step()
            for obj in moving:
                require(events.count(["awake", obj.name]) == 1, f"Awake replayed for {obj.name}")
                require(events.count(["start", obj.name]) == 1, f"Start lost/replayed for {obj.name}")
                require(["destroy", obj.name] not in events, f"source unload destroyed {obj.name}")
                require(events.count(["enable", obj.name]) == (2 if scenario == "live_inactive" else 1),
                        f"OnEnable lost/replayed for {obj.name}")
                require(events.count(["disable", obj.name]) == (1 if scenario == "live_inactive" else 0),
                        f"OnDisable lost/replayed for {obj.name}")
        (project / "migration-evidence.json").write_text(json.dumps(dict(
            status="failed" if failures else "passed", scenario=scenario, layout=layout,
            options=options, events=events, failures=failures,
        ), indent=2), encoding="utf-8")
        assert not failures, failures
    finally:
        engine.exit()


if __name__ == "__main__":
    exercise(Path(sys.argv[1]), *sys.argv[2:])
