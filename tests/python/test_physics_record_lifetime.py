"""Retained physics values must not resurrect retired native targets."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


QUERIES = ("raycast", "raycast_all", "collider", "sphere", "box", "capsule")
ACTIONS = ("destroy", "unload", "replace", "remove_collider", "reuse_component_id", "migrate", "body_rebuild")


def _run(project, query, action):
    import infernux

    result = subprocess.run(
        [sys.executable, "-X", "utf8", "-B", str(Path(__file__).resolve()), str(project), query, action],
        env={**os.environ, "PYTHONPATH": str(Path(infernux.__file__).resolve().parent.parent)},
        capture_output=True, text=True, encoding="utf-8", timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads((project / "record-evidence.json").read_text(encoding="utf-8"))["status"] == "passed"


@pytest.mark.parametrize("query", QUERIES)
@pytest.mark.parametrize("action", ACTIONS)
def test_retained_query_targets_follow_native_lifetimes(tmp_path, query, action):
    _run(tmp_path, query, action)


@pytest.mark.parametrize("action", ACTIONS)
def test_retained_collision_targets_follow_native_lifetimes(tmp_path, action):
    _run(tmp_path, "collision", action)


def exercise(project, query, action):
    from infernux.components import InxComponent
    from infernux.engine.engine import Engine
    from infernux.engine.preferences_store import PreferencesStore
    from infernux.lib import LogLevel, Physics, RuntimeMode, SceneManager, Vector3

    if sys.platform == "win32":
        import ctypes
        ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002)

    (project / "Assets").mkdir(parents=True)
    (project / "ProjectSettings").mkdir()
    PreferencesStore()._path = str(project / "preferences.json")
    engine = Engine(LogLevel.Warn, RuntimeMode.Headless)
    captured = []

    class RetainCollision(InxComponent):
        def on_collision_enter(self, value):
            captured.append(value)

    def vector(value):
        return value.x, value.y, value.z

    def snapshot(record):
        if query == "collision":
            return vector(record.contact_point), vector(record.contact_normal), vector(record.relative_velocity)
        return vector(record.point), vector(record.normal), record.distance, record.body_id, record.sub_shape_id

    try:
        engine.init_headless(str(project))
        manager = SceneManager.instance()
        source = manager.create_scene("RecordSource")
        destination = manager.create_scene("RecordDestination")
        owner = source.create_game_object("RecordTarget")
        collider = owner.add_component("BoxCollider")
        object_handle, collider_handle = owner.handle, collider.handle
        if query == "collision":
            mover = source.create_game_object("Mover")
            mover.transform.position = Vector3(0, 0.75, 0)
            mover.add_component("BoxCollider")
            body = mover.add_component("Rigidbody")
            body.use_gravity = False
            mover.add_component(RetainCollision)
            manager.play()
            manager.pause()
            manager.step()
            assert captured, "real Jolt collision was not delivered"
            record = captured[0]
        else:
            origin, direction = Vector3(0, 3, 0), Vector3(0, -1, 0)
            if query == "raycast":
                record = Physics.raycast(origin, direction, 10)
            elif query == "raycast_all":
                records = Physics.raycast_all(origin, direction, 10)
                assert len(records) == 1
                record = records[0]
            elif query == "collider":
                record = collider.raycast(origin, direction, 10)
            elif query == "sphere":
                record = Physics.sphere_cast(origin, 0.1, direction, 10)
            elif query == "box":
                record = Physics.box_cast(origin, Vector3(0.1, 0.1, 0.1), direction, max_distance=10)
            else:
                record = Physics.capsule_cast(origin, Vector3(0, 3.2, 0), 0.1, direction, 10)
        assert record is not None
        assert record.game_object.handle == object_handle
        assert record.collider.handle == collider_handle
        before = snapshot(record)
        source_world = source.world_id
        print(json.dumps(dict(query=query, action=action, stage="captured-live-record")), flush=True)

        if action == "destroy":
            source.destroy_game_object(owner)
            source.process_pending_destroys()
            assert source.resolve_game_object(object_handle) is None
        elif action == "unload":
            manager.unload_scene(source)
            assert all(manager.get_scene_at(i).world_id != source_world for i in range(manager.scene_count))
        elif action == "replace":
            assert source._commit_document(source.serialize_document())
            assert source.resolve_game_object(object_handle) is None
            replacement = source.find_by_id(object_handle.id)
            assert replacement is not None and replacement.handle.generation != object_handle.generation
        elif action in ("remove_collider", "reuse_component_id"):
            assert owner.remove_component(collider)
            assert source.resolve_component(collider_handle) is None
            if action == "reuse_component_id":
                replacement = owner.add_component("BoxCollider")
                replacement_native = replacement._cpp_component
                replacement_native._set_component_id(collider_handle.id)
                assert replacement_native.component_id == collider_handle.id
                assert replacement_native.handle.generation != collider_handle.generation
        elif action == "migrate":
            parent = destination.create_game_object("NewParent")
            owner.set_parent(parent)
            assert owner.handle.world_id != source_world
        else:
            collider.enabled = False
            collider.enabled = True
        print(json.dumps(dict(query=query, action=action, stage="after-mutation")), flush=True)
        assert snapshot(record) == before, "retained numerical snapshot changed"
        live_owner = action in ("remove_collider", "reuse_component_id", "migrate", "body_rebuild")
        live_collider = action in ("migrate", "body_rebuild")
        if live_owner:
            assert record.game_object.handle == owner.handle
        else:
            assert record.game_object is None, "retained record revived an object"
        if live_collider:
            assert record.collider.handle == collider.handle
        else:
            assert record.collider is None, "retained record revived a collider"
        # CollisionInfo repr used to read its dead GameObject pointer too.
        description = repr(record)
        if query == "collision":
            assert description == f"<CollisionInfo other='{owner.name if live_owner else 'null'}'>"
        else:
            assert description.startswith("<RaycastHit dist=")
        (project / "record-evidence.json").write_text(json.dumps(dict(
            status="passed", query=query, action=action, snapshot=before,
        ), indent=2), encoding="utf-8")
    finally:
        engine.exit()


if __name__ == "__main__":
    exercise(Path(sys.argv[1]), *sys.argv[2:4])
