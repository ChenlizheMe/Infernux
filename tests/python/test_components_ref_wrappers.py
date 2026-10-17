"""Stable scene and asset reference identity across copy, rename and retirement."""

import copy
import json

import pytest

from infernux.components.ref_wrappers import GameObjectRef, PrefabRef, ComponentRef
from infernux.components import ref_wrappers


def test_component_ref_rebinds_invalidated_wrapper_when_native_identity_survives(scene):
    from infernux.components.builtin_component import BuiltinComponent

    obj = scene.create_game_object('ReferencedCamera')
    obj.add_component('Camera')
    ref = ComponentRef(go_id=obj.id, component_type='Camera')
    old = ref.resolve()
    handle = ref._cached_handle
    assert old.is_valid
    BuiltinComponent._clear_cache()
    assert old._is_destroyed
    assert scene.resolve_component(handle) is not None
    fresh = ref.resolve()
    assert fresh is not old
    assert fresh.is_valid
    assert fresh.camera_to_world_matrix.shape == (4, 4)


def test_retiring_scene_reference_scope_keeps_other_worlds_and_restores_after_error(scene):
    from infernux.lib import SceneManager

    manager = SceneManager.instance()
    other = manager.create_scene('ReferenceScopeAdditive')
    obj = scene.create_game_object('RetiringTarget')
    obj.add_component('Camera')
    resident = other.create_game_object('ResidentTarget')
    resident.add_component('Camera')
    object_ref = GameObjectRef(obj)
    component_ref = ComponentRef(go_id=obj.id, component_type='Camera')
    resident_ref = ComponentRef(go_id=resident.id, component_type='Camera')
    assert component_ref.resolve().is_valid
    try:
        with pytest.raises(RuntimeError, match='cleanup failed'):
            with ref_wrappers.retiring_scene_references(scene.world_id):
                assert object_ref.resolve() is None
                assert component_ref.resolve() is None
                assert ComponentRef(go_id=obj.id, component_type='Camera').resolve() is None
                assert resident_ref.resolve().is_valid
                raise RuntimeError('cleanup failed')
        assert object_ref.resolve() is obj
        assert component_ref.resolve().is_valid
    finally:
        manager.unload_scene(other)


def test_game_object_reference_identity_empty_state_and_copy():
    empty = GameObjectRef()
    assert not empty and empty == None  # noqa: E711
    assert empty.persistent_id == 0 and empty.name is None and empty.transform is None
    original = GameObjectRef(persistent_id=99)
    same = GameObjectRef(persistent_id=99)
    assert original == same and hash(original) == hash(same)
    assert original != GameObjectRef(persistent_id=100)
    shallow, deep = copy.copy(original), copy.deepcopy(original)
    assert shallow == deep == original
    assert shallow is not original and deep is not original


def test_component_reference_identity_serialization_and_copy():
    empty = ComponentRef()
    assert not empty and empty.go_id == 0 and empty.component_type == ""
    assert empty.display_name == "None" and empty.some_method is None
    original = ComponentRef(go_id=42, component_type="Rigidbody")
    same = ComponentRef(go_id=42, component_type="Rigidbody")
    assert original == same and hash(original) == hash(same)
    assert original != ComponentRef(go_id=42, component_type="Light")
    assert original != ComponentRef(go_id=43, component_type="Rigidbody")
    copied = copy.copy(original)
    document = original._serialize()
    restored = ComponentRef._from_dict(document)
    assert document["$type"] == "component_ref"
    assert copied == restored == original and copied is not original


def test_prefab_reference_resolves_guid_after_rename_and_root_name_change(engine, scene, tmp_path):
    from pathlib import Path
    from infernux.engine.ui.project_file_ops import create_prefab_from_gameobject, do_rename

    database = engine.get_asset_database()
    directory = Path(database.assets_root) / tmp_path.name
    directory.mkdir()
    source = scene.create_game_object("EnemyRoot")
    ok, path = create_prefab_from_gameobject(source, str(directory), database)
    assert ok
    guid = database.get_guid_from_path(path)
    reference = PrefabRef(guid=guid, path_hint="stale.prefab")
    assert reference.game_object is reference
    assert Path(reference.path_hint) == Path(path)
    assert reference.name == "EnemyRoot"
    moved = Path(do_rename(path, "Boss", database))
    document = json.loads(moved.read_text(encoding="utf-8"))
    document["root_object"]["name"] = "BossRoot"
    moved.write_text(json.dumps(document), encoding="utf-8")
    assert database.reimport_asset(str(moved))
    assert Path(reference.path_hint) == moved
    assert reference.name == "BossRoot"
    assert database.get_guid_from_path(str(moved)) == guid


def test_prefab_missing_guid_does_not_promote_a_path_hint(monkeypatch):
    monkeypatch.setattr(ref_wrappers, "_get_prefab_asset_database", lambda: None)
    reference = PrefabRef(guid="enemy-guid", path_hint="Assets/Legacy.prefab")
    assert reference.path_hint == "" and reference.name == "enemy-gu"
    assert reference._serialize() == {
        "$type": "asset_ref", "asset_type": "Prefab", "guid": "enemy-guid",
    }
    empty = PrefabRef._from_dict("", "Assets/Legacy.prefab")
    assert not empty and empty.path_hint == "" and empty.name == "None"


def test_reference_resolution_tracks_live_persistent_scene_lifetime(scene):
    from infernux.lib import SceneManager

    manager = SceneManager.instance()
    target = scene.create_game_object("PersistentReferenceTarget")
    target.add_component("Camera")
    object_reference = GameObjectRef(target)
    component_reference = ComponentRef(go_id=target.id, component_type="Camera")
    manager.play()
    try:
        manager.dont_destroy_on_load(target)
        manager.prepare_active_scene_replacement()
        persistent = manager.get_runtime_persistent_scene()
        assert target.scene is persistent and persistent is not scene
        assert object_reference.resolve() is target
        assert target.game_object is target
        assert component_reference.resolve().game_object is target
        persistent.destroy_game_object(target)
        persistent.process_pending_destroys()
        assert object_reference.resolve() is None
        assert component_reference.resolve() is None
    finally:
        manager.stop()
