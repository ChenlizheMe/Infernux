"""Importer-authored static colliders share the normal scene/undo/physics path."""
from pathlib import Path

import numpy as np
import pytest

from Infernux.core.asset_types import MeshImportSettings, read_mesh_import_settings
from Infernux.core.assets import AssetManager
from Infernux.lib import Physics, Vector3
from Infernux.lib._Infernux import make_model_mesh_reference
from test_model_hierarchy_instances import hierarchy_asset, descendants, world_matrix


def enable_colliders(engine, monkeypatch, database, source, enabled=True):
    monkeypatch.setattr(AssetManager, '_engine', engine)
    monkeypatch.setattr(AssetManager, '_asset_database', database)
    settings = read_mesh_import_settings(str(source))
    settings.generate_colliders = enabled
    result = AssetManager.apply_import_settings('mesh', str(source), settings)
    assert result, result.error


def test_collider_option_defaults_validation_and_model_page():
    from Infernux.engine.ui import asset_details_renderer as renderer
    settings = MeshImportSettings()
    assert not settings.generate_colliders
    old = settings.to_dict()
    del old['generate_colliders']
    assert not MeshImportSettings.from_dict(old).generate_colliders
    for value in (1, 'true', None):
        with pytest.raises(TypeError, match='generate_colliders'):
            MeshImportSettings.from_dict({**old, 'generate_colliders': value})
    renderer._ensure_categories()
    field = next(f for f in renderer._categories['mesh'].editable_fields if f.key == 'generate_colliders')
    assert field.page == 'model'


def test_static_colliders_preserve_selection_transform_and_scene_roundtrip(scene, hierarchy_asset, engine, monkeypatch):
    database, source, guid = hierarchy_asset
    root = scene.create_from_model(guid, 'Before')
    assert not any(o.get_component('MeshCollider') for o in descendants(root).values())
    enable_colliders(engine, monkeypatch, database, source)
    root = scene.create_from_model(guid, 'With collision')
    objects = descendants(root)
    for obj in (root, *objects.values()):
        collider = obj.get_component('MeshCollider')
        assert bool(collider) == bool(obj.get_component('MeshRenderer'))
        assert obj.get_component('Rigidbody') is None
        if collider:
            assert not collider.convex
    Physics.sync_transforms()
    upper = objects['Upper']
    point = (world_matrix(upper) @ np.array([.2, .2, 0, 1]))[:3]
    assert not upper.get_component('MeshCollider').shape_error
    hit = Physics.raycast(Vector3(*(point + [0, 0, 5])), Vector3(0, 0, -1), 10)
    assert hit is not None and hit.distance == pytest.approx(5)
    assert hit  # A hit is a value record, not an entity exposing an id.
    root.transform.position = Vector3(20, 0, 0)
    Physics.sync_transforms()
    assert Physics.raycast(Vector3(*(point + [0, 0, 5])), Vector3(0, 0, -1), 10) is None
    assert Physics.raycast(Vector3(*(point + [20, 0, 5])), Vector3(0, 0, -1), 10) is not None
    # This is an instantiation policy, not permission to remove author components.
    enable_colliders(engine, monkeypatch, database, source, False)
    after = scene.create_from_model(guid, 'After')
    assert not any(o.get_component('MeshCollider') for o in descendants(after).values())
    document = scene.serialize_document()
    assert scene._commit_document(document)
    restored = next(o for o in scene.get_root_objects() if o.name == 'With collision')
    assert descendants(restored)['Upper'].get_component('MeshCollider')
    Physics.sync_transforms()
    assert Physics.raycast(Vector3(*(point + [20, 0, 5])), Vector3(0, 0, -1), 10) is not None


@pytest.mark.parametrize('child_only', [False, True])
def test_generated_colliders_participate_in_single_undo(scene, hierarchy_asset, engine, monkeypatch, child_only):
    from Infernux.engine.undo import UndoManager
    from Infernux.engine.interaction import ClipboardService, SelectionService, SceneObjectCommandService
    database, source, guid = hierarchy_asset
    enable_colliders(engine, monkeypatch, database, source)
    reference = make_model_mesh_reference(guid, ['Assembly', 'Empty pivot', 'Upper']) if child_only else guid
    previous = UndoManager._instance
    manager = UndoManager()
    try:
        service = SceneObjectCommandService(SelectionService(), ClipboardService())
        root = service.create_model_object(reference, is_guid=True)
        object_id = root.id
        expected = 1 if child_only else 2
        count = lambda obj: sum(bool(o.get_component('MeshCollider')) for o in (obj, *descendants(obj).values()))
        assert count(root) == expected
        assert len(manager.action_journal.entries) == 1
        Physics.sync_transforms()
        for obj in (root, *descendants(root).values()):
            collider = obj.get_component('MeshCollider')
            if collider:
                assert not collider.shape_error
        manager.undo()
        assert scene.find_by_id(object_id) is None
        manager.redo()
        assert count(scene.find_by_id(object_id)) == expected
    finally:
        manager.clear()
        UndoManager._instance = previous


def test_skinned_model_does_not_get_static_colliders(scene, engine, tmp_path, monkeypatch):
    database = engine.get_asset_database()
    source = Path(database.assets_root) / tmp_path.name / 'Skinned.fbx'
    source.parent.mkdir()
    fixture = Path(__file__).resolve().parents[2] / 'external/assimp/test/models/FBX/animation_with_skeleton.fbx'
    source.write_bytes(fixture.read_bytes())
    result = database.import_asset(str(source))
    assert result, result.error
    try:
        enable_colliders(engine, monkeypatch, database, source)
        root = scene.create_from_model(result.guid)
        objects = (root, *descendants(root).values())
        assert any(o.get_component('SkinnedMeshRenderer') for o in objects)
        assert not any(o.get_component('MeshCollider') for o in objects)
    finally:
        database.delete_asset(str(source))
        source.unlink(missing_ok=True)
        Path(str(source) + '.meta').unlink(missing_ok=True)
