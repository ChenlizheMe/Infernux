"""Public physics and camera queries preserve shared resource proxy identity."""
from __future__ import annotations

import gc
from pathlib import Path
from types import SimpleNamespace
import weakref

import pytest
import infernux as inx
from infernux.application import Application
from infernux.core.render_texture import PixelFormat


def test_physic_material_constructor_reuses_live_native_proxy():
    material = inx.PhysicMaterial()
    assert inx.PhysicMaterial(material.native) is material
    assert inx.PhysicMaterial() is not material


def test_physic_material_subclass_can_initialize_through_super():
    class GameplayMaterial(inx.PhysicMaterial):
        def __init__(self, native=None):
            super().__init__(native)
            self.gameplay_tag = 'ice'
    material = GameplayMaterial()
    assert material.gameplay_tag == 'ice'
    assert GameplayMaterial(material.native) is material


def test_colliders_share_the_assigned_physic_material_proxy(scene):
    material = inx.PhysicMaterial()
    box = scene.create_game_object('Box owner').add_component(inx.BoxCollider)
    sphere = scene.create_game_object('Sphere owner').add_component(inx.SphereCollider)
    box.physic_material = material
    sphere.physic_material = material
    assert box.physic_material.resolve() is material
    assert sphere.physic_material.resolve() is material
    assert box.physic_material.resolve() is box.physic_material.resolve()
    sphere.physic_material.resolve().friction = 0.73
    assert box.physic_material.resolve().friction == pytest.approx(0.73)
    box.physic_material = None
    assert box.physic_material.resolve() is None
    assert sphere.physic_material.resolve() is material


def test_physic_material_proxy_cache_does_not_own_unused_resources():
    material = inx.PhysicMaterial()
    native = material.native
    reference = weakref.ref(material)
    del material
    gc.collect()
    assert reference() is None
    replacement = inx.PhysicMaterial(native)
    assert replacement.native is native
    assert inx.PhysicMaterial(native) is replacement


def test_physic_material_asset_queries_and_deletion_follow_current_guid_owner(engine, tmp_path, monkeypatch):
    from infernux.core.assets import AssetManager
    from infernux.core.asset_ref import PhysicMaterialRef
    from infernux.engine.ui.project_file_ops import create_physic_material

    database = engine.get_asset_database()
    monkeypatch.setattr(AssetManager, '_engine', engine)
    monkeypatch.setattr(AssetManager, '_asset_database', database)
    monkeypatch.setattr(Application, 'data_path', staticmethod(lambda: str(Path(database.assets_root).parent)))
    directory = Path(database.assets_root)/tmp_path.name
    directory.mkdir()
    assert create_physic_material(str(directory), 'SharedPhysics') == (True, '')
    path = directory/'SharedPhysics.physicMaterial'
    imported = AssetManager.import_asset(str(path), database=database)
    assert imported, imported.error
    material = inx.PhysicMaterial.load(str(path))
    assert material is not None
    reference = PhysicMaterialRef(guid=imported.guid)
    assert reference.resolve() is material
    assert inx.PhysicMaterial.load_by_guid(imported.guid) is material
    assert AssetManager.load_by_guid(imported.guid) is material
    assert AssetManager.load_by_guid(imported.guid, asset_type=inx.PhysicMaterial) is material
    assert AssetManager.load(f'Assets/{tmp_path.name}/SharedPhysics.physicMaterial') is material
    document = path.read_bytes()
    metadata_path = Path(str(path)+'.meta')
    metadata = metadata_path.read_bytes()
    deleted = AssetManager.delete_asset(str(path), database=database)
    assert deleted, deleted.error
    assert reference.resolve() is None
    assert reference.is_missing
    path.write_bytes(document)
    metadata_path.write_bytes(metadata)
    restored = AssetManager.import_asset(str(path), database=database)
    assert restored and restored.guid == imported.guid
    replacement = reference.resolve()
    assert replacement is not None and replacement is not material
    assert inx.PhysicMaterial.load_by_guid(restored.guid) is replacement
    assert not reference.is_missing


@pytest.fixture
def graphical_engine(engine, monkeypatch):
    facade = SimpleNamespace(get_native_engine=lambda: engine)
    monkeypatch.setattr(Application, '_current_engine', staticmethod(lambda: facade))
    return engine


def test_camera_queries_return_the_created_target_proxy(graphical_engine, scene):
    target = inx.RenderTexture(16, 12, depth_format=PixelFormat.D32_SFLOAT)
    first = scene.create_game_object('First camera').add_component(inx.Camera)
    second = scene.create_game_object('Second camera').add_component(inx.Camera)
    try:
        first.target_texture = second.target_texture = target
        assert first.target_texture is target
        assert second.target_texture is target
        assert first.target_texture is first.target_texture
        target.resize(24, 18)
        assert second.target_texture is target
        assert (first.target_texture.width, second.target_texture.height) == (24, 18)
        first.target_texture = None
        assert first.target_texture is None
        assert second.target_texture is target
    finally:
        first.target_texture = second.target_texture = None


def test_render_target_subclass_can_initialize_through_super(graphical_engine):
    class GameplayTarget(inx.RenderTexture):
        def __init__(self, width, height):
            super().__init__(width, height)
            self.gameplay_tag = 'monitor'
    target = GameplayTarget(8, 6)
    assert target.gameplay_tag == 'monitor'
    assert (target.width, target.height) == (8, 6)


def test_render_target_proxy_cache_is_weak(graphical_engine, scene):
    target = inx.RenderTexture(8, 6, depth_format=PixelFormat.D32_SFLOAT)
    camera = scene.create_game_object('Retained target camera').add_component(inx.Camera)
    try:
        camera.target_texture = target
        reference = weakref.ref(target)
        native = target._native
        del target
        gc.collect()
        assert reference() is None
        replacement = camera.target_texture
        assert replacement._native is native
        assert camera.target_texture is replacement
    finally:
        camera.target_texture = None
