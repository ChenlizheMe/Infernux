"""Folder deletion publishes child retirement without waiting for a watcher."""
import numpy as np
import pytest

from infernux.core.assets import AssetManager
from infernux.core.material import Material
from infernux.engine.undo import ProjectAssetDeleteCommand
from infernux.lib import AssetRegistry
from tests.gpu.mesh_normal_map_case import source_obj


@pytest.fixture
def imported_folder(engine, scene, tmp_path, monkeypatch):
    database = engine.get_asset_database()
    monkeypatch.setattr(AssetManager, '_engine', engine)
    monkeypatch.setattr(AssetManager, '_asset_database', database)
    folder = tmp_path / 'Assets/Folder'
    folder.joinpath('Nested').mkdir(parents=True)
    mesh = folder / 'Nested/Geometry.obj'
    mesh.write_text(source_obj(False), encoding='utf-8')
    material = folder / 'Surface.mat'
    assert Material.create_lit().save(str(material))
    guids = {}
    for path in (mesh,material):
        result = AssetManager.import_asset(str(path), database=database)
        assert result.succeeded, result.error
        guids[path] = result.guid
    renderer = scene.create_game_object('Folder Mesh').add_component('MeshRenderer')
    renderer.set_mesh_asset_guid(guids[mesh])
    assert AssetRegistry.instance().load_material_by_guid(guids[material]) is not None
    command = ProjectAssetDeleteCommand([str(folder)], project_root=str(tmp_path),
        backup_root=str(tmp_path / 'Library/EditorUndo'), asset_database=database)
    try:
        yield folder, mesh, guids, database, renderer, command
    finally:
        command.dispose()
        # Independent teardown also works when the pre-fix delete left stale mappings.
        for path in guids:
            if path.exists():
                path.unlink()
            if database.get_guid_from_path(str(path)):
                assert AssetManager.delete_asset(str(path), database=database)


def assert_retired(folder, guids, database):
    assert not folder.exists()
    for path,guid in guids.items():
        assert database.get_guid_from_path(str(path)) == ''
        assert database.get_path_from_guid(guid) == ''
        assert not AssetRegistry.instance().is_loaded(guid)


def test_folder_delete_undo_redo_retires_all_children(imported_folder):
    folder, mesh, guids, database, renderer, command = imported_folder
    original = np.asarray(renderer.get_tangents()).copy()
    command.execute()
    for _ in range(2):
        assert_retired(folder, guids, database)
        assert renderer.mesh_asset_guid == guids[mesh], 'Missing references retain their authored identity'
        command.undo()
        for path,guid in guids.items():
            assert path.is_file()
            assert database.get_guid_from_path(str(path)) == guid
        np.testing.assert_array_equal(renderer.get_tangents(), original)
        command.redo()
    assert_retired(folder, guids, database)


def test_recreated_folder_cannot_reuse_previous_mesh(imported_folder):
    folder, mesh, guids, database, renderer, command = imported_folder
    command.execute()
    assert_retired(folder, guids, database)
    mesh.parent.mkdir(parents=True)
    mesh.write_text(source_obj(True), encoding='utf-8')
    result = AssetManager.import_asset(str(mesh), database=database)
    assert result.succeeded, result.error
    assert result.guid != guids[mesh]
    renderer.set_mesh_asset_guid(result.guid)
    np.testing.assert_array_equal(np.asarray(renderer.get_tangents())[:,3], -1.)
