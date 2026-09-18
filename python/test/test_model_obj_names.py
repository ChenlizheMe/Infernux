"""Memory IO must not leak Assimp's virtual source name into authored objects."""
from pathlib import Path

import pytest

from Infernux.lib import AssetRegistry
from Infernux.lib._Infernux import make_model_mesh_reference


@pytest.mark.parametrize('filename', ['Bridge.obj', '模型 试验.OBJ'])
def test_obj_root_name_authored_children_and_legacy_references(engine, scene, tmp_path, filename):
    database = engine.get_asset_database()
    source = Path(database.assets_root) / tmp_path.name / filename
    source.parent.mkdir()
    # An author is allowed to use even the old reserved-looking name for a child.
    source.write_text('o $$$___magic___$$$.obj\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n', encoding='utf-8')
    result = database.import_asset(str(source))
    assert result, result.error
    registry = AssetRegistry.instance()
    try:
        mesh = registry.load_mesh(str(source))
        assert mesh.get_model_nodes()[0]['name'] == filename
        assert mesh.get_model_nodes()[-1]['name'] == '$$$___magic___$$$.obj'
        root = scene.create_from_model(result.guid)
        assert root.name == source.stem
        assert root.get_children()[0].name == filename
        # Existing scene/picker references upgrade only the synthetic first element.
        path = ['$$$___magic___$$$.obj', '$$$___magic___$$$.obj']
        obj = scene.create_from_model(make_model_mesh_reference(result.guid, path), 'Legacy child')
        renderer = obj.get_component('MeshRenderer')
        canonical = [filename, '$$$___magic___$$$.obj']
        assert renderer.serialize_document()['modelNodePath'] == canonical
        saved = renderer.serialize_document()
        saved['modelNodePath'] = path
        assert renderer.deserialize_document(saved)
        assert renderer.serialize_document()['modelNodePath'] == canonical
        assert scene._commit_document(scene.serialize_document())
        restored = next(o for o in scene.get_root_objects() if o.name == 'Legacy child')
        assert restored.get_component('MeshRenderer').serialize_document()['modelNodePath'] == canonical
        assert registry.reload_asset(result.guid)
        assert mesh.get_model_nodes()[0]['name'] == filename
    finally:
        registry.invalidate_asset(result.guid)
        database.delete_asset(str(source))
        source.unlink(missing_ok=True)
        Path(str(source) + '.meta').unlink(missing_ok=True)
