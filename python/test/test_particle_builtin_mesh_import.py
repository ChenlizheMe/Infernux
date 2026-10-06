"""Particle built-in mesh symbols never become durable asset GUID edges."""
import json

import pytest

from infernux.graph.types import AssetReference, BUILTIN_MESH_NAMES, TypeRef, ValueType, builtin_mesh_reference
from infernux.lib import AssetDependencyGraph
from infernux.particle import ParticleGraphAsset, ParticleParameter


def _document(name):
    return ParticleGraphAsset(parameters=(
        ParticleParameter(stable_id="builtin", name="Builtin", value_type=TypeRef(ValueType.MESH),
                          default=builtin_mesh_reference(name).to_dict()),
        ParticleParameter(stable_id="project", name="Project", value_type=TypeRef(ValueType.MESH),
                          default=AssetReference(guid="c" * 32).to_dict()),
    )).to_dict()


@pytest.mark.parametrize("name", BUILTIN_MESH_NAMES)
def test_builtin_mesh_import_keeps_only_project_asset_dependencies(engine, tmp_path, name):
    database = engine.get_asset_database()
    path = tmp_path / "builtin.particlegraph"
    document = _document(name)
    path.write_text(json.dumps(document), encoding="utf-8")
    result = database.import_asset(str(path))
    try:
        assert result.succeeded, result.error
        assert AssetDependencyGraph.instance().get_dependencies(result.guid) == {"c" * 32}
        assert database.get_guid_from_path(str(path)) == result.guid
        # Reimport has the same identity and does not invent a meta for the symbol.
        repeated = database.reimport_asset(str(path))
        assert repeated.succeeded and repeated.guid == result.guid, repeated.error
        assert AssetDependencyGraph.instance().get_dependencies(result.guid) == {"c" * 32}
        assert not database.get_path_from_guid(f"builtin-mesh:{name}")
    finally:
        if database.contains_path(str(path)):
            assert database.delete_asset(str(path))


@pytest.mark.parametrize("invalid", ["builtin-mesh:Unknown", "builtin-mesh:cube", "builtin-mesh:", "builtin-mesh:../Cube"])
def test_unknown_builtin_mesh_cannot_replace_published_dependency_graph(engine, tmp_path, invalid):
    database = engine.get_asset_database()
    path = tmp_path / "invalid.particlegraph"
    document = _document("Cube")
    path.write_text(json.dumps(document), encoding="utf-8")
    original = database.import_asset(str(path))
    try:
        assert original.succeeded, original.error
        document["parameters"][0]["default"]["guid"] = invalid
        path.write_text(json.dumps(document), encoding="utf-8")
        result = database.reimport_asset(str(path))
        assert not result.succeeded
        assert "non-GUID dependency identity" in result.error
        assert database.get_guid_from_path(str(path)) == original.guid
        assert AssetDependencyGraph.instance().get_dependencies(original.guid) == {"c" * 32}
    finally:
        if database.contains_path(str(path)):
            assert database.delete_asset(str(path))
