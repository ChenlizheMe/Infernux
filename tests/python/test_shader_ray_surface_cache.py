"""Actual native author-cache hits and world Surface membership publication."""

import json
from pathlib import Path
import uuid

from infernux._compiler import shader
from infernux.core.assets import AssetManager
from infernux.core.shader import Shader
from infernux.lib import AssetRegistry, _Infernux


ORDINARY = '''import infernux.shader as shader
class Ordinary(shader.fullscreen):
    def main(self) -> shader.vec4:
        return shader.vec4(0.2, 0.4, 0.6, 1.0)
'''
QUERY = '''import infernux.shader as shader
class Query(shader.fullscreen):
    def main(self) -> shader.vec4:
        hit = shader.ray.trace_closest(shader.vec3(0.0), shader.vec3(0.0, 0.0, 1.0),
                                      0.0, 10.0, shader.uint(0xffffffff), 0.0)
        return shader.vec4(hit.surface.albedo, shader.float(hit.available))
'''
SURFACE = '''import infernux.shader as shader
class Emitter(shader.surface):
    def surface(self, ctx: shader.SurfaceContext) -> shader.SurfaceData:
        return shader.SurfaceData(emission=shader.vec3(1.0), normal_ws=ctx.normal_ws)
'''


def test_only_world_program_cache_tracks_surface_membership(engine, monkeypatch):
    database = engine.get_asset_database()
    registry = AssetRegistry.instance()
    monkeypatch.setattr(AssetManager, "_engine", engine)
    monkeypatch.setattr(AssetManager, "_asset_database", database)
    monkeypatch.setattr(AssetManager, "_registry", registry)
    prefix = "RayCatalog_" + uuid.uuid4().hex
    created = []
    def create(label, source):
        path = Path(database.assets_root) / (prefix + label + ".frag.py")
        path.write_text(source, encoding="utf-8")
        result = AssetManager.import_asset(str(path))
        assert result, result.error
        created.append((path, result.guid))
        return path, result.guid
    prepared = []
    prepare = shader._prepare_source
    def observe(source, path, *args, **kwargs):
        if not kwargs["metadata_only"]:
            prepared.append(Path(path))
        return prepare(source, path, *args, **kwargs)
    monkeypatch.setattr(shader, "_prepare_source", observe)
    try:
        ordinary, _ = create("Ordinary", ORDINARY)
        assert Shader.reload(str(ordinary))
        prepared.clear()
        create("Unrelated", ORDINARY)
        prepared.clear()
        assert Shader.reload(str(ordinary))
        assert ordinary not in prepared

        query, query_guid = create("Query", QUERY)
        assert Shader.reload(str(query))
        prepared.clear()
        assert Shader.reload(str(query))
        assert query not in prepared

        surface, surface_guid = create("Emitter", SURFACE)
        prepared.clear()
        assert Shader.reload(str(query))
        assert prepared.count(query) == 1
        schema = json.loads(_Infernux._shader_source_schema(query_guid))
        assert surface_guid in schema["ray_surface_catalog"]
        assert surface_guid in {item["guid"] for item in schema["ray_surfaces"]}

        prepared.clear()
        surface.write_text(SURFACE.replace("shader.vec3(1.0)", "shader.vec3(2.0)"), encoding="utf-8")
        assert Shader.reload(str(surface))
        assert query in prepared

        surface.unlink()
        deleted = AssetManager.delete_asset(str(surface))
        assert deleted, deleted.error
        prepared.clear()
        assert Shader.reload(str(query))
        schema = json.loads(_Infernux._shader_source_schema(query_guid))
        assert surface_guid not in schema["ray_surface_catalog"]
        assert surface_guid not in {item["guid"] for item in schema["ray_surfaces"]}
        prepared.clear()
        assert Shader.reload(str(ordinary))
        assert ordinary not in prepared
    finally:
        for path, guid in reversed(created):
            if database.get_path_from_guid(guid):
                path.unlink(missing_ok=True)
                result = AssetManager.delete_asset(str(path))
                assert result, result.error
