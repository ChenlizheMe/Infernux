"""Public typed shader operations exercise the real asset/renderer boundary."""

from types import SimpleNamespace
from pathlib import Path

import pytest

from infernux.application import Application
from infernux.core.assets import AssetManager
from infernux.core.shader import Shader
from infernux.lib import AssetRegistry


@pytest.fixture
def shader_assets(engine, monkeypatch):
    monkeypatch.setattr(AssetManager, "_engine", engine)
    monkeypatch.setattr(AssetManager, "_asset_database", engine.get_asset_database())
    monkeypatch.setattr(AssetManager, "_registry", AssetRegistry.instance())
    paths = []
    asset_root = Path(AssetManager.require_asset_database().assets_root)
    asset_root.mkdir(parents=True, exist_ok=True)

    def create(name, stage="fragment", filename=None, source=None):
        path = asset_root / (filename or (name + (".frag.py" if stage == "fragment" else ".vert.py")))
        if source is None:
            source = 'import infernux.shader as shader\n'
            if stage == "fragment":
                source += (f'class Public(shader.fullscreen):\n    shader_id = {name!r}\n'
                       '    hidden = True\n'
                       '    def main(self, ctx: shader.fragment_input) -> shader.vec4:\n'
                       '        return shader.vec4(1,0,0,1)\n')
            else:
                source += (f'class Public(shader.vert):\n    shader_id = {name!r}\n'
                       '    hidden = True\n'
                       '    def deform(self, vertex: shader.VertexData, ctx: shader.VertexContext) -> shader.VertexData:\n'
                       '        return vertex\n')
        path.write_text(source, encoding="utf-8")
        result = AssetManager.import_asset(str(path))
        assert result, result.error
        paths.append(path)
        return path, result.guid

    yield create
    for path in paths:
        AssetManager.delete_asset(str(path))


def test_query_tracks_resource_host_lifetime(engine, monkeypatch):
    monkeypatch.setattr(AssetManager, "_engine", None)
    assert not Shader.is_loaded("infernux/lit", "fragment")
    monkeypatch.setattr(AssetManager, "_engine", SimpleNamespace(_engine=engine))
    assert Shader.is_loaded("infernux/lit", "fragment")
    monkeypatch.setattr(AssetManager, "_engine", None)
    assert not Shader.is_loaded("infernux/lit", "fragment")


def test_reload_publishes_standalone_asset_by_name_and_path(shader_assets):
    path, guid = shader_assets("Public_Reload_040")
    assert not Shader.is_loaded("Public_Reload_040", "fragment")
    assert Shader.reload("Public_Reload_040", "fragment")
    assert Shader.is_loaded("Public_Reload_040", "fragment")
    registry = AssetRegistry.instance()
    version = registry.get_asset_version(guid)
    path.write_text(path.read_text(encoding="utf-8").replace("vec4(1,0,0,1)", "vec4(0,1,0,1)"), encoding="utf-8")
    assert Shader.reload(str(path))
    assert registry.get_asset_version(guid) > version
    assert Shader.is_loaded("Public_Reload_040", "fragment")


def test_reload_rejects_unregistered_and_wrong_stage(shader_assets):
    path, _ = shader_assets("Public_Wrong_Stage_040")
    with pytest.raises(FileNotFoundError, match="not registered"):
        Shader.reload("Unregistered_Shader_040")
    with pytest.raises(ValueError, match="matching GPU source stage"):
        Shader.reload(str(path), "vertex")


def test_duplicate_name_requires_explicit_path(shader_assets):
    first, _ = shader_assets("Public_Duplicate_040", filename="First.frag.py")
    shader_assets("Public_Duplicate_040", filename="Second.frag.py")
    with pytest.raises(ValueError, match="Ambiguous"):
        Shader.reload("Public_Duplicate_040")
    assert Shader.reload(str(first))


def test_same_name_in_two_stages_reloads_both(shader_assets):
    shader_assets("Public_Pair_040", "vertex")
    shader_assets("Public_Pair_040", "fragment")
    assert Shader.reload("Public_Pair_040")
    assert Shader.is_loaded("Public_Pair_040", "vertex")
    assert Shader.is_loaded("Public_Pair_040", "fragment")
    for stage in ("vertex", "fragment"):
        assert Shader.reload("Public_Pair_040", stage)
        assert Shader.is_loaded("Public_Pair_040", "vertex")
        assert Shader.is_loaded("Public_Pair_040", "fragment")


def test_library_reload_and_query_use_the_same_guid_identity(shader_assets):
    name = "PublicLibrary"
    path, guid = shader_assets(name, filename=name + ".shaderlib.py", source='''import infernux.shader as shader
def color() -> shader.vec4:
    return shader.vec4(0.2, 0.4, 0.6, 1.0)
''')
    assert Shader.is_loaded(guid, "library")
    previous = AssetRegistry.instance().get_asset_version(guid)
    assert Shader.reload(guid, "library")
    assert AssetRegistry.instance().get_asset_version(guid) > previous
    assert all(Shader.is_loaded(selector, "library") for selector in (guid, name, str(path)))
    assert not Shader.is_loaded(guid, "fragment")


def test_published_particle_schema_tracks_defaults_but_ignores_field_labels(shader_assets):
    source = '''import infernux.shader as shader
class Particle(shader.particle):
    shader_id = "tests/particle/schema"
    gain = shader.material.float(default=0.25, label="Gain", range=(0.0, 1.0))
    def main(self, ctx: shader.ParticleContext) -> shader.vec4:
        return shader.vec4(self.gain, 0.0, 0.0, 1.0)
'''
    path, guid = shader_assets("ParticleSchema", source=source)
    assert Shader.reload(guid)
    fields, initial_revision = Shader._surface_schemas[guid]
    assert ("gain", "float", 0.25, False) in fields
    assert initial_revision > 0
    renamed = source.replace('label="Gain"', 'label="Strength"')
    path.write_text(renamed, encoding="utf-8")
    assert Shader.reload(str(path))
    assert Shader._surface_schemas[guid][1] == initial_revision
    path.write_text(renamed.replace("default=0.25", "default=0.75"), encoding="utf-8")
    assert Shader.reload("tests/particle/schema", "fragment")
    fields, changed_revision = Shader._surface_schemas[guid]
    assert ("gain", "float", 0.75, False) in fields
    assert changed_revision > initial_revision


def test_reload_reports_compile_failure(shader_assets):
    path, _ = shader_assets("Public_Broken_040")
    assert Shader.reload(str(path))
    path.write_text(
        path.read_text(encoding="utf-8").replace("return shader.vec4(", "return invalid_symbol + shader.vec4("), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="Shader reload failed"):
        Shader.reload(str(path))
    assert Shader.is_loaded("Public_Broken_040", "fragment")


def test_frozen_player_reload_is_not_an_authoring_write(monkeypatch):
    monkeypatch.setattr(Application, "is_player", staticmethod(lambda: True))
    with pytest.raises(RuntimeError, match="read-only"):
        Shader.reload("Assets/Shader.frag.py")
