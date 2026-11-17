"""Dependency edits publish one complete set of real native shader candidates."""

import json
from pathlib import Path
import uuid

import pytest

from infernux.core.assets import AssetManager
from infernux.core.material import Material
from infernux.core.shader import Shader
from infernux.debug import DebugConsole
from infernux.engine.path_utils import path_key
from infernux.lib import AssetMutationErrorCode, AssetRegistry, InxMaterial


@pytest.fixture
def dependency_sources(engine, monkeypatch):
    database = engine.get_asset_database()
    registry = AssetRegistry.instance()
    monkeypatch.setattr(AssetManager, "_engine", engine)
    monkeypatch.setattr(AssetManager, "_asset_database", database)
    monkeypatch.setattr(AssetManager, "_registry", registry)
    root = Path(database.assets_root)
    root.mkdir(parents=True, exist_ok=True)
    prefix = "Dependency_" + uuid.uuid4().hex
    paths = []

    def create(label, extension, source):
        path = root / (prefix + label + extension)
        path.write_text(source, encoding="utf-8")
        imported = AssetManager.import_asset(str(path))
        assert imported, imported.error
        paths.append((path, imported.guid))
        return path, imported.guid

    yield prefix, create, registry
    for _, guid in reversed(paths):
        path = database.get_path_from_guid(guid)
        if path:
            # AssetManager processes a filesystem deletion notification; the
            # source must already be gone. Leaving an Unlit override on disk
            # changes the shading model used by subsequent rendering tests.
            Path(path).unlink()
            result = AssetManager.delete_asset(path)
            assert result, result.error


SHARED_COLOR = '''import infernux.shader as shader

def shared_color() -> shader.vec4:
    return shader.vec4(0.2, 0.3, 0.4, 1.0)
'''
OPTIONAL_COLOR = '''
def optional_color() -> shader.vec4:
    return shared_color()
'''
LIBRARY = SHARED_COLOR + OPTIONAL_COLOR
REVISED_SHARED_COLOR = SHARED_COLOR.replace("0.2, 0.3, 0.4", "0.7, 0.6, 0.5")


def _shaderlib_import(path, function):
    module = path.name.removesuffix(".shaderlib.py")
    return f"from .{module} import {function}\n"


def _fullscreen_source(shader_id, library=None, function="optional_color"):
    imported = _shaderlib_import(library, function) if library is not None else ""
    color = f"{function}()" if library is not None else "shader.vec4(1.0)"
    return f'''import infernux.shader as shader
{imported}
class Color(shader.fullscreen):
    shader_id = {shader_id!r}

    def main(self) -> shader.vec4:
        return {color}
'''


def _surface_source(shader_id, *, library=None, function="optional_color", signal=False):
    imported = _shaderlib_import(library, function) if library is not None else ""
    field = "    signal = shader.material.float(default=0.25)\n" if signal else ""
    albedo = f"{function}().rgb" if library is not None else (
        "shader.vec3(self.signal)" if signal else "shader.vec3(0.2, 0.3, 0.4)"
    )
    return f'''import infernux.shader as shader
{imported}
class Color(shader.surface):
    shader_id = {shader_id!r}
    shading_model = shader.ref("infernux/shading/unlit")
{field}
    def surface(self, ctx: shader.SurfaceContext) -> shader.SurfaceData:
        return shader.SurfaceData(albedo={albedo})
'''


def _vertex_source(shader_id):
    return f'''import infernux.shader as shader

class Shift(shader.vert):
    shader_id = {shader_id!r}
    signal = shader.material.float(default=0.25)

    def deform(self, vertex: shader.VertexData, ctx: shader.VertexContext) -> shader.VertexData:
        vertex.position_os.x += self.signal
        return vertex
'''


def _linked_material(engine, create, label, *, vertex, fragment):
    document = InxMaterial.create_default_unlit().serialize_document()
    document.update(builtin=False, name=fragment[1], shaders={
        stage: {"guid": guid, "shader_id": shader_id}
        for stage, (guid, shader_id) in (("vertex", vertex), ("fragment", fragment))
    })
    _, guid = create(label, ".mat", json.dumps(document))
    material = AssetManager.load(guid, Material)
    assert engine.refresh_material_pipeline(material.native)
    return material


def _compile_diagnostics(owner_path):
    return [entry for entry in DebugConsole.instance().get_entries()
            if entry.diagnostic_owner == "asset.compile" and entry.diagnostic_key == path_key(owner_path)]


def test_standalone_dependency_failure_preserves_every_published_generation(engine, dependency_sources):
    prefix, create, registry = dependency_sources
    library, _ = create("Library", ".shaderlib.py", LIBRARY)
    candidates = []
    for label in ("A", "B"):
        name = prefix + label
        path, guid = create(label, ".frag.py", _fullscreen_source(name, library, "shared_color"))
        candidates.append((guid, path, name))
    # The reload prepares consumers in GUID order. Reject the second so the
    # first, independently valid candidate is already prepared at rejection.
    _, rejecting_path, rejecting_name = max(candidates)
    rejecting_path.write_text(_fullscreen_source(rejecting_name, library), encoding="utf-8")
    stages = []
    for guid, path, _ in candidates:
        assert Shader.reload(str(path))
        stages.append(guid)
    versions = [registry.get_asset_version(guid) for guid in stages]
    assert all(versions)

    library.write_text(REVISED_SHARED_COLOR, encoding="utf-8")
    rejected = AssetManager.reimport_asset(str(library))
    assert not rejected and rejected.database_committed
    assert rejected.error_code == AssetMutationErrorCode.RUNTIME_APPLY_FAILED
    assert "has no export 'optional_color'" in rejected.error
    assert [registry.get_asset_version(guid) for guid in stages] == versions
    assert all(engine.is_shader_loaded(guid, "fragment") for guid in stages)

    library.write_text(REVISED_SHARED_COLOR + OPTIONAL_COLOR, encoding="utf-8")
    accepted = AssetManager.reimport_asset(str(library))
    assert accepted, accepted.error
    assert [registry.get_asset_version(guid) for guid in stages] == [v + 1 for v in versions]


def test_linked_dependency_failure_does_not_retire_a_successful_sibling_program(engine, dependency_sources):
    prefix, create, _ = dependency_sources
    library, _ = create("Library", ".shaderlib.py", LIBRARY)
    vertex_name = prefix + " Vertex"
    _, vertex_guid = create("Vertex", ".vert.py", _vertex_source(vertex_name))
    materials = []
    for label, function in (("A", "shared_color"), ("B", "optional_color")):
        name = prefix + label
        _, fragment_guid = create(
            label, ".frag.py", _surface_source(name, library=library, function=function),
        )
        materials.append(_linked_material(engine, create, label, vertex=(vertex_guid, vertex_name),
                                          fragment=(fragment_guid, name)))
    retired = engine.gpu_residency_snapshot["shader_hot_reload_retirement_count"]
    library.write_text(REVISED_SHARED_COLOR, encoding="utf-8")
    rejected = AssetManager.reimport_asset(str(library))
    assert not rejected and "has no export 'optional_color'" in rejected.error
    assert engine.gpu_residency_snapshot["shader_hot_reload_retirement_count"] == retired
    for material in materials:
        assert engine.refresh_material_pipeline(material.native)

    assert engine.gpu_residency_snapshot["shader_hot_reload_retirement_count"] == retired
    library.write_text(REVISED_SHARED_COLOR + OPTIONAL_COLOR, encoding="utf-8")
    accepted = AssetManager.reimport_asset(str(library))
    assert accepted, accepted.error
    assert engine.gpu_residency_snapshot["shader_hot_reload_retirement_count"] > retired


@pytest.mark.parametrize("retarget", (True, False), ids=("missing-export", "missing-module"))
def test_dependency_move_keeps_guid_and_reports_rejected_runtime_candidate(engine, dependency_sources, retarget):
    prefix, create, registry = dependency_sources
    library, library_guid = create("Library", ".shaderlib.py", LIBRARY)
    name = prefix + " Stage"
    stage, stage_guid = create("Stage", ".frag.py", _fullscreen_source(name, library))
    assert Shader.reload(str(stage))
    version = registry.get_asset_version(stage_guid)
    moved_path = library.with_name(prefix + "Moved.shaderlib.py")
    if retarget:
        stage.write_text(_fullscreen_source(name, moved_path), encoding="utf-8")
        library.write_text(REVISED_SHARED_COLOR, encoding="utf-8")
    library.replace(moved_path)

    result = AssetManager.move_asset(str(library), str(moved_path))
    assert result.database_committed and not result
    assert result.error_code == AssetMutationErrorCode.RUNTIME_APPLY_FAILED
    assert registry.get_asset_version(stage_guid) == version
    database = engine.get_asset_database()
    assert database.get_guid_from_path(str(moved_path)) == library_guid
    assert not database.get_guid_from_path(str(library))
    assert ("has no export 'optional_color'" if retarget else "ShaderLib source does not exist") in result.error

    moved_path.write_text(LIBRARY, encoding="utf-8")
    stage.write_text(_fullscreen_source(name, moved_path), encoding="utf-8")
    restored = AssetManager.reimport_asset(str(moved_path))
    assert restored, restored.error
    assert registry.get_asset_version(stage_guid) == version + 1
    stage.write_text(_fullscreen_source(name, library), encoding="utf-8")
    moved_path.replace(library)
    back = AssetManager.move_asset(str(moved_path), str(library))
    assert back, back.error
    assert database.get_guid_from_path(str(library)) == library_guid


def test_same_name_vertex_save_rejects_all_pairs_before_retiring_any_program(engine, dependency_sources):
    prefix, create, _ = dependency_sources
    name = prefix + " Vertex"
    original = _vertex_source(name)
    changed = original.replace("default=0.25", "default=0.5")
    vertex, vertex_guid = create("Vertex", ".vert.py", original)
    materials = []
    for label in ("A", "B"):
        fragment_name = prefix + label
        _, fragment_guid = create(
            label, ".frag.py", _surface_source(fragment_name, signal=label == "B"),
        )
        materials.append(_linked_material(engine, create, label, vertex=(vertex_guid, name),
                                          fragment=(fragment_guid, fragment_name)))
    before = [material.native.serialize_document() for material in materials]
    retired = engine.gpu_residency_snapshot["shader_hot_reload_retirement_count"]
    vertex.write_text(changed, encoding="utf-8")
    rejected = AssetManager.reimport_asset(str(vertex))
    assert not rejected and rejected.database_committed
    assert "disagree on material field: signal" in rejected.error
    assert engine.gpu_residency_snapshot["shader_hot_reload_retirement_count"] == retired
    assert [material.native.serialize_document() for material in materials] == before

    vertex.write_text(original, encoding="utf-8")
    restored = AssetManager.reimport_asset(str(vertex))
    assert restored, restored.error
    for material in materials:
        assert engine.refresh_material_pipeline(material.native)


@pytest.mark.parametrize("recreate_at", ("same-path", "different-path"))
def test_recreated_dependency_clears_the_resolved_deleted_source_diagnostic(engine, dependency_sources, recreate_at):
    prefix, create, registry = dependency_sources
    library, library_guid = create("DeletedLibrary", ".shaderlib.py", LIBRARY)
    name = prefix + " Consumer"
    stage, stage_guid = create("Consumer", ".frag.py", _fullscreen_source(name, library))
    assert Shader.reload(str(stage))
    version = registry.get_asset_version(stage_guid)
    library.unlink()  # External delete notification follows the filesystem mutation.
    deleted = AssetManager.delete_asset(str(library))
    assert deleted and not engine.get_asset_database().get_path_from_guid(library_guid)
    assert registry.get_asset_version(stage_guid) == version
    diagnostic, = _compile_diagnostics(library)
    assert library.name in diagnostic.message

    label = "DeletedLibrary" if recreate_at == "same-path" else "ReplacementLibrary"
    replacement, replacement_guid = create(label, ".shaderlib.py", LIBRARY)
    assert replacement_guid != library_guid
    assert _compile_diagnostics(library)
    if replacement != library:
        # Typed imports follow module paths. A replacement at another path
        # becomes the dependency when the consumer publishes that import.
        assert registry.get_asset_version(stage_guid) == version
        stage.write_text(_fullscreen_source(name, replacement), encoding="utf-8")
    before_reimport = registry.get_asset_version(stage_guid)
    accepted = AssetManager.reimport_asset(str(stage))
    assert accepted, accepted.error
    assert registry.get_asset_version(stage_guid) == before_reimport + 1
    assert engine.get_asset_database().dependency_graph.has_dependency(stage_guid, replacement_guid)
    assert not _compile_diagnostics(library)


@pytest.mark.parametrize("consumer_count", (1, 2))
def test_repaired_roots_retire_only_their_resolved_dependency_diagnostic(engine, dependency_sources, consumer_count):
    prefix, create, registry = dependency_sources
    library, _ = create("Library", ".shaderlib.py", LIBRARY)
    stages = []
    for index in range(consumer_count):
        name = prefix + str(index)
        root, guid = create(str(index), ".frag.py", _fullscreen_source(name, library))
        assert Shader.reload(str(root))
        stages.append((root, guid, name))
    versions = [registry.get_asset_version(guid) for _, guid, _ in stages]
    library.write_text(REVISED_SHARED_COLOR, encoding="utf-8")
    rejected = AssetManager.reimport_asset(str(library))
    assert not rejected and "has no export 'optional_color'" in rejected.error
    diagnostic, = _compile_diagnostics(library)
    assert Path(diagnostic.source_file) in {root for root, _, _ in stages}
    assert diagnostic.source_line == 2

    for index, (root, guid, name) in enumerate(stages):
        root.write_text(_fullscreen_source(name), encoding="utf-8")
        accepted = AssetManager.reimport_asset(str(root))
        assert accepted, accepted.error
        assert registry.get_asset_version(guid) == versions[index] + 1
        assert bool(_compile_diagnostics(library)) == (index + 1 < consumer_count)


def test_invalid_library_retains_its_diagnostic_after_its_consumer_detaches(dependency_sources):
    prefix, create, registry = dependency_sources
    library, _ = create("Library", ".shaderlib.py", LIBRARY)
    root, guid = create("Consumer", ".frag.py", _fullscreen_source(prefix, library))
    assert Shader.reload(str(root))
    version = registry.get_asset_version(guid)

    invalid_library = REVISED_SHARED_COLOR.replace("1.0)", "1.0, 2.0)")
    library.write_text(invalid_library, encoding="utf-8")
    rejected = AssetManager.reimport_asset(str(library))
    assert not rejected and rejected.database_committed
    assert rejected.error_code == AssetMutationErrorCode.RUNTIME_APPLY_FAILED
    diagnostic, = _compile_diagnostics(library)
    assert Path(diagnostic.source_file) == library
    assert diagnostic.source_line == 4
    assert registry.get_asset_version(guid) == version

    root.write_text(_fullscreen_source(prefix), encoding="utf-8")
    accepted = AssetManager.reimport_asset(str(root))
    assert accepted, accepted.error
    assert registry.get_asset_version(guid) == version + 1
    assert _compile_diagnostics(library)

    library.write_text(LIBRARY, encoding="utf-8")
    accepted = AssetManager.reimport_asset(str(library))
    assert accepted, accepted.error
    assert not _compile_diagnostics(library)
