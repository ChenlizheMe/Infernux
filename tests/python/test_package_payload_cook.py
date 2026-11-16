"""Package shaders are cooked; opaque runtime data keeps its package layout."""

import json
from pathlib import Path

import pytest

from infernux.engine.game_builder import GameBuilder
from infernux.engine.player_package_native import extract_pack, read_entry, read_manifest
from infernux.engine.player_service_graph import PlayerRuntimeAssetCatalog
from infernux.engine.runtime_artifact_catalog import build_catalog
from test_game_builder_asset_closure import _entry, _write_asset_index


VERTEX = """import infernux.shader as shader


class PackageVertex(shader.vert):
    shader_id = "tests/package/vertex"

    def deform(self, vertex: shader.VertexData, ctx: shader.VertexContext) -> shader.VertexData:
        return vertex
"""
FRAGMENT = """import infernux.shader as shader


class PackageFragment(shader.fullscreen):
    shader_id = "tests/package/fragment"

    def main(self, ctx: shader.fragment_input) -> shader.vec4:
        return shader.vec4(1.0)
"""


@pytest.mark.parametrize("suffix", [".vert.py", ".frag.py", ".json", ".txt", ".bin"])
def test_package_payload_keeps_guid_identity_in_sealed_content(tmp_path, suffix):
    project = tmp_path / "Project"
    scene = project / "Assets/Main.scene"
    scene.parent.mkdir(parents=True)
    scene.write_text(json.dumps({"identity_format": "guid-v1", "name": "Main",
                                 "isPlaying": False, "objects": []}), encoding="utf-8")
    source = project / f"Packages/team/content/runtime/Payload{suffix}"
    source.parent.mkdir(parents=True)
    (source.parents[1] / "inx_package.json").write_text(
        '{"reference": "team/content", "name": "Content"}', encoding="utf-8"
    )
    programs = {".vert.py": VERTEX, ".frag.py": FRAGMENT}
    payload = programs.get(suffix, '{ "value": 7 }\n').encode("utf-8")
    source.write_bytes(payload)
    source.with_name(source.name + ".meta").write_text(json.dumps({
        "metadata": {"guid": {"type": "string", "value": "package-payload"}},
    }), encoding="utf-8")
    settings = project / "ProjectSettings"
    settings.mkdir()
    (settings / "BuildSettings.json").write_text(json.dumps({"scene_guids": ["scene"]}), encoding="utf-8")
    entries = [_entry("scene", "Assets/Main.scene"),
               _entry("package-payload", source.relative_to(project).as_posix())]
    for entry in entries:
        entry["metadata"] = {"metadata": {
            "guid": {"type": "string", "value": entry["guid"]},
            "file_path": {"type": "string", "value": entry["normalized_path"]},
        }}
    _write_asset_index(project, entries)
    output = tmp_path / "Player"
    builder = GameBuilder(str(project), str(output), game_name="PackageCook")
    builder._copy_game_data(str(output))
    builder._write_runtime_asset_records(str(output))
    data = output / "PackageCook_Data"
    (output / "Data").rename(data)
    shader = suffix in programs
    runtime_path = (f"Library/Artifacts/Blob/{source.relative_to(project).as_posix()}" if shader
                    else source.relative_to(project).as_posix())
    assert (data / runtime_path).read_bytes() == payload
    builder._pack_content_archive(str(output))
    package = data / "Content.inxpkg"
    assert read_entry(package, runtime_path) == payload
    records = json.loads(read_entry(package, "Library/RuntimeAssetRecords.json"))
    record = next(item for item in records["entries"] if item["guid"] == "package-payload")
    assert record["runtime_artifacts"][0]["runtime_path"] == runtime_path
    packed_paths = {item["path"] for item in read_manifest(package)["files"]}
    if shader:
        assert source.relative_to(project).as_posix() not in packed_paths
    else:
        assert source.relative_to(project).as_posix() in packed_paths
        catalog = build_catalog([
            {"package": "PackageCook_Data/Content.inxpkg", "runtime_path": runtime_path,
             "bytes": len(payload), "payload": payload,
             "asset_binding": builder._runtime_asset_identity_bindings[runtime_path]},
        ], player_host={"executable": "PackageCook"}, package_records=[])
        relocated = tmp_path / "Relocated Player/Data"
        extract_pack(package, relocated)
        runtime = PlayerRuntimeAssetCatalog.from_documents(
            str(relocated), catalog, {"entries": [record]},
        )
        file = Path(runtime.resolve_package(runtime_path))
        directory = Path(runtime.resolve_package(
            source.parent.relative_to(project).as_posix(), allow_directory=True,
        ))
        assert file.parent == directory
        assert (directory / source.name).read_bytes() == payload
        assert runtime.resolve_guid("package-payload") == str(file)
    assert source.read_bytes() == payload


@pytest.mark.parametrize("source_root", ["Assets/Shaders", "Packages/team/content/runtime"])
def test_relocated_player_compiles_nested_relative_shader_libraries(tmp_path, source_root):
    from infernux.lib import _Infernux as native

    project = tmp_path / "Project"
    programs = {
        f"{source_root}/surface.frag.py": FRAGMENT.replace(
            "import infernux.shader as shader",
            "import infernux.shader as shader\nfrom .palettes.color import tint",
        ).replace("shader.vec4(1.0)", "shader.vec4(tint(), 1.0)"),
        f"{source_root}/palettes/color.shaderlib.py": (
            "import infernux.shader as shader\nfrom ..constants import gain\n\n"
            "def tint() -> shader.vec3:\n    return shader.vec3(gain(), 0.8, 0.1)\n"
        ),
        f"{source_root}/constants.shaderlib.py": (
            "import infernux.shader as shader\n\n"
            "def gain() -> float:\n    return 0.25\n"
        ),
    }
    scene = project / "Assets/Main.scene"
    scene.parent.mkdir(parents=True)
    scene.write_text(json.dumps({"identity_format": "guid-v1", "name": "Main",
                                 "isPlaying": False, "objects": []}), encoding="utf-8")
    entries = [_entry("scene", "Assets/Main.scene", ("surface",))]
    for (relative, source), guid, dependencies in zip(
        programs.items(), ("surface", "color", "constants"), (("color",), ("constants",), ()),
    ):
        path = project / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
        path.with_name(path.name + ".meta").write_text(json.dumps({
            "metadata": {"guid": {"type": "string", "value": guid}},
        }), encoding="utf-8")
        entries.append(_entry(guid, relative, dependencies))
    if source_root.startswith("Packages/"):
        (project / "Packages/team/content/inx_package.json").write_text(
            '{"reference": "team/content", "name": "Content"}', encoding="utf-8",
        )
    settings = project / "ProjectSettings"
    settings.mkdir()
    (settings / "BuildSettings.json").write_text(json.dumps({"scene_guids": ["scene"]}), encoding="utf-8")
    for entry in entries:
        entry["metadata"] = {"metadata": {
            "guid": {"type": "string", "value": entry["guid"]},
            "file_path": {"type": "string", "value": entry["normalized_path"]},
        }}
    _write_asset_index(project, entries)
    output = tmp_path / "Player"
    builder = GameBuilder(str(project), str(output), game_name="LibraryCook")
    builder._copy_game_data(str(output))
    builder._write_runtime_asset_records(str(output))
    data = output / "LibraryCook_Data"
    (output / "Data").rename(data)
    builder._pack_content_archive(str(output))
    package = data / "Content.inxpkg"
    records = json.loads(read_entry(package, "Library/RuntimeAssetRecords.json"))
    by_guid = {entry["guid"]: entry for entry in records["entries"]}
    runtime_path = by_guid["surface"]["runtime_artifacts"][0]["runtime_path"]
    assert by_guid["surface"]["dependencies"] == [by_guid["color"]["primary_runtime_artifact_id"]]
    packed = {item["path"] for item in read_manifest(package)["files"]}
    assert not packed.intersection(programs)

    # Only the relocated archive is available to the real resident compiler.
    project.rename(tmp_path / "Moved authoring project")
    relocated = tmp_path / "Relocated Player" / "Data"
    extract_pack(package, relocated)
    compiled = native._check_shader_source(str(relocated / runtime_path))
    assert compiled["valid"], compiled["diagnostics"]
    assert json.loads(compiled["schema"])["info"]["shader_id"] == "tests/package/fragment"
