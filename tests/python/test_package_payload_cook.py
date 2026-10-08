"""Enabled package assets use the same GUID cook boundary as Assets."""

import json

import pytest

from infernux.engine.game_builder import GameBuilder
from infernux.engine.player_package_native import read_entry, read_manifest
from test_game_builder_asset_closure import _entry, _write_asset_index


@pytest.mark.parametrize("suffix", [".vert", ".frag", ".json", ".txt", ".bin"])
def test_package_payload_is_guid_addressed_in_sealed_content(tmp_path, suffix):
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
    payload = b"#version 450\nvoid main() {}\n" if suffix in (".vert", ".frag") else b'{ "value": 7 }\n'
    source.write_bytes(payload)
    source.with_suffix(suffix + ".meta").write_text(json.dumps({
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
    runtime_path = f"Library/Artifacts/Blob/package-payload{suffix}"
    assert (data / runtime_path).read_bytes() == payload
    builder._pack_content_archive(str(output))
    package = data / "Content.inxpkg"
    assert read_entry(package, runtime_path) == payload
    records = json.loads(read_entry(package, "Library/RuntimeAssetRecords.json"))
    record = next(item for item in records["entries"] if item["guid"] == "package-payload")
    assert record["runtime_artifacts"][0]["runtime_path"] == runtime_path
    assert not any(item["path"].startswith("Packages/") for item in read_manifest(package)["files"])
    assert source.read_bytes() == payload
