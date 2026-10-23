import json

from infernux.engine.runtime_artifact_catalog import build_catalog


def test_runtime_catalog_uses_guid_and_ignores_path_hints_for_dependencies():
    scene_payload = json.dumps(
        {
            "material": {
                "$type": "asset_ref",
                "guid": "material-guid",
                "path_hint": "C:/Retired/Project/Assets/Materials/Bird.mat",
            },
            "retired_path_only_reference": {
                "$type": "asset_ref",
                "guid": "",
                "path_hint": "Assets/Materials/Bird.mat",
            },
        }
    ).encode("utf-8")
    material_payload = b"{}"
    entries = [
        {
            "package": "Content.inxpkg",
            "runtime_path": "Library/Artifacts/Document/scene-guid.scene",
            "bytes": len(scene_payload),
            "payload": scene_payload,
            "asset_binding": {
                "source_guid": "scene-guid",
                "source_path": "Assets/Main.scene",
                "dependencies": [],
            },
        },
        {
            "package": "Content.inxpkg",
            "runtime_path": "Library/Artifacts/Document/material-guid.mat",
            "bytes": len(material_payload),
            "payload": material_payload,
            "asset_binding": {
                "source_guid": "material-guid",
                "source_path": "Assets/Materials/Bird.mat",
                "dependencies": [],
            },
        },
    ]

    catalog = build_catalog(
        entries,
        player_host={"executable": "Game.exe"},
        package_records=[],
    )

    by_path = {
        artifact["runtime_path"]: artifact for artifact in catalog["artifacts"]
    }
    scene = by_path["Library/Artifacts/Document/scene-guid.scene"]
    material = by_path["Library/Artifacts/Document/material-guid.mat"]
    assert scene["dependencies"] == [material["runtime_artifact_id"]]
    assert scene["unresolved_dependencies"] == []


def test_runtime_type_registry_separates_type_identity_from_script_dependencies():
    script_guid = "3" * 32
    payload = json.dumps({"types": [{
        "type_guid": "1" * 32,
        "script_guid": script_guid,
        "semantic": {"type_guid": "1" * 32, "base_type_guid": "2" * 32},
    }]}).encode("utf-8")
    registry = {"package": "Content.inxpkg", "runtime_path": "Library/RuntimeTypeRegistry.json",
                "bytes": len(payload), "payload": payload}
    script = {"package": "Content.inxpkg", "runtime_path": "Library/Scripts/Authoring.pyc",
              "bytes": 8, "payload": b"compiled",
              "asset_binding": {"source_guid": script_guid,
                                "source_path": "Packages/author/plugin/Authoring.py", "dependencies": []}}
    catalog = build_catalog([registry, script], player_host={}, package_records=[])
    by_path = {artifact["runtime_path"]: artifact for artifact in catalog["artifacts"]}
    metadata = by_path[registry["runtime_path"]]
    assert metadata["dependencies"] == [by_path[script["runtime_path"]]["runtime_artifact_id"]]
    assert metadata["unresolved_dependencies"] == []

    missing_script = build_catalog([registry], player_host={}, package_records=[])["artifacts"][0]
    assert missing_script["unresolved_dependencies"] == [{"kind": "guid", "value": script_guid}]
