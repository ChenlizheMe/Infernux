import json

from Infernux.engine.runtime_artifact_catalog import build_catalog


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
