from __future__ import annotations

import json
from pathlib import Path

import pytest

from Infernux.engine.player_package_native import write_pack
from Infernux.engine.runtime_artifact_catalog import runtime_artifact_id
from scripts.acceptance.model_player_matrix import _data_root, audit_model_player_matrix


def _typed(value):
    return {"value": value}


def _write_package(path: Path, entries: dict[str, bytes]) -> None:
    staging = path.parent / f"{path.name}.staging"
    sources = []
    for index, (name, payload) in enumerate(entries.items()):
        source = staging / str(index)
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(payload)
        sources.append((name, source))
    write_pack(tuple(sources), path)


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    project = tmp_path / "Project"
    data = tmp_path / "Game_Data"
    (project / "Assets" / "Models").mkdir(parents=True)
    (project / "Library").mkdir()
    data.mkdir()
    model_entries = []
    runtime_records = []
    catalog_artifacts = []
    content: dict[str, bytes] = {}
    suffixes = (".blend", ".fbx", ".gltf", ".glb", ".obj")
    for index, suffix in enumerate(suffixes, 1):
        guid = f"{index:032x}"
        source = project / "Assets" / "Models" / f"Model{suffix}"
        source.write_bytes(f"model-{suffix}".encode())
        mesh_path = f"Library/Artifacts/Mesh/{guid}.inxmesh"
        skin_path = f"Library/Artifacts/SkinnedMesh/{guid}.inxskin"
        mesh_id = runtime_artifact_id("Content.inxpkg", mesh_path)
        skin_id = runtime_artifact_id("Content.inxpkg", skin_path)
        animation_guid = "a" * 32 if suffix == ".fbx" else ""
        metadata = {
            "material_slot_count": _typed(1),
            "bone_count": _typed(3 if suffix == ".fbx" else 0),
            "model_meshes": _typed(json.dumps([{
                "name": "Mesh",
                "path": ["Root", "Mesh"],
                "subresource_id": f"mesh-{index}",
            }])),
            "model_animations": _typed(json.dumps(
                [{"guid": animation_guid, "id": "take", "name": "Take"}]
                if animation_guid else []
            )),
        }
        dependencies = ["b" * 32] if suffix == ".gltf" else []
        model_entries.append({
            "guid": guid,
            "normalized_path": source.as_posix(),
            "dependencies": dependencies,
            "metadata": {"metadata": metadata},
        })
        runtime_records.append({
            "guid": guid,
            "runtime_path": f"Assets/Models/Model{suffix}",
            "metadata": {"metadata": metadata | {"file_extension": _typed(suffix)}},
            "dependencies": [],
            "runtime_artifacts": [
                {"runtime_path": mesh_path, "runtime_artifact_id": mesh_id},
                {"runtime_path": skin_path, "runtime_artifact_id": skin_id},
            ],
        })
        for runtime_path, artifact_id, logical_type in (
            (mesh_path, mesh_id, "mesh_artifact"),
            (skin_path, skin_id, "skinned_mesh_artifact"),
        ):
            content[runtime_path] = logical_type.encode()
            catalog_artifacts.append({
                "asset_guid": guid,
                "logical_type": logical_type,
                "payload_kind": "compiled_artifact",
                "runtime_artifact_id": artifact_id,
                "runtime_path": runtime_path,
            })
        if animation_guid:
            content[f"Library/Artifacts/Document/{animation_guid}.animclip3d"] = b"{}"

    texture_guid = "b" * 32
    texture_path = f"Library/Artifacts/Texture/{texture_guid}.inxtex"
    texture_id = runtime_artifact_id("Content.inxpkg", texture_path)
    runtime_records.append({
        "guid": texture_guid,
        "runtime_artifacts": [
            {"runtime_path": texture_path, "runtime_artifact_id": texture_id}
        ],
    })
    content[texture_path] = b"texture"
    catalog_artifacts.append({
        "asset_guid": texture_guid,
        "logical_type": "texture_artifact",
        "payload_kind": "compiled_artifact",
        "runtime_artifact_id": texture_id,
        "runtime_path": texture_path,
    })
    gltf_record = next(item for item in runtime_records if item.get("runtime_path", "").endswith(".gltf"))
    gltf_record["dependencies"] = [texture_id]
    animation_guid = "a" * 32
    animation_path = f"Library/Artifacts/Document/{animation_guid}.animclip3d"
    animation_id = runtime_artifact_id("Content.inxpkg", animation_path)
    runtime_records.append({
        "guid": animation_guid,
        "runtime_path": "Assets/Models/Model.fbx::subanim:take",
        "metadata": {"metadata": {"import_owner_guid": _typed(f"{2:032x}")}},
        "dependencies": [],
        "runtime_artifacts": [
            {"runtime_path": animation_path, "runtime_artifact_id": animation_id}
        ],
    })
    catalog_artifacts.append({
        "asset_guid": animation_guid,
        "logical_type": "animation_clip_3d_artifact",
        "payload_kind": "compiled_artifact",
        "runtime_artifact_id": animation_id,
        "runtime_path": animation_path,
    })
    content["Library/RuntimeAssetRecords.json"] = json.dumps(
        {"entries": runtime_records}
    ).encode()
    (project / "Library" / "AssetIndex.json").write_text(
        json.dumps({"entries": model_entries}), encoding="utf-8"
    )
    _write_package(data / "Content.inxpkg", content)
    _write_package(
        data / "AssetCatalog.inxcat",
        {"RuntimeAssetCatalog.json": json.dumps({"artifacts": catalog_artifacts}).encode()},
    )
    return project, data


def test_model_player_matrix_requires_all_formats_and_compiled_subresources(tmp_path):
    project, data = _fixture(tmp_path)

    result = audit_model_player_matrix(project, data)

    assert result["formats"] == [".blend", ".fbx", ".glb", ".gltf", ".obj"]
    assert result["mesh_subresource_count"] == 5
    assert result["animation_clip_count"] == 1
    assert result["raw_model_sources"] == []


def test_model_player_matrix_does_not_read_mutable_project_asset_index(tmp_path):
    project, data = _fixture(tmp_path)
    before = audit_model_player_matrix(project, data)
    (project / "Library" / "AssetIndex.json").write_text(
        json.dumps({"entries": [{"guid": "corrupt", "normalized_path": "Assets/Bad.txt"}]}),
        encoding="utf-8",
    )

    after = audit_model_player_matrix(project, data)

    assert after == before


def test_model_player_matrix_rejects_raw_source_in_content_package(tmp_path):
    project, data = _fixture(tmp_path)
    content = data / "Content.inxpkg"
    from Infernux.engine.player_package_native import extract_pack, read_manifest

    extracted = tmp_path / "unpacked"
    extract_pack(content, extracted)
    names = [item["path"] for item in read_manifest(content)["files"]]
    sources = [(name, extracted / name) for name in names]
    raw = tmp_path / "Leaked.blend"
    raw.write_bytes(b"BLENDER")
    sources.append(("Assets/Models/Leaked.blend", raw))
    write_pack(tuple(sources), content)

    with pytest.raises(RuntimeError, match="model authoring inputs"):
        audit_model_player_matrix(project, data)


def _write_player_product(root: Path, executable_name: str) -> Path:
    executable = root / executable_name
    executable.write_bytes(b"player")
    data = root / f"{Path(executable_name).stem}_Data"
    data.mkdir()
    (data / "Content.inxpkg").write_bytes(b"content")
    (data / "AssetCatalog.inxcat").write_bytes(b"catalog")
    return data


def test_model_player_matrix_resolves_linux_player_from_output_directory(tmp_path):
    expected = _write_player_product(tmp_path, "InfernuxGame")

    assert _data_root(tmp_path) == expected


def test_model_player_matrix_rejects_ambiguous_output_directory(tmp_path):
    _write_player_product(tmp_path, "FirstGame")
    _write_player_product(tmp_path, "SecondGame")

    with pytest.raises(RuntimeError, match="multiple packaged products"):
        _data_root(tmp_path)
