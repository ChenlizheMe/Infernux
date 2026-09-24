"""Audit cooked model formats and subresources in one desktop Player product."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


_SCRIPT_ROOT = Path(__file__).resolve().parent
_REPOSITORY_ROOT = _SCRIPT_ROOT.parents[1]
_PYTHON_ROOT = _REPOSITORY_ROOT / "python"
if str(_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(_PYTHON_ROOT))

from Infernux.engine.player_package_native import read_entry, read_manifest  # noqa: E402


MODEL_SUFFIXES = (".blend", ".fbx", ".gltf", ".glb", ".obj")


def _data_root(player: Path) -> Path:
    candidate = player.resolve()
    if candidate.is_dir() and candidate.name.endswith("_Data"):
        return candidate
    if candidate.is_file():
        return candidate.with_name(f"{candidate.stem}_Data")
    if candidate.is_dir():
        matches: list[Path] = []
        for executable in candidate.iterdir():
            if not executable.is_file():
                continue
            data = executable.with_name(f"{executable.stem}_Data")
            if (
                (data / "Content.inxpkg").is_file()
                and (data / "AssetCatalog.inxcat").is_file()
            ):
                matches.append(data)
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise RuntimeError(
                f"Player directory has multiple packaged products: {candidate}"
            )
    raise FileNotFoundError(f"cannot resolve Player data root from {player}")


def _metadata_values(entry: dict[str, Any]) -> dict[str, Any]:
    values = entry.get("metadata", {}).get("metadata", {})
    if not isinstance(values, dict):
        return {}
    return {
        str(name): item.get("value") if isinstance(item, dict) else item
        for name, item in values.items()
    }


def _json_value(values: dict[str, Any], name: str, default: Any) -> Any:
    value = values.get(name, default)
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"model metadata {name!r} is not valid JSON") from exc
    return value


def _load_json_entry(package: Path, name: str) -> dict[str, Any]:
    try:
        payload = read_entry(package, name)
    except Exception as exc:
        raise RuntimeError(f"Player package is missing {name}") from exc
    value = json.loads(payload.decode("utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Player package entry {name} is not an object")
    return value


def audit_model_player_matrix(project: Path | None, player: Path) -> dict[str, Any]:
    project_label = str(project.resolve()) if project is not None else ""
    data_root = _data_root(player)
    content = data_root / "Content.inxpkg"
    catalog_package = data_root / "AssetCatalog.inxcat"
    if not content.is_file() or not catalog_package.is_file():
        raise FileNotFoundError("Player is missing Content.inxpkg or AssetCatalog.inxcat")

    content_names = {str(item["path"]).replace("\\", "/") for item in read_manifest(content)["files"]}
    raw_sources = sorted(
        name for name in content_names if Path(name).suffix.casefold() in MODEL_SUFFIXES
    )
    temporary_products = sorted(
        name for name in content_names
        if name.casefold().startswith("library/modelimport/")
        or Path(name).suffix.casefold() in {".blend1", ".blend2"}
    )
    if raw_sources or temporary_products:
        raise RuntimeError(
            f"Player contains model authoring inputs: {raw_sources + temporary_products}"
        )

    runtime_records = _load_json_entry(content, "Library/RuntimeAssetRecords.json")
    records = {
        str(item.get("guid", "")): item
        for item in runtime_records.get("entries", [])
        if isinstance(item, dict) and str(item.get("guid", ""))
    }
    models: list[dict[str, Any]] = []
    model_formats: dict[str, str] = {}
    for guid, record in records.items():
        metadata = _metadata_values(record)
        suffix = str(metadata.get("file_extension", "") or "").casefold()
        if suffix not in MODEL_SUFFIXES:
            suffix = Path(str(record.get("runtime_path", ""))).suffix.casefold()
        if suffix in MODEL_SUFFIXES:
            models.append(record)
            model_formats[guid] = suffix
    formats = sorted(set(model_formats.values()))
    missing_formats = sorted(set(MODEL_SUFFIXES) - set(formats))
    if missing_formats:
        raise RuntimeError(f"cooked Player model matrix is missing formats: {missing_formats}")
    runtime_catalog = _load_json_entry(catalog_package, "RuntimeAssetCatalog.json")
    artifacts = {
        str(item.get("runtime_artifact_id", "")): item
        for item in runtime_catalog.get("artifacts", [])
        if isinstance(item, dict) and str(item.get("runtime_artifact_id", ""))
    }

    matrix: list[dict[str, Any]] = []
    mesh_subresource_ids: set[str] = set()
    animation_guids: set[str] = set()
    has_material = False
    has_texture = False
    has_skin = False
    for record in sorted(models, key=lambda item: str(item.get("runtime_path", ""))):
        guid = str(record.get("guid", ""))
        suffix = model_formats[guid]
        mesh_path = f"Library/Artifacts/Mesh/{guid}.inxmesh"
        if mesh_path not in content_names:
            raise RuntimeError(f"Player has no cooked Mesh artifact for {guid}")

        runtime_artifacts = record.get("runtime_artifacts", [])
        runtime_paths = {
            str(item.get("runtime_path", ""))
            for item in runtime_artifacts if isinstance(item, dict)
        }
        if mesh_path not in runtime_paths:
            raise RuntimeError(f"model GUID {guid} does not resolve to its cooked Mesh")
        for item in runtime_artifacts:
            artifact_id = str(item.get("runtime_artifact_id", ""))
            artifact = artifacts.get(artifact_id)
            if artifact is None or artifact.get("asset_guid") != guid:
                raise RuntimeError(f"model GUID {guid} has an invalid catalog artifact binding")
            if artifact.get("payload_kind") != "compiled_artifact":
                raise RuntimeError(f"model GUID {guid} resolves to a non-compiled payload")

        metadata = _metadata_values(record)
        meshes = _json_value(metadata, "model_meshes", [])
        animations = _json_value(metadata, "model_animations", [])
        if not isinstance(meshes, list) or not meshes:
            raise RuntimeError(f"model GUID {guid} publishes no mesh subresources")
        local_ids = [str(item.get("subresource_id", "")) for item in meshes]
        if any(not item for item in local_ids) or len(set(local_ids)) != len(local_ids):
            raise RuntimeError(f"model GUID {guid} has invalid mesh subresource identities")
        mesh_subresource_ids.update(local_ids)

        material_count = int(metadata.get("material_slot_count", 0) or 0)
        has_material = has_material or material_count > 0
        bone_count = int(metadata.get("bone_count", 0) or 0)
        skin_path = f"Library/Artifacts/SkinnedMesh/{guid}.inxskin"
        if bone_count > 0 and skin_path not in content_names:
            raise RuntimeError(f"skinned model GUID {guid} has no cooked SkinnedMesh")
        has_skin = has_skin or bone_count > 0

        owned_animations = [
            animation_guid
            for animation_guid, animation_record in records.items()
            if str(_metadata_values(animation_record).get("import_owner_guid", "")) == guid
            and any(
                str(item.get("runtime_path", "")).endswith(".animclip3d")
                for item in animation_record.get("runtime_artifacts", [])
                if isinstance(item, dict)
            )
        ]
        if isinstance(animations, list):
            owned_animations.extend(
                str(item.get("guid", "")) for item in animations
                if str(item.get("guid", ""))
            )
        owned_animations = sorted(set(owned_animations))
        for animation_guid in owned_animations:
            animation_path = f"Library/Artifacts/Document/{animation_guid}.animclip3d"
            if not animation_guid or animation_path not in content_names:
                raise RuntimeError(f"model GUID {guid} has an uncooked AnimationClip3D")
            animation_guids.add(animation_guid)

        dependency_artifacts = [
            artifacts.get(str(item), {}) for item in record.get("dependencies", [])
        ]
        has_texture = has_texture or any(
            str(artifact.get("runtime_path", "")).endswith(".inxtex")
            for artifact in dependency_artifacts
        )
        matrix.append(
            {
                "guid": guid,
                "format": suffix,
                "mesh_subresources": len(meshes),
                "material_slots": material_count,
                "bones": bone_count,
                "animations": len(owned_animations),
                "compiled_artifacts": sorted(runtime_paths),
            }
        )

    missing_capabilities = [
        name
        for name, present in (
            ("material", has_material),
            ("texture", has_texture),
            ("skinned_mesh", has_skin),
            ("animation", bool(animation_guids)),
            ("mesh_subresource", bool(mesh_subresource_ids)),
        )
        if not present
    ]
    if missing_capabilities:
        raise RuntimeError(f"Player model matrix is missing capabilities: {missing_capabilities}")

    return {
        "project_label": project_label,
        "player_data": str(data_root),
        "formats": formats,
        "models": matrix,
        "mesh_subresource_count": len(mesh_subresource_ids),
        "animation_clip_count": len(animation_guids),
        "raw_model_sources": raw_sources,
        "temporary_model_products": temporary_products,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", help="Optional report label; never read by the audit")
    parser.add_argument("--player", required=True)
    parser.add_argument("--report")
    args = parser.parse_args(argv)
    result = audit_model_player_matrix(
        Path(args.project) if args.project else None,
        Path(args.player),
    )
    output = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.report:
        report = Path(args.report).resolve()
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(output, encoding="utf-8")
    print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
