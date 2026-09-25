"""Audit the source layout of the Infernux041Labv2 delivery project.

This command intentionally audits only facts that can be read from the project
directory: scene files, GUID sidecars, scripts, materials and BuildSettings.
It does not infer that an Editor, Player, Android package or Web build works
from those files.  Runtime and platform acceptance must be supplied by their
own evidence reports.

Typical use::

    python scripts/acceptance/labv2_delivery_audit.py \
        --project D:/Users/Chenlizhe/Desktop/Infernux041Labv2 \
        --output out/acceptance/labv2-source-audit.json

Use ``--require-build-scenes`` when the project is being prepared for a Player
that must include every documented demo scene.  The current Labv2 project is
allowed to keep experimental scenes outside BuildSettings while authoring is
still in progress, so that check is opt-in.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


SCENE_SPECS: tuple[dict[str, str], ...] = (
    {
        "id": "demo_hub",
        "path": "Assets/Scenes/00_DemoHub.scene",
        "entry": "DemoHubController",
        "purpose": "common navigation and parameter reset",
    },
    {
        "id": "xpbd_jelly",
        "path": "Assets/Scenes/01_XPBD_Jelly.scene",
        "entry": "GPUJelly",
        "purpose": "resident GPU XPBD jelly",
    },
    {
        "id": "level_catalog",
        "path": "Assets/Scenes/02_LevelCatalog.scene",
        "entry": "LevelAuthoringData",
        "purpose": "data asset and level catalogue authoring",
    },
    {
        "id": "published_handles",
        "path": "Assets/Scenes/03_PublishedClassHandles.scene",
        "entry": "PrefabPropertyLab",
        "purpose": "published script handles and property queries",
    },
    {
        "id": "model_selection",
        "path": "Assets/Scenes/04_ModelSelection.scene",
        "entry": "ModelAuthoringProbe",
        "purpose": "model hierarchy, mesh and material authoring",
    },
    {
        "id": "imported_animation",
        "path": "Assets/Scenes/05_ImportedAnimation.scene",
        "entry": "AnimationImportTools",
        "purpose": "imported animation authoring",
    },
    {
        "id": "prefab_variants",
        "path": "Assets/Scenes/06_PrefabVariants.scene",
        "entry": "VariantPlayerProbe",
        "purpose": "variant authoring and Player consumption",
    },
    {
        "id": "user_material",
        "path": "Assets/Scenes/09_UserMaterialLesson.scene",
        "entry": "",
        "purpose": "material Inspector and Player lesson",
    },
    {
        "id": "current_jit",
        "path": "Assets/Scenes/12_CurrentJit.scene",
        "entry": "CpuJitDemo",
        "purpose": "public CPU JIT contract",
    },
    {
        "id": "blend_sync",
        "path": "Assets/Scenes/13_A16EditorSync.scene",
        "entry": "",
        "purpose": "Blender source synchronization",
    },
    {
        "id": "world_text",
        "path": "Assets/Scenes/14_WorldTextClarity.scene",
        "entry": "",
        "purpose": "world text readability",
    },
    {
        "id": "world_ui_camera",
        "path": "Assets/Scenes/15_WorldUICameraPolicies.scene",
        "entry": "",
        "purpose": "world UI camera and occlusion policies",
    },
    {
        "id": "custom_ui_shader",
        "path": "Assets/Acceptance/A06UIShader/16_A06CustomUIShader.scene",
        "entry": "",
        "purpose": "Screen and World UI custom shader acceptance",
    },
)

REQUIRED_SCRIPTS = (
    "Assets/Scripts/DemoHubController.py",
    "Assets/Scripts/GPUJelly.py",
    "Assets/Scripts/CpuJitDemo.py",
)
REQUIRED_AUTHORING_FILES = (
    "Assets/Materials/XPBDJelly.mat",
    "Assets/Models/A16EditorAcceptance.blend",
    "Assets/Models/AAA_Embedded_PBR.gltf",
)
GUID_RE = re.compile(r"\"guid\"\s*:\s*\{[^{}]*\"value\"\s*:\s*\"([0-9a-fA-F]{32})\"")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _guid_from_meta(path: Path) -> str:
    payload = path.read_text(encoding="utf-8-sig", errors="strict")
    match = GUID_RE.search(payload)
    if match is None:
        raise RuntimeError(f"scene meta has no GUID: {path}")
    return match.group(1).lower()


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def audit_labv2(project: Path, *, require_build_scenes: bool = False) -> dict[str, Any]:
    """Return a deterministic source-layout report for a Labv2 project."""

    root = project.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Labv2 project does not exist: {root}")

    errors: list[str] = []
    warnings: list[str] = []
    required_assets: list[dict[str, Any]] = []

    for relative in (*REQUIRED_SCRIPTS, *REQUIRED_AUTHORING_FILES):
        candidate = root / relative
        required_assets.append(
            {
                "path": relative,
                "present": candidate.is_file(),
                "size": candidate.stat().st_size if candidate.is_file() else 0,
            }
        )
        if not candidate.is_file():
            errors.append(f"missing required project file: {relative}")

    build_settings_path = root / "ProjectSettings" / "BuildSettings.json"
    build_settings: dict[str, Any] = {}
    build_scene_guids: set[str] = set()
    if not build_settings_path.is_file():
        errors.append("missing ProjectSettings/BuildSettings.json")
    else:
        try:
            build_settings = _read_json(build_settings_path)
            raw_guids = build_settings.get("scene_guids", [])
            if not isinstance(raw_guids, list) or any(not isinstance(item, str) for item in raw_guids):
                errors.append("BuildSettings.scene_guids must be a list of strings")
            else:
                build_scene_guids = {item.casefold() for item in raw_guids}
        except RuntimeError as exc:
            errors.append(str(exc))

    scenes: list[dict[str, Any]] = []
    for spec in SCENE_SPECS:
        scene_path = root / spec["path"]
        meta_path = scene_path.with_name(scene_path.name + ".meta")
        record: dict[str, Any] = {
            "id": spec["id"],
            "path": spec["path"],
            "entry": spec["entry"],
            "purpose": spec["purpose"],
            "source_present": scene_path.is_file(),
            "meta_present": meta_path.is_file(),
            "guid": "",
            "build_enabled": False,
        }
        if not scene_path.is_file():
            errors.append(f"missing documented scene: {spec['path']}")
        if not meta_path.is_file():
            errors.append(f"missing documented scene meta: {spec['path']}.meta")
        if meta_path.is_file():
            try:
                guid = _guid_from_meta(meta_path)
                record["guid"] = guid
                record["build_enabled"] = guid in build_scene_guids
            except (OSError, UnicodeError, RuntimeError) as exc:
                errors.append(str(exc))
        if record["source_present"] and record["meta_present"] and not record["build_enabled"]:
            warnings.append(f"scene is authored but not in BuildSettings.scene_guids: {spec['path']}")
        scenes.append(record)

    if require_build_scenes:
        for record in scenes:
            if not record["build_enabled"]:
                errors.append(
                    "required Player scene is not in BuildSettings.scene_guids: "
                    f"{record['path']}"
                )

    # A source audit must never silently turn runtime/platform claims into a pass.
    report: dict[str, Any] = {
        "$schema": "infernux.labv2_source_delivery",
        "project": str(root),
        "status": "passed" if not errors else "incomplete",
        "errors": sorted(set(errors)),
        "warnings": sorted(set(warnings)),
        "runtime_acceptance": "not_claimed",
        "platform_acceptance": "not_claimed",
        "commercial_asset_review": "not_claimed",
        "build_settings": {
            "path": _relative(build_settings_path, root),
            "game_name": build_settings.get("game_name", ""),
            "scene_guids": sorted(build_scene_guids),
            "scene_count": len(build_scene_guids),
        },
        "required_assets": required_assets,
        "scenes": scenes,
    }
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--require-build-scenes",
        action="store_true",
        help="fail when any documented scene is absent from BuildSettings.scene_guids",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="return failure when the source report has warnings as well as errors",
    )
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    report = audit_labv2(
        arguments.project,
        require_build_scenes=bool(arguments.require_build_scenes),
    )
    if arguments.output is not None:
        output = arguments.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        print(output)
    else:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if report["errors"] or (arguments.strict and report["warnings"]):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
