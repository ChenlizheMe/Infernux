"""Runtime-neutral access to ``ProjectSettings/BuildSettings.json``."""

from __future__ import annotations

import json
import os
import copy
from typing import Any, Optional

from Infernux.engine.project_context import get_project_root
from Infernux.engine.build_target import BuildTargetId
from Infernux.engine.path_utils import relative_path, resolve_project_path


BUILD_SETTINGS_FILE = "BuildSettings.json"


# This schema is consumed by both Editor authoring and the Player runtime.
# Keep the normalizer in this runtime-neutral module so a cooked Player never
# imports the editor-only interaction package merely to resolve its scene list.
BUILD_SETTINGS_DEFAULTS: dict[str, Any] = {
    "build_target": "",
    "game_name": "",
    "scene_guids": [],
    "output_dir": "",
    "icon_guid": "",
    "platform_options": {},
    "debug_mode": False,
    "lto": True,
    "splash_items": [],
}


def _json_copy(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def normalize_build_settings(value: Any, *, project_path: Optional[str] = None) -> dict[str, Any]:
    """Validate the current BuildSettings schema.

    This function intentionally has no Editor/document imports: scene loading
    and Player bootstrap use it from the runtime package directly.
    """
    if not isinstance(value, dict):
        raise TypeError("build settings must be a JSON object")
    unknown = value.keys() - BUILD_SETTINGS_DEFAULTS.keys()
    if unknown:
        raise ValueError(f"build settings contain unknown fields: {sorted(unknown)}")
    result = copy.deepcopy(BUILD_SETTINGS_DEFAULTS)
    result.update(copy.deepcopy(value))
    if not isinstance(result["scene_guids"], list) or not all(
        isinstance(item, str) and item for item in result["scene_guids"]
    ):
        raise TypeError("build settings scene_guids must contain non-empty strings")
    if not isinstance(result["splash_items"], list):
        raise TypeError("build settings splash_items must be an array")
    splash_keys = {"type", "asset_guid", "duration", "fade_in", "fade_out"}
    for index, item in enumerate(result["splash_items"]):
        if not isinstance(item, dict) or set(item) != splash_keys:
            raise TypeError(
                f"build settings splash_items[{index}] must use the current asset GUID schema"
            )
        if item["type"] not in {"image", "video"}:
            raise ValueError(f"build settings splash_items[{index}].type is invalid")
        if not isinstance(item["asset_guid"], str) or not item["asset_guid"]:
            raise TypeError(
                f"build settings splash_items[{index}].asset_guid must be a non-empty string"
            )
        for field in ("duration", "fade_in", "fade_out"):
            if isinstance(item[field], bool) or not isinstance(item[field], (int, float)):
                raise TypeError(
                    f"build settings splash_items[{index}].{field} must be numeric"
                )
            if item[field] < 0:
                raise ValueError(
                    f"build settings splash_items[{index}].{field} must not be negative"
                )
    for field in ("build_target", "game_name", "output_dir", "icon_guid"):
        if not isinstance(result[field], str):
            raise TypeError(f"build settings {field} must be a string")
    if project_path and result["output_dir"]:
        output = resolve_project_path(result["output_dir"], project_path)
        try:
            result["output_dir"] = relative_path(
                output, project_path, allow_root=True, allow_outside=True,
            )
        except ValueError:
            # A user-selected build destination on another Windows drive has
            # no relative representation. It is an explicit external I/O path.
            result["output_dir"] = output
    if result["build_target"]:
        BuildTargetId(result["build_target"])
    platform_options = result["platform_options"]
    if not isinstance(platform_options, dict):
        raise TypeError("build settings platform_options must be an object")
    for target_id, options in platform_options.items():
        BuildTargetId(target_id)
        if not isinstance(options, dict):
            raise TypeError(
                f"build settings platform_options.{target_id} must be an object"
            )
        for key, option_value in options.items():
            if not isinstance(key, str) or not key:
                raise TypeError("build option keys must be non-empty strings")
            if not isinstance(option_value, (str, int, float, bool)) or option_value is None:
                raise TypeError(
                    f"build settings platform_options.{target_id}.{key} "
                    "must be a JSON scalar"
                )
    for field in ("debug_mode", "lto"):
        if not isinstance(result[field], bool):
            raise TypeError(f"build settings {field} must be a boolean")
    return _json_copy(result)


def build_settings_path(project_path: Optional[str] = None) -> Optional[str]:
    root = project_path or get_project_root()
    if not root:
        return None
    return os.path.join(root, "ProjectSettings", BUILD_SETTINGS_FILE)


def load_build_settings(project_path: Optional[str] = None) -> dict:
    """Load the current BuildSettings document without editor UI imports."""
    path = build_settings_path(project_path)
    if not path:
        raise ValueError("Build settings require an explicit project root")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Build settings are missing: {path}")
    try:
        with open(path, "r", encoding="utf-8") as stream:
            data = json.load(stream)
    except (json.JSONDecodeError, UnicodeError, OSError) as error:
        raise ValueError(f"Build settings are unreadable: {path}: {error}") from error
    if not isinstance(data, dict):
        raise TypeError("Build settings must be a JSON object")
    return data


def load_build_settings_for_build(project_path: Optional[str] = None) -> dict:
    """Load and validate the authoritative settings document for a Player build.

    Editor/runtime discovery may use :func:`load_build_settings` while a project
    is being created.  A build is different: silently replacing a missing or
    malformed project document changes the produced Player.  Build callers
    therefore use this strict entry point and receive a precise failure.
    """

    path = build_settings_path(project_path)
    if not path:
        raise ValueError("Player build requires an explicit project root")
    data = load_build_settings(project_path)

    try:
        return normalize_build_settings(data)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Player build settings are invalid: {path}: {error}"
        ) from error
