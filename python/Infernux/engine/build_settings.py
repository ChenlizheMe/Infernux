"""Runtime-neutral access to ``ProjectSettings/BuildSettings.json``."""

from __future__ import annotations

import json
import os
import copy
from typing import Any, Optional

from Infernux.engine.project_context import get_project_root


BUILD_SETTINGS_FILE = "BuildSettings.json"


# This schema is consumed by both Editor authoring and the Player runtime.
# Keep the normalizer in this runtime-neutral module so a cooked Player never
# imports the editor-only interaction package merely to resolve its scene list.
BUILD_SETTINGS_DEFAULTS: dict[str, Any] = {
    "build_target": "",
    "android_artifact": "apk",
    "game_name": "",
    "scene_guids": [],
    "output_dir": "",
    "icon_guid": "",
    "display_mode": "fullscreen_borderless",
    "window_width": 1280,
    "window_height": 720,
    "window_resizable": True,
    "debug_mode": False,
    "lto": True,
    "splash_items": [],
}


def _json_copy(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def normalize_build_settings(value: Any) -> dict[str, Any]:
    """Validate and project the current BuildSettings schema.

    This function intentionally has no Editor/document imports: scene loading
    and Player bootstrap use it from the runtime package directly.
    """
    if not isinstance(value, dict):
        raise TypeError("build settings must be a JSON object")
    value = copy.deepcopy(value)
    result = copy.deepcopy(BUILD_SETTINGS_DEFAULTS)
    result.update(copy.deepcopy({
        key: item for key, item in value.items() if key in BUILD_SETTINGS_DEFAULTS
    }))
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
    for field in (
        "build_target", "android_artifact", "game_name", "output_dir",
        "icon_guid", "display_mode",
    ):
        if not isinstance(result[field], str):
            raise TypeError(f"build settings {field} must be a string")
    if result["display_mode"] not in {"fullscreen_borderless", "windowed"}:
        raise ValueError("build settings display_mode is invalid")
    if result["android_artifact"] not in {"apk", "aab"}:
        raise ValueError("build settings android_artifact is invalid")
    if result["build_target"]:
        from Infernux.engine.build import BuildTargetId

        BuildTargetId(result["build_target"])
    for field in ("window_width", "window_height"):
        if isinstance(result[field], bool) or not isinstance(result[field], int):
            raise TypeError(f"build settings {field} must be an integer")
        if result[field] <= 0:
            raise ValueError(f"build settings {field} must be positive")
    for field in ("window_resizable", "debug_mode", "lto"):
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
