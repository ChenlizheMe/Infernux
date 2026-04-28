"""Shared helpers for Infernux MCP tools."""

from __future__ import annotations

import os
from typing import Any

from Infernux.mcp.threading import MainThreadCommandQueue


def ok(data: Any = None) -> dict[str, Any]:
    return {"ok": True, "data": data if data is not None else {}}


def fail(code: str, message: str, *, hint: str = "") -> dict[str, Any]:
    error = {"code": code, "message": message}
    if hint:
        error["hint"] = hint
    return {"ok": False, "error": error}


def main_thread(name: str, fn, *, timeout_ms: int = 30000) -> dict[str, Any]:
    try:
        return ok(MainThreadCommandQueue.instance().run_sync(name, fn, timeout_ms=timeout_ms))
    except TimeoutError as exc:
        return fail("error.timeout", str(exc))
    except ValueError as exc:
        return fail("error.invalid_argument", str(exc))
    except FileExistsError as exc:
        return fail("error.exists", str(exc))
    except FileNotFoundError as exc:
        return fail("error.not_found", str(exc))
    except Exception as exc:
        return fail("error.internal", str(exc))


def get_asset_database():
    try:
        from Infernux.lib import AssetRegistry
        registry = AssetRegistry.instance()
        if registry:
            return registry.get_asset_database()
    except Exception:
        return None
    return None


def project_assets_dir(project_path: str) -> str:
    assets = os.path.join(os.path.abspath(project_path), "Assets")
    os.makedirs(assets, exist_ok=True)
    return assets


def resolve_project_dir(project_path: str, directory: str | None) -> str:
    root = os.path.abspath(project_path)
    if not directory:
        return project_assets_dir(root)
    raw = os.path.abspath(directory if os.path.isabs(directory) else os.path.join(root, directory))
    if os.path.commonpath([root, raw]) != root:
        raise ValueError("Target directory must stay inside the project.")
    os.makedirs(raw, exist_ok=True)
    return raw


def resolve_project_path(project_path: str, path: str | None, *, default: str = "") -> str:
    """Resolve a project-relative or absolute path and keep it inside the project."""
    root = os.path.abspath(project_path)
    candidate = path or default
    if not candidate:
        candidate = root
    raw = os.path.abspath(candidate if os.path.isabs(candidate) else os.path.join(root, candidate))
    if os.path.commonpath([root, raw]) != root:
        raise ValueError("Path must stay inside the project.")
    return raw


def resolve_asset_path(project_path: str, path: str | None, *, default_name: str = "") -> str:
    """Resolve a path under the project Assets directory."""
    root = os.path.abspath(project_path)
    assets = project_assets_dir(root)
    candidate = path or default_name
    if not candidate:
        return assets
    raw = os.path.abspath(candidate if os.path.isabs(candidate) else os.path.join(root, candidate))
    if os.path.commonpath([assets, raw]) != assets:
        raise ValueError("Asset path must stay inside Assets/.")
    return raw


def notify_asset_changed(path: str, action: str = "modified") -> None:
    """Best-effort AssetDatabase notification for external MCP file writes."""
    adb = get_asset_database()
    if not adb:
        return
    method_names = {
        "created": ("on_asset_created", "import_asset"),
        "modified": ("on_asset_modified", "import_asset"),
        "deleted": ("on_asset_deleted",),
    }.get(action, ("on_asset_modified",))
    for method_name in method_names:
        method = getattr(adb, method_name, None)
        if callable(method):
            try:
                method(path)
                return
            except Exception:
                pass


def serialize_vector(value) -> Any:
    if value is None:
        return None
    if all(hasattr(value, attr) for attr in ("x", "y", "z")):
        return [float(value.x), float(value.y), float(value.z)]
    if all(hasattr(value, attr) for attr in ("x", "y")):
        return [float(value.x), float(value.y)]
    return value


def coerce_vector3(value):
    from Infernux.lib import Vector3
    if isinstance(value, dict):
        return Vector3(float(value.get("x", 0.0)), float(value.get("y", 0.0)), float(value.get("z", 0.0)))
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        return Vector3(float(value[0]), float(value[1]), float(value[2]))
    raise ValueError("Expected Vector3 as [x, y, z] or {x, y, z}.")


def serialize_value(value: Any) -> Any:
    """Convert editor/Python/C++ values to JSON-friendly data."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    vector = serialize_vector(value)
    if vector is not value:
        return vector
    if hasattr(value, "name") and hasattr(value, "value") and type(value).__name__ != "GameObject":
        try:
            return {"name": str(value.name), "value": int(value.value)}
        except Exception:
            return str(value)
    if hasattr(value, "id") and hasattr(value, "name"):
        try:
            return {"id": int(value.id), "name": str(value.name), "type": type(value).__name__}
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(k): serialize_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialize_value(v) for v in value]
    return str(value)


def serialize_component(comp) -> dict[str, Any]:
    type_name = str(getattr(comp, "type_name", type(comp).__name__))
    data: dict[str, Any] = {
        "type": type_name,
        "python": bool(hasattr(comp, "_script_guid")),
    }
    component_id = getattr(comp, "component_id", None)
    if component_id:
        data["component_id"] = int(component_id)
    script_guid = getattr(comp, "_script_guid", "")
    if script_guid:
        data["script_guid"] = script_guid
    return data


def find_game_object(object_id: int):
    from Infernux.lib import SceneManager
    scene = SceneManager.instance().get_active_scene()
    if not scene:
        raise RuntimeError("No active scene.")
    obj = scene.find_by_id(int(object_id))
    if obj is None:
        raise FileNotFoundError(f"GameObject {object_id} was not found.")
    return obj
