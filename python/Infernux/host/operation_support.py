"""Shared construction and editor-authority helpers for engine operations."""

from __future__ import annotations

import copy
import os
from collections.abc import Callable, Mapping, MutableMapping, MutableSequence
from typing import Any

from Infernux.engine.path_utils import resolved_path

from .commands import MainThreadCommandQueue
from .editor import EditorAutomationHost
from .operations import Operation, OperationError, OperationKind, OperationSchema


OWNER = "infernux/engine"

_ANY = {}
_STRING = {"type": "string"}
_INTEGER = {"type": "integer"}
_BOOLEAN = {"type": "boolean"}
_ARRAY = {"type": "array"}
_OBJECT = {"type": "object"}

_ENGINE_OUTPUTS: dict[str, tuple[dict[str, object], tuple[str, ...]]] = {
    "infernux.scene.hierarchy.get": (
        {"name": _STRING, "structure_version": _INTEGER, "objects": _ARRAY},
        ("name", "structure_version", "objects"),
    ),
    "infernux.scene.loaded.get": (
        {"active_world_id": _INTEGER, "scenes": _ARRAY},
        ("active_world_id", "scenes"),
    ),
    "infernux.scene.active.set": (
        {"active_world_id": _INTEGER, "scenes": _ARRAY},
        ("active_world_id", "scenes"),
    ),
    "infernux.scene.object.kinds": ({"kinds": _ARRAY}, ("kinds",)),
    "infernux.scene.object.create": (
        {"id": _INTEGER, "name": _STRING, "kind": _STRING, "parent_id": _INTEGER, "selected": _BOOLEAN, "components": _ARRAY},
        ("id", "name", "kind", "parent_id", "selected", "components"),
    ),
    "infernux.scene.object.delete": ({"deleted": _ARRAY}, ("deleted",)),
    "infernux.scene.object.property.set": (
        {"object_id": _INTEGER, "property": _STRING, "value": _ANY},
        ("object_id", "property", "value"),
    ),
    "infernux.scene.object.transform.set": (
        {"object_id": _INTEGER, "transform": _OBJECT},
        ("object_id", "transform"),
    ),
    "infernux.scene.object.parent.set": (
        {"object_ids": _ARRAY, "parent_id": _INTEGER},
        ("object_ids", "parent_id"),
    ),
    "infernux.scene.component.add": (
        {"object_id": _INTEGER, "component": _OBJECT},
        ("object_id", "component"),
    ),
    "infernux.scene.component.remove": (
        {"object_id": _INTEGER, "component_id": _INTEGER, "removed": _BOOLEAN},
        ("object_id", "component_id", "removed"),
    ),
    "infernux.scene.component.schema": (
        {"object_id": _INTEGER, "component_id": _INTEGER, "component_type": _STRING, "python_type": _STRING, "fields": _ARRAY},
        ("object_id", "component_id", "component_type", "python_type", "fields"),
    ),
    "infernux.scene.component.property.set": (
        {"object_id": _INTEGER, "component": _OBJECT},
        ("object_id", "component"),
    ),
    "infernux.scene.mesh.assign": (
        {"object_id": _INTEGER, "component": _OBJECT},
        ("object_id", "component"),
    ),
    "infernux.scene.open": (
        {"asset_guid": _STRING, "path": _STRING, "scheduled": _BOOLEAN},
        ("asset_guid", "path", "scheduled"),
    ),
    "infernux.scene.additive.load": (
        {"asset_guid": _STRING, "path": _STRING, "scene": _OBJECT, "active_world_id": _INTEGER, "scenes": _ARRAY},
        ("asset_guid", "path", "scene", "active_world_id", "scenes"),
    ),
    "infernux.scene.save": (
        {"saved": _BOOLEAN, "path": _STRING},
        ("saved", "path"),
    ),
    "infernux.scene.reload": (
        {"scheduled": _BOOLEAN, "path": _STRING, "discarded_changes": _BOOLEAN},
        ("scheduled", "path", "discarded_changes"),
    ),
    "infernux.asset.list": (
        {"assets": _ARRAY, "returned": _INTEGER, "root": _STRING, "catalog_count": _INTEGER, "global_catalog_count": _INTEGER},
        ("assets", "returned", "root", "catalog_count", "global_catalog_count"),
    ),
    "infernux.asset.inspect": ({"asset": _OBJECT}, ("asset",)),
    "infernux.asset.text.read": (
        {"asset": _OBJECT, "content": _STRING},
        ("asset", "content"),
    ),
    "infernux.asset.text.set": (
        {"asset": _OBJECT, "content": _STRING},
        ("asset", "content"),
    ),
    "infernux.asset.create.kinds": ({"kinds": _ARRAY}, ("kinds",)),
    "infernux.asset.create": (
        {"asset": _OBJECT, "already_exists": _BOOLEAN},
        ("asset", "already_exists"),
    ),
    "infernux.asset.mesh.save-copy": (
        {"asset": _OBJECT, "source_guid": _STRING},
        ("asset", "source_guid"),
    ),
    "infernux.asset.delete": ({"deleted": _ARRAY}, ("deleted",)),
    "infernux.asset.move": ({"asset": _OBJECT}, ("asset",)),
    "infernux.asset.refresh": (
        {"asset_count": _INTEGER, "imported": _INTEGER, "reused": _INTEGER, "scanned": _INTEGER},
        ("asset_count", "imported", "reused", "scanned"),
    ),
    "infernux.data_asset.inspect": (
        {"asset_guid": _STRING, "path": _STRING, "document": _OBJECT},
        ("asset_guid", "path", "document"),
    ),
    "infernux.data_asset.schema": (
        {"asset_guid": _STRING, "path": _STRING, "type_id": _STRING, "schema_version": _INTEGER, "fields": _ARRAY},
        ("asset_guid", "path", "type_id", "schema_version", "fields"),
    ),
    "infernux.data_asset.property.set": (
        {"asset_guid": _STRING, "path": _STRING, "pointer": _STRING, "document": _OBJECT},
        ("asset_guid", "path", "pointer", "document"),
    ),
}


def operation(
    operation_id: str,
    kind: OperationKind,
    summary: str,
    handler: Callable[..., Any],
    *,
    capability: str,
    input_properties: Mapping[str, object] | None = None,
    required: tuple[str, ...] = (),
    side_effects: tuple[str, ...] = (),
    reversible: bool = False,
    tags: tuple[str, ...] = (),
    owner: str = OWNER,
    thread: str = "owner",
    availability: tuple[str, ...] = ("editor",),
) -> Operation:
    output_properties, output_required = _ENGINE_OUTPUTS.get(
        operation_id, ({}, ())
    )
    schema = OperationSchema(
        id=operation_id,
        kind=kind,
        summary=summary,
        input_schema={
            "type": "object",
            "properties": dict(input_properties or {}),
            "required": list(required),
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": output_properties,
            "required": list(output_required),
            "additionalProperties": True,
        },
        errors=(
            {"code": "operation.invalid_arguments"},
            {"code": "operation.permission_denied"},
            {"code": "operation.failed"},
            {"code": "operation.invalid_result"},
            {"code": "editor.unavailable"},
            {"code": "asset.not_found"},
            {"code": "scene.object_not_found"},
        ),
        thread=thread,
        side_effects=side_effects,
        reversible=bool(reversible),
        capabilities=(capability,),
        cost={"class": "interactive", "risk": "low" if not side_effects else "medium"},
        tags=tags,
        availability=availability,
        phase=(
            "editor.read"
            if kind is OperationKind.QUERY
            else "editor.authoring"
            if kind is OperationKind.COMMAND
            else "editor.workflow"
        ),
    )
    return Operation(schema=schema, handler=handler, owner=str(owner))


def on_editor(operation_id: str, callback: Callable[[], Any]) -> Any:
    return MainThreadCommandQueue.instance().run_sync(
        operation_id,
        callback,
        timeout_ms=30000,
    )


def interaction_core():
    return EditorAutomationHost.instance().interaction_core()


def plugin_manager():
    return EditorAutomationHost.instance().plugin_manager()


def asset_database():
    return EditorAutomationHost.instance().asset_database()


def asset_path(asset_guid: str, *, suffix: str = "") -> str:
    guid = str(asset_guid or "").strip()
    if not guid:
        raise OperationError("operation.invalid_arguments", "asset_guid must not be empty")
    path = str(asset_database().get_path_from_guid(guid) or "")
    if not path or not os.path.exists(path):
        raise OperationError("asset.not_found", f"Asset GUID was not found: {guid}")
    if suffix and not path.casefold().endswith(suffix.casefold()):
        raise OperationError(
            "asset.type_mismatch",
            f"Asset {guid} is not a {suffix} resource.",
            details={"path": path},
        )
    return path


def asset_identity(path: str) -> dict[str, object]:
    database = asset_database()
    resolved = resolved_path(path)
    meta = database.get_meta_by_path(resolved)
    is_directory = os.path.isdir(resolved)
    if meta is None and not is_directory:
        raise OperationError("asset.not_found", f"Asset metadata is unavailable: {resolved}")
    return {
        "guid": str(database.get_guid_from_path(resolved) or ""),
        "path": resolved,
        "name": os.path.basename(resolved),
        "is_directory": is_directory,
        "resource_type": "folder" if is_directory else meta.get_resource_type().name.lower(),
    }


def active_scene():
    return EditorAutomationHost.instance().active_scene()


def game_object(object_id: int):
    return EditorAutomationHost.instance().scene_object(int(object_id))


def components(object_id: int) -> list[Any]:
    return list(EditorAutomationHost.instance().scene_components(int(object_id)))


def component(object_id: int, component_id: int):
    obj = game_object(object_id)
    return obj, EditorAutomationHost.instance().scene_component(
        int(object_id), int(component_id)
    )


def _component_document(
    document: Mapping[str, Any], *, include_mesh_data: bool,
) -> dict[str, object]:
    result = dict(document)
    if include_mesh_data:
        return result
    summary: dict[str, int] = {}
    if "inlineVertices" in result:
        summary["vertexCount"] = len(result.pop("inlineVertices"))
    if "inlineIndices" in result:
        summary["indexCount"] = len(result.pop("inlineIndices"))
    if summary:
        result["inlineMeshSummary"] = summary
    return result


def serializable_component(
    value: Any, *, include_mesh_data: bool = False,
) -> dict[str, object]:
    serializer = getattr(value, "serialize_document", None)
    if not callable(serializer):
        serializer = getattr(value, "_serialize_fields_document", None)
    document = serializer() if callable(serializer) else {}
    serialized = (
        _component_document(document, include_mesh_data=include_mesh_data)
        if isinstance(document, Mapping)
        else {}
    )
    result = {
        "component_id": int(getattr(value, "component_id", 0) or 0),
        "type": str(getattr(value, "type_name", "") or type(value).__name__),
        "enabled": bool(getattr(value, "enabled", True)),
        "document": serialized,
    }
    script_guid = str(getattr(value, "_script_guid", "") or "")
    script_path = str(getattr(value, "_script_path", "") or "")
    type_guid_getter = getattr(type(value), "_get_type_guid", None)
    type_guid = str(type_guid_getter() or "") if callable(type_guid_getter) else ""
    if script_guid or script_path or type_guid:
        result["python"] = {
            "script_guid": script_guid,
            "type_guid": type_guid,
            "script_path": script_path,
            "type_name": f"{type(value).__module__}.{type(value).__qualname__}",
        }
    return result


def set_json_pointer(document: Mapping[str, Any], pointer: str, value: Any) -> dict[str, Any]:
    result = copy.deepcopy(dict(document))
    text = str(pointer or "").strip()
    if not text.startswith("/"):
        raise OperationError("operation.invalid_arguments", "JSON pointer must start with '/'.")
    parts = [part.replace("~1", "/").replace("~0", "~") for part in text[1:].split("/")]
    current: Any = result
    for part in parts[:-1]:
        if isinstance(current, MutableMapping):
            if part not in current:
                raise OperationError("operation.invalid_arguments", f"JSON pointer does not exist: {text}")
            current = current[part]
        elif isinstance(current, MutableSequence):
            try:
                current = current[int(part)]
            except (IndexError, TypeError, ValueError) as exc:
                raise OperationError("operation.invalid_arguments", f"JSON pointer does not exist: {text}") from exc
        else:
            raise OperationError("operation.invalid_arguments", f"JSON pointer does not exist: {text}")
    leaf = parts[-1]
    if isinstance(current, MutableMapping):
        if leaf not in current:
            raise OperationError("operation.invalid_arguments", f"JSON pointer does not exist: {text}")
        current[leaf] = copy.deepcopy(value)
    elif isinstance(current, MutableSequence):
        try:
            current[int(leaf)] = copy.deepcopy(value)
        except (IndexError, TypeError, ValueError) as exc:
            raise OperationError("operation.invalid_arguments", f"JSON pointer does not exist: {text}") from exc
    else:
        raise OperationError("operation.invalid_arguments", f"JSON pointer does not exist: {text}")
    return result


__all__ = [
    "OWNER",
    "active_scene",
    "asset_database",
    "asset_identity",
    "asset_path",
    "component",
    "components",
    "game_object",
    "interaction_core",
    "on_editor",
    "operation",
    "plugin_manager",
    "serializable_component",
    "set_json_pointer",
]
