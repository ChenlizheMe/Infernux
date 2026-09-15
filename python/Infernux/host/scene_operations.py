"""Scene and component editing operations backed by editor transactions."""

from __future__ import annotations

from typing import Any

from .editor import EditorAutomationHost
from .operations import Operation, OperationError, OperationKind

from .operation_support import (
    active_scene,
    asset_path,
    components,
    on_editor,
    operation,
    serializable_component,
)


_VECTOR3 = {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3}


def build_scene_operations() -> tuple[Operation, ...]:
    return (
        operation(
            "infernux.scene.hierarchy.get",
            OperationKind.QUERY,
            "Read the active scene hierarchy and serialized component state; inline mesh arrays are summarized by default.",
            _hierarchy,
            capability="scene.read",
            input_properties={
                "include_mesh_data": {"type": "boolean", "default": False},
            },
            tags=("scene", "hierarchy", "component", "inspect"),
        ),
        operation(
            "infernux.scene.loaded.get",
            OperationKind.QUERY,
            "List every Scene currently resident in the editor World.",
            _loaded_scenes,
            capability="scene.read",
            tags=("scene", "loaded", "additive", "inspect"),
        ),
        operation(
            "infernux.scene.active.set",
            OperationKind.COMMAND,
            "Make one loaded Scene and its authoring document active.",
            _activate_loaded_scene,
            capability="scene.write",
            input_properties={"world_id": {"type": "integer"}},
            required=("world_id",),
            side_effects=("Changes the active authoring Scene and document.",),
            tags=("scene", "loaded", "active", "authoring"),
        ),
        operation(
            "infernux.scene.object.kinds",
            OperationKind.QUERY,
            "List object kinds accepted by the editor's hierarchy creation service.",
            _object_kinds,
            capability="scene.read",
            tags=("scene", "object", "create", "schema"),
        ),
        operation(
            "infernux.scene.object.create",
            OperationKind.COMMAND,
            "Create one GameObject through hierarchy history.",
            _create_object,
            capability="scene.write",
            input_properties={
                "kind": {"type": "string"},
                "parent_id": {"type": "integer", "default": 0},
                "name": {"type": "string", "default": ""},
            },
            required=("kind",),
            side_effects=("Creates a scene object and records an Undo entry.",),
            reversible=True,
            tags=("scene", "object", "create", "authoring"),
        ),
        operation(
            "infernux.scene.object.delete",
            OperationKind.COMMAND,
            "Delete explicit GameObjects through one hierarchy transaction.",
            _delete_objects,
            capability="scene.write",
            input_properties={"object_ids": {"type": "array", "items": {"type": "integer"}}},
            required=("object_ids",),
            side_effects=("Deletes scene objects and records an Undo entry.",),
            reversible=True,
            tags=("scene", "object", "delete", "authoring"),
        ),
        operation(
            "infernux.scene.object.property.set",
            OperationKind.COMMAND,
            "Set one supported GameObject property through serialized-property history.",
            _set_object_property,
            capability="scene.write",
            input_properties={
                "object_id": {"type": "integer"},
                "property": {"type": "string", "enum": ["name", "active", "tag", "layer"]},
                "value": {},
            },
            required=("object_id", "property", "value"),
            side_effects=("Changes a GameObject property and records an Undo entry.",),
            reversible=True,
            tags=("scene", "object", "property", "authoring"),
        ),
        operation(
            "infernux.scene.object.transform.set",
            OperationKind.COMMAND,
            "Set the complete local transform of one GameObject.",
            _set_transform,
            capability="scene.write",
            input_properties={
                "object_id": {"type": "integer"},
                "position": _VECTOR3,
                "rotation": _VECTOR3,
                "scale": _VECTOR3,
            },
            required=("object_id", "position", "rotation", "scale"),
            side_effects=("Changes a Transform and records an Undo entry.",),
            reversible=True,
            tags=("scene", "object", "transform", "move", "authoring"),
        ),
        operation(
            "infernux.scene.object.parent.set",
            OperationKind.COMMAND,
            "Move GameObjects under a parent or to the scene root.",
            _set_parent,
            capability="scene.write",
            input_properties={
                "object_ids": {"type": "array", "items": {"type": "integer"}},
                "parent_id": {"type": "integer", "default": 0},
            },
            required=("object_ids",),
            side_effects=("Changes the hierarchy and records an Undo entry.",),
            reversible=True,
            tags=("scene", "hierarchy", "parent", "move", "authoring"),
        ),
        operation(
            "infernux.scene.component.add",
            OperationKind.COMMAND,
            "Attach a registered native or Python component to a GameObject.",
            _add_component,
            capability="scene.write",
            input_properties={
                "object_id": {"type": "integer"},
                "component_type": {"type": "string"},
                "script_guid": {"type": "string", "default": ""},
            },
            required=("object_id", "component_type"),
            side_effects=("Attaches a component and records an Undo entry.",),
            reversible=True,
            tags=("scene", "component", "add", "authoring"),
        ),
        operation(
            "infernux.scene.component.remove",
            OperationKind.COMMAND,
            "Remove one component through component history.",
            _remove_component,
            capability="scene.write",
            input_properties={
                "object_id": {"type": "integer"},
                "component_id": {"type": "integer"},
            },
            required=("object_id", "component_id"),
            side_effects=("Removes a component and records an Undo entry.",),
            reversible=True,
            tags=("scene", "component", "remove", "authoring"),
        ),
        operation(
            "infernux.scene.component.schema",
            OperationKind.QUERY,
            "Describe the authoritative writable fields of one component.",
            _component_schema,
            capability="scene.read",
            input_properties={
                "object_id": {"type": "integer"},
                "component_id": {"type": "integer"},
            },
            required=("object_id", "component_id"),
            tags=("scene", "component", "schema", "property"),
        ),
        operation(
            "infernux.scene.component.property.set",
            OperationKind.COMMAND,
            "Set one serialized component field through the shared component service.",
            _set_component_property,
            capability="scene.write",
            input_properties={
                "object_id": {"type": "integer"},
                "component_id": {"type": "integer"},
                "field": {"type": "string"},
                "value": {},
            },
            required=("object_id", "component_id", "field", "value"),
            side_effects=("Changes a component field and records an Undo entry.",),
            reversible=True,
            tags=("scene", "component", "property", "authoring"),
        ),
        operation(
            "infernux.scene.mesh.assign",
            OperationKind.COMMAND,
            "Assign a registered Mesh asset, including mesh source state, through component history.",
            _assign_mesh,
            capability="scene.write",
            input_properties={
                "object_id": {"type": "integer"},
                "component_id": {"type": "integer"},
                "asset_guid": {"type": "string"},
            },
            required=("object_id", "component_id", "asset_guid"),
            side_effects=("Changes the renderer mesh source and records an Undo entry.",),
            reversible=True,
            tags=("scene", "mesh", "asset", "authoring"),
        ),
        operation(
            "infernux.scene.open",
            OperationKind.COMMAND,
            "Schedule a scene asset load by GUID. The returned scheduled flag confirms that loading was queued.",
            _open_scene,
            capability="scene.write",
            input_properties={"asset_guid": {"type": "string"}},
            required=("asset_guid",),
            side_effects=("Replaces the active scene document.",),
            tags=("scene", "document", "open", "asset"),
        ),
        operation(
            "infernux.scene.additive.load",
            OperationKind.COMMAND,
            "Load a scene asset additively into the current editor World.",
            _load_additive_scene,
            capability="scene.write",
            input_properties={"asset_guid": {"type": "string"}},
            required=("asset_guid",),
            side_effects=("Adds one resident Scene without replacing existing Scenes.",),
            tags=("scene", "document", "load", "additive", "asset"),
        ),
        operation(
            "infernux.scene.save",
            OperationKind.COMMAND,
            "Save the active scene through its registered document controller.",
            _save_scene,
            capability="scene.write",
            side_effects=("Durably writes the active scene asset.",),
            tags=("scene", "document", "save"),
        ),
        operation(
            "infernux.scene.reload",
            OperationKind.COMMAND,
            "Reload the active scene from disk, optionally discarding unsaved edits.",
            _reload_scene,
            capability="scene.write",
            input_properties={
                "discard_changes": {"type": "boolean", "default": False},
            },
            side_effects=("Replaces the active scene with its persisted document.",),
            tags=("scene", "document", "reload", "discard"),
        ),
    )


def _hierarchy(include_mesh_data: bool = False) -> dict[str, object]:
    def read() -> dict[str, object]:
        scene = active_scene()

        def serialize(obj) -> dict[str, object]:
            transform = obj.get_transform()
            values = components(obj.id)
            return {
                "id": int(obj.id),
                "name": str(obj.name),
                "active": bool(getattr(obj, "active", True)),
                "tag": str(getattr(obj, "tag", "")),
                "layer": int(getattr(obj, "layer", 0)),
                "transform": {
                    "position": list(transform.local_position),
                    "rotation": list(transform.local_euler_angles),
                    "scale": list(transform.local_scale),
                },
                "components": [
                    serializable_component(
                        value, include_mesh_data=bool(include_mesh_data),
                    )
                    for value in values
                ],
                "children": [serialize(child) for child in obj.get_children()],
            }

        return {
            "name": str(scene.name),
            "structure_version": int(getattr(scene, "structure_version", 0)),
            "objects": [serialize(obj) for obj in scene.get_root_objects()],
        }

    return on_editor("infernux.scene.hierarchy.get", read)


def _loaded_scene_snapshot() -> dict[str, object]:
    return EditorAutomationHost.instance().loaded_scenes()


def _loaded_scenes() -> dict[str, object]:
    return on_editor("infernux.scene.loaded.get", _loaded_scene_snapshot)


def _activate_loaded_scene(world_id: int) -> dict[str, object]:
    return on_editor(
        "infernux.scene.active.set",
        lambda: EditorAutomationHost.instance().activate_loaded_scene(int(world_id)),
    )


def _object_kinds() -> dict[str, object]:
    def read():
        return {"kinds": EditorAutomationHost.instance().hierarchy_create_kinds()}

    return on_editor("infernux.scene.object.kinds", read)


def _create_object(kind: str, parent_id: int = 0, name: str = "") -> dict[str, object]:
    def create():
        return EditorAutomationHost.instance().create_scene_object(
            kind, int(parent_id), str(name or "")
        )

    return on_editor("infernux.scene.object.create", create)


def _delete_objects(object_ids: list[int]) -> dict[str, object]:
    def delete():
        ids = [int(value) for value in object_ids]
        return {"deleted": EditorAutomationHost.instance().delete_scene_objects(ids)}

    return on_editor("infernux.scene.object.delete", delete)


def _set_object_property(object_id: int, property: str, value: Any) -> dict[str, object]:
    def edit():
        EditorAutomationHost.instance().set_scene_object_property(object_id, property, value)
        return {"object_id": int(object_id), "property": property, "value": value}

    return on_editor("infernux.scene.object.property.set", edit)


def _set_transform(
    object_id: int,
    position: list[float],
    rotation: list[float],
    scale: list[float],
) -> dict[str, object]:
    values = {"position": position, "rotation": rotation, "scale": scale}

    def edit():
        EditorAutomationHost.instance().set_scene_transforms(object_id, values)
        return {"object_id": int(object_id), "transform": values}

    return on_editor("infernux.scene.object.transform.set", edit)


def _set_parent(object_ids: list[int], parent_id: int = 0) -> dict[str, object]:
    def edit():
        EditorAutomationHost.instance().set_scene_parent(object_ids, int(parent_id))
        return {"object_ids": [int(value) for value in object_ids], "parent_id": int(parent_id)}

    return on_editor("infernux.scene.object.parent.set", edit)


def _add_component(
    object_id: int, component_type: str, script_guid: str = ""
) -> dict[str, object]:
    def edit():
        value = EditorAutomationHost.instance().add_scene_component(
            object_id, component_type, script_guid=script_guid
        )
        return {"object_id": int(object_id), "component": serializable_component(value)}

    return on_editor("infernux.scene.component.add", edit)


def _remove_component(object_id: int, component_id: int) -> dict[str, object]:
    def edit():
        EditorAutomationHost.instance().remove_scene_component(object_id, component_id)
        return {
            "object_id": int(object_id),
            "component_id": int(component_id),
            "removed": True,
        }

    return on_editor("infernux.scene.component.remove", edit)


def _component_schema(object_id: int, component_id: int) -> dict[str, object]:
    return on_editor(
        "infernux.scene.component.schema",
        lambda: EditorAutomationHost.instance().scene_component_schema(
            object_id, component_id
        ),
    )


def _set_component_property(
    object_id: int,
    component_id: int,
    field: str,
    value: Any,
) -> dict[str, object]:
    def edit():
        target = EditorAutomationHost.instance().set_scene_component_field(
            object_id, component_id, field, value
        )
        return {"object_id": int(object_id), "component": serializable_component(target)}

    return on_editor("infernux.scene.component.property.set", edit)


def _assign_mesh(object_id: int, component_id: int, asset_guid: str) -> dict[str, object]:
    def edit():
        target = EditorAutomationHost.instance().assign_scene_mesh(
            object_id, component_id, asset_guid,
        )
        return {"object_id": int(object_id), "component": serializable_component(target)}

    return on_editor("infernux.scene.mesh.assign", edit)


def _open_scene(asset_guid: str) -> dict[str, object]:
    def open_document():
        path = asset_path(asset_guid, suffix=".scene")
        if not EditorAutomationHost.instance().open_scene(path):
            raise OperationError("scene.open_rejected", "The scene could not be opened.")
        return {"asset_guid": asset_guid, "path": path, "scheduled": True}

    return on_editor("infernux.scene.open", open_document)


def _load_additive_scene(asset_guid: str) -> dict[str, object]:
    def load():
        host = EditorAutomationHost.instance()
        path = asset_path(asset_guid, suffix=".scene")
        before = {
            int(item["world_id"])
            for item in _loaded_scene_snapshot()["scenes"]
        }
        if not host.load_additive_scene(path):
            raise OperationError(
                "scene.additive.load_rejected",
                "The scene could not be loaded additively.",
            )
        snapshot = _loaded_scene_snapshot()
        added = [
            item for item in snapshot["scenes"]
            if int(item["world_id"]) not in before
        ]
        if len(added) != 1:
            raise OperationError(
                "scene.additive.commit_mismatch",
                "Additive load did not publish exactly one resident Scene.",
            )
        return {
            "asset_guid": str(asset_guid),
            "path": path,
            "scene": added[0],
            **snapshot,
        }

    return on_editor("infernux.scene.additive.load", load)


def _save_scene() -> dict[str, object]:
    def save():
        return {
            "saved": True,
            "path": EditorAutomationHost.instance().save_scene(),
        }

    return on_editor("infernux.scene.save", save)


def _reload_scene(discard_changes: bool = False) -> dict[str, object]:
    return on_editor(
        "infernux.scene.reload",
        lambda: EditorAutomationHost.instance().reload_scene(
            discard_changes=bool(discard_changes)
        ),
    )


__all__ = ["build_scene_operations"]
