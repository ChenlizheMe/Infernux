"""Material document and renderer-slot operations."""

from __future__ import annotations

from infernux.host import EditorAutomationHost, Operation, OperationError, OperationKind

from infernux.host.operation_support import (
    asset_path,
    component,
    interaction_core,
    on_editor,
    operation,
    set_json_pointer,
)


def build_material_operations() -> tuple[Operation, ...]:
    return (
        operation(
            "infernux.material.inspect",
            OperationKind.QUERY,
            "Read one material's canonical document by asset GUID.",
            _inspect_material,
            capability="material.read",
            input_properties={"asset_guid": {"type": "string"}},
            required=("asset_guid",),
            tags=("material", "asset", "guid", "inspect"),
        ),
        operation(
            "infernux.material.property.set",
            OperationKind.COMMAND,
            "Set an existing value in a material document using a JSON pointer.",
            _set_material_property,
            capability="material.write",
            input_properties={
                "asset_guid": {"type": "string"},
                "pointer": {"type": "string"},
                "value": {},
            },
            required=("asset_guid", "pointer", "value"),
            side_effects=("Changes and durably saves a material through editor history.",),
            reversible=True,
            tags=("material", "asset", "guid", "property", "authoring"),
        ),
        operation(
            "infernux.material.slot.assign",
            OperationKind.COMMAND,
            "Assign a GUID-addressed material to a renderer slot.",
            _assign_material_slot,
            capability="material.write",
            input_properties={
                "object_id": {"type": "integer"},
                "component_id": {"type": "integer"},
                "slot": {"type": "integer", "default": 0},
                "material_guid": {"type": "string"},
            },
            required=("object_id", "component_id", "material_guid"),
            side_effects=("Changes a renderer material slot and records an Undo entry.",),
            reversible=True,
            tags=("material", "renderer", "slot", "scene", "authoring"),
        ),
        operation(
            "infernux.renderer.parameter.get",
            OperationKind.QUERY,
            "Read one effective per-renderer material parameter override.",
            _get_renderer_parameter,
            capability="material.read",
            input_properties={
                "object_id": {"type": "integer"},
                "component_id": {"type": "integer"},
                "name": {"type": "string"},
                "slot": {"type": "integer", "default": 0},
                "persistent_only": {"type": "boolean", "default": False},
            },
            required=("object_id", "component_id", "name"),
            tags=("renderer", "material", "parameter", "inspect"),
        ),
        operation(
            "infernux.renderer.parameter.set",
            OperationKind.COMMAND,
            "Set one reflected material parameter on a renderer without mutating its shared material.",
            _set_renderer_parameter,
            capability="material.write",
            input_properties={
                "object_id": {"type": "integer"},
                "component_id": {"type": "integer"},
                "name": {"type": "string"},
                "value": {},
                "slot": {"type": "integer", "default": 0},
                "persistent": {"type": "boolean", "default": False},
            },
            required=("object_id", "component_id", "name", "value"),
            side_effects=("Publishes a renderer-local material parameter override.",),
            tags=("renderer", "material", "parameter", "override"),
        ),
        operation(
            "infernux.renderer.parameter.remove",
            OperationKind.COMMAND,
            "Remove one renderer-local material parameter override.",
            _remove_renderer_parameter,
            capability="material.write",
            input_properties={
                "object_id": {"type": "integer"},
                "component_id": {"type": "integer"},
                "name": {"type": "string"},
                "slot": {"type": "integer", "default": 0},
                "persistent": {"type": "boolean", "default": False},
            },
            required=("object_id", "component_id", "name"),
            side_effects=("Removes a renderer-local material parameter override.",),
            tags=("renderer", "material", "parameter", "override", "remove"),
        ),
        operation(
            "infernux.renderer.parameter.clear",
            OperationKind.COMMAND,
            "Clear one renderer material slot's parameter override layer.",
            _clear_renderer_parameters,
            capability="material.write",
            input_properties={
                "object_id": {"type": "integer"},
                "component_id": {"type": "integer"},
                "slot": {"type": "integer", "default": 0},
                "persistent": {"type": "boolean", "default": False},
            },
            required=("object_id", "component_id"),
            side_effects=("Clears a renderer-local material parameter override layer.",),
            tags=("renderer", "material", "parameter", "override", "clear"),
        ),
    )


def _load_material(asset_guid: str):
    path = asset_path(asset_guid, suffix=".mat")
    material, document = EditorAutomationHost.instance().material_document(path)
    return path, material, document


def _inspect_material(asset_guid: str) -> dict[str, object]:
    def read():
        path, _material, document = _load_material(asset_guid)
        return {"asset_guid": asset_guid, "path": path, "document": document}

    return on_editor("infernux.material.inspect", read)


def _set_material_property(asset_guid: str, pointer: str, value) -> dict[str, object]:
    def edit():
        path, _material, before = _load_material(asset_guid)
        after = set_json_pointer(before, pointer, value)
        after = EditorAutomationHost.instance().prepare_material_document_edit(
            after, property_pointer=pointer,
        )
        EditorAutomationHost.instance().publish_material_document(
            path,
            asset_guid,
            after,
            edit_key=f"material:{pointer}",
            description=f"Set Material {pointer}",
        )
        return {"asset_guid": asset_guid, "path": path, "pointer": pointer, "document": after}

    return on_editor("infernux.material.property.set", edit)


def _assign_material_slot(
    object_id: int,
    component_id: int,
    material_guid: str,
    slot: int = 0,
) -> dict[str, object]:
    def edit():
        asset_path(material_guid, suffix=".mat")
        _, renderer = component(object_id, component_id)
        getter = getattr(renderer, "get_material_guids", None)
        if not callable(getter):
            raise OperationError("material.renderer_required", "Target component has no material slots.")
        values = list(getter())
        index = int(slot)
        if index < 0:
            raise OperationError("operation.invalid_arguments", "slot must be non-negative")
        old_guid = str(values[index] or "") if index < len(values) else ""
        # False means the requested GUID is already assigned. Rejected edits
        # raise from the authoritative component service instead.
        interaction_core().components.set_material_slot(
            renderer,
            index,
            old_guid,
            material_guid,
        )
        return {
            "object_id": int(object_id),
            "component_id": int(component_id),
            "slot": index,
            "material_guid": material_guid,
        }

    return on_editor("infernux.material.slot.assign", edit)


def _renderer(object_id: int, component_id: int):
    _, renderer = component(object_id, component_id)
    required = ("get_parameter", "set_parameter", "remove_parameter", "clear_parameters")
    if not all(callable(getattr(renderer, name, None)) for name in required):
        raise OperationError(
            "material.renderer_required",
            "Target component does not support renderer parameter overrides.",
        )
    return renderer


def _parameter_value(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_parameter_value(item) for item in value]
    try:
        return [_parameter_value(item) for item in value]
    except TypeError:
        return str(value)


def _get_renderer_parameter(
    object_id: int,
    component_id: int,
    name: str,
    slot: int = 0,
    persistent_only: bool = False,
) -> dict[str, object]:
    def read():
        renderer = _renderer(object_id, component_id)
        value = renderer.get_parameter(
            str(name),
            material_slot=int(slot),
            persistent_only=bool(persistent_only),
        )
        return {
            "object_id": int(object_id),
            "component_id": int(component_id),
            "slot": int(slot),
            "name": str(name),
            "value": _parameter_value(value),
            "inherited": value is None,
        }

    return on_editor("infernux.renderer.parameter.get", read)


def _set_renderer_parameter(
    object_id: int,
    component_id: int,
    name: str,
    value,
    slot: int = 0,
    persistent: bool = False,
) -> dict[str, object]:
    def edit():
        renderer = _renderer(object_id, component_id)
        renderer.set_parameter(
            str(name), value, material_slot=int(slot), persistent=bool(persistent), owner="mcp"
        )
        return {
            "object_id": int(object_id),
            "component_id": int(component_id),
            "slot": int(slot),
            "name": str(name),
            "value": _parameter_value(
                renderer.get_parameter(str(name), material_slot=int(slot))
            ),
            "persistent": bool(persistent),
        }

    return on_editor("infernux.renderer.parameter.set", edit)


def _remove_renderer_parameter(
    object_id: int,
    component_id: int,
    name: str,
    slot: int = 0,
    persistent: bool = False,
) -> dict[str, object]:
    def edit():
        renderer = _renderer(object_id, component_id)
        removed = bool(
            renderer.remove_parameter(
                str(name), material_slot=int(slot), persistent=bool(persistent), owner="mcp"
            )
        )
        return {
            "object_id": int(object_id),
            "component_id": int(component_id),
            "slot": int(slot),
            "name": str(name),
            "removed": removed,
            "persistent": bool(persistent),
        }

    return on_editor("infernux.renderer.parameter.remove", edit)


def _clear_renderer_parameters(
    object_id: int,
    component_id: int,
    slot: int = 0,
    persistent: bool = False,
) -> dict[str, object]:
    def edit():
        renderer = _renderer(object_id, component_id)
        renderer.clear_parameters(
            material_slot=int(slot), persistent=bool(persistent), owner="mcp"
        )
        return {
            "object_id": int(object_id),
            "component_id": int(component_id),
            "slot": int(slot),
            "persistent": bool(persistent),
            "cleared": True,
        }

    return on_editor("infernux.renderer.parameter.clear", edit)


__all__ = ["build_material_operations"]
