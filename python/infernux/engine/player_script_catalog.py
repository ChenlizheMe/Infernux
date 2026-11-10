"""Owner-published script identities and field schemas frozen for Player builds."""

from __future__ import annotations

import ast
import os
import sys


def capture_runtime_type_records(project_root, entries, sources):
    """Reflect the complete published gameplay closure, never a build worker's imports."""
    from infernux.engine.path_utils import relative_path
    from infernux.engine.runtime_artifact_catalog import source_path_for_entry

    components = []
    script_paths = {}
    for entry in entries:
        path = source_path_for_entry(project_root, entry)
        guid = str(entry["guid"])
        runtime_path = relative_path(path, project_root) + "c"
        script_paths[guid] = runtime_path
        if path not in sources:
            continue
        components.extend(runtime_component_type_records(
            sources[path].decode("utf-8-sig"), script_guid=guid,
            runtime_path=runtime_path, data_root=str(project_root),
        ))
    cook_runtime_component_semantics(components)
    return components + runtime_serializable_type_records(
        script_paths=script_paths, data_root=str(project_root),
    )


def runtime_script_module_name(
    runtime_path: str,
    *,
    data_root: str = "",
) -> str:
    normalized = runtime_path.replace("\\", "/")
    if not normalized.endswith(".pyc"):
        raise RuntimeError(f"Invalid Player script runtime path: {runtime_path}")
    if data_root:
        from infernux.engine.project_context import get_script_module_name

        module_name = get_script_module_name(
            os.path.join(data_root, *normalized.split("/")),
            data_root,
        )
        if not module_name:
            raise RuntimeError(
                f"Player script has no stable module identity: {runtime_path}"
            )
        return module_name
    if not normalized.startswith("Assets/"):
        raise RuntimeError(f"Invalid Player script runtime path: {runtime_path}")
    parts = normalized[len("Assets/"):-4].split("/")
    if parts and parts[-1] == "__init__":
        parts.pop()
    if not parts or any(not part.isidentifier() for part in parts):
        raise RuntimeError(
            f"Player component script has no stable module identity: {runtime_path}"
        )
    return ".".join(parts)


def runtime_component_type_records(
    source_text: str,
    *,
    script_guid: str,
    runtime_path: str,
    data_root: str = "",
) -> list[dict[str, object]]:
    """Extract Player component identities without executing author code."""
    tree = ast.parse(source_text, filename=runtime_path)
    component_bases = {"InxComponent"}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module in {
            "infernux",
            "infernux.components",
        }:
            for alias in node.names:
                if alias.name == "InxComponent" or alias.name == "*":
                    component_bases.add(alias.asname or "InxComponent")

    def base_name(expression: ast.expr) -> str:
        if isinstance(expression, ast.Name):
            return expression.id
        if isinstance(expression, ast.Attribute):
            return expression.attr
        return ""

    lifecycle_names = {
        "awake", "start", "fixed_update", "physics_pre_step",
        "physics_post_step", "update", "animation_update", "late_update",
        "on_enable", "on_disable", "on_destroy",
        "on_collision_enter", "on_collision_stay", "on_collision_exit",
        "on_trigger_enter", "on_trigger_stay", "on_trigger_exit",
    }
    module_name = runtime_script_module_name(
        runtime_path,
        data_root=data_root,
    )
    from infernux.components.component_identity import component_type_guid
    from infernux.components.registry import get_type_by_identity

    records: list[dict[str, object]] = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        type_guid = component_type_guid(script_guid, node.name) if script_guid else ""
        # Imported and aliased bases are resolved by the published registry,
        # not by executing imports or guessing their names from source.
        published_type = get_type_by_identity(node.name, script_guid, type_guid)
        if published_type is None and not any(
            base_name(base) in component_bases for base in node.bases
        ):
            continue
        if not script_guid:
            raise RuntimeError(
                f"Player component script has no Asset GUID: {runtime_path}"
            )
        component_bases.add(node.name)
        lifecycle = sorted(
            child.name
            for child in node.body
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            and child.name in lifecycle_names
        )
        startup_warmup = any(
            isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            and child.name == "_infernux_startup_warmup"
            for child in node.body
        )
        records.append(
            {
                "script_guid": script_guid,
                "type_guid": type_guid,
                "type_id": (
                    f"python:{script_guid}:{type_guid}:{module_name}:{node.name}"
                ),
                "module": module_name,
                "qualname": node.name,
                "runtime_path": runtime_path,
                "lifecycle": lifecycle,
                "startup_warmup": startup_warmup,
            }
        )
    return records


def _runtime_field_schemas(value_type):
    """Components and data must refer to the currently published data classes."""
    from infernux.components.fields import get_field_schema, get_serialized_fields
    from infernux.components.serializable_object import (
        SerializableObject, get_serializable_class, get_serializable_type_id,
    )

    fields = []
    for name, metadata in get_serialized_fields(value_type).items():
        for referenced in (metadata.serializable_class, metadata.element_class):
            if (isinstance(referenced, type) and issubclass(referenced, SerializableObject)
                    and referenced is not SerializableObject
                    and get_serializable_class(get_serializable_type_id(referenced)) is not referenced):
                raise RuntimeError(
                    f"Runtime field {value_type.__module__}.{value_type.__qualname__}.{name} "
                    "references a retired data type. Reload the dependent preload before building."
                )
        fields.append(get_field_schema(value_type, name).to_document())
    return fields


def cook_runtime_component_semantics(
    records: list[dict[str, object]],
) -> None:
    """Attach the already-published authoring schema to Player identities."""
    from infernux.components.registry import get_type_by_identity

    published_types = {}
    for record in records:
        script_guid = str(record["script_guid"])
        type_guid = str(record["type_guid"])
        qualname = str(record["qualname"])
        component_type = get_type_by_identity(
            qualname.rsplit(".", 1)[-1],
            script_guid,
            type_guid,
        )
        if component_type is None:
            raise RuntimeError(
                "Player semantic cook cannot resolve the published component "
                f"identity: {record['type_id']}"
            )
        published_types[component_type] = type_guid

    types_by_guid = {guid: value_type for value_type, guid in published_types.items()}
    for record in records:
        script_guid = str(record["script_guid"])
        type_guid = str(record["type_guid"])
        qualname = str(record["qualname"])
        component_type = types_by_guid[type_guid]
        owner = f"script:{script_guid}"
        record["semantic"] = {
            "type_guid": type_guid,
            "readable_id": str(record["type_id"]),
            "owner": owner,
            "origin": "python",
            "display_name": qualname,
            "base_type_guid": next(
                (published_types[base] for base in component_type.__mro__[1:] if base in published_types),
                "",
            ),
            "constructible": True,
            "serializable": True,
            "runtime_available": True,
            "runtime_profiles": ["player"],
            "lifecycle": list(record["lifecycle"]),
            "fields": _runtime_field_schemas(component_type),
        }


def runtime_serializable_type_records(
    *,
    script_paths: dict[str, str],
    data_root: str = "",
) -> list[dict[str, object]]:
    """Describe project SerializableObject types for the Player catalog.

    Components and authored data used to have different build visibility:
    components were emitted into ``RuntimeTypeRegistry.json`` while
    ``DataAsset``/nested ``SerializableObject`` classes were only visible
    as a side effect of importing a project script.  That made the binary
    artifact readable in a warm editor but left the Player semantic
    catalog incomplete.  Reuse the published field schema here and emit
    the same immutable semantic descriptor used by components.
    """
    from infernux.components.serializable_object import (
        SerializableObject,
        get_registered_serializable_types,
        get_serializable_type_id,
    )
    from infernux.core.data_asset import DataAsset
    from infernux.engine.path_utils import path_key

    scripts_by_module = {
        runtime_script_module_name(
            path.replace("\\", "/"), data_root=data_root
        ): (
            guid, path.replace("\\", "/")
        )
        for guid, path in script_paths.items()
    }
    scripts_by_source = {
        path_key(os.path.join(data_root, path.replace("\\", "/")[:-1])): (guid, path)
        for guid, path in script_paths.items()
    } if data_root else {}

    def script_for_type(value_type):
        # Preload-owned libraries also use ordinary Python import identities.
        # Their actual source still belongs to the same frozen GUID closure.
        module = sys.modules.get(value_type.__module__)
        source = getattr(module, "__file__", None)
        if data_root and source:
            if source.endswith(".pyc"):
                source = source[:-1]
            return scripts_by_source.get(path_key(source))
        return scripts_by_module.get(value_type.__module__)

    registered_types = dict(get_registered_serializable_types())
    project_types = [
        (type_id, value_type)
        for type_id, value_type in registered_types.items()
        if script_for_type(value_type) is not None
        and issubclass(value_type, SerializableObject)
        and value_type is not SerializableObject
    ]
    type_guids = {
        value_type: f"python-data:{type_id}"
        for type_id, value_type in project_types
    }
    records: list[dict[str, object]] = []
    for type_id, value_type in project_types:
        script_guid, runtime_path = script_for_type(value_type)
        type_guid = type_guids[value_type]
        for base in value_type.__mro__[1:]:
            if (issubclass(base, SerializableObject)
                    and base is not SerializableObject
                    and registered_types.get(get_serializable_type_id(base)) is not base):
                raise RuntimeError(
                    f"Runtime data type {value_type.__module__}.{value_type.__qualname__} "
                    f"inherits a retired class {base.__module__}.{base.__qualname__}. "
                    "Reload the dependent preload before building."
                )
        # Resolve against the whole build closure, using registered class
        # identity. A parent's script and explicit-ID policy are independent
        # of the child's; getattr would inherit an ancestor's explicit ID.
        base_type_guid = next(
            (type_guids[base] for base in value_type.__mro__[1:] if base in type_guids),
            "",
        )
        fields = _runtime_field_schemas(value_type)
        semantic = {
            "type_guid": type_guid,
            "readable_id": f"python:data:{type_id}",
            "owner": f"script:{script_guid}",
            "origin": "python",
            "display_name": value_type.__qualname__,
            "base_type_guid": base_type_guid,
            "constructible": True,
            "serializable": True,
            "runtime_available": True,
            "runtime_profiles": ["player"],
            "lifecycle": [],
            "fields": fields,
        }
        records.append(
            {
                "kind": "data",
                "script_guid": script_guid,
                "type_guid": type_guid,
                "type_id": f"python:data:{type_id}",
                "module": value_type.__module__,
                "qualname": value_type.__qualname__,
                "runtime_path": runtime_path,
                "lifecycle": [],
                "semantic": semantic,
                "data_asset": issubclass(value_type, DataAsset),
            }
        )
    return records
