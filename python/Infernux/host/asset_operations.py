"""GUID-addressed project asset operations."""

from __future__ import annotations

import os

from Infernux.engine.path_utils import is_path_within, resolved_path

from .editor import EditorAutomationHost
from .operations import Operation, OperationError, OperationKind

from .operation_support import (
    asset_database,
    asset_identity,
    asset_path,
    interaction_core,
    on_editor,
    operation,
)


_CREATE_KINDS = {
    "folder": "",
    "script": ".py",
    "shader": ".glsl",
    "material": ".mat",
    "physic_material": ".physicMaterial",
    "data_asset": ".inxdata",
    "render_texture": ".rendertexture",
    "scene": ".scene",
    "animation_clip": ".animclip2d",
    "animation_clip3d": ".animclip3d",
    "animation_fsm": ".animfsm",
    "particle_graph": ".particlegraph",
    "render_effect": ".effect",
    "render_effect_group": ".effectgroup",
    "animation_timeline": ".animtimeline",
    "timeline_fsm": ".timelinefsm",
}


def build_asset_operations(project_path: str) -> tuple[Operation, ...]:
    return (
        operation(
            "infernux.asset.list",
            OperationKind.QUERY,
            "List project assets with stable GUID identities.",
            lambda query="", extension="", root="assets", limit=200: _list_assets(
                project_path, query, extension, root, limit
            ),
            capability="asset.read",
            input_properties={
                "query": {"type": "string", "default": ""},
                "extension": {"type": "string", "default": ""},
                "root": {
                    "type": "string",
                    "enum": ["assets", "packages", "all"],
                    "default": "assets",
                },
                "limit": {"type": "integer", "default": 200},
            },
            tags=("asset", "guid", "list", "search"),
        ),
        operation(
            "infernux.asset.inspect",
            OperationKind.QUERY,
            "Inspect one project asset addressed by GUID.",
            _inspect_asset,
            capability="asset.read",
            input_properties={"asset_guid": {"type": "string"}},
            required=("asset_guid",),
            tags=("asset", "guid", "inspect", "metadata"),
        ),
        operation(
            "infernux.asset.text.read",
            OperationKind.QUERY,
            "Read one UTF-8 project text asset addressed by GUID.",
            _read_text_asset,
            capability="asset.read",
            input_properties={"asset_guid": {"type": "string"}},
            required=("asset_guid",),
            tags=("asset", "guid", "text", "read"),
        ),
        operation(
            "infernux.asset.text.set",
            OperationKind.COMMAND,
            "Replace one UTF-8 project text asset through Project history.",
            _set_text_asset,
            capability="asset.write",
            input_properties={
                "asset_guid": {"type": "string"},
                "content": {"type": "string"},
            },
            required=("asset_guid", "content"),
            side_effects=("Replaces and reimports a text asset with Undo support.",),
            reversible=True,
            tags=("asset", "guid", "text", "write", "authoring"),
        ),
        operation(
            "infernux.asset.create.kinds",
            OperationKind.QUERY,
            "List asset kinds accepted by the project creation service.",
            lambda: {"kinds": [{"kind": key, "extension": value} for key, value in _CREATE_KINDS.items()]},
            capability="asset.read",
            tags=("asset", "create", "schema"),
        ),
        operation(
            "infernux.asset.create",
            OperationKind.COMMAND,
            "Create a project asset through the same transaction as the Project panel.",
            lambda kind, directory, name, variant="": _create_asset(
                project_path, kind, directory, name, variant
            ),
            capability="asset.write",
            input_properties={
                "kind": {"type": "string"},
                "directory": {"type": "string"},
                "name": {"type": "string"},
                "variant": {"type": "string", "default": ""},
            },
            required=("kind", "directory", "name"),
            side_effects=("Creates and imports a project asset with Undo support.",),
            reversible=True,
            tags=("asset", "create", "import", "authoring"),
        ),
        operation(
            "infernux.asset.mesh.save-copy",
            OperationKind.COMMAND,
            "Save the current mesh geometry as an independent static asset with Undo/Redo.",
            lambda asset_guid, destination: _save_mesh_copy(project_path, asset_guid, destination),
            capability="asset.write",
            input_properties={"asset_guid": {"type": "string"}, "destination": {"type": "string"}},
            required=("asset_guid", "destination"),
            side_effects=("Creates and imports a new .inxmesh asset; never overwrites an existing path.",),
            reversible=True,
            tags=("asset", "mesh", "save", "authoring"),
        ),
        operation(
            "infernux.asset.delete",
            OperationKind.COMMAND,
            "Delete project assets addressed by GUID through asset history.",
            _delete_assets,
            capability="asset.write",
            input_properties={"asset_guids": {"type": "array", "items": {"type": "string"}}},
            required=("asset_guids",),
            side_effects=("Deletes asset files and metadata with Undo support.",),
            reversible=True,
            tags=("asset", "guid", "delete", "authoring"),
        ),
        operation(
            "infernux.asset.move",
            OperationKind.COMMAND,
            "Move or rename one GUID-addressed asset inside the project.",
            _move_asset,
            capability="asset.write",
            input_properties={
                "asset_guid": {"type": "string"},
                "destination": {"type": "string"},
            },
            required=("asset_guid", "destination"),
            side_effects=("Moves an asset while preserving its GUID and records Undo.",),
            reversible=True,
            tags=("asset", "guid", "move", "rename", "authoring"),
        ),
        operation(
            "infernux.asset.refresh",
            OperationKind.COMMAND,
            "Refresh the authoritative AssetDatabase catalog.",
            _refresh_assets,
            capability="asset.write",
            side_effects=("Scans project asset roots and imports changed assets.",),
            tags=("asset", "database", "refresh", "import"),
        ),
    )


def _list_assets(
    project_path: str,
    query: str = "",
    extension: str = "",
    root: str = "assets",
    limit: int = 200,
) -> dict[str, object]:
    def read():
        database = asset_database()
        needle = str(query or "").casefold()
        suffix = str(extension or "").casefold()
        scope = str(root or "assets").strip().casefold()
        if scope not in {"assets", "packages", "all"}:
            raise OperationError("operation.invalid_arguments", f"Unknown asset root: {root}")
        project_root = resolved_path(project_path)
        roots = {
            "assets": (os.path.join(project_root, "Assets"),),
            "packages": (os.path.join(project_root, "Packages"),),
            "all": (
                os.path.join(project_root, "Assets"),
                os.path.join(project_root, "Packages"),
            ),
        }[scope]
        values = []
        scoped_count = 0
        for path in database.get_all_asset_paths():
            text = resolved_path(path)
            if not any(is_path_within(text, candidate) for candidate in roots):
                continue
            scoped_count += 1
            if needle and needle not in text.casefold():
                continue
            if suffix and not text.casefold().endswith(suffix):
                continue
            values.append(asset_identity(text))
            if len(values) >= max(1, min(int(limit), 2000)):
                break
        return {
            "assets": values,
            "returned": len(values),
            "root": scope,
            "catalog_count": scoped_count,
            "global_catalog_count": int(database.asset_count),
        }

    return on_editor("infernux.asset.list", read)


def _inspect_asset(asset_guid: str) -> dict[str, object]:
    return on_editor(
        "infernux.asset.inspect",
        lambda: {"asset": asset_identity(asset_path(asset_guid))},
    )


def _read_text_asset(asset_guid: str) -> dict[str, object]:
    def read():
        path = asset_path(asset_guid)
        return {
            "asset": asset_identity(path),
            "content": EditorAutomationHost.instance().project_asset_text(path),
        }

    return on_editor("infernux.asset.text.read", read)


def _set_text_asset(asset_guid: str, content: str) -> dict[str, object]:
    def publish():
        path = asset_path(asset_guid)
        EditorAutomationHost.instance().set_project_asset_text(path, content)
        return {"asset": asset_identity(path), "content": str(content)}

    return on_editor("infernux.asset.text.set", publish)


def _create_asset(
    project_path: str,
    kind: str,
    directory: str,
    name: str,
    variant: str = "",
) -> dict[str, object]:
    def create():
        normalized = str(kind or "").strip()
        if normalized not in _CREATE_KINDS:
            raise OperationError("operation.invalid_arguments", f"Unknown asset kind: {kind}")
        project_root = resolved_path(project_path)
        resolved_directory = resolved_path(
            directory if os.path.isabs(directory) else os.path.join(project_root, directory)
        )
        if not any(
            is_path_within(resolved_directory, os.path.join(project_root, candidate))
            for candidate in ("Assets", "Packages")
        ):
            raise OperationError(
                "operation.invalid_arguments",
                "Assets may only be created inside the project Assets or Packages roots.",
            )
        if normalized == "folder":
            existing = os.path.join(resolved_directory, str(name or "").strip())
            if os.path.isdir(existing):
                return {"asset": asset_identity(existing), "already_exists": True}
        path = EditorAutomationHost.instance().create_project_asset(
            normalized,
            resolved_directory,
            name,
            _CREATE_KINDS[normalized],
            variant,
        )
        if not path:
            raise OperationError("asset.create_rejected", "Project asset creation was rejected.")
        return {"asset": asset_identity(path), "already_exists": False}

    return on_editor("infernux.asset.create", create)


def _save_mesh_copy(project_path: str, asset_guid: str, destination: str) -> dict[str, object]:
    def save():
        asset_path(asset_guid)
        target = destination if os.path.isabs(destination) else os.path.join(project_path, destination)
        saved = EditorAutomationHost.instance().save_mesh_copy(asset_guid, target)
        return {"asset": asset_identity(saved), "source_guid": asset_guid}

    return on_editor("infernux.asset.mesh.save-copy", save)


def _delete_assets(asset_guids: list[str]) -> dict[str, object]:
    def delete():
        paths = [asset_path(guid) for guid in asset_guids]
        deleted = interaction_core().project_assets.delete(paths)
        return {"deleted": [{"guid": guid, "path": path} for guid, path in zip(asset_guids, deleted)]}

    return on_editor("infernux.asset.delete", delete)


def _move_asset(asset_guid: str, destination: str) -> dict[str, object]:
    def move():
        source = asset_path(asset_guid)
        target = interaction_core().project_assets.move(source, destination)
        database = asset_database()
        resolved_guid = str(database.get_guid_from_path(target) or asset_guid)
        return {"asset": {**asset_identity(target), "guid": resolved_guid}}

    return on_editor("infernux.asset.move", move)


def _refresh_assets() -> dict[str, object]:
    def refresh():
        database = asset_database()
        database.refresh()
        return {
            "asset_count": int(database.asset_count),
            "imported": int(database.last_refresh_imported_count),
            "reused": int(database.last_refresh_reused_count),
            "scanned": int(database.last_refresh_scanned_count),
        }

    return on_editor("infernux.asset.refresh", refresh)


__all__ = ["build_asset_operations"]
