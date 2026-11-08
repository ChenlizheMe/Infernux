"""Build-authored Python component type and lifecycle contract for Player."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Mapping, Optional


RUNTIME_TYPE_REGISTRY_SCHEMA = "infernux.runtime_type_registry"
_runtime_types: dict[str, dict[str, Any]] = {}
_runtime_registry_installed = False
_runtime_semantic_owners: set[str] = set()
_RUNTIME_LIFECYCLE_METHODS = frozenset(
    {
        "awake",
        "start",
        "fixed_update",
        "physics_pre_step",
        "physics_post_step",
        "update",
        "animation_update",
        "late_update",
        "on_enable",
        "on_disable",
        "on_destroy",
        "on_collision_enter",
        "on_collision_stay",
        "on_collision_exit",
        "on_trigger_enter",
        "on_trigger_stay",
        "on_trigger_exit",
    }
)


def _prepare_semantic_edits(edits: list[dict[str, Any]]) -> Any:
    """Prepare semantic edits for desktop Players with the native catalog ABI.

    Web and Android Players use the cooked runtime type registry directly.
    Their minimal native runtimes intentionally do not ship the editor-only
    semantic catalog transaction ABI.
    """
    if not edits:
        return
    from infernux.lib import _Infernux as native

    prepare = getattr(native, "_semantic_catalog_prepare", None)
    if prepare is None:
        if os.environ.get("INFERNUX_WEB_RUNTIME") == "1" or os.sys.platform in {
            "emscripten",
            "android",
        }:
            return
        raise AttributeError("native semantic catalog transaction is unavailable")
    return prepare(edits)


@dataclass(slots=True)
class _PreparedRuntimeTypes:
    records: Optional[dict[str, dict[str, Any]]]
    owners: set[str]
    semantic_publication: Any

    def publish(self) -> int:
        global _runtime_types, _runtime_registry_installed, _runtime_semantic_owners

        if self.records is None:
            return len(_runtime_types)
        if self.semantic_publication is not None:
            self.semantic_publication.publish()
        _runtime_types = self.records
        _runtime_semantic_owners = self.owners
        _runtime_registry_installed = True
        return len(self.records)


def install_runtime_type_registry(path: str) -> int:
    with open(path, "r", encoding="utf-8") as stream:
        return _prepare_runtime_type_registry(json.load(stream)).publish()


def _prepare_runtime_type_registry(
    document: object,
    *,
    new_scripts: Optional[Mapping[str, str]] = None,
) -> _PreparedRuntimeTypes:
    if (
        not isinstance(document, dict)
        or document.get("$schema") != RUNTIME_TYPE_REGISTRY_SCHEMA
        or set(document) != {"$schema", "types"}
    ):
        raise RuntimeError("Unsupported Player runtime type registry schema")
    entries = document.get("types")
    if not isinstance(entries, list):
        raise RuntimeError("Player runtime type registry has no type list")

    prepared: dict[str, dict[str, Any]] = {}
    semantic_types: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise RuntimeError("Player runtime type registry contains a malformed entry")
        kind = entry.get("kind", "component")
        if kind not in {"component", "data"}:
            raise RuntimeError("Player runtime type registry entry has unknown kind")
        required = (
            "script_guid",
            "type_guid",
            "type_id",
            "module",
            "qualname",
            "runtime_path",
        )
        if any(not isinstance(entry.get(key), str) or not entry[key] for key in required):
            raise RuntimeError("Player runtime type registry entry has incomplete identity")
        if new_scripts is not None:
            expected_module = new_scripts.get(entry["script_guid"])
            if expected_module is None:
                continue
            if entry["module"] != expected_module:
                raise RuntimeError(f"Player runtime type has the wrong cooked module: {entry['module']}")
        phases = entry.get("lifecycle", [])
        if not isinstance(phases, list) or any(
            not isinstance(phase, str) or not phase for phase in phases
        ):
            raise RuntimeError("Player runtime type registry entry has invalid lifecycle data")
        if len(phases) != len(set(phases)) or any(
            phase not in _RUNTIME_LIFECYCLE_METHODS for phase in phases
        ):
            raise RuntimeError("Player runtime type registry entry has unknown lifecycle data")
        if not isinstance(entry.get("startup_warmup", False), bool):
            raise RuntimeError("Player runtime component has invalid startup warmup data")
        type_guid = entry["type_guid"]
        if type_guid in prepared:
            raise RuntimeError(f"Duplicate Player runtime component type: {type_guid}")
        semantic = entry.get("semantic")
        if not isinstance(semantic, dict):
            raise RuntimeError("Player runtime component has no cooked semantic descriptor")
        if semantic.get("type_guid") != type_guid:
            raise RuntimeError("Player runtime component semantic identity disagrees with its record")
        if semantic.get("readable_id") != entry["type_id"]:
            raise RuntimeError("Player runtime component semantic readable ID is invalid")
        owner = semantic.get("owner")
        if not isinstance(owner, str) or owner != f"script:{entry['script_guid']}":
            raise RuntimeError("Player runtime component semantic owner is invalid")
        if semantic.get("origin") != "python" or semantic.get("lifecycle") != phases:
            raise RuntimeError("Player runtime component semantic contract disagrees with its record")
        if kind == "data" and phases:
            raise RuntimeError("Player runtime data type cannot declare lifecycle methods")
        semantic_types.setdefault(owner, []).append(semantic)
        prepared[type_guid] = dict(
            entry,
            kind=kind,
            lifecycle=tuple(sorted(set(phases))),
        )

    owners = set(semantic_types)
    if new_scripts is None:
        changed_owners = _runtime_semantic_owners | owners
    else:
        if not prepared:
            return _PreparedRuntimeTypes(None, set(), None)
        if owners & _runtime_semantic_owners or prepared.keys() & _runtime_types.keys():
            raise RuntimeError("New Player runtime types conflict with published script identities")
        changed_owners = owners
        owners = _runtime_semantic_owners | owners
        prepared = _runtime_types | prepared
    edits = [
        {"owner": owner, "types": semantic_types.get(owner, [])}
        for owner in sorted(changed_owners)
    ]
    return _PreparedRuntimeTypes(prepared, owners, _prepare_semantic_edits(edits))


def runtime_type_contract(type_guid: str) -> Optional[dict[str, Any]]:
    record = _runtime_types.get(str(type_guid))
    return dict(record) if record is not None else None


def validate_runtime_component_identity(
    *,
    script_guid: str,
    type_guid: str,
    module_name: str,
    qualified_name: str,
) -> Optional[dict[str, Any]]:
    if not _runtime_registry_installed:
        if os.environ.get("_INFERNUX_PLAYER_MODE") == "1":
            raise RuntimeError("RuntimeTypeRegistry is not installed for Player")
        return None
    record = _runtime_types.get(type_guid)
    if record is None:
        raise RuntimeError(f"Player component type is absent from RuntimeTypeRegistry: {type_guid}")
    if record.get("kind", "component") != "component":
        raise RuntimeError(f"Player type is not an InxComponent: {type_guid}")
    expected = {
        "script_guid": script_guid,
        "module": module_name,
        "qualname": qualified_name,
    }
    mismatches = [
        key for key, value in expected.items() if record.get(key) != value
    ]
    if mismatches:
        raise RuntimeError(
            "Player component identity disagrees with RuntimeTypeRegistry: "
            + ", ".join(mismatches)
        )
    return dict(record)


def bind_runtime_lifecycle_contract(component_type: type, record: Optional[dict[str, Any]]) -> None:
    if record is None:
        if os.environ.get("_INFERNUX_PLAYER_MODE") == "1":
            raise RuntimeError("Player component has no RuntimeTypeRegistry contract")
        return
    lifecycle = frozenset(str(item) for item in record.get("lifecycle", ()))
    actual = frozenset(
        name
        for name in _RUNTIME_LIFECYCLE_METHODS
        if callable(getattr(component_type, name, None))
        and name in component_type.__dict__
    )
    if actual != lifecycle:
        raise RuntimeError(
            f"Player component lifecycle differs from RuntimeTypeRegistry: "
            f"{component_type.__module__}.{component_type.__qualname__}"
        )


def clear_runtime_type_registry() -> None:
    global _runtime_types, _runtime_registry_installed, _runtime_semantic_owners
    if _runtime_semantic_owners:
        publication = _prepare_semantic_edits(
            [
                {"owner": owner, "types": []}
                for owner in sorted(_runtime_semantic_owners)
            ]
        )
        if publication is not None:
            publication.publish()
    _runtime_types = {}
    _runtime_registry_installed = False
    _runtime_semantic_owners = set()


__all__ = [
    "RUNTIME_TYPE_REGISTRY_SCHEMA",
    "bind_runtime_lifecycle_contract",
    "clear_runtime_type_registry",
    "install_runtime_type_registry",
    "runtime_type_contract",
    "validate_runtime_component_identity",
]
