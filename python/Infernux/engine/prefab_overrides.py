"""
Prefab override diff system.

Compares a live prefab instance hierarchy against its source .prefab asset
to compute property-level overrides. Supports apply (write overrides back
to the .prefab file) and revert (reset instance to match the prefab).

Nodes retain their asset-local source identity independently of scene IDs.
Name/occurrence matching is limited to legacy instances without source IDs.
"""

import copy
from collections import defaultdict, deque
from dataclasses import dataclass
import json
import os
from typing import List, Optional

from Infernux.components.value_document import TYPE_KEY, GAME_OBJECT_REF, COMPONENT_REF
from Infernux.debug import Debug


# ─── Public data types ────────────────────────────────────────────────────

class Override:
    """A single property-level override on one node."""
    __slots__ = ("node_path", "key", "prefab_value", "instance_value")

    def __init__(self, node_path: str, key: str, prefab_value, instance_value):
        self.node_path = node_path
        self.key = key
        self.prefab_value = prefab_value
        self.instance_value = instance_value

    def __repr__(self):
        return f"Override({self.node_path!r}, {self.key!r})"


@dataclass(frozen=True, slots=True)
class _PrefabApplyState:
    prefab_document: dict
    instance_documents: tuple[tuple[int, dict], ...]


# ─── Core diff ────────────────────────────────────────────────────────────

_SKIP_KEYS = frozenset({
    "id", "local_id", "children", "components",
    "transform", "prefab_guid", "prefab_root", "prefab_source_id", "prefab_source",
})

_TRANSFORM_KEYS = ("position", "rotation", "scale")
_ROOT_INSTANCE_KEYS = frozenset({"name", "active", "is_static", "tag", "layer"})
_REFERENCE_ID_KEYS = {GAME_OBJECT_REF: "object_id", COMPONENT_REF: "game_object_id"}


def resolve_prefab_instance_root(instance_obj):
    """Return the linked root for the prefab instance containing *instance_obj*.

    Every node in an instantiated prefab carries the same ``prefab_guid``.
    ``prefab_root`` is authoritative.
    """
    if instance_obj is None:
        return None
    guid = getattr(instance_obj, "prefab_guid", "") or ""
    if not guid:
        return None

    current = instance_obj
    while current is not None:
        if (getattr(current, "prefab_guid", "") or "") != guid:
            break
        if bool(getattr(current, "prefab_root", False)):
            return current
        current = current.get_parent()
    raise LookupError(f"prefab instance has no marked root for GUID: {guid}")


def compute_overrides(instance_obj, prefab_path: str,
                      asset_database=None) -> List[Override]:
    """Compare *instance_obj* (live GameObject) against the .prefab file.

    Returns a list of Override objects describing every property difference.
    """
    instance_obj = resolve_prefab_instance_root(instance_obj) or instance_obj
    prefab_data = _load_prefab_root(prefab_path)
    instance_data = _serialize_obj(instance_obj)

    overrides: List[Override] = []
    object_ids = {0: 0}
    _match_object_ids(instance_data, prefab_data, object_ids)
    _diff_node(instance_data, prefab_data, "", overrides, object_ids, is_root=True)
    return overrides


def apply_overrides_to_prefab(instance_obj, prefab_path: str,
                               asset_database=None) -> bool:
    """Write the current instance state back to the .prefab file.

    Resets the instance to non-overridden state (the prefab file now
    matches the instance).
    """
    instance_obj = resolve_prefab_instance_root(instance_obj) or instance_obj
    try:
        from Infernux.engine.prefab_manager import (
            _read_prefab_document, _serialize_prefab_snapshot, save_prefab_document,
        )
        prefab_file = _read_prefab_document(prefab_path)
    except (OSError, ValueError) as exc:
        Debug.log_error(f"Failed to read prefab for apply: {exc}")
        return False

    prefab_guid = getattr(instance_obj, "prefab_guid", "") or ""
    runtime_document = _serialize_obj(instance_obj)
    updated_document, source_ids = _serialize_prefab_snapshot(
        runtime_document,
        source_canvas_name=prefab_file.get("source_canvas_name", ""),
        root_document_template=prefab_file["root_object"],
        next_local_id=prefab_file["next_local_id"],
    )
    instance_snapshots = _snapshot_linked_instances(
        instance_obj.scene, prefab_guid, instance_root=instance_obj,
        source_document=runtime_document, source_ids=source_ids,
        base_root=prefab_file["root_object"],
    )

    if not save_prefab_document(updated_document, prefab_path, asset_database=asset_database):
        return False

    if not _propagate_applied_prefab(
        prefab_file["root_object"],
        updated_document["root_object"],
        instance_snapshots,
        prefab_guid,
        asset_database,
    ):
        return False

    return True


def build_prefab_apply_command(instance_obj, prefab_path: str,
                               asset_database=None):
    """Build the single reversible command for one Prefab Apply operation."""
    instance_root = resolve_prefab_instance_root(instance_obj) or instance_obj
    if instance_root is None or not prefab_path:
        raise ValueError("Prefab Apply requires a linked instance and asset path")
    from Infernux.engine.undo import PrefabApplyOverridesCommand

    prefab_guid = getattr(instance_root, "prefab_guid", "") or ""

    def capture_state():
        return _capture_prefab_apply_state(instance_root, prefab_path, prefab_guid)

    return PrefabApplyOverridesCommand(
        capture_state,
        lambda: apply_overrides_to_prefab(
            instance_root,
            prefab_path,
            asset_database,
        ),
        lambda state: _restore_prefab_apply_state(
            state,
            instance_root,
            prefab_path,
            asset_database,
        ),
    )


def _capture_prefab_apply_state(instance_root, prefab_path: str,
                                prefab_guid: str) -> _PrefabApplyState:
    from Infernux.engine.prefab_manager import _read_prefab_document

    prefab_document = copy.deepcopy(_read_prefab_document(prefab_path))
    scene = getattr(instance_root, "scene", None)
    documents: list[tuple[int, dict]] = []
    if scene is not None and prefab_guid:
        for obj in scene.get_all_objects():
            if (getattr(obj, "prefab_guid", "") or "") != prefab_guid:
                continue
            if not bool(getattr(obj, "prefab_root", False)):
                continue
            document = _serialize_obj(obj)
            if document is None:
                raise RuntimeError(
                    f"Failed to capture prefab instance '{getattr(obj, 'name', '')}'"
                )
            documents.append((int(obj.id), copy.deepcopy(document)))
    return _PrefabApplyState(prefab_document, tuple(documents))


def _write_prefab_apply_document(prefab_path: str, document: dict,
                                 asset_database=None) -> None:
    from Infernux.engine.prefab_manager import (
        _invalidate_prefab_template_cache,
        _validate_prefab_document,
    )

    payload = copy.deepcopy(document)
    _validate_prefab_document(payload, prefab_path)
    from Infernux.core.document_store import write_document_text

    write_document_text(
        prefab_path,
        json.dumps(payload, indent=2, ensure_ascii=False),
    )
    guid = ""
    if asset_database is not None:
        from Infernux.core.assets import AssetManager

        mutation = AssetManager.import_asset(prefab_path, database=asset_database)
        if not mutation:
            raise RuntimeError(
                getattr(mutation, "error", "Prefab asset reimport failed")
            )
        guid = str(getattr(mutation, "guid", "") or "")
    _invalidate_prefab_template_cache(prefab_path, guid)


def _restore_prefab_instance_documents(instance_root, documents,
                                       asset_database=None) -> None:
    if not documents:
        return
    scene = getattr(instance_root, "scene", None)
    if scene is None:
        raise RuntimeError("Prefab instance scene is unavailable")
    from Infernux.engine.component_restore import (
        commit_prepared_game_object_document,
        preflight_game_object_python_components,
    )

    prepared = []
    try:
        for object_id, document in documents:
            obj = scene.find_by_id(int(object_id))
            if obj is None:
                raise RuntimeError(f"Prefab instance root {object_id} is unavailable")
            plan = preflight_game_object_python_components(
                copy.deepcopy(document),
                asset_database,
                preserve_document_ids=True,
                reference_scene=scene,
            )
            prepared.append((obj, document, plan))
        for obj, document, plan in prepared:
            if not commit_prepared_game_object_document(
                obj,
                copy.deepcopy(document),
                plan,
                preserve_document_ids=True,
            ):
                raise RuntimeError("Prefab ObjectGraph restore failed")
    except Exception:
        for _obj, _document, plan in prepared:
            try:
                plan.discard()
            except Exception:
                pass
        raise


def _restore_prefab_apply_state(state: _PrefabApplyState, instance_root,
                                prefab_path: str, asset_database=None) -> None:
    if not isinstance(state, _PrefabApplyState):
        raise TypeError("Prefab Apply restore state is invalid")
    current = _capture_prefab_apply_state(
        instance_root,
        prefab_path,
        getattr(instance_root, "prefab_guid", "") or "",
    )
    try:
        _write_prefab_apply_document(
            prefab_path,
            state.prefab_document,
            asset_database,
        )
        _restore_prefab_instance_documents(
            instance_root,
            state.instance_documents,
            asset_database,
        )
    except Exception:
        try:
            _write_prefab_apply_document(
                prefab_path,
                current.prefab_document,
                asset_database,
            )
            _restore_prefab_instance_documents(
                instance_root,
                current.instance_documents,
                asset_database,
            )
        except Exception as rollback_exc:
            Debug.log_error(f"Prefab Apply rollback failed: {rollback_exc}")
        raise


def revert_overrides(instance_obj, prefab_path: str,
                     asset_database=None) -> bool:
    """Reset the instance hierarchy to match the source .prefab file.

    Preserves the instance's transform (position in scene) and its
    prefab linkage fields.
    """
    instance_obj = resolve_prefab_instance_root(instance_obj) or instance_obj
    prefab_data = _build_reverted_prefab_document(instance_obj, prefab_path)
    if prefab_data is None:
        Debug.log_error("Failed to load prefab for revert.")
        return False

    try:
        from Infernux.engine.component_restore import deserialize_game_object_document_transactionally
        if not deserialize_game_object_document_transactionally(
            instance_obj,
            prefab_data,
            asset_database,
            preserve_document_ids=True,
        ):
            Debug.log_error("Failed to apply prefab document during revert.")
            return False
    except Exception as exc:
        Debug.log_error(f"Failed to deserialize during revert: {exc}")
        return False

    return True


def build_prefab_revert_command(instance_obj, prefab_path: str,
                                asset_database=None):
    """Build the single reversible command for one Prefab Revert operation."""
    instance_obj = resolve_prefab_instance_root(instance_obj) or instance_obj
    if instance_obj is None or not prefab_path:
        raise ValueError("Prefab Revert requires a linked instance and asset path")
    reverted_document = _build_reverted_prefab_document(instance_obj, prefab_path)
    if reverted_document is None:
        raise RuntimeError("Failed to load prefab for revert")
    from Infernux.engine.component_restore import (
        serialize_game_object_document_authoritatively,
    )
    from Infernux.engine.undo import PrefabRevertCommand

    before_document = serialize_game_object_document_authoritatively(instance_obj)
    return PrefabRevertCommand(
        instance_obj.id,
        before_document,
        reverted_document,
        asset_database,
    )


def _build_reverted_prefab_document(instance_obj, prefab_path: str):
    instance_obj = resolve_prefab_instance_root(instance_obj) or instance_obj
    prefab_data = _load_prefab_root(prefab_path)
    if prefab_data is None:
        return None

    # Root position and rotation place the instance in its scene. They are
    # not prefab overrides, so preserve them while reverting prefab-owned
    # scale and every child transform.
    current_document = _serialize_obj(instance_obj)
    current_transform = current_document.get("transform")

    # Keep prefab linkage
    prefab_guid = getattr(instance_obj, 'prefab_guid', '')

    # Stamp prefab linkage into the template
    from Infernux.engine.prefab_manager import _stamp_prefab_guid
    source_document = copy.deepcopy(prefab_data)
    _stamp_prefab_guid(prefab_data, prefab_guid, is_root=True)
    if prefab_guid:
        prefab_data["prefab_source"] = source_document

    # These fields describe this scene instance, not the prefab asset. Revert
    # must not rename the placed object or change its scene organization.
    if current_document:
        for key in _ROOT_INSTANCE_KEYS:
            if key in current_document:
                prefab_data[key] = copy.deepcopy(current_document[key])

    # Restore transform
    if current_transform:
        prefab_transform = prefab_data.get("transform")
        if isinstance(prefab_transform, dict):
            for key in ("position", "rotation"):
                if key in current_transform:
                    prefab_transform[key] = copy.deepcopy(current_transform[key])

    return _project_prefab_document(prefab_data, current_document)


def _object_nodes(root):
    yield root
    for child in root["children"]:
        yield from _object_nodes(child)


def _project_prefab_document(source, current, *, object_id_map=None, reserve_ids=None):
    """Resolve source identities to scene identities before ObjectGraph preflight.

    Unchanged nodes/components retain IDs; new records reserve IDs from the
    native allocators without publishing temporary objects or firing lifecycle.
    The returned document is reused by Redo, not allocated again on each replay.
    """
    from Infernux.engine.component_restore import _remap_local_reference_document

    if reserve_ids is None:
        from Infernux.lib import GameObject
        reserve_ids = GameObject._reserve_document_ids

    runtime_to_local = object_id_map
    if runtime_to_local is None:
        runtime_to_local = {0: 0}
        _match_object_ids(current, source, runtime_to_local)
    current_by_local = {
        runtime_to_local[node["id"]]: node for node in _object_nodes(current)
        if node["id"] in runtime_to_local
    }
    result = copy.deepcopy(source)
    new_objects, new_components = [], []
    local_to_runtime = {0: 0}
    for node in _object_nodes(result):
        old = current_by_local.get(node["local_id"])
        if old is None:
            new_objects.append(node)
        else:
            node["id"] = old["id"]
            local_to_runtime[node["local_id"]] = old["id"]
        pairs = [(node["transform"], old["transform"] if old else None)]
        pairs.extend(_match_records(node["components"], old["components"] if old else [], "type_id"))
        for component, previous in pairs:
            if component is None:
                continue
            if previous is None:
                new_components.append(component)
            else:
                component["component_id"] = previous["component_id"]
                if "instance_guid" in previous:
                    component["instance_guid"] = previous["instance_guid"]
    object_ids, component_ids = reserve_ids(len(new_objects), len(new_components))
    for node, object_id in zip(new_objects, object_ids, strict=True):
        node["id"] = object_id
        local_to_runtime[node["local_id"]] = object_id
    for component, component_id in zip(new_components, component_ids, strict=True):
        component["component_id"] = component_id

    def map_scene_references(value):
        if isinstance(value, dict):
            key = _REFERENCE_ID_KEYS.get(value.get(TYPE_KEY))
            if key and value[key] < 0:
                local_to_runtime.setdefault(value[key], -value[key])
            for child in value.values():
                map_scene_references(child)
        elif isinstance(value, list):
            for child in value:
                map_scene_references(child)

    for node in _object_nodes(result):
        node.pop("local_id")
        for component in node["components"]:
            map_scene_references(component["data"])
            component["data"] = _remap_local_reference_document(component["data"], local_to_runtime, "prefab")
    return result


def _snapshot_linked_instances(scene, prefab_guid: str, *, base_root,
                               instance_root=None, source_document=None, source_ids=None):
    """Capture linked roots for Apply or a saved Prefab Mode document change."""
    if scene is None:
        return []

    from Infernux.engine.prefab_manager import (
        _strip_prefab_fields,
        _strip_prefab_runtime_fields,
    )

    snapshots = []
    instances = [obj for obj in scene.get_all_objects() if prefab_guid
                 and obj.prefab_guid == prefab_guid and obj.prefab_root]
    if instance_root is not None and instance_root not in instances:
        instances.append(instance_root)
    for obj in instances:
        runtime_document = source_document if obj is instance_root else _serialize_obj(obj)
        if runtime_document is None:
            raise RuntimeError(
                f"Failed to snapshot linked prefab instance '{getattr(obj, 'name', '')}'"
            )
        local_document = copy.deepcopy(runtime_document)
        ids = source_ids if obj is instance_root else None
        if ids is None and not runtime_document.get("prefab_source_id"):
            ids = {}
            _match_object_ids(runtime_document, base_root, ids)
        object_ids, _ = _strip_prefab_runtime_fields(
            local_document, instance_snapshot=True, object_id_map=ids,
        )
        _strip_prefab_fields(local_document)
        snapshots.append((obj, runtime_document, local_document, object_ids))
    return snapshots


_MERGE_IDENTITY_KEYS = frozenset({
    "id", "component_id", "instance_guid",
    "prefab_guid", "prefab_root", "prefab_source_id", "prefab_source",
})
_MISSING = object()


def _prefab_content_equal(left, right) -> bool:
    """Compare prefab content while ignoring runtime/local identity metadata."""
    if isinstance(left, dict) and isinstance(right, dict):
        left_keys = set(left) - _MERGE_IDENTITY_KEYS
        right_keys = set(right) - _MERGE_IDENTITY_KEYS
        return left_keys == right_keys and all(
            _prefab_content_equal(left[key], right[key]) for key in left_keys
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _prefab_content_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


def _three_way_merge_prefab(base, local, remote, *, node_kind=None):
    """Merge one instance's old overrides onto an updated prefab document."""
    if _prefab_content_equal(local, base):
        return copy.deepcopy(remote)
    if _prefab_content_equal(remote, base) or _prefab_content_equal(local, remote):
        return copy.deepcopy(local)

    if isinstance(base, dict) and isinstance(local, dict) and isinstance(remote, dict):
        merged = {}
        for key in set(base) | set(local) | set(remote):
            if key in _MERGE_IDENTITY_KEYS:
                value = remote.get(key, local.get(key, base.get(key, _MISSING)))
            else:
                base_value = base.get(key, _MISSING)
                local_value = local.get(key, _MISSING)
                remote_value = remote.get(key, _MISSING)
                if local_value is _MISSING:
                    value = _MISSING if remote_value == base_value else remote_value
                elif remote_value is _MISSING:
                    value = local_value if local_value != base_value else _MISSING
                elif base_value is _MISSING:
                    value = local_value if local_value != remote_value else remote_value
                else:
                    value = _three_way_merge_prefab(
                        base_value, local_value, remote_value,
                        node_kind="children" if node_kind == "object" and key == "children" else None,
                    )
            if value is not _MISSING:
                merged[key] = copy.deepcopy(value)
        return merged

    if isinstance(base, list) and isinstance(local, list) and isinstance(remote, list):
        if node_kind == "children":
            base_nodes = {item["local_id"]: item for item in base}
            local_nodes = {item["local_id"]: item for item in local}
            remote_nodes = {item["local_id"]: item for item in remote}
            merged = []
            for source_id in dict.fromkeys([*remote_nodes, *local_nodes]):
                before = base_nodes.get(source_id)
                ours = local_nodes.get(source_id)
                theirs = remote_nodes.get(source_id)
                if ours is None:
                    if before is None and theirs is not None:
                        merged.append(copy.deepcopy(theirs))
                elif theirs is None:
                    if before is None or not _prefab_content_equal(ours, before):
                        merged.append(copy.deepcopy(ours))
                else:
                    merged.append(_three_way_merge_prefab(before, ours, theirs, node_kind="object"))
            return merged
        if len(base) == len(local) == len(remote):
            return [
                _three_way_merge_prefab(base_item, local_item, remote_item)
                for base_item, local_item, remote_item in zip(base, local, remote)
            ]
        # Concurrent structural edits cannot be matched safely without stable
        # element identities. Preserve the explicit instance override.
        return copy.deepcopy(local)

    # Both asset and instance changed the same scalar: the explicit instance
    # override wins, matching Unity-style per-instance override semantics.
    return copy.deepcopy(local)


def _merge_prefab_instance_document(runtime_document, local_document, object_ids,
                                    updated_root, prefab_guid, *, base_root=None,
                                    reserve_ids=None):
    """One source/override merge for live publication and offline Player cook."""
    from Infernux.engine.prefab_manager import _stamp_prefab_guid, _validate_game_object_document

    # A legacy scene has no historical baseline. Preserve authored values as
    # overrides on first adoption rather than guessing what the old source was.
    baseline = base_root if base_root is not None else runtime_document.get("prefab_source", updated_root)
    _validate_game_object_document(baseline)
    merged = _three_way_merge_prefab(baseline, local_document, updated_root, node_kind="object")
    source_ids = {node["local_id"] for node in _object_nodes(updated_root)}
    _stamp_prefab_guid(merged, prefab_guid, is_root=True, source_ids=source_ids)
    if prefab_guid:
        merged["prefab_source"] = copy.deepcopy(updated_root)
    for key in _ROOT_INSTANCE_KEYS:
        if key in runtime_document:
            merged[key] = copy.deepcopy(runtime_document[key])
    runtime_transform = runtime_document.get("transform")
    merged_transform = merged.get("transform")
    if isinstance(runtime_transform, dict) and isinstance(merged_transform, dict):
        for key in ("position", "rotation"):
            if key in runtime_transform:
                merged_transform[key] = copy.deepcopy(runtime_transform[key])
    return _project_prefab_document(
        merged, runtime_document, object_id_map=object_ids, reserve_ids=reserve_ids,
    )


def resolve_scene_prefab_documents(document: dict, load_source) -> dict:
    """Resolve a saved scene without instantiating objects or running scripts.

    Cook IDs are deterministic and allocated above every ID in this document,
    independent of the Editor's live object allocator. The caller owns source
    lookup through the frozen build catalog; unavailable sources are errors.
    """
    from Infernux.engine.prefab_manager import _strip_prefab_fields, _strip_prefab_runtime_fields

    result = copy.deepcopy(document)
    nodes = list(result.get("objects", ()))
    for node in nodes:
        nodes.extend(node.get("children", ()))
    if not any(node.get("prefab_guid") and node.get("prefab_root") for node in nodes):
        return result
    next_object = max((node["id"] for node in nodes), default=0) + 1
    next_component = max((component["component_id"] for node in nodes
                          for component in [node["transform"], *node["components"]]), default=0) + 1

    def reserve_ids(object_count, component_count):
        nonlocal next_object, next_component
        objects = range(next_object, next_object + object_count)
        components = range(next_component, next_component + component_count)
        next_object += object_count
        next_component += component_count
        return objects, components

    sources = {}

    def resolve(node):
        guid = node.get("prefab_guid")
        if guid and node.get("prefab_root"):
            if guid not in sources:
                sources[guid] = load_source(guid)
            updated_root = sources[guid]
            if node.get("prefab_source") == updated_root:
                return node
            local = copy.deepcopy(node)
            ids = None
            if not node.get("prefab_source_id"):
                ids = {}
                _match_object_ids(node, updated_root, ids)
            object_ids, _ = _strip_prefab_runtime_fields(local, instance_snapshot=True, object_id_map=ids)
            _strip_prefab_fields(local)
            return _merge_prefab_instance_document(
                node, local, object_ids, updated_root, guid, reserve_ids=reserve_ids,
            )
        node["children"] = [resolve(child) for child in node["children"]]
        return node

    result["objects"] = [resolve(root) for root in result["objects"]]
    return result


def _propagate_applied_prefab(base_root: dict, updated_root: dict, snapshots,
                              prefab_guid: str, asset_database=None) -> bool:
    if not snapshots:
        return True

    from Infernux.engine.component_restore import (
        commit_prepared_game_object_document,
        preflight_game_object_python_components,
    )
    prepared_updates = []
    try:
        for obj, runtime_document, local_document, object_ids in snapshots:
            merged = _merge_prefab_instance_document(
                runtime_document, local_document, object_ids, updated_root,
                prefab_guid, base_root=base_root,
            )
            prepared = preflight_game_object_python_components(
                merged,
                asset_database,
                preserve_document_ids=True,
                reference_scene=obj.scene,
            )
            prepared_updates.append((obj, merged, prepared))
    except Exception as exc:
        for _obj, _merged, prepared in prepared_updates:
            prepared.discard()
        Debug.log_error(f"Failed to preflight prefab instance propagation: {exc}")
        return False

    for index, (obj, merged, prepared) in enumerate(prepared_updates):
        try:
            if not commit_prepared_game_object_document(
                obj,
                merged,
                prepared,
                preserve_document_ids=True,
            ):
                raise RuntimeError("native ObjectGraph commit failed")
        except Exception as exc:
            prepared.discard()
            for _obj, _merged, remaining in prepared_updates[index + 1:]:
                remaining.discard()
            Debug.log_error(f"Failed to propagate applied prefab to scene instances: {exc}")
            return False
    return True


# ─── Internal helpers ─────────────────────────────────────────────────────

def _load_prefab_root(prefab_path: str) -> Optional[dict]:
    """Load and return the root_object dict from a .prefab file."""
    if not prefab_path:
        raise ValueError("prefab override comparison requires an asset path")
    from Infernux.engine.prefab_manager import _read_prefab_document

    return _read_prefab_document(prefab_path)["root_object"]


def _serialize_obj(obj) -> Optional[dict]:
    """Serialize a live GameObject to a dict."""
    from Infernux.engine.component_restore import (
        serialize_game_object_document_authoritatively,
    )

    return serialize_game_object_document_authoritatively(obj)


def _match_records(instances: list, sources: list, key: str):
    """Match each record once, retaining occurrence order for duplicate names/types."""
    remaining = defaultdict(deque)
    for source in sources:
        remaining[source[key]].append(source)
    for instance in instances:
        matches = remaining[instance[key]]
        yield instance, matches.popleft() if matches else None
    for matches in remaining.values():
        for source in matches:
            yield None, source


def _match_object_ids(instance: dict, prefab: dict, object_ids: dict):
    object_ids[instance["id"]] = prefab["local_id"]
    if instance.get("prefab_source_id"):
        sources = {node["local_id"] for node in _object_nodes(prefab)}
        for node in _object_nodes(instance):
            source_id = node.get("prefab_source_id", 0)
            if source_id in sources:
                object_ids[node["id"]] = source_id
        return
    for child, source in _match_records(instance["children"], prefab["children"], "name"):
        if child is not None and source is not None:
            _match_object_ids(child, source, object_ids)


def _match_child_nodes(instance, prefab):
    if not instance.get("prefab_source_id"):
        yield from _match_records(instance["children"], prefab["children"], "name")
        return
    sources = {child["local_id"]: child for child in prefab["children"]}
    for child in instance["children"]:
        yield child, sources.pop(child.get("prefab_source_id", 0), None)
    for source in sources.values():
        yield None, source


def _same_value(instance, prefab, object_ids: dict) -> bool:
    """Compare typed references across scene and asset identity domains."""
    if isinstance(instance, dict) and isinstance(prefab, dict):
        if instance.keys() != prefab.keys():
            return False
        reference_key = _REFERENCE_ID_KEYS.get(instance.get(TYPE_KEY))
        for key, value in instance.items():
            if key == reference_key:
                # External/added scene objects are overrides even if their numeric
                # runtime ID happens to equal an asset-local ID.
                if value not in object_ids or object_ids[value] != prefab[key]:
                    return False
            elif not _same_value(value, prefab[key], object_ids):
                return False
        return True
    if isinstance(instance, list) and isinstance(prefab, list):
        return len(instance) == len(prefab) and all(
            _same_value(value, source, object_ids) for value, source in zip(instance, prefab)
        )
    return instance == prefab


def _diff_node(instance: dict, prefab: dict, path: str,
               out: List[Override], object_ids: dict, *, is_root: bool = False):
    """Recursively diff one node."""
    node_name = instance.get("name", "")
    current_path = f"{path}/{node_name}" if path else node_name

    # Compare top-level scalar properties
    for key in set(instance.keys()) | set(prefab.keys()):
        if key in _SKIP_KEYS:
            continue
        if is_root and key in _ROOT_INSTANCE_KEYS:
            continue
        iv = instance.get(key)
        pv = prefab.get(key)
        if iv != pv:
            out.append(Override(current_path, key, pv, iv))

    # Compare transform sub-keys
    i_transform = instance.get("transform", {})
    p_transform = prefab.get("transform", {})
    for tk in _TRANSFORM_KEYS:
        if is_root and tk in ("position", "rotation"):
            continue
        iv = i_transform.get(tk)
        pv = p_transform.get(tk)
        if iv != pv:
            out.append(Override(current_path, f"transform.{tk}", pv, iv))

    # Compare components by type name matching
    _diff_components(
        instance.get("components", []),
        prefab.get("components", []),
        current_path, "components", out, object_ids,
    )

    if instance.get("prefab_source_id"):
        source_order = [child["local_id"] for child in prefab["children"]]
        current_order = [child.get("prefab_source_id", 0) for child in instance["children"]]
        common = set(source_order) & set(current_order)
        if [value for value in current_order if value in common] != [value for value in source_order if value in common]:
            out.append(Override(current_path, "children.order", source_order, current_order))
    for i_child, p_child in _match_child_nodes(instance, prefab):
        if i_child is None:
            child_name = p_child["name"]
            out.append(Override(current_path, f"removed_child:{child_name}", child_name, None))
        elif p_child is None:
            child_name = i_child["name"]
            out.append(Override(current_path, f"added_child:{child_name}", None, child_name))
        else:
            _diff_node(i_child, p_child, current_path, out, object_ids)


def _diff_components(instance_comps: list, prefab_comps: list,
                     node_path: str, section: str,
                     out: List[Override], object_ids: dict):
    """Diff component lists by type and occurrence."""
    for ic, pc in _match_records(instance_comps, prefab_comps, "type_id"):
        if ic is None:
            tn = pc["type_id"]
            out.append(Override(node_path, f"removed_{section}:{tn}", tn, None))
            continue
        tn = ic["type_id"]
        if pc is None:
            out.append(Override(node_path, f"added_{section}:{tn}", None, tn))
            continue
        # Compare fields within this component
        skip = {"type_id", "component_id", "instance_guid"}
        for key in set(ic.keys()) | set(pc.keys()):
            if key in skip:
                continue
            if not _same_value(ic.get(key), pc.get(key), object_ids):
                out.append(Override(node_path, f"{section}:{tn}.{key}",
                                   pc.get(key), ic.get(key)))
