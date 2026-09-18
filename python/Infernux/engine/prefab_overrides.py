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
class PropertyModification:
    """Detached serialized-field difference, identified without names or indices.

    Values use the existing value-document codec: source references are asset
    local, instance references are scene local. A zero component ID targets a
    GameObject; Transform has its actual instance component ID and source ID 0.
    Added/removed objects and components are structural overrides, not fields.
    """
    object_id: int
    component_id: int
    source_object_id: int
    source_component_id: int
    property_path: str
    source_value: object
    instance_value: object
    is_default_override: bool = False


@dataclass(frozen=True, slots=True)
class _PrefabApplyState:
    prefab_document: dict
    instance_documents: tuple[tuple[int, int, dict, object], ...]
    prefab_guid: str


# ─── Core diff ────────────────────────────────────────────────────────────

_SKIP_KEYS = frozenset({
    "id", "local_id", "children", "components",
    "transform", "prefab_guid", "prefab_root", "prefab_source_id", "prefab_source", "nested_prefab",
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
    object_ids, component_ids = _comparison_identities(instance_data, prefab_data)
    _diff_node(instance_data, prefab_data, "", overrides, object_ids, component_ids, is_root=True)
    return overrides


def _comparison_identities(instance, source):
    object_ids = {0: 0}
    _match_object_ids(instance, source, object_ids)
    component_ids = _instance_component_ids(instance, source)
    for node in _object_nodes(instance):
        object_ids[(node["id"], node["transform"]["component_id"])] = 0
        for component in node["components"]:
            object_ids[(node["id"], component["component_id"])] = component_ids[component["component_id"]]
    return object_ids, component_ids


def get_property_modifications(instance_obj, prefab_path: str):
    """Snapshot field differences for the containing instance, including children.

    Root placement/organization differences are included and marked as defaults,
    matching the existing whole-instance Apply/Revert exclusions. Lists and
    nested values remain one declared field, so they can be reverted atomically.
    This reads current source content; it neither records edits nor advances the
    merge baseline. Removed fields in the source are reported with a None value.
    """
    root = resolve_prefab_instance_root(instance_obj)
    if root is None:
        return ()
    source = _load_prefab_root(prefab_path)
    instance = _serialize_obj(root)
    object_ids, component_ids = _comparison_identities(instance, source)
    sources = {node["local_id"]: node for node in _object_nodes(source)}
    result = []

    def fields(node, original, instance_component_id, source_component_id,
               current, previous, prefix="", defaults=()):
        for key in sorted(current.keys() | previous.keys()):
            if key in current and key in previous and _same_value(current[key], previous[key], object_ids):
                continue
            result.append(PropertyModification(
                node["id"], instance_component_id, original["local_id"], source_component_id,
                prefix + key, copy.deepcopy(previous.get(key)), copy.deepcopy(current.get(key)),
                key in defaults,
            ))

    for node in _object_nodes(instance):
        original = sources.get(object_ids.get(node["id"]))
        if original is None:
            continue
        is_root = node["id"] == root.id
        fields(node, original, 0, 0,
               {key: value for key, value in node.items() if key not in _SKIP_KEYS},
               {key: value for key, value in original.items() if key not in _SKIP_KEYS},
               defaults=_ROOT_INSTANCE_KEYS if is_root else ())
        fields(node, original, node["transform"]["component_id"], 0,
               {key: node["transform"][key] for key in _TRANSFORM_KEYS},
               {key: original["transform"][key] for key in _TRANSFORM_KEYS},
               defaults=("position", "rotation") if is_root else ())
        original_components = {record["component_id"]: record for record in original["components"]}
        for record in node["components"]:
            previous = original_components.get(component_ids[record["component_id"]])
            if previous is None:
                continue
            fields(node, original, record["component_id"], previous["component_id"],
                   record["data"], previous["data"], prefix="data.")
            fields(node, original, record["component_id"], previous["component_id"],
                   {key: value for key, value in record.items() if key not in _COMPONENT_IDENTITY_KEYS},
                   {key: value for key, value in previous.items() if key not in _COMPONENT_IDENTITY_KEYS})
    return tuple(result)


_COMPONENT_IDENTITY_KEYS = frozenset({
    "type_id", "component_id", "instance_guid", "prefab_source_id", "data",
})


def is_property_override(component, field_name, prefab_path):
    section, key = _component_property_storage(component, field_name, writable=False)
    path = f"data.{key}" if section == "data" else key
    return any(item.component_id == component.component_id and item.property_path == path
               for item in get_property_modifications(component.game_object, prefab_path))


def apply_overrides_to_prefab(instance_obj, prefab_path: str,
                               asset_database=None) -> bool:
    """Write the current instance state back to the .prefab file.

    Resets the instance to non-overridden state (the prefab file now
    matches the instance).
    """
    instance_obj = resolve_prefab_instance_root(instance_obj) or instance_obj
    if asset_database is not None and asset_database.get_guid_from_path(prefab_path):
        try:
            build_prefab_apply_command(instance_obj, prefab_path, asset_database).execute()
            return True
        except Exception as exc:
            Debug.log_error(f"Failed to apply Prefab asset transaction: {exc}")
            return False
    try:
        from Infernux.engine.prefab_manager import (
            _read_prefab_document, _serialize_prefab_snapshot, save_prefab_document,
            _validate_nested_source_ancestry,
        )
        prefab_file = _read_prefab_document(prefab_path)
    except (OSError, ValueError) as exc:
        Debug.log_error(f"Failed to read prefab for apply: {exc}")
        return False

    prefab_guid = getattr(instance_obj, "prefab_guid", "") or ""
    runtime_document = _serialize_obj(instance_obj)
    updated_document, source_ids, component_ids = _serialize_prefab_snapshot(
        runtime_document,
        source_canvas_name=prefab_file.get("source_canvas_name", ""),
        root_document_template=prefab_file["root_object"],
        next_local_id=prefab_file["next_local_id"],
        next_component_id=prefab_file["next_component_id"],
    )
    instance_snapshots = _snapshot_all_linked_instances(
        prefab_guid, instance_root=instance_obj,
        source_document=runtime_document, source_ids=source_ids,
        component_ids=component_ids,
        base_root=prefab_file["root_object"],
    )

    # Resolve topology and preflight all instances before publishing the asset.
    # A cross-author parent cycle must not leave a changed source on disk.
    try:
        _validate_nested_source_ancestry(updated_document["root_object"], (prefab_guid,) if prefab_guid else ())
        prepared = _prepare_applied_prefab(
            prefab_file["root_object"], updated_document["root_object"],
            instance_snapshots, prefab_guid, asset_database,
        )
    except Exception as exc:
        Debug.log_error(f"Failed to preflight prefab instance propagation: {exc}")
        return False
    if not save_prefab_document(updated_document, prefab_path, asset_database=asset_database):
        for _obj, _merged, plan in prepared:
            plan.discard()
        return False
    return _publish_applied_prefab(prepared)


def build_prefab_apply_command(instance_obj, prefab_path: str,
                               asset_database=None):
    """Build the single reversible command for one Prefab Apply operation."""
    instance_root = resolve_prefab_instance_root(instance_obj) or instance_obj
    if instance_root is None or not prefab_path:
        raise ValueError("Prefab Apply requires a linked instance and asset path")
    from Infernux.engine.undo import PrefabApplyOverridesCommand

    prefab_guid = getattr(instance_root, "prefab_guid", "") or ""

    if asset_database is not None and asset_database.get_guid_from_path(prefab_path):
        from Infernux.engine.prefab_manager import _read_prefab_document, _serialize_prefab_snapshot
        from Infernux.engine.prefab_variant import edit_variant_document

        previous = _read_prefab_document(prefab_path)
        runtime = _serialize_obj(instance_root)
        updated, source_ids, component_ids = _serialize_prefab_snapshot(
            runtime, source_canvas_name=previous.get("source_canvas_name", ""),
            root_document_template=previous["root_object"], next_local_id=previous["next_local_id"],
            next_component_id=previous["next_component_id"],
        )
        if "variant" in previous:
            updated = edit_variant_document(previous, updated)
        return build_prefab_asset_edit_command(
            prefab_path, updated, asset_database, description="Apply Prefab Overrides",
            source_snapshot=dict(instance_root=instance_root, source_document=runtime,
                                 source_ids=source_ids, component_ids=component_ids),
        )

    def capture_state():
        return _capture_prefab_apply_state(None, prefab_path, prefab_guid)

    return PrefabApplyOverridesCommand(
        capture_state,
        lambda: apply_overrides_to_prefab(
            instance_root,
            prefab_path,
            asset_database,
        ),
        lambda state: _restore_prefab_apply_state(
            state,
            prefab_path,
            asset_database,
        ),
        scene_world_ids=_linked_prefab_world_ids(prefab_guid, instance_root),
        validate_replay=lambda expected: _validate_prefab_source_replay(expected, prefab_path, asset_database),
    )


def _capture_prefab_apply_state(instance_root, prefab_path: str,
                                prefab_guid: str) -> _PrefabApplyState:
    from Infernux.engine.prefab_manager import _read_prefab_document

    prefab_document = copy.deepcopy(_read_prefab_document(prefab_path))
    from Infernux.engine.scene_manager import SceneFileManager
    from Infernux.engine.interaction import DocumentRegistry

    files = SceneFileManager.instance()
    documents = []
    for scene in _loaded_prefab_scenes(instance_root):
        for obj in scene.get_all_objects():
            if (getattr(obj, "prefab_guid", "") or "") != prefab_guid:
                continue
            locator = (DocumentRegistry.instance().locate(files.document_id_for_scene(scene))
                       if files is not None else None)
            if not bool(getattr(obj, "prefab_root", False)):
                continue
            document = _serialize_obj(obj)
            if document is None:
                raise RuntimeError(
                    f"Failed to capture prefab instance '{getattr(obj, 'name', '')}'"
                )
            documents.append((int(scene.world_id), int(obj.id), copy.deepcopy(document), locator))
    return _PrefabApplyState(prefab_document, tuple(documents), prefab_guid)


def _loaded_prefab_scenes(instance_root=None):
    """All open worlds; isolated authoring copies are not scene instances."""
    from Infernux.lib import SceneManager

    manager = SceneManager.instance()
    scenes = [manager.get_scene_at(index) for index in range(manager.scene_count)]
    source_scene = getattr(instance_root, "scene", None)
    if source_scene is not None and not source_scene.is_preview and source_scene not in scenes:
        scenes.append(source_scene)
    return scenes


def _snapshot_all_linked_instances(prefab_guid, *, base_root, instance_root=None, **kwargs):
    return [snapshot for scene in _loaded_prefab_scenes(instance_root)
            for snapshot in _snapshot_linked_instances(
                scene, prefab_guid, base_root=base_root,
                instance_root=instance_root if instance_root is not None and instance_root.scene is scene else None,
                **kwargs,
            )]


def _linked_prefab_world_ids(prefab_guid, instance_root=None):
    return tuple(int(scene.world_id) for scene in _loaded_prefab_scenes(instance_root)
                 if any(obj.prefab_root and obj.prefab_guid == prefab_guid
                        for obj in scene.get_all_objects()))


def build_prefab_asset_edit_command(prefab_path, document, asset_database, *, source_snapshot=None,
                                   description="Save Prefab Asset"):
    """One source write and every open instance projection share the same Undo."""
    from Infernux.engine.undo import PrefabApplyOverridesCommand
    from Infernux.engine.prefab_manager import _read_prefab_document, _validate_nested_source_ancestry
    from Infernux.engine.prefab_variant import dependent_variant_updates

    guid = str(asset_database.get_guid_from_path(prefab_path) or "")
    if not guid:
        raise ValueError("Prefab source must be registered before editing")
    updates = dependent_variant_updates(prefab_path, copy.deepcopy(document), asset_database)
    identities = {path: str(asset_database.get_guid_from_path(path)) for path in updates}

    def capture():
        return {path: _capture_prefab_apply_state(None, path, identities[path]) for path in updates}

    def apply():
        prepared = []
        try:
            for path, updated in updates.items():
                previous = _read_prefab_document(path)
                identity = identities[path]
                _validate_nested_source_ancestry(updated["root_object"], (identity,))
                snapshots = _snapshot_all_linked_instances(
                    identity, base_root=previous["root_object"],
                    **(source_snapshot if path == prefab_path and source_snapshot else {}),
                )
                prepared.extend(_prepare_applied_prefab(
                    previous["root_object"], updated["root_object"], snapshots, identity, asset_database,
                ))
            for path, updated in updates.items():
                _write_prefab_apply_document(path, updated, asset_database)
        except BaseException:
            for _obj, _document, plan in prepared:
                plan.discard()
            raise
        return _publish_applied_prefab(prepared)

    return PrefabApplyOverridesCommand(
        capture, apply,
        lambda state: _restore_prefab_edit_batch(state, asset_database),
        description=description,
        scene_world_ids=tuple(dict.fromkeys(world for identity in identities.values()
                                           for world in _linked_prefab_world_ids(identity))),
        validate_replay=lambda expected: [_validate_prefab_source_replay(state, path, asset_database)
                                          for path, state in expected.items()],
    )


def _validate_prefab_source_replay(expected, prefab_path, asset_database):
    """History may replace its own source revision, not a different author's edit."""
    from Infernux.engine.prefab_manager import _read_prefab_document

    if asset_database is not None:
        guid = str(asset_database.get_guid_from_path(prefab_path) or "")
        if guid != expected.prefab_guid:
            raise RuntimeError("Prefab source identity changed; Undo/Redo cannot replace a different asset")
    current = _read_prefab_document(prefab_path)
    if current != expected.prefab_document:
        raise RuntimeError("Prefab source changed outside this history; Undo/Redo left it unchanged")


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


def _prepare_prefab_instance_restore(documents, asset_database=None):
    from Infernux.lib import SceneManager
    from Infernux.engine.component_restore import preflight_game_object_python_components
    from Infernux.engine.scene_manager import SceneFileManager
    from Infernux.engine.interaction import DocumentRegistry

    prepared = []
    try:
        for world_id, object_id, document, locator in documents:
            scene = SceneManager.instance().get_scene_by_world_id(world_id)
            if locator is not None:
                files = SceneFileManager.instance()
                bound = (DocumentRegistry.instance().get(files.document_id_for_scene(scene))
                         if files is not None and scene is not None else None)
                if bound is None or bound.stable_id != locator.stable_id:
                    if files is None:
                        raise RuntimeError("Prefab instance editor document owner is unavailable")
                    scene = files.restore_loaded_scene_locator(locator)
            if scene is None:
                raise RuntimeError(f"Prefab instance world {world_id} is unavailable")
            obj = scene.find_by_id(int(object_id))
            if obj is None:
                raise RuntimeError(f"Prefab instance root {object_id} is unavailable")
            plan = preflight_game_object_python_components(
                copy.deepcopy(document),
                asset_database,
                preserve_document_ids=True,
                reference_scene=scene,
                prefer_loaded_types=True,
            )
            prepared.append((obj, document, plan))
    except Exception:
        for _obj, _document, plan in prepared:
            plan.discard()
        raise
    return prepared


def _restore_prefab_apply_state(state: _PrefabApplyState, prefab_path: str,
                                asset_database=None) -> None:
    if not isinstance(state, _PrefabApplyState):
        raise TypeError("Prefab Apply restore state is invalid")
    _restore_prefab_edit_batch({prefab_path: state}, asset_database)


def _restore_prefab_edit_batch(states, asset_database):
    # All worlds/types must resolve before any source asset is replaced.
    documents = tuple(document for state in states.values() for document in state.instance_documents)
    prepared = _prepare_prefab_instance_restore(documents, asset_database)
    # Include owners just reopened above in transaction compensation as well.
    current = {path: _capture_prefab_apply_state(None, path, state.prefab_guid) for path, state in states.items()}
    try:
        for path, state in states.items():
            _write_prefab_apply_document(path, state.prefab_document, asset_database)
        if not _publish_applied_prefab(prepared):
            raise RuntimeError("Prefab ObjectGraph restore failed")
    except Exception:
        for _obj, _document, plan in prepared:
            plan.discard()
        try:
            documents = tuple(document for state in current.values() for document in state.instance_documents)
            previous = _prepare_prefab_instance_restore(documents, asset_database)
            for path, state in current.items():
                _write_prefab_apply_document(path, state.prefab_document, asset_database)
            if not _publish_applied_prefab(previous):
                raise RuntimeError("Prefab ObjectGraph compensation failed")
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


def _component_property_storage(component, field_name, *, writable=True):
    """Resolve public field spelling through the existing serialized schema."""
    from Infernux.components.builtin_component import BuiltinComponent, CppProperty
    from Infernux.components.fields import get_field_schema
    from Infernux.field_schema import get_native_field_schema

    if getattr(component, "type_name", "") == "Transform":
        schema = get_native_field_schema("native:infernux.Transform", field_name)
        section = "transform"
    elif isinstance(component, BuiltinComponent):
        descriptor = getattr(type(component), field_name, None)
        if not isinstance(descriptor, CppProperty) or descriptor.schema is None:
            raise ValueError(f"Not a declared serialized property: {field_name}")
        schema = descriptor.schema
        section = "data"
    else:
        schema = get_field_schema(type(component), field_name)
        section = "data"
    if writable and schema.read_only:
        raise ValueError(f"Property is read-only: {schema.property_path}")
    return section, schema.attributes.get("serialized_name", field_name)


def build_prefab_property_revert_command(component, field_name, prefab_path,
                                         asset_database=None):
    """Restore one declared field, retaining other overrides and exact identities."""
    section, key = _component_property_storage(component, field_name)
    owner = component.game_object
    root = resolve_prefab_instance_root(owner)
    if root is None:
        raise ValueError("Property target is not a Prefab instance member")
    source = _load_prefab_root(prefab_path)
    before = _serialize_obj(root)
    objects = {0: 0}
    _match_object_ids(before, source, objects)
    components = _instance_component_ids(before, source)
    source_id = objects.get(owner.id)
    source_node = next((node for node in _object_nodes(source)
                        if node["local_id"] == source_id), None)
    if source_node is None:
        raise ValueError("Added instance objects have no source property to revert")

    def storage(node, component_id):
        if section == "transform":
            return node["transform"]
        record = next((item for item in node["components"]
                       if item["component_id"] == component_id), None)
        if record is None:
            raise ValueError("Added or removed components have no source property to revert")
        return record["data"]

    source_component_id = components.get(component.component_id)
    source_value = storage(source_node, source_component_id)[key]
    identities = {local: runtime for runtime, local in objects.items()}
    for node in _object_nodes(before):
        local = objects.get(node["id"])
        if local is None:
            continue
        identities[(local, 0)] = node["transform"]["component_id"]
        for record in node["components"]:
            identities[(local, components[record["component_id"]])] = record["component_id"]

    def remap(value):
        if isinstance(value, list):
            return [remap(item) for item in value]
        if not isinstance(value, dict):
            return value
        kind = value.get(TYPE_KEY)
        reference_key = _REFERENCE_ID_KEYS.get(kind)
        if reference_key and value[reference_key]:
            result = copy.deepcopy(value)
            local = value[reference_key]
            # No local-ID fallback: a removed source member must not bind an
            # unrelated runtime object whose numeric ID happens to match.
            result[reference_key] = identities[local]
            if kind == COMPONENT_REF and "component_id" in value:
                result["component_id"] = identities[(local, value["component_id"])]
            return result
        return {name: remap(item) for name, item in value.items()}

    value = remap(source_value)
    after = copy.deepcopy(before)
    target = next(node for node in _object_nodes(after) if node["id"] == owner.id)
    destination = storage(target, component.component_id)
    destination[key] = value

    # Advance only this field's merge baseline. Updating the entire baseline
    # would hide unrelated source changes still waiting to reach this instance.
    if after.get("prefab_source"):
        from Infernux.engine.prefab_manager import _prefab_baseline_root
        baseline = _prefab_baseline_root(after["prefab_source"])
        baseline_node = next((node for node in _object_nodes(baseline)
                              if node["local_id"] == source_id), None)
        if baseline_node is not None:
            storage(baseline_node, source_component_id)[key] = copy.deepcopy(source_value)

    if after == before:
        return None
    from Infernux.engine.undo import PrefabRevertCommand
    return PrefabRevertCommand(root.id, before, after, asset_database,
                               description=f"Revert {type(component).__name__}.{field_name}")


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
    from Infernux.engine.prefab_manager import _stamp_prefab_guid, _make_prefab_baseline
    source_document = copy.deepcopy(prefab_data)
    _stamp_prefab_guid(prefab_data, prefab_guid, is_root=True)
    if prefab_guid:
        prefab_data["prefab_source"] = _make_prefab_baseline(
            source_document, outer_source_id=current_document.get("prefab_source", {}).get("outer_source_id", 0),
        )

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


def _project_prefab_document(source, current, *, object_id_map=None, component_id_map=None, reserve_ids=None):
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
    if component_id_map is None:
        component_id_map = _instance_component_ids(current, source)
    current_components = {component_id_map[component["component_id"]]: component
                          for node in _object_nodes(current) for component in node["components"]
                          if component["component_id"] in component_id_map}
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
        pairs.extend((component, current_components.get(component["component_id"]))
                     for component in node["components"])
        for component, previous in pairs:
            if component is None:
                continue
            reference_key = (node["local_id"], component.get("component_id", 0))
            if previous is None:
                new_components.append((reference_key, component))
            else:
                component["component_id"] = previous["component_id"]
                local_to_runtime[reference_key] = previous["component_id"]
                if "instance_guid" in previous:
                    component["instance_guid"] = previous["instance_guid"]
    object_ids, component_ids = reserve_ids(len(new_objects), len(new_components))
    for node, object_id in zip(new_objects, object_ids, strict=True):
        node["id"] = object_id
        local_to_runtime[node["local_id"]] = object_id
    for (reference_key, component), component_id in zip(new_components, component_ids, strict=True):
        component["component_id"] = component_id
        local_to_runtime[reference_key] = component_id

    def map_scene_references(value):
        if isinstance(value, dict):
            key = _REFERENCE_ID_KEYS.get(value.get(TYPE_KEY))
            if key and value[key] < 0:
                local_to_runtime.setdefault(value[key], -value[key])
            if value.get(TYPE_KEY) == COMPONENT_REF and value.get("component_id", 0) < 0:
                local_to_runtime[(value["game_object_id"], value["component_id"])] = -value["component_id"]
            for child in value.values():
                map_scene_references(child)
        elif isinstance(value, list):
            for child in value:
                map_scene_references(child)

    for node in _object_nodes(result):
        node.pop("local_id")
        node.pop("nested_prefab", None)
        for component in node["components"]:
            map_scene_references(component["data"])
            component["data"] = _remap_local_reference_document(component["data"], local_to_runtime, "prefab")
    return result


def _snapshot_linked_instances(scene, prefab_guid: str, *, base_root,
                               instance_root=None, source_document=None, source_ids=None, component_ids=None):
    """Capture linked roots for Apply or a saved Prefab Mode document change."""
    if scene is None:
        return []

    from Infernux.engine.prefab_manager import (
        _localize_prefab_snapshot,
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
        component_map = component_ids if obj is instance_root and component_ids is not None else _instance_component_ids(runtime_document, base_root)
        object_ids, _, component_map, _ = _localize_prefab_snapshot(
            local_document, source=base_root, instance_snapshot=True, object_id_map=ids,
            component_id_map=component_map,
        )
        snapshots.append((obj, runtime_document, local_document, object_ids, component_map))
    return snapshots


_MERGE_IDENTITY_KEYS = frozenset({
    "id", "instance_guid",
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
                        node_kind=key if node_kind == "object" and key in ("children", "components") else None,
                    )
            if value is not _MISSING:
                merged[key] = copy.deepcopy(value)
        return merged

    if isinstance(base, list) and isinstance(local, list) and isinstance(remote, list):
        if node_kind in ("children", "components"):
            identity_key = "local_id" if node_kind == "children" else "component_id"
            base_nodes = {item[identity_key]: item for item in base}
            local_nodes = {item[identity_key]: item for item in local}
            remote_nodes = {item[identity_key]: item for item in remote}
            merged = []
            common = set(base_nodes) & set(local_nodes)
            reordered = [key for key in local_nodes if key in common] != [key for key in base_nodes if key in common]
            order = [*local_nodes, *remote_nodes] if reordered else [*remote_nodes, *local_nodes]
            for source_id in dict.fromkeys(order):
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
                    merged.append(_three_way_merge_prefab(before, ours, theirs, node_kind="object" if node_kind == "children" else None))
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


def _merge_prefab_hierarchy(base, local, remote):
    """Merge node content and parent edges independently, keyed by source ID.

    A reparent is an edge edit, not deletion plus creation. Components still
    use the existing field/identity merge within their owning node.
    """
    from Infernux.engine.prefab_manager import PrefabDocumentError

    def index(root):
        nodes, parents, orders = {}, {}, {}

        def visit(node, parent):
            identity = node["local_id"]
            nodes[identity] = {key: value for key, value in node.items() if key != "children"}
            parents[identity] = parent
            orders[identity] = [child["local_id"] for child in node["children"]]
            for child in node["children"]:
                visit(child, identity)

        visit(root, None)
        return nodes, parents, orders

    before, old_parents, old_order = index(base)
    ours, our_parents, our_order = index(local)
    theirs, their_parents, their_order = index(remote)
    merged, parents = {}, {}
    for identity in dict.fromkeys([*theirs, *ours]):
        old = before.get(identity)
        own = ours.get(identity)
        new = theirs.get(identity)
        if own is None:
            if old is not None:  # Explicit instance deletion.
                continue
            merged[identity] = copy.deepcopy(new)
            parents[identity] = their_parents[identity]
        elif new is None:
            if old is not None and _prefab_content_equal(own, old) and our_parents[identity] == old_parents[identity]:
                continue
            merged[identity] = copy.deepcopy(own)
            parents[identity] = our_parents[identity]
        else:
            merged[identity] = _three_way_merge_prefab(old, own, new, node_kind="object")
            parents[identity] = _three_way_merge_prefab(
                old_parents.get(identity), our_parents[identity], their_parents[identity],
            )

    # A retained local override also retains its author-owned ancestry. New
    # source children below a locally deleted parent disappear with that parent.
    pending = list(merged)
    for identity in pending:
        parent = parents[identity]
        if parent is None or parent in merged:
            continue
        if parent in ours:
            merged[parent] = copy.deepcopy(ours[parent])
            parents[parent] = our_parents[parent]
            pending.append(parent)

    # Validate the selected edges before materializing a recursive hierarchy.
    # Two independently valid edits can form A->B->A; reject that conflict,
    # rather than guessing a parent or silently detaching either object.
    resolved = {}
    for identity in merged:
        chain, visiting = [], set()
        current = identity
        while current in merged and current not in resolved:
            if current in visiting:
                raise PrefabDocumentError("Prefab parent edits form a cycle; resolve the hierarchy conflict before Apply")
            visiting.add(current)
            chain.append(current)
            current = parents[current]
        connected = current is None or resolved.get(current, False)
        for member in chain:
            resolved[member] = connected

    children = defaultdict(list)
    for identity, node in merged.items():
        node["children"] = []
        if resolved[identity] and parents[identity] is not None:
            children[parents[identity]].append(identity)
    for parent, identities in children.items():
        old = old_order.get(parent, ())
        own = our_order.get(parent, ())
        new = their_order.get(parent, ())
        common = set(old) & set(own)
        reordered = [item for item in own if item in common] != [item for item in old if item in common]
        order = dict.fromkeys([*(own if reordered else new), *(new if reordered else own), *identities])
        selected = set(identities)
        merged[parent]["children"] = [merged[item] for item in order if item in selected]
    return merged[local["local_id"]]


def _merge_prefab_instance_document(runtime_document, local_document, object_ids,
                                    updated_root, prefab_guid, *, base_root=None,
                                    component_ids=None, reserve_ids=None):
    """One source/override merge for live publication and offline Player cook."""
    from Infernux.engine.prefab_manager import (
        _stamp_prefab_guid, _validate_game_object_document, _make_prefab_baseline, _prefab_baseline_root,
    )

    # A legacy scene has no historical baseline. Preserve authored values as
    # overrides on first adoption rather than guessing what the old source was.
    baseline = base_root if base_root is not None else _prefab_baseline_root(runtime_document.get("prefab_source", updated_root))
    _validate_game_object_document(baseline)
    merged = _merge_prefab_hierarchy(baseline, local_document, updated_root)
    source_ids = {node["local_id"] for node in _object_nodes(updated_root)}
    component_sources = {component["component_id"] for node in _object_nodes(updated_root) for component in node["components"]}
    _stamp_prefab_guid(merged, prefab_guid, is_root=True, source_ids=source_ids, component_source_ids=component_sources)
    if prefab_guid:
        merged["prefab_source"] = _make_prefab_baseline(
            updated_root, outer_source_id=runtime_document.get("prefab_source", {}).get("outer_source_id", 0),
        )
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
        merged, runtime_document, object_id_map=object_ids, component_id_map=component_ids, reserve_ids=reserve_ids,
    )


def resolve_scene_prefab_documents(document: dict, load_source, *, reserve_ids=None, affected_guids=None) -> dict:
    """Resolve a saved scene without instantiating objects or running scripts.

    Cook IDs are deterministic and allocated above every ID in this document,
    independent of the Editor's live object allocator. The caller owns source
    lookup through the frozen build catalog; unavailable sources are errors.
    Editor publication supplies the native allocator instead of cook-local IDs.
    """
    from Infernux.engine.prefab_manager import _localize_prefab_snapshot, _make_prefab_baseline

    result = copy.deepcopy(document)
    nodes = list(result.get("objects", ()))
    for node in nodes:
        nodes.extend(node.get("children", ()))
    if not any(node.get("prefab_guid") and node.get("prefab_root") for node in nodes):
        return result
    next_object = max((node["id"] for node in nodes), default=0) + 1
    next_component = max((component["component_id"] for node in nodes
                          for component in [node["transform"], *node["components"]]), default=0) + 1

    def reserve_document_ids(object_count, component_count):
        nonlocal next_object, next_component
        objects = range(next_object, next_object + object_count)
        components = range(next_component, next_component + component_count)
        next_object += object_count
        next_component += component_count
        return objects, components

    if reserve_ids is None:
        reserve_ids = reserve_document_ids

    sources = {}

    def resolve(node, ancestry=(), resolve_all=False):
        guid = node.get("prefab_guid")
        if guid and node.get("prefab_root") and (resolve_all or affected_guids is None or guid in affected_guids):
            if guid in ancestry:
                from Infernux.engine.prefab_manager import PrefabDocumentError
                raise PrefabDocumentError("Nested Prefab source cycle: " + " -> ".join((*ancestry, guid)))
            ancestry = (*ancestry, guid)
            if guid not in sources:
                sources[guid] = load_source(guid)
            updated_root = sources[guid]
            expected = _make_prefab_baseline(updated_root, outer_source_id=node.get("prefab_source", {}).get("outer_source_id", 0))
            if node.get("prefab_source") != expected:
                local = copy.deepcopy(node)
                ids = None
                if not node.get("prefab_source_id"):
                    ids = {}
                    _match_object_ids(node, updated_root, ids)
                object_ids, _, component_ids, _ = _localize_prefab_snapshot(
                    local, source=updated_root, instance_snapshot=True, object_id_map=ids,
                    component_id_map=_instance_component_ids(node, updated_root),
                )
                node = _merge_prefab_instance_document(
                    node, local, object_ids, updated_root, guid, component_ids=component_ids, reserve_ids=reserve_ids,
                )
            resolve_all = True
        node["children"] = [resolve(child, ancestry, resolve_all) for child in node["children"]]
        return node

    result["objects"] = [resolve(root) for root in result["objects"]]
    return result


def _propagate_applied_prefab(base_root: dict, updated_root: dict, snapshots,
                              prefab_guid: str, asset_database=None) -> bool:
    try:
        prepared = _prepare_applied_prefab(base_root, updated_root, snapshots, prefab_guid, asset_database)
    except Exception as exc:
        Debug.log_error(f"Failed to preflight prefab instance propagation: {exc}")
        return False
    return _publish_applied_prefab(prepared)


def _prepare_applied_prefab(base_root, updated_root, snapshots, prefab_guid, asset_database):
    from Infernux.engine.component_restore import preflight_game_object_python_components
    prepared_updates = []
    try:
        for obj, runtime_document, local_document, object_ids, component_ids in snapshots:
            merged = _merge_prefab_instance_document(
                runtime_document, local_document, object_ids, updated_root,
                prefab_guid, base_root=base_root, component_ids=component_ids,
            )
            prepared = preflight_game_object_python_components(
                merged,
                asset_database,
                preserve_document_ids=True,
                reference_scene=obj.scene,
                prefer_loaded_types=True,
            )
            prepared_updates.append((obj, merged, prepared))
    except Exception:
        for _obj, _merged, prepared in prepared_updates:
            prepared.discard()
        raise
    return prepared_updates


def _publish_applied_prefab(prepared_updates):
    from Infernux.engine.component_restore import commit_prepared_game_object_document

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
    from Infernux.engine.prefab_manager import _read_resolved_prefab_document

    return _read_resolved_prefab_document(prefab_path)["root_object"]


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


def _instance_component_ids(instance, source):
    """Project components by persisted identity; only old scenes need adoption.

    A versioned baseline distinguishes a new private component from a legacy
    unlinked component, even after the author deletes every source component.
    """
    from Infernux.engine.prefab_manager import _prefab_baseline_root, _nested_source_projection
    if any(node.get("prefab_root") for child in instance["children"] for node in _object_nodes(child)):
        return _nested_source_projection(instance, source)[1]
    baseline = instance.get("prefab_source", {})
    result = {component["component_id"]: component.get("prefab_source_id", -component["component_id"])
              for node in _object_nodes(instance) for component in node["components"]}
    if "component_identity_version" in baseline:
        _prefab_baseline_root(baseline)
        return result
    previous = _prefab_baseline_root(baseline) if baseline else source
    if previous is None:
        return result
    objects = {}
    _match_object_ids(instance, previous, objects)
    sources = {node["local_id"]: node for node in _object_nodes(previous)}
    for node in _object_nodes(instance):
        old = sources.get(objects.get(node["id"]))
        if old is not None:
            for component, original in _match_records(node["components"], old["components"], "type_id"):
                if component is not None and original is not None and not component.get("prefab_source_id"):
                    result[component["component_id"]] = original["component_id"]
    return result


def _match_object_ids(instance: dict, prefab: dict, object_ids: dict):
    object_ids[instance["id"]] = prefab["local_id"]
    if any(node.get("prefab_root") for child in instance["children"] for node in _object_nodes(child)):
        from Infernux.engine.prefab_manager import _nested_source_projection
        object_ids.update(_nested_source_projection(instance, prefab)[0])
        return
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


def _match_child_nodes(instance, prefab, object_ids):
    if not instance.get("prefab_source_id"):
        yield from _match_records(instance["children"], prefab["children"], "name")
        return
    sources = {child["local_id"]: child for child in prefab["children"]}
    for child in instance["children"]:
        yield child, sources.pop(object_ids.get(child["id"], 0), None)
    for source in sources.values():
        yield None, source


def _same_value(instance, prefab, object_ids: dict) -> bool:
    """Compare typed references across scene and asset identity domains."""
    if isinstance(instance, dict) and isinstance(prefab, dict):
        if instance.get(TYPE_KEY) == prefab.get(TYPE_KEY) == COMPONENT_REF:
            instance = dict(instance)
            if "component_id" not in prefab:
                instance.pop("component_id", None)
            elif "component_id" in instance:
                key = (instance["game_object_id"], instance["component_id"])
                instance["component_id"] = object_ids.get(key, -instance["component_id"])
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
               out: List[Override], object_ids: dict, component_ids: dict, *, is_root: bool = False):
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
        current_path, "components", out, object_ids, component_ids,
    )

    if instance.get("prefab_source_id"):
        source_order = [child["local_id"] for child in prefab["children"]]
        current_order = [object_ids.get(child["id"], 0) for child in instance["children"]]
        common = set(source_order) & set(current_order)
        if [value for value in current_order if value in common] != [value for value in source_order if value in common]:
            out.append(Override(current_path, "children.order", source_order, current_order))
    for i_child, p_child in _match_child_nodes(instance, prefab, object_ids):
        if i_child is None:
            child_name = p_child["name"]
            out.append(Override(current_path, f"removed_child:{child_name}", child_name, None))
        elif p_child is None:
            child_name = i_child["name"]
            out.append(Override(current_path, f"added_child:{child_name}", None, child_name))
        else:
            _diff_node(i_child, p_child, current_path, out, object_ids, component_ids)


def _diff_components(instance_comps: list, prefab_comps: list,
                     node_path: str, section: str,
                     out: List[Override], object_ids: dict, component_ids: dict):
    """Diff component lists by source identity, not by same-type occurrence."""
    sources = {component["component_id"]: component for component in prefab_comps}
    pairs = [(component, sources.pop(component_ids[component["component_id"]], None)) for component in instance_comps]
    pairs.extend((None, source) for source in sources.values())
    source_order = [component["component_id"] for component in prefab_comps]
    instance_order = [component_ids[component["component_id"]] for component in instance_comps]
    common = set(source_order) & set(instance_order)
    if [value for value in instance_order if value in common] != [value for value in source_order if value in common]:
        out.append(Override(node_path, "components.order", source_order, instance_order))
    for ic, pc in pairs:
        if ic is None:
            tn = pc["type_id"]
            out.append(Override(node_path, f"removed_{section}:{tn}", tn, None))
            continue
        tn = ic["type_id"]
        if pc is None:
            out.append(Override(node_path, f"added_{section}:{tn}", None, tn))
            continue
        # Compare fields within this component
        skip = {"type_id", "component_id", "instance_guid", "prefab_source_id"}
        for key in set(ic.keys()) | set(pc.keys()):
            if key in skip:
                continue
            if not _same_value(ic.get(key), pc.get(key), object_ids):
                out.append(Override(node_path, f"{section}:{tn}.{key}",
                                   pc.get(key), ic.get(key)))
