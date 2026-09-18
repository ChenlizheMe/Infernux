"""Synchronize live model instances after an external source reimport.

Imported models are references, not flattened scene content.  A scene keeps
the instance hierarchy and its authored transforms while the source asset owns
the node list and geometry.  This module reconciles the two at the owner
safe-point after a model publication.
"""

from __future__ import annotations

import json
from typing import Any


def _object_path_map(root: Any) -> dict[tuple[str, ...], Any]:
    result: dict[tuple[str, ...], Any] = {(): root}

    def visit(parent: Any, prefix: tuple[str, ...]) -> None:
        for child in tuple(parent.get_children() or ()):
            path = prefix + (str(child.name),)
            result[path] = child
            visit(child, path)

    visit(root, ())
    return result


def _descendants(root: Any) -> tuple[Any, ...]:
    result: list[Any] = []

    def visit(parent: Any) -> None:
        for child in tuple(parent.get_children() or ()):
            result.append(child)
            visit(child)

    visit(root)
    return tuple(result)


def _model_instance_roots(scene: Any, guid: str) -> tuple[Any, ...]:
    """Find model containers from persisted model-source identities.

    The source binding is deliberately independent from the visible object
    name.  The renderer-path walk remains only for scenes authored before the
    model_source field was introduced.
    """
    roots: dict[int, Any] = {}
    for obj in tuple(scene.get_all_objects() or ()):
        if str(getattr(obj, "_model_source_guid", "") or "") == guid and not tuple(
            getattr(obj, "_model_source_path", ()) or ()
        ):
            roots[int(obj.id)] = obj
            continue
        renderer = obj.get_component("MeshRenderer")
        if renderer is None or str(renderer.mesh_asset_guid or "") != guid:
            continue
        source_path = tuple(str(part) for part in (renderer.model_node_path or ()))
        if not source_path:
            continue

        chain = [obj]
        parent = obj.get_parent()
        while parent is not None:
            chain.append(parent)
            parent = parent.get_parent()
        reverse_path = source_path[::-1]
        for offset in range(0, len(chain) - len(reverse_path)):
            names = tuple(str(item.name) for item in chain[offset:offset + len(reverse_path)])
            if names != reverse_path:
                continue
            root = chain[offset + len(reverse_path)]
            roots[int(root.id)] = root
            break
    return tuple(roots.values())


def _source_paths(mesh: Any) -> dict[tuple[str, ...], dict]:
    nodes = tuple(mesh.model_nodes or ())
    # Keep the node path construction identical to the native CreateModelObject
    # contract: paths are source names, never scene object IDs or transforms.
    result: dict[tuple[str, ...], dict] = {}
    computed: list[tuple[str, ...]] = []
    for node in nodes:
        parent_index = int(node.get("parent_index", -1))
        prefix = computed[parent_index] if parent_index >= 0 else ()
        path = prefix + (str(node["name"]),)
        computed.append(path)
        result[path] = node
    return result


def _source_subresource_ids(guid: str) -> dict[tuple[str, ...], str]:
    """Return unique imported mesh identities for the current model source.

    The importer owns this manifest.  Missing/duplicate identities are left
    out so reconciliation keeps the path-based behavior instead of guessing.
    """
    from Infernux.core.assets import AssetManager

    database = AssetManager.require_asset_database()
    meta = database.get_meta_by_guid(guid)
    if meta is None or not meta.has_key("model_meshes"):
        return {}
    manifest = json.loads(meta.get_string("model_meshes"))
    if not isinstance(manifest, list):
        raise ValueError("model_meshes metadata must be a list")
    counts: dict[str, int] = {}
    entries: list[tuple[tuple[str, ...], str]] = []
    for entry in manifest:
        if not isinstance(entry, dict):
            continue
        path = entry.get("path")
        identifier = str(entry.get("subresource_id") or "")
        if not isinstance(path, list) or not identifier:
            continue
        key = tuple(str(part) for part in path)
        counts[identifier] = counts.get(identifier, 0) + 1
        entries.append((key, identifier))
    return {path: identifier for path, identifier in entries if counts[identifier] == 1}


def _migrate_source_paths_by_identity(scene: Any, root: Any, guid: str, source_ids: dict[tuple[str, ...], str]) -> bool:
    """Move a live model hierarchy's source bindings across unambiguous renames.

    The object itself is retained, so authored TRS, materials and user-added
    components remain intact.  A user-renamed GameObject keeps its display
    name; only its hidden source binding follows the DCC path.
    """
    if not source_ids:
        return False
    candidates = (root, *_descendants(root))
    by_id: dict[str, Any] = {}
    for obj in candidates:
        if str(getattr(obj, "_model_source_guid", "") or "") != guid:
            continue
        renderer = obj.get_component("MeshRenderer")
        identifier = str(getattr(renderer, "model_subresource_id", "") or "") if renderer else ""
        if identifier and identifier in source_ids.values() and identifier not in by_id:
            by_id[identifier] = obj

    prefix_maps: dict[tuple[str, ...], tuple[str, ...]] = {}
    for new_path, identifier in source_ids.items():
        obj = by_id.get(identifier)
        if obj is None:
            continue
        old_path = tuple(str(part) for part in (getattr(obj, "_model_source_path", ()) or ()))
        if not old_path or old_path == new_path or len(old_path) != len(new_path):
            continue
        for length in range(1, len(old_path) + 1):
            old_prefix = old_path[:length]
            new_prefix = new_path[:length]
            previous = prefix_maps.get(old_prefix)
            if previous is not None and previous != new_prefix:
                prefix_maps.clear()
                break
            prefix_maps[old_prefix] = new_prefix
        if not prefix_maps:
            break
    if not prefix_maps:
        return False

    changed = False
    for obj in candidates:
        if str(getattr(obj, "_model_source_guid", "") or "") != guid:
            continue
        old_path = tuple(str(part) for part in (getattr(obj, "_model_source_path", ()) or ()))
        new_path = prefix_maps.get(old_path)
        if new_path is None or new_path == old_path:
            continue
        if old_path and str(obj.name) == old_path[-1]:
            obj.name = new_path[-1]
        obj._set_model_source(guid, list(new_path))
        renderer = obj.get_component("MeshRenderer")
        if renderer is not None and renderer.model_node_path == list(old_path):
            renderer.set_model_mesh(guid, list(new_path))
        changed = True

    # A source node may have moved under a different DCC parent as well as
    # being renamed. Rebuild only that source hierarchy edge and preserve the
    # authored world pose; source-local TRS is for newly created instances.
    updated_by_path = {
        tuple(str(part) for part in (getattr(obj, "_model_source_path", ()) or ())): obj
        for obj in candidates
        if str(getattr(obj, "_model_source_guid", "") or "") == guid
    }
    for path, obj in sorted(updated_by_path.items(), key=lambda item: (len(item[0]), item[0])):
        if not path:
            continue
        parent = updated_by_path.get(path[:-1], root if len(path) == 1 else None)
        if parent is None or obj.get_parent() is parent:
            continue
        obj.set_parent(parent, world_position_stays=True)
        changed = True
    return changed


def _destroy_stale_geometry(
    scene: Any, root: Any, guid: str, source: dict[tuple[str, ...], dict]
) -> bool:
    paths = _object_path_map(root)
    source_objects: dict[tuple[str, ...], Any] = {}
    for candidate in (root, *_descendants(root)):
        if str(getattr(candidate, "_model_source_guid", "") or "") != guid:
            continue
        source_path = tuple(str(part) for part in (getattr(candidate, "_model_source_path", ()) or ()))
        source_objects[source_path] = candidate
    stale: dict[int, Any] = {}
    if source_objects:
        for path, obj in tuple(source_objects.items()):
            if not path:
                continue
            if path not in source:
                stale[path] = obj
                continue
            source_node = source[path]
            if int(source_node.get("node_group", -1)) < 0 and obj.get_component("MeshRenderer") is not None:
                stale[path] = obj
        if not stale:
            return False
        # Destroy only the highest stale source node. Its descendants belong
        # to the same removed source subtree and are retired with it.
        selected = []
        stale_paths = set(stale)
        for path, obj in sorted(stale.items(), key=lambda item: (len(item[0]), item[0])):
            if any(path[:index] in stale_paths for index in range(1, len(path))):
                continue
            selected.append(obj)
        for obj in selected:
            if obj is not root:
                scene.destroy_game_object(obj)
        scene.process_pending_destroys()
        return True
    for path, obj in tuple(paths.items()):
        if not path:
            continue
        renderer = obj.get_component("MeshRenderer")
        if renderer is None:
            continue
        if not str(renderer.mesh_asset_guid or ""):
            continue
        if str(renderer.mesh_asset_guid) != guid:
            # This branch is only a defensive guard for user-authored children;
            # the caller filters instances by source GUID.
            continue
        source_node = source.get(path)
        if source_node is None or int(source_node.get("node_group", -1)) < 0:
            stale[path] = obj
            continue
        # A removed source pivot must take its old geometry with it.  Destroy
        # the first missing ancestor, not each descendant independently.
        prefix: tuple[str, ...] = ()
        for part in path:
            prefix += (part,)
            if prefix not in source:
                stale[prefix] = paths.get(prefix, obj)
                break
    if not stale:
        return False
    for obj in stale.values():
        if obj is not root:
            scene.destroy_game_object(obj)
    scene.process_pending_destroys()
    return True


def _sync_instance(scene: Any, root: Any, guid: str, mesh: Any) -> bool:
    source = _source_paths(mesh)
    changed = _migrate_source_paths_by_identity(scene, root, guid, _source_subresource_ids(guid))
    changed = _destroy_stale_geometry(scene, root, guid, source) or changed
    source_bound = {
        tuple(str(part) for part in (getattr(obj, "_model_source_path", ()) or ())): obj
        for obj in (root, *_descendants(root))
        if str(getattr(obj, "_model_source_guid", "") or "") == guid
    }
    existing = source_bound or _object_path_map(root)
    missing = [path for path in source if path not in existing]
    if not missing:
        return changed

    # Native model creation already performs TRS decomposition and creates the
    # correct renderer/material bindings.  Use it as the source-node factory,
    # then move only missing nodes into the existing instance.  Existing nodes
    # and all authored transforms/components remain untouched.
    temporary = scene.create_from_model(guid, "__InfernuxModelRefresh__")
    if temporary is None:
        raise RuntimeError(f"cannot stage refreshed model instance for {guid}")
    staged = _object_path_map(temporary)
    try:
        for path in sorted(missing, key=lambda item: (len(item), item)):
            source_object = staged.get(path)
            if source_object is None:
                continue
            parent_path = path[:-1]
            parent = root if not parent_path else existing.get(parent_path)
            if parent is None:
                raise RuntimeError(f"model source parent is missing: {parent_path}")
            source_object.set_parent(parent, world_position_stays=False)
            existing[path] = source_object
            changed = True
    finally:
        # Existing source nodes stayed under the temporary container and are
        # discarded. Newly moved nodes are now owned by the real instance.
        scene.destroy_game_object(temporary)
        scene.process_pending_destroys()
    return changed


def synchronize_model_instances(scene: Any, guid: str) -> int:
    """Reconcile every live instance of *guid* in one loaded scene.

    Returns the number of instance roots changed.  The function is intentionally
    owner-thread only and is called after AssetDatabase publication.
    """
    guid = str(guid or "")
    if not guid or scene is None:
        return 0
    from Infernux.core.mesh import Mesh

    mesh = Mesh.load_guid(guid)
    if mesh is None or not mesh.model_nodes:
        return 0
    changed = 0
    for root in _model_instance_roots(scene, guid):
        if _sync_instance(scene, root, guid, mesh):
            changed += 1
    return changed


def _mark_scene_dirty(scene: Any) -> None:
    from Infernux.engine.interaction import DocumentRegistry
    from Infernux.engine.scene_manager import SceneFileManager

    manager = SceneFileManager.instance()
    if manager is None or manager._is_play_mode():
        return
    document_id = manager.document_id_for_scene(scene)
    if document_id:
        DocumentRegistry.instance().mark_changed(document_id)


def synchronize_loaded_model_instances(guid: str) -> int:
    """Synchronize all currently resident editor scenes for one model GUID."""
    from Infernux.engine.scene_manager import SceneFileManager
    from Infernux.lib import SceneManager

    editor_manager = SceneFileManager.instance()
    if editor_manager is not None and editor_manager._is_play_mode():
        return 0
    manager = SceneManager.instance()
    total = 0
    for index in range(int(manager.scene_count)):
        scene = manager.get_scene_at(index)
        changed = synchronize_model_instances(scene, guid)
        if changed:
            _mark_scene_dirty(scene)
        total += changed
    return total


def synchronize_scene_model_instances(scene: Any) -> int:
    """Reconcile every compound model source referenced by one loaded scene."""
    if scene is None:
        return 0
    from Infernux.engine.scene_manager import SceneFileManager

    editor_manager = SceneFileManager.instance()
    if editor_manager is not None and editor_manager._is_play_mode():
        return 0
    guids = {
        str(renderer.mesh_asset_guid)
        for obj in tuple(scene.get_all_objects() or ())
        for renderer in (obj.get_component("MeshRenderer"),)
        if renderer is not None and renderer.model_node_path and renderer.mesh_asset_guid
    }
    total = 0
    for guid in sorted(guids):
        changed = synchronize_model_instances(scene, guid)
        if changed:
            _mark_scene_dirty(scene)
        total += changed
    return total
