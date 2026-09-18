"""Synchronize live model instances after an external source reimport.

Imported models are references, not flattened scene content.  A scene keeps
the instance hierarchy and its authored transforms while the source asset owns
the node list and geometry.  This module reconciles the two at the owner
safe-point after a model publication.
"""

from __future__ import annotations

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


def _model_instance_roots(scene: Any, guid: str) -> tuple[Any, ...]:
    """Find model containers from their persisted renderer node identities."""
    roots: dict[int, Any] = {}
    for obj in tuple(scene.get_all_objects() or ()):
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


def _destroy_stale_geometry(
    scene: Any, root: Any, guid: str, source: dict[tuple[str, ...], dict]
) -> bool:
    paths = _object_path_map(root)
    stale: dict[int, Any] = {}
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
    changed = _destroy_stale_geometry(scene, root, guid, source)
    existing = _object_path_map(root)
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
