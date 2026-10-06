"""Game object recreation from typed documents for structural undo."""

from __future__ import annotations

from typing import Optional

def _recreate_game_object_from_document(document: dict,
                                        parent_id: Optional[int],
                                        sibling_index: int,
                                        *, scene) -> object:
    return _recreate_game_objects_from_documents([(document, parent_id, sibling_index, scene)])[0]


def _recreate_game_objects_from_documents(entries: list[tuple]) -> list:
    """Restore one owned set, attaching every reference target before callbacks."""
    from infernux.engine.component_restore import (
        _native_object_graph_document,
        _require_clean_pending_queue,
        preflight_scene_python_components,
        publish_prepared_scene_python_components,
    )
    from infernux.engine.scene_manager import SceneFileManager

    sfm = SceneFileManager.instance()
    asset_database = sfm._asset_database if sfm else None
    worlds = {}
    planned = []
    prepared_graphs = []
    created = []
    try:
        for document, parent_id, sibling_index, scene in entries:
            if not scene:
                raise RuntimeError("cannot restore GameObject without its owning scene")
            world_id = int(scene.world_id)
            if world_id not in worlds:
                _require_clean_pending_queue(scene)
                worlds[world_id] = (scene, [])
            parent = scene.find_by_id(parent_id) if parent_id is not None else None
            if parent_id is not None and parent is None:
                raise RuntimeError(f"GameObject undo restore cannot find parent {int(parent_id)}")
            worlds[world_id][1].append(document)
            planned.append((document, parent, sibling_index, scene))

        for scene, documents in worlds.values():
            prepared = preflight_scene_python_components(
                {"objects": documents}, asset_database,
                prefer_loaded_types=True, reference_scene=scene,
            )
            prepared_graphs.append((scene, prepared))

        # Native data is committed for every root before any Python component
        # is attached or activated. Unrelated resident objects retain ownership.
        for document, parent, sibling_index, scene in planned:
            obj = scene.create_game_object("__undo_restore__")
            if obj is None:
                raise RuntimeError("scene rejected GameObject allocation during undo restore")
            created.append((scene, obj))
            if not obj._commit_document(_native_object_graph_document(document), True):
                raise RuntimeError("GameObject document commit failed during undo restore")
            if int(obj.id) != document["id"]:
                raise RuntimeError("GameObject undo restore changed stable identity")
            if parent is not None:
                # The snapshot stores local coordinates, including rotated or
                # scaled parent space. Reparenting must retain those coordinates.
                obj.set_parent(parent, False)
            obj.transform.set_sibling_index(sibling_index)

        for scene, obj in created:
            scene.awake_object(obj)
        attached = []
        for scene, prepared in prepared_graphs:
            attached.extend(publish_prepared_scene_python_components(
                scene, prepared, clear_registries=False, defer_lifecycle=True,
            ))
        for _target, instance, _native in attached:
            instance._call_on_after_deserialize()
        for target, _instance, native in attached:
            target._activate_prepared_py_component(native)
        return [obj for _scene, obj in created]
    except Exception:
        for _scene, prepared in prepared_graphs:
            prepared.discard()
        # Preflight failures own no native mutations. Only drain queues for
        # worlds in which this operation actually allocated restoration roots.
        cleanup_scenes = {int(scene.world_id): scene for scene, _obj in created}
        for scene in cleanup_scenes.values():
            scene.take_pending_py_components()
        for scene, obj in reversed(created):
            if obj:
                scene._remove_game_object_immediately(obj)
        raise
