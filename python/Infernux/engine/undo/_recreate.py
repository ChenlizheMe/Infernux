"""Game object recreation from JSON (used by Create/Delete undo commands)."""

from __future__ import annotations

import json as _json
import os as _os
from typing import Optional

from Infernux.debug import Debug
from Infernux.engine.undo._helpers import _get_active_scene


def _recreate_game_object_from_json(json_str: str,
                                    parent_id: Optional[int],
                                    sibling_index: int) -> object:
    scene = _get_active_scene()
    if not scene:
        return None

    obj = scene.create_game_object("__undo_restore__")
    if not obj:
        return None

    obj.deserialize(json_str)

    if parent_id is not None:
        parent = scene.find_by_id(parent_id)
        if parent:
            obj.set_parent(parent)

    if getattr(obj, "transform", None):
        obj.transform.set_sibling_index(sibling_index)

    data = _json.loads(json_str)
    _restore_py_components_from_data(scene, data)

    scene.awake_object(obj)
    return obj


def _restore_py_components_from_data(scene, obj_data: dict) -> None:
    obj_id = obj_data.get("id")
    py_comps = obj_data.get("py_components")
    if py_comps and obj_id is not None:
        go = scene.find_by_id(obj_id)
        if go:
            _attach_py_components(go, py_comps)
    for child_data in obj_data.get("children", []):
        _restore_py_components_from_data(scene, child_data)


def _attach_py_components(go, py_comps_json: list) -> None:
    from Infernux.engine.scene_manager import SceneFileManager
    from Infernux.components.script_loader import load_and_create_component
    from Infernux.components.registry import get_type

    sfm = SceneFileManager.instance()
    asset_db = sfm._asset_database if sfm else None

    for pc_json in py_comps_json:
        type_name = pc_json.get("py_type_name", "PyComponent")
        script_guid = pc_json.get("script_guid", "")
        enabled = pc_json.get("enabled", True)
        fields_json = ""
        if "py_fields" in pc_json:
            fields_json = (_json.dumps(pc_json["py_fields"])
                           if isinstance(pc_json["py_fields"], dict)
                           else str(pc_json["py_fields"]))

        script_path = None
        if script_guid and asset_db:
            script_path = asset_db.get_path_from_guid(script_guid)

        instance = None
        if script_path:
            instance = load_and_create_component(
                script_path, asset_database=asset_db, type_name=type_name)
        if instance is None and (not script_path or not _os.path.exists(script_path or "")):
            comp_class = get_type(type_name)
            if comp_class:
                instance = comp_class()

        if instance is None:
            from Infernux.components.component import BrokenComponent
            instance = BrokenComponent()
            instance._broken_type_name = type_name
            instance._script_guid = script_guid
            instance._broken_fields_json = fields_json or "{}"
            instance._broken_error = (
                f"Script not found for '{type_name}' "
                f"(guid={script_guid}) during undo/redo")

        if fields_json:
            instance._deserialize_fields(fields_json)
        instance.enabled = enabled
        go.add_py_component(instance)
        if hasattr(instance, "_call_on_after_deserialize"):
            instance._call_on_after_deserialize()
