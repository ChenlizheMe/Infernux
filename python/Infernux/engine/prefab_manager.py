"""
Prefab system for Infernux.

Handles saving GameObjects as .prefab files and instantiating them back into scenes.
Prefab files contain the serialized JSON from GameObject.serialize(), wrapped in an
envelope with a prefab_version field.
"""

import json
import os
import copy

from Infernux.debug import Debug
from Infernux.engine.component_restore import restore_pending_py_components

PREFAB_EXTENSION = ".prefab"
PREFAB_VERSION = 1
_PREFAB_TEMPLATE_SCENE_NAME = "__InfernuxPrefabTemplateCache__"
_PREFAB_TEMPLATE_CACHE = {}


def _invalidate_prefab_template_cache(file_path: str = None, guid: str = ""):
    keys_to_remove = set()
    if guid:
        keys_to_remove.add(guid)
    if file_path:
        keys_to_remove.add(os.path.normcase(os.path.abspath(file_path)))
    for key in keys_to_remove:
        _PREFAB_TEMPLATE_CACHE.pop(key, None)


def _get_prefab_template_scene():
    from Infernux.lib import SceneManager

    manager = SceneManager.instance()
    scene = manager.get_scene(_PREFAB_TEMPLATE_SCENE_NAME)
    if scene is None:
        scene = manager.create_scene(_PREFAB_TEMPLATE_SCENE_NAME)
        scene.set_playing(False)
    return scene


def _get_file_stamp(file_path: str):
    try:
        stat = os.stat(file_path)
        return (stat.st_mtime_ns, stat.st_size)
    except OSError:
        # Missing file is not an error in the cache-stamp probe path; the
        # caller (cached prefab template lookup) treats None as "no cache
        # entry" and re-reads the prefab from disk if it exists.
        return None


def _load_prefab_template_payload(file_path: str, resolved_guid: str):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            prefab_data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        Debug.log_error(f"Failed to read prefab file: {exc}")
        return None

    root_obj_data = prefab_data.get("root_object")
    if root_obj_data is None:
        Debug.log_error("Invalid prefab file: missing 'root_object'.")
        return None

    file_version = prefab_data.get("prefab_version", 0)
    if file_version > PREFAB_VERSION:
        Debug.log_error(
            f"Prefab '{file_path}' uses version {file_version} but this "
            f"engine only supports up to version {PREFAB_VERSION}. "
            f"Please update Infernux."
        )
        return None
    if file_version < 1:
        Debug.log_warning(
            f"Prefab '{file_path}' has no version tag - treating as v1."
        )

    root_obj_data = copy.deepcopy(root_obj_data)
    _strip_prefab_runtime_fields(root_obj_data)
    if resolved_guid:
        _stamp_prefab_guid(root_obj_data, resolved_guid)

    return root_obj_data


def _get_cached_prefab_template(file_path: str, resolved_guid: str, asset_database=None):
    stamp = _get_file_stamp(file_path)
    if stamp is None:
        Debug.log_warning(f"Prefab file not found: {file_path}")
        return None

    cache_key = resolved_guid or os.path.normcase(os.path.abspath(file_path))
    cached = _PREFAB_TEMPLATE_CACHE.get(cache_key)
    if cached and cached.get("stamp") == stamp:
        template = cached.get("template")
        if template is not None:
            return template

    template_payload = _load_prefab_template_payload(file_path, resolved_guid)
    if template_payload is None:
        return None

    template_scene = _get_prefab_template_scene()

    old_template = cached.get("template") if cached else None
    if old_template is not None:
        try:
            template_scene.destroy_game_object(old_template)
            template_scene.process_pending_destroys()
        except Exception as exc:
            Debug.log_suppressed("prefab_manager._get_cached_prefab_template.destroy_old", exc)

    template = template_scene.instantiate_from_json(json.dumps(template_payload), None)
    if template is None:
        Debug.log_error("Failed to build cached prefab template from JSON.")
        return None

    try:
        restore_pending_py_components(template_scene, asset_database=asset_database)
    except Exception as exc:
        Debug.log_error(f"Failed to restore cached prefab Python components: {exc}")
        template_scene.destroy_game_object(template)
        template_scene.process_pending_destroys()
        return None

    _PREFAB_TEMPLATE_CACHE[cache_key] = {
        "stamp": stamp,
        "template": template,
    }
    return template


def _strip_prefab_runtime_fields(obj_data: dict):
    if not isinstance(obj_data, dict):
        return

    for comp in obj_data.get("components", []) or []:
        if isinstance(comp, dict):
            comp.pop("component_id", None)
            comp.pop("instance_guid", None)

    for py_comp in obj_data.get("py_components", []) or []:
        if not isinstance(py_comp, dict):
            continue
        py_comp.pop("component_id", None)
        py_comp.pop("instance_guid", None)
        py_fields = py_comp.get("py_fields")
        if isinstance(py_fields, dict):
            py_fields.pop("__component_id__", None)

    for child in obj_data.get("children", []) or []:
        _strip_prefab_runtime_fields(child)


def read_prefab_source_canvas(file_path: str = None, guid: str = None,
                              asset_database=None) -> str:
    """Return the ``source_canvas_name`` stored in a prefab, or ``""``."""
    if not file_path and guid and asset_database:
        file_path = asset_database.get_path_from_guid(guid)
    if not file_path or not os.path.isfile(file_path):
        return ""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("source_canvas_name", "")
    except Exception:
        return ""


def save_prefab(game_object, file_path: str, asset_database=None,
               source_canvas_name: str = "") -> bool:
    """Serialize a GameObject hierarchy to a .prefab file.

    Returns True on success, False on failure.
    """
    if game_object is None:
        Debug.log_warning("Cannot save prefab: no GameObject provided.")
        return False

    if not file_path.lower().endswith(PREFAB_EXTENSION):
        file_path += PREFAB_EXTENSION

    try:
        go_json_str = game_object.serialize()
        go_data = json.loads(go_json_str)
    except Exception as exc:
        Debug.log_error(f"Failed to serialize GameObject for prefab: {exc}")
        return False

    # Strip any existing prefab linkage and runtime-only IDs from the saved template.
    _strip_prefab_fields(go_data)
    _strip_prefab_runtime_fields(go_data)

    prefab_data = {
        "prefab_version": PREFAB_VERSION,
        "root_object": go_data,
    }
    if source_canvas_name:
        prefab_data["source_canvas_name"] = source_canvas_name

    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(prefab_data, f, indent=2, ensure_ascii=False)
    except OSError as exc:
        Debug.log_error(f"Failed to write prefab file: {exc}")
        return False

    if asset_database:
        try:
            guid = asset_database.import_asset(file_path)
            Debug.log_internal(f"Registered prefab: {os.path.basename(file_path)} -> {guid}")
            _invalidate_prefab_template_cache(file_path, guid)
        except Exception as exc:
            Debug.log_warning(f"Failed to register prefab in AssetDatabase: {exc}")
            _invalidate_prefab_template_cache(file_path, "")
    else:
        _invalidate_prefab_template_cache(file_path, "")

    Debug.log_internal(f"Prefab saved: {file_path}")
    return True


def instantiate_prefab(file_path: str = None, guid: str = None,
                       scene=None, parent=None, asset_database=None):
    """Instantiate a prefab into the active scene.

    Supply either *file_path* or *guid* (GUID is resolved via asset_database).
    Returns the root GameObject, or None on failure.
    """
    # Resolve path from GUID if needed
    resolved_guid = guid or ""
    if not file_path and guid and asset_database:
        file_path = asset_database.get_path_from_guid(guid)

    if not file_path or not os.path.isfile(file_path):
        Debug.log_warning(f"Prefab file not found: {file_path}")
        return None

    # If we have a path but no GUID, try to resolve GUID from the asset database
    if not resolved_guid and asset_database:
        try:
            resolved_guid = asset_database.get_guid_from_path(file_path) or ""
        except Exception:
            resolved_guid = ""

    if scene is None:
        from Infernux.lib import SceneManager
        scene = SceneManager.instance().get_active_scene()
    if scene is None:
        Debug.log_warning("No active scene — cannot instantiate prefab.")
        return None

    template = _get_cached_prefab_template(file_path, resolved_guid, asset_database)
    if template is None:
        return None

    # Repeated prefab instantiation now uses native C++ clone from a cached template.
    new_obj = scene.instantiate_game_object(template, parent)
    if new_obj is None:
        Debug.log_error("Failed to instantiate prefab from cached template.")
        return None

    # Restore Python components that were collected as pending during native clone.
    try:
        restore_pending_py_components(scene, asset_database=asset_database)
    except Exception as exc:
        Debug.log_error(f"Failed to restore prefab Python components: {exc}")

    return new_obj


def _stamp_prefab_guid(obj_data: dict, guid: str, is_root: bool = True):
    """Recursively stamp prefab_guid (and prefab_root on root) into JSON data."""
    obj_data["prefab_guid"] = guid
    if is_root:
        obj_data["prefab_root"] = True
    for child in obj_data.get("children", []):
        _stamp_prefab_guid(child, guid, is_root=False)


def _strip_prefab_fields(obj_data: dict):
    """Recursively remove prefab_guid/prefab_root so the template is clean."""
    obj_data.pop("prefab_guid", None)
    obj_data.pop("prefab_root", None)
    for child in obj_data.get("children", []):
        _strip_prefab_fields(child)
