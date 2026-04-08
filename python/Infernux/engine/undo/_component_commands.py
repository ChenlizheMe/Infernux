"""Component add/remove undo commands."""

from __future__ import annotations

from typing import Any, Optional

from Infernux.debug import Debug
from Infernux.engine.undo._base import UndoCommand
from Infernux.engine.undo._helpers import (
    _get_active_scene, _comp_type_name_of,
    _require_scene_object, _find_live_native_component,
    _invalidate_builtin_wrapper,
    _bump_inspector_structure, _notify_gizmos_scene_changed,
)
from Infernux.engine.undo._snapshots import _get_nth_live_py_component


# -- Helper functions --

def _snapshot_py_fields(py_comp: Any) -> str:
    if py_comp is None or not hasattr(py_comp, '_serialize_fields'):
        return ""
    try:
        return py_comp._serialize_fields()
    except Exception as _exc:
        Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
        return ""


def _snapshot_py_enabled(py_comp: Any) -> bool:
    try:
        return bool(getattr(py_comp, 'enabled', True))
    except Exception:
        return True


def _find_py_ordinal(object_id: int, py_comp: Any) -> int:
    scene = _get_active_scene()
    if not scene:
        return 0
    obj = scene.find_by_id(object_id)
    if obj is None or not hasattr(obj, 'get_py_components'):
        return 0
    target_type = _comp_type_name_of(py_comp)
    target_guid = getattr(py_comp, '_script_guid', '') or ''
    ordinal = 0
    try:
        for current in obj.get_py_components():
            try:
                ct = _comp_type_name_of(current)
                cg = getattr(current, '_script_guid', '') or ''
            except Exception as _exc:
                Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
                continue
            if ct != target_type or cg != target_guid:
                continue
            if current is py_comp:
                return ordinal
            ordinal += 1
    except Exception as _exc:
        Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
    return 0


def _resolve_live_py(obj, type_name: str, script_guid: str,
                     ordinal: int, fallback: Any = None):
    live = _get_nth_live_py_component(obj.id, type_name, ordinal, script_guid)
    if live is not None:
        return live
    if fallback is None:
        return None
    try:
        for current in obj.get_py_components():
            if current is fallback:
                return current
    except Exception as _exc:
        Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
    return None


def _instantiate_py_snapshot(type_name: str, script_guid: str,
                             fields_json: str, enabled: bool,
                             description: str = "") -> Any:
    from Infernux.engine.scene_manager import SceneFileManager
    from Infernux.engine.component_restore import create_component_instance

    sfm = SceneFileManager.instance()
    asset_db = sfm._asset_database if sfm else None
    instance, script_path = create_component_instance(
        script_guid, type_name, asset_database=asset_db)

    if instance is None:
        try:
            from Infernux.components.component import BrokenComponent
            broken = BrokenComponent()
            broken._broken_type_name = type_name
            broken._script_guid = script_guid
            broken._broken_fields_json = fields_json or "{}"
            broken._broken_error = (
                f"Script for '{type_name}' not found during "
                f"{description or 'undo/redo'}")
            broken.enabled = enabled
            return broken
        except Exception as exc:
            Debug.log_error(f"[Undo] Failed to recreate '{type_name}': {exc}")
            return None

    if fields_json:
        try:
            instance._deserialize_fields(fields_json, _skip_on_after_deserialize=True)
        except TypeError:
            instance._deserialize_fields(fields_json)
        except Exception as exc:
            Debug.log_warning(f"[Undo] Deserialize failed for '{type_name}': {exc}")

    try:
        instance.enabled = enabled
    except Exception as _exc:
        Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
    if script_guid:
        try:
            instance._script_guid = script_guid
        except Exception as _exc:
            Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
    return instance


def _snapshot_and_remove_native(object_id: int, type_name: str,
                                label: str) -> str:
    _scene, obj = _require_scene_object(object_id, label)
    live = _find_live_native_component(obj, type_name)
    if live is None:
        raise RuntimeError(f"[Undo] {label}: component '{type_name}' not found")
    json_snap = ""
    if hasattr(live, "serialize"):
        try:
            json_snap = live.serialize()
        except Exception as _exc:
            Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
    obj.remove_component(live)
    _invalidate_builtin_wrapper(live)
    _bump_inspector_structure()
    _notify_gizmos_scene_changed()
    return json_snap


def _add_native_from_snapshot(object_id: int, type_name: str,
                              json_snapshot: Optional[str],
                              label: str) -> None:
    _scene, obj = _require_scene_object(object_id, label)
    result = obj.add_component(type_name)
    if not result:
        raise RuntimeError(f"[Undo] {label}: add '{type_name}' failed")
    if json_snapshot and hasattr(result, "deserialize"):
        try:
            result.deserialize(json_snapshot)
        except Exception as _exc:
            Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
    _bump_inspector_structure()
    _notify_gizmos_scene_changed()


def _snapshot_and_remove_py(object_id: int, type_name: str, script_guid: str,
                            ordinal: int, py_comp_ref: Any, label: str):
    _scene, obj = _require_scene_object(object_id, label)
    live = _resolve_live_py(obj, type_name, script_guid, ordinal, py_comp_ref)
    if live is None:
        raise RuntimeError(f"[Undo] {label}: component not found")
    fields_json = _snapshot_py_fields(live)
    enabled = _snapshot_py_enabled(live)
    obj.remove_py_component(live)
    _bump_inspector_structure()
    return fields_json, enabled, live


def _add_py_from_snapshot(object_id: int, type_name: str, script_guid: str,
                          fields_json, enabled, label: str):
    _scene, obj = _require_scene_object(object_id, label)
    instance = _instantiate_py_snapshot(
        type_name, script_guid, fields_json, enabled, description=label)
    if instance is None:
        raise RuntimeError(f"[Undo] {label}: recreate failed")
    obj.add_py_component(instance)
    if hasattr(instance, '_call_on_after_deserialize'):
        try:
            instance._call_on_after_deserialize()
        except Exception as _exc:
            Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
    _bump_inspector_structure()
    return instance


# -- Command classes --

class AddNativeComponentCommand(UndoCommand):
    """Undo removes the C++ component; redo re-adds from JSON snapshot."""

    def __init__(self, object_id: int, type_name: str, comp_ref: Any = None,
                 description: str = ""):
        super().__init__(description or f"Add {type_name}")
        self._object_id = object_id
        self._type_name = type_name
        self._json_snapshot: Optional[str] = None

    def execute(self) -> None:
        pass

    def undo(self) -> None:
        self._json_snapshot = _snapshot_and_remove_native(
            self._object_id, self._type_name,
            f"AddNative('{self._type_name}').undo")

    def redo(self) -> None:
        _add_native_from_snapshot(
            self._object_id, self._type_name, self._json_snapshot,
            f"AddNative('{self._type_name}').redo")


class RemoveNativeComponentCommand(UndoCommand):
    """Undo re-adds the C++ component from JSON snapshot; redo re-removes."""

    def __init__(self, object_id: int, type_name: str, comp_ref: Any = None,
                 description: str = ""):
        super().__init__(description or f"Remove {type_name}")
        self._object_id = object_id
        self._type_name = type_name
        self._json_snapshot: Optional[str] = None
        if comp_ref is not None and hasattr(comp_ref, "serialize"):
            try:
                self._json_snapshot = comp_ref.serialize()
            except Exception as _exc:
                Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")

    def execute(self) -> None:
        self._do_remove()

    def undo(self) -> None:
        _add_native_from_snapshot(
            self._object_id, self._type_name, self._json_snapshot,
            f"RemoveNative('{self._type_name}').undo")

    def redo(self) -> None:
        self._do_remove()

    def _do_remove(self) -> None:
        self._json_snapshot = _snapshot_and_remove_native(
            self._object_id, self._type_name,
            f"RemoveNative('{self._type_name}')")


class AddPyComponentCommand(UndoCommand):
    """Undo removes the Python component; redo recreates from snapshot."""

    def __init__(self, object_id: int, py_comp_ref: Any,
                 description: str = ""):
        self._type_name_str = getattr(py_comp_ref, 'type_name', 'Script')
        super().__init__(description or f"Add {self._type_name_str}")
        self._object_id = object_id
        self._py_comp_ref = py_comp_ref
        self._script_guid = getattr(py_comp_ref, '_script_guid', '') or ''
        self._fields_json = _snapshot_py_fields(py_comp_ref)
        self._enabled = _snapshot_py_enabled(py_comp_ref)
        self._ordinal = _find_py_ordinal(object_id, py_comp_ref)

    def execute(self) -> None:
        pass

    def undo(self) -> None:
        fj, en, live = _snapshot_and_remove_py(
            self._object_id, self._type_name_str, self._script_guid,
            self._ordinal, self._py_comp_ref,
            f"AddPy('{self._type_name_str}').undo")
        self._fields_json, self._enabled, self._py_comp_ref = fj, en, live

    def redo(self) -> None:
        self._py_comp_ref = _add_py_from_snapshot(
            self._object_id, self._type_name_str, self._script_guid,
            self._fields_json, self._enabled,
            f"AddPy('{self._type_name_str}').redo")


class RemovePyComponentCommand(UndoCommand):
    """Undo recreates the Python component from snapshot; redo re-removes."""

    def __init__(self, object_id: int, py_comp_ref: Any,
                 description: str = ""):
        self._type_name_str = getattr(py_comp_ref, 'type_name', 'Script')
        super().__init__(description or f"Remove {self._type_name_str}")
        self._object_id = object_id
        self._py_comp_ref = py_comp_ref
        self._script_guid = getattr(py_comp_ref, '_script_guid', '') or ''
        self._fields_json = _snapshot_py_fields(py_comp_ref)
        self._enabled = _snapshot_py_enabled(py_comp_ref)
        self._ordinal = _find_py_ordinal(object_id, py_comp_ref)

    def execute(self) -> None:
        self._do_remove()

    def undo(self) -> None:
        self._py_comp_ref = _add_py_from_snapshot(
            self._object_id, self._type_name_str, self._script_guid,
            self._fields_json, self._enabled,
            f"RemovePy('{self._type_name_str}').undo")

    def redo(self) -> None:
        self._do_remove()

    def _do_remove(self) -> None:
        fj, en, live = _snapshot_and_remove_py(
            self._object_id, self._type_name_str, self._script_guid,
            self._ordinal, self._py_comp_ref,
            f"RemovePy('{self._type_name_str}')")
        self._fields_json, self._enabled, self._py_comp_ref = fj, en, live
