"""
Undo/Redo system for Infernux editor.

Design principles (modelled after Unity's Undo architecture):
- Generic snapshot-based change detection via InspectorUndoTracker
  (one universal tracker for ALL component types — no per-type commands)
- Structural commands for object lifecycle only
  (create / delete / reparent / add-remove component)
- Automatic merge for rapid consecutive edits (slider dragging)
- Save-point tracking for dirty state synchronisation
- Play mode isolation (stack cleared on play/stop)
- Exception-safe undo/redo (stack preserved on failure)

Usage::

    from Infernux.engine.undo import UndoManager, SetPropertyCommand

    mgr = UndoManager.instance()
    mgr.record(SetPropertyCommand(obj, "position", old, new, "Move"))
    mgr.undo()
    mgr.redo()
"""

from __future__ import annotations

import json as _json
import os as _os
import time as _time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Optional, List, Callable

from Infernux.debug import Debug


# ═══════════════════════════════════════════════════════════════════════════
# Base command
# ═══════════════════════════════════════════════════════════════════════════

class UndoCommand(ABC):
    """Base class for all undoable editor commands."""

    supports_redo: bool = True
    marks_dirty: bool = True
    _is_property_edit: bool = False

    def __init__(self, description: str = ""):
        self.description: str = description
        self.timestamp: float = _time.time()

    @abstractmethod
    def execute(self) -> None: ...

    @abstractmethod
    def undo(self) -> None: ...

    def redo(self) -> None:
        self.execute()

    def can_merge(self, other: UndoCommand) -> bool:
        return False

    def merge(self, other: UndoCommand) -> None:
        pass


class CompoundCommand(UndoCommand):
    """Group multiple commands into a single undo/redo unit."""

    def __init__(self, commands: List[UndoCommand], description: str = ""):
        super().__init__(description or (commands[0].description if commands else "Compound"))
        self._commands: List[UndoCommand] = list(commands)
        self.supports_redo = all(c.supports_redo for c in commands)

    def execute(self) -> None:
        for cmd in self._commands:
            cmd.execute()

    def undo(self) -> None:
        for cmd in reversed(self._commands):
            cmd.undo()

    def redo(self) -> None:
        for cmd in self._commands:
            cmd.redo()


# ═══════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════

def _snapshot_value(value: Any) -> Any:
    """Deep-copy mutable containers; immutable types pass through."""
    if isinstance(value, (list, dict)):
        import copy
        try:
            return copy.deepcopy(value)
        except Exception:
            return list(value) if isinstance(value, list) else dict(value)
    return value


def _get_active_scene():
    from Infernux.lib import SceneManager
    return SceneManager.instance().get_active_scene()


def _game_object_id_of(target: Any) -> int:
    goid = getattr(target, 'game_object_id', None) or 0
    if not goid:
        go = getattr(target, 'game_object', None)
        if go is not None:
            goid = getattr(go, 'id', 0) or 0
    if not goid:
        goid = getattr(target, 'id', 0) or 0
    return goid


def _comp_type_name_of(target: Any) -> str:
    tn = getattr(target, 'type_name', None)
    if tn:
        return str(tn)
    if (getattr(target, 'id', 0)
            and not getattr(target, 'component_id', 0)
            and getattr(target, 'game_object', None) is None):
        return "GameObject"
    return type(target).__name__


def _stable_target_id(target: Any) -> int:
    for attr in ("component_id", "id"):
        val = getattr(target, attr, None)
        if val is not None and val != 0:
            return int(val)
    return id(target)


def _resolve_target(stored_ref: Any, game_object_id: int,
                    comp_type_name: str) -> Any:
    """Re-fetch a live component from the scene graph.

    After delete+undo cycles pybind11 wrappers may reference freed C++
    memory.  This resolves a fresh reference by game_object_id and type
    name.  Returns *stored_ref* when resolution is not needed, or
    *None* when the target cannot be found.
    """
    if not game_object_id or not comp_type_name:
        return stored_ref
    scene = _get_active_scene()
    if not scene:
        return None
    obj = scene.find_by_id(game_object_id)
    if obj is None:
        return None
    if comp_type_name == "GameObject":
        return obj
    if comp_type_name == "Transform":
        return getattr(obj, "transform", None)
    try:
        live = obj.get_component(comp_type_name)
        if live is not None:
            return live
    except Exception as _exc:
        Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
        pass
    try:
        for pc in obj.get_py_components():
            if type(pc).__name__ == comp_type_name:
                return pc
    except Exception as _exc:
        Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
        pass
    return None


# Backward-compat alias (used by tests)
_resolve_live_ref = _resolve_target


def _find_live_native_component(obj, type_name: str):
    if hasattr(obj, 'get_component'):
        try:
            c = obj.get_component(type_name)
            if c is not None:
                return c
        except Exception as _exc:
            Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
            pass
    try:
        for c in obj.get_components():
            if getattr(c, 'type_name', None) == type_name:
                return c
    except Exception as _exc:
        Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
        pass
    return None


def _get_current_selection_ids() -> List[int]:
    try:
        from Infernux.engine.ui.selection_manager import SelectionManager
        sel = SelectionManager.instance()
        if sel:
            return sel.get_ids()
    except Exception as _exc:
        Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
        pass
    return []


def _bump_inspector_structure():
    """Bump Inspector component-structure version so its cache invalidates."""
    try:
        from Infernux.engine.ui.inspector_support import bump_component_structure_version
        bump_component_structure_version()
    except ImportError as _exc:
        Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
        pass


def _bump_inspector_values():
    """Bump the coarse Inspector value generation after editor mutations."""
    try:
        from Infernux.engine.ui.inspector_support import bump_inspector_value_generation
        bump_inspector_value_generation()
    except ImportError as _exc:
        Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
        pass


def _require_scene_object(object_id: int, label: str):
    """Return (scene, obj) or raise RuntimeError with *label* context."""
    scene = _get_active_scene()
    if not scene:
        raise RuntimeError(f"[Undo] {label}: no scene")
    obj = scene.find_by_id(object_id)
    if not obj:
        raise RuntimeError(f"[Undo] {label}: object {object_id} not found")
    return scene, obj


def _notify_gizmos_scene_changed():
    from Infernux.gizmos.collector import notify_scene_changed
    notify_scene_changed()


def _invalidate_builtin_wrapper(comp_ref):
    try:
        comp_id = comp_ref.component_id
    except Exception as _exc:
        Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
        return
    from Infernux.components.builtin_component import BuiltinComponent
    wrapper = BuiltinComponent._wrapper_cache.get(comp_id)
    if wrapper is not None:
        wrapper._invalidate_native_binding()


def _invalidate_builtin_wrappers_for_tree(obj):
    try:
        from Infernux.components.builtin_component import BuiltinComponent
    except ImportError as _exc:
        Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
        return
    cache = BuiltinComponent._wrapper_cache
    pending = [obj]
    while pending:
        current = pending.pop()
        if current is None:
            continue
        try:
            for comp in current.get_components():
                comp_id = getattr(comp, "component_id", 0) or 0
                if comp_id:
                    wrapper = cache.get(comp_id)
                    if wrapper is not None:
                        wrapper._invalidate_native_binding()
        except Exception as _exc:
            Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
            pass
        try:
            pending.extend(current.get_children())
        except Exception as _exc:
            Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
            pass


def _destroy_game_object_immediately(scene, obj):
    if scene is None or obj is None:
        return
    _invalidate_builtin_wrappers_for_tree(obj)
    scene.destroy_game_object(obj)
    if hasattr(scene, "process_pending_destroys"):
        try:
            scene.process_pending_destroys()
        except Exception as _exc:
            Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
            pass
    _bump_inspector_structure()
    _notify_gizmos_scene_changed()


def _invalidate_canvas_caches(go):
    if go is None:
        return
    from Infernux.ui import UICanvas
    cur = go
    while cur is not None:
        for comp in cur.get_py_components():
            if isinstance(comp, UICanvas):
                comp.invalidate_element_cache()
                return
        cur = cur.get_parent()


def _preserve_ui_world_position(obj, new_parent):
    """Adjust UI element local x/y to preserve canvas-space position."""
    from Infernux.ui.inx_ui_screen_component import InxUIScreenComponent, clear_rect_cache
    from Infernux.ui import UICanvas

    ui_comp = None
    for comp in obj.get_py_components():
        if isinstance(comp, InxUIScreenComponent):
            ui_comp = comp
            break
    if ui_comp is None:
        return

    def _find_canvas(go):
        while go is not None:
            for c in go.get_py_components():
                if isinstance(c, UICanvas):
                    return c
            go = go.get_parent()
        return None

    old_canvas = _find_canvas(obj.get_parent() or obj)
    if old_canvas is None:
        return
    old_cw = float(old_canvas.reference_width)
    old_ch = float(old_canvas.reference_height)
    old_abs_x, old_abs_y, _w, _h = ui_comp.get_rect(old_cw, old_ch)

    new_canvas = _find_canvas(new_parent) if new_parent is not None else None
    ncw = float(new_canvas.reference_width) if new_canvas is not None else old_cw
    nch = float(new_canvas.reference_height) if new_canvas is not None else old_ch

    if new_parent is not None:
        new_parent_ui = None
        for c in new_parent.get_py_components():
            if isinstance(c, InxUIScreenComponent):
                new_parent_ui = c
                break
        if new_parent_ui is not None:
            npx, npy, npw, nph = new_parent_ui.get_rect(ncw, nch)
        else:
            npx, npy, npw, nph = 0.0, 0.0, ncw, nch
    else:
        npx, npy, npw, nph = 0.0, 0.0, ncw, nch

    anchor_x, anchor_y = ui_comp._anchor_origin(npw, nph)
    ui_comp.x = old_abs_x - npx - anchor_x
    ui_comp.y = old_abs_y - npy - anchor_y
    clear_rect_cache(-1)


# ═══════════════════════════════════════════════════════════════════════════
# Generic property commands
# ═══════════════════════════════════════════════════════════════════════════

class SetPropertyCommand(UndoCommand):
    """Generic property-edit via ``setattr(target, name, value)``.

    Works uniformly for C++ pybind11 properties and Python fields.
    Consecutive rapid edits to the same target+property within
    ``MERGE_WINDOW`` are merged into a single undo entry.
    """

    _is_property_edit = True
    MERGE_WINDOW: float = 0.3

    def __init__(self, target: Any, prop_name: str,
                 old_value: Any, new_value: Any,
                 description: str = ""):
        super().__init__(description or f"Set {prop_name}")
        self._target = target
        self._prop_name = prop_name
        self._old_value = _snapshot_value(old_value)
        self._new_value = _snapshot_value(new_value)
        self._target_id: int = _stable_target_id(target)
        self._game_object_id: int = _game_object_id_of(target)
        self._comp_type_name: str = _comp_type_name_of(target) if self._game_object_id else ""

    def _live(self):
        return _resolve_target(self._target, self._game_object_id, self._comp_type_name)

    def execute(self) -> None:
        target = self._live()
        if target is None:
            target = self._target
        setattr(target, self._prop_name, self._new_value)

    def undo(self) -> None:
        target = self._live()
        if target is None:
            from Infernux.debug import Debug
            Debug.log_error(
                f"[Undo] SetProperty('{self._prop_name}').undo: target not found "
                f"(go={self._game_object_id}, type={self._comp_type_name})"
            )
            return
        setattr(target, self._prop_name, self._old_value)

    def redo(self) -> None:
        target = self._live()
        if target is None:
            from Infernux.debug import Debug
            Debug.log_error(
                f"[Undo] SetProperty('{self._prop_name}').redo: target not found "
                f"(go={self._game_object_id}, type={self._comp_type_name})"
            )
            return
        setattr(target, self._prop_name, self._new_value)

    def can_merge(self, other: UndoCommand) -> bool:
        if not isinstance(other, SetPropertyCommand):
            return False
        return (self._target_id == other._target_id
                and self._prop_name == other._prop_name
                and (other.timestamp - self.timestamp) <= self.MERGE_WINDOW)

    def merge(self, other: SetPropertyCommand) -> None:
        self._new_value = _snapshot_value(other._new_value)
        self.timestamp = other.timestamp


# Backward-compatible alias — identical behaviour to SetPropertyCommand
BuiltinPropertyCommand = SetPropertyCommand


class GenericComponentCommand(UndoCommand):
    """Undo/redo for a component edited via serialize/deserialize JSON."""

    _is_property_edit = True
    MERGE_WINDOW: float = 0.3

    def __init__(self, comp: Any, old_json: str, new_json: str,
                 description: str = ""):
        super().__init__(description or f"Edit {getattr(comp, 'type_name', 'Component')}")
        self._comp = comp
        self._old_json = old_json
        self._new_json = new_json
        self._comp_id: int = getattr(comp, "component_id", id(comp))
        self._game_object_id: int = _game_object_id_of(comp)
        self._comp_type_name: str = _comp_type_name_of(comp)

    def _live(self):
        return _resolve_target(self._comp, self._game_object_id, self._comp_type_name)

    def execute(self) -> None:
        self._comp.deserialize(self._new_json)

    def undo(self) -> None:
        comp = self._live()
        if comp is None:
            from Infernux.debug import Debug
            Debug.log_error(
                f"[Undo] GenericComponent('{self._comp_type_name}').undo: not found")
            return
        comp.deserialize(self._old_json)

    def redo(self) -> None:
        comp = self._live()
        if comp is None:
            from Infernux.debug import Debug
            Debug.log_error(
                f"[Undo] GenericComponent('{self._comp_type_name}').redo: not found")
            return
        comp.deserialize(self._new_json)

    def can_merge(self, other: UndoCommand) -> bool:
        if not isinstance(other, GenericComponentCommand):
            return False
        return (self._comp_id == other._comp_id
                and (other.timestamp - self.timestamp) <= self.MERGE_WINDOW)

    def merge(self, other: GenericComponentCommand) -> None:
        self._new_json = other._new_json
        self.timestamp = other.timestamp


class MaterialJsonCommand(UndoCommand):
    """Undo/redo for material asset edits (deserialize + save to disk)."""

    _is_property_edit = False
    MERGE_WINDOW: float = 0.3
    marks_dirty: bool = False   # materials save to their own files

    def __init__(self, material: Any, old_json: str, new_json: str,
                 description: str = "Edit Material",
                 refresh_callback: Optional[Callable[[Any], None]] = None,
                 edit_key: str = ""):
        super().__init__(description)
        self._material = material
        self._old_json = old_json
        self._new_json = new_json
        self._refresh_callback = refresh_callback
        self._material_id = self._stable_id(material)
        self._edit_key = edit_key or ""

    @staticmethod
    def _stable_id(material: Any) -> int:
        guid = getattr(material, "guid", "")
        if guid:
            return hash(("material-guid", guid))
        fp = getattr(material, "file_path", "")
        if fp:
            return hash(("material-file", fp))
        return id(material)

    def execute(self) -> None:
        self._apply(self._new_json)

    def undo(self) -> None:
        self._apply(self._old_json)

    def redo(self) -> None:
        self._apply(self._new_json)

    def can_merge(self, other: UndoCommand) -> bool:
        if not isinstance(other, MaterialJsonCommand):
            return False
        return (self._material_id == other._material_id
                and self._edit_key == other._edit_key
                and (other.timestamp - self.timestamp) <= self.MERGE_WINDOW)

    def merge(self, other: MaterialJsonCommand) -> None:
        self._new_json = other._new_json
        self.timestamp = other.timestamp

    def _apply(self, json_str: str) -> None:
        self._material.deserialize(json_str)
        save = getattr(self._material, "save", None)
        save_ok = False
        if callable(save):
            result = save()
            save_ok = bool(result) if result is not None else True
        if save_ok:
            fp = getattr(self._material, "file_path", "") or ""
            if fp:
                try:
                    from Infernux.core.assets import AssetManager
                    AssetManager.on_material_saved(fp)
                except Exception as _exc:
                    Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
                    pass
        if self._refresh_callback:
            self._refresh_callback(self._material)


class SetMaterialSlotCommand(UndoCommand):
    """Undo/redo for MeshRenderer material-slot assignment.

    Unlike ``SetPropertyCommand``, this calls ``renderer.set_material(slot, guid)``
    directly — because MeshRenderer has no pybind11 *property* named
    ``material_slot_N``, so ``setattr`` would silently create a Python
    instance attribute instead of touching C++.
    """

    _is_property_edit = True
    MERGE_WINDOW: float = 0.3

    def __init__(self, renderer, slot: int, old_guid: str, new_guid: str,
                 description: str = ""):
        super().__init__(description or f"Set Material Slot {slot}")
        self._renderer = renderer
        self._slot = slot
        self._old_guid = old_guid or ""
        self._new_guid = new_guid or ""
        self._game_object_id: int = _game_object_id_of(renderer)
        self._comp_type_name: str = _comp_type_name_of(renderer) if self._game_object_id else ""

    def _live(self):
        return _resolve_target(self._renderer, self._game_object_id, self._comp_type_name)

    def execute(self) -> None:
        target = self._live() or self._renderer
        target.set_material(self._slot, self._new_guid)

    def undo(self) -> None:
        target = self._live()
        if target is None:
            from Infernux.debug import Debug
            Debug.log_error(
                f"[Undo] SetMaterialSlot({self._slot}).undo: renderer not found "
                f"(go={self._game_object_id}, type={self._comp_type_name})")
            return
        target.set_material(self._slot, self._old_guid)

    def redo(self) -> None:
        target = self._live()
        if target is None:
            from Infernux.debug import Debug
            Debug.log_error(
                f"[Undo] SetMaterialSlot({self._slot}).redo: renderer not found "
                f"(go={self._game_object_id}, type={self._comp_type_name})")
            return
        target.set_material(self._slot, self._new_guid)

    def can_merge(self, other: UndoCommand) -> bool:
        if not isinstance(other, SetMaterialSlotCommand):
            return False
        return (self._game_object_id == other._game_object_id
                and self._comp_type_name == other._comp_type_name
                and self._slot == other._slot
                and (other.timestamp - self.timestamp) <= self.MERGE_WINDOW)

    def merge(self, other: SetMaterialSlotCommand) -> None:
        self._new_guid = other._new_guid
        self.timestamp = other.timestamp


# ═══════════════════════════════════════════════════════════════════════════
# Structural commands — object lifecycle
# ═══════════════════════════════════════════════════════════════════════════

class CreateGameObjectCommand(UndoCommand):
    """Undo destroys the object; redo recreates from JSON snapshot."""

    _selection_restore_fn: Optional[Callable[[List[int]], None]] = None

    def __init__(self, object_id: int, description: str = "Create GameObject"):
        super().__init__(description)
        self._object_id = object_id
        self._snapshot_json: Optional[str] = None
        self._parent_id: Optional[int] = None
        self._sibling_index: int = 0
        self._post_create_ids: List[int] = _get_current_selection_ids()

    def execute(self) -> None:
        pass  # object already created before record()

    def undo(self) -> None:
        scene = _get_active_scene()
        if scene:
            obj = scene.find_by_id(self._object_id)
            if obj:
                self._snapshot_json = obj.serialize()
                parent = obj.get_parent()
                self._parent_id = parent.id if parent else None
                t = getattr(obj, "transform", None)
                self._sibling_index = t.get_sibling_index() if t else 0
                _destroy_game_object_immediately(scene, obj)
        fn = type(self)._selection_restore_fn
        if fn:
            fn([])

    def redo(self) -> None:
        if self._snapshot_json:
            _recreate_game_object_from_json(
                self._snapshot_json, self._parent_id, self._sibling_index)
            _bump_inspector_structure()
            _notify_gizmos_scene_changed()
            fn = type(self)._selection_restore_fn
            if fn and self._post_create_ids:
                fn(self._post_create_ids)


class DeleteGameObjectCommand(UndoCommand):
    """Undo recreates from JSON snapshot; redo re-destroys."""

    _selection_restore_fn: Optional[Callable[[List[int]], None]] = None

    def __init__(self, object_id: int, description: str = "Delete GameObject"):
        super().__init__(description)
        self._object_id = object_id
        self._snapshot_json: Optional[str] = None
        self._parent_id: Optional[int] = None
        self._sibling_index: int = 0
        self._pre_delete_selection_ids: List[int] = []

        scene = _get_active_scene()
        if scene:
            obj = scene.find_by_id(object_id)
            if obj:
                self._snapshot_json = obj.serialize()
                parent = obj.get_parent()
                self._parent_id = parent.id if parent else None
                t = getattr(obj, "transform", None)
                self._sibling_index = t.get_sibling_index() if t else 0
        self._pre_delete_selection_ids = _get_current_selection_ids()

    def execute(self) -> None:
        scene = _get_active_scene()
        if scene:
            obj = scene.find_by_id(self._object_id)
            if obj:
                _destroy_game_object_immediately(scene, obj)

    def undo(self) -> None:
        if self._snapshot_json:
            _recreate_game_object_from_json(
                self._snapshot_json, self._parent_id, self._sibling_index)
            _bump_inspector_structure()
            _notify_gizmos_scene_changed()
            fn = type(self)._selection_restore_fn
            if fn and self._pre_delete_selection_ids:
                fn(self._pre_delete_selection_ids)

    def redo(self) -> None:
        scene = _get_active_scene()
        if scene:
            obj = scene.find_by_id(self._object_id)
            if obj:
                self._snapshot_json = obj.serialize()
                _destroy_game_object_immediately(scene, obj)
        fn = type(self)._selection_restore_fn
        if fn:
            fn([])


class ReparentCommand(UndoCommand):
    """Undo/redo changing the parent of a GameObject."""

    def __init__(self, object_id: int,
                 old_parent_id: Optional[int],
                 new_parent_id: Optional[int],
                 description: str = "Reparent"):
        super().__init__(description)
        self._object_id = object_id
        self._old_parent_id = old_parent_id
        self._new_parent_id = new_parent_id

    def execute(self) -> None:
        self._apply(self._new_parent_id)

    def undo(self) -> None:
        self._apply(self._old_parent_id)

    def redo(self) -> None:
        self._apply(self._new_parent_id)

    def _apply(self, parent_id: Optional[int]) -> None:
        scene = _get_active_scene()
        if not scene:
            return
        obj = scene.find_by_id(self._object_id)
        if not obj:
            return
        new_parent = scene.find_by_id(parent_id) if parent_id is not None else None
        old_parent = obj.get_parent()
        _preserve_ui_world_position(obj, new_parent)
        obj.set_parent(new_parent)
        _invalidate_canvas_caches(old_parent)
        _invalidate_canvas_caches(new_parent)


class MoveGameObjectCommand(UndoCommand):
    """Undo/redo moves that change both parent and sibling order."""

    def __init__(self, object_id: int,
                 old_parent_id: Optional[int], new_parent_id: Optional[int],
                 old_sibling_index: int, new_sibling_index: int,
                 description: str = "Move In Hierarchy"):
        super().__init__(description)
        self._object_id = object_id
        self._old_parent_id = old_parent_id
        self._new_parent_id = new_parent_id
        self._old_sibling_index = int(old_sibling_index)
        self._new_sibling_index = int(new_sibling_index)

    def execute(self) -> None:
        self._apply(self._new_parent_id, self._new_sibling_index)

    def undo(self) -> None:
        self._apply(self._old_parent_id, self._old_sibling_index)

    def redo(self) -> None:
        self._apply(self._new_parent_id, self._new_sibling_index)

    def _apply(self, parent_id: Optional[int], sibling_index: int) -> None:
        scene = _get_active_scene()
        if not scene:
            return
        obj = scene.find_by_id(self._object_id)
        if not obj:
            return
        parent = scene.find_by_id(parent_id) if parent_id is not None else None
        current_parent = obj.get_parent()
        if current_parent is not parent:
            _preserve_ui_world_position(obj, parent)
            obj.set_parent(parent)
            _invalidate_canvas_caches(current_parent)
            _invalidate_canvas_caches(parent)
        transform = getattr(obj, "transform", None)
        if transform is not None:
            transform.set_sibling_index(max(0, int(sibling_index)))


# ═══════════════════════════════════════════════════════════════════════════
# Component add / remove — generic for both native (C++) and Python
# ═══════════════════════════════════════════════════════════════════════════

def _snapshot_and_remove_native(object_id: int, type_name: str,
                                label: str) -> str:
    """Find, snapshot-serialize, remove a native component. Return JSON."""
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
            pass
    obj.remove_component(live)
    _invalidate_builtin_wrapper(live)
    _bump_inspector_structure()
    _notify_gizmos_scene_changed()
    return json_snap


def _add_native_from_snapshot(object_id: int, type_name: str,
                              json_snapshot: Optional[str],
                              label: str) -> None:
    """Add a native component and optionally deserialize from snapshot."""
    _scene, obj = _require_scene_object(object_id, label)
    result = obj.add_component(type_name)
    if not result:
        raise RuntimeError(f"[Undo] {label}: add '{type_name}' failed")
    if json_snapshot and hasattr(result, "deserialize"):
        try:
            result.deserialize(json_snapshot)
        except Exception as _exc:
            Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
            pass
    _bump_inspector_structure()
    _notify_gizmos_scene_changed()


def _snapshot_and_remove_py(object_id: int, type_name: str, script_guid: str,
                            ordinal: int, py_comp_ref: Any,
                            label: str):
    """Find live py component, snapshot fields+enabled, remove it.

    Returns ``(fields_json, enabled, live_ref)``.
    """
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
    """Instantiate py component from snapshot, add to object. Returns instance."""
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
            pass
    _bump_inspector_structure()
    return instance


class AddNativeComponentCommand(UndoCommand):
    """Undo removes the C++ component; redo re-adds from JSON snapshot."""

    def __init__(self, object_id: int, type_name: str, comp_ref: Any = None,
                 description: str = ""):
        super().__init__(description or f"Add {type_name}")
        self._object_id = object_id
        self._type_name = type_name
        self._json_snapshot: Optional[str] = None

    def execute(self) -> None:
        pass  # already added before record()

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
                pass

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


# -- Python component helpers --

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
        pass
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
        pass
    return None


def _instantiate_py_snapshot(type_name: str, script_guid: str,
                             fields_json: str, enabled: bool,
                             description: str = "") -> Any:
    """Recreate a Python component instance from stored snapshot data."""
    from Infernux.debug import Debug
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
                f"{description or 'undo/redo'}"
            )
            broken.enabled = enabled
            return broken
        except Exception as exc:
            Debug.log_error(
                f"[Undo] Failed to recreate '{type_name}': {exc}")
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
        pass
    if script_guid:
        try:
            instance._script_guid = script_guid
        except Exception as _exc:
            Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
            pass
    return instance


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
        pass  # already added before record()

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


# ═══════════════════════════════════════════════════════════════════════════
# Selection
# ═══════════════════════════════════════════════════════════════════════════

class SelectionCommand(UndoCommand):
    """Record a selection change.  Does not mark the scene dirty."""

    marks_dirty: bool = False

    def __init__(self, old_ids: List[int], new_ids: List[int],
                 apply_fn: Callable[[List[int]], None],
                 description: str = ""):
        super().__init__(description or "Change Selection")
        self._old_ids = list(old_ids)
        self._new_ids = list(new_ids)
        self._apply_fn = apply_fn

    def execute(self) -> None:
        pass  # selection already applied before record()

    def undo(self) -> None:
        self._apply_fn(self._old_ids)

    def redo(self) -> None:
        self._apply_fn(self._new_ids)


class EditorSelectionCommand(UndoCommand):
    """Record editor selection state across hierarchy and project panels."""

    marks_dirty: bool = False

    def __init__(self,
                 old_object_ids: List[int], old_file_path: str,
                 new_object_ids: List[int], new_file_path: str,
                 apply_fn: Callable[[List[int], str], None],
                 description: str = ""):
        super().__init__(description or "Change Selection")
        self._old_object_ids = list(old_object_ids)
        self._old_file_path = old_file_path or ""
        self._new_object_ids = list(new_object_ids)
        self._new_file_path = new_file_path or ""
        self._apply_fn = apply_fn

    def execute(self) -> None:
        pass

    def undo(self) -> None:
        self._apply_fn(self._old_object_ids, self._old_file_path)

    def redo(self) -> None:
        self._apply_fn(self._new_object_ids, self._new_file_path)


class PrefabModeCommand(UndoCommand):
    """Undoable enter/exit transition for Prefab Mode."""

    marks_dirty: bool = False

    def __init__(self, prefab_path: str, enter_mode: bool):
        action = "Enter Prefab Mode" if enter_mode else "Exit Prefab Mode"
        super().__init__(action)
        self._prefab_path = prefab_path or ""
        self._enter_mode = bool(enter_mode)

    def execute(self) -> None:
        from Infernux.engine.scene_manager import SceneFileManager
        sfm = SceneFileManager.instance()
        if not sfm:
            return
        if self._enter_mode:
            sfm.open_prefab_mode(self._prefab_path, preserve_undo_history=True)
        else:
            sfm._do_exit_prefab_mode(preserve_undo_history=True)

    def undo(self) -> None:
        from Infernux.engine.scene_manager import SceneFileManager
        sfm = SceneFileManager.instance()
        if not sfm:
            return
        if self._enter_mode:
            sfm._do_exit_prefab_mode(preserve_undo_history=True)
        else:
            sfm.open_prefab_mode(self._prefab_path, preserve_undo_history=True)

    def redo(self) -> None:
        self.execute()


# ═══════════════════════════════════════════════════════════════════════════
# UndoManager singleton
# ═══════════════════════════════════════════════════════════════════════════

class UndoManager:
    """Central undo/redo manager with save-point tracking.

    Exception-safe: if ``undo()`` or ``redo()`` raises, the stack entry
    is preserved (not lost).
    """

    MAX_STACK_DEPTH: int = 200
    _instance: Optional[UndoManager] = None

    def __init__(self) -> None:
        self._undo_stack: List[UndoCommand] = []
        self._redo_stack: List[UndoCommand] = []
        self._save_point: Optional[int] = 0
        self._base_scene_dirty: bool = False
        self._is_executing: bool = False
        self._enabled: bool = True
        self._suppress_property_recording: bool = False
        self._on_state_changed: Optional[Callable[[], None]] = None
        UndoManager._instance = self

    @classmethod
    def instance(cls) -> Optional[UndoManager]:
        return cls._instance

    # -- Context managers --

    @contextmanager
    def suppress(self):
        """Suppress auto-recording while active (``is_executing`` = True)."""
        prev = self._is_executing
        self._is_executing = True
        try:
            yield
        finally:
            self._is_executing = prev

    @contextmanager
    def suppress_property_recording(self):
        """Suppress property-edit commands (``_is_property_edit`` = True).

        Structural commands still record normally.  Used by
        ``InspectorUndoTracker`` to prevent double-recording.
        """
        prev = self._suppress_property_recording
        self._suppress_property_recording = True
        try:
            yield
        finally:
            self._suppress_property_recording = prev

    # -- Properties --

    @property
    def is_executing(self) -> bool:
        return self._is_executing

    @property
    def can_undo(self) -> bool:
        return len(self._undo_stack) > 0

    @property
    def can_redo(self) -> bool:
        return len(self._redo_stack) > 0

    @property
    def undo_description(self) -> str:
        return self._undo_stack[-1].description if self._undo_stack else ""

    @property
    def redo_description(self) -> str:
        return self._redo_stack[-1].description if self._redo_stack else ""

    @property
    def _dirty_depth(self) -> int:
        return sum(1 for cmd in self._undo_stack if cmd.marks_dirty)

    @property
    def is_at_save_point(self) -> bool:
        if self._save_point is None:
            return False
        return self._dirty_depth == self._save_point

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value

    # -- Core operations --

    def execute(self, cmd: UndoCommand) -> None:
        """Execute *cmd* and push onto the undo stack."""
        from Infernux.debug import Debug

        if not self._enabled:
            try:
                cmd.execute()
                _bump_inspector_values()
            except Exception as exc:
                Debug.log_exception(exc)
            return

        self._is_executing = True
        try:
            cmd.execute()
        except Exception as exc:
            self._is_executing = False
            Debug.log_exception(exc)
            self._debug_dump_stack("execute-failed")
            return
        self._is_executing = False
        _bump_inspector_values()

        if self._suppress_property_recording and cmd._is_property_edit:
            return

        self._push(cmd)

    def record(self, cmd: UndoCommand) -> None:
        """Push an already-executed command onto the undo stack."""
        if not self._enabled:
            return
        if self._suppress_property_recording and cmd._is_property_edit:
            return
        self._push(cmd)
        _bump_inspector_values()

    def undo(self) -> None:
        """Undo the most recent command (exception-safe)."""
        from Infernux.debug import Debug

        if not self._undo_stack:
            return
        cmd = self._undo_stack.pop()
        self._is_executing = True
        try:
            cmd.undo()
        except Exception as exc:
            self._is_executing = False
            # Re-push so the entry is not lost
            self._undo_stack.append(cmd)
            Debug.log_exception(exc)
            self._debug_dump_stack("undo-failed")
            return
        self._is_executing = False
        _bump_inspector_values()

        if cmd.supports_redo:
            self._redo_stack.append(cmd)

        self._sync_dirty()
        self._fire_state_changed()
        self._debug_dump_stack("undo")

    def redo(self) -> None:
        """Redo the most recently undone command (exception-safe)."""
        from Infernux.debug import Debug

        if not self._redo_stack:
            return
        cmd = self._redo_stack.pop()
        self._is_executing = True
        try:
            cmd.redo()
        except Exception as exc:
            self._is_executing = False
            # Re-push so the entry is not lost
            self._redo_stack.append(cmd)
            Debug.log_exception(exc)
            self._debug_dump_stack("redo-failed")
            return
        self._is_executing = False
        _bump_inspector_values()

        self._undo_stack.append(cmd)

        self._sync_dirty()
        self._fire_state_changed()
        self._debug_dump_stack("redo")

    def clear(self, scene_is_dirty: bool = False) -> None:
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._save_point = 0
        self._base_scene_dirty = bool(scene_is_dirty)
        self._fire_state_changed()

    def mark_save_point(self) -> None:
        self._save_point = self._dirty_depth
        self._base_scene_dirty = False

    def sync_dirty_state(self) -> None:
        self._sync_dirty()

    def set_on_state_changed(self, cb: Optional[Callable[[], None]]) -> None:
        self._on_state_changed = cb

    # -- Private --

    def _push(self, cmd: UndoCommand) -> None:
        if self._undo_stack and self._undo_stack[-1].can_merge(cmd):
            self._undo_stack[-1].merge(cmd)
        else:
            self._undo_stack.append(cmd)
            if len(self._undo_stack) > self.MAX_STACK_DEPTH:
                overflow = len(self._undo_stack) - self.MAX_STACK_DEPTH
                dirty_dropped = sum(
                    1 for c in self._undo_stack[:overflow] if c.marks_dirty)
                del self._undo_stack[:overflow]
                if self._save_point is not None:
                    self._save_point -= dirty_dropped
                    if self._save_point < 0:
                        self._save_point = None

        self._redo_stack.clear()
        self._sync_dirty()
        self._fire_state_changed()
        self._debug_dump_stack("push")

    def _debug_dump_stack(self, action: str) -> None:
        from Infernux.debug import Debug
        pos = len(self._undo_stack)
        total = pos + len(self._redo_stack)
        parts = []
        for i, cmd in enumerate(self._undo_stack, 1):
            parts.append(f"{i}: {cmd.description}")
        for j, cmd in enumerate(self._redo_stack[::-1], pos + 1):
            parts.append(f"{j}: (redo) {cmd.description}")

    def _sync_dirty(self) -> None:
        from Infernux.engine.play_mode import PlayModeManager, PlayModeState
        pm = PlayModeManager.instance()
        if pm and pm.state != PlayModeState.EDIT:
            return
        from Infernux.engine.scene_manager import SceneFileManager
        sfm = SceneFileManager.instance()
        if sfm is None:
            return
        if self._base_scene_dirty or not self.is_at_save_point:
            sfm.mark_dirty()
        else:
            sfm.clear_dirty()

    def _fire_state_changed(self) -> None:
        if self._on_state_changed:
            self._on_state_changed()


# ═══════════════════════════════════════════════════════════════════════════
# Inspector snapshot-based undo tracking
# ═══════════════════════════════════════════════════════════════════════════

class InspectorSnapshotCommand(UndoCommand):
    """Snapshot-based undo for Inspector edits — one generic command for ALL
    component types (Transform, C++, Python, RenderStack, Material, …).

    Records serialised state before and after an edit.  Supports automatic
    merge for continuous edits within ``MERGE_WINDOW``.
    """

    MERGE_WINDOW: float = 0.3

    def __init__(self, target_key: str, old_snapshot: str,
                 new_snapshot: str,
                 restore_fn: Callable[[str], None],
                 description: str = "",
                 marks_dirty: bool = True):
        super().__init__(description or "Inspector Edit")
        self.marks_dirty = marks_dirty
        self._target_key = target_key
        self._old_snapshot = old_snapshot
        self._new_snapshot = new_snapshot
        self._restore_fn = restore_fn

    def execute(self) -> None:
        self._restore_fn(self._new_snapshot)

    def undo(self) -> None:
        self._restore_fn(self._old_snapshot)

    def redo(self) -> None:
        self._restore_fn(self._new_snapshot)

    def can_merge(self, other: UndoCommand) -> bool:
        if not isinstance(other, InspectorSnapshotCommand):
            return False
        return (self._target_key == other._target_key
                and (other.timestamp - self.timestamp) <= self.MERGE_WINDOW)

    def merge(self, other: InspectorSnapshotCommand) -> None:
        self._new_snapshot = other._new_snapshot
        self.timestamp = other.timestamp


class _TrackedEntry:
    __slots__ = ('pre_snapshot', 'snapshot_fn', 'restore_fn', 'description',
                 'seen_generation', 'marks_dirty')

    def __init__(self, pre_snapshot: str,
                 snapshot_fn: Callable[[], str],
                 restore_fn: Callable[[str], None],
                 description: str,
                 marks_dirty: bool = True):
        self.pre_snapshot = pre_snapshot
        self.snapshot_fn = snapshot_fn
        self.restore_fn = restore_fn
        self.description = description
        self.seen_generation = 0
        self.marks_dirty = marks_dirty


class InspectorUndoTracker:
    """Automatic snapshot-based undo for the Inspector panel.

    Workflow each frame:
      1. ``begin_frame()`` — mark entries stale
      2. ``track(key, snapshot_fn, restore_fn)`` — register each target
      3. Inspector widgets render and mutate (property recording suppressed)
      4. ``end_frame(any_item_active)`` — compare pre/post, record changes
    """

    def __init__(self):
        self._entries: dict[str, _TrackedEntry] = {}
        self._was_active: bool = False
        self._is_active: bool = False
        self._frame_generation: int = 0
        self._all_active_current_frame: bool = False

    def begin_frame(self) -> None:
        self._frame_generation += 1
        self._all_active_current_frame = False

    def invalidate_all(self) -> None:
        self._entries.clear()
        self._all_active_current_frame = False
        self._was_active = False
        self._is_active = False

    def mark_all_active(self) -> None:
        self._all_active_current_frame = True

    def track(self, key: str, snapshot_fn: Callable[[], str],
              restore_fn: Callable[[str], None],
              description: str = "",
              marks_dirty: bool = True) -> None:
        existing = self._entries.get(key)
        if existing is not None:
            existing.seen_generation = self._frame_generation
            existing.snapshot_fn = snapshot_fn
            existing.restore_fn = restore_fn
            return
        try:
            pre = snapshot_fn()
        except Exception as _exc:
            Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
            return
        entry = _TrackedEntry(pre, snapshot_fn, restore_fn, description, marks_dirty)
        entry.seen_generation = self._frame_generation
        self._entries[key] = entry

    def end_frame(self, any_item_active: bool | None = None) -> None:
        do_compare = True
        if any_item_active is not None:
            self._was_active, self._is_active = self._is_active, any_item_active
            do_compare = any_item_active or self._was_active

        mgr = UndoManager.instance()
        if not mgr or not mgr.enabled:
            self._entries.clear()
            return

        # Prune stale entries
        if not self._all_active_current_frame:
            stale = [k for k, e in self._entries.items()
                     if e.seen_generation != self._frame_generation]
            for k in stale:
                del self._entries[k]

        if not do_compare:
            return

        for key, entry in self._entries.items():
            try:
                post = entry.snapshot_fn()
            except Exception as _exc:
                Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
                continue
            if post != entry.pre_snapshot:
                cmd = InspectorSnapshotCommand(
                    key, entry.pre_snapshot, post,
                    entry.restore_fn, entry.description,
                    marks_dirty=entry.marks_dirty)
                mgr.record(cmd)
                entry.pre_snapshot = post


# ═══════════════════════════════════════════════════════════════════════════
# Snapshot / restore helpers for inspector_panel tracker registration
# ═══════════════════════════════════════════════════════════════════════════

def _get_live_game_object(game_object_id: int):
    if not game_object_id:
        return None
    scene = _get_active_scene()
    if not scene:
        return None
    try:
        return scene.find_by_id(game_object_id)
    except Exception as _exc:
        Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
        return None


def _get_live_transform(game_object_id: int):
    obj = _get_live_game_object(game_object_id)
    if obj is None:
        return None
    try:
        t = obj.get_transform()
        if t is not None:
            return t
    except Exception as _exc:
        Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
        pass
    return getattr(obj, 'transform', None)


def _get_nth_live_native_component(game_object_id: int, type_name: str,
                                   ordinal: int = 0):
    if type_name == "Transform":
        return _get_live_transform(game_object_id)
    obj = _get_live_game_object(game_object_id)
    if obj is None or not hasattr(obj, 'get_components'):
        return None
    match_index = 0
    try:
        try:
            from Infernux.components.component import InxComponent
        except Exception:
            InxComponent = ()
        for comp in obj.get_components():
            try:
                if getattr(comp, 'type_name', None) != type_name:
                    continue
                if isinstance(comp, InxComponent) or hasattr(comp, 'get_py_component'):
                    continue
            except Exception as _exc:
                Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
                continue
            if match_index == ordinal:
                return comp
            match_index += 1
    except Exception as _exc:
        Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
        pass
    return None


def _get_nth_live_py_component(game_object_id: int, type_name: str,
                               ordinal: int = 0, script_guid: str = ""):
    obj = _get_live_game_object(game_object_id)
    if obj is None or not hasattr(obj, 'get_py_components'):
        return None
    match_index = 0
    try:
        for comp in obj.get_py_components():
            try:
                ct = getattr(comp, 'type_name', type(comp).__name__)
                if ct != type_name and type(comp).__name__ != type_name:
                    continue
                if script_guid and getattr(comp, '_script_guid', '') != script_guid:
                    continue
                if getattr(comp, '_is_destroyed', False):
                    continue
            except Exception as _exc:
                Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
                continue
            if match_index == ordinal:
                return comp
            match_index += 1
    except Exception as _exc:
        Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
        pass
    return None


# -- Table-driven snapshot/restore for inspector_panel tracker registration --

def _snap_game_object(obj) -> str:
    return _json.dumps({"name": obj.name, "active": obj.active})

def _restore_game_object(obj, snapshot: str) -> None:
    data = _json.loads(snapshot)
    obj.name = data["name"]
    obj.active = data["active"]

# Registry: kind -> (resolver, snap_fn(target)->str, restore_fn(target,str)->None)
_SNAPSHOT_REGISTRY: dict[str, tuple] = {
    "game_object":  (_get_live_game_object,           _snap_game_object,          _restore_game_object),
    "transform":    (_get_live_transform,              lambda t: t.serialize(),    lambda t, s: t.deserialize(s)),
    "native":       (_get_nth_live_native_component,   lambda c: c.serialize(),    lambda c, s: c.deserialize(s)),
    "py":           (_get_nth_live_py_component,       lambda c: c._serialize_fields(), lambda c, s: c._deserialize_fields(s)),
    "renderstack":  (_get_nth_live_py_component,       lambda c: snapshot_renderstack(c), lambda c, s: restore_renderstack(c, s)),
}


def _resolve_and_snap(kind: str, game_object_id: int, **kwargs) -> str:
    resolver, snap_fn, _ = _SNAPSHOT_REGISTRY[kind]
    target = resolver(game_object_id, **kwargs) if kwargs else resolver(game_object_id)
    if target is None:
        return ""
    return snap_fn(target)


def _resolve_and_restore(kind: str, game_object_id: int, snapshot: str, **kwargs) -> None:
    _, _, restore_fn = _SNAPSHOT_REGISTRY[kind]
    resolver = _SNAPSHOT_REGISTRY[kind][0]
    target = resolver(game_object_id, **kwargs) if kwargs else resolver(game_object_id)
    if target is None:
        return
    restore_fn(target, snapshot)


# Public API — thin wrappers for each kind (preserves call signatures)

def snapshot_live_game_object(game_object_id: int) -> str:
    return _resolve_and_snap("game_object", game_object_id)

def restore_live_game_object(game_object_id: int, snapshot: str) -> None:
    _resolve_and_restore("game_object", game_object_id, snapshot)

def snapshot_live_transform(game_object_id: int) -> str:
    return _resolve_and_snap("transform", game_object_id)

def restore_live_transform(game_object_id: int, snapshot: str) -> None:
    _resolve_and_restore("transform", game_object_id, snapshot)

def snapshot_live_native_component(game_object_id: int, type_name: str,
                                   ordinal: int = 0) -> str:
    return _resolve_and_snap("native", game_object_id,
                             type_name=type_name, ordinal=ordinal)

def restore_live_native_component(game_object_id: int, type_name: str,
                                  ordinal: int, snapshot: str) -> None:
    _resolve_and_restore("native", game_object_id, snapshot,
                         type_name=type_name, ordinal=ordinal)

def snapshot_live_py_component(game_object_id: int, type_name: str,
                               ordinal: int = 0, script_guid: str = "") -> str:
    return _resolve_and_snap("py", game_object_id,
                             type_name=type_name, ordinal=ordinal,
                             script_guid=script_guid)

def restore_live_py_component(game_object_id: int, type_name: str,
                              ordinal: int, snapshot: str,
                              script_guid: str = "") -> None:
    _resolve_and_restore("py", game_object_id, snapshot,
                         type_name=type_name, ordinal=ordinal,
                         script_guid=script_guid)

def snapshot_live_renderstack_component(game_object_id: int, type_name: str,
                                        ordinal: int = 0,
                                        script_guid: str = "") -> str:
    return _resolve_and_snap("renderstack", game_object_id,
                             type_name=type_name, ordinal=ordinal,
                             script_guid=script_guid)

def restore_live_renderstack_component(game_object_id: int, type_name: str,
                                       ordinal: int, snapshot: str,
                                       script_guid: str = "") -> None:
    _resolve_and_restore("renderstack", game_object_id, snapshot,
                         type_name=type_name, ordinal=ordinal,
                         script_guid=script_guid)


# -- RenderStack snapshot/restore --

def _serialize_simple(val: Any) -> Any:
    import enum
    if isinstance(val, enum.Enum):
        return val.value
    if isinstance(val, (int, float, str, bool, type(None))):
        return val
    if isinstance(val, (list, tuple)):
        return [_serialize_simple(v) for v in val]
    if isinstance(val, dict):
        return {str(k): _serialize_simple(v) for k, v in val.items()}
    if hasattr(val, 'x') and hasattr(val, 'y'):
        if hasattr(val, 'w'):
            return [val.x, val.y, val.z, val.w]
        if hasattr(val, 'z'):
            return [val.x, val.y, val.z]
        return [val.x, val.y]
    return str(val)


def snapshot_renderstack(stack: Any) -> str:
    from Infernux.components.serialized_field import get_serialized_fields

    data: dict = {
        "pipeline_class_name": stack.pipeline_class_name or "",
        "pipeline_params": {},
        "pass_entries": [],
    }
    pipeline = stack.pipeline
    if pipeline:
        for name, meta in get_serialized_fields(pipeline.__class__).items():
            val = getattr(pipeline, name, meta.default)
            data["pipeline_params"][name] = _serialize_simple(val)

    for entry in stack.pass_entries:
        ed: dict = {
            "class": type(entry.render_pass).__name__,
            "name": entry.render_pass.name,
            "enabled": entry.enabled,
            "order": entry.order,
        }
        if hasattr(entry.render_pass, 'get_params_dict'):
            ed["params"] = entry.render_pass.get_params_dict()
        data["pass_entries"].append(ed)

    return _json.dumps(data, sort_keys=True)


def restore_renderstack(stack: Any, json_str: str) -> None:
    from Infernux.components.serialized_field import get_serialized_fields

    data = _json.loads(json_str)

    new_pipeline_name = data.get("pipeline_class_name", "")
    if (stack.pipeline_class_name or "") != new_pipeline_name:
        stack.set_pipeline(new_pipeline_name)

    pipeline = stack.pipeline
    if pipeline and "pipeline_params" in data:
        for name, val in data["pipeline_params"].items():
            try:
                setattr(pipeline, name, val)
            except (AttributeError, TypeError) as _exc:
                Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
                pass

    current_names = [e.render_pass.name for e in list(stack.pass_entries)]
    for name in current_names:
        stack.remove_pass(name)

    from Infernux.renderstack.discovery import discover_passes
    from Infernux.renderstack.fullscreen_effect import FullScreenEffect

    all_passes = discover_passes()
    for ed in data.get("pass_entries", []):
        cls_name = ed.get("class", "")
        pass_name = ed.get("name", "")
        cls = all_passes.get(pass_name)
        if cls is None:
            for pcls in all_passes.values():
                if pcls.__name__ == cls_name:
                    cls = pcls
                    break
        if cls is None:
            continue
        inst = cls()
        if isinstance(inst, FullScreenEffect) and "params" in ed:
            inst.set_params_dict(ed["params"])
        inst.enabled = ed.get("enabled", True)
        stack.add_pass(inst)
        stack.set_pass_enabled(pass_name, ed.get("enabled", True))
        stack.reorder_pass(pass_name, ed.get("order", 0))

    stack.invalidate_graph()


# ═══════════════════════════════════════════════════════════════════════════
# Game object recreation from JSON (used by Create/Delete commands)
# ═══════════════════════════════════════════════════════════════════════════

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
                f"(guid={script_guid}) during undo/redo"
            )

        if fields_json:
            instance._deserialize_fields(fields_json)
        instance.enabled = enabled
        go.add_py_component(instance)
        if hasattr(instance, "_call_on_after_deserialize"):
            instance._call_on_after_deserialize()


# ═══════════════════════════════════════════════════════════════════════════
# Hierarchy undo tracker — facade for hierarchy panel operations
# ═══════════════════════════════════════════════════════════════════════════

class HierarchyUndoTracker:
    """Unified undo interface for Hierarchy panel mutations.

    Provides a single entry-point for create, delete, reparent, and
    reorder.  Selection changes are NOT tracked (consistent with Unity).
    """

    @staticmethod
    def _mgr() -> Optional[UndoManager]:
        return UndoManager.instance()

    def record_create(self, object_id: int,
                      description: str = "Create GameObject") -> None:
        mgr = self._mgr()
        if mgr:
            mgr.record(CreateGameObjectCommand(object_id, description))
            return
        from Infernux.engine.scene_manager import SceneFileManager
        sfm = SceneFileManager.instance()
        if sfm:
            sfm.mark_dirty()

    def record_delete(self, object_id: int,
                      description: str = "Delete GameObject") -> None:
        mgr = self._mgr()
        if mgr:
            mgr.execute(DeleteGameObjectCommand(object_id, description))
            return
        scene = _get_active_scene()
        if scene:
            obj = scene.find_by_id(object_id)
            if obj:
                scene.destroy_game_object(obj)
                from Infernux.engine.scene_manager import SceneFileManager
                sfm = SceneFileManager.instance()
                if sfm:
                    sfm.mark_dirty()

    def record_reparent(self, object_id: int,
                        old_parent_id: Optional[int],
                        new_parent_id: Optional[int],
                        description: str = "Reparent") -> None:
        mgr = self._mgr()
        if mgr:
            mgr.execute(ReparentCommand(
                object_id, old_parent_id, new_parent_id, description))
            return
        scene = _get_active_scene()
        if scene:
            obj = scene.find_by_id(object_id)
            if obj:
                parent = (scene.find_by_id(new_parent_id)
                          if new_parent_id is not None else None)
                old_parent = obj.get_parent()
                _preserve_ui_world_position(obj, parent)
                obj.set_parent(parent)
                _invalidate_canvas_caches(old_parent)
                _invalidate_canvas_caches(parent)
                from Infernux.engine.scene_manager import SceneFileManager
                sfm = SceneFileManager.instance()
                if sfm:
                    sfm.mark_dirty()

    def record_move(self, object_id: int,
                    old_parent_id: Optional[int],
                    new_parent_id: Optional[int],
                    old_sibling_index: int,
                    new_sibling_index: int,
                    description: str = "Move In Hierarchy") -> None:
        mgr = self._mgr()
        if mgr:
            mgr.execute(MoveGameObjectCommand(
                object_id, old_parent_id, new_parent_id,
                old_sibling_index, new_sibling_index, description))
            return
        scene = _get_active_scene()
        if scene:
            obj = scene.find_by_id(object_id)
            if obj:
                parent = (scene.find_by_id(new_parent_id)
                          if new_parent_id is not None else None)
                old_parent = obj.get_parent()
                _preserve_ui_world_position(obj, parent)
                obj.set_parent(parent)
                _invalidate_canvas_caches(old_parent)
                _invalidate_canvas_caches(parent)
                transform = getattr(obj, 'transform', None)
                if transform is not None:
                    transform.set_sibling_index(max(0, int(new_sibling_index)))
                from Infernux.engine.scene_manager import SceneFileManager
                sfm = SceneFileManager.instance()
                if sfm:
                    sfm.mark_dirty()


# ═══════════════════════════════════════════════════════════════════════════
# Backward-compatibility shims for removed RenderStack command classes
# (these were eliminated in favour of InspectorUndoTracker snapshots)
# ═══════════════════════════════════════════════════════════════════════════

class RenderStackFieldCommand(SetPropertyCommand):
    """Property edit on a RenderStack target with graph invalidation."""

    def __init__(self, stack: Any, target: Any, field_name: str,
                 old_value: Any, new_value: Any, description: str = ""):
        super().__init__(target, field_name, old_value, new_value,
                         description or f"Set {field_name}")
        self._stack = stack

    def execute(self) -> None:
        super().execute()
        self._stack.invalidate_graph()

    def undo(self) -> None:
        super().undo()
        self._stack.invalidate_graph()

    def redo(self) -> None:
        super().redo()
        self._stack.invalidate_graph()


class RenderStackSetPipelineCommand(UndoCommand):
    _is_property_edit = True

    def __init__(self, stack, old_pipeline: str, new_pipeline: str,
                 description: str = "Set Render Pipeline"):
        super().__init__(description)
        self._stack = stack
        self._old_pipeline = old_pipeline
        self._new_pipeline = new_pipeline

    def execute(self) -> None:
        self._stack.set_pipeline(self._new_pipeline)

    def undo(self) -> None:
        self._stack.set_pipeline(self._old_pipeline)

    def redo(self) -> None:
        self._stack.set_pipeline(self._new_pipeline)


class RenderStackAddPassCommand(UndoCommand):
    _is_property_edit = True

    def __init__(self, stack, effect_cls: type,
                 description: str = "Add Effect"):
        super().__init__(description)
        self._stack = stack
        self._effect_cls = effect_cls
        self._pass_name: str = getattr(effect_cls, "name", effect_cls.__name__)
        self._snapshot: Optional[str] = None

    def execute(self) -> None:
        self._snapshot = snapshot_renderstack(self._stack)
        inst = self._effect_cls()
        self._stack.add_pass(inst)
        self._pass_name = inst.name

    def undo(self) -> None:
        if self._snapshot:
            restore_renderstack(self._stack, self._snapshot)
        else:
            self._stack.remove_pass(self._pass_name)

    def redo(self) -> None:
        self.execute()


class RenderStackMovePassCommand(UndoCommand):
    _is_property_edit = True

    def __init__(self, stack, old_orders: dict, new_orders: dict,
                 description: str = "Reorder Effect"):
        super().__init__(description)
        self._stack = stack
        self._old_orders = dict(old_orders)
        self._new_orders = dict(new_orders)

    def execute(self) -> None:
        self._apply(self._new_orders)

    def undo(self) -> None:
        self._apply(self._old_orders)

    def redo(self) -> None:
        self._apply(self._new_orders)

    def _apply(self, orders):
        for entry in self._stack.pass_entries:
            name = entry.render_pass.name
            if name in orders:
                entry.order = int(orders[name])
        self._stack.invalidate_graph()


class RenderStackTogglePassCommand(UndoCommand):
    _is_property_edit = True

    def __init__(self, stack, pass_name: str, old_enabled: bool,
                 new_enabled: bool, description: str = "Toggle Effect"):
        super().__init__(description)
        self._stack = stack
        self._pass_name = pass_name
        self._old_enabled = bool(old_enabled)
        self._new_enabled = bool(new_enabled)

    def execute(self) -> None:
        self._stack.set_pass_enabled(self._pass_name, self._new_enabled)

    def undo(self) -> None:
        self._stack.set_pass_enabled(self._pass_name, self._old_enabled)

    def redo(self) -> None:
        self._stack.set_pass_enabled(self._pass_name, self._new_enabled)


class RenderStackRemovePassCommand(UndoCommand):
    _is_property_edit = True

    def __init__(self, stack, pass_name: str,
                 description: str = "Remove Effect"):
        super().__init__(description)
        self._stack = stack
        self._pass_name = pass_name
        self._snapshot: Optional[str] = None

    def execute(self) -> None:
        self._snapshot = snapshot_renderstack(self._stack)
        self._stack.remove_pass(self._pass_name)

    def undo(self) -> None:
        if self._snapshot:
            restore_renderstack(self._stack, self._snapshot)

    def redo(self) -> None:
        self._stack.remove_pass(self._pass_name)
