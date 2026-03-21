"""
Undo/Redo system for InfEngine editor operations.

Implements a command pattern with:
- Per-property undo with automatic merge for rapid edits (slider dragging)
- Save-point tracking for clean/dirty state synchronisation
- Integration with SceneFileManager for dirty flag management
- Play mode isolation (stack cleared on play/stop)

Usage::

    from InfEngine.engine.undo import UndoManager, SetPropertyCommand

    mgr = UndoManager.instance()
    mgr.execute(SetPropertyCommand(obj, "position", old, new, "Move Object"))
    mgr.undo()   # restores old value
    mgr.redo()   # re-applies new value
"""

from __future__ import annotations

import json as _json_mod
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Optional, List, Callable


# ---------------------------------------------------------------------------
# Base command
# ---------------------------------------------------------------------------

class UndoCommand(ABC):
    """Base class for all undoable editor commands."""

    #: Whether this command supports redo after undo.  Commands that cannot
    #: fully recreate their effect (e.g. *CreateGameObject*) should set this
    #: to ``False`` so they are discarded from the redo stack on undo.
    supports_redo: bool = True

    #: Whether this command represents a scene modification that affects the
    #: dirty/save-point state.  Commands like selection changes set this to
    #: ``False`` so that selecting an object does not mark the scene dirty.
    marks_dirty: bool = True

    #: Whether this command is a property/data edit that should be suppressed
    #: when the InspectorUndoTracker is active.  Structural commands (create,
    #: delete, reparent, add/remove component) leave this as ``False`` so
    #: they are always recorded.
    _is_property_edit: bool = False

    def __init__(self, description: str = ""):
        self.description: str = description
        self.timestamp: float = time.time()

    @abstractmethod
    def execute(self) -> None:
        """Perform the action.  Called by :meth:`UndoManager.execute`."""

    @abstractmethod
    def undo(self) -> None:
        """Reverse the action."""

    def redo(self) -> None:
        """Re-apply the action.  Defaults to :meth:`execute`."""
        self.execute()

    # -- Merging (for consecutive rapid edits on the same property) --

    def can_merge(self, other: UndoCommand) -> bool:
        """Return *True* if *other* can be folded into this command."""
        return False

    def merge(self, other: UndoCommand) -> None:
        """Absorb *other* into this command (called only when :meth:`can_merge` returned *True*)."""


# ---------------------------------------------------------------------------
# Value snapshot helper
# ---------------------------------------------------------------------------

def _snapshot_value(value: Any) -> Any:
    """Return a deep copy of *value* if it is a mutable container.

    Immutable types (int, float, str, bool, None, tuple, enums, pybind11
    value types like Vector3) are returned as-is.  Lists and dicts are
    deep-copied so that later in-place mutations do not corrupt undo
    snapshots.
    """
    if isinstance(value, list):
        import copy
        return copy.deepcopy(value)
    if isinstance(value, dict):
        import copy
        return copy.deepcopy(value)
    return value


def _resolve_live_ref(stored: Any, game_object_id: int,
                      comp_type_name: Optional[str]) -> Any:
    """Return the live component after a delete+undo-delete recreate cycle.

    When ``DeleteGameObjectCommand.undo()`` recreates an object from JSON all
    previously-cached pybind11 component wrappers point to freed C++ memory.
    This helper re-fetches the live component of the same type from the
    reconstructed game object using the stored ``game_object_id`` and
    component type name (string).

    Falls back to the stored (stale) reference when the scene or object
    cannot be found, so callers get a meaningful error rather than a segfault.
    """
    if not game_object_id or not comp_type_name:
        return stored
    scene = _get_active_scene()
    if not scene:
        return stored
    obj = scene.find_by_id(game_object_id)
    if obj is None:
        return stored
    try:
        # Transform is accessible directly via obj.transform
        if comp_type_name == "Transform":
            t = getattr(obj, "transform", None)
            if t is not None:
                return t
        # C++ components: get_component() expects a string type name
        live = obj.get_component(comp_type_name)
        if live is not None:
            return live
    except Exception:
        pass
    return stored


def _comp_type_name_of(target: Any) -> str:
    """Return the component type name string suitable for ``get_component()``."""
    # C++ pybind11 components expose type_name
    tn = getattr(target, 'type_name', None)
    if tn:
        return str(tn)
    # Fallback: Python class name (works for both InfComponent and pybind11)
    return type(target).__name__


# ---------------------------------------------------------------------------
# Concrete commands
# ---------------------------------------------------------------------------

class SetPropertyCommand(UndoCommand):
    """Set a property on a target object via ``setattr``.

    Works uniformly for C++ components (pybind11 properties) and Python
    ``InfComponent`` fields (``SerializedFieldDescriptor``).

    Consecutive rapid edits to the **same target + property** within
    :attr:`MERGE_WINDOW` seconds are merged into a single undo entry
    (e.g. dragging a slider).

    Mutable values (lists, dicts) are deep-copied on capture so that
    later in-place mutations do not corrupt the undo snapshot.
    """

    _is_property_edit = True
    MERGE_WINDOW: float = 0.3  # seconds

    def __init__(self, target: Any, prop_name: str,
                 old_value: Any, new_value: Any,
                 description: str = ""):
        super().__init__(description or f"Set {prop_name}")
        self._target = target
        self._prop_name = prop_name
        self._old_value = _snapshot_value(old_value)
        self._new_value = _snapshot_value(new_value)
        self._target_id: int = self._stable_id(target)
        # Cache owner game object ID + type name so undo/redo can re-fetch
        # the live component after a delete+recreate cycle (stale C++ pointer).
        _goid = getattr(target, 'game_object_id', None) or 0
        if not _goid:
            _go = getattr(target, 'game_object', None)
            if _go is not None:
                _goid = getattr(_go, 'id', 0) or 0
        self._game_object_id: int = _goid
        self._comp_type_name: str = _comp_type_name_of(target) if _goid else ""

    # -- stable identity for merge comparisons --
    @staticmethod
    def _stable_id(target: Any) -> int:
        for attr in ("component_id", "id"):
            val = getattr(target, attr, None)
            if val is not None and val != 0:
                return int(val)
        return id(target)

    def execute(self) -> None:
        setattr(self._target, self._prop_name, self._new_value)

    def undo(self) -> None:
        target = _resolve_live_ref(self._target, self._game_object_id, self._comp_type_name)
        setattr(target, self._prop_name, self._old_value)

    def redo(self) -> None:
        target = _resolve_live_ref(self._target, self._game_object_id, self._comp_type_name)
        setattr(target, self._prop_name, self._new_value)

    def can_merge(self, other: UndoCommand) -> bool:
        if not isinstance(other, SetPropertyCommand):
            return False
        return (self._target_id == other._target_id
                and self._prop_name == other._prop_name
                and (other.timestamp - self.timestamp) <= self.MERGE_WINDOW)

    def merge(self, other: SetPropertyCommand) -> None:  # type: ignore[override]
        self._new_value = _snapshot_value(other._new_value)
        self.timestamp = other.timestamp


class GenericComponentCommand(UndoCommand):
    """Undo/redo for a C++ component edited via the generic
    *serialize → edit → deserialize* path in the Inspector.
    """

    _is_property_edit = True
    MERGE_WINDOW: float = 0.3

    def __init__(self, comp: Any, old_json: str, new_json: str,
                 description: str = ""):
        super().__init__(description or f"Edit {getattr(comp, 'type_name', 'Component')}")
        self._comp = comp
        self._old_json = old_json
        self._new_json = new_json
        self._comp_id: int = getattr(comp, "component_id", id(comp))
        _goid = getattr(comp, 'game_object_id', None) or 0
        if not _goid:
            _go = getattr(comp, 'game_object', None)
            if _go is not None:
                _goid = getattr(_go, 'id', 0) or 0
        self._game_object_id: int = _goid
        self._comp_type_name: str = _comp_type_name_of(comp)

    def execute(self) -> None:
        self._comp.deserialize(self._new_json)

    def undo(self) -> None:
        comp = _resolve_live_ref(self._comp, self._game_object_id, self._comp_type_name)
        comp.deserialize(self._old_json)

    def redo(self) -> None:
        comp = _resolve_live_ref(self._comp, self._game_object_id, self._comp_type_name)
        comp.deserialize(self._new_json)

    def can_merge(self, other: UndoCommand) -> bool:
        if not isinstance(other, GenericComponentCommand):
            return False
        return (self._comp_id == other._comp_id
                and (other.timestamp - self.timestamp) <= self.MERGE_WINDOW)

    def merge(self, other: GenericComponentCommand) -> None:  # type: ignore[override]
        self._new_json = other._new_json
        self.timestamp = other.timestamp


class BuiltinPropertyCommand(UndoCommand):
    """Undo/redo for a C++ component property edited via direct setter.

    Unlike ``GenericComponentCommand`` (which goes through JSON
    serialize/deserialize), this command uses ``setattr(comp, attr, val)``
    which calls the pybind11 property setter → C++ ``SetXXX()`` →
    ``RebuildShape()`` / physics sync.  This is the preferred path for
    BuiltinComponent wrappers (colliders, rigidbody, etc.) because it
    guarantees immediate physics world updates at runtime.
    """

    _is_property_edit = True
    MERGE_WINDOW: float = 0.3

    def __init__(self, comp: Any, cpp_attr: str, old_value: Any,
                 new_value: Any, description: str = ""):
        super().__init__(description or f"Set {cpp_attr}")
        self._comp = comp
        self._cpp_attr = cpp_attr
        self._old_value = _snapshot_value(old_value)
        self._new_value = _snapshot_value(new_value)
        self._comp_id: int = getattr(comp, "component_id", id(comp))
        _goid = getattr(comp, 'game_object_id', None) or 0
        if not _goid:
            _go = getattr(comp, 'game_object', None)
            if _go is not None:
                _goid = getattr(_go, 'id', 0) or 0
        self._game_object_id: int = _goid
        self._comp_type_name: str = _comp_type_name_of(comp)

    def execute(self) -> None:
        setattr(self._comp, self._cpp_attr, self._new_value)

    def undo(self) -> None:
        comp = _resolve_live_ref(self._comp, self._game_object_id, self._comp_type_name)
        setattr(comp, self._cpp_attr, self._old_value)

    def redo(self) -> None:
        comp = _resolve_live_ref(self._comp, self._game_object_id, self._comp_type_name)
        setattr(comp, self._cpp_attr, self._new_value)

    def can_merge(self, other: UndoCommand) -> bool:
        if not isinstance(other, BuiltinPropertyCommand):
            return False
        return (self._comp_id == other._comp_id
                and self._cpp_attr == other._cpp_attr
                and (other.timestamp - self.timestamp) <= self.MERGE_WINDOW)

    def merge(self, other: BuiltinPropertyCommand) -> None:  # type: ignore[override]
        self._new_value = _snapshot_value(other._new_value)
        self.timestamp = other.timestamp


class CreateGameObjectCommand(UndoCommand):
    """Record the creation of a GameObject.  Undo destroys it; redo recreates.

    On undo the object is serialized to JSON before destruction so that
    redo can recreate it (including children and Python components).
    """
    supports_redo = True

    #: Callback ``fn(ids: list[int])`` to restore selection state.
    #: Set once by bootstrap so that undo/redo can update the full
    #: selection pipeline (SelectionManager, inspector, outline, hierarchy).
    _selection_restore_fn: Optional[Callable[[List[int]], None]] = None

    def __init__(self, object_id: int, description: str = "Create GameObject"):
        super().__init__(description)
        self._object_id = object_id
        self._snapshot_json: Optional[str] = None
        self._parent_id: Optional[int] = None
        self._sibling_index: int = 0
        # Selection state just after creation (usually [object_id]).
        self._post_create_ids: List[int] = []
        self._capture_selection()

    def _capture_selection(self) -> None:
        try:
            from InfEngine.engine.ui.selection_manager import SelectionManager
            sel = SelectionManager.instance()
            if sel:
                self._post_create_ids = sel.get_ids()
        except Exception:
            pass

    def execute(self) -> None:
        pass  # already created before record()

    def undo(self) -> None:
        scene = _get_active_scene()
        if scene:
            obj = scene.find_by_id(self._object_id)
            if obj:
                self._snapshot_json = obj.serialize()
                parent = obj.get_parent()
                self._parent_id = parent.id if parent else None
                self._sibling_index = (obj.transform.get_sibling_index()
                                       if getattr(obj, "transform", None) else 0)
                scene.destroy_game_object(obj)
        # After destroying, clear selection if it contained this object.
        fn = type(self)._selection_restore_fn
        if fn:
            fn([])

    def redo(self) -> None:
        if self._snapshot_json:
            _recreate_game_object_from_json(
                self._snapshot_json, self._parent_id, self._sibling_index)
            # Re-select the recreated object.
            fn = type(self)._selection_restore_fn
            if fn and self._post_create_ids:
                fn(self._post_create_ids)


class DeleteGameObjectCommand(UndoCommand):
    """Record the deletion of a GameObject.  Undo recreates it from JSON.

    The object is serialized in the constructor (before ``execute``
    destroys it) so that all component/child data is preserved for undo.
    Pre-delete selection state is also captured so that undo can
    re-select the object.
    """

    #: Callback ``fn(ids: list[int])`` — set by bootstrap (shared with
    #: :class:`CreateGameObjectCommand`).
    _selection_restore_fn: Optional[Callable[[List[int]], None]] = None

    def __init__(self, object_id: int, description: str = "Delete GameObject"):
        super().__init__(description)
        self._object_id = object_id
        self._snapshot_json: Optional[str] = None
        self._parent_id: Optional[int] = None
        self._sibling_index: int = 0
        # Selection state *before* the deletion so undo can restore it.
        self._pre_delete_selection_ids: List[int] = []

        # Capture snapshot + selection before destruction
        scene = _get_active_scene()
        if scene:
            obj = scene.find_by_id(object_id)
            if obj:
                self._snapshot_json = obj.serialize()
                parent = obj.get_parent()
                self._parent_id = parent.id if parent else None
                self._sibling_index = (obj.transform.get_sibling_index()
                                       if getattr(obj, "transform", None) else 0)
        self._capture_selection()

    def _capture_selection(self) -> None:
        try:
            from InfEngine.engine.ui.selection_manager import SelectionManager
            sel = SelectionManager.instance()
            if sel:
                self._pre_delete_selection_ids = sel.get_ids()
        except Exception:
            pass

    def execute(self) -> None:
        scene = _get_active_scene()
        if scene:
            obj = scene.find_by_id(self._object_id)
            if obj:
                scene.destroy_game_object(obj)

    def undo(self) -> None:
        if self._snapshot_json:
            _recreate_game_object_from_json(
                self._snapshot_json, self._parent_id, self._sibling_index)
            # Restore the selection that was active before delete.
            fn = type(self)._selection_restore_fn
            if fn and self._pre_delete_selection_ids:
                fn(self._pre_delete_selection_ids)

    def redo(self) -> None:
        scene = _get_active_scene()
        if scene:
            obj = scene.find_by_id(self._object_id)
            if obj:
                # Refresh snapshot before re-deleting
                self._snapshot_json = obj.serialize()
                scene.destroy_game_object(obj)
        # Clear selection after re-delete.
        fn = type(self)._selection_restore_fn
        if fn:
            fn([])


class ReparentCommand(UndoCommand):
    """Undo/redo reparenting of a GameObject."""

    def __init__(self, object_id: int,
                 old_parent_id: Optional[int],
                 new_parent_id: Optional[int],
                 description: str = "Reparent"):
        super().__init__(description)
        self._object_id = object_id
        self._old_parent_id = old_parent_id
        self._new_parent_id = new_parent_id

    def execute(self) -> None:
        self._apply_parent(self._new_parent_id)

    def undo(self) -> None:
        self._apply_parent(self._old_parent_id)

    def redo(self) -> None:
        self._apply_parent(self._new_parent_id)

    def _apply_parent(self, parent_id: Optional[int]) -> None:
        scene = _get_active_scene()
        if not scene:
            return
        obj = scene.find_by_id(self._object_id)
        if not obj:
            return
        new_parent = scene.find_by_id(parent_id) if parent_id is not None else None
        _preserve_ui_world_position(obj, new_parent)
        obj.set_parent(new_parent)


class MoveGameObjectCommand(UndoCommand):
    """Undo/redo hierarchy moves that may change both parent and sibling order."""

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

        transform = getattr(obj, "transform", None)
        if transform is not None:
            transform.set_sibling_index(max(0, int(sibling_index)))


class MaterialJsonCommand(UndoCommand):
    """Undo/redo for material asset edits represented as serialized JSON."""

    _is_property_edit = True
    MERGE_WINDOW: float = 0.3
    marks_dirty: bool = False

    def __init__(self, material: Any, old_json: str, new_json: str,
                 description: str = "Edit Material",
                 refresh_callback: Optional[Callable[[Any], None]] = None):
        super().__init__(description)
        self._material = material
        self._old_json = old_json
        self._new_json = new_json
        self._refresh_callback = refresh_callback
        self._material_id = self._stable_id(material)

    @staticmethod
    def _stable_id(material: Any) -> int:
        guid = getattr(material, "guid", "")
        if guid:
            return hash(("material-guid", guid))
        file_path = getattr(material, "file_path", "")
        if file_path:
            return hash(("material-file", file_path))
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
                and (other.timestamp - self.timestamp) <= self.MERGE_WINDOW)

    def merge(self, other: MaterialJsonCommand) -> None:  # type: ignore[override]
        self._new_json = other._new_json
        self.timestamp = other.timestamp

    def _apply(self, json_str: str) -> None:
        self._material.deserialize(json_str)
        if self._refresh_callback:
            self._refresh_callback(self._material)
        save = getattr(self._material, "save", None)
        if callable(save):
            save()


class RenderStackSetPipelineCommand(UndoCommand):
    """Undo/redo changing the selected RenderStack pipeline."""

    _is_property_edit = True

    def __init__(self, stack: Any, old_pipeline: str, new_pipeline: str,
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


class RenderStackFieldCommand(UndoCommand):
    """Undo/redo a RenderStack-owned field change and rebuild the graph."""

    _is_property_edit = True
    MERGE_WINDOW: float = 0.3

    def __init__(self, stack: Any, target: Any, field_name: str,
                 old_value: Any, new_value: Any, description: str = ""):
        super().__init__(description or f"Set {field_name}")
        self._stack = stack
        self._target = target
        self._field_name = field_name
        self._old_value = _snapshot_value(old_value)
        self._new_value = _snapshot_value(new_value)
        self._target_id = id(target)

    def execute(self) -> None:
        setattr(self._target, self._field_name, self._new_value)
        self._stack.invalidate_graph()

    def undo(self) -> None:
        setattr(self._target, self._field_name, self._old_value)
        self._stack.invalidate_graph()

    def redo(self) -> None:
        setattr(self._target, self._field_name, self._new_value)
        self._stack.invalidate_graph()

    def can_merge(self, other: UndoCommand) -> bool:
        if not isinstance(other, RenderStackFieldCommand):
            return False
        return (self._target_id == other._target_id
                and self._field_name == other._field_name
                and (other.timestamp - self.timestamp) <= self.MERGE_WINDOW)

    def merge(self, other: RenderStackFieldCommand) -> None:  # type: ignore[override]
        self._new_value = _snapshot_value(other._new_value)
        self.timestamp = other.timestamp


class RenderStackTogglePassCommand(UndoCommand):
    """Undo/redo toggling an effect enabled state."""

    _is_property_edit = True
    MERGE_WINDOW: float = 0.3

    def __init__(self, stack: Any, pass_name: str, old_enabled: bool,
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

    def can_merge(self, other: UndoCommand) -> bool:
        if not isinstance(other, RenderStackTogglePassCommand):
            return False
        return (self._pass_name == other._pass_name
                and (other.timestamp - self.timestamp) <= self.MERGE_WINDOW)

    def merge(self, other: RenderStackTogglePassCommand) -> None:  # type: ignore[override]
        self._new_enabled = other._new_enabled
        self.timestamp = other.timestamp


class RenderStackMovePassCommand(UndoCommand):
    """Undo/redo effect ordering within one injection point."""

    _is_property_edit = True

    def __init__(self, stack: Any, old_orders: dict[str, int],
                 new_orders: dict[str, int], description: str = "Reorder Effect"):
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

    def _apply(self, orders: dict[str, int]) -> None:
        for entry in self._stack.pass_entries:
            name = entry.render_pass.name
            if name in orders:
                entry.order = int(orders[name])
        self._stack.invalidate_graph()


class RenderStackAddPassCommand(UndoCommand):
    """Undo/redo adding an effect to RenderStack."""

    _is_property_edit = True

    def __init__(self, stack: Any, effect_cls: type, description: str = "Add Effect"):
        super().__init__(description)
        self._stack = stack
        self._effect_cls = effect_cls
        self._entry_state: Optional[dict[str, Any]] = None
        self._pass_name: str = getattr(effect_cls, "name", effect_cls.__name__)

    def execute(self) -> None:
        entry = self._restore_or_create_entry()
        if entry is None:
            return
        self._pass_name = entry.render_pass.name
        self._entry_state = self._snapshot_entry(entry.render_pass.name)

    def undo(self) -> None:
        if self._entry_state is None:
            self._entry_state = self._snapshot_entry(self._pass_name)
        self._stack.remove_pass(self._pass_name)

    def redo(self) -> None:
        self.execute()

    def _restore_or_create_entry(self):
        render_pass = self._create_pass_from_state()
        if render_pass is None:
            return None
        if not self._stack.add_pass(render_pass):
            return self._find_entry(self._pass_name)
        entry = self._find_entry(render_pass.name)
        if entry is not None and self._entry_state is not None:
            entry.enabled = bool(self._entry_state.get("enabled", True))
            entry.render_pass.enabled = entry.enabled
            entry.order = int(self._entry_state.get("order", entry.order))
            self._stack.invalidate_graph()
        return entry

    def _create_pass_from_state(self):
        inst = self._effect_cls()
        if self._entry_state is not None:
            params = self._entry_state.get("params")
            if params is not None and hasattr(inst, "set_params_dict"):
                inst.set_params_dict(params)
            inst.enabled = bool(self._entry_state.get("enabled", True))
        return inst

    def _find_entry(self, pass_name: str):
        for entry in self._stack.pass_entries:
            if entry.render_pass.name == pass_name:
                return entry
        return None

    def _snapshot_entry(self, pass_name: str) -> Optional[dict[str, Any]]:
        entry = self._find_entry(pass_name)
        if entry is None:
            return None
        state = {
            "enabled": entry.enabled,
            "order": entry.order,
        }
        params = getattr(entry.render_pass, "get_params_dict", None)
        if callable(params):
            state["params"] = params()
        return state


class RenderStackRemovePassCommand(UndoCommand):
    """Undo/redo removing an effect from RenderStack."""

    _is_property_edit = True

    def __init__(self, stack: Any, pass_name: str, description: str = "Remove Effect"):
        super().__init__(description)
        self._stack = stack
        self._pass_name = pass_name
        self._entry_state = self._snapshot_entry(pass_name)

    def execute(self) -> None:
        self._entry_state = self._snapshot_entry(self._pass_name)
        self._stack.remove_pass(self._pass_name)

    def undo(self) -> None:
        if self._entry_state is None:
            return
        entry = self._restore_entry(self._entry_state)
        if entry is not None:
            self._stack.invalidate_graph()

    def redo(self) -> None:
        self._stack.remove_pass(self._pass_name)

    def _find_entry(self, pass_name: str):
        for entry in self._stack.pass_entries:
            if entry.render_pass.name == pass_name:
                return entry
        return None

    def _snapshot_entry(self, pass_name: str) -> Optional[dict[str, Any]]:
        entry = self._find_entry(pass_name)
        if entry is None:
            return None
        cls = type(entry.render_pass)
        state = {
            "class": cls,
            "enabled": entry.enabled,
            "order": entry.order,
        }
        params = getattr(entry.render_pass, "get_params_dict", None)
        if callable(params):
            state["params"] = params()
        return state

    def _restore_entry(self, state: dict[str, Any]):
        cls = state.get("class")
        if cls is None:
            return None
        inst = cls()
        params = state.get("params")
        if params is not None and hasattr(inst, "set_params_dict"):
            inst.set_params_dict(params)
        inst.enabled = bool(state.get("enabled", True))
        if not self._stack.add_pass(inst):
            return self._find_entry(inst.name)
        entry = self._find_entry(inst.name)
        if entry is not None:
            entry.enabled = bool(state.get("enabled", True))
            entry.render_pass.enabled = entry.enabled
            entry.order = int(state.get("order", entry.order))
        return entry


class AddNativeComponentCommand(UndoCommand):
    """Record adding a C++ component.  Undo removes it; redo re-adds."""

    def __init__(self, object_id: int, type_name: str, comp_ref: Any = None,
                 description: str = ""):
        super().__init__(description or f"Add {type_name}")
        self._object_id = object_id
        self._type_name = type_name
        self._comp_ref = comp_ref

    def execute(self) -> None:
        pass  # already added before record()

    def undo(self) -> None:
        scene = _get_active_scene()
        if scene and self._comp_ref:
            obj = scene.find_by_id(self._object_id)
            if obj and hasattr(obj, "remove_component"):
                obj.remove_component(self._comp_ref)
                _notify_gizmos_scene_changed()

    def redo(self) -> None:
        scene = _get_active_scene()
        if scene:
            obj = scene.find_by_id(self._object_id)
            if obj:
                result = obj.add_component(self._type_name)
                if result:
                    self._comp_ref = result
                    _notify_gizmos_scene_changed()


class RemoveNativeComponentCommand(UndoCommand):
    """Undo/redo removal of a C++ component.

    Snapshots the component's serialized JSON before removal so that
    undo can re-add plus restore state.
    """

    def __init__(self, object_id: int, type_name: str, comp_ref: Any,
                 description: str = ""):
        super().__init__(description or f"Remove {type_name}")
        self._object_id = object_id
        self._type_name = type_name
        self._comp_ref = comp_ref
        # Snapshot serialized state for undo restoration
        self._json_snapshot: Optional[str] = None
        if hasattr(comp_ref, "serialize"):
            try:
                self._json_snapshot = comp_ref.serialize()
            except Exception as exc:
                from InfEngine.debug import Debug
                Debug.log_warning(f"Undo: failed to snapshot component '{type_name}': {exc}")

    def execute(self) -> None:
        scene = _get_active_scene()
        if scene and self._comp_ref:
            obj = scene.find_by_id(self._object_id)
            if obj and hasattr(obj, "remove_component"):
                _invalidate_builtin_wrapper(self._comp_ref)
                obj.remove_component(self._comp_ref)
                _notify_gizmos_scene_changed()

    def undo(self) -> None:
        scene = _get_active_scene()
        if not scene:
            return
        obj = scene.find_by_id(self._object_id)
        if not obj:
            return
        result = obj.add_component(self._type_name)
        if result and self._json_snapshot and hasattr(result, "deserialize"):
            try:
                result.deserialize(self._json_snapshot)
            except Exception as exc:
                from InfEngine.debug import Debug
                Debug.log_warning(f"Undo: failed to restore component '{self._type_name}' state: {exc}")
        self._comp_ref = result
        _notify_gizmos_scene_changed()

    def redo(self) -> None:
        self.execute()


class RemovePyComponentCommand(UndoCommand):
    """Undo/redo removal of a Python component.

    Redo is not supported because recreating the exact Python component
    instance is non-trivial.
    """
    supports_redo = False

    def __init__(self, object_id: int, py_comp_ref: Any,
                 description: str = ""):
        type_name = getattr(py_comp_ref, 'type_name', 'Script')
        super().__init__(description or f"Remove {type_name}")
        self._object_id = object_id
        self._py_comp_ref = py_comp_ref

    def execute(self) -> None:
        scene = _get_active_scene()
        if scene and self._py_comp_ref:
            obj = scene.find_by_id(self._object_id)
            if obj and hasattr(obj, "remove_py_component"):
                obj.remove_py_component(self._py_comp_ref)

    def undo(self) -> None:
        scene = _get_active_scene()
        if scene and self._py_comp_ref:
            obj = scene.find_by_id(self._object_id)
            if obj and hasattr(obj, "add_py_component"):
                obj.add_py_component(self._py_comp_ref)


class AddPyComponentCommand(UndoCommand):
    """Record adding a Python component.  Undo removes it.

    Redo is not supported because recreating an arbitrary Python component
    instance with the exact same state is not trivial.
    """
    supports_redo = False

    def __init__(self, object_id: int, py_comp_ref: Any,
                 description: str = ""):
        super().__init__(
            description or f"Add {getattr(py_comp_ref, 'type_name', 'Script')}")
        self._object_id = object_id
        self._py_comp_ref = py_comp_ref

    def execute(self) -> None:
        pass  # already added before record()

    def undo(self) -> None:
        scene = _get_active_scene()
        if scene and self._py_comp_ref:
            obj = scene.find_by_id(self._object_id)
            if obj and hasattr(obj, "remove_py_component"):
                obj.remove_py_component(self._py_comp_ref)


class SelectionCommand(UndoCommand):
    """Record a selection change so it can be undone/redone.

    *apply_fn* receives a list of object IDs and is responsible for
    updating :class:`SelectionManager`, the Inspector, the hierarchy
    panel highlight, and the outline.
    """

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


class CompoundCommand(UndoCommand):
    """Group multiple commands into a single undo/redo unit."""

    def __init__(self, commands: List[UndoCommand], description: str = ""):
        desc = description or (commands[0].description if commands else "Compound")
        super().__init__(desc)
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_active_scene():
    """Return the active C++ Scene, or *None* on failure."""
    from InfEngine.lib import SceneManager
    return SceneManager.instance().get_active_scene()


def _notify_gizmos_scene_changed():
    """Notify the gizmos collector that the scene structure has changed."""
    from InfEngine.gizmos.collector import notify_scene_changed
    notify_scene_changed()


def _invalidate_builtin_wrapper(comp_ref):
    """Invalidate the BuiltinComponent wrapper for a C++ component being removed."""
    try:
        comp_id = comp_ref.component_id
    except Exception:
        return
    from InfEngine.components.builtin_component import BuiltinComponent
    wrapper = BuiltinComponent._wrapper_cache.get(comp_id)
    if wrapper is not None:
        wrapper._invalidate_native_binding()


def _preserve_ui_world_position(obj, new_parent) -> None:
    """Adjust UI element local x/y so that its world position is preserved after reparenting.

    Called *before* ``obj.set_parent(new_parent)``.  Only affects GameObjects
    that carry an ``InfUIScreenComponent``; silently no-ops otherwise.
    """
    from InfEngine.ui.inf_ui_screen_component import InfUIScreenComponent
    from InfEngine.ui import UICanvas

    # Find the screen-component on the dragged object
    ui_comp = None
    for comp in obj.get_py_components():
        if isinstance(comp, InfUIScreenComponent):
            ui_comp = comp
            break
    if ui_comp is None:
        return

    # Find the canvas that owns this element (walk up from current parent)
    def _find_canvas(go):
        while go is not None:
            for c in go.get_py_components():
                if isinstance(c, UICanvas):
                    return c
            go = go.get_parent()
        return None

    canvas = _find_canvas(obj.get_parent() or obj)
    if canvas is None:
        return
    cw = float(canvas.reference_width)
    ch = float(canvas.reference_height)

    # Compute current world rect (x, y, w, h) under the OLD parent
    world_x, world_y, w, h = ui_comp.get_rect(cw, ch)

    # Compute what the new parent's world rect will be
    if new_parent is not None:
        new_parent_ui = None
        for c in new_parent.get_py_components():
            if isinstance(c, InfUIScreenComponent):
                new_parent_ui = c
                break
        if new_parent_ui is not None:
            npx, npy, npw, nph = new_parent_ui.get_rect(cw, ch)
        else:
            npx, npy, npw, nph = 0.0, 0.0, cw, ch
    else:
        npx, npy, npw, nph = 0.0, 0.0, cw, ch

    # Recompute local x/y so that (npx + anchor_in_new_parent + x) == world_x
    anchor_x, anchor_y = ui_comp._anchor_origin(npw, nph)
    ui_comp.x = world_x - npx - anchor_x
    ui_comp.y = world_y - npy - anchor_y


def _recreate_game_object_from_json(json_str: str,
                                    parent_id: Optional[int],
                                    sibling_index: int) -> object:
    """Recreate a game object (and its children/components) from serialised JSON.

    Used by :class:`CreateGameObjectCommand` (redo) and
    :class:`DeleteGameObjectCommand` (undo) to restore previously-destroyed
    GameObjects, including their Python components.
    """
    scene = _get_active_scene()
    if not scene:
        return None

    # Create a placeholder object and deserialize into it.
    # Deserialize will overwrite the name, ID, and all components.
    obj = scene.create_game_object("__undo_restore__")
    if not obj:
        return None

    obj.deserialize(json_str)

    # Re-parent
    if parent_id is not None:
        parent = scene.find_by_id(parent_id)
        if parent:
            obj.set_parent(parent)

    # Restore sibling index
    if getattr(obj, "transform", None):
        obj.transform.set_sibling_index(sibling_index)

    # Restore Python components from the JSON payload
    data = _json_mod.loads(json_str)
    _restore_py_components_from_data(scene, data)

    # Initialise C++ components (MeshRenderer registration, etc.) that
    # were recreated by deserialize but not yet Awoken.
    scene.awake_object(obj)

    return obj


def _restore_py_components_from_data(scene, obj_data: dict) -> None:
    """Recursively restore Python components for an object tree from parsed JSON."""
    obj_id = obj_data.get("id")
    py_comps = obj_data.get("py_components")
    if py_comps and obj_id is not None:
        go = scene.find_by_id(obj_id)
        if go:
            _attach_py_components(go, py_comps)

    for child_data in obj_data.get("children", []):
        _restore_py_components_from_data(scene, child_data)


def _attach_py_components(go, py_comps_json: list) -> None:
    """Instantiate and attach Python components from a list of JSON dicts."""
    from InfEngine.engine.scene_manager import SceneFileManager
    from InfEngine.components.script_loader import load_and_create_component
    from InfEngine.components.registry import get_type

    sfm = SceneFileManager.instance()
    asset_db = sfm._asset_database if sfm else None

    for pc_json in py_comps_json:
        type_name = pc_json.get("py_type_name", "PyComponent")
        script_guid = pc_json.get("script_guid", "")
        enabled = pc_json.get("enabled", True)
        fields_json = ""
        if "py_fields" in pc_json:
            fields_json = (_json_mod.dumps(pc_json["py_fields"])
                           if isinstance(pc_json["py_fields"], dict)
                           else str(pc_json["py_fields"]))

        # Resolve script path from GUID
        script_path = None
        if script_guid and asset_db:
            script_path = asset_db.get_path_from_guid(script_guid)

        instance = None
        if script_path:
            instance = load_and_create_component(
                script_path, asset_database=asset_db)
        if instance is None:
            comp_class = get_type(type_name)
            if comp_class:
                instance = comp_class()

        if instance is None:
            # Create BrokenComponent placeholder to preserve data
            from InfEngine.components.component import BrokenComponent
            instance = BrokenComponent()
            instance._broken_type_name = type_name
            instance._script_guid = script_guid
            instance._broken_fields_json = fields_json or "{}"
            instance._broken_error = (
                f"Script not found for component '{type_name}' "
                f"(guid={script_guid}) during undo/redo"
            )

        if fields_json:
            instance._deserialize_fields(fields_json)
        instance.enabled = enabled
        go.add_py_component(instance)
        if hasattr(instance, "_call_on_after_deserialize"):
            instance._call_on_after_deserialize()


# ---------------------------------------------------------------------------
# UndoManager singleton
# ---------------------------------------------------------------------------

class UndoManager:
    """Central undo/redo manager.

    Maintains an undo and a redo stack of :class:`UndoCommand` objects with
    automatic merge for rapid successive property edits, save-point tracking,
    and integration with :class:`SceneFileManager` for dirty-flag management.
    """

    MAX_STACK_DEPTH: int = 200

    _instance: Optional[UndoManager] = None

    def __init__(self) -> None:
        self._undo_stack: List[UndoCommand] = []
        self._redo_stack: List[UndoCommand] = []
        # ``None`` means the save state has been evicted from the history.
        self._save_point: Optional[int] = 0
        # Dirty baseline for states that intentionally have no usable history.
        self._base_scene_dirty: bool = False
        self._is_executing: bool = False
        self._enabled: bool = True
        self._suppress_property_recording: bool = False
        self._on_state_changed: Optional[Callable[[], None]] = None
        UndoManager._instance = self

    @classmethod
    def instance(cls) -> Optional[UndoManager]:
        """Return the singleton, or *None* if not yet created."""
        return cls._instance

    @contextmanager
    def suppress(self):
        """Context manager to suppress auto-recording by ``SerializedFieldDescriptor``.

        While active, ``is_executing`` returns *True* so that descriptor
        ``__set__`` calls skip undo recording.  Use this around continuous
        interactions (drag, resize, rotate) and record a single undo
        command manually at the end.
        """
        prev = self._is_executing
        self._is_executing = True
        try:
            yield
        finally:
            self._is_executing = prev

    @contextmanager
    def suppress_property_recording(self):
        """Temporarily suppress recording of property-edit commands.

        Commands flagged with ``_is_property_edit = True`` still *execute*
        (values are written) but are not pushed to the undo/redo stacks.
        Structural commands (create/delete/reparent/add-remove component)
        are unaffected and continue to record normally.

        Used by :class:`InspectorUndoTracker` to prevent double-recording
        when the tracker captures panel-level snapshots.
        """
        prev = self._suppress_property_recording
        self._suppress_property_recording = True
        try:
            yield
        finally:
            self._suppress_property_recording = prev

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_executing(self) -> bool:
        """*True* while a command is being executed / undone / redone.

        Used by :class:`SerializedFieldDescriptor` to skip auto-recording
        when the undo system itself is driving the property change.
        """
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
        """Count of commands in the undo stack that mark the scene dirty."""
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

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def execute(self, cmd: UndoCommand) -> None:
        """Execute *cmd*, push it onto the undo stack, and clear the redo
        stack.  Automatically merges with the stack-top command if possible.
        """
        if not self._enabled:
            # disabled → still execute, just don't record history
            cmd.execute()
            return

        self._is_executing = True
        cmd.execute()
        self._is_executing = False

        # Skip recording for property-edit commands when suppressed by
        # InspectorUndoTracker (the tracker captures panel-level snapshots).
        if self._suppress_property_recording and cmd._is_property_edit:
            return

        self._push(cmd)

    def record(self, cmd: UndoCommand) -> None:
        """Push an **already-executed** command onto the undo stack.

        Use this when the action was performed outside the command's
        :meth:`~UndoCommand.execute` (e.g. complex creation logic in the
        hierarchy panel).
        """
        if not self._enabled:
            return
        if self._suppress_property_recording and cmd._is_property_edit:
            return
        self._push(cmd)

    def undo(self) -> None:
        """Undo the most recent command."""
        if not self._undo_stack:
            return
        cmd = self._undo_stack.pop()
        self._is_executing = True
        cmd.undo()
        self._is_executing = False

        if cmd.supports_redo:
            self._redo_stack.append(cmd)

        self._sync_dirty()
        self._fire_state_changed()

    def redo(self) -> None:
        """Redo the most recently undone command."""
        if not self._redo_stack:
            return
        cmd = self._redo_stack.pop()
        self._is_executing = True
        cmd.redo()
        self._is_executing = False

        self._undo_stack.append(cmd)

        self._sync_dirty()
        self._fire_state_changed()

    def clear(self, scene_is_dirty: bool = False) -> None:
        """Clear both stacks and reset the dirty baseline."""
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._save_point = 0
        self._base_scene_dirty = bool(scene_is_dirty)
        self._fire_state_changed()

    def mark_save_point(self) -> None:
        """Record the current undo depth as the *clean* state.

        Called by :class:`SceneFileManager` after a successful save.
        """
        self._save_point = self._dirty_depth
        self._base_scene_dirty = False

    def sync_dirty_state(self) -> None:
        """Re-apply dirty state to SceneFileManager from current history."""
        self._sync_dirty()

    def set_on_state_changed(self, cb: Optional[Callable[[], None]]) -> None:
        self._on_state_changed = cb

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _push(self, cmd: UndoCommand) -> None:
        """Push *cmd*, merge if possible, enforce depth limit, clear redo."""
        if self._undo_stack and self._undo_stack[-1].can_merge(cmd):
            self._undo_stack[-1].merge(cmd)
        else:
            self._undo_stack.append(cmd)
            # enforce depth limit
            if len(self._undo_stack) > self.MAX_STACK_DEPTH:
                overflow = len(self._undo_stack) - self.MAX_STACK_DEPTH
                dirty_dropped = sum(1 for c in self._undo_stack[:overflow] if c.marks_dirty)
                del self._undo_stack[:overflow]
                if self._save_point is not None:
                    self._save_point -= dirty_dropped
                    if self._save_point < 0:
                        self._save_point = None  # save state lost

        self._redo_stack.clear()
        self._sync_dirty()
        self._fire_state_changed()

    def _sync_dirty(self) -> None:
        """Update :class:`SceneFileManager` dirty flag based on save-point.

        Skipped in play mode — runtime property tweaks are transient and
        must not mark the scene file as dirty.
        """
        from InfEngine.engine.play_mode import PlayModeManager, PlayModeState
        pm = PlayModeManager.instance()
        if pm and pm.state != PlayModeState.EDIT:
            return
        from InfEngine.engine.scene_manager import SceneFileManager
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


# ---------------------------------------------------------------------------
# Unified Inspector Undo — Snapshot-based panel-level tracking
# ---------------------------------------------------------------------------

class InspectorSnapshotCommand(UndoCommand):
    """Snapshot-based undo command for Inspector panel edits.

    Records the serialized state of an inspectable target before and after
    an edit.  A single command replaces the many per-widget command types
    (SetPropertyCommand, RenderStackFieldCommand, MaterialJsonCommand, …).

    Supports automatic merging for continuous edits within
    :attr:`MERGE_WINDOW` seconds (e.g. slider drags).
    """

    MERGE_WINDOW: float = 0.3

    def __init__(self, target_key: str, old_snapshot: str,
                 new_snapshot: str,
                 restore_fn: Callable[[str], None],
                 description: str = ""):
        super().__init__(description or "Inspector Edit")
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

    def merge(self, other: InspectorSnapshotCommand) -> None:  # type: ignore[override]
        self._new_snapshot = other._new_snapshot
        self.timestamp = other.timestamp


class _TrackedEntry:
    """Internal bookkeeping for a single tracked target within a frame."""
    __slots__ = ('pre_snapshot', 'snapshot_fn', 'restore_fn', 'description')

    def __init__(self, pre_snapshot: str,
                 snapshot_fn: Callable[[], str],
                 restore_fn: Callable[[str], None],
                 description: str):
        self.pre_snapshot = pre_snapshot
        self.snapshot_fn = snapshot_fn
        self.restore_fn = restore_fn
        self.description = description


class InspectorUndoTracker:
    """Automatic undo tracking for Inspector and Hierarchy panels.

    Replaces scattered per-widget undo recording with a unified
    panel-level snapshot approach:

    1. ``begin_frame()`` — clear previous entries.
    2. ``track(key, snapshot_fn, restore_fn, desc)`` — register each
       inspectable target (component, material, RenderStack, …).
       A pre-edit snapshot is captured immediately.
    3. Inspector widgets render and mutate state as usual (old undo calls
       are suppressed via ``UndoManager.suppress_property_recording``).
    4. ``end_frame()`` — re-capture each target's state, compare with
       pre-snapshot, and record :class:`InspectorSnapshotCommand` for
       any detected changes.

    Benefits:
    - **No forgotten undo**: every property visible in the Inspector is
      automatically covered.
    - **Uniform**: one command type handles Transform, components,
      materials, and RenderStack edits.
    - **Merge**: slider drags produce a single undo entry (0.3 s window).
    """

    def __init__(self):
        self._entries: dict[str, _TrackedEntry] = {}

    def begin_frame(self) -> None:
        """Start tracking for a new render frame."""
        self._entries.clear()

    def track(self, key: str, snapshot_fn: Callable[[], str],
              restore_fn: Callable[[str], None],
              description: str = "") -> None:
        """Register a target for change tracking.

        Args:
            key: Stable identity string for merge across frames.
            snapshot_fn: Returns the target's serialized state.
            restore_fn: Restores the target from a serialized state.
            description: Human-readable label for the undo menu entry.
        """
        if key in self._entries:
            return
        try:
            pre = snapshot_fn()
        except Exception:
            return
        self._entries[key] = _TrackedEntry(
            pre, snapshot_fn, restore_fn, description)

    def end_frame(self) -> None:
        """Compare pre/post snapshots and record undo for any changes."""
        mgr = UndoManager.instance()
        if not mgr or not mgr.enabled:
            self._entries.clear()
            return

        for key, entry in self._entries.items():
            try:
                post = entry.snapshot_fn()
            except Exception:
                continue
            if post != entry.pre_snapshot:
                cmd = InspectorSnapshotCommand(
                    key, entry.pre_snapshot, post,
                    entry.restore_fn, entry.description,
                )
                mgr.record(cmd)

        self._entries.clear()


# ---------------------------------------------------------------------------
# RenderStack snapshot / restore helpers
# ---------------------------------------------------------------------------

def _serialize_simple(val: Any) -> Any:
    """Serialize a value for JSON snapshot (handles common field types)."""
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
    """Capture the complete state of a RenderStack as a JSON string.

    Captures pipeline selection, pipeline parameters, and all mounted
    effects with their parameters, enabled state, and ordering.
    """
    from InfEngine.components.serialized_field import get_serialized_fields

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

    return _json_mod.dumps(data, sort_keys=True)


def restore_renderstack(stack: Any, json_str: str) -> None:
    """Restore a RenderStack from a snapshot produced by :func:`snapshot_renderstack`."""
    from InfEngine.components.serialized_field import get_serialized_fields

    data = _json_mod.loads(json_str)

    # --- Pipeline selection ---
    new_pipeline_name = data.get("pipeline_class_name", "")
    if (stack.pipeline_class_name or "") != new_pipeline_name:
        stack.set_pipeline(new_pipeline_name)

    # --- Pipeline parameters ---
    pipeline = stack.pipeline
    if pipeline and "pipeline_params" in data:
        for name, val in data["pipeline_params"].items():
            try:
                setattr(pipeline, name, val)
            except (AttributeError, TypeError):
                pass

    # --- Effect pass entries ---
    current_names = [e.render_pass.name for e in list(stack.pass_entries)]
    for name in current_names:
        stack.remove_pass(name)

    from InfEngine.renderstack.discovery import discover_passes
    from InfEngine.renderstack.fullscreen_effect import FullScreenEffect

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


# ---------------------------------------------------------------------------
# Hierarchy undo tracker
# ---------------------------------------------------------------------------

class HierarchyUndoTracker:
    """Unified undo interface for Hierarchy panel operations.

    Provides a single entry-point for all hierarchy mutations —
    create, delete, reparent, and reorder — so the panel code does not
    need to import individual command classes or call ``UndoManager``
    directly.  Selection changes are intentionally **not** tracked
    (consistent with Unity's behaviour).

    Usage in HierarchyPanel::

        self._undo = HierarchyUndoTracker()
        # after creating a primitive:
        self._undo.record_create(new_obj.id, "Create Cube")
        # delete:
        self._undo.record_delete(obj.id)
        # reparent:
        self._undo.record_reparent(obj_id, old_parent_id, new_parent_id)
        # reorder (parent + sibling):
        self._undo.record_move(obj_id, old_pid, new_pid, old_idx, new_idx)
    """

    @staticmethod
    def _mgr() -> Optional['UndoManager']:
        return UndoManager.instance()

    # -- create --------------------------------------------------------
    def record_create(self, object_id: int,
                      description: str = "Create GameObject") -> None:
        """Record that *object_id* was just created (already exists in scene)."""
        mgr = self._mgr()
        if mgr:
            mgr.record(CreateGameObjectCommand(object_id, description))
            return
        # Fallback: just mark dirty
        from InfEngine.engine.scene_manager import SceneFileManager
        sfm = SceneFileManager.instance()
        if sfm:
            sfm.mark_dirty()

    # -- delete --------------------------------------------------------
    def record_delete(self, object_id: int,
                      description: str = "Delete GameObject") -> None:
        """Snapshot the object, then destroy it via the undo system."""
        mgr = self._mgr()
        if mgr:
            mgr.execute(DeleteGameObjectCommand(object_id, description))
            return
        # Fallback when undo is unavailable
        scene = _get_active_scene()
        if scene:
            obj = scene.find_by_id(object_id)
            if obj:
                scene.destroy_game_object(obj)
                from InfEngine.engine.scene_manager import SceneFileManager
                sfm = SceneFileManager.instance()
                if sfm:
                    sfm.mark_dirty()

    # -- reparent ------------------------------------------------------
    def record_reparent(self, object_id: int,
                        old_parent_id: Optional[int],
                        new_parent_id: Optional[int],
                        description: str = "Reparent") -> None:
        """Record a parent change."""
        mgr = self._mgr()
        if mgr:
            mgr.execute(ReparentCommand(
                object_id, old_parent_id, new_parent_id, description))
            return
        # Fallback
        scene = _get_active_scene()
        if scene:
            obj = scene.find_by_id(object_id)
            if obj:
                parent = (scene.find_by_id(new_parent_id)
                          if new_parent_id is not None else None)
                _preserve_ui_world_position(obj, parent)
                obj.set_parent(parent)
                from InfEngine.engine.scene_manager import SceneFileManager
                sfm = SceneFileManager.instance()
                if sfm:
                    sfm.mark_dirty()

    # -- move (parent + sibling order) ---------------------------------
    def record_move(self, object_id: int,
                    old_parent_id: Optional[int],
                    new_parent_id: Optional[int],
                    old_sibling_index: int,
                    new_sibling_index: int,
                    description: str = "Move In Hierarchy") -> None:
        """Record a combined parent + sibling-order change."""
        mgr = self._mgr()
        if mgr:
            mgr.execute(MoveGameObjectCommand(
                object_id, old_parent_id, new_parent_id,
                old_sibling_index, new_sibling_index, description))
            return
        # Fallback
        scene = _get_active_scene()
        if scene:
            obj = scene.find_by_id(object_id)
            if obj:
                parent = (scene.find_by_id(new_parent_id)
                          if new_parent_id is not None else None)
                _preserve_ui_world_position(obj, parent)
                obj.set_parent(parent)
                transform = getattr(obj, 'transform', None)
                if transform is not None:
                    transform.set_sibling_index(max(0, int(new_sibling_index)))
                from InfEngine.engine.scene_manager import SceneFileManager
                sfm = SceneFileManager.instance()
                if sfm:
                    sfm.mark_dirty()
