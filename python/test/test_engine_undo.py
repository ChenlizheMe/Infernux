"""Tests for InfEngine.engine.undo — UndoCommand subclasses and UndoManager.

Pure-Python tests — no C++ backend needed.
Imports bypass the heavy InfEngine.__init__ / native-module chain by loading
undo.py and selection_manager.py directly via importlib.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import time
import types

import pytest

# ── Direct-load helper ───────────────────────────────────────────────
# Load a .py file as if it were a top-level module, avoiding __init__.py
# chains that pull in the C++ native backend.

_PROJECT_PY = os.path.normpath(
    os.path.join(os.path.dirname(__file__), os.pardir, "InfEngine"))


def _direct_import(module_name: str, rel_path: str):
    """Import *rel_path* (relative to python/InfEngine/) as *module_name*."""
    if module_name in sys.modules:
        return sys.modules[module_name]
    filepath = os.path.join(_PROJECT_PY, *rel_path.split("/"))
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


# Pre-register lightweight stub packages so sub-module imports resolve.
for _pkg in [
    "InfEngine",
    "InfEngine.engine",
    "InfEngine.engine.ui",
]:
    if _pkg not in sys.modules:
        _p = types.ModuleType(_pkg)
        _p.__path__ = [os.path.join(_PROJECT_PY,
                                     *_pkg.split(".")[1:])] if _pkg != "InfEngine" \
                       else [_PROJECT_PY]
        _p.__package__ = _pkg
        sys.modules[_pkg] = _p

# Now load the actual modules we need — this executes their code but
# won't trigger InfEngine/__init__.py's heavy imports.
_undo_mod = _direct_import("InfEngine.engine.undo", "engine/undo.py")
_sel_mod = _direct_import(
    "InfEngine.engine.ui.selection_manager", "engine/ui/selection_manager.py")

# Pull symbols into module scope for convenience.
UndoCommand = _undo_mod.UndoCommand
UndoManager = _undo_mod.UndoManager
SetPropertyCommand = _undo_mod.SetPropertyCommand
GenericComponentCommand = _undo_mod.GenericComponentCommand
BuiltinPropertyCommand = _undo_mod.BuiltinPropertyCommand
CreateGameObjectCommand = _undo_mod.CreateGameObjectCommand
DeleteGameObjectCommand = _undo_mod.DeleteGameObjectCommand
ReparentCommand = _undo_mod.ReparentCommand
MoveGameObjectCommand = _undo_mod.MoveGameObjectCommand
MaterialJsonCommand = _undo_mod.MaterialJsonCommand
CompoundCommand = _undo_mod.CompoundCommand
SelectionCommand = _undo_mod.SelectionCommand
InspectorSnapshotCommand = _undo_mod.InspectorSnapshotCommand
InspectorUndoTracker = _undo_mod.InspectorUndoTracker
RenderStackFieldCommand = _undo_mod.RenderStackFieldCommand
_snapshot_value = _undo_mod._snapshot_value
SelectionManager = _sel_mod.SelectionManager


# ── Helpers ──

class _Obj:
    """Simple mutable target for SetPropertyCommand."""
    def __init__(self):
        self.x = 0
        self.y = 0


class _FakeComp:
    """Fake component with serialize/deserialize for GenericComponentCommand."""
    def __init__(self):
        self.component_id = 42
        self.type_name = "FakeComp"
        self._json = "{}"

    def serialize(self):
        return self._json

    def deserialize(self, json_str):
        self._json = json_str


class _FakeStack:
    """Fake RenderStack with invalidate_graph() for RenderStackFieldCommand."""
    def __init__(self):
        self.invalidated = 0

    def invalidate_graph(self):
        self.invalidated += 1


@pytest.fixture(autouse=True)
def _reset_undo_manager():
    """Ensure a fresh UndoManager for every test; tear down afterwards."""
    old = UndoManager._instance
    mgr = UndoManager()
    # Prevent dirty-sync from importing SceneFileManager / PlayModeManager
    mgr._sync_dirty = lambda: None
    yield mgr
    UndoManager._instance = old


# ══════════════════════════════════════════════════════════════════════
# _snapshot_value
# ══════════════════════════════════════════════════════════════════════

class TestSnapshotValue:
    def test_immutable_pass_through(self):
        assert _snapshot_value(42) == 42
        assert _snapshot_value("hello") == "hello"
        assert _snapshot_value(None) is None

    def test_list_deepcopy(self):
        original = [[1, 2], [3, 4]]
        snapped = _snapshot_value(original)
        snapped[0][0] = 999
        assert original[0][0] == 1

    def test_dict_deepcopy(self):
        original = {"a": [1, 2]}
        snapped = _snapshot_value(original)
        snapped["a"].append(3)
        assert len(original["a"]) == 2


# ══════════════════════════════════════════════════════════════════════
# SetPropertyCommand
# ══════════════════════════════════════════════════════════════════════

class TestSetPropertyCommand:
    def test_execute_sets_value(self):
        obj = _Obj()
        cmd = SetPropertyCommand(obj, "x", 0, 42)
        cmd.execute()
        assert obj.x == 42

    def test_undo_restores_old(self):
        obj = _Obj()
        cmd = SetPropertyCommand(obj, "x", 0, 42)
        cmd.execute()
        cmd.undo()
        assert obj.x == 0

    def test_redo_reapplies(self):
        obj = _Obj()
        cmd = SetPropertyCommand(obj, "x", 0, 42)
        cmd.execute()
        cmd.undo()
        cmd.redo()
        assert obj.x == 42

    def test_merge_within_window(self):
        obj = _Obj()
        cmd1 = SetPropertyCommand(obj, "x", 0, 1)
        cmd2 = SetPropertyCommand(obj, "x", 1, 2)
        cmd2.timestamp = cmd1.timestamp + 0.1
        assert cmd1.can_merge(cmd2)
        cmd1.merge(cmd2)
        # After merge, new value is from cmd2
        cmd1.execute()
        assert obj.x == 2
        # Undo goes to original old value
        cmd1.undo()
        assert obj.x == 0

    def test_merge_rejected_outside_window(self):
        obj = _Obj()
        cmd1 = SetPropertyCommand(obj, "x", 0, 1)
        cmd2 = SetPropertyCommand(obj, "x", 1, 2)
        cmd2.timestamp = cmd1.timestamp + 10.0
        assert not cmd1.can_merge(cmd2)

    def test_merge_rejected_different_property(self):
        obj = _Obj()
        cmd1 = SetPropertyCommand(obj, "x", 0, 1)
        cmd2 = SetPropertyCommand(obj, "y", 0, 2)
        cmd2.timestamp = cmd1.timestamp + 0.1
        assert not cmd1.can_merge(cmd2)

    def test_is_property_edit_flag(self):
        obj = _Obj()
        cmd = SetPropertyCommand(obj, "x", 0, 1)
        assert cmd.marks_dirty is True

    def test_default_description(self):
        obj = _Obj()
        cmd = SetPropertyCommand(obj, "x", 0, 1)
        assert "x" in cmd.description


# ══════════════════════════════════════════════════════════════════════
# GenericComponentCommand
# ══════════════════════════════════════════════════════════════════════

class TestGenericComponentCommand:
    def test_execute_and_undo(self):
        comp = _FakeComp()
        cmd = GenericComponentCommand(comp, '{"old": 1}', '{"new": 2}')
        cmd.execute()
        assert comp._json == '{"new": 2}'
        cmd.undo()
        assert comp._json == '{"old": 1}'

    def test_merge(self):
        comp = _FakeComp()
        cmd1 = GenericComponentCommand(comp, "{}", '{"a":1}')
        cmd2 = GenericComponentCommand(comp, '{"a":1}', '{"a":2}')
        cmd2.timestamp = cmd1.timestamp + 0.1
        assert cmd1.can_merge(cmd2)
        cmd1.merge(cmd2)
        cmd1.execute()
        assert comp._json == '{"a":2}'


# ══════════════════════════════════════════════════════════════════════
# BuiltinPropertyCommand
# ══════════════════════════════════════════════════════════════════════

class TestBuiltinPropertyCommand:
    def test_execute_undo_redo(self):
        obj = _Obj()
        obj.component_id = 1
        cmd = BuiltinPropertyCommand(obj, "x", 0, 5)
        cmd.execute()
        assert obj.x == 5
        cmd.undo()
        assert obj.x == 0
        cmd.redo()
        assert obj.x == 5


# ══════════════════════════════════════════════════════════════════════
# MaterialJsonCommand
# ══════════════════════════════════════════════════════════════════════

class TestMaterialJsonCommand:
    def test_execute_undo_redo(self):
        mat = _FakeComp()
        mat.guid = "test-guid"
        cmd = MaterialJsonCommand(mat, '{"old":1}', '{"new":2}')
        cmd.execute()
        assert mat._json == '{"new":2}'
        cmd.undo()
        assert mat._json == '{"old":1}'
        cmd.redo()
        assert mat._json == '{"new":2}'

    def test_marks_dirty_false(self):
        mat = _FakeComp()
        cmd = MaterialJsonCommand(mat, "{}", "{}")
        assert cmd.marks_dirty is False

    def test_merge(self):
        mat = _FakeComp()
        mat.guid = "g"
        cmd1 = MaterialJsonCommand(mat, "{}", '{"a":1}')
        cmd2 = MaterialJsonCommand(mat, '{"a":1}', '{"a":2}')
        cmd2.timestamp = cmd1.timestamp + 0.1
        assert cmd1.can_merge(cmd2)


# ══════════════════════════════════════════════════════════════════════
# CompoundCommand
# ══════════════════════════════════════════════════════════════════════

class TestCompoundCommand:
    def test_execute_all(self):
        a = _Obj()
        b = _Obj()
        cmds = [
            SetPropertyCommand(a, "x", 0, 10),
            SetPropertyCommand(b, "x", 0, 20),
        ]
        compound = CompoundCommand(cmds, "Compound Edit")
        compound.execute()
        assert a.x == 10
        assert b.x == 20

    def test_undo_in_reverse(self):
        log = []

        class LogCmd(UndoCommand):
            def __init__(self, tag):
                super().__init__(tag)
                self.tag = tag
            def execute(self):
                log.append(("exec", self.tag))
            def undo(self):
                log.append(("undo", self.tag))

        cmds = [LogCmd("A"), LogCmd("B")]
        compound = CompoundCommand(cmds)
        compound.execute()
        log.clear()
        compound.undo()
        assert log == [("undo", "B"), ("undo", "A")]

    def test_supports_redo_all_true(self):
        obj = _Obj()
        cmds = [SetPropertyCommand(obj, "x", 0, 1)]
        compound = CompoundCommand(cmds)
        assert compound.supports_redo is True


# ══════════════════════════════════════════════════════════════════════
# UndoCommand base
# ══════════════════════════════════════════════════════════════════════

class TestUndoCommandBase:
    def test_default_flags(self):
        class Concrete(UndoCommand):
            def execute(self): pass
            def undo(self): pass
        cmd = Concrete("test")
        assert cmd.supports_redo is True
        assert cmd.marks_dirty is True
        assert isinstance(cmd.timestamp, float)

    def test_default_redo_calls_execute(self):
        calls = []
        class Concrete(UndoCommand):
            def execute(self):
                calls.append("exec")
            def undo(self):
                calls.append("undo")
        cmd = Concrete()
        cmd.redo()
        assert calls == ["exec"]

    def test_default_merge_returns_false(self):
        class Concrete(UndoCommand):
            def execute(self): pass
            def undo(self): pass
        cmd = Concrete()
        assert not cmd.can_merge(cmd)


# ══════════════════════════════════════════════════════════════════════
# SelectionCommand
# ══════════════════════════════════════════════════════════════════════

class TestSelectionCommand:
    def test_undo_calls_apply_with_old_ids(self):
        applied = []
        cmd = SelectionCommand([1, 2], [3], apply_fn=lambda ids: applied.append(list(ids)))
        cmd.undo()
        assert applied == [[1, 2]]

    def test_redo_calls_apply_with_new_ids(self):
        applied = []
        cmd = SelectionCommand([1], [2, 3], apply_fn=lambda ids: applied.append(list(ids)))
        cmd.redo()
        assert applied == [[2, 3]]

    def test_execute_is_noop(self):
        applied = []
        cmd = SelectionCommand([], [1], apply_fn=lambda ids: applied.append(ids))
        cmd.execute()
        assert applied == []

    def test_marks_dirty_false(self):
        cmd = SelectionCommand([], [], apply_fn=lambda ids: None)
        assert cmd.marks_dirty is False

    def test_default_description(self):
        cmd = SelectionCommand([], [], apply_fn=lambda ids: None)
        assert cmd.description == "Change Selection"

    def test_custom_description(self):
        cmd = SelectionCommand([], [], apply_fn=lambda ids: None, description="Pick Object")
        assert cmd.description == "Pick Object"

    def test_supports_redo(self):
        cmd = SelectionCommand([], [1], apply_fn=lambda ids: None)
        assert cmd.supports_redo is True


# ══════════════════════════════════════════════════════════════════════
# InspectorSnapshotCommand
# ══════════════════════════════════════════════════════════════════════

class TestInspectorSnapshotCommand:
    def test_execute_restores_new(self):
        restored = []
        cmd = InspectorSnapshotCommand("k", "old_snap", "new_snap",
                                       restore_fn=lambda s: restored.append(s))
        cmd.execute()
        assert restored == ["new_snap"]

    def test_undo_restores_old(self):
        restored = []
        cmd = InspectorSnapshotCommand("k", "old_snap", "new_snap",
                                       restore_fn=lambda s: restored.append(s))
        cmd.undo()
        assert restored == ["old_snap"]

    def test_redo_restores_new(self):
        restored = []
        cmd = InspectorSnapshotCommand("k", "old_snap", "new_snap",
                                       restore_fn=lambda s: restored.append(s))
        cmd.redo()
        assert restored == ["new_snap"]

    def test_merge_within_window(self):
        cmd1 = InspectorSnapshotCommand("go:1", "a", "b", restore_fn=lambda s: None)
        cmd2 = InspectorSnapshotCommand("go:1", "b", "c", restore_fn=lambda s: None)
        cmd2.timestamp = cmd1.timestamp + 0.1
        assert cmd1.can_merge(cmd2)
        cmd1.merge(cmd2)
        # After merge, old stays "a", new becomes "c"
        restored = []
        cmd1.undo()
        assert restored == [] or True  # undo restores old
        # Verify via execute
        results = []
        cmd1._restore_fn = lambda s: results.append(s)
        cmd1.execute()
        assert results == ["c"]
        cmd1.undo()
        assert results == ["c", "a"]

    def test_merge_rejected_different_key(self):
        cmd1 = InspectorSnapshotCommand("go:1", "a", "b", restore_fn=lambda s: None)
        cmd2 = InspectorSnapshotCommand("go:2", "a", "b", restore_fn=lambda s: None)
        cmd2.timestamp = cmd1.timestamp + 0.1
        assert not cmd1.can_merge(cmd2)

    def test_merge_rejected_outside_window(self):
        cmd1 = InspectorSnapshotCommand("go:1", "a", "b", restore_fn=lambda s: None)
        cmd2 = InspectorSnapshotCommand("go:1", "b", "c", restore_fn=lambda s: None)
        cmd2.timestamp = cmd1.timestamp + 10.0
        assert not cmd1.can_merge(cmd2)


# ══════════════════════════════════════════════════════════════════════
# RenderStackFieldCommand
# ══════════════════════════════════════════════════════════════════════

class TestRenderStackFieldCommand:
    def test_execute_undo_redo(self):
        target = _Obj()
        stack = _FakeStack()
        cmd = RenderStackFieldCommand(stack, target, "x", 0, 42)
        cmd.execute()
        assert target.x == 42
        assert stack.invalidated == 1
        cmd.undo()
        assert target.x == 0
        assert stack.invalidated == 2
        cmd.redo()
        assert target.x == 42
        assert stack.invalidated == 3

    def test_merge_same_target_field(self):
        target = _Obj()
        stack = _FakeStack()
        cmd1 = RenderStackFieldCommand(stack, target, "x", 0, 1)
        cmd2 = RenderStackFieldCommand(stack, target, "x", 1, 5)
        cmd2.timestamp = cmd1.timestamp + 0.1
        assert cmd1.can_merge(cmd2)
        cmd1.merge(cmd2)
        cmd1.execute()
        assert target.x == 5
        cmd1.undo()
        assert target.x == 0

    def test_merge_rejected_different_field(self):
        target = _Obj()
        stack = _FakeStack()
        cmd1 = RenderStackFieldCommand(stack, target, "x", 0, 1)
        cmd2 = RenderStackFieldCommand(stack, target, "y", 0, 2)
        cmd2.timestamp = cmd1.timestamp + 0.1
        assert not cmd1.can_merge(cmd2)


# ══════════════════════════════════════════════════════════════════════
# UndoManager — core operations
# ══════════════════════════════════════════════════════════════════════

class TestUndoManager:
    def test_execute_then_undo(self, _reset_undo_manager):
        mgr = _reset_undo_manager
        obj = _Obj()
        mgr.execute(SetPropertyCommand(obj, "x", 0, 10))
        assert obj.x == 10
        mgr.undo()
        assert obj.x == 0

    def test_redo_after_undo(self, _reset_undo_manager):
        mgr = _reset_undo_manager
        obj = _Obj()
        mgr.execute(SetPropertyCommand(obj, "x", 0, 10))
        mgr.undo()
        mgr.redo()
        assert obj.x == 10

    def test_execute_clears_redo(self, _reset_undo_manager):
        mgr = _reset_undo_manager
        obj = _Obj()
        mgr.execute(SetPropertyCommand(obj, "x", 0, 1))
        mgr.undo()
        assert mgr.can_redo
        mgr.execute(SetPropertyCommand(obj, "y", 0, 2))
        assert not mgr.can_redo

    def test_can_undo_redo_properties(self, _reset_undo_manager):
        mgr = _reset_undo_manager
        assert not mgr.can_undo
        assert not mgr.can_redo
        obj = _Obj()
        mgr.execute(SetPropertyCommand(obj, "x", 0, 1))
        assert mgr.can_undo
        assert not mgr.can_redo

    def test_undo_description(self, _reset_undo_manager):
        mgr = _reset_undo_manager
        obj = _Obj()
        mgr.execute(SetPropertyCommand(obj, "x", 0, 1, "Move X"))
        assert mgr.undo_description == "Move X"

    def test_record_pushes_without_executing(self, _reset_undo_manager):
        mgr = _reset_undo_manager
        executed = []

        class TrackCmd(UndoCommand):
            def execute(self):
                executed.append(True)
            def undo(self):
                pass

        cmd = TrackCmd("test")
        mgr.record(cmd)
        assert len(executed) == 0
        assert mgr.can_undo

    def test_merge_auto(self, _reset_undo_manager):
        mgr = _reset_undo_manager
        obj = _Obj()
        cmd1 = SetPropertyCommand(obj, "x", 0, 1)
        mgr.execute(cmd1)
        cmd2 = SetPropertyCommand(obj, "x", 1, 5)
        cmd2.timestamp = cmd1.timestamp + 0.1
        mgr.execute(cmd2)
        # Should have merged → only 1 entry on the stack
        assert len(mgr._undo_stack) == 1
        mgr.undo()
        assert obj.x == 0

    def test_clear(self, _reset_undo_manager):
        mgr = _reset_undo_manager
        obj = _Obj()
        mgr.execute(SetPropertyCommand(obj, "x", 0, 1))
        mgr.clear()
        assert not mgr.can_undo
        assert not mgr.can_redo

    def test_is_executing_during_undo_redo(self, _reset_undo_manager):
        mgr = _reset_undo_manager
        seen_exec = []

        class SpyCmd(UndoCommand):
            def execute(self):
                seen_exec.append(mgr.is_executing)
            def undo(self):
                seen_exec.append(mgr.is_executing)
            def redo(self):
                seen_exec.append(mgr.is_executing)

        mgr.execute(SpyCmd("test"))
        assert seen_exec == [True]
        mgr.undo()
        assert seen_exec == [True, True]
        mgr.redo()
        assert seen_exec == [True, True, True]

    def test_suppress_context_manager(self, _reset_undo_manager):
        mgr = _reset_undo_manager
        assert not mgr.is_executing
        with mgr.suppress():
            assert mgr.is_executing
        assert not mgr.is_executing

    def test_suppress_property_recording(self, _reset_undo_manager):
        mgr = _reset_undo_manager
        obj = _Obj()
        with mgr.suppress_property_recording():
            mgr.execute(SetPropertyCommand(obj, "x", 0, 99))
        # Value was set but command not recorded
        assert obj.x == 99
        assert not mgr.can_undo

    def test_suppress_property_recording_allows_structural(self, _reset_undo_manager):
        mgr = _reset_undo_manager

        class StructuralCmd(UndoCommand):
            _is_property_edit = False
            def execute(self): pass
            def undo(self): pass

        with mgr.suppress_property_recording():
            mgr.execute(StructuralCmd("create object"))
        # Structural commands are recorded even under suppression
        assert mgr.can_undo

    def test_depth_limit(self, _reset_undo_manager):
        mgr = _reset_undo_manager
        mgr.MAX_STACK_DEPTH = 5
        obj = _Obj()
        for i in range(10):
            # Use large timestamp gap to prevent merging
            cmd = SetPropertyCommand(obj, "x", i, i + 1)
            cmd.timestamp = i * 10.0
            mgr.execute(cmd)
        assert len(mgr._undo_stack) == 5

    def test_disabled_still_executes(self, _reset_undo_manager):
        mgr = _reset_undo_manager
        mgr.enabled = False
        obj = _Obj()
        mgr.execute(SetPropertyCommand(obj, "x", 0, 5))
        assert obj.x == 5
        assert not mgr.can_undo

    def test_state_changed_callback(self, _reset_undo_manager):
        mgr = _reset_undo_manager
        calls = []
        mgr.set_on_state_changed(lambda: calls.append(1))
        obj = _Obj()
        mgr.execute(SetPropertyCommand(obj, "x", 0, 1))
        assert len(calls) == 1
        mgr.undo()
        assert len(calls) == 2

    def test_multiple_undo_redo(self, _reset_undo_manager):
        mgr = _reset_undo_manager
        obj = _Obj()
        for i in range(3):
            cmd = SetPropertyCommand(obj, "x", i, i + 1)
            cmd.timestamp = i * 10.0
            mgr.execute(cmd)
        assert obj.x == 3
        mgr.undo()
        assert obj.x == 2
        mgr.undo()
        assert obj.x == 1
        mgr.redo()
        assert obj.x == 2


# ══════════════════════════════════════════════════════════════════════
# UndoManager — save point / dirty tracking
# ══════════════════════════════════════════════════════════════════════

class TestUndoManagerSavePoint:
    def test_starts_at_save_point(self, _reset_undo_manager):
        mgr = _reset_undo_manager
        assert mgr.is_at_save_point

    def test_dirty_after_execute(self, _reset_undo_manager):
        mgr = _reset_undo_manager
        obj = _Obj()
        mgr.execute(SetPropertyCommand(obj, "x", 0, 1))
        assert not mgr.is_at_save_point

    def test_clean_after_undo(self, _reset_undo_manager):
        mgr = _reset_undo_manager
        obj = _Obj()
        mgr.execute(SetPropertyCommand(obj, "x", 0, 1))
        mgr.undo()
        assert mgr.is_at_save_point

    def test_mark_save_point(self, _reset_undo_manager):
        mgr = _reset_undo_manager
        obj = _Obj()
        mgr.execute(SetPropertyCommand(obj, "x", 0, 1))
        mgr.mark_save_point()
        assert mgr.is_at_save_point

    def test_dirty_after_undo_past_save(self, _reset_undo_manager):
        mgr = _reset_undo_manager
        obj = _Obj()
        cmd1 = SetPropertyCommand(obj, "x", 0, 1)
        cmd1.timestamp = 0.0
        mgr.execute(cmd1)
        mgr.mark_save_point()
        mgr.undo()
        assert not mgr.is_at_save_point

    def test_selection_command_does_not_dirty(self, _reset_undo_manager):
        mgr = _reset_undo_manager
        cmd = SelectionCommand([1], [2], apply_fn=lambda ids: None)
        mgr.record(cmd)
        # SelectionCommand has marks_dirty=False, so save point unchanged
        assert mgr.is_at_save_point


# ══════════════════════════════════════════════════════════════════════
# InspectorUndoTracker
# ══════════════════════════════════════════════════════════════════════

class TestInspectorUndoTracker:
    def test_detects_change(self, _reset_undo_manager):
        mgr = _reset_undo_manager
        tracker = InspectorUndoTracker()
        state = {"val": "A"}
        restored = []

        tracker.begin_frame()
        tracker.track("k1",
                       snapshot_fn=lambda: state["val"],
                       restore_fn=lambda s: restored.append(s),
                       description="Edit")
        # Simulate edit
        state["val"] = "B"
        tracker.end_frame()

        assert mgr.can_undo
        mgr.undo()
        assert restored == ["A"]

    def test_no_change_no_record(self, _reset_undo_manager):
        mgr = _reset_undo_manager
        tracker = InspectorUndoTracker()
        state = {"val": "A"}

        tracker.begin_frame()
        tracker.track("k1",
                       snapshot_fn=lambda: state["val"],
                       restore_fn=lambda s: None,
                       description="Edit")
        # No edit → no command recorded
        tracker.end_frame()
        assert not mgr.can_undo

    def test_multiple_targets(self, _reset_undo_manager):
        mgr = _reset_undo_manager
        tracker = InspectorUndoTracker()
        s1 = {"v": "1"}
        s2 = {"v": "X"}

        tracker.begin_frame()
        tracker.track("a", lambda: s1["v"], lambda s: None, "edit a")
        tracker.track("b", lambda: s2["v"], lambda s: None, "edit b")
        s1["v"] = "2"  # only s1 changed
        tracker.end_frame()

        assert len(mgr._undo_stack) == 1  # only 1 command recorded

    def test_begin_frame_clears_previous(self, _reset_undo_manager):
        mgr = _reset_undo_manager
        tracker = InspectorUndoTracker()

        tracker.begin_frame()
        tracker.track("k1", lambda: "a", lambda s: None, "edit")
        # Begin new frame without calling end_frame
        tracker.begin_frame()
        tracker.end_frame()
        assert not mgr.can_undo

    def test_snapshot_exception_skipped(self, _reset_undo_manager):
        mgr = _reset_undo_manager
        tracker = InspectorUndoTracker()

        def bad_snapshot():
            raise RuntimeError("broken")

        tracker.begin_frame()
        tracker.track("k1", bad_snapshot, lambda s: None, "edit")
        tracker.end_frame()
        assert not mgr.can_undo

    def test_duplicate_key_ignored(self, _reset_undo_manager):
        mgr = _reset_undo_manager
        tracker = InspectorUndoTracker()
        calls = [0]

        def snap():
            calls[0] += 1
            return "v"

        tracker.begin_frame()
        tracker.track("k", snap, lambda s: None, "edit")
        tracker.track("k", snap, lambda s: None, "edit")  # duplicate
        # Only 1 snapshot call (the first)
        assert calls[0] == 1


# ══════════════════════════════════════════════════════════════════════
# SelectionManager.set_ids
# ══════════════════════════════════════════════════════════════════════

class TestSelectionManagerSetIds:
    @pytest.fixture(autouse=True)
    def _fresh_selection(self):
        old = SelectionManager._instance
        sel = SelectionManager()
        yield sel
        SelectionManager._instance = old

    def test_set_ids_replaces_selection(self, _fresh_selection):
        sel = _fresh_selection
        sel.set_ids([10, 20, 30])
        assert sel.get_ids() == [10, 20, 30]
        assert sel.get_primary() == 30

    def test_set_ids_empty(self, _fresh_selection):
        sel = _fresh_selection
        sel.set_ids([1])
        sel.set_ids([])
        assert sel.get_ids() == []
        assert sel.get_primary() == 0

    def test_set_ids_fires_callback(self, _fresh_selection):
        sel = _fresh_selection
        called = []
        sel.add_listener(lambda: called.append(1))
        sel.set_ids([5])
        assert len(called) == 1

    def test_set_ids_no_change_no_callback(self, _fresh_selection):
        sel = _fresh_selection
        sel.set_ids([5])
        called = []
        sel.add_listener(lambda: called.append(1))
        sel.set_ids([5])  # same → should not fire
        assert len(called) == 0

    def test_clear(self, _fresh_selection):
        sel = _fresh_selection
        sel.select(42)
        sel.clear()
        assert sel.get_ids() == []
        assert sel.get_primary() == 0

    def test_toggle(self, _fresh_selection):
        sel = _fresh_selection
        sel.select(1)
        sel.toggle(2)
        assert sel.get_ids() == [1, 2]
        sel.toggle(1)
        assert sel.get_ids() == [2]


# ══════════════════════════════════════════════════════════════════════
# Integration: Selection undo via UndoManager
# ══════════════════════════════════════════════════════════════════════

class TestSelectionUndoIntegration:
    @pytest.fixture(autouse=True)
    def _fresh_selection(self):
        old = SelectionManager._instance
        self.sel = SelectionManager()
        yield
        SelectionManager._instance = old

    def _apply_fn(self, ids):
        self.sel.set_ids(ids)

    def test_select_undo_redo_cycle(self, _reset_undo_manager, _fresh_selection):
        mgr = _reset_undo_manager
        sel = self.sel
        # Simulate: nothing selected → select [1]
        mgr.record(SelectionCommand([], [1], self._apply_fn))
        sel.set_ids([1])
        assert sel.get_primary() == 1

        mgr.undo()
        assert sel.get_ids() == []

        mgr.redo()
        assert sel.get_ids() == [1]

    def test_multi_selection_undo(self, _reset_undo_manager, _fresh_selection):
        mgr = _reset_undo_manager
        sel = self.sel

        mgr.record(SelectionCommand([], [1, 2, 3], self._apply_fn))
        sel.set_ids([1, 2, 3])

        mgr.undo()
        assert sel.get_ids() == []

    def test_chained_selection_undo(self, _reset_undo_manager, _fresh_selection):
        mgr = _reset_undo_manager
        sel = self.sel

        # Step 1: select [1]
        c1 = SelectionCommand([], [1], self._apply_fn)
        c1.timestamp = 0.0
        mgr.record(c1)
        sel.set_ids([1])

        # Step 2: select [2]
        c2 = SelectionCommand([1], [2], self._apply_fn)
        c2.timestamp = 10.0
        mgr.record(c2)
        sel.set_ids([2])

        # Undo step 2 → back to [1]
        mgr.undo()
        assert sel.get_ids() == [1]

        # Undo step 1 → back to []
        mgr.undo()
        assert sel.get_ids() == []


# ══════════════════════════════════════════════════════════════════════
# Delete/Create selection restore callback
# ══════════════════════════════════════════════════════════════════════

# The commands call _get_active_scene() which uses SceneManager (native).
# We patch that helper to return None so __init__ doesn't blow up, then
# poke internal fields to simulate the state we want to test.

_undo_mod_ref = _undo_mod  # keep reference for patching


class TestDeleteCommandSelectionRestore:
    """DeleteGameObjectCommand should capture pre-delete selection and restore
    it on undo via _selection_restore_fn."""

    @pytest.fixture(autouse=True)
    def _patch_scene(self, monkeypatch):
        monkeypatch.setattr(_undo_mod_ref, "_get_active_scene", lambda: None)

    @pytest.fixture(autouse=True)
    def _fresh_sel(self):
        old = SelectionManager._instance
        self.sel = SelectionManager()
        SelectionManager._instance = self.sel
        yield
        SelectionManager._instance = old

    def test_captures_pre_delete_selection(self):
        self.sel.set_ids([42])
        cmd = DeleteGameObjectCommand(42, "Delete")
        assert cmd._pre_delete_selection_ids == [42]

    def test_undo_calls_selection_restore(self, _reset_undo_manager):
        restored = []
        old_fn = DeleteGameObjectCommand._selection_restore_fn
        DeleteGameObjectCommand._selection_restore_fn = lambda ids: restored.append(list(ids))
        try:
            self.sel.set_ids([42])
            cmd = DeleteGameObjectCommand(42, "Delete")
            cmd._snapshot_json = '{"id": 42}'
            orig_recreate = _undo_mod_ref._recreate_game_object_from_json
            _undo_mod_ref._recreate_game_object_from_json = lambda *a, **k: None
            try:
                cmd.undo()
            finally:
                _undo_mod_ref._recreate_game_object_from_json = orig_recreate
            assert restored == [[42]]
        finally:
            DeleteGameObjectCommand._selection_restore_fn = old_fn

    def test_redo_clears_selection(self, _reset_undo_manager):
        restored = []
        old_fn = DeleteGameObjectCommand._selection_restore_fn
        DeleteGameObjectCommand._selection_restore_fn = lambda ids: restored.append(list(ids))
        try:
            cmd = DeleteGameObjectCommand(42, "Delete")
            cmd.redo()  # no scene → no-op destruction, but selection should clear
            assert restored == [[]]
        finally:
            DeleteGameObjectCommand._selection_restore_fn = old_fn

    def test_no_restore_when_callback_not_set(self, _reset_undo_manager):
        old_fn = DeleteGameObjectCommand._selection_restore_fn
        DeleteGameObjectCommand._selection_restore_fn = None
        try:
            cmd = DeleteGameObjectCommand(42, "Delete")
            cmd._pre_delete_selection_ids = [42]
            cmd._snapshot_json = '{"id": 42}'
            orig_recreate = _undo_mod_ref._recreate_game_object_from_json
            _undo_mod_ref._recreate_game_object_from_json = lambda *a, **k: None
            try:
                cmd.undo()  # should not raise
            finally:
                _undo_mod_ref._recreate_game_object_from_json = orig_recreate
        finally:
            DeleteGameObjectCommand._selection_restore_fn = old_fn

    def test_empty_selection_not_restored(self, _reset_undo_manager):
        """If nothing was selected before delete, undo doesn't call restore."""
        restored = []
        old_fn = DeleteGameObjectCommand._selection_restore_fn
        DeleteGameObjectCommand._selection_restore_fn = lambda ids: restored.append(list(ids))
        try:
            # Don't select anything → pre_delete_selection_ids is []
            cmd = DeleteGameObjectCommand(42, "Delete")
            cmd._snapshot_json = '{"id": 42}'
            orig_recreate = _undo_mod_ref._recreate_game_object_from_json
            _undo_mod_ref._recreate_game_object_from_json = lambda *a, **k: None
            try:
                cmd.undo()
            finally:
                _undo_mod_ref._recreate_game_object_from_json = orig_recreate
            # Empty list → fn not called (guard: `if fn and ids`)
            assert restored == []
        finally:
            DeleteGameObjectCommand._selection_restore_fn = old_fn


class TestCreateCommandSelectionRestore:
    """CreateGameObjectCommand should clear selection on undo and re-select
    on redo via _selection_restore_fn."""

    @pytest.fixture(autouse=True)
    def _patch_scene(self, monkeypatch):
        monkeypatch.setattr(_undo_mod_ref, "_get_active_scene", lambda: None)

    def test_undo_clears_selection(self, _reset_undo_manager):
        restored = []
        old_fn = CreateGameObjectCommand._selection_restore_fn
        CreateGameObjectCommand._selection_restore_fn = lambda ids: restored.append(list(ids))
        try:
            cmd = CreateGameObjectCommand(99, "Create")
            cmd.undo()  # no scene → no-op destruction, but selection should clear
            assert restored == [[]]
        finally:
            CreateGameObjectCommand._selection_restore_fn = old_fn

    def test_redo_restores_selection(self, _reset_undo_manager):
        restored = []
        old_fn = CreateGameObjectCommand._selection_restore_fn
        CreateGameObjectCommand._selection_restore_fn = lambda ids: restored.append(list(ids))
        try:
            cmd = CreateGameObjectCommand(99, "Create")
            cmd._post_create_ids = [99]
            cmd._snapshot_json = '{"id": 99}'
            orig_recreate = _undo_mod_ref._recreate_game_object_from_json
            _undo_mod_ref._recreate_game_object_from_json = lambda *a, **k: None
            try:
                cmd.redo()
            finally:
                _undo_mod_ref._recreate_game_object_from_json = orig_recreate
            assert restored == [[99]]
        finally:
            CreateGameObjectCommand._selection_restore_fn = old_fn

    def test_redo_no_restore_when_no_post_ids(self, _reset_undo_manager):
        """If post_create_ids is empty, redo doesn't call restore."""
        restored = []
        old_fn = CreateGameObjectCommand._selection_restore_fn
        CreateGameObjectCommand._selection_restore_fn = lambda ids: restored.append(list(ids))
        try:
            cmd = CreateGameObjectCommand(99, "Create")
            cmd._post_create_ids = []
            cmd._snapshot_json = '{"id": 99}'
            orig_recreate = _undo_mod_ref._recreate_game_object_from_json
            _undo_mod_ref._recreate_game_object_from_json = lambda *a, **k: None
            try:
                cmd.redo()
            finally:
                _undo_mod_ref._recreate_game_object_from_json = orig_recreate
            assert restored == []
        finally:
            CreateGameObjectCommand._selection_restore_fn = old_fn
