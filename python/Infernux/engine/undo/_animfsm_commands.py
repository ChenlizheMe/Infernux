"""Snapshot undo commands for node-graph style editors.

Provides a generic reusable snapshot command for any panel that edits
node graphs, plus backwards-compatible AnimFSM aliases.
"""

from __future__ import annotations

import copy
from typing import Any, Callable

from Infernux.engine.undo._base import UndoCommand
from Infernux.engine.undo._manager import UndoManager


class NodeGraphSnapshotCommand(UndoCommand):
    """Undo/redo command for snapshot-based node-graph edits.

    The command stores two snapshots and replays them through an apply
    callback owned by the panel.
    """

    _is_property_edit = False
    marks_dirty = False

    def __init__(
        self,
        description: str,
        before_snapshot: Any,
        after_snapshot: Any,
        apply_snapshot: Callable[[Any], None],
    ) -> None:
        super().__init__(description)
        self._before_snapshot = copy.deepcopy(before_snapshot)
        self._after_snapshot = copy.deepcopy(after_snapshot)
        self._apply_snapshot = apply_snapshot

    def execute(self) -> None:
        self._apply_snapshot(copy.deepcopy(self._after_snapshot))

    def undo(self) -> None:
        self._apply_snapshot(copy.deepcopy(self._before_snapshot))

    def redo(self) -> None:
        self._apply_snapshot(copy.deepcopy(self._after_snapshot))


def record_node_graph_snapshot(
    *,
    description: str,
    before_snapshot: Any,
    after_snapshot: Any,
    apply_snapshot: Callable[[Any], None],
) -> bool:
    """Record a node-graph snapshot command through the shared UndoManager."""

    if before_snapshot == after_snapshot:
        return False

    mgr = UndoManager.instance()
    if not mgr or not mgr.enabled:
        return False

    from Infernux.engine.play_mode import PlayModeManager, PlayModeState

    pmm = PlayModeManager.instance()
    if pmm and pmm.state != PlayModeState.EDIT:
        return False

    mgr.record(
        NodeGraphSnapshotCommand(
            description=description,
            before_snapshot=before_snapshot,
            after_snapshot=after_snapshot,
            apply_snapshot=apply_snapshot,
        )
    )
    return True


# Backwards-compatible aliases for existing AnimFSM imports.
AnimFSMSnapshotCommand = NodeGraphSnapshotCommand


def record_animfsm_snapshot(
    *,
    description: str,
    before_snapshot: Any,
    after_snapshot: Any,
    apply_snapshot: Callable[[Any], None],
) -> bool:
    return record_node_graph_snapshot(
        description=description,
        before_snapshot=before_snapshot,
        after_snapshot=after_snapshot,
        apply_snapshot=apply_snapshot,
    )
