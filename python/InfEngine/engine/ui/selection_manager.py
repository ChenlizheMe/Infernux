"""
Centralised selection state for the editor.

Hierarchy, Scene View, and other panels all read/write through this
single authority so multi-selection stays consistent everywhere.

Usage
-----
    from InfEngine.engine.ui.selection_manager import SelectionManager

    sel = SelectionManager.instance()
    sel.select(obj_id)           # single select
    sel.toggle(obj_id)           # ctrl+click
    sel.range_select(obj_id)     # shift+click (needs ordered list)
    sel.box_select([id1, id2])   # replace with box-select result
    sel.clear()
"""
from __future__ import annotations

from typing import Callable, List, Optional, Sequence


class SelectionManager:
    """Singleton that owns the set of selected GameObject IDs."""

    _instance: Optional["SelectionManager"] = None

    @classmethod
    def instance(cls) -> "SelectionManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        self._ids: list[int] = []          # ordered; last = primary
        self._primary: int = 0             # shortcut for the "main" selection
        self._callbacks: list[Callable] = []
        # Ordered ID list used for shift-range selection.
        # Set by the panel that owns display order (hierarchy / project).
        self._ordered_ids: list[int] = []

    # ── Registration ──────────────────────────────────────────────────

    def add_listener(self, cb: Callable[[], None]) -> None:
        if cb not in self._callbacks:
            self._callbacks.append(cb)

    def remove_listener(self, cb: Callable[[], None]) -> None:
        try:
            self._callbacks.remove(cb)
        except ValueError:
            pass

    def _notify(self) -> None:
        for cb in self._callbacks:
            try:
                cb()
            except Exception:
                pass

    # ── Ordered-ID hint (for shift-range) ─────────────────────────────

    def set_ordered_ids(self, ids: Sequence[int]) -> None:
        """Provide the visible ordering so shift-select can compute ranges."""
        self._ordered_ids = list(ids)

    # ── Mutation ──────────────────────────────────────────────────────

    def select(self, obj_id: int) -> None:
        """Replace selection with a single object."""
        if obj_id == 0:
            self.clear()
            return
        if self._ids == [obj_id]:
            return
        self._ids = [obj_id]
        self._primary = obj_id
        self._notify()

    def toggle(self, obj_id: int) -> None:
        """Ctrl+click: add or remove *obj_id* from the selection."""
        if obj_id == 0:
            return
        if obj_id in self._ids:
            self._ids.remove(obj_id)
            self._primary = self._ids[-1] if self._ids else 0
        else:
            self._ids.append(obj_id)
            self._primary = obj_id
        self._notify()

    def range_select(self, obj_id: int) -> None:
        """Shift+click: select contiguous range from primary → *obj_id*
        using the ordered-ID list provided by the panel."""
        if obj_id == 0 or not self._ordered_ids:
            self.select(obj_id)
            return
        anchor = self._primary or obj_id
        try:
            idx_a = self._ordered_ids.index(anchor)
        except ValueError:
            self.select(obj_id)
            return
        try:
            idx_b = self._ordered_ids.index(obj_id)
        except ValueError:
            self.select(obj_id)
            return
        lo, hi = min(idx_a, idx_b), max(idx_a, idx_b)
        new_ids = self._ordered_ids[lo : hi + 1]
        if self._ids == new_ids and self._primary == anchor:
            return
        self._ids = new_ids
        self._primary = anchor
        self._notify()

    def box_select(self, ids: Sequence[int], *, additive: bool = False) -> None:
        """Replace (or union) selection with the result of a box/lasso drag."""
        new = list(dict.fromkeys(ids))  # dedupe, preserve order
        if additive:
            combined = list(dict.fromkeys(self._ids + new))
            if combined == self._ids:
                return
            self._ids = combined
        else:
            if self._ids == new:
                return
            self._ids = new
        self._primary = self._ids[-1] if self._ids else 0
        self._notify()

    def clear(self) -> None:
        if not self._ids:
            return
        self._ids.clear()
        self._primary = 0
        self._notify()

    def set_ids(self, ids: Sequence[int]) -> None:
        """Replace the entire selection with *ids* (last element = primary).

        Used by undo/redo to restore a previous selection state.
        """
        new = list(ids)
        if new == self._ids:
            return
        self._ids = new
        self._primary = new[-1] if new else 0
        self._notify()

    # ── Queries ───────────────────────────────────────────────────────

    def get_ids(self) -> list[int]:
        """Ordered list of selected IDs (last = most recently added)."""
        return list(self._ids)

    def get_primary(self) -> int:
        return self._primary

    def is_selected(self, obj_id: int) -> bool:
        return obj_id in self._ids

    def count(self) -> int:
        return len(self._ids)

    def is_empty(self) -> bool:
        return len(self._ids) == 0

    def is_single(self) -> bool:
        return len(self._ids) == 1

    def is_multi(self) -> bool:
        return len(self._ids) > 1
