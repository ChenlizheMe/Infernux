"""RenderPassManagementMixin — extracted from RenderStack."""
from __future__ import annotations

"""
RenderStack — Scene-level rendering configuration component.

RenderStack is a scene-singleton InxComponent that manages:
- The active RenderPipeline (topology skeleton + injection points)
- All mounted RenderPass instances (user effects + built-in passes)
- Graph construction: combines pipeline topology with injected passes

Architecture::

    RenderStack (InxComponent, scene singleton)
      ├── selected_pipeline: RenderPipeline  (defines topology skeleton)
      └── pass_entries: List[PassEntry]      (user-mounted passes)

    Each frame:
      1. RenderStack.render(context, camera)
      2. Lazy-build graph if invalidated
      3. context.apply_graph(desc) + context.submit_culling(culling)

Build flow (Section 7.1)::

    graph = RenderGraph("Pipeline+Stack")
    bus = ResourceBus()
    pipeline.define_topology(graph, bus, callback)
      └── callback triggers _inject_passes_at for each injection point
    graph.set_output(bus.get("color"))
    graph.build() → RenderGraphDescription

Usage::

    # In a scene setup script
    stack = game_object.add_component(RenderStack)
    stack.set_pipeline("Default Forward")
    stack.add_pass(BloomPass())
"""


import json as _json
import sys
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, TYPE_CHECKING

from Infernux.components.component import InxComponent
from Infernux.components.decorators import disallow_multiple, add_component_menu
from Infernux.renderstack._pipeline_common import (
    COLOR_TEXTURE,
    ensure_standard_post_process_points,
)
from Infernux.renderstack.injection_point import InjectionPoint
from Infernux.renderstack.resource_bus import ResourceBus
from Infernux.debug import Debug


if TYPE_CHECKING:
    from Infernux.renderstack.render_stack import PassEntry


class RenderPassManagementMixin:
    """RenderPassManagementMixin method group for RenderStack."""

    @property
    def pass_entries(self) -> List[PassEntry]:
        """Mounted pass entries (read-only view for UI/integration code)."""
        if self._pass_entries is None:
            self._pass_entries = []
        return self._pass_entries

    def add_pass(self, render_pass: "RenderPass") -> bool:
        """Mount a RenderPass onto the RenderStack.

        The pass is assigned automatically based on
        ``render_pass.injection_point``.

        Returns:
            ``False`` if the injection point does not exist.
        """
        valid_points = {p.name for p in self.injection_points}
        if render_pass.injection_point not in valid_points:
            import logging
            logging.getLogger("Infernux.RenderStack").warning(
                "RenderPass '%s' has unknown injection_point '%s'. "
                "Valid points: %s",
                render_pass.name,
                render_pass.injection_point,
                ", ".join(sorted(valid_points)),
            )
            return False
        for entry in self._pass_entries:
            if (entry.render_pass.injection_point == render_pass.injection_point and
                    entry.render_pass.name == render_pass.name):
                return False
        from Infernux.renderstack.render_stack import PassEntry as _PassEntry
        entry = _PassEntry(
            render_pass=render_pass,
            enabled=render_pass.enabled,
            order=render_pass.default_order,
        )
        self._pass_entries.append(entry)
        self.invalidate_graph()
        return True

    def remove_pass(self, pass_name: str) -> bool:
        """Remove a mounted RenderPass by name."""
        for i, entry in enumerate(self._pass_entries):
            if entry.render_pass.name == pass_name:
                self._pass_entries.pop(i)
                self.invalidate_graph()
                return True
        return False

    def set_pass_enabled(self, pass_name: str, enabled: bool) -> None:
        """Enable or disable a mounted pass."""
        for entry in self._pass_entries:
            if entry.render_pass.name == pass_name:
                entry.enabled = enabled
                entry.render_pass.enabled = enabled
                self.invalidate_graph()
                return

    def reorder_pass(self, pass_name: str, new_order: int) -> None:
        """Change a pass order within its injection point."""
        for entry in self._pass_entries:
            if entry.render_pass.name == pass_name:
                entry.order = new_order
                self.invalidate_graph()
                return

    def move_pass_before(self, dragged_name: str, target_name: str) -> None:
        """Move ``dragged_name`` before ``target_name`` within one injection point.

        This reassigns order values while preserving the relative order of the
        other passes.
        """
        dragged_entry = None
        target_entry = None
        for e in self._pass_entries:
            if e.render_pass.name == dragged_name:
                dragged_entry = e
            if e.render_pass.name == target_name:
                target_entry = e
        if dragged_entry is None or target_entry is None:
            return
        if dragged_entry.render_pass.injection_point != target_entry.render_pass.injection_point:
            return

        ip = dragged_entry.render_pass.injection_point
        entries = self.get_passes_at(ip)

        # Remove dragged, insert before target, reassign orders
        ordered_names = [e.render_pass.name for e in entries if e.render_pass.name != dragged_name]
        try:
            idx = ordered_names.index(target_name)
        except ValueError as _exc:
            Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
            return
        ordered_names.insert(idx, dragged_name)

        # Reassign orders with stable spacing
        name_to_entry = {e.render_pass.name: e for e in self._pass_entries}
        for i, name in enumerate(ordered_names):
            entry = name_to_entry.get(name)
            if entry is not None:
                entry.order = (i + 1) * 10

        self.invalidate_graph()

    def get_passes_at(self, injection_point: str) -> List[PassEntry]:
        """Return all passes at an injection point, sorted by order."""
        match_names = {injection_point}

        entries = [
            e
            for e in self._pass_entries
            if e.render_pass.injection_point in match_names
        ]
        entries.sort(key=lambda e: e.order)
        return entries

    @property
    def injection_points(self) -> List[InjectionPoint]:
        """Read-only injection points defined by the current pipeline.

        Injection points come from ``define_topology()``. The first call runs a
        dry-build probe to discover them.
        """
        if not hasattr(self, "_cached_ips") or self._cached_ips is None:
            g = self._build_full_topology_probe()
            self._cached_ips = g.injection_points
        return self._cached_ips

