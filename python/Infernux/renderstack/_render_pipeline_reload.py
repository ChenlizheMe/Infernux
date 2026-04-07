"""PipelineReloadMixin — extracted from RenderStack."""
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


class PipelineReloadMixin:
    """PipelineReloadMixin method group for RenderStack."""

    def _register_pipeline_reload(self, pipeline_cls) -> None:
        """Subscribe to watchdog file-change events for the pipeline's source file."""
        import sys as _sys
        mod = _sys.modules.get(pipeline_cls.__module__)
        if mod is None:
            return
        src = getattr(mod, '__file__', None)
        if not src:
            return
        self._pipeline_module = mod
        from Infernux.engine.resources_manager import ResourcesManager
        rm = ResourcesManager.instance()
        if rm is not None:
            rm.register_script_reload_callback(src, self._on_pipeline_file_changed)

    def _unregister_pipeline_reload(self) -> None:
        """Unsubscribe from watchdog callbacks."""
        from Infernux.engine.resources_manager import ResourcesManager
        rm = ResourcesManager.instance()
        if rm is not None:
            rm.unregister_script_reload_callback(self._on_pipeline_file_changed)
        self._pipeline_module = None

    def _on_pipeline_file_changed(self, file_path: str) -> None:
        """Watchdog callback — called on main thread when pipeline source is saved."""
        import importlib
        from Infernux.renderstack.discovery import invalidate_discovery_cache
        mod = self._pipeline_module
        if mod is None:
            return
        print(f"[RenderStack] Pipeline file changed, reloading...", file=sys.stderr)
        self._save_current_pipeline_params()
        invalidate_discovery_cache()
        importlib.reload(mod)
        self._pipeline = None   # re-instantiate on next .pipeline access
        self.invalidate_graph() # clears _build_failed + _graph_desc
        print(f"[RenderStack] Pipeline reloaded.", file=sys.stderr)

    def _sync_pipeline_catalog(self) -> None:
        """Refresh available pipeline catalog and enforce fallback policy."""
        names = set(self.discover_pipelines().keys())
        signature = tuple(sorted(names))
        if signature == self._pipeline_catalog_signature:
            return

        self._pipeline_catalog_signature = signature

        current = self.pipeline_class_name
        if current and current not in names:
            warnings.warn(
                f"[RenderStack] Pipeline '{current}' was removed. Falling back to DefaultForwardPipeline.",
                stacklevel=2,
            )
            self.set_pipeline("")
            return

        # Refresh pipeline type on catalog changes so newly edited classes can be re-instantiated.
        if self._pipeline is not None:
            self._save_current_pipeline_params()
            self._pipeline = None
            self._cached_ips = None
            self.invalidate_graph()

    def _register_pipeline_catalog_reload(self) -> None:
        """Subscribe to watchdog-driven script catalog changes."""
        from Infernux.engine.resources_manager import ResourcesManager
        rm = ResourcesManager.instance()
        if rm is not None:
            rm.register_script_catalog_callback(self._on_script_catalog_changed)

    def _unregister_pipeline_catalog_reload(self) -> None:
        """Unsubscribe from watchdog-driven script catalog changes."""
        from Infernux.engine.resources_manager import ResourcesManager
        rm = ResourcesManager.instance()
        if rm is not None:
            rm.unregister_script_catalog_callback(self._on_script_catalog_changed)

    def _on_script_catalog_changed(self, file_path: str, event_type: str) -> None:
        """ResourcesManager callback for create/delete/move/modify of python scripts."""
        from Infernux.renderstack.discovery import invalidate_discovery_cache
        invalidate_discovery_cache()
        self._sync_pipeline_catalog()

