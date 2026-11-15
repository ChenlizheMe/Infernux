"""Hot-reload support for RenderStack pipeline definitions."""
from __future__ import annotations

import sys
import warnings


class PipelineReloadMixin:
    """PipelineReloadMixin method group for RenderStack."""

    def _register_pipeline_reload(self, pipeline_cls) -> None:
        """Subscribe to watchdog file-change events for the pipeline's source file."""
        from infernux.application import Application

        mod = sys.modules.get(pipeline_cls.__module__)
        if mod is None:
            return
        src = getattr(mod, '__file__', None)
        if not src:
            return
        self._pipeline_module = mod
        from infernux.engine.reload.owners import owner_for
        owner = owner_for(pipeline_cls)
        self._pipeline_owner = owner
        if owner is not None:
            import weakref
            reference = weakref.ref(self)
            def retire_source(expected=mod, source=src):
                target = reference()
                if target is None or target._pipeline_module is not expected:
                    return
                target._save_current_pipeline_params()
                target._pipeline_module = None
                target._pipeline = None
                target._cached_ips = None
                from infernux.renderstack.discovery import invalidate_discovery_cache
                invalidate_discovery_cache(source)
                target.invalidate_graph()
            owner.add_cleanup(retire_source)
        if not Application.is_editor():
            return
        from infernux.engine.resources_manager import ResourcesManager
        rm = ResourcesManager.instance()
        if rm is not None:
            rm.unregister_script_reload_callback(self._on_pipeline_file_changed)
            rm.register_script_reload_callback(src, self._on_pipeline_file_changed)

    def _unregister_pipeline_reload(self) -> None:
        """Unsubscribe from watchdog callbacks."""
        from infernux.application import Application

        if not Application.is_editor():
            self._pipeline_module = None
            self._pipeline_owner = None
            return
        from infernux.engine.resources_manager import ResourcesManager
        rm = ResourcesManager.instance()
        if rm is not None:
            rm.unregister_script_reload_callback(self._on_pipeline_file_changed)
        self._pipeline_module = None
        self._pipeline_owner = None

    def _on_pipeline_file_changed(self, file_path: str) -> None:
        """Watchdog callback — called on main thread when pipeline source is saved."""
        from infernux.renderstack.discovery import invalidate_discovery_cache
        previous_module = self._pipeline_module
        if previous_module is None:
            from infernux.engine.path_utils import path_key
            if path_key(file_path) == path_key(self._pipeline_source_file):
                invalidate_discovery_cache(file_path)
                self.invalidate_graph()
            return
        module_name = getattr(previous_module, "__name__", "")
        published_module = sys.modules.get(module_name)
        if published_module is None:
            # The transactional script loader owns module publication.  A
            # callback must never re-execute source behind that transaction;
            # the catalog callback will recover a module that was retired.
            self._save_current_pipeline_params()
            self._pipeline_module = None
            self._pipeline = None
            self.invalidate_graph()
            return
        self._save_current_pipeline_params()
        invalidate_discovery_cache()
        self._pipeline_module = published_module
        self._pipeline = None   # re-instantiate on next .pipeline access
        self.invalidate_graph() # clears _build_failed + _graph_desc

    def _sync_pipeline_catalog(self) -> None:
        """Refresh the available pipeline catalog without changing selection."""
        names = set(self.discover_pipelines().keys())
        signature = tuple(sorted(names))
        if signature == self._pipeline_catalog_signature:
            return

        previous_names = self._pipeline_catalog_signature
        self._pipeline_catalog_signature = signature

        current = self.pipeline_class_name
        if current != self.DEFAULT_PIPELINE_NAME and current not in names:
            self._save_current_pipeline_params()
            warnings.warn(
                f"[RenderStack] Selected pipeline '{current}' was removed. "
                "Rendering is disabled until the pipeline is restored "
                "or another pipeline is selected.",
                RuntimeWarning,
                stacklevel=2,
            )
            self._pipeline = None
            self._cached_ips = None
            self.invalidate_graph()
            return

        # Refresh pipeline type on catalog changes so newly edited classes can be re-instantiated.
        if self._pipeline is not None:
            self._save_current_pipeline_params()
            self._pipeline = None
            self._cached_ips = None
            self.invalidate_graph()
        elif current in names and current not in previous_names:
            # A missing/ambiguous selection has no instance to refresh, but
            # its failed graph must retry once when the name becomes usable.
            self._cached_ips = None
            self.invalidate_graph()

    def _register_pipeline_catalog_reload(self) -> None:
        """Subscribe to watchdog-driven script catalog changes."""
        from infernux.application import Application

        if not Application.is_editor():
            return
        from infernux.engine.resources_manager import ResourcesManager
        rm = ResourcesManager.instance()
        if rm is not None:
            rm.register_script_catalog_callback(self._on_script_catalog_changed)

    def _unregister_pipeline_catalog_reload(self) -> None:
        """Unsubscribe from watchdog-driven script catalog changes."""
        from infernux.application import Application

        if not Application.is_editor():
            return
        from infernux.engine.resources_manager import ResourcesManager
        rm = ResourcesManager.instance()
        if rm is not None:
            rm.unregister_script_catalog_callback(self._on_script_catalog_changed)

    def _on_script_catalog_changed(self, file_path: str, event_type: str) -> None:
        """ResourcesManager callback for create/delete/move/modify of python scripts."""
        from infernux.renderstack.discovery import (
            script_may_affect_pipeline_catalog,
        )
        if not script_may_affect_pipeline_catalog(file_path, event_type):
            return
        from infernux.renderstack.discovery import invalidate_discovery_cache
        invalidate_discovery_cache(file_path)
        self._sync_pipeline_catalog()
        if self._pipeline is None:
            self.invalidate_graph()

