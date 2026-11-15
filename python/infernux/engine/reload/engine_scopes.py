"""Explicit built-in source scopes, using the ordinary reload coordinator."""

import importlib
from pathlib import Path
from threading import Lock

from watchdog.events import FileSystemEventHandler

from infernux.engine.path_utils import path_key, resolved_path


# Only views whose data lives outside their presentation instances are rebuilt.
# The execution machinery and native extension require a real process restart.
PANEL_MODULES = (
    "infernux.engine.ui.catalog_browser",
    "infernux.engine.ui.plugin_panel",
    "infernux.engine.ui.theme_panel",
)
CORE_MODULES = (
    "infernux.engine.reload.coordinator",
    "infernux.engine.reload.modules",
    "infernux.engine.reload.owners",
    "infernux.engine.reload.engine_scopes",
    "infernux.engine.ui.editor_panel",
    "infernux.engine.ui.panel_registry",
    "infernux.lib._Infernux",
)


class EngineSourceScopes(FileSystemEventHandler):
    """Watch actual loaded sources; file events never import engine assets."""

    def __init__(self, resources):
        self.resources = resources
        self.coordinator = resources._ensure_event_handler().reload
        self.paths = {}
        self.restart_paths = set()
        self._pending = {}
        self._lock = Lock()
        self._views = {}
        for name in (*PANEL_MODULES, *CORE_MODULES):
            module = importlib.import_module(name)
            path = resolved_path(module.__file__)
            restart = name in CORE_MODULES
            self.coordinator.register_source_scope(path, name, restart_only=restart, loaded_module=module)
            self.paths[path_key(path)] = path
            if restart:
                self.restart_paths.add(path)
        self.coordinator.adapter.add(self)
        self._attribute_registrations(PANEL_MODULES)

    @property
    def watch_directories(self):
        return tuple(sorted({str(Path(path).parent) for path in self.paths.values()}))

    def on_any_event(self, event):
        if event.is_directory or event.event_type not in {"created", "modified", "deleted", "moved"}:
            return
        changed = []
        for path in (event.src_path, getattr(event, "dest_path", "")):
            source = self.paths.get(path_key(path)) if path else None
            if source is not None:
                changed.append(source)
        if changed:
            with self._lock:
                self._pending.update(dict.fromkeys(changed))
            self.resources._engine.request_editor_wake()

    def submit_pending(self):
        with self._lock:
            pending, self._pending = tuple(self._pending), {}
        if not pending:
            return 0
        removed = tuple(path for path in pending if not Path(path).is_file())
        self.coordinator.request(tuple(path for path in pending if path not in removed), removed=removed)
        self.resources._frontend_wake_event.set()
        return len(pending)

    @staticmethod
    def _attribute_registrations(names):
        from infernux.engine.ui.panel_registry import PanelRegistry

        for registration in PanelRegistry.get_registrations():
            name = registration.panel_class.__module__
            if name in names:
                module = importlib.import_module(name)
                registration.owner = "source:" + path_key(module.__file__)

    def guid_for_path(self, path):
        return ""

    def retire(self, paths):
        from infernux.engine.ui import panel_registry
        from infernux.engine.ui import _EXPORTS
        import infernux.engine.ui as ui

        registry = panel_registry.PanelRegistry
        for path in paths:
            if path_key(path) not in self.paths:
                continue
            owner = "source:" + path_key(path)
            views = registry.capture_open_views_for_owners((owner,))
            self._views[path] = tuple(dict.fromkeys((*self._views.get(path, ()), *views)))
            registry.remove_owner(owner)
            record = self.coordinator.graph.module_for_path(path)
            name = record.id.module_name if record is not None else ""
            for alias, (module_name, _attribute) in _EXPORTS.items():
                if module_name.startswith("."):
                    module_name = importlib.util.resolve_name(module_name, ui.__name__)
                if module_name == name:
                    vars(ui).pop(alias, None)
        restart = self.restart_paths.intersection(paths)
        if restart:
            from infernux.debug import Debug
            from infernux.engine.play_mode import PlayModeManager

            play = PlayModeManager.instance()
            if play is not None and play.is_playing:
                play.pause()
            Debug.log_error("Loaded engine core/native source changed; restart the editor: " + ", ".join(sorted(restart)))

    def prepare(self, modules, owner):
        return tuple(resolved_path(module.__file__) for name, module in modules.items() if name in PANEL_MODULES)

    def bind(self, prepared):
        pass

    def initialize(self, prepared, owner):
        pass

    def activate(self, prepared):
        from infernux.engine.ui.panel_registry import PanelRegistry

        for path in prepared:
            PanelRegistry.restore_reloaded_views(self._views.pop(path, ()))

    def fail(self, paths, error):
        pass
