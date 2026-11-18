"""Construct current modules with normal Python imports and no stale pyc reuse."""

import importlib
import importlib.abc
import importlib.util
import importlib.machinery
import os
import sys
import builtins
from contextlib import ExitStack, contextmanager, nullcontext
from contextvars import ContextVar
from infernux.engine.path_utils import resolved_path
from types import ModuleType


_constructing = ContextVar("reload_constructing_modules", default=None)
_importing = ContextVar("reload_importing_module", default=None)
_coordinators = set()
_import_module = importlib.import_module


def is_native_module(module):
    spec = getattr(module, "__spec__", None)
    origin = str(getattr(spec, "origin", "") or getattr(module, "__file__", ""))
    return isinstance(getattr(spec, "loader", None), importlib.machinery.ExtensionFileLoader) or any(
        origin.endswith(suffix) for suffix in importlib.machinery.EXTENSION_SUFFIXES)


def loaded_native_modules(paths):
    """Resolve actual extension modules at a file publication boundary."""
    from infernux.engine.path_utils import path_key

    targets = {path_key(path) for path in paths}
    return tuple(module for module in tuple(sys.modules.values())
                 if module is not None and is_native_module(module)
                 and path_key(getattr(module, "__file__", "")) in targets)


def _remember_import(module, importer):
    from .owners import current_owner

    owner = current_owner()
    if owner is None or owner.coordinator is None or module is None:
        return
    batch = _constructing.get()
    if batch is not None and batch.coordinator is owner.coordinator:
        batch.remember_import(importer, module)
    elif owner.active:
        owner.coordinator.record_import(importer, module, owner)


def _observed_import_module(name, package=None):
    importer = sys._getframe(1).f_globals.get("__name__", "")
    token = _importing.set(importer)
    try:
        module = _import_module(name, package)
    finally:
        _importing.reset(token)
    _remember_import(module, importer)
    return module


def observe_runtime_imports(coordinator):
    if not _coordinators:
        importlib.import_module = _observed_import_module
    _coordinators.add(coordinator)


def stop_observing_runtime_imports(coordinator):
    _coordinators.discard(coordinator)
    if not _coordinators and importlib.import_module is _observed_import_module:
        importlib.import_module = _import_module


def coordinator_for_path(path):
    """Find the current source authority even for an explicit scene harness."""
    matches = [item for item in _coordinators if not item._closed and item.graph.owns_path(path)]
    if len(matches) > 1:
        raise RuntimeError(f"Several reload coordinators own this source: {path}")
    return matches[0] if matches else None


def _source_builtins(namespace):
    def source_import(name, globals=None, locals=None, fromlist=(), level=0):
        importer = namespace["__name__"]
        token = _importing.set(importer)
        try:
            imported = builtins.__import__(name, globals, locals, fromlist, level)
        finally:
            _importing.reset(token)
        absolute = importlib.util.resolve_name("." * level + name, globals["__package__"]) if level else name
        _remember_import(imported, importer)
        _remember_import(sys.modules.get(absolute), importer)
        for child in fromlist or ():
            if child != "*":
                _remember_import(sys.modules.get(absolute + "." + child), importer)
        return imported

    return dict(vars(builtins), __import__=source_import)


def remove_module(name):
    module = sys.modules.pop(name, None)
    parent, _, child = name.rpartition(".")
    namespace = sys.modules.get(parent)
    if module is not None and namespace is not None and vars(namespace).get(child) is module:
        delattr(namespace, child)


class ModuleBatch(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def __init__(self, sources, *, unavailable=(), coordinator=None, owner=None):
        # name -> exact compiled source; the owner has already invalidated old modules.
        self.sources = sources
        self.unavailable = frozenset(unavailable)
        self.modules = {}
        self.serializable_types = {}
        self.effect_features = {}
        self.effect_sources = {}
        self.coordinator = coordinator
        self.runtime_imports = set()
        self.catalog_sources = set()
        self.owner = owner
        self.error = None

    @contextmanager
    def imports(self):
        sys.meta_path.insert(0, self)
        token = _constructing.set(self)
        try:
            yield
        finally:
            _constructing.reset(token)
            sys.meta_path.remove(self)

    def remember_import(self, importer, module):
        path = getattr(module, "__file__", None)
        if not path or self.coordinator is None or not self.coordinator.graph.owns_path(path):
            return
        if is_native_module(module):
            self.coordinator.record_import(importer, module, self.owner, verify_native=True)
            return
        if module.__name__ not in self.sources:
            state = self.coordinator.states.get(resolved_path(path))
            if ((state is None or state.status != "active")
                    and not self.coordinator.is_current_candidate(module.__name__)):
                raise ImportError(f"Project dependency was not prepared from current source: {path}")
        self.runtime_imports.add((importer, module.__name__))

    def find_spec(self, fullname, path=None, target=None):
        result = self.sources.get(fullname)
        if result is None and self.coordinator is not None:
            candidate = self.coordinator.candidate_batch(fullname, consumer=self)
            if candidate is not None and candidate is not self:
                self._remember_attempt(fullname)
                return candidate.find_spec(fullname, path, target)
        if result is None and fullname not in self.unavailable and self.coordinator is not None:
            spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
            if (spec is not None and spec.origin and spec.origin.endswith(".py")
                    and self.coordinator.graph.owns_path(spec.origin)):
                from pathlib import Path
                from .source import SourceResult
                from infernux.engine.path_utils import resolved_path

                source_path = resolved_path(spec.origin)
                source = None
                code = error = None
                try:
                    source = Path(source_path).read_bytes()
                    code = compile(source, source_path, "exec", dont_inherit=True)
                except Exception as exc:
                    error = exc
                result = SourceResult(source_path, 0, source, code, error)
                self.coordinator.graph.set_module_name(source_path, fullname)
                self.sources[fullname] = result
        if result is not None:
            self._remember_attempt(fullname)
            if result.error is not None:
                raise result.error
            if self.error is not None:
                raise ImportError(f"Current module construction failed: {fullname}") from self.error
            package = os.path.basename(result.path) in {"__init__.py", "__init__.pyc"}
            return importlib.util.spec_from_file_location(
                fullname,
                result.path,
                loader=self,
                submodule_search_locations=[os.path.dirname(result.path)] if package else None,
            )
        if fullname in self.unavailable:
            raise ImportError(f"Current script is unavailable: {fullname}")
        return None

    def _remember_attempt(self, fullname):
        importer = _importing.get()
        if not importer or self.coordinator is None:
            return
        if importer in self.sources or self.coordinator.graph.module_for_name(importer) is not None:
            self.runtime_imports.add((importer, fullname))
        elif self.owner is not None:
            self.runtime_imports.update((record.id.module_name, fullname)
                for path in self.owner.paths
                if (record := self.coordinator.graph.module_for_path(path)) is not None)

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        # A normal import may enter a different SCC in the current publication.
        # Its definitions and derived registrations belong to that SCC's owner.
        try:
            from infernux.components._component_registration import candidate_component_registration_scope
            from infernux.engine.project_context import temporary_script_import_paths
            with (self.owner.enter() if self.owner is not None else nullcontext()), self.imports(), \
                    candidate_component_registration_scope(), temporary_script_import_paths(self.sources[module.__name__].path):
                self._exec_module(module)
        except Exception as exc:
            self.error = exc
            raise

    def _exec_module(self, module):
        from infernux.components.serializable_object import _candidate_serializable_scope
        from infernux.renderstack.render_effect_compiler import _capture_source_effect_features
        from .contributions import contribution_scope
        from infernux.engine.path_utils import path_key

        result = self.sources[module.__name__]
        self.modules[module.__name__] = module
        from .owners import current_owner
        module.__infernux_reload_owner__ = current_owner()
        module.__builtins__ = _source_builtins(vars(module))
        with (
            _candidate_serializable_scope(self.serializable_types, module.__name__),
            _capture_source_effect_features(result.path, self.effect_features, self.effect_sources),
            contribution_scope("source:" + path_key(result.path)),
        ):
            exec(result.code, vars(module))
            for value in vars(module).values():
                if isinstance(value, type) and value.__module__ == module.__name__:
                    value.__infernux_reload_owner__ = module.__infernux_reload_owner__
                    for module_name, base_name in (
                        ("infernux.renderstack.render_pipeline", "RenderPipeline"),
                        ("infernux.renderstack.render_pass", "RenderPass"),
                    ):
                        namespace = sys.modules.get(module_name)
                        base = getattr(namespace, base_name, None)
                        if base is not None and issubclass(value, base):
                            self.catalog_sources.add(result.path)
            if result.path in self.catalog_sources and module.__infernux_reload_owner__ is not None:
                def retire_catalog(source=result.path):
                    from infernux.renderstack.discovery import invalidate_discovery_cache
                    invalidate_discovery_cache(source)
                module.__infernux_reload_owner__.add_cleanup(retire_catalog)

    def _parents(self, name, path):
        parent, _, _ = name.rpartition(".")
        folder = os.path.dirname(path)
        if os.path.basename(path) in {"__init__.py", "__init__.pyc"}:
            folder = os.path.dirname(folder)
        missing = []
        while parent:
            if parent not in sys.modules and parent not in self.sources:
                missing.append((parent, folder))
            folder = os.path.dirname(folder)
            parent, _, _ = parent.rpartition(".")
        for parent, folder in reversed(missing):
            if any(os.path.isfile(os.path.join(folder, name)) for name in ("__init__.py", "__init__.pyc")):
                _import_module(parent)
                continue
            package = ModuleType(parent)
            package.__path__ = [folder]
            package.__package__ = parent
            package.__spec__ = importlib.util.spec_from_loader(parent, loader=None, is_package=True)
            sys.modules[parent] = package
            above, _, child = parent.rpartition(".")
            if above in sys.modules:
                setattr(sys.modules[above], child, package)

    def load(self, names=None):
        from infernux.components._component_registration import candidate_component_registration_scope
        from infernux.engine.project_context import temporary_script_import_paths

        sys.meta_path.insert(0, self)
        try:
            with ExitStack() as stack:
                stack.enter_context(candidate_component_registration_scope())
                for result in self.sources.values():
                    stack.enter_context(temporary_script_import_paths(result.path))
                selected = tuple(self.sources) if names is None else tuple(names)
                for name in selected:
                    result = self.sources[name]
                    self._parents(name, result.path)
                for name in selected:
                    # Constructing a source is not a runtime dependency from
                    # this loader onto that source. User imports inside exec
                    # still pass through their normal observed entry points.
                    _import_module(name)
        except BaseException:
            self.discard()
            raise
        finally:
            sys.meta_path.remove(self)
        return self.modules

    def publish(self):
        from infernux.components.serializable_object import _publish_serializable_types
        from infernux.renderstack.render_effect_compiler import _publish_effect_features
        from infernux.engine.path_utils import path_key

        _publish_serializable_types(self.serializable_types, set(self.sources))
        _publish_effect_features(
            self.effect_features.values(),
            source_keys={path_key(item.path) for item in self.sources.values()},
        )
        if self.catalog_sources:
            from infernux.renderstack.discovery import invalidate_discovery_cache
            for source in self.catalog_sources:
                invalidate_discovery_cache(source)

    def publish_components(self, guid_for_path):
        """The common type/dispatch publication for source and Player code."""
        from infernux.components.component import InxComponent
        from infernux.components.component_identity import bind_asset_script_guid
        from infernux.components.registry import publish_component_script_types_batch
        from infernux.engine.runtime_dispatch import publish_runtime_dispatch_epoch

        declarations = []
        for name, module in self.modules.items():
            classes = tuple(value for value in vars(module).values()
                            if isinstance(value, type) and value is not InxComponent
                            and issubclass(value, InxComponent) and value.__module__ == name)
            path = self.sources[name].path
            guid = guid_for_path(path)
            if guid:
                for cls in classes:
                    bind_asset_script_guid(cls, guid, register=False)
            declarations.append((path, classes))
        publish_component_script_types_batch(declarations)
        publish_runtime_dispatch_epoch(tuple(cls for _path, classes in declarations for cls in classes)).commit()

    def discard(self):
        for name in self.modules:
            remove_module(name)


class RuntimeModuleBatch(ModuleBatch):
    """Fresh source imports admitted by an already active callback lifetime."""

    def __init__(self, sources, *, coordinator, owner):
        super().__init__(sources, coordinator=coordinator, unavailable=coordinator._unavailable)
        self.owner = owner.source_owner.child()
        self._depth = 0

    def exec_module(self, module):
        from infernux.components._component_registration import candidate_component_registration_scope

        self._depth += 1
        try:
            with self.owner.enter(), self.imports(), candidate_component_registration_scope():
                super().exec_module(module)
            if self._depth == 1:
                self.publish()
                self.coordinator.adopt_runtime_imports(self, self.owner)
        except BaseException:
            self.owner.close()
            self.discard()
            raise
        finally:
            self._depth -= 1
