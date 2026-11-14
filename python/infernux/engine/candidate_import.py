"""Private import table for runtime script reload candidates.

Candidate scripts are ordinary Python modules, but they must not be executed
through the process-wide import table.  This module provides a small,
transaction-scoped importer instead.  Project modules are loaded into a
private table and may import one another, including cycles. Imports outside
that table use Python's ordinary engine and installed dependency modules.

This is an isolation boundary for reload correctness, not a security sandbox:
the static candidate policy diagnoses explicit import-time side effects.
"""

from __future__ import annotations

import builtins
import importlib
import importlib.util
import inspect
import os
import sys
import tokenize
import types
import weakref
from collections import ChainMap
from dataclasses import dataclass
from functools import lru_cache

from .path_utils import is_path_within, path_key, resolved_path
from .module_classification import is_stdlib_module
from .project_context import (
    get_project_script_roots,
    get_script_module_name,
    is_preload_python_library,
)


class CandidateImportError(ImportError):
    """Raised when a candidate import cannot be proven to be controlled."""


@dataclass(frozen=True, slots=True)
class CandidateModuleSpec:
    name: str
    file_path: str
    source: bytes | str | None = None
    code: types.CodeType | None = None
    namespace: bool = False


def _module_source(module: types.ModuleType) -> str | None:
    return types.ModuleType.__getattribute__(module, "__dict__").get("__file__")


def _published_module(module: types.ModuleType) -> types.ModuleType:
    name = types.ModuleType.__getattribute__(module, "__name__")
    current = sys.modules.get(name)
    if type(current) is types.ModuleType and _module_source(current) == _module_source(module):
        return current
    return module


class _PublishedModuleAlias(types.ModuleType):
    """An earlier module reference follows publication of the same script."""

    __slots__ = ()

    def __getattribute__(self, name):
        return types.ModuleType.__getattribute__(_published_module(self), name)

    def __setattr__(self, name, value):
        types.ModuleType.__setattr__(_published_module(self), name, value)

    def __delattr__(self, name):
        types.ModuleType.__delattr__(_published_module(self), name)

    def __dir__(self):
        return types.ModuleType.__dir__(_published_module(self))


@dataclass(frozen=True, slots=True)
class _PublishedFunction:
    module: str
    source: str
    name: str
    original: types.FunctionType

    @property
    def __wrapped__(self):
        module = sys.modules.get(self.module)
        if type(module) is not types.ModuleType or _module_source(module) != self.source:
            return self.original
        function = vars(module).get(self.name)
        if callable(function) and getattr(function, "__wrapped__", None) is not self:
            return function
        raise ReferenceError(f"Project function '{self.module}.{self.name}' is no longer published")


def _method_function(descriptor):
    return descriptor.__func__ if isinstance(descriptor, (staticmethod, classmethod)) else descriptor


@dataclass(frozen=True, slots=True)
class _PublishedMethod:
    owner: type
    name: str
    descriptor_type: type

    @property
    def __wrapped__(self):
        descriptor = vars(self.owner).get(self.name)
        if type(descriptor) is self.descriptor_type:
            function = _method_function(descriptor)
            if getattr(function, "__wrapped__", None) is not self:
                return function
        raise ReferenceError(f"Project method '{self.owner.__qualname__}.{self.name}' is no longer published")


__infernux_published_functions__: dict[str, _PublishedFunction | _PublishedMethod] = {}


def _forward_published_function(*args, **kwargs):
    return __infernux_published_functions__["<function>"].__wrapped__(*args, **kwargs)


@lru_cache(maxsize=32)
def _bridge_code(capture_count):
    if not capture_count:
        return _forward_published_function.__code__
    # CPython requires the same number of closure cells when replacing a body.
    # The bridge delegates to a new body that receives the original cells.
    names = [f"cell_{index}" for index in range(capture_count)]
    source = (
        "def build():\n"
        f"    {' = '.join(names)} = None\n"
        "    def forward(*args, **kwargs):\n"
        f"        if False: ({', '.join(names)},)\n"
        '        return __infernux_published_functions__["<function>"].__wrapped__(*args, **kwargs)\n'
        "    return forward.__code__\n"
    )
    scope = {}
    exec(source, scope)
    return scope["build"]()


_NON_SYNCHRONOUS = inspect.CO_GENERATOR | inspect.CO_COROUTINE | inspect.CO_ASYNC_GENERATOR


def _plain_project_function(function, namespace, name, *, method=False):
    free_variables = ((), ("__class__",)) if method else ((),)
    return (type(function) is types.FunctionType and function.__globals__ is namespace
            and function.__qualname__ == name and function.__code__.co_freevars in free_variables
            and not function.__code__.co_flags & _NON_SYNCHRONOUS
            and "__wrapped__" not in function.__dict__)


class CandidateImportTransaction:
    """Execute a candidate dependency closure without publishing modules.

    ``load`` inserts an empty module into ``modules`` before executing it, so
    normal circular imports observe the same object.  The module table is
    never exposed through ``sys.modules`` until ``commit`` is called. The owner
    calls ``finalize`` only after its outer publication becomes durable; import
    hooks then resolve current published modules and no longer retain history.
    """

    def __init__(self) -> None:
        self._specs: dict[str, CandidateModuleSpec] = {}
        self._modules: dict[str, types.ModuleType] = {}
        self._reused_lkg: dict[str, types.ModuleType] = {}
        self._roots = get_project_script_roots()
        from .runtime_worlds import current_world_runtime
        self._world_domain = current_world_runtime()
        self._external_roots: dict[str, bool] = {}
        self._trusted_proxies: dict[str, types.ModuleType] = {}
        self._overlay_names: set[str] = set()
        self._parent_before: list[tuple[types.ModuleType, str, bool, object]] = []
        self._parent_before_keys: set[tuple[int, str]] = set()
        self._before: dict[str, object] = {}
        self._published_aliases: list[types.ModuleType] = []
        self._forwarded_functions: list[tuple[types.FunctionType, types.CodeType]] = []
        self._forwarded_namespaces: dict[int, tuple[dict, object, dict]] = {}
        self._closure_bodies = []
        self._serializable_types: dict[str, type] = {}
        self._serializable_before: dict[str, type | None] = {}
        self._effect_features = {}
        self._effect_module_sources: dict[str, str] = {}
        self._effect_sources: set[str] = set()
        self._effect_before = {}
        self._committed = False
        self._rolled_back = False
        self._finalized = False
        self._published_names: frozenset[str] = frozenset()
        self._accepted_code_before = {}

    @property
    def finalized(self) -> bool:
        return self._finalized

    @property
    def modules(self) -> dict[str, types.ModuleType]:
        """Return the private candidate module table."""
        return dict(self._modules)

    @property
    def publishable_modules(self) -> tuple[types.ModuleType, ...]:
        """Return candidate modules, excluding shallow live-package overlays."""
        return tuple(
            module
            for name, module in self._modules.items()
            if name not in self._overlay_names
        )

    @property
    def loaded_module_names(self) -> tuple[str, ...]:
        return tuple(self._modules)

    def register(
        self,
        name: str,
        file_path: str,
        *,
        source: bytes | str | None = None,
        code: types.CodeType | None = None,
    ) -> None:
        if self._rolled_back or self._committed:
            raise CandidateImportError("cannot register modules after candidate publication or rollback")
        if not name or not isinstance(name, str):
            raise ValueError("candidate module name must be non-empty")
        if name in self._specs:
            raise CandidateImportError(f"candidate module registered twice: {name}")
        path = resolved_path(file_path)
        if not path or not os.path.isfile(path):
            raise CandidateImportError(f"candidate module file not found: {file_path}")
        self._specs[name] = CandidateModuleSpec(name, path, source, code)
        self._effect_module_sources[name] = path_key(path)

    def module_for(self, name: str) -> types.ModuleType | None:
        return self._modules.get(name)

    def load(self, name: str) -> types.ModuleType:
        name = self._qualified(name)
        if self._rolled_back:
            raise CandidateImportError("candidate import transaction has been rolled back")
        if self._finalized:
            # Functions retain the import hook captured when they were compiled.
            # After the rollback window closes it must resolve current published
            # modules, without retaining or re-executing a historical candidate.
            module = self._reuse_project_lkg(name)
            if module is not None:
                return module
            if self._is_external_module(name):
                return self._load_external(name)
            if self._world_domain is not None:
                canonical = self._world_domain.modules._canonical(name)
                if self._world_domain.modules._path(canonical) is not None:
                    return self._world_domain.modules.load(canonical)
            if self._has_lkg_descendant(name):
                return self._create_namespace(name)
            raise CandidateImportError(f"project dependency '{name}' has no published module")
        if name in self._modules:
            return self._modules[name]
        spec = self._specs.get(name)
        if spec is None:
            lkg = self._reuse_project_lkg(name)
            if lkg is not None:
                if self._has_registered_descendant(name):
                    return self._project_package_overlay(name, lkg)
                return lkg
            if self._has_registered_descendant(name):
                return self._create_namespace(name)
            if self._is_external_module(name):
                return self._load_external(name)
            if self._has_lkg_descendant(name):
                return self._create_namespace(name)
            raise CandidateImportError(
                f"candidate import rejected: project dependency '{name}' is not registered in this transaction "
                "and has no valid preloaded LKG module"
            )
        return self._execute(spec)

    def _has_registered_descendant(self, name: str) -> bool:
        prefix = name + "."
        return any(candidate.startswith(prefix) for candidate in self._specs)

    def _has_lkg_descendant(self, name: str) -> bool:
        prefix = name + "."
        return any(
            candidate.startswith(prefix)
            and self._reuse_project_lkg(candidate) is not None
            for candidate in tuple(sys.modules)
        )

    def _create_namespace(self, name: str) -> types.ModuleType:
        if name in self._modules:
            return self._modules[name]
        child_paths = [
            os.path.dirname(spec.file_path)
            for child_name, spec in self._specs.items()
            if child_name.startswith(name + ".") and spec.file_path
        ]
        # Imports performed while validating a live child can publish another
        # module.  Namespace assembly needs one deterministic view of the
        # interpreter table for the whole transaction step.
        live_modules = tuple(sys.modules.items())
        child_paths.extend(
            os.path.dirname(resolved_path(getattr(module, "__file__", "")))
            for child_name, module in live_modules
            if child_name.startswith(name + ".")
            and self._reuse_project_lkg(child_name) is not None
            and getattr(module, "__file__", "")
        )
        if not child_paths:
            raise CandidateImportError(f"candidate namespace is not registered: '{name}'")
        module = types.ModuleType(name)
        module.__path__ = [resolved_path(child_paths[0])]
        module.__package__ = name
        module.__file__ = None
        module.__spec__ = None
        module.__builtins__ = self._builtins_for(module)
        if self._finalized:
            # A child can have been loaded directly, without importing its
            # namespace parent during staging. Ordinary runtime imports may
            # materialize that parent, but never stage another private revision.
            module = sys.modules.setdefault(name, module)
        else:
            self._modules[name] = module
        self._attach_child(name, module)
        return module

    def _project_package_overlay(
        self,
        name: str,
        module: types.ModuleType,
    ) -> types.ModuleType:
        """Return a private package view for candidate child attachment.

        Reusing a project's last-known-good package object directly is safe for
        reads, but attaching a candidate child to that live object before the
        owner commits leaks staged state to the running game.  The shallow
        overlay preserves the package API while keeping child attributes in the
        transaction-private module table.
        """
        cached = self._modules.get(name)
        if cached is not None:
            return cached
        if not hasattr(module, "__path__"):
            return module
        overlay = types.ModuleType(name)
        overlay.__dict__.update(vars(module))
        self._modules[name] = overlay
        self._overlay_names.add(name)
        self._attach_child(name, overlay)
        return overlay

    def _reuse_project_lkg(self, name: str) -> types.ModuleType | None:
        if not self._finalized and name in self._reused_lkg:
            return self._reused_lkg[name]
        module = sys.modules.get(name)
        if module is None:
            return None
        if self._finalized and name in self._published_names:
            # These names were proved while staging. Keep only their identities,
            # never module objects: a later owner publication wins, and a source
            # or package-control save cannot invalidate the live revision first.
            return module
        module_path = resolved_path(getattr(module, "__file__", "") or "")
        project_roots = self._roots
        if (
            self._finalized
            and module_path
            and any(is_path_within(module_path, root) for root in project_roots)
            and not is_preload_python_library(module_path)
            and self._source_name(module_path) == name
        ):
            # A save/rename/delete on disk is not a runtime publication. Keep
            # using the current LKG until its owner replaces/retires sys.modules,
            # including the gap between an atomic save's unlink and replacement.
            return module
        if not module_path or not os.path.isfile(module_path):
            # InxPreload creates path-bound namespace parents for isolated
            # package names. Their synthetic ``__file__`` names an absent
            # __init__.py, so validating that path as a gameplay module would
            # incorrectly reject reversible reference encodings such as
            # ``multiplatform_probe`` -> ``multiplatform_5fprobe``. Prove the
            # namespace against a registered descendant and its real path.
            search_paths = tuple(
                resolved_path(path)
                for path in (getattr(module, "__path__", ()) or ())
                if path
            )
            candidates = ((candidate_name, spec.file_path) for candidate_name, spec in self._specs.items())
            if self._finalized:
                # Synthetic preload/namespace parents have no real __init__.py.
                # Prove them against current children instead of retaining the
                # transaction's source snapshots or old namespace overlays.
                candidates = (
                    (candidate_name, resolved_path(getattr(child, "__file__", "") or ""))
                    for candidate_name, child in tuple(sys.modules.items())
                    if candidate_name.startswith(name + ".") and getattr(child, "__file__", None)
                )
            for candidate_name, candidate_path in candidates:
                if not candidate_name.startswith(name + "."):
                    continue
                if not any(
                    is_path_within(candidate_path, search_path, allow_root=False)
                    and any(is_path_within(search_path, root) for root in project_roots)
                    for search_path in search_paths
                ):
                    continue
                expected = self._source_name(candidate_path)
                if candidate_name != expected:
                    raise CandidateImportError(
                        f"registered project module '{candidate_name}' has a path/name mismatch"
                    )
                if not self._finalized:
                    self._reused_lkg[name] = module
                return module
            return None
        if not any(is_path_within(module_path, root) for root in project_roots):
            # Trusted interpreter/engine modules are not project LKG entries.
            return None
        if is_preload_python_library(module_path):
            # Preload owns this library's imports and teardown, independently
            # of the component transaction's isolated module names.
            return None
        expected = self._source_name(module_path)
        if name != expected:
            raise CandidateImportError(
                f"preloaded project module '{name}' has a path/name mismatch"
            )
        if not self._finalized:
            self._reused_lkg[name] = module
        return module

    def _source_name(self, path):
        name = get_script_module_name(path)
        return self._world_domain.modules.prefix + '.' + name if name and self._world_domain is not None else name

    def _qualified(self, name):
        if self._world_domain is None:
            return name
        modules = self._world_domain.modules
        if name == modules.prefix or name.startswith(modules.prefix + '.'):
            return name
        path = modules._path(name)
        if path is None or (name == 'random' and not any(path.is_relative_to(root) for root in modules.roots)):
            return name
        return modules.prefix + '.' + name

    def _sync_world_modules(self, names):
        if self._world_domain is None:
            return
        modules = self._world_domain.modules
        for name in names:
            if name.startswith(modules.prefix + '.'):
                canonical = modules._canonical(name)
                if name in sys.modules:
                    modules.modules[canonical] = sys.modules[name]
                else:
                    modules.modules.pop(canonical, None)

    def _is_external_module(self, name: str) -> bool:
        """Separate interpreter dependencies from transaction-owned code.

        Resolve the root without executing project package initializers.
        Project code must come from this transaction's source snapshots or
        its published revision, including code reached by dynamic imports.
        """
        if self._world_domain is not None and (name == self._world_domain.modules.prefix
                or name.startswith(self._world_domain.modules.prefix + '.')):
            return False
        root = name.partition(".")[0]
        if root in self._external_roots:
            return self._external_roots[root]
        if is_stdlib_module(root):
            return True
        loaded = sys.modules.get(root)
        spec = getattr(loaded, "__spec__", None) if loaded is not None else importlib.util.find_spec(root)
        if loaded is None and spec is None:
            return False
        locations = tuple(getattr(loaded, "__path__", ()) or ())
        if spec is not None:
            locations += tuple(spec.submodule_search_locations or ())
        origin = getattr(loaded, "__file__", None) if loaded is not None else spec.origin
        if origin and origin not in {"built-in", "frozen"}:
            locations += (origin,)
        external = any(is_preload_python_library(location) for location in locations) or not any(
            is_path_within(resolved_path(location), project_root)
            for location in locations
            for project_root in self._roots
        )
        self._external_roots[root] = external
        return external

    def _load_external(self, name: str):
        if self._world_domain is not None:
            if name == 'sys':
                proxy = self._trusted_proxies.get('sys')
                if proxy is None:
                    proxy = types.ModuleType('sys')
                    proxy.__dict__.update(vars(sys))
                    proxy.modules = ChainMap(self._modules, self._world_domain.modules.sys.modules)
                    self._trusted_proxies['sys'] = proxy
                return proxy
            if name == 'random':
                return self._world_domain.modules.load('random')
        module = sys.modules.get(name)
        if module is None:
            module = importlib.import_module(name)
        if name == "dataclasses":
            return self._dataclasses_proxy(module)
        if name == "importlib":
            return self._importlib_proxy(module)
        return module

    def _importlib_proxy(self, module: types.ModuleType) -> types.ModuleType:
        cached = self._trusted_proxies.get("importlib")
        if cached is not None:
            return cached
        proxy = types.ModuleType("importlib")
        proxy.__dict__.update(vars(module))

        def import_module(name, package=None):
            absolute = importlib.util.resolve_name(name, package) if name.startswith(".") else name
            return self.load(absolute)

        proxy.import_module = import_module
        self._trusted_proxies["importlib"] = proxy
        return proxy

    def _dataclasses_proxy(self, module: types.ModuleType) -> types.ModuleType:
        cached = self._trusted_proxies.get("dataclasses")
        if cached is not None:
            return cached
        proxy = types.ModuleType("dataclasses")
        proxy.__dict__.update(vars(module))
        # Keep CPython's annotation classification and code generation intact.
        # Its functions look up class namespaces through their module's sys
        # global, so bind them to a private interpreter-table view. Class/Field
        # identities stay shared with stdlib; no candidate enters sys.modules,
        # no annotation is evaluated, and slots classes keep their real module.
        private_sys = types.ModuleType("sys")
        private_sys.__dict__.update(vars(sys))
        private_sys.modules = ChainMap({"dataclasses": proxy}, self._modules,
            self._world_domain.modules.sys.modules if self._world_domain is not None else sys.modules)
        proxy.sys = private_sys
        for name, value in vars(module).items():
            if isinstance(value, types.FunctionType) and value.__globals__ is vars(module):
                rebound = types.FunctionType(value.__code__, proxy.__dict__, value.__name__,
                                             value.__defaults__, value.__closure__)
                rebound.__kwdefaults__ = value.__kwdefaults__
                rebound.__annotations__ = dict(value.__annotations__)
                rebound.__qualname__ = value.__qualname__
                rebound.__dict__.update(value.__dict__)
                proxy.__dict__[name] = rebound
        self._trusted_proxies["dataclasses"] = proxy
        return proxy

    def _execute(self, spec: CandidateModuleSpec) -> types.ModuleType:
        if spec.name in self._modules:
            return self._modules[spec.name]
        if spec.namespace:
            module = types.ModuleType(spec.name)
            module.__file__ = None
            module.__path__ = [spec.file_path]
            module.__package__ = spec.name
            module.__spec__ = None
            self._modules[spec.name] = module
            return module
        import_spec = importlib.util.spec_from_file_location(spec.name, spec.file_path)
        if import_spec is None or import_spec.loader is None:
            raise CandidateImportError(f"failed to create candidate module spec: {spec.file_path}")
        if spec.file_path.endswith((os.sep + "__init__.py", "/__init__.py", "\\__init__.py")):
            import_spec.submodule_search_locations = [os.path.dirname(spec.file_path)]
        module = importlib.util.module_from_spec(import_spec)
        if self._world_domain is not None:
            module.__infernux_source_name__ = self._world_domain.modules._canonical(spec.name)
            module.__infernux_serializable_catalog__ = self._world_domain.serializable_types.maps[0]
        self._modules[spec.name] = module
        module.__builtins__ = self._builtins_for(module)
        try:
            code = spec.code
            if code is None and spec.source is not None:
                if not spec.file_path.endswith(".py"):
                    raise CandidateImportError("source candidate must point to a .py file")
                if isinstance(spec.source, bytes):
                    source = spec.source.decode("utf-8")
                else:
                    source = spec.source
                code = compile(source, spec.file_path, "exec", dont_inherit=True)
            elif code is None and spec.file_path.endswith(".py"):
                # A reload candidate must read the current source even when
                # the frontend did not provide a snapshot; never reuse a
                # timestamp/size-matching stale pyc during staging.
                with tokenize.open(spec.file_path) as source_file:
                    source = source_file.read()
                code = compile(source, spec.file_path, "exec", dont_inherit=True)
            if code is None:
                loader = import_spec.loader
                get_code = getattr(loader, "get_code", None)
                if not callable(get_code):
                    raise CandidateImportError(
                        f"candidate loader does not expose code: {spec.file_path}"
                    )
                loaded_code = get_code(spec.name)
                if loaded_code is None:
                    raise CandidateImportError(f"candidate loader returned no code: {spec.file_path}")
                code = loaded_code
            from infernux.components.serializable_object import _candidate_serializable_scope
            from infernux.renderstack.render_effect_compiler import _capture_source_effect_features

            from contextlib import nullcontext
            from infernux.components.value_ownership import value_world
            execution = (value_world(self._world_domain.identity)
                         if self._world_domain is not None else nullcontext())
            with execution, _candidate_serializable_scope(self._serializable_types, spec.name), _capture_source_effect_features(
                spec.file_path, self._effect_features, self._effect_module_sources,
            ):
                exec(code, module.__dict__)
            from .script_source import prepare_module_code
            module.__infernux_accepted_code__ = prepare_module_code(
                spec.name, spec.file_path, source=spec.source, code=code)
            from infernux.components.script_loader import _remember_module_literals
            from infernux.engine.script_closure_reload import remember_closure_code

            _remember_module_literals(module)
            remember_closure_code(module, code)
            previous = sys.modules.get(spec.name)
            if type(previous) is types.ModuleType and _module_source(previous) == spec.file_path:
                remember_closure_code(previous)
        except Exception:
            self._modules.pop(spec.name, None)
            for identity, cls in tuple(self._serializable_types.items()):
                if cls.__module__ == spec.name:
                    del self._serializable_types[identity]
            for type_id, feature in tuple(self._effect_features.items()):
                if feature.source_identity[0] == path_key(spec.file_path):
                    del self._effect_features[type_id]
            raise
        self._effect_sources.add(path_key(spec.file_path))
        self._attach_child(spec.name, module)
        return module

    def _attach_child(self, name: str, module: types.ModuleType) -> None:
        parent_name, _, child_name = name.rpartition(".")
        if not parent_name:
            return
        if self._finalized:
            parent = sys.modules.get(parent_name)
            if parent is not None:
                setattr(parent, child_name, module)
            return
        parent = self._modules.get(parent_name)
        if parent is None:
            live_parent = self._reuse_project_lkg(parent_name)
            if live_parent is not None:
                parent = self._project_package_overlay(parent_name, live_parent)
        if parent is not None:
            setattr(parent, child_name, module)

    def _try_load_child(self, parent_name: str, child_name: str) -> None:
        child = f"{parent_name}.{child_name}"
        if child in self._modules:
            # The child may have been loaded before its namespace package was
            # materialized (for example ``from .helper import VALUE`` followed
            # by ``from . import helper``).  Attach it now to the private parent.
            self._attach_child(child, self._modules[child])
            return
        child_spec = self._specs.get(child)
        if child_spec is not None:
            self.load(child)
            return
        lkg = self._reuse_project_lkg(child)
        if lkg is not None:
            # A preloaded helper can be reused without republishing it, but a
            # later ``from . import helper`` still needs that child on this
            # transaction's private package view.  Never attach it to the live
            # project namespace before commit.
            self._attach_child(child, lkg)
            return
        if self._is_external_module(child):
            # Installed dependencies use ordinary lazy from-list imports.
            try:
                self._load_external(child)
            except ModuleNotFoundError as exc:
                if exc.name != child:
                    raise

    def _builtins_for(self, module: types.ModuleType) -> dict[str, object]:
        values = dict(vars(builtins))
        values["__import__"] = self._import
        return values

    def _import(
        self,
        name: str,
        globals: dict[str, object] | None = None,
        locals: dict[str, object] | None = None,
        fromlist: tuple[str, ...] | list[str] = (),
        level: int = 0,
    ):
        del locals
        package = str((globals or {}).get("__package__") or "")
        try:
            relative_name = ("." * level) + name if level else name
            absolute = importlib.util.resolve_name(relative_name, package) if level else name
        except (ImportError, ValueError) as exc:
            raise CandidateImportError(f"invalid relative candidate import '{name}'") from exc
        absolute = self._qualified(absolute)
        module = self.load(absolute)
        if fromlist:
            for item in fromlist:
                if item == "*":
                    continue
                child = f"{absolute}.{item}"
                # A registered candidate child wins over an attribute copied
                # from the live package into its private overlay.  Otherwise
                # ``from package import child`` could silently retain the LKG
                # child and bypass this transaction's dependency closure.
                if child in self._specs or child in self._modules:
                    self._try_load_child(absolute, item)
                elif not hasattr(module, item):
                    self._try_load_child(absolute, item)
            return self._modules.get(absolute, module)
        root_name = absolute.split(".", 1)[0]
        if self._world_domain is not None and absolute.startswith(self._world_domain.modules.prefix + '.'):
            root_name = self._world_domain.modules.prefix + '.' + self._world_domain.modules._canonical(absolute).split('.', 1)[0]
        if root_name == absolute:
            return module
        root = self.load(root_name)
        self._attach_child(absolute, module)
        return root

    def serializable_type_scope(self):
        """Resolve staged data types while preparing private instance values."""
        from infernux.components.serializable_object import _candidate_serializable_scope

        return _candidate_serializable_scope(self._serializable_types, "")

    def commit(self) -> None:
        if self._rolled_back:
            raise CandidateImportError("candidate import transaction has been rolled back")
        if self._committed:
            return
        try:
            from infernux.engine.runtime_dispatch import assert_runtime_dispatch_safe_point

            assert_runtime_dispatch_safe_point()
            published_names = tuple(
                name for name in self._modules if name not in self._overlay_names
            )
            published_set = set(published_names)
            from infernux.components.serializable_object import _publish_serializable_types

            self._serializable_before = _publish_serializable_types(
                self._serializable_types, published_set, modules=self._modules,
            )
            from infernux.components.script_loader import _carry_module_literal_state

            for name in published_names:
                module = self._modules[name]
                previous = sys.modules.get(name, _MODULE_ABSENT)
                self._before.setdefault(name, previous)
                source = _module_source(module)
                if source is not None and type(previous) is types.ModuleType and _module_source(previous) == source:
                    _carry_module_literal_state(previous, module)
                    self._published_aliases.append(previous)
                sys.modules[name] = module
                accepted = vars(module).get('__infernux_accepted_code__')
                if accepted is not None and self._world_domain is None:
                    from .script_source import publish_module_code
                    self._accepted_code_before.setdefault(path_key(source), publish_module_code(source, accepted))
            # A private overlay is never published as a replacement package.
            # Once every child module is visible, update its live parent as the
            # final package-level side effect and retain an exact before-image.
            for name in published_names:
                parent_name, _, child_name = name.rpartition(".")
                if not parent_name or parent_name in published_set:
                    continue
                parent = sys.modules.get(parent_name)
                if parent is None:
                    continue
                key = (id(parent), child_name)
                if key not in self._parent_before_keys:
                    values = vars(parent)
                    self._parent_before.append(
                        (parent, child_name, child_name in values, values.get(child_name))
                    )
                    self._parent_before_keys.add(key)
                setattr(parent, child_name, self._modules[name])
            from infernux.renderstack.render_effect_compiler import (
                _publish_effect_features, _snapshot_effect_features,
            )

            self._effect_before = _snapshot_effect_features(self._effect_features, self._effect_sources)
            _publish_effect_features(self._effect_features.values(), source_keys=self._effect_sources)
            self._sync_world_modules(published_names)
            self._forward_closures()
            for previous in self._published_aliases:
                self._forward_module_functions(previous, self._modules[previous.__name__])
                previous.__class__ = _PublishedModuleAlias
            self._committed = True
        except Exception:
            self.rollback()
            raise

    def _forward_module_functions(self, previous: types.ModuleType, current: types.ModuleType) -> None:
        from infernux.engine.runtime_dispatch import _copy_function_body

        before, after = vars(previous), vars(current)
        for name, original in tuple(before.items()):
            replacement = after.get(name)
            if not _plain_project_function(original, before, name):
                continue
            if callable(replacement) and not _plain_project_function(replacement, after, name):
                continue
            reference = _PublishedFunction(previous.__name__, previous.__file__, name, _copy_function_body(original))
            self._forward_callable(original, reference)

    def forward_replaced_methods(self, class_snapshots) -> None:
        """Keep callbacks bound before the class body was replaced callable."""
        for owner, attributes in class_snapshots:
            for name, (present, old_descriptor) in attributes.items():
                original = _method_function(old_descriptor)
                if not present or type(original) is not types.FunctionType or original.__module__ != owner.__module__:
                    continue
                qualified = f"{owner.__qualname__}.{name}"
                if not _plain_project_function(original, original.__globals__, qualified, method=True):
                    continue
                descriptor = vars(owner).get(name)
                replacement = _method_function(descriptor)
                if replacement is original or (callable(replacement) and type(descriptor) is not type(old_descriptor)):
                    continue
                self._forward_callable(original, _PublishedMethod(owner, name, type(old_descriptor)))

    def _forward_closures(self) -> None:
        from infernux.engine.runtime_dispatch import _copy_function_body
        from infernux.engine.script_closure_reload import ClosureBody, closure_updates

        pairs = ((previous, self._modules[previous.__name__]) for previous in self._published_aliases)
        for retained, function, (code, namespace) in closure_updates(pairs):
            body = _copy_function_body(function, code, namespace)
            body.__infernux_closure_body__ = True
            if type(retained) is ClosureBody:
                self._closure_bodies.append((retained, function))
                retained.function = body
            else:
                self._forward_callable(function, ClosureBody(body), str(id(function)))

    def _forward_callable(self, original: types.FunctionType, reference, closure_key: str | None = None) -> None:
        namespace = original.__globals__
        identity = id(namespace)
        key = "__infernux_published_functions__"
        if identity not in self._forwarded_namespaces:
            previous = namespace.get(key, _MODULE_ABSENT)
            references = dict(previous) if type(previous) is dict else {}
            self._forwarded_namespaces[identity] = namespace, previous, references
            namespace[key] = references
        references = self._forwarded_namespaces[identity][2]
        name = closure_key if closure_key is not None else original.__qualname__
        references[name] = weakref.proxy(reference) if closure_key is not None else reference
        old_code = original.__code__
        bridge = _bridge_code(len(old_code.co_freevars))
        self._forwarded_functions.append((original, old_code))
        original.__code__ = bridge.replace(
            co_consts=tuple(name if value == "<function>" else value for value in bridge.co_consts),
            co_freevars=old_code.co_freevars,
            co_filename=old_code.co_filename,
            co_name=old_code.co_name, co_qualname=old_code.co_qualname,
        )
        original.__wrapped__ = reference

    def rollback(self) -> None:
        if self._rolled_back:
            return
        if self._finalized:
            raise CandidateImportError("candidate import transaction is durable and cannot be rolled back")
        if self._committed:
            from infernux.engine.runtime_dispatch import assert_runtime_dispatch_safe_point

            assert_runtime_dispatch_safe_point()
        for closure, previous in self._closure_bodies:
            closure.function = previous
        self._closure_bodies.clear()
        for function, code in self._forwarded_functions:
            function.__code__ = code
            function.__dict__.pop("__wrapped__", None)
        self._forwarded_functions.clear()
        for namespace, previous, _ in self._forwarded_namespaces.values():
            key = "__infernux_published_functions__"
            if previous is _MODULE_ABSENT:
                namespace.pop(key, None)
            else:
                namespace[key] = previous
        self._forwarded_namespaces.clear()
        for module in self._published_aliases:
            types.ModuleType.__setattr__(module, "__class__", types.ModuleType)
        self._published_aliases.clear()
        for parent, child_name, present, previous in reversed(self._parent_before):
            if present:
                setattr(parent, child_name, previous)
            else:
                try:
                    delattr(parent, child_name)
                except AttributeError:
                    pass
        for name, previous in self._before.items():
            if previous is _MODULE_ABSENT:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
        self._sync_world_modules(self._before)
        from .script_source import restore_module_codes
        restore_module_codes(self._accepted_code_before)
        self._accepted_code_before.clear()
        from infernux.components.serializable_object import _restore_serializable_types

        _restore_serializable_types(self._serializable_before)
        from infernux.renderstack.render_effect_compiler import _restore_effect_features

        _restore_effect_features(self._effect_before)
        self._effect_features.clear()
        self._effect_sources.clear()
        self._effect_before.clear()
        self._serializable_types.clear()
        self._serializable_before.clear()
        self._modules.clear()
        self._overlay_names.clear()
        self._parent_before.clear()
        self._parent_before_keys.clear()
        self._rolled_back = True
        self._committed = False

    def finalize(self) -> None:
        """Close the rollback window without keeping candidate history alive.

        Published functions and importlib proxies can retain this importer.
        Only proven names, project roots and external classification remain
        useful to runtime imports; all module tables, source snapshots and registry
        before-images belong to the completed publication, not to those hooks.
        """
        if self._rolled_back:
            raise CandidateImportError("candidate import transaction has been rolled back")
        if not self._committed:
            raise CandidateImportError("candidate import transaction must commit before finalization")
        if self._finalized:
            return
        from infernux.engine.runtime_dispatch import assert_runtime_dispatch_safe_point

        assert_runtime_dispatch_safe_point()
        self._published_names = frozenset((*self._modules, *self._reused_lkg))
        self._finalized = True
        self._before.clear()
        self._accepted_code_before.clear()
        self._published_aliases.clear()
        self._forwarded_functions.clear()
        self._forwarded_namespaces.clear()
        self._closure_bodies.clear()
        self._parent_before.clear()
        self._parent_before_keys.clear()
        self._serializable_before.clear()
        self._effect_before.clear()
        self._specs.clear()
        self._modules.clear()
        self._reused_lkg.clear()
        self._trusted_proxies.clear()
        self._overlay_names.clear()
        self._serializable_types.clear()
        self._effect_features.clear()
        self._effect_sources.clear()
        self._effect_module_sources.clear()


_MODULE_ABSENT = object()


__all__ = ["CandidateImportError", "CandidateImportTransaction", "CandidateModuleSpec"]
