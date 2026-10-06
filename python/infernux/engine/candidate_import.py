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
import os
import sys
import tokenize
import types
from dataclasses import dataclass
from typing import Iterable

from .path_utils import is_path_within, path_key, resolved_path
from .module_classification import is_stdlib_module
from .project_context import (
    get_project_script_roots,
    get_script_module_name,
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


class CandidateImportTransaction:
    """Execute a candidate dependency closure without publishing modules.

    ``load`` inserts an empty module into ``modules`` before executing it, so
    normal circular imports observe the same object.  The module table is
    never exposed through ``sys.modules`` until ``commit`` is called.
    """

    def __init__(self, *, trusted_modules: Iterable[str] = ()) -> None:
        self._specs: dict[str, CandidateModuleSpec] = {}
        self._modules: dict[str, types.ModuleType] = {}
        self._reused_lkg: dict[str, types.ModuleType] = {}
        self._roots = get_project_script_roots()
        self._external_roots: dict[str, bool] = {}
        # Retain the old keyword for callers; installed imports no longer
        # require registration in an engine-maintained allowlist.
        del trusted_modules
        self._trusted_proxies: dict[str, types.ModuleType] = {}
        self._overlay_names: set[str] = set()
        self._parent_before: list[tuple[types.ModuleType, str, bool, object]] = []
        self._parent_before_keys: set[tuple[int, str]] = set()
        self._before: dict[str, object] = {}
        self._serializable_types: dict[str, type] = {}
        self._serializable_before: dict[str, type | None] = {}
        self._effect_features = {}
        self._effect_module_sources: dict[str, str] = {}
        self._effect_sources: set[str] = set()
        self._effect_before = {}
        self._committed = False
        self._rolled_back = False

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
        if self._rolled_back:
            raise CandidateImportError("candidate import transaction has been rolled back")
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
        if name in self._reused_lkg:
            return self._reused_lkg[name]
        module = sys.modules.get(name)
        if module is None:
            return None
        module_path = resolved_path(getattr(module, "__file__", "") or "")
        project_roots = self._roots
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
            for candidate_name, spec in self._specs.items():
                if not candidate_name.startswith(name + "."):
                    continue
                if not any(
                    is_path_within(spec.file_path, search_path, allow_root=False)
                    and any(is_path_within(search_path, root) for root in project_roots)
                    for search_path in search_paths
                ):
                    continue
                expected = get_script_module_name(spec.file_path)
                if candidate_name != expected:
                    raise CandidateImportError(
                        f"registered project module '{candidate_name}' has a path/name mismatch"
                    )
                self._reused_lkg[name] = module
                return module
            return None
        if not any(is_path_within(module_path, root) for root in project_roots):
            # Trusted interpreter/engine modules are not project LKG entries.
            return None
        expected = get_script_module_name(module_path)
        if name != expected:
            raise CandidateImportError(
                f"preloaded project module '{name}' has a path/name mismatch"
            )
        self._reused_lkg[name] = module
        return module

    def _is_external_module(self, name: str) -> bool:
        """Separate interpreter dependencies from transaction-owned code.

        Resolve the root without executing project package initializers.
        Project code must come from this transaction's source snapshots or
        its published revision, including code reached by dynamic imports.
        """
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
        external = not any(
            is_path_within(resolved_path(location), project_root)
            for location in locations
            for project_root in self._roots
        )
        self._external_roots[root] = external
        return external

    def _load_external(self, name: str):
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
        real_dataclass = module.dataclass

        def candidate_dataclass(cls=None, /, **kwargs):
            def decorate(target):
                annotations = getattr(target, "__annotations__", None)
                original = dict(annotations) if isinstance(annotations, dict) else None
                original_module = getattr(target, "__module__", None)
                if annotations is not None:
                    for key, value in tuple(annotations.items()):
                        if not isinstance(value, str):
                            continue
                        try:
                            resolved = value
                            for _ in range(2):
                                if not isinstance(resolved, str):
                                    break
                                resolved = eval(
                                    resolved,
                                    target.__dict__,
                                    target.__dict__,
                                )
                            annotations[key] = resolved
                        except Exception:
                            pass
                if any(isinstance(value, str) for value in (annotations or {}).values()):
                    # dataclasses uses sys.modules for a few string-annotation
                    # checks. Point that lookup at an existing trusted module,
                    # never at the private candidate module.
                    target.__module__ = "builtins"
                try:
                    return real_dataclass(target, **kwargs)
                finally:
                    if original_module is not None:
                        target.__module__ = original_module
                    if original is not None:
                        target.__annotations__ = original

            return decorate if cls is None else decorate(cls)

        proxy.dataclass = candidate_dataclass
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

            with _candidate_serializable_scope(self._serializable_types, spec.name), _capture_source_effect_features(
                spec.file_path, self._effect_features, self._effect_module_sources,
            ):
                exec(code, module.__dict__)
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
                self._serializable_types, published_set,
            )
            for name in published_names:
                module = self._modules[name]
                self._before.setdefault(name, sys.modules.get(name, _MODULE_ABSENT))
                sys.modules[name] = module
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
            self._committed = True
        except Exception:
            self.rollback()
            raise

    def rollback(self) -> None:
        if self._rolled_back:
            return
        if self._committed:
            from infernux.engine.runtime_dispatch import assert_runtime_dispatch_safe_point

            assert_runtime_dispatch_safe_point()
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


_MODULE_ABSENT = object()


__all__ = ["CandidateImportError", "CandidateImportTransaction", "CandidateModuleSpec"]
