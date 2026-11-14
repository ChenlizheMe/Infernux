"""Source execution facts for preload-owned Python libraries.

Use Python's normal import machinery, but compile the selected source itself.
Timestamp bytecode caches cannot identify the source of a live data schema.
"""

from __future__ import annotations

import hashlib
import builtins
import importlib.machinery
import importlib.util
import sys
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from types import CodeType

from infernux.engine.path_utils import is_path_within, path_key


_import_roots = ContextVar("infernux_source_import_roots", default=())
_executing_sources = ContextVar("infernux_executing_sources", default=())


@dataclass(frozen=True)
class ModuleCode:
    code: CodeType
    source_sha256: str | None


_accepted_module_codes = {}


def prepare_module_code(name, path, *, source=None, code=None):
    """One first-import/stage compiler; published code never consults disk again."""
    key = path_key(path)
    if source is None and code is None and key in _accepted_module_codes:
        return _accepted_module_codes[key]
    if code is None:
        if source is None and path.endswith('.pyc'):
            return ModuleCode(importlib.machinery.SourcelessFileLoader(name, path).get_code(name), None)
        if source is None:
            source = importlib.machinery.SourceFileLoader(name, path).get_data(path)
        code = compile(source, path, 'exec', dont_inherit=True)
    raw = source.encode('utf8') if isinstance(source, str) else source
    return ModuleCode(code, hashlib.sha256(raw).hexdigest() if raw is not None else None)


def publish_module_code(path, value):
    key = path_key(path)
    previous = _accepted_module_codes.get(key)
    _accepted_module_codes[key] = value
    return previous


def restore_module_codes(before):
    for key, previous in before.items():
        if previous is None:
            _accepted_module_codes.pop(key, None)
        else:
            _accepted_module_codes[key] = previous


def retire_module_code(path):
    _accepted_module_codes.pop(path_key(path), None)


def clear_module_codes():
    _accepted_module_codes.clear()


class SourceFileLoader(importlib.machinery.SourceFileLoader):
    def get_code(self, fullname):
        if self.path.endswith('.pyc'):
            code = importlib.machinery.SourcelessFileLoader(fullname, self.path).get_code(fullname)
            self.prepared_code = prepare_module_code(fullname, self.path, code=code)
        else:
            source = self.get_data(self.path)
            self.prepared_code = prepare_module_code(fullname, self.path, source=source)
        self.source_sha256 = self.prepared_code.source_sha256
        return self.prepared_code.code

    def get_source(self, fullname):
        return None if self.path.endswith('.pyc') else super().get_source(fullname)

    def exec_module(self, module):
        from infernux.components.serializable_object import (
            _candidate_serializable_scope, _publish_serializable_types,
        )

        code = self.get_code(module.__name__)
        module.__infernux_source__ = (path_key(self.path), self.source_sha256)
        # Copy facts, not live modules: a dependent can retain a constant or
        # field default after its provider has been unloaded and imported anew.
        namespace = module.__dict__
        namespace["__infernux_source_dependencies__"] = frozenset()

        def remember(imported):
            fact = getattr(imported, "__infernux_source__", None)
            if fact is not None:
                facts = {fact, *getattr(imported, "__infernux_source_dependencies__", ())}
                for scope in (namespace, *_executing_sources.get()):
                    scope["__infernux_source_dependencies__"] |= facts

        def source_import(name, globals=None, locals=None, fromlist=(), level=0):
            imported = builtins.__import__(name, globals, locals, fromlist, level)
            absolute = (importlib.util.resolve_name("." * level + name, globals["__package__"])
                        if level else name)
            parts = absolute.split(".")
            for index in range(1, len(parts) + 1):
                remember(sys.modules.get(".".join(parts[:index])))
            for item in fromlist or ():
                exports = (vars(imported).get("__all__", tuple(
                    name for name in vars(imported) if not name.startswith("_")
                )) if item == "*" else (item,))
                for export in exports:
                    remember(vars(imported).get(export))
            return imported

        module.__builtins__ = dict(vars(builtins), __import__=source_import)
        types = {}
        remember(module)
        token = _executing_sources.set((*_executing_sources.get(), namespace))
        try:
            with _candidate_serializable_scope(types, module.__name__):
                exec(code, namespace)
        finally:
            _executing_sources.reset(token)
        _publish_serializable_types(types, {module.__name__})
        module.__infernux_accepted_code__ = self.prepared_code
        publish_module_code(self.path, self.prepared_code)
        from infernux.components.script_loader import (
            _class_literal_defaults, _declared_class_literals, _remember_module_literals,
        )
        from .script_closure_reload import remember_closure_code
        _class_literal_defaults.update(_declared_class_literals((module,)))
        _remember_module_literals(module)
        remember_closure_code(module, code)


class _PreloadSourceFinder:
    def __init__(self):
        self.path_finder = importlib.machinery.PathFinder

    def find_spec(self, fullname, path=None, target=None):
        from infernux.engine.project_context import (
            is_preload_python_library, get_project_script_roots, _script_project_root,
        )

        spec = self.path_finder.find_spec(fullname, path, target)
        if (spec is not None
                and isinstance(spec.loader, (importlib.machinery.SourceFileLoader,
                                             importlib.machinery.SourcelessFileLoader))
                and (is_preload_python_library(spec.origin)
                     or any(is_path_within(spec.origin, root, allow_root=False)
                            for root in (*_import_roots.get(), *get_project_script_roots(_script_project_root(spec.origin)))))):
            spec.loader = SourceFileLoader(fullname, spec.origin)
        return spec


_finder = _PreloadSourceFinder()


def track_preload_sources() -> None:
    """Keep one PathFinder adapter while any preload owns a Python library."""
    from infernux.engine.project_context import _preload_python_libraries, get_project_root

    active = bool(_preload_python_libraries) or bool(_import_roots.get()) or bool(get_project_root())
    if active and _finder not in sys.meta_path:
        for index, finder in enumerate(sys.meta_path):
            if isinstance(finder, type) and issubclass(finder, importlib.machinery.PathFinder):
                _finder.path_finder = finder
                sys.meta_path.insert(index, _finder)
                break
        else:
            raise RuntimeError("Preload source tracking requires a Python path finder")
    elif not active and _finder in sys.meta_path:
        sys.meta_path.remove(_finder)


@contextmanager
def source_imports(project_root):
    """Also cover helper imports made before own_python_library is declared."""
    token = _import_roots.set((*_import_roots.get(), project_root))
    track_preload_sources()
    try:
        yield
    finally:
        _import_roots.reset(token)
        track_preload_sources()


def validate_loaded_preload_sources(paths) -> None:
    """Reject old live schemas instead of silently restarting active preloads."""
    from infernux.engine.project_context import is_preload_python_library
    from infernux.lib import _Infernux as native

    sources = {path_key(path): path for path in paths}
    hashes = {}
    for module in tuple(sys.modules.values()):
        path = getattr(module, "__file__", None)
        if not path or not is_preload_python_library(path) or (key := path_key(path)) not in sources:
            continue
        facts = {*getattr(module, "__infernux_source_dependencies__", ()),
                 getattr(module, "__infernux_source__", (key, None))}
        for dependency, expected in facts:
            if dependency not in sources:
                raise RuntimeError(f"Loaded preload dependency is absent from the build: {dependency}")
            if dependency not in hashes:
                with native._FileInput(sources[dependency]) as source:
                    hashes[dependency] = source.sha256()
            if expected != hashes[dependency]:
                raise RuntimeError(
                    f"Loaded preload library {path} does not match source {sources[dependency]}. "
                    "Reload the owning preload before building."
                )
