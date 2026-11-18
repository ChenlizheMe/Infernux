"""Private import table for build-time source analysis; never publishes live code.

Project imports come only from this build's snapshots. External engine/library
modules keep their ordinary identity. Cycles and dataclasses use a private table.
"""
from __future__ import annotations
import builtins
import importlib
import importlib.util
import os
import sys
import tokenize
import types
from pathlib import Path
from collections import ChainMap
from dataclasses import dataclass
from infernux.engine.path_utils import is_path_within, path_key, resolved_path
from infernux.engine.module_classification import is_stdlib_module
from infernux.engine.project_context import get_project_script_roots, is_preload_python_library


@dataclass(frozen=True, slots=True)
class BuildModuleSpec:
    name: str
    file_path: str
    source: bytes | str | None = None
    code: types.CodeType | None = None
    namespace: bool = False


class BuildSourceImports:
    def __init__(self):
        self._specs = {}
        self._modules = {}
        self._roots = get_project_script_roots()
        self._external_roots = {}
        self._trusted_proxies = {}
        self._serializable_types = {}
        self._effect_features = {}
        self._effect_module_sources = {}
        self._effect_sources = set()

    def register(self, name, file_path, *, source=None, code=None):
        self._specs[name] = BuildModuleSpec(name, resolved_path(file_path), source, code)
        self._effect_module_sources[name] = path_key(file_path)

    def load(self, name):
        if name in self._modules:
            return self._modules[name]
        if name in self._specs:
            return self._execute(self._specs[name])
        if any(child.startswith(name + ".") for child in self._specs):
            return self._create_namespace(name)
        if self._is_external_module(name):
            return self._load_external(name)
        raise ImportError(f"Build source is absent from the current snapshot: {name}")

    def _create_namespace(self, name):
        child, spec = next((child, spec) for child, spec in self._specs.items() if child.startswith(name + "."))
        folder = Path(spec.file_path).parent
        depth = len(child.split(".")) - len(name.split(".")) - 1
        if Path(spec.file_path).name == "__init__.py":
            depth += 1
        for _ in range(depth):
            folder = folder.parent
        module = types.ModuleType(name)
        module.__path__ = [str(folder)]
        module.__package__ = name
        module.__file__ = None
        self._modules[name] = module
        for child_name, value in tuple(self._modules.items()):
            if child_name.rpartition(".")[0] == name:
                setattr(module, child_name.rpartition(".")[2], value)
        return module

    def _attach_child(self, name, module):
        parent, _, child = name.rpartition(".")
        if parent in self._modules:
            setattr(self._modules[parent], child, module)

    def _try_load_child(self, parent, child):
        name = parent + "." + child
        if name in self._specs or name in self._modules:
            self._attach_child(name, self.load(name))
        elif self._is_external_module(name):
            try:
                self._load_external(name)
            except ModuleNotFoundError as exc:
                if exc.name != name:
                    raise

    def close(self):
        self._modules.clear()
        self._trusted_proxies.clear()
        self._specs.clear()

    def _is_external_module(self, name: str) -> bool:
        """Separate interpreter dependencies from transaction-owned code.

            Resolve the root without executing project package initializers.
            Project code must come from the build's frozen source snapshots, including code reached by dynamic imports.
            """
        root = name.partition('.')[0]
        if root in self._external_roots:
            return self._external_roots[root]
        if is_stdlib_module(root):
            return True
        loaded = sys.modules.get(root)
        spec = getattr(loaded, '__spec__', None) if loaded is not None else importlib.util.find_spec(root)
        if loaded is None and spec is None:
            return False
        locations = tuple(getattr(loaded, '__path__', ()) or ())
        if spec is not None:
            locations += tuple(spec.submodule_search_locations or ())
        origin = getattr(loaded, '__file__', None) if loaded is not None else spec.origin
        if origin and origin not in {'built-in', 'frozen'}:
            locations += (origin,)
        external = any((is_preload_python_library(location) for location in locations)) or not any((is_path_within(resolved_path(location), project_root) for location in locations for project_root in self._roots))
        self._external_roots[root] = external
        return external

    def _load_external(self, name: str):
        module = sys.modules.get(name)
        if module is None:
            module = importlib.import_module(name)
        if name == 'dataclasses':
            return self._dataclasses_proxy(module)
        if name == 'importlib':
            return self._importlib_proxy(module)
        return module

    def _importlib_proxy(self, module: types.ModuleType) -> types.ModuleType:
        cached = self._trusted_proxies.get('importlib')
        if cached is not None:
            return cached
        proxy = types.ModuleType('importlib')
        proxy.__dict__.update(vars(module))

        def import_module(name, package=None):
            absolute = importlib.util.resolve_name(name, package) if name.startswith('.') else name
            return self.load(absolute)
        proxy.import_module = import_module
        self._trusted_proxies['importlib'] = proxy
        return proxy

    def _dataclasses_proxy(self, module: types.ModuleType) -> types.ModuleType:
        cached = self._trusted_proxies.get('dataclasses')
        if cached is not None:
            return cached
        proxy = types.ModuleType('dataclasses')
        proxy.__dict__.update(vars(module))
        private_sys = types.ModuleType('sys')
        private_sys.__dict__.update(vars(sys))
        private_sys.modules = ChainMap({'dataclasses': proxy}, self._modules, sys.modules)
        proxy.sys = private_sys
        for name, value in vars(module).items():
            if isinstance(value, types.FunctionType) and value.__globals__ is vars(module):
                rebound = types.FunctionType(value.__code__, proxy.__dict__, value.__name__, value.__defaults__, value.__closure__)
                rebound.__kwdefaults__ = value.__kwdefaults__
                rebound.__annotations__ = dict(value.__annotations__)
                rebound.__qualname__ = value.__qualname__
                rebound.__dict__.update(value.__dict__)
                proxy.__dict__[name] = rebound
        self._trusted_proxies['dataclasses'] = proxy
        return proxy

    def _execute(self, spec: BuildModuleSpec) -> types.ModuleType:
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
            raise ImportError(f'failed to create candidate module spec: {spec.file_path}')
        if spec.file_path.endswith((os.sep + '__init__.py', '/__init__.py', '\\__init__.py')):
            import_spec.submodule_search_locations = [os.path.dirname(spec.file_path)]
        module = importlib.util.module_from_spec(import_spec)
        self._modules[spec.name] = module
        module.__builtins__ = self._builtins_for(module)
        try:
            code = spec.code
            if code is None and spec.source is not None:
                if not spec.file_path.endswith('.py'):
                    raise ImportError('source candidate must point to a .py file')
                if isinstance(spec.source, bytes):
                    source = spec.source.decode('utf-8')
                else:
                    source = spec.source
                code = compile(source, spec.file_path, 'exec', dont_inherit=True)
            elif code is None and spec.file_path.endswith('.py'):
                with tokenize.open(spec.file_path) as source_file:
                    source = source_file.read()
                code = compile(source, spec.file_path, 'exec', dont_inherit=True)
            if code is None:
                loader = import_spec.loader
                get_code = getattr(loader, 'get_code', None)
                if not callable(get_code):
                    raise ImportError(f'candidate loader does not expose code: {spec.file_path}')
                loaded_code = get_code(spec.name)
                if loaded_code is None:
                    raise ImportError(f'candidate loader returned no code: {spec.file_path}')
                code = loaded_code
            from infernux.components.serializable_object import _candidate_serializable_scope
            from infernux.renderstack.render_effect_compiler import _capture_source_effect_features
            from infernux.components._component_registration import candidate_component_registration_scope
            with candidate_component_registration_scope(), _candidate_serializable_scope(self._serializable_types, spec.name), _capture_source_effect_features(spec.file_path, self._effect_features, self._effect_module_sources):
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

    def _builtins_for(self, module: types.ModuleType) -> dict[str, object]:
        values = dict(vars(builtins))
        values['__import__'] = self._import
        return values

    def _import(self, name: str, globals: dict[str, object] | None=None, locals: dict[str, object] | None=None, fromlist: tuple[str, ...] | list[str]=(), level: int=0):
        del locals
        package = str((globals or {}).get('__package__') or '')
        try:
            relative_name = '.' * level + name if level else name
            absolute = importlib.util.resolve_name(relative_name, package) if level else name
        except (ImportError, ValueError) as exc:
            raise ImportError(f"invalid relative candidate import '{name}'") from exc
        module = self.load(absolute)
        if fromlist:
            for item in fromlist:
                if item == '*':
                    continue
                child = f'{absolute}.{item}'
                if child in self._specs or child in self._modules:
                    self._try_load_child(absolute, item)
                elif not hasattr(module, item):
                    self._try_load_child(absolute, item)
            return self._modules.get(absolute, module)
        root_name = absolute.split('.', 1)[0]
        if root_name == absolute:
            return module
        root = self.load(root_name)
        self._attach_child(absolute, module)
        return root
