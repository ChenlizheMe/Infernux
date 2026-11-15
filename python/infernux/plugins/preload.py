"""Static discovery and GUID-keyed runtime for :class:`InxPreload`."""

from __future__ import annotations

import ast
import importlib.util
import inspect
import json
import os
import sys
import time
import types
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from infernux.debug import Debug
from infernux.engine.path_utils import (
    VERSION_CONTROL_DIRECTORIES,
    is_path_within,
    lexical_path_key,
    path_key,
    portable_path,
    relative_path,
    resolved_path,
)
from infernux.lifecycle import InxPreload, PreloadContext

from .registry import PluginRegistry
from .project_index import project_guid_paths


_SCRIPT_NAMESPACE = uuid.UUID("90c16393-d740-4fa3-a45f-8003e2459753")
_TYPE_NAMESPACE = uuid.UUID("11a563a9-a178-45b4-b5bc-93c50b8da675")
_SKIPPED_DIRECTORIES = VERSION_CONTROL_DIRECTORIES | {
    "__pycache__", ".venv", "venv", "build", "dist", ".runtime",
}


@dataclass(frozen=True, slots=True)
class _ClassDeclaration:
    path: str
    module: str
    name: str
    bases: tuple[str, ...]

    @property
    def key(self) -> str:
        return f"{self.module}.{self.name}"


@dataclass(slots=True)
class PreloadState:
    identity: str
    script_guid: str
    type_id: str
    type_name: str
    source_path: str
    package_reference: str = ""
    loaded: bool = False
    error: str = ""
    import_ms: float = 0.0
    preload_ms: float = 0.0
    unload_ms: float = 0.0
    module_name: str = ""
    module_names: tuple[str, ...] = ()
    restart_required: bool = False
    restart_reason: str = ""
    contribution_owner: str = ""
    instance: InxPreload | None = field(default=None, repr=False)
    cleanup_callbacks: list[Any] = field(default_factory=list, repr=False)
    unload_completed: bool = field(default=False, repr=False)
    reload_owner: Any = field(default=None, repr=False)

    def snapshot(self) -> dict[str, object]:
        return {
            "identity": self.identity,
            "script_guid": self.script_guid,
            "type_id": self.type_id,
            "type_name": self.type_name,
            "source_path": self.source_path,
            "package_reference": self.package_reference,
            "loaded": self.loaded,
            "error": self.error,
            "import_ms": self.import_ms,
            "preload_ms": self.preload_ms,
            "unload_ms": self.unload_ms,
            "module_names": list(self.module_names),
            "restart_required": self.restart_required,
            "restart_reason": self.restart_reason,
            "contribution_owner": self.contribution_owner,
        }


@dataclass(frozen=True, slots=True)
class _ModuleOnlyImport:
    source_path: str
    package_reference: str
    contribution_owner: str
    module_names: tuple[str, ...]


class PreloadManager:
    """Project-scoped authority for early-import lifecycle candidates."""

    def __init__(
        self,
        project_root: str,
        *,
        engine: Any = None,
        runtime: bool = False,
        registry: PluginRegistry | None = None,
    ) -> None:
        self.project_root = resolved_path(project_root)
        if not self.project_root:
            raise ValueError("PreloadManager requires a project root")
        self.engine = engine
        self.runtime = bool(runtime)
        self.registry = registry or PluginRegistry(self.project_root)
        self.states: dict[str, PreloadState] = {}
        self._module_only_imports: dict[str, _ModuleOnlyImport] = {}
        self.failures: dict[str, str] = {}
        self._ownership_cache: dict[str, dict[str, object]] | None = None
        self._declarations_by_path: dict[str, tuple[_ClassDeclaration, ...]] = {}
        self._declaration_stamps: dict[str, tuple[int, int]] = {}
        self._catalog_initialized = False
        self._translation_packages: set[str] = set()
        self._reload = None
        self._owns_reload = False
        self._reload_adapter = None
        self._bytecode_owners = {}
        self._readiness_records = None
        self._package_readiness = {}
        self.on_failure = None
        # ``infernux`` is the process-wide public facade. Gameplay modules and
        # plugin preloads may retain direct references to it for the whole
        # Editor lifetime, so a reload transaction may mutate attributes but
        # must never replace the module object.
        import infernux as public_namespace

        self._public_namespace = public_namespace

    def _assert_public_namespace_stable(self) -> None:
        current = sys.modules.get("infernux")
        if current is self._public_namespace:
            return
        sys.modules["infernux"] = self._public_namespace
        raise RuntimeError(
            "Plugin lifecycle replaced the process-wide 'infernux' module object"
        )

    def _coordinator(self):
        if self._reload is None:
            from infernux.engine.resources_manager import ResourcesManager
            from infernux.engine.reload.coordinator import ReloadCoordinator
            from infernux.engine.reload.components import ComponentAdapter
            resources = ResourcesManager.instance()
            if resources is not None and path_key(resources._project_path) == path_key(self.project_root):
                self._reload = resources._ensure_event_handler().reload
            else:
                self._reload = ReloadCoordinator(self.project_root, adapter=ComponentAdapter(lambda: ()))
                self._owns_reload = True
            self._reload_adapter = _ReloadPreloads(self)
            self._reload.adapter.add(self._reload_adapter)
        return self._reload

    def _publish_paths(self, paths):
        paths = tuple(paths)
        if not paths:
            return ()
        coordinator = self._coordinator()
        try:
            coordinator.ensure_ready(paths)
        except Exception as exc:
            Debug.log_error(f"Plugin source activation failed: {exc}")
        return tuple(state for state in self.states.values() if path_key(state.source_path) in {path_key(p) for p in paths})

    def reload_all(self) -> tuple[PreloadState, ...]:
        self.unload_all()
        self._refresh_package_readiness()
        self._coordinator()
        self._retire_unready_sources()
        self._ownership_cache = None
        self.failures.clear()
        self._refresh_declaration_catalog()
        candidates = self._candidate_declarations(self._catalog_declarations())
        result = []
        for path in self._ordered_candidate_paths(candidates):
            result.extend(self._load_path(path, {item.name for item in candidates if path_key(item.path) == path_key(path)}))
        return tuple(result)

    def catch_up(self) -> tuple[PreloadState, ...]:
        """Load lifecycle declarations that appeared after the initial preload.

        Editor startup preloads plugins before the AssetDatabase refresh has
        committed, so a brand-new script (no ``.meta`` yet) is not in the GUID
        catalog at that point. Once the refresh has committed, this pass
        re-reads the catalog and loads only the candidates that have no live
        state yet — already-loaded lifecycles (e.g. a running MCP server) are
        left untouched.
        """
        self._refresh_package_readiness()
        self._coordinator()
        self._retire_unready_sources()
        self._ownership_cache = None
        self._refresh_declaration_catalog()
        candidates = self._candidate_declarations(self._catalog_declarations())
        loaded_paths = {path_key(state.source_path) for state in self.states.values()}
        new_by_path: dict[str, set[str]] = {}
        for declaration in candidates:
            key = path_key(declaration.path)
            if key in loaded_paths or key in self.failures:
                continue
            new_by_path.setdefault(key, set()).add(declaration.name)
        if not new_by_path:
            return ()
        ordered = self._ordered_candidate_paths(
            item for item in candidates if path_key(item.path) in new_by_path
        )
        result: list[PreloadState] = []
        for path in ordered:
            result.extend(self._load_path(path, new_by_path[path_key(path)]))
        return tuple(result)

    def reload_path(self, file_path: str) -> tuple[PreloadState, ...]:
        self._refresh_package_readiness()
        self._retire_unready_sources()
        target = resolved_path(file_path)
        coordinator = self._coordinator()
        if not coordinator.graph.owns_path(target):
            return ()
        if os.path.isfile(target):
            state = coordinator.states.get(target)
            if state is not None and state.status == "active" and coordinator.graph.published_source(target) == Path(target).read_bytes():
                return ()
            return self._publish_paths((target,))
        coordinator.request((), removed=(target,))
        coordinator.compile_pending()
        coordinator.drain()
        return ()

    def _unload_paths(self, paths):
        paths = tuple(paths)
        source_paths = []
        for path in paths:
            owner = self._bytecode_owners.pop(resolved_path(path), None)
            if owner is not None:
                try:
                    owner.close()
                except Exception as exc:
                    Debug.log_exception(exc)
            elif path.endswith(".py"):
                source_paths.append(path)
        if self._reload is not None and source_paths:
            self._reload.request((), removed=source_paths)
            self._reload.compile_pending()
            self._reload.drain()

    def unload_all(self) -> tuple[PreloadState, ...]:
        states = tuple(self.states.values())
        self._unload_paths(tuple(self._bytecode_owners))
        if self._reload is not None:
            paths = {state.source_path for state in states}
            paths.update(record.source_path for record in self._module_only_imports.values())
            paths.update(item.path for item in self._candidate_declarations(self._catalog_declarations()))
            self._unload_paths(paths)
        for state in reversed(states):
            if not state.unload_completed:
                self._retire_state(state)
        self._unload_module_only_imports()
        for reference in tuple(self._translation_packages):
            self._unregister_package_translations(reference)
        return tuple(state for state in states if state.restart_required and state.error)

    def close(self):
        failures = self.unload_all()
        if self._reload is not None:
            self._reload.adapter.remove(self._reload_adapter)
            if self._owns_reload:
                self._reload.close()
            self._reload = None
            self._reload_adapter = None
        return failures

    def unload_package(self, reference: str) -> tuple[PreloadState, ...]:
        states = tuple(state for state in self.states.values()
                       if state.package_reference.casefold() == reference.casefold())
        self._unload_paths(tuple(path for path in self._bytecode_owners
                                 if self._path_belongs_to_package(path, reference)))
        if self._reload is not None:
            paths = tuple(record.source_path for record in self._reload.graph.snapshot().modules
                          if self._path_belongs_to_package(record.source_path, reference))
            self._unload_paths(paths)
        for state in reversed(states):
            if not state.unload_completed:
                self._retire_state(state)
        self._unload_module_only_imports(reference)
        self._unregister_package_translations(reference)
        return tuple(state for state in states if state.restart_required and state.error)

    def reload_package(self, reference: str) -> tuple[PreloadState, ...]:
        self._refresh_package_readiness()
        self._coordinator()
        self._retire_unready_sources()
        self._ownership_cache = None
        if self.runtime:
            self.unload_package(reference)
            self._refresh_declaration_catalog()
            candidates = [item for item in self._candidate_declarations(self._catalog_declarations())
                          if self._path_belongs_to_package(item.path, reference)]
            result = []
            for path in self._ordered_candidate_paths(candidates):
                result.extend(self._load_path(path, {item.name for item in candidates if path_key(item.path) == path_key(path)}))
            return tuple(result)
        paths = tuple(self._package_source_paths(reference))
        coordinator = self._coordinator()
        existing = tuple(record.source_path for record in coordinator.graph.snapshot().modules
                         if self._path_belongs_to_package(record.source_path, reference))
        self._unload_paths(existing)
        for path in paths:
            key = path_key(path)
            try:
                self._declarations_by_path[key] = self._read_path_declarations(path)
                self._declaration_stamps[key] = _file_stamp(path)
                self.failures.pop(key, None)
            except (OSError, SyntaxError, ValueError) as exc:
                self._declarations_by_path[key] = ()
                self.failures[key] = f"{type(exc).__name__}: {exc}"
        candidates = [item for item in self._candidate_declarations(self._catalog_declarations())
                      if self._path_belongs_to_package(item.path, reference)]
        loaded = []
        for path in self._ordered_candidate_paths(candidates):
            loaded.extend(self._publish_paths((path,)))
        return tuple(loaded)

    def reload_package_translations(self, reference: str) -> bool:
        """Publish one package catalog without restarting its Editor lifecycle."""

        if self.runtime or not reference:
            return False
        catalog_path = self._package_translation_path(reference)
        try:
            document = self._translation_document(reference)
            if document is None:
                self._unregister_package_translations(reference)
            else:
                from infernux.engine.i18n import register_translation_catalog

                register_translation_catalog(
                    self._translation_owner(reference), document
                )
                self._translation_packages.add(reference.casefold())
            self.failures.pop(path_key(catalog_path), None)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            self.failures[path_key(catalog_path)] = f"{type(exc).__name__}: {exc}"
            Debug.log_error(
                f"Plugin translation reload failed [{catalog_path}]: {exc}"
            )
            return False

        from infernux.engine.ui.window_manager import WindowManager

        windows = WindowManager.instance()
        if windows is not None:
            windows.refresh_type_labels()
        return True

    def forget_package(self, reference: str) -> None:
        """Drop cached declarations after a package has been removed."""

        for key, declarations in tuple(self._declarations_by_path.items()):
            path = declarations[0].path if declarations else key
            if self._path_belongs_to_package(path, reference):
                self._declarations_by_path.pop(key, None)
                self._declaration_stamps.pop(key, None)
                self.failures.pop(key, None)
        self._ownership_cache = None
        self._unregister_package_translations(reference)

    def snapshots(self) -> tuple[dict[str, object], ...]:
        return tuple(state.snapshot() for state in self.states.values())

    def _capture_editor_views(
        self,
        states: tuple[PreloadState, ...] | list[PreloadState],
    ) -> tuple[tuple[str, str], ...]:
        if self.runtime:
            return ()
        owners = tuple(
            dict.fromkeys(
                state.contribution_owner
                for state in states
                if state.contribution_owner
            )
        )
        if not owners:
            return ()
        from infernux.engine.ui.panel_registry import PanelRegistry

        return PanelRegistry.capture_open_views_for_owners(owners)

    def _restore_editor_views(
        self,
        views: tuple[tuple[str, str], ...],
    ) -> None:
        if self.runtime or not views:
            return
        from infernux.engine.ui.panel_registry import PanelRegistry

        PanelRegistry.restore_reloaded_views(views)

    def _refresh_declaration_catalog(self) -> None:
        declarations: dict[str, tuple[_ClassDeclaration, ...]] = {}
        stamps: dict[str, tuple[int, int]] = {}
        package_files = self._package_file_records() if self.runtime else {}
        for path in self._source_paths():
            key = path_key(path)
            try:
                stamp = _file_stamp(path)
                if self._declaration_stamps.get(key) == stamp:
                    declarations[key] = self._declarations_by_path.get(key, ())
                    stamps[key] = stamp
                    continue
                # _source_paths already applies the filesystem role rules.
                declarations[key] = self._read_path_declarations(
                    path, package_files=package_files, role_checked=True
                )
                stamps[key] = stamp
                self.failures.pop(key, None)
            except (OSError, SyntaxError, ValueError) as exc:
                declarations[key] = ()
                self.failures[key] = f"{type(exc).__name__}: {exc}"
        self._declarations_by_path = declarations
        self._declaration_stamps = stamps
        self._catalog_initialized = True

    def _catalog_declarations(self) -> list[_ClassDeclaration]:
        return [
            declaration
            for values in self._declarations_by_path.values()
            for declaration in values
        ]

    def _package_file_records(self) -> dict[str, Mapping[str, object]]:
        records: dict[str, Mapping[str, object]] = {}
        for package in self.registry.installed():
            for item in package.get("files", []):
                if not isinstance(item, Mapping):
                    continue
                for field in ("path_hint", "compiled_path_hint"):
                    hint = portable_path(str(item.get(field, ""))).strip("/")
                    if not hint:
                        continue
                    candidate = os.path.join(self.project_root, *hint.split("/"))
                    records.setdefault(path_key(candidate), item)
        return records

    def _read_path_declarations(
        self, path: str, *, package_files: Mapping[str, Mapping[str, object]] | None = None,
        role_checked: bool = False,
    ) -> tuple[_ClassDeclaration, ...]:
        reference = self._package_for_path(path)
        owner = self.registry.installed_record(reference) if reference else None
        if owner is not None and not bool(owner.get("enabled", True)):
            return ()
        if owner is not None and self._package_readiness.get(reference.casefold()):
            return ()
        if self.runtime and not role_checked and _is_editor_source(path, self.project_root, owner):
            return ()
        if not path.casefold().endswith(".pyc"):
            declarations = _read_declarations(path, self.project_root)
            return declarations
        if package_files is None:
            package_files = self._package_file_records()
        record = package_files.get(path_key(path))
        raw_declarations = (
            record.get("preload_declarations", []) if record is not None else []
        )
        if not isinstance(raw_declarations, list):
            raise ValueError("Compiled preload declarations must be a list")
        if not raw_declarations:
            return ()
        module = _module_name(path, self.project_root)
        declarations: list[_ClassDeclaration] = []
        for raw in raw_declarations:
            if not isinstance(raw, Mapping):
                raise ValueError("Compiled preload declaration must be an object")
            name = str(raw.get("name", ""))
            bases = raw.get("bases", [])
            if not name.isidentifier() or not isinstance(bases, list) or any(
                not isinstance(base, str) or not base for base in bases
            ):
                raise ValueError("Compiled preload declaration is invalid")
            declarations.append(
                _ClassDeclaration(path, module, name, tuple(bases))
            )
        return tuple(declarations)

    @staticmethod
    def _affected_candidate_paths(
        target: str,
        previous: Iterable[_ClassDeclaration],
        current: Iterable[_ClassDeclaration],
        previous_candidates: Iterable[_ClassDeclaration],
        current_candidates: Iterable[_ClassDeclaration],
    ) -> set[str]:
        old_values = list(previous)
        new_values = list(current)
        old_candidates = list(previous_candidates)
        new_candidates = list(current_candidates)
        target_key = path_key(target)
        old_keys = {item.key for item in old_candidates}
        new_keys = {item.key for item in new_candidates}
        affected = {
            target_key
            for item in (*old_values, *new_values)
            if item.key in old_keys or item.key in new_keys
        }
        symbols = {
            value
            for item in (*old_values, *new_values)
            for value in (item.key, item.name)
        }
        changed = True
        all_candidates = (*old_candidates, *new_candidates)
        while changed:
            changed = False
            for item in all_candidates:
                if path_key(item.path) in affected:
                    symbols.update((item.key, item.name))
                    continue
                if not any(
                    base in symbols or base.rsplit(".", 1)[-1] in symbols
                    for base in item.bases
                ):
                    continue
                affected.add(path_key(item.path))
                symbols.update((item.key, item.name))
                changed = True
        return affected

    @staticmethod
    def _candidate_declarations(
        declarations: Iterable[_ClassDeclaration],
    ) -> list[_ClassDeclaration]:
        values = list(declarations)
        by_key = {item.key: item for item in values}
        simple: dict[str, set[str]] = {}
        for item in values:
            simple.setdefault(item.name, set()).add(item.key)
        known = {
            "infernux.lifecycle.InxPreload",
            "infernux.InxPreload",
            "infernux.lifecycle.InxPreload",
            "InxPreload",
        }
        selected: set[str] = set()
        changed = True
        while changed:
            changed = False
            for item in values:
                if item.key in selected:
                    continue
                resolved: set[str] = set()
                for base in item.bases:
                    resolved.add(base)
                    if "." not in base:
                        resolved.add(f"{item.module}.{base}")
                        if len(simple.get(base, ())) == 1:
                            resolved.update(simple[base])
                if not any(base in known or base in selected for base in resolved):
                    continue
                selected.add(item.key)
                known.add(item.key)
                changed = True
        return [item for item in values if item.key in selected]

    def _source_paths(self) -> Iterator[str]:
        guid_paths, native = project_guid_paths(
            self.project_root,
            engine=self.engine,
        )
        if native:
            candidates = {
                path_key(path): path
                for path in guid_paths.values()
                if path.casefold().endswith((".py", ".pyc"))
                and os.path.isfile(path)
            }
            for package in self.registry.installed():
                for item in package.get("files", []):
                    if not isinstance(item, Mapping):
                        continue
                    hint = portable_path(
                        str(
                            item.get("compiled_path_hint", "")
                            or item.get("path_hint", "")
                            if self.runtime
                            else item.get("path_hint", "")
                        )
                    ).strip("/")
                    if not hint.casefold().endswith((".py", ".pyc")):
                        continue
                    path = resolved_path(
                        os.path.join(self.project_root, *hint.split("/"))
                    )
                    if os.path.isfile(path):
                        candidates[path_key(path)] = path
            ownership = self._path_ownership(guid_paths)
            for path in sorted(candidates.values(), key=path_key):
                owner = ownership.get(path_key(path))
                if owner is not None and (not bool(owner.get("enabled", True))
                        or self._package_readiness.get(str(owner.get("reference", "")).casefold())):
                    continue
                if self.runtime and _is_editor_source(
                    path, self.project_root, owner
                ):
                    continue
                yield path
            return
        roots = [
            os.path.join(self.project_root, "Assets"),
            os.path.join(self.project_root, "Packages"),
        ]
        ownership = self._path_ownership()
        for root in roots:
            if not os.path.isdir(root):
                continue
            for walk_root, dirs, names in os.walk(root):
                dirs[:] = sorted(
                    name
                    for name in dirs
                    if name not in _SKIPPED_DIRECTORIES and not name.startswith(".")
                )
                for name in sorted(names):
                    if not name.endswith((".py", ".pyc")) or name.startswith("."):
                        continue
                    path = os.path.join(walk_root, name)
                    owner = ownership.get(path_key(path))
                    if owner is not None and (not bool(owner.get("enabled", True))
                            or self._package_readiness.get(str(owner.get("reference", "")).casefold())):
                        continue
                    if self.runtime and _is_editor_source(path, self.project_root, owner):
                        continue
                    yield path

    def _package_source_paths(self, reference: str) -> tuple[str, ...]:
        record = self.registry.installed_record(reference)
        if record is None or not bool(record.get("enabled", True)):
            return ()
        guid_paths, _native = project_guid_paths(
            self.project_root,
            engine=self.engine,
        )
        result: dict[str, str] = {}
        # Current catalog membership includes scripts authored after install;
        # the durable file ledger only describes uninstall ownership.
        for path in guid_paths.values():
            if (
                path.casefold().endswith((".py", ".pyc"))
                and self._package_for_path(path).casefold() == reference.casefold()
            ):
                result[path_key(path)] = path
        for item in record.get("files", []):
            if not isinstance(item, Mapping):
                continue
            hint = portable_path(
                str(
                    item.get("compiled_path_hint", "")
                    or item.get("path_hint", "")
                    if self.runtime
                    else item.get("path_hint", "")
                )
            ).strip("/")
            if not hint.casefold().endswith((".py", ".pyc")):
                continue
            guid = str(item.get("guid", "")).casefold()
            path = guid_paths.get(guid)
            if not path or not os.path.isfile(path):
                path = resolved_path(
                    os.path.join(self.project_root, *hint.split("/"))
                )
            if path and os.path.isfile(path):
                result[path_key(path)] = path
        return tuple(sorted(result.values(), key=path_key))

    def _path_belongs_to_package(self, path: str, reference: str) -> bool:
        if self._package_for_path(path).casefold() == str(reference).casefold():
            return True
        roots = (
            os.path.join(
                self.project_root, "Packages", *str(reference).split("/")
            ),
        )
        return any(is_path_within(path, root, allow_root=False) for root in roots)

    def _path_ownership(
        self, guid_paths: Mapping[str, str] | None = None
    ) -> dict[str, dict[str, object]]:
        if self._ownership_cache is not None:
            return self._ownership_cache
        result: dict[str, dict[str, object]] = {}
        if guid_paths is None:
            guid_paths, _native = project_guid_paths(
                self.project_root,
                engine=self.engine,
            )
        for package in self.registry.installed():
            for item in package.get("files", []):
                if not isinstance(item, Mapping):
                    continue
                guid = str(item.get("guid", "")).casefold()
                path = guid_paths.get(guid)
                if path and os.path.isfile(path):
                    result[path_key(path)] = package
                compiled_hint = portable_path(
                    str(item.get("compiled_path_hint", ""))
                ).strip("/")
                if compiled_hint:
                    compiled_path = resolved_path(
                        os.path.join(self.project_root, *compiled_hint.split("/"))
                    )
                    if os.path.isfile(compiled_path):
                        result[path_key(compiled_path)] = package
        self._ownership_cache = result
        return result

    def _package_for_path(self, path: str) -> str:
        owner = self._path_ownership().get(path_key(path))
        if owner is not None:
            return str(owner.get("reference", ""))
        from infernux.engine.project_context import package_script_reference

        return package_script_reference(path, self.project_root)

    def package_reference_for_path(self, path: str) -> str:
        """Return the installed package that owns *path*, or an empty string."""

        return self._package_for_path(resolved_path(path))

    def package_errors(self, reference: str) -> tuple[str, ...]:
        """Report discovery/import errors owned by one package, including new files."""
        return tuple(
            f"{portable_path(relative_path(path, self.project_root))}: {error}"
            for path, error in sorted(self.failures.items())
            if self._path_belongs_to_package(path, reference)
        )

    def _refresh_package_readiness(self):
        """Evaluate installed package facts at catalog/lifecycle boundaries."""
        from packaging.specifiers import SpecifierSet
        from packaging.version import Version
        from infernux.version import ENGINE_COMPATIBILITY_VERSION

        records = {str(item["reference"]).casefold(): item for item in self.registry.installed()}
        errors = {}
        visiting = []
        def inspect_package(key):
            if key in errors:
                return errors[key]
            if key in visiting:
                return ("Circular package dependency: " + " -> ".join((*visiting, key)),)
            visiting.append(key)
            record = records[key]
            messages = []
            requirement = str(record.get("engine", ""))
            try:
                if Version(ENGINE_COMPATIBILITY_VERSION) not in SpecifierSet(requirement):
                    messages.append(f"Package {record['reference']} requires Infernux {requirement}; current engine is {ENGINE_COMPATIBILITY_VERSION}")
            except ValueError as exc:
                messages.append(f"Package {record['reference']} has invalid engine requirement {requirement!r}: {exc}")
            for dependency in record.get("dependencies", ()):
                target = str(dependency).casefold()
                if target not in records:
                    messages.append(f"Package {record['reference']} requires missing dependency {dependency}")
                elif not bool(records[target].get("enabled", True)):
                    messages.append(f"Package {record['reference']} requires disabled dependency {dependency}")
                else:
                    messages.extend(f"Dependency {record['reference']} -> {dependency}: {message}"
                                    for message in inspect_package(target))
            visiting.pop()
            errors[key] = tuple(dict.fromkeys(messages))
            return errors[key]
        for key in records:
            inspect_package(key)
        self._readiness_records = records
        self._package_readiness = errors

    def package_readiness_errors(self, reference, *, refreshing=()):
        if self._readiness_records is None:
            self._refresh_package_readiness()
        key = str(reference).casefold()
        messages = list(self._package_readiness.get(key, ()))
        refreshing = {str(item).casefold() for item in refreshing}
        visited = {key}
        def inspect_dependency(parent):
            for dependency in self._readiness_records.get(parent, {}).get("dependencies", ()):
                target = str(dependency).casefold()
                if target in visited:
                    continue
                visited.add(target)
                if target not in refreshing:
                    messages.extend(f"Dependency {reference} -> {dependency} is unavailable: {message}"
                                    for message in self.package_errors(target))
                inspect_dependency(target)
        inspect_dependency(key)
        return tuple(dict.fromkeys(messages))

    def _retire_unready_sources(self):
        if self._reload is None:
            return
        paths = {record.source_path for record in self._reload.graph.snapshot().modules
                 if self._package_readiness.get(self._package_for_path(record.source_path).casefold())}
        paths.update(path for path in self._bytecode_owners
                     if self._package_readiness.get(self._package_for_path(path).casefold()))
        self._unload_paths(paths)

    def _ordered_candidate_paths(
        self, candidates: Iterable[_ClassDeclaration]
    ) -> tuple[str, ...]:
        unique = {path_key(item.path): item.path for item in candidates}
        ranks = self._package_ranks()
        return tuple(
            path
            for _key, path in sorted(
                unique.items(),
                key=lambda item: (
                    ranks.get(self._package_for_path(item[1]).casefold(), -1),
                    portable_path(relative_path(item[1], self.project_root)).casefold(),
                ),
            )
        )

    def _package_ranks(self) -> dict[str, int]:
        installed = {
            str(item.get("reference", "")).casefold(): item
            for item in self.registry.installed()
        }
        ranks: dict[str, int] = {}
        visiting: set[str] = set()

        def visit(key: str) -> int:
            if key in ranks:
                return ranks[key]
            if key in visiting:
                raise RuntimeError(f"Circular InxPackage dependency: {key}")
            visiting.add(key)
            record = installed.get(key, {})
            dependencies = [str(item).casefold() for item in record.get("dependencies", [])]
            rank = 1 + max((visit(item) for item in dependencies if item in installed), default=-1)
            visiting.remove(key)
            ranks[key] = rank
            return rank

        for key in installed:
            if not self._package_readiness.get(key):
                visit(key)
        return ranks

    def _load_bytecode_path(self, path, expected_classes):
        """Player code objects use the same construction and registration owner."""
        from infernux.engine.reload.modules import ModuleBatch, remove_module
        from infernux.engine.reload.owners import ReloadOwner
        from infernux.engine.reload.source import SourceResult

        path = resolved_path(path)
        self._unload_paths((path,))
        module_name = _module_name(path, self.project_root)
        try:
            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot load packaged preload: {path}")
            code = spec.loader.get_code(module_name)
        except Exception as exc:
            self.failures[path_key(path)] = f"{type(exc).__name__}: {exc}"
            Debug.log_error(f"Cannot load packaged preload [{path}]: {exc}")
            return []
        owner = ReloadOwner((path,))
        batch = ModuleBatch({module_name: SourceResult(path, 0, None, code)})
        modules_before = set(sys.modules)
        remove_module(module_name)
        states = []
        owned_names = []
        owner.add_cleanup(lambda: [_unpublish_module(name) for name in owned_names])
        try:
            with owner.enter(), batch.imports():
                modules = batch.load()
                owner.bind_modules(modules)
                records = self._package_file_records()
                batch.publish_components(lambda source: str(records.get(path_key(source), {}).get("guid", "")))
                self._bytecode_owners[path] = owner
                owner.add_cleanup(lambda: self._bytecode_owners.pop(path, None))
                self.failures.pop(path_key(path), None)
                states = self._load_path(path, expected_classes, module=modules[module_name], owner=owner)
                errors = [state.error for state in states if not state.loaded]
                if self.failures.get(path_key(path)):
                    errors.append(self.failures[path_key(path)])
                if errors:
                    raise RuntimeError("; ".join(errors))
                batch.publish()
                owner.active = True
        except Exception as exc:
            self.failures[path_key(path)] = f"{type(exc).__name__}: {exc}"
            try:
                owner.close()
            except Exception as cleanup_error:
                Debug.log_exception(cleanup_error)
            Debug.log_error(f"Packaged preload activation failed [{path}]: {exc}")
        finally:
            names = _new_project_modules(self.project_root, modules_before)
            owned_names.extend(names)
            if owner.closed:
                for name in names:
                    _unpublish_module(name)
            else:
                owner.bind_modules(names)
                for name in names:
                    namespace = sys.modules[name]
                    namespace.__infernux_reload_owner__ = owner
        return states

    def _load_path(self, path: str, expected_classes: set[str], *, module=None, owner=None) -> list[PreloadState]:
        if module is None:
            reference = self._package_for_path(path)
            readiness = self.package_readiness_errors(reference, refreshing=(reference,))
            if readiness:
                self.failures[path_key(path)] = "; ".join(readiness)
                return []
            return (self._load_bytecode_path(path, expected_classes) if path.endswith(".pyc")
                    else list(self._publish_paths((path,))))
        record = self._package_file_records().get(path_key(path))
        script_guid = (
            str(record.get("guid", "")).strip().casefold()
            if record is not None
            else ""
        ) or _ensure_script_guid(path, self.project_root)
        package_reference = self._package_for_path(path)
        try:
            self._register_package_translations(package_reference)
        except Exception as exc:
            self.failures[path_key(path)] = f"{type(exc).__name__}: {exc}"
            Debug.log_error(
                f"InxPackage editor translations failed [{package_reference}]: {exc}"
            )
            return []
        module_name = module.__name__ if module is not None else _module_name(path, self.project_root)
        contribution_owner = "source:" + path_key(path)
        modules_before = set(sys.modules)
        started = time.perf_counter()
        import_ms = (time.perf_counter() - started) * 1000.0
        states: list[PreloadState] = []
        classes = sorted(
            (
                value
                for value in vars(module).values()
                if inspect.isclass(value)
                and value is not InxPreload
                and issubclass(value, InxPreload)
                and value.__module__ == module.__name__
                and not inspect.isabstract(value)
                and value.__name__ in expected_classes
            ),
            key=lambda value: value.__qualname__,
        )
        for preload_type in classes:
            type_id = uuid.uuid5(
                _TYPE_NAMESPACE, f"{script_guid}:{preload_type.__qualname__}"
            ).hex
            identity = f"{script_guid}:{type_id}"
            state = PreloadState(
                identity,
                script_guid,
                type_id,
                preload_type.__qualname__,
                path,
                package_reference,
                import_ms=import_ms,
                module_name=module_name,
                contribution_owner=contribution_owner,
                reload_owner=owner,
            )
            if owner is not None:
                owner.add_cleanup(lambda target=state: self._retire_state(target))
            try:
                instance = preload_type()
                state.instance = instance
                context = PreloadContext(
                    self.project_root,
                    path,
                    script_guid,
                    type_id,
                    package_reference,
                    self.engine,
                    self.runtime,
                    lambda reason, target=state: _mark_restart_required(
                        target, reason
                    ),
                    state.cleanup_callbacks.append,
                    lambda message, target=state: self._schedule_failure(target, message),
                )
                started = time.perf_counter()
                with _editor_contribution_scope(
                    contribution_owner,
                    runtime=self.runtime,
                ):
                    with _temporary_import_paths(
                        path, self.project_root, package_reference
                    ):
                        instance.preload(context)
                self._assert_public_namespace_stable()
                state.preload_ms = (time.perf_counter() - started) * 1000.0
                state.instance = instance
                state.loaded = True
            except Exception as exc:
                state.error = f"{type(exc).__name__}: {exc}"
                cleanup_error = self._run_cleanup_callbacks(state)
                if cleanup_error:
                    state.error += f"; cleanup failed: {cleanup_error}"
                    _mark_restart_required(state, state.error)
                try:
                    _remove_editor_contribution_owner(
                        contribution_owner,
                        runtime=self.runtime,
                    )
                except Exception as cleanup_exc:
                    state.error += f"; contribution cleanup failed: {cleanup_exc}"
                    _mark_restart_required(state, state.error)
                if not state.cleanup_callbacks:
                    from infernux.engine.project_context import release_preload_python_libraries
                    release_preload_python_libraries(f"{self.project_root}:{identity}")
                Debug.log_error(
                    f"InxPreload.preload failed [{path}:{preload_type.__qualname__}]: {exc}"
                )
            self.states[identity] = state
            states.append(state)
        imported_project_modules = _new_project_modules(
            self.project_root, modules_before
        )
        for state in states:
            state.module_names = imported_project_modules
        if not classes:
            # An abstract candidate may also be an authored parent package.
            # Keep its import identity until explicit retirement; descendants
            # must reuse it, and module-level contributions still need an owner.
            key = path_key(path)
            previous = self._module_only_imports.get(key)
            module_names = set(imported_project_modules) | {module_name}
            if previous is not None:
                module_names.update(previous.module_names)
            self._module_only_imports[key] = _ModuleOnlyImport(
                path, package_reference, contribution_owner, tuple(sorted(module_names)),
            )
        return states

    def _unload_module_only_imports(
        self, reference: str = "", *, paths: set[str] | None = None,
    ) -> None:
        for key, record in reversed(tuple(self._module_only_imports.items())):
            if reference:
                if record.package_reference.casefold() != reference.casefold():
                    continue
            elif paths is not None and key not in paths:
                continue
            if not _remove_editor_contribution_owner(record.contribution_owner, runtime=self.runtime):
                raise RuntimeError(
                    f"Cannot unload preload module '{record.source_path}': contributed editor panel refused to close"
                )
            for name in reversed(record.module_names):
                _unpublish_module(name)
            self._module_only_imports.pop(key)

    @staticmethod
    def _translation_owner(reference: str) -> str:
        return f"package:{str(reference or '').strip().casefold()}"

    def _package_translation_path(self, reference: str) -> str:
        return os.path.join(
            self.project_root,
            "Packages",
            *str(reference).split("/"),
            "editor",
            "translations.json",
        )

    def _translation_document(self, reference: str) -> dict[str, object] | None:
        catalog_path = self._package_translation_path(reference)
        if not os.path.isfile(catalog_path):
            return None
        with open(catalog_path, "r", encoding="utf-8") as stream:
            document = json.load(stream)
        if not isinstance(document, dict):
            raise ValueError("editor/translations.json must contain a JSON object")
        return document

    def _validate_package_translations(self, reference: str) -> None:
        if self.runtime or not reference:
            return
        document = self._translation_document(reference)
        if document is None:
            return
        from infernux.engine.i18n import validate_translation_catalog

        validate_translation_catalog(self._translation_owner(reference), document)

    def _register_package_translations(self, reference: str) -> None:
        if self.runtime or not reference:
            return
        key = reference.casefold()
        if key in self._translation_packages:
            return
        document = self._translation_document(reference)
        if document is not None:
            from infernux.engine.i18n import register_translation_catalog

            register_translation_catalog(self._translation_owner(reference), document)
        self._translation_packages.add(key)

    def _unregister_package_translations(self, reference: str) -> None:
        key = str(reference or "").casefold()
        if not key or key not in self._translation_packages:
            return
        if not self.runtime:
            from infernux.engine.i18n import unregister_translation_catalog

            unregister_translation_catalog(self._translation_owner(reference))
        self._translation_packages.discard(key)

    @staticmethod
    def _run_cleanup_callbacks(state: PreloadState) -> str:
        failures = []
        while state.cleanup_callbacks:
            callback = state.cleanup_callbacks.pop()
            try:
                callback()
            except Exception as exc:
                failures.append(f"{type(exc).__name__}: {exc}")
        return "; ".join(failures)

    def _retire_state(self, state):
        self._unload_state(state)
        self.states.pop(state.identity, None)

    def _schedule_failure(self, state, message):
        from infernux.engine.runtime_event_queue import enqueue
        enqueue(lambda: self._fail_current(state, message))

    def _fail_current(self, state, message):
        if self.states.get(state.identity) is not state:
            return
        if self._reload is not None:
            self._reload.request((), removed=(state.source_path,))
        else:
            self._retire_state(state)
        self.failures[path_key(state.source_path)] = message
        Debug.log_error(message, source_file=state.source_path)
        self._notify_failure()

    def _notify_failure(self):
        if self.on_failure is not None:
            try:
                self.on_failure()
            except Exception as exc:
                Debug.log_exception(exc)

    def _unload_state(self, state: PreloadState) -> bool:
        state.loaded = False
        failures = []
        if state.instance is not None and not state.unload_completed:
            state.unload_completed = True
            started = time.perf_counter()
            try:
                with _temporary_import_paths(state.source_path, self.project_root, state.package_reference):
                    state.instance.unload()
            except Exception as exc:
                failures.append(f"unload failed: {type(exc).__name__}: {exc}")
            state.unload_ms = (time.perf_counter() - started) * 1000.0
        state.instance = None
        cleanup_error = self._run_cleanup_callbacks(state)
        if cleanup_error:
            failures.append(cleanup_error)
        try:
            _remove_editor_contribution_owner(state.contribution_owner, runtime=self.runtime)
        except Exception as exc:
            failures.append(f"contribution cleanup failed: {exc}")
        from infernux.engine.project_context import release_preload_python_libraries
        release_preload_python_libraries(f"{self.project_root}:{state.identity}")
        if state.reload_owner is None:
            for name in {*state.module_names, state.module_name}:
                if name:
                    _unpublish_module(name)
        if failures:
            state.error = "; ".join(failures)
            _mark_restart_required(state, state.error)
            Debug.log_error(f"InxPreload retirement failed [{state.identity}]: {state.error}")
        return not failures


class _ReloadPreloads:
    def __init__(self, manager):
        self.manager = manager

    def guid_for_path(self, path):
        return ""

    def validate_sources(self, paths):
        references = {self.manager._package_for_path(path) for path in paths}
        messages = tuple(dict.fromkeys(message for reference in references if reference
                         for message in self.manager.package_readiness_errors(reference, refreshing=references)))
        if messages:
            raise RuntimeError("; ".join(messages))

    def record_dependencies(self, graph):
        if self.manager._readiness_records is None:
            self.manager._refresh_package_readiness()
        by_package = {}
        for record in graph.snapshot().modules:
            reference = self.manager._package_for_path(record.source_path).casefold()
            if reference:
                by_package.setdefault(reference, []).append(record.id.module_name)
        edges = []
        for reference, sources in by_package.items():
            pending = list(self.manager._readiness_records.get(reference, {}).get("dependencies", ()))
            visited = {reference}
            while pending:
                dependency = str(pending.pop()).casefold()
                if dependency in visited:
                    continue
                visited.add(dependency)
                edges.extend((source, target) for source in sources for target in by_package.get(dependency, ()))
                pending.extend(self.manager._readiness_records.get(dependency, {}).get("dependencies", ()))
        graph.record_runtime_imports(edges)

    def retire(self, paths):
        keys = {path_key(path) for path in paths}
        for state in tuple(self.manager.states.values()):
            if path_key(state.source_path) in keys:
                self.manager._retire_state(state)
        for path in paths:
            _remove_editor_contribution_owner("source:" + path_key(path), runtime=self.manager.runtime)

    def prepare(self, modules, owner):
        return tuple(modules.values())

    def bind(self, prepared):
        pass

    def initialize(self, prepared, owner):
        for module in prepared:
            path = resolved_path(module.__file__)
            self.manager.failures.pop(path_key(path), None)
            classes = {value.__name__ for value in vars(module).values()
                       if inspect.isclass(value) and value is not InxPreload
                       and issubclass(value, InxPreload) and not inspect.isabstract(value)
                       and value.__module__ == module.__name__}
            if not classes:
                continue
            states = self.manager._load_path(path, classes, module=module, owner=owner)
            failed = [state.error for state in states if not state.loaded]
            if self.manager.failures.get(path_key(path)):
                failed.append(self.manager.failures[path_key(path)])
            if failed:
                raise RuntimeError("; ".join(failed))
            self.manager.failures.pop(path_key(path), None)
        if owner.coordinator is not None:
            for path in owner.paths:
                state = owner.coordinator.states.get(path)
                if state is not None and state.status == "active":
                    self.manager.failures.pop(path_key(path), None)

    def activate(self, prepared):
        pass

    def fail(self, paths, error):
        for path in paths:
            self.manager.failures[path_key(path)] = error
        self.manager._notify_failure()



def _file_stamp(path: str) -> tuple[int, int]:
    stat = os.stat(path)
    return stat.st_mtime_ns, stat.st_size


def _read_declarations(path: str, project_root: str) -> tuple[_ClassDeclaration, ...]:
    with open(path, "r", encoding="utf-8") as stream:
        tree = ast.parse(stream.read(), filename=path)
    module = _module_name(path, project_root)
    aliases: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                local_name = alias.asname or alias.name.split(".", 1)[0]
                aliases[local_name] = alias.name if alias.asname else local_name
        elif isinstance(node, ast.ImportFrom):
            imported_module = _resolve_import_module(module, node.module or "", node.level)
            for alias in node.names:
                if alias.name == "*":
                    continue
                target = f"{imported_module}.{alias.name}" if imported_module else alias.name
                aliases[alias.asname or alias.name] = target
    result = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        bases = tuple(
            value
            for base in node.bases
            if (value := _expression_name(base, aliases))
        )
        result.append(_ClassDeclaration(path, module, node.name, bases))
    return tuple(result)


def _expression_name(node: ast.AST, aliases: Mapping[str, str]) -> str:
    if isinstance(node, ast.Name):
        return aliases.get(node.id, node.id)
    if isinstance(node, ast.Attribute):
        prefix = _expression_name(node.value, aliases)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    if isinstance(node, ast.Subscript):
        return _expression_name(node.value, aliases)
    return ""


def _module_name(path: str, project_root: str) -> str:
    from infernux.engine.project_context import get_script_module_name

    module_name = get_script_module_name(path, project_root)
    if module_name:
        return module_name
    return f"infernux_project_script_{_ensure_script_guid(path, project_root)}"


def _resolve_import_module(current: str, target: str, level: int) -> str:
    if level <= 0:
        return target
    parts = current.split(".")[:-1]
    trim = max(0, level - 1)
    if trim:
        parts = parts[:-trim]
    if target:
        parts.extend(target.split("."))
    return ".".join(parts)


def _ensure_script_guid(path: str, project_root: str) -> str:
    meta_path = path + ".meta"
    if os.path.isfile(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as stream:
                document = json.load(stream)
            entry = document["metadata"]["guid"]
            if entry.get("type") != "string":
                raise ValueError("metadata.guid.type must be 'string'")
            guid = str(entry["value"]).strip().casefold()
            if len(guid) != 32 or any(char not in "0123456789abcdef" for char in guid):
                raise ValueError("metadata.guid.value must be 32 hexadecimal characters")
            return guid
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid script identity metadata '{meta_path}': {exc}") from exc
    relative = portable_path(relative_path(path, project_root))
    guid = uuid.uuid5(_SCRIPT_NAMESPACE, relative.casefold()).hex
    document = {"metadata": {"guid": {"type": "string", "value": guid}}}
    with open(meta_path, "w", encoding="utf-8", newline="\n") as stream:
        json.dump(document, stream, ensure_ascii=False, indent=4)
        stream.write("\n")
    return guid


def _is_editor_source(
    path: str, project_root: str, owner: Mapping[str, object] | None
) -> bool:
    relative = portable_path(relative_path(path, project_root))
    folded_relative = relative.casefold()
    from infernux.engine.project_context import is_editor_asset_path

    if is_editor_asset_path(relative):
        return True

    if folded_relative.startswith("packages/"):
        from infernux.engine.project_context import package_script_role

        role = package_script_role(path, project_root)
        if role:
            return role == "editor"
        if owner is None:
            return False
        for item in owner.get("files", []):
            if not isinstance(item, Mapping):
                continue
            hints = (
                item.get("path_hint", ""),
                item.get("compiled_path_hint", ""),
            )
            if any(
                hint
                and path_key(
                    os.path.join(
                        project_root,
                        *portable_path(str(hint)).split("/"),
                    )
                )
                == path_key(path)
                for hint in hints
            ):
                return str(item.get("role", "")).casefold() == "editor"
    return False


def _execute_module_file(module_name: str, path: str) -> types.ModuleType:
    """Execute one selected source/bytecode module without stale source bytecode."""
    from infernux.engine.script_source import SourceFileLoader

    loader = None if path.casefold().endswith(".pyc") else SourceFileLoader(module_name, path)
    spec = importlib.util.spec_from_file_location(module_name, path, loader=loader)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load InxPreload candidate: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    parent_name, _, child_name = module_name.rpartition(".")
    if parent_name:
        setattr(sys.modules[parent_name], child_name, module)
    return module


def _unpublish_module(name: str) -> None:
    module = sys.modules.pop(name, None)
    if module is not None:
        from infernux.components.serializable_object import _publish_serializable_types
        _publish_serializable_types({}, {name})
    parent_name, _, child_name = name.rpartition(".")
    parent = sys.modules.get(parent_name)
    if module is not None and parent is not None and getattr(parent, child_name, None) is module:
        delattr(parent, child_name)


def _ensure_module_parents(module_name: str, path: str, package_reference: str) -> None:
    """Bind isolated namespace prefixes and execute authored package parents."""

    parts = module_name.split(".")
    parent_names = [".".join(parts[:index]) for index in range(1, len(parts))]
    directory = os.path.dirname(path)
    if os.path.basename(path) in {"__init__.py", "__init__.pyc"}:
        directory = os.path.dirname(directory)
    directories: dict[str, str] = {}
    for name in reversed(parent_names):
        directories[name] = resolved_path(directory)
        directory = os.path.dirname(directory)

    synthetic_count = 1 + len(package_reference.split("/")) if package_reference else 0
    for index, name in enumerate(parent_names):
        expected = directories[name]
        existing = sys.modules.get(name)
        if existing is not None:
            search_path = getattr(existing, "__path__", None)
            search_locations = tuple(str(item) for item in (search_path or ()))
            if any(path_key(item) == path_key(expected) for item in search_locations):
                continue
            spec = getattr(existing, "__spec__", None)
            if search_path is not None and getattr(spec, "loader", object()) is None:
                search_path.append(expected)
                continue
            existing_path = str(getattr(existing, "__file__", "") or "")
            if existing_path and is_path_within(existing_path, expected):
                continue
            raise ImportError(
                f"InxPreload package identity '{name}' is already owned by "
                f"{existing_path or type(existing).__name__}"
            )
        if index >= synthetic_count:
            source = os.path.join(expected, "__init__.py")
            bytecode = source + "c"
            if os.path.isfile(source):
                _execute_module_file(name, source)
                continue
            if os.path.isfile(bytecode):
                _execute_module_file(name, bytecode)
                continue
        package = types.ModuleType(name)
        package.__package__ = name
        package.__path__ = [expected]
        package.__file__ = os.path.join(expected, "__init__.py")
        package.__spec__ = importlib.util.spec_from_loader(
            name,
            loader=None,
            is_package=True,
        )
        sys.modules[name] = package
        parent_name, _, child_name = name.rpartition(".")
        if parent_name:
            setattr(sys.modules[parent_name], child_name, package)


def _new_project_modules(
    project_root: str, modules_before: set[str]
) -> tuple[str, ...]:
    result: list[str] = []
    project_key = lexical_path_key(project_root)
    for name in set(sys.modules) - modules_before:
        module = sys.modules.get(name)
        path = str(getattr(module, "__file__", "") or "")
        if path and _lexically_below(path, project_key) and is_path_within(
            path, project_root, allow_root=False
        ):
            result.append(name)
    return tuple(sorted(result))


def _package_module_names(
    project_root: str, package_reference: str
) -> tuple[str, ...]:
    roots = (os.path.join(project_root, "Packages", *package_reference.split("/")),)
    root_keys = tuple(lexical_path_key(root) for root in roots)
    result: list[str] = []
    for name, module in tuple(sys.modules.items()):
        path = str(getattr(module, "__file__", "") or "")
        if path and any(
            _lexically_below(path, root_key)
            and is_path_within(path, root, allow_root=False)
            for root, root_key in zip(roots, root_keys)
        ):
            result.append(name)
    return tuple(sorted(result))


def _lexically_below(path: str, root_key: str) -> bool:
    if not path or not root_key:
        return False
    try:
        relative_path(path, root_key, resolve=False, allow_root=False)
    except ValueError:
        return False
    return True


def _mark_restart_required(state: PreloadState, reason: str) -> None:
    state.restart_required = True
    state.restart_reason = str(reason)


@contextmanager
def _editor_contribution_scope(owner: str, *, runtime: bool) -> Iterator[None]:
    if runtime:
        yield
        return
    from infernux.engine.interaction._contributions import contribution_scope

    with contribution_scope(owner):
        yield


def _remove_editor_contribution_owner(owner: str, *, runtime: bool) -> bool:
    if runtime:
        return True
    from infernux.engine.ui.panel_registry import PanelRegistry

    PanelRegistry.remove_owner(owner)
    from infernux.engine.interaction.commands import EditorCommandRegistry
    from infernux.engine.interaction.handles import EditorHandleRegistry
    from infernux.engine.interaction.shortcuts import ShortcutRouter
    # Do not instantiate editor services solely to tear a preload down.
    if EditorHandleRegistry._instance is not None:
        EditorHandleRegistry._instance.unregister_owner(owner)
    if ShortcutRouter._instance is not None:
        ShortcutRouter._instance.unregister_owner(owner)
    if EditorCommandRegistry._instance is not None:
        EditorCommandRegistry._instance.unregister_owner(owner)
    return True


@contextmanager
def _temporary_import_paths(
    path: str, project_root: str, package_reference: str
) -> Iterator[None]:
    candidates = [
        os.path.dirname(path),
        os.path.join(project_root, "Assets"),
        os.path.join(project_root, "Packages"),
    ]
    if package_reference:
        from infernux.engine.project_context import _package_role_root

        package_root = os.path.join(
            project_root, "Packages", *package_reference.split("/")
        )
        candidates.extend(
            [
                package_root,
                _package_role_root(package_root, "runtime"),
                _package_role_root(package_root, "editor"),
            ]
        )
    before = sys.path
    temporary = before.copy()
    try:
        for candidate in reversed(candidates):
            candidate = resolved_path(candidate)
            if candidate and os.path.isdir(candidate) and candidate not in temporary:
                temporary.insert(0, candidate)
        # PathFinder may be iterating this list on a plugin's background
        # thread. Publish a separate list and never shorten its live iterator.
        sys.path = temporary
        from infernux.engine.script_source import source_imports
        with source_imports(project_root):
            yield
    finally:
        sys.path = before


__all__ = ["PreloadManager", "PreloadState"]
