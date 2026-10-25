"""Transactional InxPackage installation and project lifecycle authority."""

from __future__ import annotations

import importlib.util
import importlib.metadata
import importlib.machinery
import json
import os
import posixpath
import re
import shlex
import shutil
import subprocess
import sys
import threading
import uuid
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Iterable, Mapping, TYPE_CHECKING
from urllib.parse import unquote, urlsplit

from packaging.specifiers import SpecifierSet
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import Version

from infernux.core.file_read_cache import FileReadCache
from infernux.debug import Debug
from infernux.engine.path_utils import (
    process_executable_path,
    is_path_within,
    path_key,
    portable_path,
    relative_path,
    resolved_path,
    same_path,
)
from infernux.engine.player_package_native import read_entry
from infernux.engine.python_abi import PYTHON_RUNTIME_DIRECTORY
from infernux.version import ENGINE_VERSION

from .content import (
    PAGE_EXTENSIONS,
    discover_plugin_pages,
    localized_intro as select_plugin_intro,
    read_plugin_pages,
    resolve_plugin_page_asset,
    split_markdown_images,
)
from .cache import SharedPackageCache
from .package import (
    InxPackage,
    InxPackagePreview,
    PACKAGE_EXTENSION,
    PACKAGE_MANIFEST,
    REPOSITORY_PACKAGE_DIRECTORY,
    SOURCE_MANIFEST,
    current_meta_bytes,
    portable_meta_bytes,
    package_control_root,
    package_destination,
    validate_reference,
)
from .preload import PreloadManager
from .project_index import _asset_database, project_guid_paths
from .registry import PluginRegistry
from .source_content import same_source_content

if TYPE_CHECKING:
    from .preload import PreloadState


_InstallProgress = Callable[[str, float, str], None]
_GIT_PROGRESS_PERCENT = re.compile(r"(?<!\d)(\d{1,3})%(?!\d)")
_SOURCE_DESCRIPTOR_FIELDS = frozenset(
    {
        "type",
        "location",
        "reference",
        "revision",
        "subdirectory",
        "official",
        "builtin",
        "repository",
        "cache_scope",
        "cache_location",
        "release_tag",
        "release_url",
        "version",
        "acquisition",
        "commit",
        "source_snapshot",
    }
)


def _require_package_identity(
    metadata: Mapping[str, object], *, expected_reference: str = "", expected_version: str = "",
) -> str:
    reference = validate_reference(str(metadata.get("reference", "")))
    if expected_reference and reference.casefold() != validate_reference(expected_reference).casefold():
        raise RuntimeError(
            f"Plugin reference mismatch: expected {expected_reference}, found {reference}"
        )
    version = str(metadata.get("version", ""))
    if expected_version and version != expected_version:
        raise RuntimeError(
            f"Plugin version mismatch: expected {reference}@{expected_version}, found {version}"
        )
    return reference


def _report_progress(
    callback: _InstallProgress | None,
    stage: str,
    progress: float,
    detail: str = "",
) -> None:
    if callback is not None:
        callback(
            str(stage),
            max(0.0, min(1.0, float(progress))),
            str(detail or ""),
        )


def _scaled_progress(
    callback: _InstallProgress | None,
    start: float,
    end: float,
) -> _InstallProgress | None:
    if callback is None:
        return None
    span = float(end) - float(start)
    return lambda stage, value, detail="": callback(
        stage,
        float(start) + span * float(value),
        detail,
    )


class PackageConflictError(RuntimeError):
    """Raised when installation would overwrite a different durable asset."""


class PackageUpdateConflict(PackageConflictError):
    """Local edits requiring explicit consent before package replacement."""

    def __init__(self, paths: Iterable[str]) -> None:
        self.paths = tuple(sorted(set(paths)))
        super().__init__("Plugin update would replace local edits: " + ", ".join(self.paths))


class _DependencyInUse(RuntimeError):
    def __init__(self, names: Iterable[str]) -> None:
        self.names = tuple(names)
        super().__init__("Loaded Python dependencies require an editor restart: " + ", ".join(self.names))


@dataclass(slots=True)
class PluginState:
    reference: str
    root: str
    enabled: bool = True
    loaded: bool = False
    error: str = ""
    resources: dict[str, str] = field(default_factory=dict)
    lifecycle: tuple[dict[str, object], ...] = ()
    restart_required: bool = False

    def snapshot(self) -> dict[str, object]:
        return {
            "reference": self.reference,
            "root": self.root,
            "enabled": self.enabled,
            "loaded": self.loaded,
            "error": self.error,
            "resources": sorted(self.resources),
            "preload_instances": sum(
                1 for item in self.lifecycle if bool(item.get("loaded"))
            ),
            "lifecycle": list(self.lifecycle),
            "restart_required": self.restart_required,
        }


@dataclass(frozen=True, slots=True)
class _PlannedFile:
    logical_path: str
    destination: str
    destination_relative: str
    guid: str
    role: str
    payload: bytes
    meta_payload: bytes
    owned: bool


@dataclass(frozen=True, slots=True)
class _PluginRequirement:
    reference: str
    archive: bytes | None = None
    preview: InxPackagePreview | None = None


@dataclass(frozen=True, slots=True)
class _PackageRequirements:
    plugins: tuple[_PluginRequirement, ...] = ()
    python_lines: tuple[str, ...] = ()

    @property
    def references(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(item.reference for item in self.plugins))


@dataclass(slots=True)
class _PipInstallEffect:
    before: dict[str, str]
    after: dict[str, str]
    requirements: tuple[dict[str, str], ...]
    command: tuple[str, ...]
    output: str
    rolled_back: bool = False

    @property
    def changes(self) -> tuple[dict[str, str], ...]:
        names = sorted(set(self.before) | set(self.after))
        return tuple(
            {
                "name": name,
                "before": self.before.get(name, ""),
                "after": self.after.get(name, ""),
            }
            for name in names
            if self.before.get(name, "") != self.after.get(name, "")
        )


class PluginManager:
    """One project-scoped authority for InxPackage plugins and dependencies."""

    _instance: "PluginManager | None" = None

    def __init__(
        self, project_root: str, *, engine: Any = None, runtime: bool = False,
        installation_only: bool = False,
    ) -> None:
        self.project_root = resolved_path(project_root)
        if not self.project_root:
            raise ValueError("PluginManager requires a project root")
        self.engine = engine
        self.runtime = bool(runtime)
        self.installation_only = installation_only
        self._prepared_python_baseline: dict[str, str] | None = None
        self.pending_install = None
        if not self.runtime:
            from infernux_editor import read_pending

            self.pending_install = read_pending(self.project_root)
        self.official_catalog_error = ""
        self.python_requirement_error = ""
        self.registry = PluginRegistry(self.project_root)
        self.preloads = PreloadManager(
            self.project_root,
            engine=engine,
            runtime=runtime,
            registry=self.registry,
        )
        self.states: dict[str, PluginState] = {}
        self._resource_manager = None
        self._installing: set[str] = set()
        self._dependency_plans: dict[str, tuple[str, ...]] = {}
        self._deferred_catalog_changes: set[str] = set()
        self._page_workspaces = ExitStack()
        self._cached_page_roots = FileReadCache()
        self._shared_package_cache: SharedPackageCache | None = None
        self._content_reads = FileReadCache()
        self._content_asset_reads = FileReadCache()
        self._archive_reads = FileReadCache()

    def _package_cache(self) -> SharedPackageCache:
        if self._shared_package_cache is None:
            self._shared_package_cache = SharedPackageCache(
                staging_root=os.path.join(
                    self.project_root, "Cache", "Plugins", ".staging"
                )
            )
        return self._shared_package_cache

    @classmethod
    def instance(cls) -> "PluginManager | None":
        return cls._instance

    @classmethod
    def startup(
        cls, project_root: str, *, engine: Any = None, runtime: bool = False
    ) -> "PluginManager":
        current = cls._instance
        normalized = resolved_path(project_root)
        if (
            current is not None
            and current.project_root == normalized
            and current.runtime == bool(runtime)
        ):
            current.engine = engine
            current.preloads.engine = engine
            if not runtime:
                current._reconcile_python_requirements_for_startup()
            current.reload_all()
            return current
        if current is not None:
            current.shutdown()
        manager = cls(normalized, engine=engine, runtime=runtime)
        cls._instance = manager
        if not runtime:
            from .official import (
                OfficialCatalogError,
                install_bundled_packages,
                sync_official_registry,
            )

            try:
                sync_official_registry(normalized)
            except OfficialCatalogError as exc:
                manager.official_catalog_error = str(exc)
                Debug.log_warning(
                    "Official plugin catalog is unavailable; continuing with "
                    f"installed, local, Git, and pip sources: {exc}"
                )
            # An existing built-in package may be reloaded by
            # install_bundled_packages(). Its Python dependencies therefore
            # have to exist before that call; restoring them afterwards leaves
            # the current Editor process with a failed preload even if pip later
            # succeeds.
            manager._reconcile_python_requirements_for_startup()
            install_bundled_packages(normalized, manager=manager)
        # Bundled packages load their own lifecycle exactly once while being
        # synchronized.  Catch up the remaining project/plugin declarations
        # without unloading those live instances and importing them again.
        manager.preloads.catch_up()
        manager._rebuild_states()
        if runtime:
            missing = [state.error for state in manager.states.values()
                       if state.enabled and "Missing package asset" in state.error]
            if missing:
                manager.shutdown()
                raise RuntimeError("Project package checkout is incomplete: " + "; ".join(missing))
        if not runtime:
            manager._attach_resource_events()
        return manager

    def shutdown(self) -> None:
        self._content_reads.clear()
        self._archive_reads.clear()
        self._page_workspaces.close()
        self._cached_page_roots.clear()
        if self._resource_manager is not None:
            self._resource_manager.unregister_script_catalog_callback(
                self._on_script_catalog_changed
            )
        self._resource_manager = None
        failures = self.preloads.unload_all()
        if failures:
            raise RuntimeError(
                "Plugin lifecycle shutdown failed; process restart required: "
                + "; ".join(state.error for state in failures)
            )
        self.states.clear()
        if PluginManager._instance is self:
            PluginManager._instance = None

    def reload_all(self) -> tuple[PluginState, ...]:
        self.preloads.reload_all()
        return self._rebuild_states()

    def _rebuild_states(self) -> tuple[PluginState, ...]:
        snapshots = self.preloads.snapshots()
        installed = self.registry.installed()
        guid_index = self._guid_index() if installed else {}
        self.states.clear()
        for record in installed:
            state = self._state_for_record(record, snapshots, guid_index)
            self.states[state.reference.casefold()] = state
        return tuple(self.states.values())

    def reload(self, reference: str) -> PluginState:
        reference = validate_reference(reference)
        if self.registry.installed_record(reference) is None:
            raise KeyError(f"Plugin is not installed: {reference}")
        if self.installation_only:
            return PluginState(reference, package_control_root(self.project_root, reference))
        self.preloads.reload_package(reference)
        self._rebuild_states()
        if self.states[reference.casefold()].loaded:
            self._publish_package_runtime_scripts(reference)
        return self.states[reference.casefold()]

    def catch_up_preloads(self) -> tuple[PreloadState, ...]:
        """Load lifecycles that entered the GUID catalog after startup preload.

        Called once the startup AssetDatabase refresh has committed, so
        scripts created without a ``.meta`` sidecar still preload on their
        first editor session instead of waiting for the next restart.
        """
        previous_failures = dict(self.preloads.failures)
        loaded = self.preloads.catch_up()
        if loaded or self.preloads.failures != previous_failures:
            self._rebuild_states()
        return loaded

    def _content_root(self, reference: str) -> str:
        if self.registry.is_installed(reference):
            return package_control_root(self.project_root, reference)
        archive = self.cached_reference_path(reference)
        if not archive:
            return package_control_root(self.project_root, reference)
        def prepare(observed):
            observed.watch(archive)
            if not os.path.isfile(archive):
                raise FileNotFoundError(archive)
            # Preview documents only: never install or execute a downloaded plugin.
            preview = InxPackage.inspect(archive)
            root = self._page_workspaces.enter_context(self._package_cache().workspace("pages"))
            files = {str(item["logical_path"]): item for item in preview.file_records}

            def materialize(logical: str) -> None:
                destination = Path(root, *logical.split("/"))
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(read_entry(archive, str(files[logical]["archive_path"])))

            for logical in files:
                if logical.startswith("plugin_pages/"):
                    materialize(logical)
            for page in read_plugin_pages(root, list(preview.metadata["pages"])):
                for block in split_markdown_images(page["content"]):
                    if block["kind"] != "image":
                        continue
                    image = resolve_plugin_page_asset(
                        root, page["path"], block["source"], require_file=False
                    )
                    if image:
                        logical = Path(image).relative_to(root).as_posix()
                        if logical in files and not os.path.isfile(image):
                            materialize(logical)
            return root

        return self._cached_page_roots.get(archive, prepare, current=True)

    def content_pages(
        self,
        record: Mapping[str, object],
        *,
        locale: str | None = None,
    ) -> tuple[dict[str, str], ...]:
        reference = str(record.get("reference", ""))
        if locale is None:
            try:
                from infernux.engine.i18n import get_locale

                selected_locale = get_locale()
            except ImportError:
                selected_locale = "en"
        else:
            selected_locale = locale
        installed = self.registry.is_installed(reference)
        archive = "" if installed else self.cached_reference_path(reference)
        declared_pages = record.get("pages", [])

        def prepare(observed):
            # Observe the lexical project location as well as any extracted
            # preview. Replacing a package directory or retargeting a junction
            # must not keep an old, already-resolved content root alive.
            observed.watch(os.path.join(self.project_root, "Packages", *reference.split("/")))
            if archive:
                observed.watch(archive)
            control_root = self._content_root(reference)
            observed.watch_tree(os.path.join(control_root, "plugin_pages"), suffixes=PAGE_EXTENSIONS)
            descriptors = record.get("pages", [])
            if not descriptors:
                descriptors = list(discover_plugin_pages(control_root))
            return read_plugin_pages(
                control_root,
                descriptors,
                locale=selected_locale,
            )

        try:
            pages = self._content_reads.get(
                (reference, installed, archive, selected_locale,
                 json.dumps(declared_pages, sort_keys=True)), prepare,
            )
        except (OSError, ValueError):
            pages = ()
        if pages:
            return tuple(dict(page) for page in pages)
        intro = select_plugin_intro(record, selected_locale)
        if not intro:
            return ()
        return (
            {
                "id": "intro",
                "title": "Description",
                "path": "",
                "format": "text",
                "content": intro,
            },
        )

    def content_asset_path(
        self,
        record: Mapping[str, object],
        page: Mapping[str, object],
        source: str,
    ) -> str:
        reference = str(record.get("reference", ""))
        installed = self.registry.is_installed(reference)
        archive = "" if installed else self.cached_reference_path(reference)
        page_path = str(page.get("path", ""))

        def prepare(observed):
            observed.watch(os.path.join(self.project_root, "Packages", *reference.split("/")))
            if archive:
                observed.watch(archive)
            try:
                control_root = self._content_root(reference)
                path = resolve_plugin_page_asset(control_root, page_path, source, observations=observed)
                if path:
                    return path
            except ValueError:
                pass
            if not installed:
                return ""
            content_root = os.path.join(self.project_root, "Assets", "Plugins")
            try:
                return resolve_plugin_page_asset(content_root, page_path, source, observations=observed)
            except ValueError:
                return ""

        return self._content_asset_reads.get((reference, installed, archive, page_path, source), prepare)

    def install_package(
        self,
        package_path: str,
        *,
        selected: Iterable[str] | None = None,
        install_dependencies: bool = True,
        source: Mapping[str, object] | None = None,
        progress: _InstallProgress | None = None,
        update: bool = False,
        overwrite_modified: bool = False,
        expected_reference: str = "",
        expected_version: str = "",
    ) -> PluginState:
        if self.pending_install is not None and not self.installation_only:
            raise RuntimeError("Complete or cancel the pending dependency installation first.")
        _report_progress(progress, "inspect_package", 0.36)
        package_path = resolved_path(package_path)
        preview = InxPackage.inspect(package_path)
        # A registry request owns the identity. Reject mismatches before cache,
        # project files, dependency installation or preload publication.
        reference = _require_package_identity(
            preview.metadata, expected_reference=expected_reference, expected_version=expected_version,
        )
        compatibility = str(preview.metadata.get("engine", "")).strip()
        if compatibility and Version(ENGINE_VERSION) not in SpecifierSet(compatibility):
            raise RuntimeError(
                f"InxPackage requires Infernux {compatibility}, current engine is {ENGINE_VERSION}"
            )
        key = reference.casefold()
        if key in self._installing:
            raise RuntimeError(f"Circular plugin dependency: {reference}")
        current = self.registry.installed_record(reference)
        if update:
            if current is None:
                raise KeyError(f"Plugin is not installed: {reference}")
            if threading.current_thread() is not threading.main_thread():
                raise RuntimeError("Plugin update publication must run on the editor thread")
        if current is not None and not update:
            if _same_install(current, preview):
                return self.reload(reference)
            control = current.get("control", {})
            control_path = self._guid_index().get(str(control.get("guid", "")).casefold())
            if control_path:
                with open(control_path, "r", encoding="utf-8") as stream:
                    original = json.load(stream)
                if (
                    str(current.get("version", "")) == str(preview.metadata.get("version", ""))
                    and _package_file_identities(original.get("files", ()))
                    == _package_file_identities(preview.file_records)
                ):
                    return self._extend_package_install(current, preview, selected, progress)
            raise RuntimeError(
                f"Plugin is already installed with different content: {reference}; "
                "select an explicit package update"
            )
        removed: list[str] = []
        if update:
            if selected is not None:
                raise ValueError("An update preserves the existing package selection")
            previous_path = self._prepare_update_baseline(
                current, progress=_scaled_progress(progress, 0.36, 0.40),
            )
            planned, control, removed = self._plan_update(
                current, preview, previous_path=previous_path, overwrite_modified=overwrite_modified,
            )
        requirements = self._plan_requirements(preview)
        dependencies = list(requirements.references)
        enabled = bool(current.get("enabled", True)) if update else True
        # Skipping dependency installation does not erase declared edges. The
        # required plugins must already be present in that mode.
        self._validate_dependency_plan(
            reference, dependencies, enabled=enabled,
            require_installed=not install_dependencies,
        )
        if install_dependencies and not self.installation_only and not self._installing:
            if self._defer_package_python(
                preview, current, selected=selected, source=source, update=update,
                overwrite_modified=overwrite_modified,
            ):
                return PluginState(reference, package_control_root(self.project_root, reference), restart_required=True)
        installed_before = {
            str(item.get("reference", "")).casefold()
            for item in self.registry.installed()
        }
        pip_effect: _PipInstallEffect | None = None
        python_baseline: dict[str, str] | None = None
        stopped = False
        committed = False
        publication_prepared = False

        def prepare_publication() -> None:
            nonlocal stopped, python_baseline, publication_prepared
            if publication_prepared:
                return
            # Dependencies have now been prepared. Validate their actual state
            # before retiring the old version or changing its Python environment.
            self._validate_dependency_plan(
                reference, dependencies, enabled=enabled, require_installed=True,
            )
            if not self.installation_only and threading.current_thread() is threading.main_thread() and any(
                dependency.casefold() not in self.states for dependency in dependencies
            ):
                # A fresh manager can see installed files before its startup
                # pass. Initialize them in graph order without restarting live
                # dependencies, including when installation was skipped.
                self.preloads.catch_up()
                self._rebuild_states()
            if update and not self.installation_only:
                failures = self.preloads.unload_package(reference)
                if failures:
                    raise RuntimeError("Plugin lifecycle could not be stopped; restart required: "
                                       + "; ".join(state.error for state in failures))
                stopped = True
                self._retire_package_runtime_scripts(current)
                if install_dependencies and current.get("python_requirements"):
                    python_baseline = self._python_environment_snapshot(
                        self._project_python_executable(),
                    )
            publication_prepared = True

        self._installing.add(key)
        self._dependency_plans[key] = tuple(dependencies)
        try:
            cache_path, cache_relative = self._cache_package(
                package_path,
                reference,
                str(preview.metadata.get("version", "")),
            )
            resolved_source = dict(source or (current.get("source", {}) if update else {}))
            resolved_source.setdefault("type", "local")
            resolved_source.setdefault("location", package_path)
            resolved_source["cache_location"] = cache_relative
            resolved_source["cache_scope"] = "hub"
            if install_dependencies:
                _report_progress(progress, "resolve_dependencies", 0.48)
                requirement_dependencies, pip_effect = self._install_requirements(
                    preview,
                    progress=_scaled_progress(progress, 0.57, 0.66),
                    plan=requirements,
                    before_python_install=prepare_publication,
                )
                if pip_effect:
                    other_requirements = [
                        str(requirement)
                        for dependency in self.registry.load()["python_dependencies"]
                        for owner in dependency["owners"]
                        if str(owner["reference"]).casefold() != key
                        for requirement in owner["requirements"]
                    ]
                    if not _requirements_satisfied(other_requirements, pip_effect.after):
                        raise PackageConflictError(
                            "Plugin Python requirements conflict with another installed package"
                        )
                dependencies = list(requirement_dependencies)
                if update:
                    self._release_python_dependencies(
                        reference,
                        keep_names={item["name"] for item in pip_effect.requirements}
                        if pip_effect else set(),
                    )
            prepare_publication()
            _report_progress(progress, "plan_assets", 0.68)
            if not update:
                planned, control = self._plan_install(preview, selected)
            transaction = _InstallTransaction(self.project_root)
            registry_before = self.registry.load()
            registry_changed = False
            try:
                _report_progress(progress, "write_assets", 0.76)
                for path in removed:
                    for bytecode in _script_bytecode_paths(path):
                        transaction.remove(bytecode)
                    transaction.remove(path)
                    transaction.remove(path + ".meta")
                for item in planned:
                    if not item.owned:
                        continue
                    if update:
                        for bytecode in _script_bytecode_paths(item.destination):
                            transaction.remove(bytecode)
                    transaction.write(item.destination, item.payload)
                    transaction.write(item.destination + ".meta", portable_meta_bytes(item.meta_payload, item.destination_relative))
                control_payload = (
                    json.dumps(preview.metadata, ensure_ascii=False, indent=2) + "\n"
                ).encode("utf-8")
                if bool(control["owned"]):
                    transaction.write(str(control["absolute_path"]), control_payload)
                    transaction.write(
                        str(control["absolute_path"]) + ".meta",
                        portable_meta_bytes(current_meta_bytes(str(control["guid"])), str(control["path_hint"])),
                    )
                file_records = [
                    {
                        "logical_path": item.logical_path,
                        "path_hint": item.destination_relative,
                        "guid": item.guid,
                        "role": item.role,
                        "owned": item.owned,
                    }
                    for item in planned
                ]
                control_record = {
                    "logical_path": PACKAGE_MANIFEST,
                    "path_hint": str(control["path_hint"]),
                    "guid": str(control["guid"]),
                    "role": "control",
                    "owned": bool(control["owned"]),
                }
                _report_progress(progress, "update_registry", 0.84)
                registry_changed = True
                self.registry.record_install(
                    preview.metadata,
                    files=file_records,
                    control=control_record,
                    package_path=cache_path,
                    source=resolved_source,
                    dependencies=dependencies,
                    enabled=enabled,
                    transaction_id=transaction.id,
                    python_requirements=(
                        pip_effect.requirements if pip_effect else
                        current.get("python_requirements", ())
                        if update and not install_dependencies else ()
                    ),
                    python_changes=(pip_effect.changes if pip_effect else ()),
                    python_environment=(pip_effect.after if pip_effect else None),
                    python_install=(
                        {
                            "syntax": "requirements.txt",
                            "command": list(pip_effect.command),
                            "output": pip_effect.output[-4000:],
                        }
                        if pip_effect else None
                    ),
                )
                transaction.commit()
                committed = True
            except BaseException:
                transaction.rollback()
                if registry_changed:
                    self.registry.restore(registry_before)
                raise
            self._prune_package_directories(removed)
            if not self.installation_only:
                self._acknowledge_installed_scripts(planned)
            if not self.installation_only and threading.current_thread() is threading.main_thread():
                _report_progress(progress, "refresh_assets", 0.90)
                self._refresh_editor_assets()
                _report_progress(progress, "preload_plugin", 0.95)
                state = self.reload(reference)
            else:
                state = PluginState(
                    reference=reference,
                    root=package_control_root(self.project_root, reference),
                )
            _report_progress(progress, "complete", 1.0)
            return state
        except BaseException as install_error:
            if committed or self.installation_only:
                raise
            self._rollback_pip_effect(pip_effect)
            if python_baseline is not None:
                self._restore_python_environment(
                    python_baseline, executable=self._project_python_executable(),
                )
            rollback_errors = self._rollback_new_plugin_dependencies(
                installed_before,
                excluding=reference,
            )
            if rollback_errors:
                raise RuntimeError(
                    f"Plugin install failed and dependency rollback was incomplete for "
                    f"{reference}: {'; '.join(rollback_errors)}"
                ) from install_error
            if stopped:
                self.reload(reference)
            if isinstance(install_error, _DependencyInUse) and len(self._installing) == 1:
                if self._defer_package_python(
                    preview, current, selected=selected, source=source, update=update,
                    overwrite_modified=overwrite_modified,
                ):
                    return PluginState(reference, package_control_root(self.project_root, reference), restart_required=True)
            raise
        finally:
            self._installing.discard(key)
            self._dependency_plans.pop(key, None)

    def _extend_package_install(
        self,
        current: Mapping[str, object],
        preview: InxPackagePreview,
        selected: Iterable[str] | None,
        progress: _InstallProgress | None,
    ) -> PluginState:
        """Add previously unchecked members of the same package, not an upgrade."""
        reference = str(current["reference"])
        installed = {str(item["logical_path"]) for item in current["files"]}
        selected_paths = None if selected is None else {
            portable_path(str(path)).strip("/") for path in selected
        }
        additions = tuple(
            str(item["logical_path"])
            for item in preview.file_records
            if str(item["logical_path"]) not in installed
            and (
                selected_paths is None
                or str(item["logical_path"]) in selected_paths
                or package_destination(reference, str(item["logical_path"])) in selected_paths
            )
        )
        if not additions:
            return self.reload(reference)

        self._installing.add(reference.casefold())
        try:
            planned, _control = self._plan_install(preview, additions)
            transaction = _InstallTransaction(self.project_root)
            registry_before = self.registry.load()
            registry_changed = False
            try:
                _report_progress(progress, "write_assets", 0.76)
                for item in planned:
                    if item.owned:
                        transaction.write(item.destination, item.payload)
                        transaction.write(item.destination + ".meta", portable_meta_bytes(item.meta_payload, item.destination_relative))
                document = self.registry.load()
                record = next(
                    item for item in document["installed"]
                    if str(item["reference"]).casefold() == reference.casefold()
                )
                record["files"].extend({
                    "logical_path": item.logical_path,
                    "path_hint": item.destination_relative,
                    "guid": item.guid,
                    "role": item.role,
                    "owned": item.owned,
                } for item in planned)
                record["transaction_id"] = transaction.id
                _report_progress(progress, "update_registry", 0.84)
                registry_changed = True
                self.registry.save(document)
                transaction.commit()
            except BaseException:
                transaction.rollback()
                if registry_changed:
                    self.registry.restore(registry_before)
                raise
            if not self.installation_only:
                self._acknowledge_installed_scripts(planned)
            if not self.installation_only and threading.current_thread() is threading.main_thread():
                self._refresh_editor_assets()
                state = self.reload(reference)
            else:
                state = PluginState(reference=reference, root=package_control_root(self.project_root, reference))
            _report_progress(progress, "complete", 1.0)
            return state
        finally:
            self._installing.discard(reference.casefold())

    @staticmethod
    def _acknowledge_installed_scripts(planned: Iterable[_PlannedFile]) -> None:
        """The install publication owns these writes, not their delayed echoes."""
        from infernux.core.assets import AssetManager

        for item in planned:
            # Bare content scripts in Assets/Plugins still use the normal asset
            # watcher. Only package Editor/Runtime scripts are published here
            # (or by finalize_background_install on the editor thread).
            if (
                item.owned
                and item.role in {"editor", "runtime"}
                and item.destination.casefold().endswith(".py")
            ):
                for event in ("created", "modified", "deleted"):
                    AssetManager._suppress_watcher_echo(event, item.destination)

    def _rollback_new_plugin_dependencies(
        self,
        installed_before: set[str],
        *,
        excluding: str,
    ) -> tuple[str, ...]:
        """Remove plugin dependencies introduced by one rejected install."""

        excluded = str(excluding).casefold()
        pending = {
            str(item.get("reference", "")).casefold(): str(
                item.get("reference", "")
            )
            for item in self.registry.installed()
            if str(item.get("reference", "")).casefold() not in installed_before
            and str(item.get("reference", "")).casefold() != excluded
        }
        failures: list[str] = []
        while pending:
            records = {
                str(item.get("reference", "")).casefold(): item
                for item in self.registry.installed()
                if str(item.get("reference", "")).casefold() in pending
            }
            removable = [
                key
                for key in pending
                if not any(
                    pending_key != key
                    and pending[key].casefold()
                    in {
                        str(value).casefold()
                        for value in records.get(pending_key, {}).get(
                            "dependencies", []
                        )
                    }
                    for pending_key in pending
                )
            ]
            if not removable:
                failures.append(
                    "dependency cycle prevented rollback: "
                    + ", ".join(sorted(pending.values()))
                )
                break
            for dependency_key in sorted(removable, reverse=True):
                dependency = pending.pop(dependency_key)
                try:
                    self.uninstall(dependency)
                except Exception as exc:
                    failures.append(
                        f"{dependency}: {type(exc).__name__}: {exc}"
                    )
        return tuple(failures)

    def install_reference(
        self,
        reference: str,
        *,
        install_dependencies: bool = True,
        progress: _InstallProgress | None = None,
    ) -> PluginState:
        if self.registry.installed_record(reference) is not None:
            return self.reload(reference)
        record = self.registry.find(reference)
        if record is None:
            raise KeyError(
                f"Plugin reference was not found in the official registry: {reference}"
            )
        source = record.get("source")
        if not isinstance(source, Mapping):
            raise ValueError(f"Plugin registry source is invalid: {reference}")
        descriptor = dict(source)
        pinned_version = str(record.get("version", "")).strip()
        if (
            str(descriptor.get("type", "")).strip().casefold() == "local"
            and descriptor.get("builtin") is True
        ):
            return self.install_source(
                descriptor,
                install_dependencies=install_dependencies,
                progress=progress,
                expected_reference=reference,
                expected_version=pinned_version,
            )
        cache_path = self.cached_reference_path(reference)
        if cache_path:
            return self.install_package(
                cache_path,
                source=descriptor,
                install_dependencies=install_dependencies,
                progress=progress,
                expected_reference=reference,
                expected_version=pinned_version,
            )
        return self.install_source(
            descriptor,
            install_dependencies=install_dependencies,
            progress=progress,
            expected_reference=reference,
            expected_version=pinned_version,
        )

    def available_releases(self, reference: str) -> tuple[dict[str, object], ...]:
        """Read versions from a package's source without changing its project pin."""

        location = self.release_repository(reference)
        if not location:
            return ()
        from .github_releases import list_github_releases

        return list_github_releases(location, expected_reference=reference)

    def release_repository(self, reference: str) -> str:
        """The installed publisher wins over catalog discovery and local caches."""
        record = self.registry.installed_record(reference) or self.registry.find(reference)
        if record is None:
            raise KeyError(f"Plugin reference was not found: {reference}")
        source = record.get("source", {})
        if source.get("type") == "github":
            location = str(source["location"])
        elif source.get("acquisition") == "github-release" and source.get("repository"):
            # Older cache imports recorded the local archive as their location.
            # Their explicit repository still identifies the original publisher.
            location = str(source["repository"])
        elif source.get("official") and source.get("repository"):
            location = str(source["repository"])
        else:
            return ""
        return location

    def download_update(
        self, reference: str, release_tag: str, *, progress: _InstallProgress | None = None,
    ) -> dict[str, object]:
        """Stage an explicitly selected release without changing any project pin."""
        current = self.registry.installed_record(reference)
        if current is None:
            raise KeyError(f"Plugin is not installed: {reference}")
        if not release_tag.strip():
            raise ValueError("Plugin updates require an explicit release tag")
        repository = self.release_repository(reference)
        if not repository:
            raise ValueError("Local author packages are not replaced by remote updates")
        from .github_releases import resolve_github_release

        with self._package_cache().workspace("update") as workspace:
            self._prepare_update_baseline(current, progress=_scaled_progress(progress, 0.0, 0.35))
            release = resolve_github_release(
                repository, workspace, expected_reference=reference,
                release_tag=release_tag, progress=_scaled_progress(progress, 0.35, 1.0),
            )
            assert release is not None  # An exact tag requires the release protocol.
            path, _relative = self._cache_package(
                release.path, reference, str(release.source["version"]),
            )
            source = {**release.source, "type": "github", "location": repository,
                      "repository": repository, "acquisition": "github-release"}
            return {"reference": reference, "path": path, "source": source}

    def update_reference(
        self, reference: str, release_tag: str, *, overwrite_modified: bool = False,
        progress: _InstallProgress | None = None,
    ) -> PluginState:
        release = self.download_update(reference, release_tag, progress=progress)
        return self.install_package(
            str(release["path"]), source=release["source"], update=True,
            overwrite_modified=overwrite_modified, progress=progress,
        )

    def cached_reference_path(self, reference: str) -> str:
        coordinates = self.registry._download_coordinates(reference)
        if coordinates is None:
            return ""
        cache = self._package_cache()
        version, location = coordinates
        if location:
            location = SharedPackageCache.validate_location(location)

        def prepare(observed):
            candidates = []
            if location:
                candidates.append(os.path.join(cache.root, *location.split("/")))
            if version:
                candidates.append(os.path.join(cache.root, *cache.relative_path(reference, version).split("/")))
            for candidate in dict.fromkeys(candidates):
                observed.watch(candidate)
                if os.path.isfile(candidate):
                    return resolved_path(candidate)
            return ""

        return self._archive_reads.get((reference, location, version), prepare)

    def _installed_archive_path(self, record: Mapping[str, object]) -> str:
        location = SharedPackageCache.validate_location(str(record["source"]["cache_location"]))
        return resolved_path(os.path.join(self._package_cache().root, *location.split("/")))

    def _prepare_update_baseline(
        self, record: Mapping[str, object], *, progress: _InstallProgress | None = None,
    ) -> str:
        path = self._installed_archive_path(record)
        if os.path.isfile(path):
            return path
        source = self._source_descriptor(record["source"])
        if source["type"] == "local" and source.get("builtin"):
            from .official import official_release_source

            published = official_release_source(str(record["reference"]), str(record["version"]))
            if published is not None:
                source = published
        source["reference"] = str(record["reference"])
        if source["type"] in {"git", "github"}:
            if source["type"] == "git" or source.get("source_snapshot"):
                if not source.get("commit"):
                    raise ValueError("Restoring a pinned Git package requires its exact commit")
            elif not source.get("release_tag"):
                raise ValueError("Restoring a pinned GitHub package requires its exact release tag")
        with self._package_cache().workspace("update-baseline") as workspace:
            acquired, _source = self._materialize_source(source, workspace, progress=progress)
            preview = InxPackage.inspect(acquired)
            reference, version = str(record["reference"]), str(record["version"])
            _require_package_identity(preview.metadata, expected_reference=reference, expected_version=version)
            path, _relative = self._cache_package(acquired, reference, version)
            return path

    def download_reference(
        self,
        reference: str,
        *,
        force: bool = False,
        release_tag: str = "",
        progress: _InstallProgress | None = None,
    ) -> dict[str, object]:
        """Download one registry package into the shared Hub library."""

        record = self.registry.find(reference)
        if record is None:
            raise KeyError(f"Plugin reference was not found: {reference}")
        cached = self.cached_reference_path(reference)
        if cached and not force and not release_tag:
            _require_package_identity(
                InxPackage.inspect(cached).metadata, expected_reference=reference,
                expected_version=str(record.get("version", "")).strip(),
            )
            return {"reference": reference, "path": cached, "cached": True}
        source = record.get("source")
        if not isinstance(source, Mapping):
            raise ValueError(f"Plugin registry source is invalid: {reference}")
        descriptor = self._source_descriptor(source)
        with self._package_cache().workspace("download") as workspace:
            package_path, acquired_source = self._materialize_source(
                descriptor,
                workspace,
                progress=progress,
                release_tag=release_tag,
            )
            preview = InxPackage.inspect(package_path)
            actual_reference = _require_package_identity(
                preview.metadata, expected_reference=reference,
                expected_version="" if release_tag else str(record.get("version", "")).strip(),
            )
            compatibility = str(preview.metadata.get("engine", "")).strip()
            if compatibility and Version(ENGINE_VERSION) not in SpecifierSet(
                compatibility
            ):
                raise RuntimeError(
                    f"InxPackage requires Infernux {compatibility}, current engine "
                    f"is {ENGINE_VERSION}"
                )
            version = str(preview.metadata.get("version", ""))
            cache = self._package_cache()
            destination = cache.store(
                package_path,
                reference=actual_reference,
                version=version,
            )
            cached_source = dict(acquired_source)
            cached_source.update(
                {
                    "cache_scope": "hub",
                    "cache_location": cache.relative_path(actual_reference, version),
                }
            )
            self.registry.add_package(
                actual_reference,
                name=str(preview.metadata.get("name", record.get("name", ""))),
                version=str(preview.metadata.get("version", "")),
                engine=compatibility,
                dependencies=(),
                intro=str(preview.metadata.get("intro", record.get("intro", ""))),
                intros=preview.metadata.get("intros", record.get("intros", {})),
                pages=preview.metadata.get("pages", record.get("pages", ())),
                category=str(record.get("category", "Other")),
                targets=record.get("targets", ()),
                source=cached_source,
            )
            _report_progress(progress, "download_complete", 1.0)
            return {
                "reference": actual_reference,
                "path": destination,
                "cached": False,
            }

    def download_source(
        self, source: Mapping[str, object] | str, *, progress: _InstallProgress | None = None,
        update: bool = False,
    ) -> dict[str, object]:
        """Acquire a source into the shared cache without publishing project state."""
        _report_progress(progress, "resolve_source", 0.04)
        descriptor = self._source_descriptor(source)
        descriptor.update(dict(source) if isinstance(source, Mapping) else {})
        with self._package_cache().workspace("source") as workspace:
            package_path, acquired_source = self._materialize_source(descriptor, workspace, progress=progress)
            preview = InxPackage.inspect(package_path)
            reference = validate_reference(str(preview.metadata["reference"]))
            path, _relative = self._cache_package(package_path, reference, str(preview.metadata["version"]))
            if update:
                current = self.registry.installed_record(reference)
                if current is None:
                    raise KeyError(f"Plugin is not installed: {reference}")
                self._prepare_update_baseline(current, progress=_scaled_progress(progress, 0.36, 0.40))
            return {"reference": reference, "path": path, "source": acquired_source}

    def install_source(
        self,
        source: Mapping[str, object] | str,
        *,
        install_dependencies: bool = True,
        progress: _InstallProgress | None = None,
        update: bool = False,
        expected_reference: str = "",
        expected_version: str = "",
    ) -> PluginState:
        _report_progress(progress, "resolve_source", 0.04)
        descriptor = self._source_descriptor(source)
        descriptor.update(dict(source) if isinstance(source, Mapping) else {})
        with self._package_cache().workspace("source") as workspace:
            package_path, acquired_source = self._materialize_source(
                descriptor,
                workspace,
                progress=progress,
            )
            return self.install_package(
                package_path,
                install_dependencies=install_dependencies,
                source=acquired_source,
                progress=progress,
                update=update,
                expected_reference=expected_reference,
                expected_version=expected_version,
            )

    def install_pip(
        self,
        syntax: str,
        *,
        progress: _InstallProgress | None = None,
    ) -> dict[str, object]:
        if self.pending_install is not None:
            raise RuntimeError("Complete or cancel the pending dependency installation first.")
        _report_progress(progress, "parse_pip", 0.08)
        raw = str(syntax or "").strip()
        if not raw:
            raise ValueError("pip syntax cannot be empty")
        tokens = [token.strip("\"'") for token in shlex.split(raw, posix=os.name != "nt")]
        lowered = [token.casefold() for token in tokens]
        if len(tokens) >= 2 and os.path.basename(lowered[0]).startswith("pip") and lowered[1] == "install":
            arguments = tokens[2:]
        elif len(tokens) >= 4 and lowered[1:4] == ["-m", "pip", "install"]:
            arguments = tokens[4:]
        elif lowered[:3] == ["-m", "pip", "install"]:
            arguments = tokens[3:]
        else:
            arguments = tokens
        if not arguments:
            raise ValueError("pip install syntax contains no arguments")
        executable = self._project_python_executable()
        _report_progress(progress, "inspect_python_environment", 0.22)
        before = self._python_environment_snapshot(executable)
        requirements = _pip_requirement_targets(arguments)
        install = self._resolve_python_install(arguments, project_targets=(item["name"] for item in requirements))
        from infernux_editor import resolved_arguments

        command = [executable, "-m", "pip", "install", "--no-deps", "--force-reinstall", *resolved_arguments(install)]
        try:
            _report_progress(progress, "install_python_dependencies", 0.46)
            result = (
                self._run_python_change(command, {canonicalize_name(item["metadata"]["name"]) for item in install} & before.keys()) if install else
                subprocess.CompletedProcess(command, 0, "Requirements already satisfied.")
            )
            _report_progress(progress, "verify_python_environment", 0.86)
            after = self._python_environment_snapshot(executable)
            requested = tuple(item["requirement"] for item in requirements)
            if not _requirements_satisfied(requested, after):
                raise RuntimeError("pip completed but requested Python requirements remain unresolved")
            changes = _python_environment_changes(before, after)
            self._activate_installed_python_paths(
                before, after, executable=executable,
                reinstalled=(canonicalize_name(item["metadata"]["name"]) for item in install),
            )
            self.registry.record_python_install(
                syntax=raw,
                command=command,
                output=result.stdout,
                requirements=arguments,
                dependency_requirements=requirements,
                changes=changes,
                owner="@project",
                python_environment=after,
            )
        except _DependencyInUse as used:
            self._stage_python_install({
                "request": {"kind": "pip", "syntax": raw, "arguments": arguments},
                "before": before, "install": install, "loaded": used.names,
            })
            _report_progress(progress, "complete", 1.0)
            return {"ok": True, "restart_required": True}
        except BaseException as install_error:
            try:
                self._restore_python_environment(before, executable=executable)
            except Exception as rollback_error:
                raise RuntimeError(
                    "pip install failed and Python environment rollback was "
                    f"incomplete: {rollback_error}"
                ) from install_error
            raise
        _report_progress(progress, "complete", 1.0)
        return {
            "ok": True,
            "command": command,
            "output": result.stdout[-4000:],
        }

    def _resolve_python_install(
        self, arguments: Iterable[str], *, replacing: Iterable[str] = (), project_targets: Iterable[str] = (),
    ) -> list[dict]:
        replacing = self._installing | {name.casefold() for name in replacing}
        project_targets = set(project_targets)
        constraints = list(importlib.metadata.distribution("infernux").requires or ())
        for dependency in self.registry.load()["python_dependencies"]:
            for owner in dependency["owners"]:
                reference = str(owner["reference"]).casefold()
                if reference in replacing or (reference.startswith("@") and dependency["name"] in project_targets):
                    continue
                constraints.extend(owner["requirements"])
        with self._package_cache().workspace("python-plan") as workspace:
            report = Path(workspace) / "report.json"
            constraint = Path(workspace) / "constraints.txt"
            normalized = []
            for value in constraints:
                requirement = Requirement(value)
                requirement.extras.clear()
                normalized.append(str(requirement))
            constraint.write_text("\n".join(dict.fromkeys(normalized)) + "\n", encoding="utf-8")
            self._run_process([
                self._project_python_executable(), "-m", "pip", "install",
                "--dry-run", "--report", str(report), "--constraint", str(constraint), *arguments,
            ], cwd=self.project_root)
            return [{
                "metadata": {key: item["metadata"][key] for key in ("name", "version")},
                "download_info": {key: value for key, value in item["download_info"].items()
                                  if key in {"url", "vcs_info", "dir_info", "subdirectory"}},
            } for item in json.loads(report.read_text(encoding="utf-8"))["install"]]

    def _stage_python_install(self, operation: dict) -> None:
        from infernux_editor import save_pending

        save_pending(self.project_root, operation)
        self.pending_install = {**operation, "prepared": False}
        from infernux.engine.interaction import EditorInteractionCore

        if self.engine is not None and EditorInteractionCore.instance() is not None:
            from infernux.engine.runtime_event_queue import enqueue

            def prompt() -> None:
                from infernux.engine.ui.plugin_install_confirmation import PluginInstallConfirmationCoordinator

                if self.pending_install is not None:
                    PluginInstallConfirmationCoordinator.instance().request_restart(self)

            enqueue(prompt)

    @contextmanager
    def _python_change_scope(self, names: Iterable[str]):
        from importlib._bootstrap import _ModuleLockManager

        executable = self._project_python_executable()
        if not (same_path(executable, sys.executable) or same_path(os.path.dirname(executable), sys.prefix)):
            yield ()
            return
        distributions = {name: importlib.metadata.distribution(name) for name in sorted(set(names))}
        paths: dict[str, set[str]] = {}
        modules: set[str] = set()
        for name, distribution in distributions.items():
            files = distribution.files or ()
            paths[name] = {path_key(distribution.locate_file(file)) for file in files}
            for file in files:
                parts = str(file).replace("\\", "/").split("/")
                filename = parts[-1]
                for suffix in [".py", ".pyc", *importlib.machinery.EXTENSION_SUFFIXES]:
                    if filename.endswith(suffix):
                        parts[-1] = filename[:-len(suffix)]
                        if parts[-1] == "__init__":
                            parts.pop()
                        if parts and all(part.isidentifier() for part in parts):
                            modules.update(".".join(parts[:i]) for i in range(1, len(parts) + 1))
                        break
        # Serialize with CPython's actual loaders, including imports in flight.
        with ExitStack() as locks:
            for name in sorted(modules):
                locks.enter_context(_ModuleLockManager(name))
            loaded_paths = set()
            for module in tuple(sys.modules.values()):
                if isinstance(module, ModuleType):
                    filename = object.__getattribute__(module, "__dict__").get("__file__")
                    if filename:
                        loaded_paths.add(path_key(filename))
            yield tuple(name for name, files in paths.items() if files & loaded_paths)

    def _run_python_change(self, command: list[str], names: Iterable[str]):
        with self._python_change_scope(names) as loaded:
            if loaded:
                raise _DependencyInUse(loaded)
            return self._run_process(command, cwd=self.project_root)

    def _python_dependencies_in_use(self, names: Iterable[str]) -> tuple[str, ...]:
        with self._python_change_scope(names) as loaded:
            return loaded

    def cancel_pending_install(self) -> None:
        from infernux_editor import pending_path

        pending_path(self.project_root).unlink()
        self.pending_install = None

    def _defer_package_python(
        self, preview: InxPackagePreview, current: Mapping[str, object] | None,
        *, selected: Iterable[str] | None, source: Mapping[str, object] | None,
        update: bool, overwrite_modified: bool,
    ) -> bool:
        python_lines: list[str] = []
        visiting: set[str] = set()
        incoming: set[str] = set()

        def collect(package: InxPackagePreview) -> None:
            reference = str(package.metadata["reference"])
            if reference in visiting:
                raise RuntimeError(f"Circular plugin dependency: {reference}")
            visiting.add(reference)
            incoming.add(reference)
            plan = self._plan_requirements(package)
            python_lines.extend(plan.python_lines)
            for dependency in plan.plugins:
                installed = self.registry.installed_record(dependency.reference)
                if dependency.archive is not None:
                    if installed is None or not _same_install(installed, dependency.preview):
                        with self._package_cache().workspace("nested-plan") as workspace:
                            path = Path(workspace) / "dependency.inxpkg"
                            path.write_bytes(dependency.archive)
                            collect(InxPackage.inspect(str(path)))
                elif installed is None:
                    record = self.registry.find(dependency.reference)
                    if record is None:
                        raise KeyError(f"Plugin reference was not found: {dependency.reference}")
                    acquired = self.download_source(record["source"])
                    child = InxPackage.inspect(str(acquired["path"]))
                    _require_package_identity(
                        child.metadata, expected_reference=dependency.reference,
                        expected_version=str(record.get("version", "")),
                    )
                    collect(child)
            visiting.remove(reference)

        collect(preview)
        keep = {item["name"] for item in _pip_requirement_targets(self._plan_requirements(preview).python_lines)}
        restore, uninstall = self._python_dependency_release(str(preview.metadata["reference"]), keep_names=keep) if current else ([], [])
        python_lines.extend(restore)
        if not uninstall and not any(line.strip() and not line.lstrip().startswith("#") for line in python_lines):
            return False
        before = self._python_environment_snapshot()
        install: list[dict] = []
        requested = tuple(item["requirement"] for item in _pip_requirement_targets(python_lines))
        if python_lines and (not requested or not _requirements_satisfied(requested, before)):
            with self._pip_requirement_file(python_lines) as path:
                install = self._resolve_python_install(["-r", path], replacing=incoming)
        if not install and not uninstall:
            return False
        changed = ({canonicalize_name(item["metadata"]["name"]) for item in install} & before.keys()) | set(uninstall)
        loaded = self._python_dependencies_in_use(changed)
        if not loaded:
            return False
        reference = str(preview.metadata["reference"])
        path, _ = self._cache_package(preview.package_path, reference, str(preview.metadata["version"]))
        self._stage_python_install({
            "request": {"kind": "package", "path": path, "options": {
                "selected": list(selected) if selected is not None else None,
                "source": dict(source or (current.get("source", {}) if current else {})),
                "update": update, "overwrite_modified": overwrite_modified,
                "expected_reference": reference, "expected_version": str(preview.metadata["version"]),
            }},
            "before": before, "install": install, "uninstall": uninstall, "loaded": loaded,
        })
        return True

    def _complete_pending_install(self) -> None:
        from infernux_editor import read_pending

        operation = read_pending(self.project_root)
        if not self.installation_only or operation is None or not operation["prepared"]:
            raise RuntimeError("Dependency publication requires a prepared installation.")
        self._prepared_python_baseline = operation["before"]
        request = operation["request"]
        if request["kind"] == "pip":
            after = self._python_environment_snapshot()
            requirements = _pip_requirement_targets(request["arguments"])
            if not _requirements_satisfied(tuple(item["requirement"] for item in requirements), after):
                raise RuntimeError("Prepared Python requirements remain unresolved")
            self.registry.record_python_install(
                syntax=request["syntax"], command=operation["command"],
                output=operation.get("output", ""), requirements=request["arguments"],
                dependency_requirements=requirements,
                changes=_python_environment_changes(operation["before"], after),
                owner="@project", python_environment=after,
            )
        elif request["kind"] == "package":
            self.install_package(request["path"], **request["options"])
        elif request["kind"] == "uninstall":
            self.uninstall(request["reference"])
        elif request["kind"] == "startup":
            after = self._python_environment_snapshot()
            if not _requirements_satisfied(request["requirements"], after, check_sources=True):
                raise RuntimeError("Prepared startup requirements remain unresolved")
            self.registry.record_python_reconciliation(
                requirements=request["requirements"],
                changes=_python_environment_changes(operation["before"], after),
                python_environment=after, owners=request["owners"],
            )
        else:
            raise ValueError(f"Unknown pending installation: {request['kind']}")

    def finalize_background_install(self, reference: str) -> PluginState:
        """Publish worker-written plugin assets and preload them on the editor thread."""

        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError("Plugin installation must be finalized on the editor thread")
        normalized = validate_reference(reference)
        self._refresh_editor_assets()
        self.preloads.catch_up()
        self._deferred_catalog_changes.clear()
        self._rebuild_states()
        if self.states[normalized.casefold()].loaded:
            self._publish_package_runtime_scripts(normalized)
        return self.states[normalized.casefold()]

    def _startup_python_requirements(self) -> dict[str, tuple[str, ...]]:
        requirements = {
            str(record["reference"]): tuple(str(item["requirement"]) for item in record.get("python_requirements", ()))
            for record in self.registry.installed() if record.get("enabled", True)
        }
        project = tuple(dict.fromkeys(
            str(requirement)
            for dependency in self.registry.load()["python_dependencies"]
            for owner in dependency["owners"] if str(owner["reference"]).startswith("@")
            for requirement in owner["requirements"]
        ))
        from infernux.engine.project_requirements import _parse_requirements, requirements_path

        path = requirements_path(self.project_root)
        authored = tuple(spec for spec, _ in _parse_requirements(path)) if os.path.isfile(path) else ()
        authored_names = {item["name"] for item in _pip_requirement_targets(authored)}
        requirements["@project"] = authored + tuple(
            spec for spec in project if canonicalize_name(Requirement(spec).name) not in authored_names
        )
        return {reference: values for reference, values in requirements.items() if values}

    def _plan_startup_python(self) -> dict | None:
        owners = self._startup_python_requirements()
        requirements = list(dict.fromkeys(spec for values in owners.values() for spec in values))
        if not requirements:
            return None
        targets = _pip_requirement_targets(requirements)
        for dependency in self.registry.load()["python_dependencies"]:
            name, version = dependency["name"], dependency["installed_version"]
            constraints = [item["requirement"] for item in targets if item["name"] == name]
            if (version and constraints and not any(Requirement(spec).url for spec in constraints)
                    and _requirements_satisfied(constraints, {name: version})):
                requirements.append(f"{name}=={version}")
        before = self._python_environment_snapshot()
        if _requirements_satisfied(requirements, before, check_sources=True):
            return None
        with self._pip_requirement_file(requirements) as path:
            install = self._resolve_python_install(["-r", path], replacing=owners)
        return {
            "request": {"kind": "startup", "requirements": requirements,
                        "owners": {key: _pip_requirement_targets(values) for key, values in owners.items()}},
            "before": before, "install": install,
        }

    def _reconcile_python_requirements_for_startup(self) -> tuple[str, ...]:
        """Launch preparation owns installation; the live host only checks its result."""
        requirements = [spec for values in self._startup_python_requirements().values() for spec in values]
        before = self._python_environment_snapshot() if requirements else {}
        missing = [spec for spec in requirements if not _requirements_satisfied((spec,), before, check_sources=True)]
        self.python_requirement_error = (
            "Open this project from Hub or 'python -m infernux_editor <project>' to prepare: " + ", ".join(missing)
            if missing else ""
        )
        if missing:
            raise RuntimeError(self.python_requirement_error)
        return ()

    def _install_pip_lines(
        self, lines: Iterable[str]
    ) -> _PipInstallEffect:
        values = tuple(str(line) for line in lines)
        executable = self._project_python_executable()
        before = self._python_environment_snapshot(executable)
        requirements = _pip_requirement_targets(values)
        requested = tuple(item["requirement"] for item in requirements)
        if self.installation_only:
            if not _requirements_satisfied(requested, before):
                raise RuntimeError("Prepared plugin Python requirements remain unresolved")
            return _PipInstallEffect(
                self._prepared_python_baseline, before, requirements, (),
                "Dependencies prepared before editor startup.",
            )
        if requested and _requirements_satisfied(requested, before):
            return _PipInstallEffect(
                before,
                dict(before),
                requirements,
                (),
                "Requirements already satisfied by the project Python environment.",
            )
        try:
            with self._pip_requirement_file(values) as filtered:
                install = self._resolve_python_install(["-r", filtered])
            from infernux_editor import resolved_arguments

            command = (executable, "-m", "pip", "install", "--no-deps", "--force-reinstall", *resolved_arguments(install))
            changed = {canonicalize_name(item["metadata"]["name"]) for item in install} & before.keys()
            result = self._run_python_change(list(command), changed)
            after = self._python_environment_snapshot(executable)
            if not _requirements_satisfied(requested, after):
                raise RuntimeError("pip completed but requested Python requirements remain unresolved")
            self._activate_installed_python_paths(
                before, after, executable=executable,
                reinstalled=(canonicalize_name(item["metadata"]["name"]) for item in install),
            )
            return _PipInstallEffect(
                before,
                after,
                requirements,
                command,
                result.stdout,
            )
        except _DependencyInUse:
            raise
        except BaseException as install_error:
            try:
                self._restore_python_environment(before, executable=executable)
            except Exception as rollback_error:
                raise RuntimeError(
                    "Plugin requirement installation failed and Python environment "
                    f"rollback was incomplete: {rollback_error}"
                ) from install_error
            raise

    def _activate_installed_python_paths(
        self,
        before: Mapping[str, str],
        after: Mapping[str, str],
        *,
        executable: str,
        reinstalled: Iterable[str] = (),
    ) -> None:
        """Publish newly installed site hooks before importing plugin lifecycles.

        pip runs in a child interpreter. Its successful exit does not process
        new .pth files in this Editor (notably pywin32's import paths and DLL
        bootstrap). Use CPython's site processing for the changed distributions,
        not hard-coded dependency paths or a second pass over unrelated hooks.
        An installation into a different interpreter must not affect this one.
        """
        if not (same_path(executable, sys.executable) or same_path(os.path.dirname(executable), sys.prefix)):
            return
        changed = sorted(set(reinstalled) | {
            name for name, version in after.items() if before.get(name) != version
        })
        if not changed:
            return

        def publish() -> None:
            import importlib
            import importlib.metadata
            import site

            importlib.invalidate_caches()
            hooks: set[str] = set()
            for name in changed:
                distribution = importlib.metadata.distribution(name)
                root = resolved_path(distribution.locate_file(""))
                for entry in distribution.files or ():
                    path = resolved_path(distribution.locate_file(entry))
                    if path.endswith(".pth") and same_path(os.path.dirname(path), root):
                        hooks.add(path)
            for path in sorted(hooks, key=path_key):
                # addpackage is the CPython 3.13 site primitive used by
                # addsitedir; unlike addsitedir it processes only this hook.
                site.addpackage(os.path.dirname(path), os.path.basename(path), None)
            importlib.invalidate_caches()

        if self.engine is not None and threading.current_thread() is not threading.main_thread():
            from infernux.host.commands import MainThreadCommandQueue

            MainThreadCommandQueue.instance().run_sync("plugin.python-site-publish", publish)
        else:
            publish()

    def _python_environment_snapshot(
        self, executable: str | None = None
    ) -> dict[str, str]:
        python = executable or self._project_python_executable()
        # When the project environment is the interpreter we are already
        # running in, enumerate installed distributions in-process instead of
        # spawning `python -m pip list` (which costs seconds on every editor
        # startup, under the "Preloading project plugins…" splash step).
        same_interpreter = same_path(python, sys.executable) or same_path(os.path.dirname(python), sys.prefix)
        if same_interpreter:
            import importlib.metadata

            snapshot: dict[str, str] = {}
            for distribution in importlib.metadata.distributions():
                try:
                    name = canonicalize_name(str(distribution.metadata["Name"] or ""))
                    version = str(distribution.version or "")
                except (OSError, ValueError) as exc:
                    Debug.log_suppressed("PluginManager.distribution_metadata", exc)
                    continue
                if name and version and name not in snapshot:
                    snapshot[name] = version
            return snapshot
        result = self._run_process(
            [python, "-m", "pip", "list", "--format=json"],
            cwd=self.project_root,
        )
        try:
            values = json.loads(result.stdout or "[]")
        except json.JSONDecodeError as exc:
            raise RuntimeError("pip list returned invalid JSON") from exc
        if not isinstance(values, list):
            raise RuntimeError("pip list returned a non-list document")
        return {
            canonicalize_name(str(item.get("name", ""))): str(
                item.get("version", "")
            )
            for item in values
            if isinstance(item, Mapping)
            and str(item.get("name", "")).strip()
            and str(item.get("version", "")).strip()
        }

    def _rollback_pip_effect(self, effect: _PipInstallEffect | None) -> None:
        if effect is None or effect.rolled_back:
            return
        effect.rolled_back = True
        self._restore_python_environment(effect.before)

    def _restore_python_environment(
        self,
        baseline: Mapping[str, str],
        *,
        executable: str | None = None,
    ) -> None:
        python = executable or self._project_python_executable()
        current = self._python_environment_snapshot(python)
        added = sorted(set(current) - set(baseline))
        if added:
            self._run_process(
                [python, "-m", "pip", "uninstall", "-y", *added],
                cwd=self.project_root,
            )
        restore = [
            f"{name}=={version}"
            for name, version in sorted(baseline.items())
            if current.get(name) != version
        ]
        if restore:
            self._run_process(
                [python, "-m", "pip", "install", *restore],
                cwd=self.project_root,
            )

    def _python_dependency_release(
        self, reference: str, *, keep_names: set[str] | None = None,
    ) -> tuple[list[str], list[str]]:
        plan = self.registry.python_release_plan(reference)
        actionable = [
            item
            for item in plan
            if (item.get("remaining_owners") or bool(item.get("managed", False)))
            and str(item.get("name", "")) not in (keep_names or set())
        ]
        if not actionable:
            return [], []
        remaining_requirements: list[str] = []
        uninstall: list[str] = []
        restore: list[str] = []
        for item in actionable:
            owners = item.get("remaining_owners", [])
            if owners:
                for owner in owners:
                    if isinstance(owner, Mapping):
                        remaining_requirements.extend(
                            str(value)
                            for value in owner.get("requirements", [])
                            if str(value).strip()
                        )
                continue
            if not bool(item.get("managed", False)):
                continue
            name = canonicalize_name(str(item.get("name", "")))
            previous = str(item.get("baseline_version", "")).strip()
            if previous:
                restore.append(f"{name}=={previous}")
            elif name:
                uninstall.append(name)
        return [*remaining_requirements, *restore], sorted(set(uninstall))

    def _release_python_dependencies(
        self, reference: str, *, keep_names: set[str] | None = None,
    ) -> dict[str, str] | None:
        requirements, uninstall = self._python_dependency_release(reference, keep_names=keep_names)
        if not requirements and not uninstall:
            return None
        if self.installation_only:
            return self._prepared_python_baseline
        from infernux_editor import resolved_arguments

        executable = self._project_python_executable()
        baseline = self._python_environment_snapshot(executable)
        install = self._resolve_python_install(requirements, replacing=(reference,)) if requirements and not _requirements_satisfied(requirements, baseline) else []
        uninstall = [name for name in uninstall if name in baseline]
        changed = ({canonicalize_name(item["metadata"]["name"]) for item in install} & baseline.keys()) | set(uninstall)
        with self._python_change_scope(changed) as loaded:
            if loaded:
                raise _DependencyInUse(loaded)
            try:
                if install:
                    self._run_process([
                        executable, "-m", "pip", "install", "--no-deps", "--force-reinstall", *resolved_arguments(install),
                    ], cwd=self.project_root)
                if uninstall:
                    self._run_process([executable, "-m", "pip", "uninstall", "-y", *uninstall], cwd=self.project_root)
            except BaseException:
                self._restore_python_environment(baseline, executable=executable)
                raise
        return baseline

    def _defer_uninstall_python(self, reference: str) -> bool:
        requirements, uninstall = self._python_dependency_release(reference)
        if not requirements and not uninstall:
            return False
        before = self._python_environment_snapshot()
        install = self._resolve_python_install(requirements, replacing=(reference,)) if requirements and not _requirements_satisfied(requirements, before) else []
        uninstall = [name for name in uninstall if name in before]
        changed = ({canonicalize_name(item["metadata"]["name"]) for item in install} & before.keys()) | set(uninstall)
        loaded = self._python_dependencies_in_use(changed)
        if not loaded:
            return False
        self._stage_python_install({
            "request": {"kind": "uninstall", "reference": reference},
            "before": before, "install": install, "uninstall": uninstall, "loaded": loaded,
        })
        return True

    @contextmanager
    def _pip_requirement_file(self, requirements: Iterable[str]):
        with self._package_cache().workspace("requirements") as workspace:
            path = Path(workspace) / "requirements.txt"
            path.write_text(
                "".join(str(value).rstrip("\r\n") + "\n" for value in requirements),
                encoding="utf-8",
            )
            yield str(path)

    def set_enabled(self, reference: str, enabled: bool) -> PluginState:
        reference = validate_reference(reference)
        record = self.registry.installed_record(reference)
        if record is None:
            raise KeyError(f"Plugin is not installed: {reference}")
        requested = bool(enabled)
        if bool(record.get("enabled", True)) == requested:
            return self.states.get(reference.casefold()) or self.reload(reference)
        installed = self.registry.installed()
        reference_key = reference.casefold()
        if not requested:
            blockers = [
                str(item.get("reference", ""))
                for item in installed
                if bool(item.get("enabled", True))
                and any(
                    str(dependency).casefold() == reference_key
                    for dependency in item.get("dependencies", [])
                )
            ]
            if blockers:
                raise RuntimeError(
                    f"Plugin {reference} is required by enabled plugins: "
                    + ", ".join(sorted(blockers))
                )
            failures = self.preloads.unload_package(reference)
            if failures:
                raise RuntimeError(
                    "Plugin lifecycle could not be stopped; restart required: "
                    + "; ".join(state.error for state in failures)
                )
            try:
                self.registry.set_enabled(reference, False)
                self._retire_package_runtime_scripts(record)
            except BaseException:
                self.registry.set_enabled(reference, True)
                self.reload(reference)
                raise
            self._rebuild_states()
            return self.states[reference_key]
        self._validate_dependency_plan(
            reference, record.get("dependencies", ()), enabled=True, require_installed=True,
        )
        self.registry.set_enabled(reference, True)
        return self.reload(reference)

    def uninstall(self, reference: str) -> dict[str, object]:
        if self.pending_install is not None and not self.installation_only:
            raise RuntimeError("Complete or cancel the pending dependency installation first.")
        reference = validate_reference(reference)
        record = self.registry.installed_record(reference)
        if record is None:
            raise KeyError(f"Plugin is not installed: {reference}")
        reference_key = reference.casefold()
        blockers = [
            str(item.get("reference", ""))
            for item in self.registry.installed()
            if any(
                str(dependency).casefold() == reference_key
                for dependency in item.get("dependencies", [])
            )
        ]
        if blockers:
            raise RuntimeError(
                f"Plugin {reference} is required by: {', '.join(sorted(blockers))}"
            )
        if not self.installation_only and self._defer_uninstall_python(reference):
            return {"reference": reference, "restart_required": True}
        failures = self.preloads.unload_package(reference) if not self.installation_only else ()
        if failures:
            raise RuntimeError(
                "Plugin lifecycle could not be stopped; uninstall aborted and restart required: "
                + "; ".join(state.error for state in failures)
            )
        python_baseline: dict[str, str] | None = None
        try:
            python_baseline = self._release_python_dependencies(reference)
        except BaseException as error:
            self.preloads.reload_package(reference)
            self._rebuild_states()
            if isinstance(error, _DependencyInUse) and self._defer_uninstall_python(reference):
                return {"reference": reference, "restart_required": True}
            raise
        removed: list[str] = []
        preserved: list[str] = []
        guid_index = self._guid_index()
        transaction = _InstallTransaction(self.project_root)
        registry_before = self.registry.load()
        registry_changed = False
        removed_caches: list[str] = []
        try:
            for item in [*record.get("files", []), record.get("control")]:
                if not isinstance(item, Mapping) or not bool(item.get("owned", True)):
                    continue
                guid = str(item.get("guid", "")).casefold()
                if self.registry.users_for_guid(guid, excluding=reference):
                    preserved.append(guid_index.get(guid, str(item.get("path_hint", ""))))
                    continue
                path = guid_index.get(guid)
                if not path:
                    continue
                for cache in _script_bytecode_paths(path):
                    transaction.remove(cache)
                    removed_caches.append(cache)
                transaction.remove(path)
                transaction.remove(path + ".meta")
                removed.append(path)
            registry_changed = True
            self.registry.remove_install(reference)
            if not self.installation_only:
                self._retire_package_runtime_scripts(record)
            transaction.commit()
        except BaseException as uninstall_error:
            transaction.rollback()
            if registry_changed:
                self.registry.restore(registry_before)
            python_rollback_error = None
            if python_baseline is not None and not self.installation_only:
                try:
                    self._restore_python_environment(python_baseline)
                except Exception as python_error:
                    python_rollback_error = python_error
            if not self.installation_only:
                try:
                    self.reload(reference)
                except Exception as reload_error:
                    Debug.log_suppressed("PluginManager.uninstall.rollback_reload", reload_error)
            if python_rollback_error is not None:
                raise RuntimeError(
                    "Plugin uninstall failed and Python environment rollback "
                    f"was incomplete: {python_rollback_error}"
                ) from uninstall_error
            raise
        self._prune_package_directories([*removed, *removed_caches])
        if self.installation_only:
            return {**record, "removed_files": removed, "preserved_shared_files": preserved}
        from infernux.core.assets import AssetManager

        for path in removed:
            if path.endswith(".py") and is_path_within(
                path, os.path.join(self.project_root, "Packages"), allow_root=False
            ):
                # Lifecycle/runtime retirement was committed above. A delayed
                # delete echo must not restart a replacement package at this
                # reference (including a legacy Editor -> editor migration).
                AssetManager._suppress_watcher_echo("deleted", path)
        self.preloads.forget_package(reference)
        self._rebuild_states()
        self._refresh_editor_assets()
        return {**record, "removed_files": removed, "preserved_shared_files": preserved}

    def _materialize_source(
        self,
        source: Mapping[str, object],
        workspace: str,
        *,
        progress: _InstallProgress | None = None,
        release_tag: str = "",
    ) -> tuple[str, dict[str, object]]:
        descriptor = dict(source)
        source_type = str(descriptor["type"])
        location = str(descriptor["location"])
        if release_tag and source_type != "github":
            raise ValueError("A release version can only be selected for a GitHub plugin source")
        if source_type == "local":
            _report_progress(progress, "read_local_source", 0.16)
            if descriptor.get("cache_scope") == "hub" and location == descriptor.get("cache_location"):
                location = SharedPackageCache.validate_location(location)
                local = resolved_path(os.path.join(self._package_cache().root, *location.split("/")))
                if not os.path.isfile(local):
                    raise FileNotFoundError(f"Install the pinned local package archive first: {location}")
                return self._materialize_local(local, descriptor, workspace, progress)
            local = resolved_path(
                location
                if os.path.isabs(location)
                else os.path.join(self.project_root, location)
            )
            return self._materialize_local(local, descriptor, workspace, progress)
        if source_type == "url":
            from ._download import download_file

            target = os.path.join(workspace, "download.inxpkg")
            _report_progress(progress, "download_package", 0.08)
            try:
                download_file(location, target, progress=progress)
            except OSError:
                repository = str(descriptor.get("repository", "")).strip()
                reference = str(descriptor.get("reference", "")).strip()
                fallback_tag = str(descriptor.get("release_tag", "")).strip()
                if not descriptor.get("official") or not all(
                    (repository, reference, fallback_tag)
                ):
                    raise
                from .github_releases import resolve_github_release

                _report_progress(progress, "resolve_releases", 0.08)
                released = resolve_github_release(
                    repository,
                    workspace,
                    expected_reference=reference,
                    progress=progress,
                    release_tag=fallback_tag,
                )
                assert released is not None
                fallback_source = dict(descriptor)
                fallback_source.update(
                    {
                        key: value
                        for key, value in released.source.items()
                        if key not in {"type", "location"}
                    }
                )
                fallback_source["acquisition"] = "github-release"
                return released.path, fallback_source
            return target, descriptor
        revision = str(descriptor.get("commit") or descriptor.get("revision", "")).strip()
        if source_type == "github":
            from .github_releases import (
                download_github_source,
                resolve_github_release,
            )
        if source_type == "github" and not descriptor.get("source_snapshot"):
            _report_progress(progress, "resolve_releases", 0.06)
            released = resolve_github_release(
                location,
                workspace,
                expected_reference=str(descriptor.get("reference", "")),
                progress=progress,
                release_tag=release_tag or str(descriptor.get("release_tag", "")),
            )
            if released is not None:
                released_source = dict(descriptor)
                released_source.update(
                    {
                        key: value
                        for key, value in released.source.items()
                        if key not in {"type", "location"}
                    }
                )
                released_source["acquisition"] = "github-release"
                return released.path, released_source

        subdirectory = portable_path(
            str(descriptor.get("subdirectory", ""))
        ).strip("/")
        if source_type == "github":
            _report_progress(progress, "download_source", 0.08)
            snapshot = download_github_source(
                location,
                workspace,
                revision=revision,
                subdirectory=subdirectory,
                progress=progress,
            )
            descriptor["commit"] = snapshot.commit
            descriptor["source_snapshot"] = True
            return self._materialize_local(
                snapshot.root,
                descriptor,
                workspace,
                progress,
            )

        checkout = os.path.join(workspace, "checkout")
        command = ["git", "clone", "--depth", "1", "--progress"]
        if subdirectory:
            command.extend(["--filter=blob:none", "--sparse"])
        if revision and not descriptor.get("commit"):
            command.extend(["--branch", revision])
        command.extend([location, checkout])
        _report_progress(progress, "clone_repository", 0.08)
        self._run_process_with_progress(
            command,
            stage="clone_repository",
            progress=progress,
            start=0.08,
            end=0.24,
        )
        if descriptor.get("commit"):
            self._run_process(["git", "fetch", "--depth", "1", "origin", revision], cwd=checkout)
            self._run_process(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=checkout)
        if subdirectory:
            self._run_process(
                ["git", "sparse-checkout", "set", subdirectory],
                cwd=checkout,
            )
        commit = self._run_process(
            ["git", "rev-parse", "HEAD"], cwd=checkout
        ).stdout.strip()
        if len(commit) != 40 or any(
            character not in "0123456789abcdefABCDEF" for character in commit
        ):
            raise RuntimeError("Git plugin source did not resolve to a commit SHA")
        if descriptor.get("commit") and commit.casefold() != revision.casefold():
            raise RuntimeError("Git plugin checkout does not match its pinned commit")
        descriptor["commit"] = commit.casefold()
        descriptor["source_snapshot"] = True
        _report_progress(progress, "read_repository", 0.28)
        root = (
            resolved_path(os.path.join(checkout, *subdirectory.split("/")))
            if subdirectory
            else checkout
        )
        if not is_path_within(root, checkout, allow_root=True):
            raise ValueError("Git plugin subdirectory escapes the checkout")
        if not os.path.isdir(root):
            source_revision = revision or "the repository default branch"
            raise RuntimeError(
                f"Git plugin source {location} at {source_revision} does not contain "
                f"the package directory {subdirectory}"
            )
        return self._materialize_local(root, descriptor, workspace, progress)

    def _materialize_local(
        self,
        path: str,
        source: Mapping[str, object],
        workspace: str,
        progress: _InstallProgress | None,
    ) -> tuple[str, dict[str, object]]:
        if os.path.isfile(path):
            if not path.casefold().endswith(PACKAGE_EXTENSION):
                raise ValueError("A local plugin file must be an .inxpkg")
            return path, dict(source)
        if not os.path.isdir(path):
            raise FileNotFoundError(path)
        configured = portable_path(str(source.get("package", ""))).strip("/")
        if configured:
            package_path = resolved_path(os.path.join(path, *configured.split("/")))
            if not is_path_within(package_path, path, allow_root=False):
                raise ValueError("Plugin package path escapes the source root")
            if not os.path.isfile(package_path):
                raise FileNotFoundError(package_path)
            return package_path, dict(source)
        repository_root = os.path.join(path, REPOSITORY_PACKAGE_DIRECTORY)
        repository_manifest = os.path.isfile(os.path.join(repository_root, SOURCE_MANIFEST))
        if bool(source.get("source_snapshot")) and not repository_manifest:
            raise ValueError(
                "Repository plugin source must put distributable files under "
                "package/inx_package.json"
            )
        source_name = Path(path).name
        if bool(source.get("source_snapshot")):
            location_path = urlsplit(str(source.get("location", ""))).path.rstrip("/")
            source_name = Path(location_path).stem or source_name
        package = os.path.join(workspace, f"{source_name}.inxpkg")
        _report_progress(progress, "build_source_package", 0.28)
        InxPackage.export_source(path, package)
        return package, dict(source)

    def _validate_dependency_plan(
        self, reference: str, dependencies: Iterable[str], *, enabled: bool,
        require_installed: bool = False,
    ) -> None:
        """Validate the candidate graph, including outer in-flight installs."""
        installed = {
            str(item["reference"]).casefold(): item for item in self.registry.installed()
        }
        proposed = tuple(str(item).casefold() for item in dependencies)
        graph = {
            key: tuple(str(item).casefold() for item in record.get("dependencies", ()))
            for key, record in installed.items()
        }
        graph.update({key: tuple(item.casefold() for item in values)
                      for key, values in self._dependency_plans.items()})
        graph[reference.casefold()] = proposed
        visiting: list[str] = []
        complete: set[str] = set()

        def visit(key: str) -> None:
            if key in complete:
                return
            if key in visiting:
                raise RuntimeError("Circular plugin dependency: " + " -> ".join(visiting + [key]))
            visiting.append(key)
            for dependency in graph.get(key, ()):
                visit(dependency)
            visiting.pop()
            complete.add(key)

        for key in graph:
            visit(key)
        unavailable = [
            key for key in proposed
            if (key not in installed and require_installed)
            or (enabled and key in installed and not bool(installed[key].get("enabled", True)))
        ]
        if unavailable:
            raise RuntimeError(
                f"Plugin {reference} requires enabled plugins: " + ", ".join(sorted(unavailable))
            )

    def _plan_requirements(self, preview: InxPackagePreview) -> _PackageRequirements:
        requirement_name = "requirements.txt"
        requirement_record = next(
            (
                item
                for item in preview.file_records
                if str(item.get("logical_path", "")) == requirement_name
            ),
            None,
        )
        if requirement_record is None:
            return _PackageRequirements()
        text = read_entry(
            preview.package_path, str(requirement_record["archive_path"])
        ).decode("utf-8")
        plugins: list[_PluginRequirement] = []
        pip_lines: list[str] = []
        for line in text.splitlines(keepends=True):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                pip_lines.append(line)
                continue
            nested = self._nested_requirement(preview, requirement_name, stripped)
            if nested is not None:
                with self._package_cache().workspace("nested") as workspace:
                    path = os.path.join(workspace, "nested.inxpkg")
                    Path(path).write_bytes(nested)
                    child = InxPackage.inspect(path)
                    reference = _require_package_identity(child.metadata)
                    plugins.append(_PluginRequirement(reference, nested, child))
                continue
            reference = self._registry_reference_for_requirement(stripped)
            own_reference = str(preview.metadata["reference"])
            if stripped.removeprefix("inx:").strip().casefold() == own_reference.casefold():
                reference = own_reference
            if reference is not None:
                plugins.append(_PluginRequirement(reference))
            else:
                pip_lines.append(line)
        return _PackageRequirements(tuple(plugins), tuple(pip_lines))

    def _install_requirements(
        self, preview: InxPackagePreview, *, progress: _InstallProgress | None = None,
        plan: _PackageRequirements | None = None,
        before_python_install: Callable[[], None] | None = None,
    ) -> tuple[tuple[str, ...], _PipInstallEffect | None]:
        plan = plan if plan is not None else self._plan_requirements(preview)
        for item in plan.plugins:
            installed = self.registry.installed_record(item.reference)
            if item.archive is not None:
                if installed is not None and _same_install(installed, item.preview):
                    continue
                with self._package_cache().workspace("nested") as workspace:
                    path = os.path.join(workspace, "nested.inxpkg")
                    Path(path).write_bytes(item.archive)
                    self.install_package(path, progress=progress, expected_reference=item.reference)
            elif installed is None:
                self.install_reference(item.reference, progress=progress)
        if before_python_install is not None:
            before_python_install()
        effect = None
        if any(line.strip() and not line.lstrip().startswith("#") for line in plan.python_lines):
            _report_progress(progress, "install_python_dependencies", 0.58)
            effect = self._install_pip_lines(plan.python_lines)
        return plan.references, effect

    def _nested_requirement(
        self, preview: InxPackagePreview, requirement_name: str, requirement: str
    ) -> bytes | None:
        if requirement.startswith(("-e ", "--editable ")):
            return None
        raw = portable_path(requirement)
        if not raw.casefold().endswith(PACKAGE_EXTENSION):
            return None
        base = posixpath.dirname(requirement_name)
        logical = posixpath.normpath(posixpath.join(base, raw)).strip("/")
        if logical == ".." or logical.startswith("../"):
            raise ValueError(f"Nested InxPackage requirement escapes package: {requirement}")
        record = next(
            (
                item
                for item in preview.file_records
                if str(item.get("logical_path", "")) == logical
            ),
            None,
        )
        if record is None:
            raise FileNotFoundError(logical)
        return read_entry(preview.package_path, str(record["archive_path"]))

    def _registry_reference_for_requirement(self, requirement: str) -> str | None:
        candidate = requirement.removeprefix("inx:").strip()
        if candidate.casefold() in self._dependency_plans:
            return validate_reference(candidate)
        try:
            direct = self.registry.find(candidate)
        except ValueError:
            direct = None
        if direct is not None:
            return str(direct["reference"])
        lowered = candidate.casefold()
        if (
            candidate.startswith(("-", ".", "/", "\\"))
            or " @ " in candidate
            or lowered.startswith(("git+", "http://", "https://", "file:"))
            or lowered.endswith((".whl", ".tar.gz", ".zip", PACKAGE_EXTENSION))
            or (len(candidate) >= 2 and candidate[1] == ":")
        ):
            return None
        try:
            parsed = Requirement(candidate)
            if parsed.url or (parsed.marker is not None and not parsed.marker.evaluate()):
                return None
            name = parsed.name
        except Exception:
            parsed = None
            name = candidate.split(";", 1)[0].split("@", 1)[0].strip()
            for marker in ("===", "==", ">=", "<=", "~=", "!=", ">", "<", "["):
                name = name.split(marker, 1)[0].strip()
        matched = self.registry.installed_record(name) or self.registry.find(name)
        if matched is not None and parsed is not None and parsed.specifier:
            version = str(matched.get("version", "")).strip()
            if version and not parsed.specifier.contains(version, prereleases=True):
                raise RuntimeError(
                    f"Official InxPackage {matched['reference']} {version} "
                    f"does not satisfy {parsed.specifier}"
                )
        return str(matched["reference"]) if matched is not None else None

    def _plan_update(
        self, current: Mapping[str, object], preview: InxPackagePreview, *,
        previous_path: str,
        overwrite_modified: bool,
    ) -> tuple[list[_PlannedFile], dict[str, object], list[str]]:
        """Compare only at the update boundary; reuse the normal GUID routing."""
        reference = str(current["reference"])
        previous = InxPackage.inspect(previous_path)
        old_files = {str(item["guid"]).casefold(): item for item in current["files"]}
        old_contents = {
            str(item["guid"]).casefold(): item for item in previous.file_records
        }
        old_logical = {str(item["logical_path"]): str(item["guid"]).casefold()
                       for item in current["files"]}
        for item in preview.file_records:
            logical = str(item["logical_path"])
            if logical in old_logical and old_logical[logical] != str(item["guid"]).casefold():
                raise PackageConflictError(f"Updated asset must retain its GUID: {logical}")
        # A selective import stays selective; newly introduced members are included.
        omitted = set(old_contents) - set(old_files)
        selected = [str(item["logical_path"]) for item in preview.file_records
                    if str(item["guid"]).casefold() not in omitted]
        planned, control = self._plan_install(preview, selected)
        if str(control["guid"]).casefold() != str(current["control"]["guid"]).casefold():
            raise PackageConflictError("An update must preserve the package control GUID")
        conflicts: list[str] = []

        def check_edit(path: str, baseline: bytes | None, incoming: bytes | None) -> None:
            actual = Path(path).read_bytes() if os.path.isfile(path) else None
            if not same_source_content(path, actual, incoming) and (
                baseline is None or not same_source_content(path, actual, baseline)
            ):
                conflicts.append(portable_path(relative_path(path, self.project_root)))

        def baseline_payload(guid: str) -> bytes | None:
            record = old_contents.get(guid)
            return read_entry(previous_path, str(record["archive_path"])) if record else None

        def settings(payload: bytes) -> dict[str, object]:
            document = json.loads(payload)
            keep_readable = document["metadata"].get("resource_type", {}).get("value") == "Mesh"
            # InxResourceMeta and the default file loaders regenerate these
            # observations on import. They are not authored importer settings.
            for key in (
                "content_hash", "file_path", "last_modified", "resource_type",
                "file_type", "file_extension", "file_size",
                "line_count", "character_count", "encoding", "binary_type",
                "size_category", "language",
            ):
                document["metadata"].pop(key, None)
            if not keep_readable:
                document["metadata"].pop("is_readable", None)
            return document

        def local_meta(path: str, guid: str) -> tuple[bytes | None, bool]:
            meta_path = Path(path + ".meta")
            if not meta_path.is_file():
                return None, False
            payload = meta_path.read_bytes()
            record = old_contents.get(guid)
            baseline = read_entry(previous_path, str(record["meta_archive_path"])) if record else None
            return payload, baseline is None or settings(payload) != settings(baseline)

        replacements: list[_PlannedFile] = []
        removed: list[str] = []
        for item in planned:
            guid = item.guid.casefold()
            old = old_files.get(guid)
            if old is None:
                if not item.owned and not same_source_content(
                    item.destination, Path(item.destination).read_bytes(), item.payload,
                ):
                    raise PackageConflictError(f"Shared asset has different content: {item.destination}")
                replacements.append(item)
                continue
            shared = self.registry.users_for_guid(guid, excluding=reference)
            owned = bool(old.get("owned", True))
            if shared or not owned:
                if not os.path.isfile(item.destination) or not same_source_content(
                    item.destination, Path(item.destination).read_bytes(), item.payload,
                ):
                    raise PackageConflictError(f"Cannot replace a shared asset: {item.destination}")
                # Retain ownership without writing through another package's asset.
                replacements.append(replace(item, owned=owned,
                    meta_payload=Path(item.destination + ".meta").read_bytes()))
                continue
            check_edit(item.destination, baseline_payload(guid), item.payload)
            existing_meta, modified_meta = local_meta(item.destination, guid)
            meta_payload = current_meta_bytes(
                item.guid,
                existing=existing_meta if modified_meta else item.meta_payload,
            )
            old_destination = os.path.join(self.project_root, package_destination(
                reference, str(old["logical_path"]), project_root=self.project_root,
            ))
            if str(old["logical_path"]) != item.logical_path and same_path(item.destination, old_destination):
                # Follow publisher renames unless the user already moved this GUID.
                destination_relative = package_destination(
                    reference, item.logical_path, project_root=self.project_root,
                )
                destination = resolved_path(os.path.join(self.project_root, destination_relative))
                if os.path.exists(destination):
                    raise PackageConflictError(f"Updated destination is occupied: {destination}")
                removed.append(item.destination)
                item = replace(item, destination=destination,
                               destination_relative=portable_path(destination_relative))
            replacements.append(replace(item, owned=True, meta_payload=meta_payload))

        new_guids = {item.guid.casefold() for item in replacements}
        guid_index = self._guid_index()
        for guid, old in old_files.items():
            if guid in new_guids or not bool(old.get("owned", True)):
                continue
            if self.registry.users_for_guid(guid, excluding=reference):
                # The other package must inherit ownership when we stop using it.
                continue
            path = guid_index.get(guid)
            if path:
                check_edit(path, baseline_payload(guid), None)
                if local_meta(path, guid)[1]:
                    conflicts.append(portable_path(relative_path(path + ".meta", self.project_root)))
                removed.append(path)
        control_path = str(control["absolute_path"])
        old_control = (json.dumps(previous.metadata, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        new_control = (json.dumps(preview.metadata, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        check_edit(control_path, old_control, new_control)
        if not bool(current["control"].get("owned", True)):
            raise PackageConflictError("Cannot replace a shared package control file")
        control["owned"] = True
        if conflicts and not overwrite_modified:
            raise PackageUpdateConflict(conflicts)
        return replacements, control, removed

    def _plan_install(
        self, preview: InxPackagePreview, selected: Iterable[str] | None
    ) -> tuple[list[_PlannedFile], dict[str, object]]:
        selected_set = None if selected is None else {
            portable_path(str(item)).strip("/") for item in selected
        }
        reference = str(preview.metadata["reference"])
        guid_index = self._guid_index()
        planned: list[_PlannedFile] = []
        for record in preview.file_records:
            logical = str(record["logical_path"])
            destination_relative = package_destination(reference, logical)
            if selected_set is not None and not (
                logical in selected_set or destination_relative in selected_set
            ):
                continue
            physical_relative = package_destination(
                reference, logical, project_root=self.project_root
            )
            destination = resolved_path(
                os.path.join(self.project_root, *physical_relative.split("/"))
            )
            payload = read_entry(preview.package_path, str(record["archive_path"]))
            meta_payload = read_entry(
                preview.package_path, str(record["meta_archive_path"])
            )
            owned, actual_destination = self._preflight_file(
                destination,
                str(record["guid"]),
                guid_index,
            )
            planned.append(
                _PlannedFile(
                    logical,
                    actual_destination,
                    portable_path(relative_path(actual_destination, self.project_root)),
                    str(record["guid"]),
                    str(record["role"]),
                    payload,
                    meta_payload,
                    owned,
                )
            )
        control_root = package_control_root(self.project_root, reference)
        control_path = os.path.join(control_root, PACKAGE_MANIFEST)
        control_guid = str(preview.metadata["control_guid"])
        manifest_payload = (
            json.dumps(preview.metadata, ensure_ascii=False, indent=2) + "\n"
        ).encode("utf-8")
        owned, actual_control = self._preflight_file(
            control_path,
            control_guid,
            guid_index,
        )
        control = {
            "guid": control_guid,
            "owned": owned,
            "absolute_path": actual_control,
            "path_hint": portable_path(relative_path(actual_control, self.project_root)),
        }
        return planned, control

    def _preflight_file(
        self,
        destination: str,
        guid: str,
        guid_index: Mapping[str, str],
    ) -> tuple[bool, str]:
        existing_guid_path = guid_index.get(guid.casefold())
        if existing_guid_path:
            return False, existing_guid_path
        if os.path.exists(destination):
            target_guid = _meta_guid(destination)
            if target_guid.casefold() != guid.casefold():
                raise PackageConflictError(
                    f"Destination is occupied by another GUID: {destination}"
                )
            return False, destination
        return True, destination

    def _guid_index(self) -> dict[str, str]:
        from infernux.application import Application

        if self.runtime and Application.is_player():
            from infernux.engine.project_context import resolve_runtime_asset_guid

            # A sealed Player owns cooked GUID bindings, not the Editor's
            # source/.meta checkout. Resolve exported files through that one
            # authority; the installation control manifest is not exported.
            result = {}
            for package in self.registry.installed():
                for item in package.get("files", []):
                    if not isinstance(item, Mapping):
                        continue
                    guid = str(item.get("guid", "")).casefold()
                    path = resolve_runtime_asset_guid(guid) if guid else None
                    if path:
                        result[guid] = path
            return result
        try:
            result, _native = project_guid_paths(
                self.project_root,
                engine=self.engine,
            )
            for package in self.registry.installed():
                for item in [*package.get("files", []), package.get("control")]:
                    if not isinstance(item, Mapping):
                        continue
                    guid = str(item.get("guid", "")).casefold()
                    hint = portable_path(str(item.get("path_hint", ""))).strip("/")
                    if not guid or not hint or guid in result:
                        continue
                    path = resolved_path(
                        os.path.join(self.project_root, *hint.split("/"))
                    )
                    if os.path.isfile(path) and _meta_guid(path).casefold() == guid:
                        result[guid] = path
            return result
        except ValueError as exc:
            raise PackageConflictError(str(exc)) from exc

    def _state_for_record(
        self,
        record: Mapping[str, object],
        lifecycle: tuple[dict[str, object], ...],
        guid_index: Mapping[str, str],
    ) -> PluginState:
        reference = str(record.get("reference", ""))
        package_lifecycle = tuple(
            item
            for item in lifecycle
            if str(item.get("package_reference", "")).casefold()
            == reference.casefold()
        )
        errors = [str(item.get("error", "")) for item in package_lifecycle if item.get("error")]
        errors.extend(self.preloads.package_errors(reference))
        enabled = bool(record.get("enabled", True))
        resources = {}
        files = list(record.get("files", []))
        if not self.runtime:
            files.append(record.get("control"))
        for item in files:
            if not isinstance(item, Mapping):
                continue
            guid = str(item.get("guid", "")).casefold()
            path = guid_index.get(guid)
            if not path:
                errors.append(f"Missing package asset {reference}: {item.get('logical_path', '')} ({guid})")
            elif item.get("role") != "control":
                resources[str(item.get("logical_path", ""))] = path
        return PluginState(
            reference,
            f"Packages/{reference}",
            enabled,
            loaded=enabled and not errors,
            error="; ".join(errors),
            resources=resources,
            lifecycle=package_lifecycle,
            restart_required=any(
                bool(item.get("restart_required")) for item in package_lifecycle
            ),
        )

    def _attach_resource_events(self) -> None:
        from infernux.engine.resources_manager import ResourcesManager

        manager = ResourcesManager.instance()
        if manager is not None:
            manager.register_script_catalog_callback(self._on_script_catalog_changed)
            self._resource_manager = manager

    def _on_script_catalog_changed(self, file_path: str, event_type: str) -> None:
        if self._installing:
            self._deferred_catalog_changes.add(resolved_path(file_path))
            return
        normalized = portable_path(str(file_path)).casefold()
        reloaded = ()
        if normalized.endswith("/editor/translations.json"):
            reference = self.preloads.package_reference_for_path(file_path)
            if not reference:
                return
            self.preloads.reload_package_translations(reference)
        elif normalized.endswith(".py"):
            reloaded = self.preloads.reload_path(file_path)
        else:
            return
        self._rebuild_states()
        references = sorted({state.package_reference for state in reloaded
                             if state.loaded and state.package_reference}, key=str.casefold)
        if references:
            from infernux.engine.resources_manager import ResourcesManager

            manager = ResourcesManager.instance()
            if manager is not None:
                for reference in references:
                    if self.states[reference.casefold()].loaded:
                        manager.defer_script_catalog_publication(
                            self._publish_package_runtime_scripts, reference,
                        )

    def _publish_package_runtime_scripts(self, reference: str) -> None:
        """Publish installed Runtime components through the resource authority."""

        record = self.registry.installed_record(reference)
        if record is None:
            raise KeyError(f"Plugin is not installed: {reference}")
        if not bool(record.get("enabled", True)):
            return
        from infernux.engine.resources_manager import ResourcesManager

        manager = ResourcesManager.instance()
        if manager is None:
            return
        if not same_path(getattr(manager, "_project_path", ""), self.project_root):
            raise RuntimeError(
                "Active ResourcesManager belongs to a different project during plugin install"
            )

        from infernux.engine.project_context import is_preload_python_library, is_project_component_script

        runtime_paths = [
            path for path in self._package_runtime_script_paths(record)
            if is_project_component_script(path, self.project_root)
        ]
        root = package_control_root(self.project_root, reference)
        library_names = tuple(
            name for name, module in tuple(sys.modules.items())
            if (source := getattr(module, "__file__", ""))
            and is_path_within(source, root, allow_root=False)
            and is_preload_python_library(source)
        )
        if library_names:
            runtime_paths = sorted(set(runtime_paths).union(
                manager.script_library_dependents(library_names)), key=path_key)
        if not runtime_paths:
            return

        transaction_id = manager.begin_script_transaction(runtime_paths)
        submitted = False
        for path in runtime_paths:
            change = manager.submit_script_change(
                path,
                origin="editor",
                catalog_event=None,
                change_kind="created",
                transaction_id=transaction_id,
                force=True,
            )
            submitted = submitted or change is not None
        if submitted:
            manager.process_pending_reloads(force=True)

    def _retire_package_runtime_scripts(self, record: Mapping[str, object]) -> None:
        """Remove package gameplay types and modules from the active runtime."""

        from infernux.engine.resources_manager import ResourcesManager

        manager = ResourcesManager.instance()
        if manager is None:
            return
        if not same_path(getattr(manager, "_project_path", ""), self.project_root):
            raise RuntimeError(
                "Active ResourcesManager belongs to a different project during plugin retirement"
            )
        paths = self._package_runtime_script_paths(record, retiring=True)
        if paths:
            manager.retire_script_paths(paths)

    def _package_runtime_script_paths(
        self, record: Mapping[str, object], *, retiring: bool = False
    ) -> list[str]:
        """Current Runtime scripts, plus historical module paths for retirement."""

        from infernux.engine.project_context import package_script_reference

        guid_paths = self._guid_index()
        paths: dict[str, str] = {}
        if retiring:
            for item in record.get("files", []):
                if not isinstance(item, Mapping) or item.get("role") != "runtime":
                    continue
                hint = portable_path(str(item.get("path_hint", ""))).strip("/")
                if hint.casefold().endswith(".py"):
                    # Removed or moved files can still have live modules at
                    # their old paths. Never reinterpret the current Editor
                    # destination of that GUID as a gameplay module to retire.
                    path = resolved_path(os.path.join(self.project_root, *hint.split("/")))
                    paths[path_key(path)] = path
        reference = str(record["reference"])
        root = package_control_root(self.project_root, reference)
        for path in guid_paths.values():
            if not path.casefold().endswith(".py") or not is_path_within(path, root, allow_root=False):
                continue
            logical = portable_path(relative_path(path, root))
            if logical.partition("/")[0].casefold() != "runtime":
                continue
            current_reference = package_script_reference(path, self.project_root)
            # A nested package is independent. During uninstall the removed
            # control file may already be gone; the transaction's record still
            # defines which surviving, user-authored scripts must be retired.
            if current_reference and current_reference.casefold() != reference.casefold():
                continue
            paths[path_key(path)] = path
        return [paths[key] for key in sorted(paths)]

    def _refresh_editor_assets(self) -> None:
        if self.installation_only or threading.current_thread() is not threading.main_thread():
            return
        database = _asset_database(self.engine)
        if database is not None and same_path(database.project_root, self.project_root):
            database.refresh()

    def _prune_package_directories(self, removed_paths: Iterable[str]) -> None:
        boundaries = (
            os.path.join(self.project_root, "Packages"),
            os.path.join(self.project_root, "Assets", "Plugins"),
        )
        roots = {
            resolved_path(os.path.dirname(path))
            for path in removed_paths
            if path
        }
        for root in sorted(roots, key=lambda path: len(Path(path).parts), reverse=True):
            boundary = next(
                (
                    candidate
                    for candidate in boundaries
                    if is_path_within(root, candidate, allow_root=False)
                ),
                "",
            )
            if not boundary:
                continue
            current = root
            while is_path_within(current, boundary, allow_root=False):
                try:
                    os.rmdir(current)
                except OSError:
                    break
                current = os.path.dirname(current)

    def _project_python_executable(self) -> str:
        candidates = (
            os.path.join(
                self.project_root,
                ".runtime",
                PYTHON_RUNTIME_DIRECTORY,
                "python.exe",
            ),
            os.path.join(self.project_root, ".venv", "Scripts", "python.exe"),
            os.path.join(self.project_root, ".venv", "bin", "python"),
        )
        for candidate in candidates:
            if os.path.isfile(candidate):
                return candidate
        return sys.executable

    def _cache_package(
        self,
        package_path: str,
        reference: str,
        version: str,
    ) -> tuple[str, str]:
        cache = self._package_cache()
        destination = cache.store(
            package_path,
            reference=reference,
            version=version,
        )
        return destination, cache.relative_path(reference, version)

    def _source_descriptor(
        self, source: Mapping[str, object] | str
    ) -> dict[str, object]:
        if isinstance(source, str):
            value = source.strip()
            lower = value.casefold()
            if lower.startswith(("http://", "https://")) and lower.endswith(
                PACKAGE_EXTENSION
            ):
                source_type = "url"
            elif lower.startswith(("http://", "https://", "git@")) or lower.endswith(
                ".git"
            ):
                source_type = "github" if "github.com" in lower else "git"
            else:
                source_type = "local"
            return {"type": source_type, "location": value}
        descriptor = dict(source)
        unknown = set(descriptor).difference(_SOURCE_DESCRIPTOR_FIELDS)
        if unknown:
            fields = ", ".join(sorted(map(str, unknown)))
            raise ValueError(f"Plugin source contains unknown fields: {fields}")
        source_type = str(descriptor.get("type", "")).strip().casefold()
        location = str(descriptor.get("location", "")).strip()
        if source_type not in {"local", "github", "git", "url"} or not location:
            raise ValueError(
                "Plugin source must be local, git, github, or url and include location"
            )
        descriptor["type"] = source_type
        descriptor["location"] = location
        return descriptor

    def _run_process(
        self, command: list[str], *, cwd: str | None = None
    ) -> subprocess.CompletedProcess[str]:
        is_pip = command[1:3] == ["-m", "pip"]
        workspace_scope = (
            self._package_cache().workspace("pip") if is_pip else nullcontext(None)
        )
        with workspace_scope as workspace:
            env = None
            if is_pip:
                if command[3:4] == ["install"]:
                    constraint = Path(workspace) / "engine-constraint.txt"
                    constraint.write_text(
                        f"Infernux=={ENGINE_VERSION}\n", encoding="utf-8", newline="\n"
                    )
                    command = [*command[:4], "--constraint", str(constraint), *command[4:]]
                env = dict(os.environ)
                shared = env.get("INFERNUX_SHARED_DATA_ROOT", "").strip()
                cache_root = (
                    resolved_path(os.path.expandvars(os.path.expanduser(shared)))
                    if shared else self.project_root
                )
                if not env.get("PIP_CACHE_DIR", "").strip():
                    env["PIP_CACHE_DIR"] = os.path.join(cache_root, "Cache", "Python", "Pip")
                # pip and its build subprocesses own no system-temp residue.
                # Only this invocation sees these paths; the Editor's process
                # environment and unrelated subprocesses remain untouched.
                env.update(TMPDIR=workspace, TEMP=workspace, TMP=workspace)
            result = subprocess.run(
                command,
                executable=process_executable_path(command[0]),
                cwd=cwd,
                env=env,
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                check=False,
            )
        if result.returncode:
            detail = (result.stderr or result.stdout).strip()
            raise RuntimeError(
                f"Command failed ({result.returncode}): {detail[-4000:]}"
            )
        return result

    def _run_process_with_progress(
        self,
        command: list[str],
        *,
        stage: str,
        progress: _InstallProgress | None,
        start: float,
        end: float,
        cwd: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        if progress is None:
            return self._run_process(command, cwd=cwd)
        process = subprocess.Popen(
            command,
            executable=process_executable_path(command[0]),
            cwd=cwd,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        output: list[str] = []
        assert process.stdout is not None
        for raw_line in process.stdout:
            line = raw_line.strip()
            if not line:
                continue
            output.append(line)
            matches = _GIT_PROGRESS_PERCENT.findall(line)
            fraction = int(matches[-1]) / 100.0 if matches else 0.0
            _report_progress(
                progress,
                stage,
                start + (end - start) * fraction,
                line,
            )
        return_code = process.wait()
        combined = "\n".join(output)
        if return_code:
            raise RuntimeError(f"Command failed ({return_code}): {combined[-4000:]}")
        return subprocess.CompletedProcess(command, return_code, combined, "")


def _script_bytecode_paths(source: str) -> tuple[str, ...]:
    """Derived caches of this owned source, never other author files."""

    if not source.endswith(".py"):
        return ()
    directory = os.path.join(os.path.dirname(source), "__pycache__")
    if not os.path.isdir(directory) or os.path.islink(directory):
        return ()
    result = []
    for entry in os.scandir(directory):
        if entry.is_symlink() or not entry.is_file() or not entry.name.endswith(".pyc"):
            continue
        try:
            cached_source = importlib.util.source_from_cache(entry.path)
        except ValueError:
            continue
        if same_path(cached_source, source):
            result.append(entry.path)
    return tuple(sorted(result))


class _InstallTransaction:
    def __init__(self, project_root: str) -> None:
        self.project_root = project_root
        self.id = uuid.uuid4().hex
        self.root = os.path.join(
            project_root, "Cache", "Plugins", ".transactions", self.id
        )
        self.backups: list[tuple[str, str | None]] = []
        self.committed = False
        os.makedirs(self.root, exist_ok=True)

    def write(self, destination: str, payload: bytes) -> None:
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        backup = None
        if os.path.exists(destination):
            backup = os.path.join(self.root, f"backup-{len(self.backups)}")
            shutil.copy2(destination, backup)
        temporary = os.path.join(self.root, f"write-{len(self.backups)}")
        with open(temporary, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        self.backups.append((destination, backup))

    def remove(self, destination: str) -> None:
        if not os.path.exists(destination):
            return
        backup = os.path.join(self.root, f"backup-{len(self.backups)}")
        shutil.copy2(destination, backup)
        self.backups.append((destination, backup))
        os.remove(destination)

    def commit(self) -> None:
        self.committed = True
        shutil.rmtree(self.root, ignore_errors=True)

    def rollback(self) -> None:
        if self.committed:
            return
        for destination, backup in reversed(self.backups):
            if backup and os.path.isfile(backup):
                os.replace(backup, destination)
            else:
                try:
                    os.remove(destination)
                except FileNotFoundError:
                    pass
        shutil.rmtree(self.root, ignore_errors=True)


def _python_environment_changes(
    before: Mapping[str, str], after: Mapping[str, str]
) -> tuple[dict[str, str], ...]:
    return tuple(
        {
            "name": name,
            "before": str(before.get(name, "")),
            "after": str(after.get(name, "")),
        }
        for name in sorted(set(before) | set(after))
        if before.get(name, "") != after.get(name, "")
    )


def _pip_requirement_targets(
    values: Iterable[str],
) -> tuple[dict[str, str], ...]:
    result: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for raw in values:
        requirement = str(raw).strip()
        if not requirement or requirement.startswith("#"):
            continue
        candidate = requirement
        if candidate.startswith(("-e ", "--editable ")):
            candidate = candidate.split(None, 1)[1].strip()
        if candidate.startswith("-"):
            continue
        name = ""
        try:
            parsed = Requirement(candidate)
            if parsed.marker is not None and not parsed.marker.evaluate():
                continue
            name = parsed.name
        except Exception:
            marker = candidate.rsplit("#egg=", 1)
            if len(marker) == 2:
                name = marker[1].split("&", 1)[0].strip()
            else:
                basename = os.path.basename(
                    unquote(urlsplit(candidate).path or candidate)
                )
                if basename.casefold().endswith(".whl"):
                    try:
                        wheel_name, wheel_version, _, _ = parse_wheel_filename(basename)
                        name = str(wheel_name)
                        requirement = f"{name}=={wheel_version}"
                    except Exception:
                        name = ""
        canonical = canonicalize_name(name) if name else ""
        if canonical == "infernux":
            raise ValueError(
                "Plugin requirements cannot install the engine; use the exact "
                "engine version pinned in .infernux-version"
            )
        key = (canonical, requirement)
        if canonical and key not in seen:
            result.append({"name": canonical, "requirement": requirement})
            seen.add(key)
    return tuple(result)


def _requirements_satisfied(
    requirements: Iterable[str], environment: Mapping[str, str], *, check_sources: bool = False,
) -> bool:
    for value in requirements:
        targets = _pip_requirement_targets((value,))
        if len(targets) != 1:
            return False
        target = targets[0]
        version = environment.get(target["name"], "")
        if not version:
            return False
        try:
            parsed = Requirement(target["requirement"])
        except Exception:
            return False
        if parsed.specifier and not parsed.specifier.contains(
            version, prereleases=True
        ):
            return False
        if check_sources and parsed.url:
            try:
                distribution = importlib.metadata.distribution(parsed.name)
            except importlib.metadata.PackageNotFoundError:
                return False
            direct_url = distribution.read_text("direct_url.json")
            if direct_url is None or json.loads(direct_url).get("url") != parsed.url:
                return False
    return True


def _package_file_identities(files: Iterable[Mapping[str, object]]) -> set[tuple[str, str]]:
    return {
        (str(item.get("logical_path", "")), str(item.get("guid", "")))
        for item in files
        if isinstance(item, Mapping)
    }


def _same_install(record: Mapping[str, object], preview: InxPackagePreview) -> bool:
    current = _package_file_identities(record.get("files", []))
    incoming = _package_file_identities(preview.file_records)
    return current == incoming and str(record.get("version", "")) == str(
        preview.metadata.get("version", "")
    )


def _meta_guid(asset_path: str) -> str:
    try:
        with open(asset_path + ".meta", "r", encoding="utf-8") as stream:
            document = json.load(stream)
        return str(document["metadata"]["guid"]["value"]).strip()
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return ""


__all__ = ["PackageConflictError", "PluginManager", "PluginState"]
