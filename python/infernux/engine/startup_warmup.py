"""Persistent CPU/GPU startup warmup registry."""

from __future__ import annotations

import ast
import inspect
import json
import os
import platform
import sys
import tempfile
import time
from collections import deque
from pathlib import Path
from typing import Callable, Iterable

from infernux.debug import Debug
from infernux._source_kinds import is_host_python_source
from infernux.engine.path_utils import lexical_path

_SCHEMA = "infernux.startup-warmup"
_FILENAME = "StartupWarmup.json"
_GPU_PUMP_BUDGET_SECONDS = 0.004


class PlayerStartupWarmup:
    """Resolve declarations and install GPU pipelines on the Player thread."""

    def __init__(self, records: tuple[dict, ...], *, scope: str):
        self._records = deque(records)
        self._scope = scope
        self._prepares = deque()
        self._pending_gpu = None
        self._started = time.perf_counter()
        self._completed = False
        self._prepare_count = 0

    @property
    def completed(self) -> bool:
        return self._completed

    def close(self) -> None:
        pending, self._pending_gpu = self._pending_gpu, None
        if pending is not None:
            pending.cancel()
        self._prepares.clear()
        self._records.clear()
        self._completed = True

    def _pump_gpu(self) -> None:
        from infernux.engine.player_log import write_player_log

        pump_started = time.perf_counter()
        while self._pending_gpu is not None or self._prepares:
            if self._pending_gpu is not None:
                pending = self._pending_gpu
                if not pending.finish():
                    return
                self._pending_gpu = None
            else:
                declaration, params, autodiff_mode, gradient_parameters = self._prepares.popleft()
                self._pending_gpu = declaration._begin_prepare(
                    params, autodiff_mode=autodiff_mode, gradient_parameters=gradient_parameters,
                )
                if self._pending_gpu is not None:
                    # Admit one cold compile at a time. Slang serializes its
                    # resident session, and the same cache serves later requests.
                    return
            self._prepare_count += 1
            write_player_log(f"[Startup.Warmup] GPU specialization {self._prepare_count} ready")
            if time.perf_counter() - pump_started >= _GPU_PUMP_BUDGET_SECONDS:
                return

    def pump(self) -> bool:
        """Run one bounded warmup unit; return true when all work is complete."""
        if self._completed:
            return True
        if self._pending_gpu is not None or self._prepares:
            self._pump_gpu()
            return False
        if self._records:
            record = self._records.popleft()
            from infernux import compute
            with compute.defer_prepares() as pending:
                hook_elapsed, declaration_elapsed = _run_player_warmup_record(record)
            self._prepares.extend(pending)
            from infernux.engine.player_log import write_player_log

            write_player_log(
                "[Startup.Warmup] declaration "
                f"{record['module']}.{record['qualname']}: "
                f"{declaration_elapsed:.1f} ms, hook={hook_elapsed:.1f} ms, "
                f"queued_gpu={len(pending)}"
            )
            Debug.log_debug(
                "INFERNUX_PLAYER_WARMUP_DECLARED "
                f"type={record['module']}.{record['qualname']} "
                f"prepares={len(pending)} "
                f"elapsed_ms={hook_elapsed:.1f}"
            )
            return False
        self._completed = True
        from infernux.engine.player_log import write_player_log

        elapsed_ms = (time.perf_counter() - self._started) * 1000.0
        write_player_log(
            "[Startup.Warmup] ready "
            f"gpu_specializations={self._prepare_count} elapsed_ms={elapsed_ms:.1f}"
        )
        Debug.log_debug(
            f"INFERNUX_PLAYER_WARMUP_READY scope={self._scope} "
            f"elapsed_ms={elapsed_ms:.1f}"
        )
        return True


def _run_player_warmup_record(record: dict) -> tuple[float, float]:
    hook, resolution_elapsed = _resolve_player_warmup_record(record)
    return _run_resolved_player_warmup_record(record, hook, resolution_elapsed)


def _resolve_player_warmup_record(record: dict) -> tuple[object, float]:
    """Resolve and validate one authored hook on the runtime owner thread."""
    resolution_started = time.perf_counter()
    from infernux.engine.component_restore import create_component_instance
    from infernux.engine.runtime_worlds import source_module_name

    type_name = str(record["qualname"]).rsplit(".", 1)[-1]
    component, resolved_script_path = create_component_instance(
        str(record["script_guid"]),
        str(record["type_guid"]),
        type_name,
        prefer_loaded_type=True,
    )
    if (
        component is None
        or component.__class__._get_type_guid() != str(record["type_guid"])
        or source_module_name(component.__class__.__module__) != str(record["module"])
        or component.__class__.__qualname__ != str(record["qualname"])
    ):
        if component is None:
            from infernux.components.script_loader import get_script_error_by_path

            actual = (
                get_script_error_by_path(resolved_script_path)
                if resolved_script_path
                else "script path was not resolved"
            )
        else:
            actual = (
                f"module={component.__class__.__module__!r}, "
                f"qualname={component.__class__.__qualname__!r}, "
                f"type_guid={component.__class__._get_type_guid()!r}"
            )
        raise RuntimeError(
            "Player startup warmup type is not loadable by its build identity: "
            f"{record['module']}.{record['qualname']}: {actual}"
        )
    hook = getattr(component, "_infernux_startup_warmup", None)
    if not callable(hook):
        raise RuntimeError(
            "Player startup warmup hook is missing: "
            f"{record['module']}.{record['qualname']}"
        )
    return hook, (time.perf_counter() - resolution_started) * 1000.0


def _run_resolved_player_warmup_record(
    record: dict, hook: object, resolution_elapsed: float
) -> tuple[float, float]:
    """Execute a resolved hook without touching component registry state."""
    hook_started = time.perf_counter()
    ready = hook()
    hook_elapsed = (time.perf_counter() - hook_started) * 1000.0
    if ready is False:
        raise RuntimeError(
            "Player startup warmup rejected its build-selected declaration: "
            f"{record['module']}.{record['qualname']}"
        )
    return hook_elapsed, resolution_elapsed + hook_elapsed


def player_startup_warmup_records(project_path: str) -> tuple[dict, ...]:
    """Read the build-authored declaration identities before opening a world."""
    root = Path(lexical_path(project_path))
    return tuple(sorted(
        _runtime_warmup_records(root),
        key=lambda record: (
            str(record["runtime_path"]).casefold(),
            str(record["qualname"]).casefold(),
        ),
    ))
def _runtime_warmup_records(root: Path) -> tuple[dict, ...]:
    registry_path = root / "Library" / "RuntimeTypeRegistry.json"
    with registry_path.open("r", encoding="utf-8") as stream:
        registry = json.load(stream)
    return tuple(
        record
        for record in registry["types"]
        if isinstance(record, dict) and record.get("startup_warmup") is True
    )


def _platform_identity() -> str:
    if sys.platform == "win32":
        return "windows"
    if sys.platform == "emscripten" or os.environ.get("INFERNUX_WEB_RUNTIME") == "1":
        return "web"
    return sys.platform.casefold() or platform.system().casefold()


def _source_identity(component: object) -> tuple[str, str]:
    cls = type(component)
    try:
        source = inspect.getsourcefile(cls) or inspect.getfile(cls) or ""
    except (OSError, TypeError):
        source = ""
    source = _canonical_source(source, cls.__module__) if source else ""
    hook = getattr(cls, "_infernux_startup_warmup", None)
    qualname = getattr(hook, "__qualname__", "_infernux_startup_warmup")
    return f"{cls.__module__}.{cls.__qualname__}:{qualname}", source


def _canonical_source(source: str, module_name: str) -> str:
    """Map stale authored absolute paths to the active project copy."""
    candidate = lexical_path(source) if source else ""
    if candidate and os.path.isfile(candidate):
        return candidate
    try:
        from infernux.engine.project_context import get_project_root

        root = get_project_root()
    except Exception:
        root = ""
    if root and module_name and not module_name.startswith("infernux."):
        relative = os.path.join(*module_name.split(".")) + ".py"
        project_candidate = os.path.join(root, "Assets", relative)
        if os.path.isfile(project_candidate):
            return lexical_path(project_candidate)
    return candidate


def _registry_path(project_path: str | None) -> str | None:
    if not project_path:
        return None
    return os.path.join(lexical_path(project_path), "Library", _FILENAME)


def _load(path: str) -> dict:
    if not os.path.isfile(path):
        return {"schema": _SCHEMA, "platform": _platform_identity(), "entries": {}}
    with open(path, "r", encoding="utf-8") as stream:
        document = json.load(stream)
    if not isinstance(document, dict) or document.get("schema") != _SCHEMA:
        raise RuntimeError(f"Invalid startup warmup registry: {path}")
    if not isinstance(document.get("entries"), dict):
        raise RuntimeError(f"Startup warmup registry entries are invalid: {path}")
    return document


def _write(path: str, document: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="StartupWarmup.", suffix=".tmp", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(document, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.remove(temporary)
        except FileNotFoundError:
            pass


def invalidate_source(file_path: str, *, project_path: str | None = None) -> int:
    """Remove successful records owned by one saved script."""
    path = _registry_path(project_path)
    if path is None or not os.path.isfile(path):
        return 0
    document = _load(path)
    source = lexical_path(file_path)
    entries = document["entries"]
    removed = []
    for key, value in entries.items():
        recorded = lexical_path(str(value.get("source", "") or ""))
        if recorded == source:
            removed.append(key)
            continue
        # Older registries may contain the authored absolute path from a
        # machine-specific .meta file. Resolve the stable module identity
        # against this project before deciding that a save invalidates it.
        module_name = str(key).split(":", 1)[0]
        canonical = _canonical_source("", module_name.rsplit(".", 1)[0])
        if canonical and canonical == source:
            removed.append(key)
    if removed:
        for key in removed:
            del entries[key]
        document["updated_at"] = time.time()
        _write(path, document)
        Debug.log_debug(f"INFERNUX_STARTUP_WARMUP_INVALIDATED source={source} entries={len(removed)}")
    return len(removed)


def run_component_warmups(
    components: Iterable[object], *, scope: str, project_path: str | None = None,
    can_continue: Callable[[], bool] | None = None,
) -> int:
    """Scan persisted results and execute only missing or changed warmups."""
    started = time.perf_counter()
    path = _registry_path(project_path)
    document = _load(path) if path else {"schema": _SCHEMA, "entries": {}}
    entries = document["entries"]
    executed = 0
    cached = 0
    current_platform = _platform_identity()
    for component in tuple(components):
        if can_continue is not None and not can_continue():
            return executed
        hook = getattr(component, "_infernux_startup_warmup", None)
        if not callable(hook):
            continue
        key, source = _source_identity(component)
        record = entries.get(key)
        if (
            isinstance(record, dict)
            and record.get("platform") == current_platform
            and record.get("status") == "ready"
        ):
            cached += 1
            continue
        hook_started = time.perf_counter()
        from infernux.execution import component_execution
        with component_execution(component):
            ready = hook()
        if can_continue is not None and not can_continue():
            return executed
        hook_elapsed = (time.perf_counter() - hook_started) * 1000.0
        # A component may not be ready to prepare its native resources until
        # its authored scene has created the required buffers/renderers.  Such
        # a hook is allowed to return False; do not persist a false positive
        # that would suppress the real scene-bound warmup later.
        if ready is False:
            continue
        executed += 1
        Debug.log_debug(
            f"INFERNUX_STARTUP_WARMUP_HOOK key={key} elapsed_ms={hook_elapsed:.1f}"
        )
        if path:
            entries[key] = {
                "source": source,
                "platform": current_platform,
                "status": "ready",
                "updated_at": time.time(),
            }
            document["updated_at"] = time.time()
            _write(path, document)
    elapsed = (time.perf_counter() - started) * 1000.0
    Debug.log_debug(
        f"INFERNUX_STARTUP_WARMUP_READY scope={scope} hooks={executed} "
        f"cached={cached} elapsed_ms={elapsed:.1f}"
    )
    return executed


def run_project_script_warmups(
    *, project_path: str | None, scope: str, can_continue: Callable[[], bool] | None = None,
) -> int:
    """Discover warmups from the host's explicit Editor/Player contract.

    An Editor project with no scripts has no script warmups. A synchronous
    Player host uses its build-authored registry regardless of source files.
    Packaged Players advance ``PlayerStartupWarmup`` inside World preparation.
    Scene-bound GPU components return ``False`` until their buffers exist
    and are warmed by scene activation.
    """
    if not project_path:
        return 0
    from infernux.application import Application

    root = Path(lexical_path(project_path))
    paths: list[str] = []
    if Application.is_player():
        paths = sorted({
            str(root / str(record["runtime_path"]))
            for record in _runtime_warmup_records(root)
        }, key=lambda value: value.casefold())
    else:
        scripts_root = root / "Assets"
        source_paths = (
            sorted((path for path in scripts_root.rglob("*.py") if is_host_python_source(path)),
                   key=lambda item: item.as_posix().casefold())
            if scripts_root.is_dir() else []
        )
        for path in source_paths:
            try:
                source = path.read_text(encoding="utf-8")
                if "_infernux_startup_warmup" not in source:
                    continue
                tree = ast.parse(source, filename=str(path))
            except (OSError, UnicodeError, SyntaxError) as exc:
                raise RuntimeError(f"Cannot inspect startup warmup script: {path}") from exc
            if any(
                isinstance(node, ast.FunctionDef)
                and node.name == "_infernux_startup_warmup"
                for node in ast.walk(tree)
            ):
                paths.append(str(path))
    if not paths:
        return 0
    from infernux.components.registry import component_types_for_script_path
    from infernux.components.script_loader import load_all_components_from_file

    components: list[object] = []
    for path in paths:
        if can_continue is not None and not can_continue():
            return 0
        source = lexical_path(path)
        types = component_types_for_script_path(source)
        if not types:
            types = tuple(load_all_components_from_file(path, register=True))
        for component_type in types:
            if can_continue is not None and not can_continue():
                return 0
            component = component_type()
            components.append(component)
    return run_component_warmups(components, scope=scope, project_path=project_path,
                                 can_continue=can_continue)


def run_script_warmup(
    file_path: str, *, project_path: str | None, scope: str
) -> int:
    """Warm one saved script immediately after its durable commit."""
    source = lexical_path(file_path)
    from infernux.components.component import InxComponent
    from infernux.components.registry import component_types_for_script_path

    # An empty registry is expected for helper scripts. Re-importing here
    # would replace the module and classes that the reload just published.
    types = component_types_for_script_path(source)
    if not types:
        return 0
    active = tuple(
        component
        for values in InxComponent._active_instances.values()
        for component in values
        if lexical_path(_source_identity(component)[1]) == source
    )
    components = active or tuple(component_type() for component_type in types)
    if not components:
        return 0
    return run_component_warmups(components, scope=scope, project_path=project_path)
