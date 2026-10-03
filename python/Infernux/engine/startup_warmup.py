"""Persistent CPU/GPU startup warmup registry."""

from __future__ import annotations

import inspect
import json
import os
import platform
import sys
import tempfile
import threading
import time
import ast
from pathlib import Path
from typing import Iterable

from Infernux.debug import Debug
from Infernux.engine.path_utils import lexical_path

_SCHEMA = "infernux.startup-warmup"
_FILENAME = "StartupWarmup.json"
_cpu_runtime_preload_lock = threading.Lock()
_cpu_runtime_preload_thread: threading.Thread | None = None
_cpu_runtime_preload_done = threading.Event()
_cpu_runtime_preload_error: Exception | None = None
_GPU_PUMP_BUDGET_SECONDS = 0.004


def start_cpu_runtime_preload() -> None:
    """Load the packaged CPU compiler while native startup owns the main thread."""
    global _cpu_runtime_preload_thread
    # Web and Android Players are cooked without the desktop Numba/llvmlite
    # runtime. Their CPU kernels are already lowered to ordinary Python during
    # Cook, so importing the desktop JIT facade here only competes with Vulkan
    # and scene startup for CPU and storage bandwidth.
    if _platform_identity() in {"web", "android"}:
        _cpu_runtime_preload_done.set()
        return
    with _cpu_runtime_preload_lock:
        if _cpu_runtime_preload_thread is not None:
            return
        _cpu_runtime_preload_thread = threading.Thread(
            target=_preload_cpu_runtime,
            name="InfernuxCpuRuntimePreload",
            daemon=True,
        )
        _cpu_runtime_preload_thread.start()


def _preload_cpu_runtime() -> None:
    global _cpu_runtime_preload_error
    started = time.perf_counter()
    try:
        from Infernux import jit

        if jit.JIT_AVAILABLE:
            from Infernux._jit_cache import cpu_cache_root

            cpu_cache_root()
        from Infernux.engine.player_log import write_player_log

        write_player_log(
            "[Startup.Warmup] CPU runtime preloaded: "
            f"{(time.perf_counter() - started) * 1000.0:.1f} ms"
        )
    except Exception as exc:
        _cpu_runtime_preload_error = exc
    finally:
        _cpu_runtime_preload_done.set()


def wait_cpu_runtime_preload() -> None:
    """Join the compiler-only preload before publishing authored CPU code."""
    start_cpu_runtime_preload()
    _cpu_runtime_preload_done.wait()
    if _cpu_runtime_preload_error is not None:
        raise RuntimeError("Player CPU runtime preload failed") from _cpu_runtime_preload_error


class PlayerStartupWarmup:
    """Materialize GPU work on the owner thread and CPU work in parallel."""

    def __init__(
        self,
        project_path: str,
        records: tuple[dict, ...],
        *,
        cpu_records: tuple[dict, ...] = (),
        scope: str,
    ):
        self._project_path = project_path
        self._scope = scope
        self._records = list(records)
        self._cpu_records = list(cpu_records)
        self._cpu_jobs: list[tuple[dict, object, float]] = []
        self._cpu_records_resolved = False
        self._prepares: list[tuple[object, tuple]] = []
        # The Player window is still hidden while bootstrap pumps this queue.
        # Starting immediately lets persisted GPU work and the CPU worker
        # overlap scene preparation instead of charging the first interaction.
        self._delay_frames = 0
        self._started = time.perf_counter()
        self._completed = False
        self._prepare_count = 0
        self._cpu_count = 0
        self._cpu_elapsed_ms = 0.0
        self._cpu_thread: threading.Thread | None = None
        self._cpu_done = threading.Event()
        self._cpu_error: BaseException | None = None

    def start_cpu(self) -> None:
        """Start already-resolved CPU hooks without delaying the owner thread."""
        if self._cpu_thread is not None or self._cpu_done.is_set():
            return
        if not self._cpu_records_resolved:
            raise RuntimeError("CPU warmup hooks must be resolved on the owner thread")
        if not self._cpu_jobs:
            self._cpu_done.set()
            return
        self._cpu_thread = threading.Thread(
            target=self._prepare_cpu_worker,
            name="InfernuxProjectCpuWarmup",
            daemon=True,
        )
        self._cpu_thread.start()

    def _prepare_cpu_worker(self) -> None:
        try:
            self._prepare_cpu_records()
        except BaseException as exc:
            self._cpu_error = exc
        finally:
            self._cpu_done.set()

    def prepare_cpu(self) -> None:
        """Wait for CPU declarations when an explicit synchronous gate needs them."""
        self._resolve_cpu_records()
        self.start_cpu()
        self._cpu_done.wait()
        if self._cpu_error is not None:
            raise RuntimeError("Player CPU startup warmup failed") from self._cpu_error

    def _prepare_cpu_records(self) -> None:
        """Publish build-selected CPU-only declarations on the warmup worker."""
        if not self._cpu_jobs:
            return
        wait_cpu_runtime_preload()
        started = time.perf_counter()
        from Infernux import compute
        from Infernux.engine.player_log import write_player_log

        while self._cpu_jobs:
            record, hook, resolution_elapsed = self._cpu_jobs.pop(0)
            with compute.defer_prepares() as pending:
                hook_elapsed, declaration_elapsed = _run_resolved_player_warmup_record(
                    record, hook, resolution_elapsed
                )
            if pending:
                raise RuntimeError(
                    "CPU startup warmup queued GPU work from a module without "
                    f"GPU AOT artifacts: {record['module']}.{record['qualname']}"
                )
            self._cpu_count += 1
            write_player_log(
                "[Startup.Warmup] CPU declaration "
                f"{record['module']}.{record['qualname']}: "
                f"{declaration_elapsed:.1f} ms, hook={hook_elapsed:.1f} ms"
            )
        self._cpu_elapsed_ms += (time.perf_counter() - started) * 1000.0

    def _resolve_cpu_records(self) -> None:
        """Resolve components on the runtime owner before worker execution."""
        if self._cpu_records_resolved:
            return
        while self._cpu_records:
            record = self._cpu_records.pop(0)
            hook, resolution_elapsed = _resolve_player_warmup_record(record)
            self._cpu_jobs.append((record, hook, resolution_elapsed))
        self._cpu_records_resolved = True

    @property
    def completed(self) -> bool:
        return self._completed

    def pump(self) -> bool:
        """Run one bounded warmup unit; return true when all work is complete."""
        if self._completed:
            return True
        if self._delay_frames:
            self._delay_frames -= 1
            return False
        self._resolve_cpu_records()
        self.start_cpu()
        if self._cpu_done.is_set() and self._cpu_error is not None:
            raise RuntimeError("Player CPU startup warmup failed") from self._cpu_error
        if self._prepares:
            from Infernux import compute
            from Infernux.engine.player_log import write_player_log

            # A cooked Player normally resolves each persisted GPU
            # specialization in well under a millisecond. Draining only one
            # per frame stretched a small batch over hundreds of milliseconds
            # and allowed the first scene interaction to race the warmup. Keep
            # one strict owner-thread budget instead: cached work is grouped
            # into a few frames while an unexpectedly expensive item still
            # yields immediately after it completes.
            pump_started = time.perf_counter()
            while self._prepares:
                declaration, params = self._prepares.pop(0)
                prepare_started = time.perf_counter()
                compute._prepare_now(declaration, params)
                self._prepare_count += 1
                write_player_log(
                    "[Startup.Warmup] GPU specialization "
                    f"{self._prepare_count}: "
                    f"{(time.perf_counter() - prepare_started) * 1000.0:.1f} ms"
                )
                if time.perf_counter() - pump_started >= _GPU_PUMP_BUDGET_SECONDS:
                    break
            return False
        if self._records:
            record = self._records.pop(0)
            from Infernux import compute
            with compute.defer_prepares() as pending:
                hook_elapsed, declaration_elapsed = _run_player_warmup_record(record)
            self._prepares.extend(pending)
            from Infernux.engine.player_log import write_player_log

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
        if not self._cpu_done.is_set():
            return False
        self._completed = True
        from Infernux.engine.player_log import write_player_log

        elapsed_ms = (time.perf_counter() - self._started) * 1000.0
        write_player_log(
            "[Startup.Warmup] ready "
            f"cpu_declarations={self._cpu_count} "
            f"cpu_elapsed_ms={self._cpu_elapsed_ms:.1f} "
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
    from Infernux.engine.component_restore import create_component_instance

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
        or component.__class__.__module__ != str(record["module"])
        or component.__class__.__qualname__ != str(record["qualname"])
    ):
        if component is None:
            from Infernux.components.script_loader import get_script_error_by_path

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


def create_player_startup_warmup(
    *, project_path: str, scope: str = "player-startup"
) -> PlayerStartupWarmup:
    """Create the exact build-authored warmup queue without importing scripts."""
    root = Path(lexical_path(project_path))
    registry_path = root / "Library" / "RuntimeTypeRegistry.json"
    with registry_path.open("r", encoding="utf-8") as stream:
        registry = json.load(stream)
    records = tuple(
        record
        for record in registry["types"]
        if isinstance(record, dict) and record.get("startup_warmup") is True
    )
    records = tuple(sorted(
        records,
        key=lambda record: (
            str(record["runtime_path"]).casefold(),
            str(record["qualname"]).casefold(),
        ),
    ))
    # Emscripten owns one browser execution thread. Its build-selected CPU
    # compute path is already source lowered, so declarations remain in the
    # bounded owner queue. Native Players can overlap pure CPU JIT publication
    # with scene activation, while every module represented by the GPU AOT
    # manifest stays on the RHI owner thread.
    if _platform_identity() == "web":
        return PlayerStartupWarmup(str(root), records, scope=scope)

    gpu_modules = _gpu_aot_modules(root)
    gpu_records = tuple(
        record for record in records if str(record["module"]) in gpu_modules
    )
    cpu_records = tuple(
        record for record in records if str(record["module"]) not in gpu_modules
    )
    return PlayerStartupWarmup(
        str(root), gpu_records, cpu_records=cpu_records, scope=scope
    )


def _gpu_aot_modules(root: Path) -> frozenset[str]:
    """Read the exact project modules represented by packaged GPU artifacts."""
    manifest_path = root / "Library" / "Artifacts" / "Compute" / "AotManifest.json"
    if not manifest_path.is_file():
        return frozenset()
    with manifest_path.open("r", encoding="utf-8") as stream:
        document = json.load(stream)
    if not isinstance(document, dict) or set(document) != {"artifacts"}:
        raise RuntimeError(f"Invalid GPU AOT manifest: {manifest_path}")
    artifacts = document["artifacts"]
    if not isinstance(artifacts, list):
        raise RuntimeError(f"GPU AOT manifest artifacts are invalid: {manifest_path}")

    functions: list[str] = []
    for index, record in enumerate(artifacts):
        if not isinstance(record, dict):
            raise RuntimeError(
                f"GPU AOT manifest artifact {index} is invalid: {manifest_path}"
            )
        function_name = record.get("function")
        if not isinstance(function_name, str) or not function_name:
            raise RuntimeError(
                f"GPU AOT manifest artifact {index} has no function: {manifest_path}"
            )
        functions.append(function_name)

    modules: set[str] = set()
    for record in _runtime_warmup_records(root):
        module_name = str(record["module"])
        prefix = module_name + "."
        if any(name == module_name or name.startswith(prefix) for name in functions):
            modules.add(module_name)
    return frozenset(modules)


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


def _type_source(component_type: type) -> str:
    try:
        source = inspect.getsourcefile(component_type) or inspect.getfile(component_type) or ""
    except (OSError, TypeError):
        source = ""
    return _canonical_source(source, component_type.__module__) if source else ""


def _canonical_source(source: str, module_name: str) -> str:
    """Map stale authored absolute paths to the active project copy."""
    candidate = lexical_path(source) if source else ""
    if candidate and os.path.isfile(candidate):
        return candidate
    try:
        from Infernux.engine.project_context import get_project_root

        root = get_project_root()
    except Exception:
        root = ""
    if root and module_name and not module_name.startswith("Infernux."):
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
    components: Iterable[object], *, scope: str, project_path: str | None = None
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
        ready = hook()
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


def run_project_script_warmups(*, project_path: str | None, scope: str) -> int:
    """Discover warmups from the host's explicit Editor/Player contract.

    An Editor project with no scripts has no script warmups. A synchronous
    Player host uses its build-authored registry regardless of source files.
    Native Players use ``create_player_startup_warmup`` for scheduled work.
    Scene-bound GPU components return ``False`` until their buffers exist
    and are warmed by scene activation.
    """
    if not project_path:
        return 0
    from Infernux.application import Application

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
            sorted(scripts_root.rglob("*.py"), key=lambda item: item.as_posix().casefold())
            if scripts_root.is_dir() else []
        )
        for path in source_paths:
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
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
    from Infernux.components.registry import get_all_types
    from Infernux.components.script_loader import load_all_components_from_file

    components: list[object] = []
    for path in paths:
        source = lexical_path(path)
        types = tuple(
            component_type
            for component_type in get_all_types().values()
            if _type_source(component_type) == source
        )
        if not types:
            types = tuple(load_all_components_from_file(path, register=True))
        for component_type in types:
            component = component_type()
            components.append(component)
    return run_component_warmups(components, scope=scope, project_path=project_path)


def run_script_warmup(
    file_path: str, *, project_path: str | None, scope: str
) -> int:
    """Warm one saved script immediately after its durable commit."""
    source = lexical_path(file_path)
    from Infernux.components.component import InxComponent
    from Infernux.components.registry import get_all_types
    from Infernux.components.script_loader import load_all_components_from_file

    types = tuple(
        component_type
        for component_type in get_all_types().values()
        if _type_source(component_type) == source
    )
    if not types and os.path.isfile(source):
        types = tuple(load_all_components_from_file(source, register=True))
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
