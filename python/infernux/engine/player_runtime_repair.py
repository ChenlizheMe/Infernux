"""Rebuild incompatible platform payloads using the running SDK's source release.

Inspection stays read-only. Export workers call ``prepare_player_runtime`` before
cooking; only a validated replacement overwrites the installed plugin's payload.
Compiler intermediates stay in the project cache; the Python SDK is not replaced.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
from pathlib import Path
import runpy
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager

from infernux.version import ENGINE_RELEASE


def rebuild_diagnostic(error, *, source):
    from .build.contracts import BuildDiagnostic, DiagnosticSeverity

    return BuildDiagnostic(
        DiagnosticSeverity.WARNING, "player.runtime.rebuild-required",
        f"{error}. The Player will be rebuilt from engine {ENGINE_RELEASE} during export.",
        source=source, detail={"runtime_error": str(error), "engine_release": ENGINE_RELEASE},
    )


@contextmanager
def _build_lock(path, request):
    with path.open("a+b") as stream:
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            def acquire():
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            def release():
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            def acquire():
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            def release():
                fcntl.flock(stream, fcntl.LOCK_UN)
        while True:
            request.cancellation.raise_if_cancelled()
            try:
                acquire()
                break
            except (BlockingIOError, PermissionError):
                time.sleep(.1)
        try:
            yield
        finally:
            release()


def _run(command, *, cwd, request, log):
    request.cancellation.raise_if_cancelled()
    request.report("runtime-rebuild", 0, 1, "Rebuilding compatible Player", log=str(log))
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    # Child release tools choose their own staged SDK. Never inherit a developer
    # module override into the payload compiler.
    env.pop("PYTHONPATH", None)
    env.pop("INFERNUX_NATIVE_MODULE_DIR", None)
    with log.open("a", encoding="utf-8") as stream:
        stream.write("\n" + subprocess.list2cmdline([str(x) for x in command]) + "\n")
        stream.flush()
        with subprocess.Popen(command, cwd=cwd, env=env, stdout=stream,
                              stderr=subprocess.STDOUT, start_new_session=os.name != "nt") as process:
            while process.poll() is None:
                if request.cancellation.cancelled:
                    if os.name == "nt":
                        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    else:
                        os.killpg(process.pid, signal.SIGTERM)
                    process.wait()
                    request.cancellation.raise_if_cancelled()
                time.sleep(.1)
            if process.returncode:
                raise RuntimeError(f"Player rebuild failed (exit {process.returncode}). Log: {log}")


def _check_source(root):
    version = root / "python/infernux/version.py"
    if not (root / "CMakeLists.txt").is_file() or not version.is_file():
        raise RuntimeError(f"Player build source is incomplete: {root}")
    actual = runpy.run_path(str(version))["ENGINE_RELEASE"]
    if actual != ENGINE_RELEASE:
        raise RuntimeError(f"Player build source is {actual}; the running engine is {ENGINE_RELEASE}")
    from infernux.plugins.platform_releases import PLAYER_SOURCE_REVISION
    if PLAYER_SOURCE_REVISION is not None:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        if revision != PLAYER_SOURCE_REVISION:
            raise RuntimeError(f"Player source revision {revision} does not match this SDK's {PLAYER_SOURCE_REVISION}")
    return root


def _source_root(cache, request, log):
    configured = os.environ.get("INFERNUX_ENGINE_SOURCE_ROOT", "")
    if configured:
        return _check_source(Path(configured).resolve())
    checkout = Path(__file__).resolve().parents[3]
    if (checkout / "cpp/apps/player/PlayerHost.cpp").is_file():
        return _check_source(checkout)
    provenance = Path(__file__).resolve().parents[1] / "resources/player-build-source.json"
    if provenance.is_file():
        identity = json.loads(provenance.read_text(encoding="utf-8"))
        if identity["engine_release"] != ENGINE_RELEASE:
            raise RuntimeError("Installed SDK source provenance does not match its release")
        if identity.get("dirty"):
            raise RuntimeError("This development SDK contains uncommitted source changes. "
                               "Set INFERNUX_ENGINE_SOURCE_ROOT to its original source checkout before rebuilding Players.")
        revision = identity["revision"]
    else:
        raise RuntimeError("The installed SDK has no Player source provenance")
    source = cache / "source"
    if not (source / ".git").exists():
        _run(["git", "clone", "--no-checkout", "--filter=blob:none",
              "https://github.com/ChenlizheMe/Infernux.git", str(source)],
             cwd=cache, request=request, log=log)
    _run(["git", "fetch", "--depth", "1", "origin", revision],
         cwd=source, request=request, log=log)
    _run(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=source, request=request, log=log)
    _run(["git", "submodule", "update", "--init", "--recursive", "--depth", "1"],
         cwd=source, request=request, log=log)
    return _check_source(source)


def _rebuild(root, destination, build, request, options, log):
    target = str(request.target)
    cmake = shutil.which("cmake")
    if target in {"windows-x64", "linux-x64"}:
        if not cmake:
            raise RuntimeError("CMake is required to rebuild the desktop Player")
        preset = "windows-msvc-release" if target == "windows-x64" else "linux-clang-release"
        _run([cmake, "--preset", preset, "-S", str(root), "-B", str(build),
              f"-DPython3_EXECUTABLE={sys.executable}",
              f"-DINFERNUX_PYTHON_SYNC_DIR={build / 'python-sync'}",
              f"-DINFERNUX_PLATFORM_PLAYER_OUTPUT_DIR={destination}",
              "-DINFERNUX_PLAYER_PACKAGE_PLUGIN=OFF", "-DINFERNUX_BUILD_TESTS=OFF"],
             cwd=root, request=request, log=log)
        _run([cmake, "--build", str(build), "--config", "Release", "--target",
              "prebuild_player_runtime", "--parallel", "4"],
             cwd=root, request=request, log=log)
    elif target in {"android-arm64", "android-x64-emulator"}:
        abi = "x86_64" if target == "android-x64-emulator" else "arm64-v8a"
        _run([sys.executable, str(root / "external/plugins/infernux_android/native/build.py"),
              "--sdk", str(options["sdk_root"]), "--python-prefix", str(options["python_prefix"]),
              "--abi", abi, "--build-root", str(build), "--player-output", str(destination)],
             cwd=root, request=request, log=log)
    elif target == "web-wasm32":
        missing = [key for key in ("emsdk", "cpython_source", "python_runtime_manifest", "numpy_payload_root")
                   if not options.get(key)]
        if missing:
            raise RuntimeError("Web Player rebuild requires its source toolchain: " + ", ".join(missing))
        _run([sys.executable, str(root / "external/plugins/infernux_web/native/build.py"),
              "--emsdk", str(options["emsdk"]), "--cpython-source", str(options["cpython_source"]),
              "--python-runtime-manifest", str(options["python_runtime_manifest"]),
              "--numpy-payload-root", str(options["numpy_payload_root"]),
              "--host-python", sys.executable, "--build-root", str(build),
              "--player-output", str(destination)], cwd=root, request=request, log=log)
    else:
        from infernux.plugins.platform_releases import PLATFORM_PLAYERS
        declaration = next((item for item in PLATFORM_PLAYERS.values() if target in item["targets"]), None)
        if declaration is None or not declaration.get("build_commands"):
            raise ValueError(f"No Player rebuild recipe for {target}")
        context = {"python": sys.executable, "source": str(root), "build_root": str(build),
                   "payload": str(destination), "env": os.environ,
                   **{key: str(value) for key, value in options.items()}}
        for command in declaration["build_commands"]:
            _run([argument.format_map(context) for argument in command], cwd=root, request=request, log=log)


def prepare_player_runtime(request, installed_root, inspect, *, options=None):
    """Repair an already installed plugin and return its compatible payload.

    ``inspect`` validates target-specific files and ABI before either an existing
    or newly compiled payload can be returned. Failures never publish a partial
    replacement. Build logs survive failure and cancellation.
    """
    installed = Path(installed_root)
    try:
        inspect(installed)
        return installed
    except (OSError, ValueError, RuntimeError) as mismatch:
        request.report("runtime-rebuild", 0, 1, str(mismatch))
    if not any((parent / "inx_package.json").is_file() for parent in installed.parents):
        raise RuntimeError("Install the platform plugin before rebuilding its Player")
    identity = f"{ENGINE_RELEASE}/{request.target}/cp{sys.version_info.major}{sys.version_info.minor}/{sys.platform}/{platform.machine()}"
    key = hashlib.sha256(identity.encode()).hexdigest()[:16]
    cache = Path(request.project_root) / "Library/PlayerRuntime" / key
    cache.mkdir(parents=True, exist_ok=True)
    # OS locks release on process exit. Concurrent exports share one compile.
    with _build_lock(installed.parent / ".player-rebuild.lock", request):
        try:
            inspect(installed)
            return installed
        except (OSError, ValueError, RuntimeError):
            pass
        log = cache / "rebuild.log"
        source = _source_root(cache, request, log)
        native_build_root = Path(request.profile.options.get("player_native_build_root") or cache / "build").resolve()
        # Keep publication on the installed plugin's volume, so the final
        # directory renames cannot turn into a cross-volume partial copy.
        work = Path(tempfile.mkdtemp(prefix=".player-candidate-", dir=installed.parent))
        retain_backup = False
        try:
            candidate = work / "player"
            _rebuild(source, candidate, native_build_root, request, options or {}, log)
            inspect(candidate)
            # The installer owns these asset GUIDs. Retain matching sidecars
            # while replacing the generated bytes, so its ledger stays stable.
            if installed.is_dir():
                for metadata in installed.rglob("*.meta"):
                    relative = metadata.relative_to(installed)
                    target = candidate / relative
                    if target.with_suffix("").exists():
                        shutil.copy2(metadata, target)
            request.cancellation.raise_if_cancelled()
            previous = work / "previous"
            if installed.exists():
                installed.rename(previous)
            try:
                candidate.rename(installed)
            except OSError as publish_error:
                if previous.exists():
                    try:
                        previous.rename(installed)
                    except OSError as restore_error:
                        retain_backup = True
                        raise RuntimeError(f"Player payload restore failed; previous payload retained at {previous}: {restore_error}") from publish_error
                raise
            if previous.exists():
                shutil.rmtree(previous)
        finally:
            if not retain_backup:
                shutil.rmtree(work)
        request.report("runtime-rebuild", 1, 1, "Compatible Player rebuilt", log=str(log))
        return installed
