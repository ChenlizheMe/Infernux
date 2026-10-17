"""Infernux Game — compiled entry point."""
import os
import sys
import time

_BOOT_STARTED = time.perf_counter()
_BOOT_PHASES = []

def _mark_boot_phase(_name):
    _BOOT_PHASES.append((_name, time.perf_counter() - _BOOT_STARTED))

# Activate Player mode before importing the engine package.
os.environ["_INFERNUX_PLAYER_MODE"] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True

_PLAYER_ROOT = os.path.dirname(sys.executable)
_GAME_NAME = os.path.splitext(os.path.basename(sys.argv[0]))[0] or "InfernuxPlayer"
_SAFE_GAME_NAME = "".join(
    _ch if _ch not in '<>:"/\\|?*' else '_' for _ch in _GAME_NAME
)
_STATE_HOME = (
    os.environ.get("LOCALAPPDATA", "").strip()
    or os.environ.get("XDG_STATE_HOME", "").strip()
    or os.path.join(os.path.expanduser("~"), ".local", "state")
)
_PLAYER_STATE_ROOT = os.path.join(
    _STATE_HOME, "Infernux", "Players", _SAFE_GAME_NAME
)
# Log locations depend only on the executable name and the user state root, so
# every later startup step — including a missing Data or Runtime directory —
# is reported through crash.log instead of failing silently in a GUI Player.
_LOGS_DIR = os.path.join(_PLAYER_STATE_ROOT, "Logs")
_LOG = os.path.join(_LOGS_DIR, "player.log")
_CRASH_LOG = os.path.join(_LOGS_DIR, "crash.log")
os.environ["_INFERNUX_PLAYER_LOG"] = _LOG

_NATIVE_PACK = None
_DEBUG_MODE = False
_debug_fh = None
# os.add_dll_directory handles must stay alive for the whole process.
_DLL_DIR_HANDLES = []

def _log(_message):
    try:
        with open(_LOG, "a", encoding="utf-8") as _stream:
            _stream.write(str(_message) + "\n")
    except OSError:
        pass
    if _DEBUG_MODE:
        print(_message, flush=True)

def _crash_report(_exc):
    try:
        # The stdlib path is configured during startup; the import may be unavailable.
        _traceback = __import__("traceback").format_exc()
    except Exception:
        _traceback = type(_exc).__name__ + ": " + repr(_exc)
    _log("CRASH: " + _traceback)
    try:
        with open(_CRASH_LOG, "w", encoding="utf-8") as _stream:
            _stream.write(_traceback)
    except OSError:
        pass
    show_error = getattr(_NATIVE_PACK, "_inxplayer_show_error", None)
    if show_error is None or os.environ.get("_INFERNUX_PLAYER_CONTROL_FILE"):
        return
    try:
        show_error(
            "Infernux Error",
            "Failed to start. Details in crash.log\n\n" + _traceback[-800:],
        )
    except Exception:
        pass

def _register_player_dll_directory(_dll_dir):
    if sys.platform != "win32" or not os.path.isdir(_dll_dir):
        return
    try:
        _DLL_DIR_HANDLES.append(os.add_dll_directory(_dll_dir))
    except OSError:
        pass

def _start():
    global _NATIVE_PACK, _DEBUG_MODE, _debug_fh
    os.makedirs(_LOGS_DIR, exist_ok=True)
    if os.path.isfile(_CRASH_LOG):
        os.remove(_CRASH_LOG)

    _DATA_ROOT = os.environ.get("_INFERNUX_PLAYER_DATA_ROOT", "").strip()
    if not _DATA_ROOT:
        raise RuntimeError("The PlayerHost did not provide the Player Data directory.")
    _CORE_RUNTIME_DIR = os.environ.get("_INFERNUX_PLAYER_RUNTIME_ROOT", "").strip()
    if not _CORE_RUNTIME_DIR:
        raise RuntimeError("The PlayerHost did not provide the Player Runtime directory.")
    os.environ["_INFERNUX_PLAYER_DATA_ROOT"] = _DATA_ROOT
    os.environ["_INFERNUX_PLAYER_RUNTIME_ROOT"] = _CORE_RUNTIME_DIR
    if _PLAYER_ROOT not in sys.path:
        sys.path.insert(0, _PLAYER_ROOT)

    try:
        import _InfernuxBootstrap as _native
    except ImportError as _bootstrap_error:
        raise RuntimeError("The native Player bootstrap API is unavailable.") from _bootstrap_error
    if not hasattr(_native, "_inxplayer_show_error"):
        raise RuntimeError("The Player bootstrap is missing API: _inxplayer_show_error")
    _NATIVE_PACK = _native
    _mark_boot_phase("native_bootstrap")

    if not os.path.isdir(_CORE_RUNTIME_DIR):
        raise RuntimeError("The direct Player Runtime directory is missing: " + _CORE_RUNTIME_DIR)
    _mark_boot_phase("runtime_ready")
    _STDLIB_RUNTIME_DIR = os.path.join(_CORE_RUNTIME_DIR, "stdlib")
    _INFERNUX_LIB_DIR = os.path.join(_CORE_RUNTIME_DIR, "infernux", "lib")
    os.environ["INFERNUX_NATIVE_MODULE_DIR"] = _INFERNUX_LIB_DIR
    for _runtime_import_dir in (_CORE_RUNTIME_DIR, _STDLIB_RUNTIME_DIR, _INFERNUX_LIB_DIR):
        if os.path.isdir(_runtime_import_dir) and _runtime_import_dir not in sys.path:
            sys.path.append(_runtime_import_dir)
    os.environ["_INFERNUX_PACKAGED_RESOURCE_ROOT"] = os.path.join(
        _CORE_RUNTIME_DIR, "infernux", "resources"
    )

    _PLAYER_PERSISTENT_DATA_ROOT = os.path.join(_PLAYER_STATE_ROOT, "Data")
    os.makedirs(_PLAYER_PERSISTENT_DATA_ROOT, exist_ok=True)
    os.environ["_INFERNUX_PLAYER_PERSISTENT_DATA_ROOT"] = _PLAYER_PERSISTENT_DATA_ROOT
    from infernux.engine.platform_player_bootstrap import prepare_platform_player
    _DATA_DIR = prepare_platform_player(
        _DATA_ROOT,
        os.path.join(_PLAYER_STATE_ROOT, "Cache"),
    )
    _mark_boot_phase("content_ready")
    _DEBUG_MODE = os.environ["_INFERNUX_PLAYER_DEBUG_BUILD"] == "1"
    if _DEBUG_MODE:
        _debug_fh = open(os.path.join(_LOGS_DIR, _SAFE_GAME_NAME + "_debug.log"), "w", encoding="utf-8")
        sys.stdout = _debug_fh
        sys.stderr = _debug_fh
    _BUILD_MANIFEST_PATH = os.path.join(_DATA_DIR, "BuildManifest.json")
    if not os.path.isfile(_BUILD_MANIFEST_PATH):
        raise RuntimeError(
            "Player package has no BuildManifest.json: " + _BUILD_MANIFEST_PATH
        )

    # Optional CPU/JIT support is shipped as one compressed archive.  Materialize
    # it only on first use into the Player-owned cache; keeping the delivery tree
    # sealed avoids duplicating the archive's expanded payload in every build.
    _RUNTIME_MODULE_DIR = os.path.join(_DATA_ROOT, "Modules", "Parallel")
    _PARALLEL_ARCHIVE = os.path.join(_DATA_ROOT, "Modules", "Parallel.inxmod")
    if os.path.isfile(_PARALLEL_ARCHIVE) and not os.path.isdir(_RUNTIME_MODULE_DIR):
        from infernux.engine.platform_player_bootstrap import _parallel_module_cache
        _RUNTIME_MODULE_DIR = _parallel_module_cache(
            _DATA_ROOT,
            os.path.join(_PLAYER_STATE_ROOT, "Cache"),
        )
    if os.path.isdir(_RUNTIME_MODULE_DIR):
        if _RUNTIME_MODULE_DIR not in sys.path:
            sys.path.insert(0, _RUNTIME_MODULE_DIR)
        for _parallel_dll_dir in (
            _RUNTIME_MODULE_DIR,
            os.path.join(_RUNTIME_MODULE_DIR, "llvmlite", "binding"),
            os.path.join(_RUNTIME_MODULE_DIR, "llvmlite.libs"),
        ):
            _register_player_dll_directory(_parallel_dll_dir)
        _mark_boot_phase("parallel_ready")
    if sys.platform == "win32":
        for _dll_dir in (
            _PLAYER_ROOT,
            _CORE_RUNTIME_DIR,
            _STDLIB_RUNTIME_DIR,
            _INFERNUX_LIB_DIR,
            os.path.join(_CORE_RUNTIME_DIR, "numpy.libs"),
        ):
            _register_player_dll_directory(_dll_dir)

    _log(
        "boot phases: "
        + ", ".join(_name + "=" + format(_elapsed, ".3f") + "s" for _name, _elapsed in _BOOT_PHASES)
    )
    _log("boot: importing run_player")
    import infernux as _public_api
    sys.modules["infernux"] = _public_api
    from infernux.engine import run_player
    from infernux.lib import LogLevel
    _log("boot: imports ready at " + format(time.perf_counter() - _BOOT_STARTED, ".3f") + "s")
    _log("boot: calling run_player")
    run_player(
        project_path=_DATA_DIR,
        engine_log_level=LogLevel.Debug if _DEBUG_MODE else LogLevel.Info,
    )
    _log("boot: run_player returned")

_exit_code = 0
try:
    _start()
except SystemExit as _exc:
    if _exc.code is None:
        _exit_code = 0
    elif isinstance(_exc.code, int):
        # CPython converts through signed 64-bit before narrowing to native int.
        if -(2**63) <= _exc.code < 2**63:
            _exit_code = (_exc.code + 2**31) % 2**32 - 2**31
        else:
            _exit_code = -1
    else:
        _exit_code = 1
        _log(_exc.code)
except BaseException as _exc:
    _exit_code = 1
    if isinstance(_exc, KeyboardInterrupt):
        # Preserve CPython's STATUS_CONTROL_C_EXIT on Windows.
        _exit_code = -1073741510 if sys.platform == "win32" else 130
    _crash_report(_exc)
finally:
    try:
        if _debug_fh is not None:
            _debug_fh.close()
    finally:
        # The Player owns this process, including failed startup. Returning through
        # SystemExit would finalize CPython while native runtime objects remain live.
        os._exit(_exit_code)
