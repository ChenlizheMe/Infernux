"""
GameBuilder — packages a standalone native game from an Infernux project.

Uses **Nuitka** to compile the Python entry script into a native EXE.
All engine code, dependencies, and the CPython runtime are bundled into
a self-contained directory.  User scripts (.py in Assets/) are compiled
to .pyc with ``py_compile`` for source protection.

Output layout::

    <OutputDir>/
        <GameName>.exe          ← Nuitka-compiled native executable
        python312.dll           ← CPython runtime (required by Nuitka)
        SDL3.dll, imgui.dll … ← engine native DLLs (also in Infernux/lib/)
        Infernux/              ← engine package
            lib/
                _Infernux.*.pyd ← pybind11 extension module
                SDL3.dll …       ← DLLs (for os.add_dll_directory)
        Data/
            Assets/             ← game scenes, scripts(.pyc), textures, models
            ProjectSettings/    ← build & tag-layer settings
            materials/
            Splash/             ← splash images + .infsplash video data
            BuildManifest.json  ← display mode, window size, splash config
"""

from __future__ import annotations

import json
import os
import py_compile
import shutil
import struct
import subprocess
import sys
import threading
import time
from typing import Callable, Dict, List, Optional

import Infernux._jit_kernels as _jit_kernels
from Infernux.debug import Debug
from Infernux.engine.i18n import t
from Infernux.engine.nuitka_builder import NuitkaBuilder


def _ensure_video_splash_packages() -> None:
    try:
        import imageio.v3  # noqa: F401
        import av  # noqa: F401
        return
    except ImportError:
        Debug.log_internal(
            "Video splash dependencies missing — installing imageio and av automatically..."
        )
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "imageio", "av", "--quiet"],
        )

    import imageio.v3  # noqa: F401
    import av  # noqa: F401


class _BuildCancelled(Exception):
    """Raised when the user cancels the build."""


class BuildOutputDirectoryError(ValueError):
    """Raised when the chosen build output directory is unsafe to reuse."""

    def __init__(
        self,
        reason: str,
        path: str,
        *,
        marker_filename: str,
        entries: Optional[list[str]] = None,
    ):
        self.reason = reason
        self.path = path
        self.marker_filename = marker_filename
        self.entries = list(entries or [])

        if reason == "required":
            message = "Output directory is required."
        elif reason == "path-is-file":
            message = f"Output path is a file, not a directory: {path}"
        elif reason == "path-not-directory":
            message = f"Output path is not a directory: {path}"
        else:
            preview = ", ".join(self.entries[:5])
            if len(self.entries) > 5:
                preview += ", ..."
            message = (
                "Output directory must be empty before building, unless it already contains "
                f"{marker_filename} from a previous Infernux build.\n"
                f"Directory: {path}"
            )
            if preview:
                message += f"\nFound: {preview}"

        super().__init__(message)


class GameBuilder:
    """Build a standalone native game distribution using Nuitka."""

    OUTPUT_MARKER_FILENAME = ".infernux-build-output"
    _GAME_DATA_DIRS = ["Assets", "ProjectSettings", "materials"]
    _EXCLUDE_PATTERNS = {"__pycache__", ".git", ".gitignore", ".infernux-engine-lock.json"}
    _ICON_EXTS = {".png", ".jpg", ".jpeg", ".ico"}

    def __init__(
        self,
        project_path: str,
        output_dir: str,
        *,
        game_name: str = "",
        icon_path: Optional[str] = None,
        display_mode: str = "fullscreen_borderless",
        window_width: int = 1280,
        window_height: int = 720,
        window_resizable: bool = True,
        splash_items: Optional[List[Dict]] = None,
        debug_mode: bool = False,
        lto: bool = True,
    ):
        self.project_path = os.path.abspath(project_path)
        self.project_name = game_name.strip() if game_name.strip() else os.path.basename(self.project_path)
        self.output_dir = os.path.abspath(output_dir)
        self.icon_path = os.path.abspath(icon_path) if icon_path else ""
        self.display_mode = display_mode
        self.window_width = window_width
        self.window_height = window_height
        self.window_resizable = window_resizable
        self.splash_items = list(splash_items) if splash_items else []
        self.debug_mode = debug_mode
        self.lto = lto

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        on_progress: Optional[Callable[[str, float], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> str:
        """Run the full build pipeline.  Returns the final output directory."""

        build_start = time.perf_counter()
        _stage_t0 = build_start

        # ── Build log file ────────────────────────────────────────────
        log_dir = os.path.join(self.project_path, "Logs")
        os.makedirs(log_dir, exist_ok=True)
        build_log_path = os.path.join(log_dir, "build.log")
        build_log = open(build_log_path, "w", encoding="utf-8")

        def _blog(msg: str):
            """Write to both the engine console and the build log file."""
            try:
                build_log.write(msg + "\n")
                build_log.flush()
            except OSError:
                pass

        def _p(msg: str, pct: float):
            nonlocal _stage_t0
            if cancel_event is not None and cancel_event.is_set():
                raise _BuildCancelled()
            now = time.perf_counter()
            elapsed = now - _stage_t0
            _stage_t0 = now
            if on_progress:
                on_progress(msg, pct)
            log_msg = (
                f"[Build {pct:.0%}] {msg}  (prev stage {elapsed:.2f}s, "
                f"total {now - build_start:.1f}s)"
            )
            Debug.log_internal(log_msg)
            _blog(log_msg)

        try:
            return self._build_inner(_p, _blog, on_progress, cancel_event, build_start)
        except _BuildCancelled:
            _blog("Build cancelled by user.")
            raise
        except Exception as exc:
            import traceback
            tb = traceback.format_exc()
            _blog(f"BUILD FAILED: {tb}")
            Debug.log_error(
                f"Build failed — see {build_log_path} for details.\n{exc}"
            )
            raise
        finally:
            try:
                build_log.close()
            except OSError:
                pass

    def _build_inner(self, _p, _blog, on_progress, cancel_event, build_start) -> str:
        """Internal build pipeline (separated for clean exception handling)."""

        _p(t("build.step.validating"), 0.00)
        self._validate()

        _p(t("build.step.cleaning_output"), 0.02)
        self._clean_output()

        _p(t("build.step.collecting_deps"), 0.04)
        user_packages = self._collect_user_dependencies()

        _p(t("build.step.generating_boot"), 0.05)
        boot_script = self._generate_boot_script()

        _p(t("build.step.nuitka_compilation"), 0.06)
        dist_dir = self._run_nuitka(boot_script, on_progress, user_packages, cancel_event)

        _p(t("build.step.organizing_output"), 0.86)
        final_dir = self._organize_output(dist_dir)

        _p(t("build.step.copying_data"), 0.88)
        self._copy_game_data(final_dir)

        _p(t("build.step.compiling_scripts"), 0.91)
        self._compile_user_scripts(final_dir)

        _p(t("build.step.processing_splash"), 0.93)
        self._process_splash_items(final_dir)

        _p(t("build.step.fixing_scenes"), 0.96)
        self._relativize_scenes(final_dir)

        _p(t("build.step.generating_manifest"), 0.97)
        self._generate_manifest(final_dir)

        _p(t("build.step.cleaning_redundant"), 0.98)
        self._cleanup_dist(final_dir)

        _p(t("build.step.writing_marker"), 0.985)
        self._write_output_marker(final_dir)

        _p(t("build.step.cleaning_temp"), 0.99)
        self._cleanup_temp(boot_script)

        # Log per-directory size breakdown so the user sees where size goes
        self._report_build_size(final_dir, _blog)

        _p(t("build.step.complete"), 1.0)
        elapsed_seconds = time.perf_counter() - build_start
        done_msg = t("build.completed_log").format(
            path=final_dir,
            seconds=elapsed_seconds,
        )
        Debug.log(done_msg)
        _blog(done_msg)
        return final_dir

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(self):
        bs = os.path.join(
            self.project_path, "ProjectSettings", "BuildSettings.json"
        )
        if not os.path.isfile(bs):
            raise FileNotFoundError(
                "BuildSettings.json not found. "
                "Open Build Settings in the editor and add at least one scene."
            )
        with open(bs, "r", encoding="utf-8", errors="replace") as f:
            data = json.load(f)
        scenes = data.get("scenes", [])
        if not scenes:
            raise ValueError(
                "Build list is empty. Add at least one scene in Build Settings."
            )
        missing = [s for s in scenes if not os.path.isfile(s)]
        if missing:
            names = ", ".join(os.path.basename(m) for m in missing)
            raise FileNotFoundError(f"Scene file(s) not found: {names}")

        if self.icon_path:
            if not os.path.isfile(self.icon_path):
                raise FileNotFoundError(f"Build icon not found: {self.icon_path}")
            ext = os.path.splitext(self.icon_path)[1].lower()
            if ext not in self._ICON_EXTS:
                raise ValueError(
                    "Build icon must be a .png, .jpg, .jpeg, or .ico file."
                )

        self._validate_output_directory()

    def _output_marker_path(self, directory: Optional[str] = None) -> str:
        target_dir = os.path.abspath(directory or self.output_dir)
        return os.path.join(target_dir, self.OUTPUT_MARKER_FILENAME)

    def _validate_output_directory(self) -> None:
        if not self.output_dir:
            raise BuildOutputDirectoryError(
                "required",
                self.output_dir,
                marker_filename=self.OUTPUT_MARKER_FILENAME,
            )

        if os.path.isfile(self.output_dir):
            raise BuildOutputDirectoryError(
                "path-is-file",
                self.output_dir,
                marker_filename=self.OUTPUT_MARKER_FILENAME,
            )

        if not os.path.exists(self.output_dir):
            return

        if not os.path.isdir(self.output_dir):
            raise BuildOutputDirectoryError(
                "path-not-directory",
                self.output_dir,
                marker_filename=self.OUTPUT_MARKER_FILENAME,
            )

        entries = [entry.name for entry in os.scandir(self.output_dir)]
        if not entries:
            return

        marker_path = self._output_marker_path(self.output_dir)
        if os.path.isfile(marker_path):
            return

        raise BuildOutputDirectoryError(
            "not-empty-unmarked",
            self.output_dir,
            marker_filename=self.OUTPUT_MARKER_FILENAME,
            entries=sorted(entries),
        )

    # ------------------------------------------------------------------
    # Clean output
    # ------------------------------------------------------------------

    def _clean_output(self):
        os.makedirs(self.output_dir, exist_ok=True)
        self._validate_output_directory()

        for name in os.listdir(self.output_dir):
            path = os.path.join(self.output_dir, name)
            if os.path.isdir(path) and not os.path.islink(path):
                if sys.platform == "win32":
                    subprocess.run(
                        ["cmd", "/c", "rd", "/s", "/q", path],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                else:
                    shutil.rmtree(path, ignore_errors=True)
            else:
                try:
                    os.remove(path)
                except FileNotFoundError:
                    continue

            if os.path.exists(path):
                raise OSError(f"Failed to clean output path: {path}")

    def _write_output_marker(self, final_dir: str) -> None:
        marker_path = self._output_marker_path(final_dir)
        marker_payload = {
            "tool": "Infernux",
            "kind": "build-output",
            "project_name": self.project_name,
            "project_path": self.project_path,
            "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        with open(marker_path, "w", encoding="utf-8") as f:
            json.dump(marker_payload, f, indent=2, ensure_ascii=False)
            f.write("\n")

    # ------------------------------------------------------------------
    # Generate boot script (temporary, fed to Nuitka)
    # ------------------------------------------------------------------

    def _generate_boot_script(self) -> str:
        """Generate the entry script that Nuitka will compile into the EXE.

        Returns the path to the temporary boot script.
        """
        _debug_mode = self.debug_mode
        _log_level_str = "LogLevel.Debug" if _debug_mode else "LogLevel.Info"

        boot_src = f'''\
"""Infernux Game — compiled entry point."""
import os
import sys
import traceback

# Activate player mode BEFORE any Infernux imports so the engine
# package skips heavy editor-only UI panels and watchdog file watcher.
os.environ["_INFERNUX_PLAYER_MODE"] = "1"

_DEBUG_MODE = {_debug_mode!r}

# Determine the directory containing the executable
_DIR = os.path.dirname(os.path.abspath(sys.argv[0]))
if not os.path.isdir(os.path.join(_DIR, "Data")):
    _DIR = os.path.dirname(os.path.abspath(sys.executable))

# Ensure raw-copied JIT packages (numba, numpy, llvmlite) are importable.
# Nuitka standalone may not include the exe directory in sys.path by default.
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

# On Windows, add the exe directory as a DLL search path so that
# native extensions inside raw-copied packages can find their .dll deps.
if sys.platform == 'win32':
    try:
        os.add_dll_directory(_DIR)
    except OSError:
        pass
    # Pre-load bundled MSVC CRT DLLs so the dynamic linker can resolve
    # them even on machines without Visual C++ Redistributable installed.
    import ctypes as _ctypes
    for _crt in ('vcruntime140.dll', 'vcruntime140_1.dll',
                 'msvcp140.dll', 'msvcp140_1.dll', 'msvcp140_2.dll',
                 'msvcp140_atomic_wait.dll', 'msvcp140_codecvt_ids.dll',
                 'concrt140.dll'):
        _crt_path = os.path.join(_DIR, _crt)
        if os.path.isfile(_crt_path):
            try:
                _ctypes.WinDLL(_crt_path)
            except OSError:
                pass
    del _ctypes

# Logs go into Data/Logs/ to keep the root directory clean
_LOGS_DIR = os.path.join(_DIR, "Data", "Logs")
os.makedirs(_LOGS_DIR, exist_ok=True)
_LOG = os.path.join(_LOGS_DIR, "player.log")
os.environ["_INFERNUX_PLAYER_LOG"] = _LOG

# Debug mode: write a detailed log next to the executable
if _DEBUG_MODE:
    _DEBUG_LOG = os.path.join(_DIR, "{self.project_name}_debug.log")
    _debug_fh = open(_DEBUG_LOG, "w", encoding="utf-8")
    sys.stdout = _debug_fh
    sys.stderr = _debug_fh

# Clear previous log
try:
    open(_LOG, "w", encoding="utf-8").close()
except OSError:
    pass

def _log(msg):
    try:
        with open(_LOG, "a", encoding="utf-8") as _f:
            _f.write(str(msg) + "\\n")
    except OSError:
        pass
    if _DEBUG_MODE:
        print(msg, flush=True)

def _crash_report(exc):
    """Write crash details to a log file and show a Windows message box."""
    tb_text = traceback.format_exc()
    _log("CRASH: " + tb_text)
    log_path = os.path.join(_LOGS_DIR, "crash.log")
    try:
        with open(log_path, "w", encoding="utf-8") as _f:
            _f.write(tb_text)
    except OSError:
        pass
    # Try to show a native message box (works even without console)
    try:
        import ctypes
        msg = f"Failed to start.  Details in crash.log\\n\\n" + tb_text[-800:]
        ctypes.windll.user32.MessageBoxW(0, msg, "Infernux Error", 0x10)
    except Exception:
        pass

try:
    _log("boot: importing run_player")
    from Infernux.engine import run_player
    from Infernux.lib import LogLevel

    _log("boot: calling run_player")
    run_player(
        project_path=os.path.join(_DIR, "Data"),
        engine_log_level={_log_level_str},
    )
    _log("boot: run_player returned")
except Exception as _exc:
    _crash_report(_exc)
    sys.exit(1)
finally:
    if _DEBUG_MODE:
        try:
            _debug_fh.close()
        except Exception:
            pass
'''
        # Write boot script to a temp location (NuitkaBuilder will copy
        # it into its ASCII-safe staging directory).
        boot_dir = os.path.join(self.output_dir, "_build_temp")
        os.makedirs(boot_dir, exist_ok=True)
        boot_path = os.path.join(boot_dir, "boot.py")
        with open(boot_path, "w", encoding="utf-8") as f:
            f.write(boot_src)
        return boot_path

    # ------------------------------------------------------------------
    # Nuitka compilation
    # ------------------------------------------------------------------

    def _run_nuitka(
        self,
        boot_script: str,
        on_progress: Optional[Callable[[str, float], None]],
        user_packages: Optional[List[str]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> str:
        """Invoke NuitkaBuilder. Returns the dist directory path."""
        from Infernux.resources import icon_path

        selected_icon = self.icon_path if self.icon_path else icon_path

        # Separate JIT-related packages that must be raw-copied (not
        # compiled by Nuitka) from ordinary packages that Nuitka should
        # compile normally.
        jit_set = NuitkaBuilder._JIT_NOFOLLOW_PACKAGES
        all_pkgs = user_packages or []
        compiled_pkgs = [p for p in all_pkgs if p not in jit_set]
        jit_pkgs = [p for p in all_pkgs if p in jit_set]

        nk = NuitkaBuilder(
            entry_script=boot_script,
            output_dir=self.output_dir,
            output_filename=f"{self.project_name}.exe",
            product_name=self.project_name,
            icon_path=selected_icon if selected_icon and os.path.isfile(selected_icon) else None,
            extra_include_packages=compiled_pkgs,
            extra_requirements_files=self._project_requirement_files(),
            raw_copy_packages=jit_pkgs,
            console_mode="force" if self.debug_mode else "disable",
            lto=self.lto,
        )

        def _nk_progress(msg: str, pct: float):
            # Map Nuitka's 0–1 range into our 0.06–0.85 range
            mapped = 0.06 + pct * 0.79
            if on_progress:
                on_progress(msg, mapped)

        return nk.build(on_progress=_nk_progress, cancel_event=cancel_event)

    def _project_requirement_files(self) -> List[str]:
        req_path = os.path.join(self.project_path, "requirements.txt")
        if os.path.isfile(req_path):
            return [req_path]
        return []

    # ------------------------------------------------------------------
    # Organize output: move dist contents to the final output directory
    # ------------------------------------------------------------------

    def _organize_output(self, dist_dir: str) -> str:
        """Move Nuitka dist contents from staging into self.output_dir.

        The dist_dir lives in an ASCII-safe staging area (e.g.
        ``C:\\_InxBuild\\<hash>\\boot.dist``).  We move every item
        into the user's chosen output directory.
        Returns the final directory path.
        """
        final_dir = self.output_dir
        os.makedirs(final_dir, exist_ok=True)

        _move_t0 = time.perf_counter()

        if sys.platform == "win32":
            # robocopy /MOVE /E is dramatically faster than per-item
            # shutil.move for large directory trees (native NTFS ops).
            rc = subprocess.call(
                ["robocopy", dist_dir, final_dir, "/E", "/MOVE",
                 "/NFL", "/NDL", "/NJH", "/NJS", "/NP"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=0x08000000,
            )
            if rc >= 8:
                Debug.log_warning(
                    f"robocopy /MOVE failed (exit {rc}), falling back to Python move"
                )
                for item in os.listdir(dist_dir):
                    src = os.path.join(dist_dir, item)
                    dst = os.path.join(final_dir, item)
                    if os.path.exists(dst):
                        if os.path.isdir(dst):
                            shutil.rmtree(dst)
                        else:
                            os.remove(dst)
                    shutil.move(src, dst)
        else:
            for item in os.listdir(dist_dir):
                src = os.path.join(dist_dir, item)
                dst = os.path.join(final_dir, item)
                if os.path.exists(dst):
                    if os.path.isdir(dst):
                        shutil.rmtree(dst)
                    else:
                        os.remove(dst)
                shutil.move(src, dst)

        Debug.log_internal(
            f"  moved dist to output in {time.perf_counter() - _move_t0:.2f}s"
        )

        # Remove the now-empty staging parent
        staging_parent = os.path.dirname(dist_dir)
        if sys.platform == "win32":
            subprocess.run(
                ["cmd", "/c", "rd", "/s", "/q", staging_parent],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            shutil.rmtree(staging_parent, ignore_errors=True)

        return final_dir

    # ------------------------------------------------------------------
    # Game data
    # ------------------------------------------------------------------

    def _copy_game_data(self, final_dir: str):
        """Copy Assets, ProjectSettings, materials to Data/."""
        data_dir = os.path.join(final_dir, "Data")
        ignore = shutil.ignore_patterns(*self._EXCLUDE_PATTERNS)
        for dirname in self._GAME_DATA_DIRS:
            src = os.path.join(self.project_path, dirname)
            dst = os.path.join(data_dir, dirname)
            if os.path.isdir(src):
                _t0 = time.perf_counter()
                if sys.platform == "win32":
                    os.makedirs(dst, exist_ok=True)
                    rc = subprocess.call(
                        ["robocopy", src, dst, "/E",
                         "/NFL", "/NDL", "/NJH", "/NJS", "/NP",
                         "/XD", "__pycache__", ".git"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        creationflags=0x08000000,
                    )
                    if rc >= 8:
                        Debug.log_warning(
                            f"robocopy failed for {dirname}/ (exit {rc}), "
                            f"falling back to shutil.copytree"
                        )
                        if os.path.isdir(dst):
                            shutil.rmtree(dst)
                        shutil.copytree(src, dst, ignore=ignore)
                else:
                    shutil.copytree(src, dst, ignore=ignore)
                Debug.log_internal(
                    f"  copied {dirname}/ in {time.perf_counter() - _t0:.2f}s"
                )

    # ------------------------------------------------------------------
    # Collect user script dependencies
    # ------------------------------------------------------------------

    # Packages that are already bundled by the engine or excluded on
    # purpose — never add them via --include-package even if a user
    # script imports them.
    _BUILTIN_MODULES = frozenset({
        # Standard library (always available in the Nuitka bundle)
        *sys.stdlib_module_names,
        # Engine packages (already followed by Nuitka via boot.py)
        "Infernux",
        # Excluded editor-only / build-only packages
        "watchdog", "PIL", "cv2", "imageio", "psd_tools",
        "tkinter", "unittest", "test", "pip", "setuptools",
        "distutils", "ensurepip",
    })

    def _collect_internal_asset_module_names(self) -> set[str]:
        """Return top-level module names that belong to the project's Assets tree."""
        names: set[str] = {"Assets"}
        assets_dir = os.path.join(self.project_path, "Assets")
        if not os.path.isdir(assets_dir):
            return names

        for entry in os.scandir(assets_dir):
            name = entry.name
            if name.startswith(".") or name in {"__pycache__"}:
                continue
            if entry.is_dir():
                names.add(name)
                continue
            stem, ext = os.path.splitext(name)
            if ext in {".py", ".pyc"} and stem and not stem.startswith("_"):
                names.add(stem)
        return names

    def _collect_user_dependencies(self) -> List[str]:
        """Scan user scripts for third-party imports and return package names.

        Detection sources (in order of priority):
        1. ``requirements.txt`` in the project root — explicit user list.
           Lines starting with ``#`` or empty lines are ignored.
           Version specifiers are stripped (``torch>=2.0`` → ``torch``).
        2. AST-based import scanning of all ``.py`` files under ``Assets/``.
           Only top-level package names are collected (``import a.b`` → ``a``).

        The results are de-duplicated, stdlib/engine names are filtered out,
        and only packages actually installed in the current environment are
        returned (to avoid Nuitka errors on typos or conditional imports).
        """
        import ast
        import importlib.util
        import re

        found: set[str] = set()
        uses_infernux_jit = False
        _t0 = time.perf_counter()

        # --- Source 1: project requirements.txt -------------------------
        req_path = os.path.join(self.project_path, "requirements.txt")
        if os.path.isfile(req_path):
            Debug.log_internal(f"Found project requirements.txt: {req_path}")
            with open(req_path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or line.startswith("-"):
                        continue
                    # Strip version specifiers: "torch>=2.0" → "torch"
                    pkg = re.split(r"[><=!;\[]", line, maxsplit=1)[0].strip()
                    if pkg:
                        found.add(pkg)
        Debug.log_internal(
            f"  requirements.txt parsed in {time.perf_counter() - _t0:.3f}s"
        )

        # --- Source 2: AST import scanning ------------------------------
        _ast_t0 = time.perf_counter()
        _ast_file_count = 0
        assets_dir = os.path.join(self.project_path, "Assets")
        if os.path.isdir(assets_dir):
            for root, _, files in os.walk(assets_dir):
                for fname in files:
                    if not fname.endswith(".py"):
                        continue
                    fpath = os.path.join(root, fname)
                    _ast_file_count += 1
                    try:
                        with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                            tree = ast.parse(f.read(), filename=fpath)
                    except SyntaxError:
                        continue
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Import):
                            for alias in node.names:
                                found.add(alias.name.split(".")[0])
                                if alias.name in {"Infernux.jit", "Infernux._jit_kernels"}:
                                    uses_infernux_jit = True
                        elif isinstance(node, ast.ImportFrom):
                            if node.module and node.level == 0:
                                found.add(node.module.split(".")[0])
                                if node.module in {"Infernux.jit", "Infernux._jit_kernels"}:
                                    uses_infernux_jit = True
                                elif node.module == "Infernux":
                                    imported_names = {alias.name for alias in node.names}
                                    if imported_names & {
                                        "jit", "njit", "warmup", "precompile_jit",
                                        "JIT_AVAILABLE",
                                    }:
                                        uses_infernux_jit = True
        Debug.log_internal(
            f"  AST scanned {_ast_file_count} .py files in "
            f"{time.perf_counter() - _ast_t0:.3f}s"
        )

        # --- Filter: remove stdlib / engine / excluded ------------------
        found -= self._BUILTIN_MODULES
        found -= self._collect_internal_asset_module_names()

        # Public JIT API ultimately depends on numba + llvmlite + numpy.
        # Make that explicit so standalone player builds include the runtime
        # pieces even when user scripts import the supported ``Infernux.jit``
        # surface instead of importing ``numba`` directly.  numpy must also
        # be raw-copied because numba introspects numpy bytecode at JIT time.
        if uses_infernux_jit or "numba" in found:
            found.add("numba")
            found.add("llvmlite")
            found.add("numpy")

        # Only keep packages that are actually importable in the current
        # environment so Nuitka doesn't error on stale or optional imports.
        _verify_t0 = time.perf_counter()
        verified: list[str] = []
        for pkg in sorted(found):
            if importlib.util.find_spec(pkg) is not None:
                verified.append(pkg)
            else:
                Debug.log_warning(
                    f"User script dependency '{pkg}' not installed — skipping"
                )
        Debug.log_internal(
            f"  import verification in {time.perf_counter() - _verify_t0:.3f}s"
        )

        if verified:
            Debug.log_internal(
                f"User dependencies to bundle: {', '.join(verified)}"
            )
        return verified

    # ------------------------------------------------------------------
    # Compile user scripts
    # ------------------------------------------------------------------

    def _compile_user_scripts(self, final_dir: str):
        """Compile .py in Data/Assets/ to .pyc and remove originals.

        Also generates ``Data/_script_guid_map.json`` so that the
        player can resolve script GUIDs without the original ``.py``
        files (the C++ AssetDatabase only recognises ``.py``).
        """
        assets_dir = os.path.join(final_dir, "Data", "Assets")
        if not os.path.isdir(assets_dir):
            return

        _compile_t0 = time.perf_counter()
        _compile_count = 0
        data_dir = os.path.join(final_dir, "Data")
        guid_map: dict[str, str] = {}

        # First pass: build GUID → .pyc relative-path map from .meta
        for root, _dirs, files in os.walk(assets_dir):
            for fname in files:
                if fname.endswith(".py"):
                    py_path = os.path.join(root, fname)
                    meta_path = py_path + ".meta"
                    if os.path.isfile(meta_path):
                        try:
                            with open(meta_path, "r", encoding="utf-8") as mf:
                                meta = json.load(mf)
                            guid = (meta.get("metadata", {})
                                        .get("guid", {})
                                        .get("value", ""))
                            if guid:
                                pyc_rel = os.path.relpath(
                                    py_path + "c", data_dir
                                ).replace("\\", "/")
                                guid_map[guid] = pyc_rel
                        except (json.JSONDecodeError, OSError):
                            pass

        # Second pass: compile and remove originals
        for root, _dirs, files in os.walk(assets_dir):
            for fname in files:
                if fname.endswith(".py"):
                    py_path = os.path.join(root, fname)
                    _compile_count += 1
                    try:
                        with open(py_path, "r", encoding="utf-8") as sf:
                            source_text = sf.read()
                        sidecar_source = _jit_kernels.build_auto_parallel_sidecar_source(source_text)
                        if sidecar_source:
                            sidecar_py = py_path[:-3] + ".autop.py"
                            with open(sidecar_py, "w", encoding="utf-8", newline="\n") as apf:
                                apf.write(sidecar_source)
                            py_compile.compile(
                                sidecar_py,
                                cfile=sidecar_py + "c",
                                optimize=2,
                                doraise=True,
                            )
                            os.remove(sidecar_py)
                            Debug.log_internal(
                                f"  auto_parallel sidecar: {os.path.basename(sidecar_py)}c"
                            )
                    except (OSError, SyntaxError, py_compile.PyCompileError) as _sc_exc:
                        Debug.log_warning(
                            f"  auto_parallel sidecar generation failed for "
                            f"{fname}: {_sc_exc}"
                        )

                    try:
                        py_compile.compile(
                            py_path,
                            cfile=py_path + "c",
                            optimize=2,
                            doraise=True,
                        )
                        os.remove(py_path)
                    except py_compile.PyCompileError:
                        pass

        Debug.log_internal(
            f"  compiled {_compile_count} scripts in "
            f"{time.perf_counter() - _compile_t0:.2f}s"
        )

        # Write manifest
        if guid_map:
            manifest_path = os.path.join(data_dir, "_script_guid_map.json")
            with open(manifest_path, "w", encoding="utf-8") as mf:
                json.dump(guid_map, mf)

    # ------------------------------------------------------------------
    # Splash items
    # ------------------------------------------------------------------

    def _process_splash_items(self, final_dir: str):
        """Copy/convert splash items into Data/Splash/."""
        if not self.splash_items:
            return

        splash_dir = os.path.join(final_dir, "Data", "Splash")
        os.makedirs(splash_dir, exist_ok=True)

        for item in self.splash_items:
            src_path = item.get("path", "")
            if not os.path.isfile(src_path):
                Debug.log_warning(f"Splash item not found: {src_path}")
                continue

            item_type = item.get("type", "image")
            base_name = os.path.splitext(os.path.basename(src_path))[0]

            if item_type == "video":
                out_name = base_name + ".infsplash"
                out_path = os.path.join(splash_dir, out_name)
                self._extract_video_frames(src_path, out_path)
                item["_built_path"] = f"Splash/{out_name}"
            else:
                ext = os.path.splitext(src_path)[1]
                out_name = base_name + ext
                shutil.copy2(src_path, os.path.join(splash_dir, out_name))
                item["_built_path"] = f"Splash/{out_name}"

    def _extract_video_frames(self, video_path: str, output_path: str):
        """Extract video frames to .infsplash binary blob."""
        _ensure_video_splash_packages()
        self._extract_with_imageio(video_path, output_path)

    def _extract_with_imageio(self, video_path: str, output_path: str):
        """Extract video frames using imageio+av."""
        import imageio.v3 as iio

        frames_data: list[bytes] = []
        width = height = 0
        for frame in iio.imiter(video_path, plugin="pyav"):
            height, width = frame.shape[:2]
            jpeg_bytes = iio.imwrite(
                "<bytes>", frame, extension=".jpg", quality=85
            )
            frames_data.append(jpeg_bytes)

        meta = iio.immeta(video_path, plugin="pyav")
        fps = meta.get("fps", 30.0) or 30.0
        self._write_infsplash(output_path, frames_data, fps, width, height)

    @staticmethod
    def _write_infsplash(
        path: str, frames: list, fps: float, width: int, height: int
    ):
        """Write .infsplash binary (magic + header + index + JPEG data)."""
        with open(path, "wb") as f:
            f.write(b"INFSPLSH")
            f.write(struct.pack("<IfII", len(frames), fps, width, height))
            offset = 0
            for data in frames:
                f.write(struct.pack("<II", offset, len(data)))
                offset += len(data)
            for data in frames:
                f.write(data)

    # ------------------------------------------------------------------
    # Relativize scene paths
    # ------------------------------------------------------------------

    def _relativize_scenes(self, final_dir: str):
        bs = os.path.join(
            final_dir, "Data", "ProjectSettings", "BuildSettings.json"
        )
        if not os.path.isfile(bs):
            return
        with open(bs, "r", encoding="utf-8", errors="replace") as f:
            data = json.load(f)

        scenes = data.get("scenes", [])
        rel_scenes = []
        for scene_path in scenes:
            try:
                rel = os.path.relpath(scene_path, self.project_path)
            except ValueError:
                rel = os.path.basename(scene_path)
            rel_scenes.append(rel.replace("\\", "/"))
        data["scenes"] = rel_scenes

        with open(bs, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Generate BuildManifest.json
    # ------------------------------------------------------------------

    def _generate_manifest(self, final_dir: str):
        """Write BuildManifest.json with display mode, splash config, etc."""
        bs = os.path.join(
            final_dir, "Data", "ProjectSettings", "BuildSettings.json"
        )
        scenes = []
        if os.path.isfile(bs):
            with open(bs, "r", encoding="utf-8", errors="replace") as f:
                scenes = json.load(f).get("scenes", [])

        splash_runtime = []
        for item in self.splash_items:
            built = item.get("_built_path")
            if not built:
                continue
            splash_runtime.append({
                "type": item.get("type", "image"),
                "path": built,
                "duration": item.get("duration", 3.0),
                "fade_in": item.get("fade_in", 0.5),
                "fade_out": item.get("fade_out", 0.5),
            })

        manifest = {
            "game_name": self.project_name,
            "display_mode": self.display_mode,
            "window_width": self.window_width,
            "window_height": self.window_height,
            "window_resizable": self.window_resizable,
            "scenes": scenes,
            "splash_items": splash_runtime,
        }

        manifest_path = os.path.join(final_dir, "Data", "BuildManifest.json")
        os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def _cleanup_dist(self, final_dir: str):
        """Remove editor-only and redundant files from the build output."""
        removed_bytes = 0
        dirs_to_remove: list[str] = []
        files_to_remove: list[str] = []

        def _queue_dir(d: str):
            if os.path.isdir(d):
                dirs_to_remove.append(d)

        def _queue_file(f: str):
            if os.path.isfile(f):
                files_to_remove.append(f)

        # Directories that are entirely unnecessary at runtime
        _queue_dir(os.path.join(final_dir, "Infernux", "lib", "_player_runtime"))
        _queue_dir(os.path.join(final_dir, "Infernux", "resources", "icons"))
        _queue_dir(os.path.join(final_dir, "Infernux", "resources", "supports"))

        # Build-time-only video packages — av (PyAV/ffmpeg) and imageio
        # are used only for splash video encoding at build time.  The
        # player reads pre-extracted .infsplash blobs via struct.
        for _build_pkg in ("av", "av.libs", "imageio"):
            _queue_dir(os.path.join(final_dir, _build_pkg))

        # Remove any leaked ffmpeg DLLs from the dist root that Nuitka's
        # DLL scanner may have copied from the av package.
        _FFMPEG_PREFIXES = (
            "avcodec", "avformat", "avutil", "avfilter", "avdevice",
            "swresample", "swscale",
        )
        for fname in os.listdir(final_dir):
            if fname.lower().endswith(".dll") and any(
                fname.lower().startswith(p) for p in _FFMPEG_PREFIXES
            ):
                _queue_file(os.path.join(final_dir, fname))

        # Individual files not needed at runtime
        _queue_file(os.path.join(final_dir, "Infernux", "lib", "_Infernux.pyi"))
        _queue_file(os.path.join(final_dir, "Infernux", "lib", "InfernuxLauncher.exe"))
        _queue_file(os.path.join(final_dir, "Data", "ProjectSettings", "EditorSettings.json"))
        _queue_file(os.path.join(final_dir, "Data", "ProjectSettings", "GameView.ini"))

        # Remove the platform-tagged .pyd duplicate — Nuitka standardises
        # to the short name (_Infernux.pyd) and --include-package-data
        # copies the original cp312-win_amd64.pyd as well.
        lib_dir_dup = os.path.join(final_dir, "Infernux", "lib")
        if os.path.isdir(lib_dir_dup):
            for fname in os.listdir(lib_dir_dup):
                if fname.endswith(".pyd") and ".cp" in fname:
                    short = fname.split(".")[0] + ".pyd"
                    if os.path.isfile(os.path.join(lib_dir_dup, short)):
                        _queue_file(os.path.join(lib_dir_dup, fname))

        # Remove duplicate engine DLLs from Infernux/lib/ — they already
        # exist in the dist root (placed by Nuitka / _inject_native_libs)
        # and the root copy is what the OS DLL loader finds.  Keep only
        # .pyd files in Infernux/lib/ (needed for relative imports).
        lib_dir = os.path.join(final_dir, "Infernux", "lib")
        if os.path.isdir(lib_dir):
            for fname in os.listdir(lib_dir):
                if fname.lower().endswith(".dll"):
                    _queue_file(os.path.join(lib_dir, fname))

        # Remove .meta files from engine shaders (editor hot-reload metadata)
        shaders_dir = os.path.join(final_dir, "Infernux", "resources", "shaders")
        if os.path.isdir(shaders_dir):
            for root, _, files in os.walk(shaders_dir):
                for fname in files:
                    if fname.endswith(".meta"):
                        _queue_file(os.path.join(root, fname))

        # ── Global cleanup: __pycache__, .dist-info, and stale .pyc ──
        jit_dirs = {os.path.join(final_dir, p) for p in ("numba", "llvmlite", "numpy")}
        for root, dirs, files in os.walk(final_dir, topdown=False):
            for dname in dirs:
                if dname == "__pycache__" or dname.endswith(".dist-info"):
                    dirs_to_remove.append(os.path.join(root, dname))
            # Remove stale .pyc from raw-copied JIT packages
            if any(root == jd or root.startswith(jd + os.sep) for jd in jit_dirs):
                for fname in files:
                    if fname.endswith(".pyc"):
                        files_to_remove.append(os.path.join(root, fname))

        # ── Execute removals ─────────────────────────────────────────
        # 1. Remove individual files (fast, no subprocess)
        for f in files_to_remove:
            try:
                removed_bytes += os.path.getsize(f)
            except OSError:
                pass
            try:
                os.remove(f)
            except OSError:
                pass

        # 2. Count bytes in queued dirs, then batch-remove
        for d in dirs_to_remove:
            if not os.path.isdir(d):
                continue
            for r, _, fs in os.walk(d):
                for fname in fs:
                    try:
                        removed_bytes += os.path.getsize(os.path.join(r, fname))
                    except OSError:
                        pass

        if sys.platform == "win32" and dirs_to_remove:
            # Single cmd process to remove all directories at once
            rd_args = []
            for d in dirs_to_remove:
                if os.path.isdir(d):
                    rd_args.extend(["rd", "/s", "/q", d, "&"])
            if rd_args:
                rd_args.pop()  # remove trailing "&"
                subprocess.run(
                    ["cmd", "/c"] + rd_args,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
        else:
            for d in dirs_to_remove:
                shutil.rmtree(d, ignore_errors=True)

        # Ensure Data/Logs exists for runtime log output
        logs_dir = os.path.join(final_dir, "Data", "Logs")
        os.makedirs(logs_dir, exist_ok=True)

        mb = removed_bytes / (1024 * 1024)
        Debug.log_internal(f"Cleaned {mb:.1f} MB of redundant files from build")

    @staticmethod
    def _cleanup_temp(boot_script: str):
        """Remove the temporary boot script directory.

        Runs in a daemon thread so the caller returns immediately;
        on Windows uses a single ``rd /s /q`` for speed.
        """
        boot_dir = os.path.dirname(boot_script)
        if not os.path.isdir(boot_dir):
            return

        def _bg():
            if sys.platform == "win32":
                subprocess.run(
                    ["cmd", "/c", "rd", "/s", "/q", boot_dir],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            else:
                shutil.rmtree(boot_dir, ignore_errors=True)

        t = threading.Thread(target=_bg, daemon=True)
        t.start()

    # ------------------------------------------------------------------
    # Build size report
    # ------------------------------------------------------------------

    @staticmethod
    def _report_build_size(final_dir: str, _blog: Callable[[str], None]) -> None:
        """Log a per-directory size breakdown of the final build output."""
        total = 0
        entries: list[tuple[str, int]] = []

        for item in os.scandir(final_dir):
            if item.is_dir(follow_symlinks=False):
                sz = 0
                for root, _, files in os.walk(item.path):
                    for f in files:
                        try:
                            sz += os.path.getsize(os.path.join(root, f))
                        except OSError:
                            pass
                entries.append((item.name + "/", sz))
            elif item.is_file(follow_symlinks=False):
                sz = item.stat().st_size
                entries.append((item.name, sz))
            else:
                continue
            total += sz

        entries.sort(key=lambda x: x[1], reverse=True)
        lines = [f"Build size report — total {total / (1024*1024):.1f} MB"]
        for name, sz in entries:
            mb = sz / (1024 * 1024)
            pct = (sz / total * 100) if total else 0
            if mb >= 0.1:
                lines.append(f"  {mb:7.1f} MB  {pct:4.1f}%  {name}")
        report = "\n".join(lines)
        Debug.log_internal(report)
        _blog(report)
