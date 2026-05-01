"""
NuitkaBuilder — compiles a Python entry script into a standalone native EXE
using Nuitka (Python → C → native binary).

This replaces the old RuntimeBuilder cache-copy approach with true native
compilation.  The output is a self-contained directory containing the EXE,
all required DLLs, and the embedded Python runtime.

On Windows, Infernux requires an MSVC toolchain for game builds.
All intermediate compilation is done in an ASCII-safe staging directory and
moved to the final destination afterwards.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Callable, List, Optional

from Infernux.debug import Debug
from Infernux.engine.i18n import t

# ASCII-safe root for Nuitka staging and temporary build artifacts.
_STAGING_ROOT = "C:\\_InxBuild"

# Persistent Nuitka compilation cache — lives outside the per-build staging
# directory so it survives across builds, dramatically speeding up rebuilds.
_NUITKA_CACHE_DIR = os.path.join(_STAGING_ROOT, "_nuitka_cache")

_AUTO_INSTALLABLE_PACKAGES = {
    "nuitka": "nuitka",
    "ordered_set": "ordered-set",
    "PIL": "Pillow",
    "numba": "numba",
    "llvmlite": "llvmlite",
}


class _BuildCancelled(Exception):
    """Raised when the user cancels the build."""


def _has_msvc_toolchain() -> bool:
    if shutil.which("cl"):
        return True

    return bool(_find_msvc_environment_scripts())


def _which_in_env(executable: str, env: dict[str, str]) -> str:
    return shutil.which(executable, path=env.get("PATH", "")) or ""


def _msvc_env_ready(env: dict[str, str]) -> bool:
    return bool(
        _which_in_env("cl.exe", env)
        and env.get("INCLUDE")
        and env.get("LIB")
    )


def _visual_studio_roots_from_vswhere() -> list[str]:
    roots: list[str] = []
    vswhere = os.path.join(
        os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
        "Microsoft Visual Studio",
        "Installer",
        "vswhere.exe",
    )
    if not os.path.isfile(vswhere):
        return roots

    try:
        completed = subprocess.run(
            [
                vswhere,
                "-products",
                "*",
                "-requires",
                "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                "-property",
                "installationPath",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError) as _exc:
        Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
        return roots

    if completed.returncode != 0:
        return roots

    for line in (completed.stdout or "").splitlines():
        root = line.strip()
        if root and os.path.isdir(root):
            roots.append(root)
    return roots


def _visual_studio_roots_from_registry() -> list[str]:
    """Discover Visual Studio installation roots from Windows registry.

    ``vswhere`` is the official and most reliable discovery API, but registry
    fallback matters for non-standard or partially repaired installs where the
    VS Installer utility is missing from its usual location.  The SxS and Setup
    keys are written with the actual installation path, so custom drives are
    covered here.
    """
    if sys.platform != "win32":
        return []

    try:
        import winreg
    except ImportError:
        return []

    roots: list[str] = []
    seen: set[str] = set()

    def _add(path: str) -> None:
        if not path:
            return
        root = os.path.abspath(os.path.expandvars(path.strip().strip('"')))
        if not os.path.isdir(root):
            return
        normalized = os.path.normcase(root)
        if normalized in seen:
            return
        seen.add(normalized)
        roots.append(root)

    hives = (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER)
    views = [0]
    for view_name in ("KEY_WOW64_64KEY", "KEY_WOW64_32KEY"):
        value = getattr(winreg, view_name, 0)
        if value and value not in views:
            views.append(value)

    # VS7 SxS values are named by major version (e.g. "17.0") and point to
    # the VS installation root, even when the user installs on a custom drive.
    sx_s_values: list[tuple[float, str]] = []
    sx_s_keys = (
        r"SOFTWARE\Microsoft\VisualStudio\SxS\VS7",
        r"SOFTWARE\WOW6432Node\Microsoft\VisualStudio\SxS\VS7",
    )
    for hive in hives:
        for access in views:
            for key_name in sx_s_keys:
                try:
                    with winreg.OpenKey(hive, key_name, 0, winreg.KEY_READ | access) as key:
                        index = 0
                        while True:
                            try:
                                name, value, _kind = winreg.EnumValue(key, index)
                            except OSError:
                                break
                            index += 1
                            try:
                                version = float(str(name).split(".", 1)[0])
                            except ValueError:
                                version = 0.0
                            if isinstance(value, str):
                                sx_s_values.append((version, value))
                except OSError:
                    continue

    for _version, path in sorted(sx_s_values, reverse=True):
        _add(path)

    # Newer installers also expose per-instance Setup keys.  These are useful
    # when SxS is absent but the installer registration is intact.
    setup_instance_keys = (
        r"SOFTWARE\Microsoft\VisualStudio\Setup\Instances",
        r"SOFTWARE\WOW6432Node\Microsoft\VisualStudio\Setup\Instances",
    )
    for hive in hives:
        for access in views:
            for key_name in setup_instance_keys:
                try:
                    with winreg.OpenKey(hive, key_name, 0, winreg.KEY_READ | access) as key:
                        index = 0
                        while True:
                            try:
                                subkey_name = winreg.EnumKey(key, index)
                            except OSError:
                                break
                            index += 1
                            try:
                                with winreg.OpenKey(key, subkey_name) as instance_key:
                                    value, _kind = winreg.QueryValueEx(instance_key, "InstallationPath")
                            except OSError:
                                continue
                            if isinstance(value, str):
                                _add(value)
                except OSError:
                    continue

    return roots


def _find_msvc_environment_scripts() -> list[tuple[str, list[str]]]:
    """Return candidate VS environment scripts for x64 MSVC builds."""
    roots: list[str] = []
    roots.extend(_visual_studio_roots_from_vswhere())
    roots.extend(_visual_studio_roots_from_registry())

    for env_name in ("VSINSTALLDIR", "VCINSTALLDIR"):
        root = os.environ.get(env_name, "")
        if env_name == "VCINSTALLDIR" and root:
            root = os.path.abspath(os.path.join(root, "..", ".."))
        if root and os.path.isdir(root):
            roots.append(root)

    program_files = os.environ.get("ProgramFiles", r"C:\Program Files")
    for year in ("2022", "2019"):
        for edition in ("BuildTools", "Community", "Professional", "Enterprise"):
            roots.append(os.path.join(program_files, "Microsoft Visual Studio", year, edition))

    candidates: list[tuple[str, list[str]]] = []
    seen_roots: set[str] = set()
    for root in roots:
        normalized_root = os.path.normcase(os.path.abspath(root))
        if normalized_root in seen_roots:
            continue
        seen_roots.add(normalized_root)

        for script, args in (
            (os.path.join(root, "Common7", "Tools", "VsDevCmd.bat"), ["-arch=x64", "-host_arch=x64"]),
            (os.path.join(root, "VC", "Auxiliary", "Build", "vcvars64.bat"), []),
            (os.path.join(root, "VC", "Auxiliary", "Build", "vcvarsall.bat"), ["x64"]),
        ):
            if os.path.isfile(script):
                candidates.append((script, args))
    return candidates


def _capture_msvc_environment(
    script_path: str,
    args: list[str],
    base_env: dict[str, str],
) -> dict[str, str]:
    env = dict(base_env)
    env["VSCMD_SKIP_SENDTELEMETRY"] = "1"
    quoted_args = " ".join(args)
    temp_root = env.get("TEMP") if os.path.isdir(env.get("TEMP", "")) else tempfile.gettempdir()
    batch_dir = tempfile.mkdtemp(prefix="inx_vcvars_", dir=temp_root)
    batch_path = os.path.join(batch_dir, "capture_env.bat")
    try:
        with open(batch_path, "w", encoding="utf-8", newline="\r\n") as f:
            f.write("@echo off\n")
            f.write(f'call "{script_path}" {quoted_args} >nul\n')
            f.write("if errorlevel 1 exit /b %errorlevel%\n")
            f.write("set\n")

        completed = subprocess.run(
            ["cmd", "/d", "/c", batch_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            timeout=90,
            creationflags=0x08000000,
        )
        if completed.returncode != 0:
            raise RuntimeError((completed.stderr or completed.stdout or "").strip())
    finally:
        shutil.rmtree(batch_dir, ignore_errors=True)

    captured: dict[str, str] = {}
    for line in (completed.stdout or "").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key:
            captured[key] = value
    return captured


def _ensure_windows_msvc_environment(env: dict[str, str]) -> dict[str, str]:
    """Merge a Visual Studio Developer Command Prompt environment.

    End users normally launch Infernux from the Hub or Explorer, not from a
    Developer Command Prompt.  Nuitka's SCons backend needs MSVC/Windows SDK
    variables such as PATH, INCLUDE, LIB, and the cl.exe location; otherwise it
    can fail with the misleading internal message "scons environment variable
    CC is not set" even when Visual Studio is installed.
    """
    if sys.platform != "win32":
        return env

    if _msvc_env_ready(env):
        return env

    failures: list[str] = []
    for script_path, args in _find_msvc_environment_scripts():
        try:
            captured = _capture_msvc_environment(script_path, args, env)
        except Exception as exc:
            failures.append(f"{script_path}: {exc}")
            continue

        merged = dict(env)
        merged.update(captured)
        if _msvc_env_ready(merged):
            merged.setdefault("CC", "cl")
            merged.setdefault("CXX", "cl")
            Debug.log_internal(f"Loaded MSVC build environment from {script_path}")
            return merged

        failures.append(f"{script_path}: cl.exe/INCLUDE/LIB not available after initialization")

    details = "\n".join(failures[-3:])
    raise RuntimeError(
        "Windows game builds require an initialized MSVC + Windows SDK build environment.\n"
        "Visual Studio was detected, but Infernux could not initialize the C++ toolchain "
        "for Nuitka/SCons.\n"
        "Install or repair Visual Studio 2022 with 'Desktop development with C++', "
        "including MSVC v143 and a Windows 10/11 SDK, then try again.\n"
        f"Details:\n{details}"
    )


def _run_python(python_exe: str, args: List[str], *, timeout: int = 60) -> subprocess.CompletedProcess:
    kwargs = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
        "timeout": timeout,
    }
    if sys.platform == "win32":
        kwargs["creationflags"] = 0x08000000
    return subprocess.run([python_exe, *args], **kwargs)


def _python_version(python_exe: str) -> str:
    try:
        completed = _run_python(
            python_exe,
            ["-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError) as _exc:
        Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
        return ""
    if completed.returncode != 0:
        return ""
    return (completed.stdout or "").strip()


def _is_embeddable_python_exe(python_exe: str) -> bool:
    try:
        root = os.path.dirname(os.path.abspath(python_exe))
        return any(name.lower().endswith("._pth") for name in os.listdir(root))
    except OSError as _exc:
        Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
        return False


def _is_valid_builder_python(python_exe: str) -> bool:
    return bool(
        python_exe
        and os.path.isfile(python_exe)
        and _python_version(python_exe) == "3.12"
        and not _is_embeddable_python_exe(python_exe)
    )


def _dedupe_paths(paths: List[str]) -> List[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for path in paths:
        if not path:
            continue
        normalized = os.path.normcase(os.path.abspath(path))
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(path)
    return deduped


def _resolve_builder_python() -> str:
    if _is_valid_builder_python(sys.executable):
        return sys.executable

    raise RuntimeError(
        "Nuitka builds must run from a non-embeddable Python 3.12 environment.\n"
        "In the packaged Hub workflow, each project owns a full Python copy "
        "under .runtime/python312/ — open the project through its runtime and build from there."
    )


def _ensure_python_packages(python_exe: str, *module_names: str) -> None:
    import time as _time
    missing_packages: list[str] = []
    _check_t0 = _time.perf_counter()

    # Check all modules in a single subprocess instead of one per module.
    check_script = (
        "import importlib.util, sys; "
        "mods = sys.argv[1:]; "
        "print(','.join(str(int(importlib.util.find_spec(m) is not None)) for m in mods))"
    )
    completed = _run_python(
        python_exe,
        ["-c", check_script, *module_names],
        timeout=30,
    )
    if completed.returncode == 0 and (completed.stdout or "").strip():
        results = (completed.stdout or "").strip().split(",")
        for module_name, available in zip(module_names, results):
            if available.strip() != "1":
                package_name = _AUTO_INSTALLABLE_PACKAGES.get(module_name)
                if package_name and package_name not in missing_packages:
                    missing_packages.append(package_name)
    else:
        # Fallback: treat all as potentially missing
        for module_name in module_names:
            package_name = _AUTO_INSTALLABLE_PACKAGES.get(module_name)
            if package_name and package_name not in missing_packages:
                missing_packages.append(package_name)

    Debug.log_internal(
        f"  package availability check for {len(module_names)} modules in "
        f"{_time.perf_counter() - _check_t0:.2f}s"
    )

    if not missing_packages:
        return

    Debug.log_internal(
        "Missing build packages detected — installing automatically: "
        + ", ".join(missing_packages)
    )
    _pip_t0 = _time.perf_counter()
    subprocess.check_call(
        [python_exe, "-m", "pip", "install", *missing_packages, "--quiet"],
    )
    Debug.log_internal(
        f"  pip install completed in {_time.perf_counter() - _pip_t0:.2f}s"
    )


def _install_requirements_files(python_exe: str, requirement_files: List[str]) -> None:
    for requirement_file in requirement_files:
        if not requirement_file or not os.path.isfile(requirement_file):
            continue
        Debug.log_internal(
            f"Installing project requirements into builder Python: {requirement_file}"
        )
        subprocess.check_call(
            [
                python_exe,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "-r",
                requirement_file,
                "--quiet",
            ],
        )


class NuitkaBuilder:
    """Wraps Nuitka compilation for Infernux standalone builds."""

    # Packages that must be excluded from Nuitka compilation and
    # injected as raw site-packages into the dist.  Numba requires
    # Python bytecode at runtime for its LLVM JIT compiler — Nuitka's
    # C compilation removes the bytecode, making @njit silently fail.
    _JIT_NOFOLLOW_PACKAGES = frozenset({"numba", "llvmlite", "numpy"})

    # Directories stripped from raw-copied JIT packages to slim down
    # the build output.  These are never needed at runtime.
    _JIT_STRIP_DIRS: dict[str, list[str]] = {
        "numba": [
            "tests", "cuda", "testing", "pycc", "scripts",
            # CUDA is ~4 MB and only needed for GPU compute
            # tests/testing are ~10 MB of test fixtures
        ],
        "numpy": [
            "tests", "f2py", "testing", "doc",
            "_pyinstaller", "distutils",
            # f2py is 1.6 MB Fortran build tooling
        ],
        "llvmlite": [
            "tests",
        ],
    }

    def __init__(
        self,
        entry_script: str,
        output_dir: str,
        *,
        output_filename: str = "Game.exe",
        product_name: str = "Infernux Game",
        file_version: str = "1.0.0.0",
        icon_path: Optional[str] = None,
        extra_include_packages: Optional[List[str]] = None,
        extra_include_data: Optional[List[str]] = None,
        extra_requirements_files: Optional[List[str]] = None,
        raw_copy_packages: Optional[List[str]] = None,
        console_mode: str = "disable",
        lto: bool = True,
    ):
        self.entry_script = os.path.abspath(entry_script)
        self.output_dir = os.path.abspath(output_dir)
        self.output_filename = output_filename
        self.product_name = product_name
        self.file_version = file_version
        self.icon_path = icon_path
        self.console_mode = console_mode
        self.lto = lto
        self.extra_include_packages = list(extra_include_packages or [])
        self.extra_include_data = list(extra_include_data or [])
        self.extra_requirements_files = [
            os.path.abspath(path)
            for path in list(extra_requirements_files or [])
            if path
        ]
        self.raw_copy_packages = list(raw_copy_packages or [])

        # Staging directory — unique per build to allow parallel builds
        tag = hashlib.md5(self.output_dir.encode()).hexdigest()[:8]
        self._staging_dir = os.path.join(_STAGING_ROOT, tag)
        self._builder_python = _resolve_builder_python()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        on_progress: Optional[Callable[[str, float], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> str:
        """Run Nuitka compilation.  Returns the dist directory path."""
        import time as _time
        _build_t0 = _time.perf_counter()
        _stage_t0 = _build_t0

        def _p(msg: str, pct: float):
            nonlocal _stage_t0
            if cancel_event is not None and cancel_event.is_set():
                raise _BuildCancelled()
            now = _time.perf_counter()
            elapsed = now - _stage_t0
            _stage_t0 = now
            if on_progress:
                on_progress(msg, pct)
            Debug.log_internal(
                f"[NuitkaBuilder {pct:.0%}] {msg}  (prev {elapsed:.2f}s, "
                f"nuitka total {now - _build_t0:.1f}s)"
            )

        _p(t("build.step.checking_nuitka"), 0.0)
        self._check_nuitka()

        _p(t("build.step.preparing_staging"), 0.03)
        self._prepare_staging()

        _p(t("build.step.building_command"), 0.05)
        cmd = self._build_command()
        _p(f"cmd: {' '.join(cmd)}", 0.05)

        _p(t("build.step.running_nuitka"), 0.10)
        dist_dir = self._run_nuitka(cmd, on_progress, cancel_event)

        _p(t("build.step.injecting_libs"), 0.85)
        self._inject_native_libs(dist_dir)

        if self.raw_copy_packages:
            _p(t("build.step.injecting_jit"), 0.87)
            self._inject_jit_packages(dist_dir)

        if sys.platform == "win32":
            _p(t("build.step.embedding_manifest"), 0.90)
            self._embed_utf8_manifest(dist_dir)

            _p(t("build.step.signing_exe"), 0.92)
            self._sign_executable(dist_dir)

        _p(t("build.step.cleaning_artifacts"), 0.95)
        self._cleanup_build_artifacts()

        _p(t("build.step.nuitka_complete"), 1.0)
        return dist_dir

    # ------------------------------------------------------------------
    # Nuitka availability check
    # ------------------------------------------------------------------

    def _check_nuitka(self):
        """Ensure Nuitka and build-time project dependencies are installed."""
        import time as _time
        try:
            _t0 = _time.perf_counter()
            _ensure_python_packages(
                self._builder_python,
                "nuitka",
                "ordered_set",
                *self.extra_include_packages,
            )
            Debug.log_internal(
                f"  _ensure_python_packages in {_time.perf_counter() - _t0:.2f}s"
            )
            _t1 = _time.perf_counter()
            _install_requirements_files(
                self._builder_python,
                self.extra_requirements_files,
            )
            Debug.log_internal(
                f"  _install_requirements_files in {_time.perf_counter() - _t1:.2f}s"
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to prepare the builder Python environment.  "
                f"Builder Python: {self._builder_python}\n"
                "Please run manually:\n"
                "    pip install nuitka ordered-set\n"
                "and install the project's requirements.txt if needed."
            ) from exc

    # ------------------------------------------------------------------
    # Staging directory
    # ------------------------------------------------------------------

    def _prepare_staging(self):
        """Create a clean ASCII-only staging directory.

        Using a short ASCII-only staging directory avoids temporary-path
        edge cases and keeps compiler output paths stable on Windows.
        """
        if os.path.isdir(self._staging_dir):
            if sys.platform == "win32":
                subprocess.run(
                    ["cmd", "/c", "rd", "/s", "/q", self._staging_dir],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            else:
                shutil.rmtree(self._staging_dir, ignore_errors=True)
        os.makedirs(self._staging_dir, exist_ok=True)

        # Copy entry script into staging (path itself may be non-ASCII)
        staged_script = os.path.join(self._staging_dir, "boot.py")
        shutil.copy2(self.entry_script, staged_script)
        self._staged_entry = staged_script

    # ------------------------------------------------------------------
    # Command construction
    # ------------------------------------------------------------------

    def _build_command(self) -> List[str]:
        """Assemble the Nuitka command line.

        All output paths point to the ASCII-safe staging directory.
        """
        cmd = [
            self._builder_python, "-m", "nuitka",
            "--standalone",
            "--assume-yes-for-downloads",
            f"--windows-console-mode={self.console_mode}",
            "--follow-imports",
            f"--output-dir={self._staging_dir}",
            f"--output-filename={self.output_filename}",
            # Disable Nuitka's deployment-time hard-crash when an excluded
            # module is imported.  Some modules are legitimately excluded
            # but lazily imported with graceful fallback (try/except or
            # None checks); the default deployment flag converts those
            # into RuntimeErrors which is counter-productive.
            "--no-deployment-flag=excluded-module-usage",
        ]

        if sys.platform == "win32":
            if not _has_msvc_toolchain():
                raise RuntimeError(
                    "Windows game builds require Microsoft Visual C++ Build Tools (MSVC).\n"
                    "MinGW fallback has been disabled.\n"
                    "Install Visual Studio 2022 Build Tools or Visual Studio with the Desktop development with C++ workload, then try again."
                )
            cmd.append("--msvc=latest")

        # Link-time optimization for smaller and faster binaries
        if self.lto:
            cmd.append("--lto=yes")

        # Strip docstrings and assert statements for smaller output
        cmd.append("--python-flag=-OO")

        # Tell Nuitka to exclude large dev-only frameworks at the module
        # level — catches transitive imports that --nofollow-import-to
        # might miss.
        cmd.append("--noinclude-pytest-mode=nofollow")
        cmd.append("--noinclude-unittest-mode=nofollow")
        cmd.append("--noinclude-setuptools-mode=nofollow")

        # Parallel C compilation
        cmd.append("--jobs=%d" % max(1, os.cpu_count() - 1))

        # Include package data (fonts, shaders, icons…) but NOT the whole
        # package as source — let --follow-imports trace only what the
        # player entry script actually needs.  This avoids compiling the
        # entire editor UI (hundreds of files) which is never used.
        cmd += [
            "--include-package-data=Infernux",
        ]

        # Explicitly ensure the pybind11 native extension is bundled
        # (Nuitka may not auto-detect it because it's a .pyd, not .py).
        cmd.append("--include-module=Infernux.lib._Infernux")

        # csv is needed by importlib.metadata (Python's own import system)
        # but Nuitka may not auto-detect it when JIT packages are excluded.
        cmd.append("--include-module=csv")

        # Prevent Nuitka from following into editor-only modules that the
        # standalone player never uses.  The _INFERNUX_PLAYER_MODE guard
        # in __init__ already prevents runtime loading, but --nofollow
        # also speeds up Nuitka's compile-time analysis significantly.
        #
        # NOTE: Do NOT exclude Infernux.engine.resources_manager here —
        # render_stack.py lazily imports ResourcesManager.instance() and
        # Nuitka's excluded-module deployment flag causes a hard crash
        # instead of allowing the graceful None fallback.
        for _editor_mod in (
            "Infernux.engine.bootstrap",
            "watchdog",
            "PIL",
            "cv2",
            "imageio",
            "psd_tools",
            "av",  # PyAV/ffmpeg — build-time splash encoding only
        ):
            cmd.append(f"--nofollow-import-to={_editor_mod}")

        # Exclude JIT packages from Nuitka compilation — they will be
        # injected as raw site-packages afterwards so numba retains the
        # Python bytecode it needs for LLVM JIT at runtime.
        _nofollow_jit = self._JIT_NOFOLLOW_PACKAGES | set(self.raw_copy_packages)
        for _jit_pkg in sorted(_nofollow_jit):
            cmd.append(f"--nofollow-import-to={_jit_pkg}")

        # Auto-discover stdlib modules that the raw-copied JIT packages
        # import transitively.  Nuitka can't discover them because the
        # packages are excluded via --nofollow-import-to, so we trace
        # them in a subprocess and include them explicitly.
        if self.raw_copy_packages:
            for _stdlib_mod in self._discover_jit_stdlib_deps():
                cmd.append(f"--include-module={_stdlib_mod}")

        # Numba's parallel backend (prange / parallel=True) imports
        # multiprocessing lazily at JIT compile time — NOT at
        # ``import numba``.  The auto-discovery above therefore misses
        # it.  Include it unconditionally when JIT packages are bundled.
        if _nofollow_jit & self._JIT_NOFOLLOW_PACKAGES:
            cmd.append("--include-module=multiprocessing")

        for pkg in self.extra_include_packages:
            if pkg not in _nofollow_jit:
                cmd.append(f"--include-package={pkg}")

        for pattern in self.extra_include_data:
            cmd.append(f"--include-package-data={pattern}")

        # Product metadata (Windows)
        if sys.platform == "win32":
            cmd.append(f"--product-name={self.product_name}")
            cmd.append(f"--file-version={self.file_version}")
            cmd.append(f"--product-version={self.file_version}")

            if self.icon_path and os.path.isfile(self.icon_path):
                ico = self._ensure_ico(self.icon_path)
                if ico:
                    cmd.append(f"--windows-icon-from-ico={ico}")

        # Exclude heavy dev/test modules that aren't needed at runtime
        for mod in ("tkinter", "unittest", "test", "pip",
                    "setuptools", "distutils", "ensurepip"):
            cmd.append(f"--nofollow-import-to={mod}")

        cmd.append(self._staged_entry)
        return cmd

    # ------------------------------------------------------------------
    # Auto-discover JIT package stdlib dependencies
    # ------------------------------------------------------------------

    def _discover_jit_stdlib_deps(self) -> List[str]:
        """Import raw_copy_packages in the builder Python and return the
        set of stdlib top-level modules they transitively load.

        This replaces the manual list approach — numba/llvmlite/numpy
        pull in dozens of stdlib modules (email, csv, html, http, …)
        that Nuitka cannot discover because we exclude these packages
        via --nofollow-import-to.
        """
        import time as _time
        _t0 = _time.perf_counter()

        pkgs_arg = ",".join(sorted(self.raw_copy_packages))
        # The subprocess: record modules before vs after importing the
        # packages, then report only stdlib top-level names.
        trace_script = (
            "import sys; "
            "before = set(sys.modules); "
            "pkgs = '$PKGS'.split(','); "
            "[__import__(p) for p in pkgs]; "
            "after = set(sys.modules); "
            "new = {m.split('.')[0] for m in after - before}; "
            "stdlib = sorted(new & sys.stdlib_module_names); "
            "print(','.join(stdlib))"
        ).replace("$PKGS", pkgs_arg)

        try:
            result = _run_python(
                self._builder_python,
                ["-c", trace_script],
                timeout=120,
            )
            if result.returncode != 0 or not result.stdout.strip():
                Debug.log_warning(
                    f"JIT stdlib trace failed (exit {result.returncode}): "
                    f"{(result.stderr or '').strip()[:200]}"
                )
                return []

            mods = [m for m in result.stdout.strip().split(",") if m]
            Debug.log_internal(
                f"  JIT stdlib trace: {len(mods)} modules in "
                f"{_time.perf_counter() - _t0:.2f}s  "
                f"({', '.join(mods[:10])}{'…' if len(mods) > 10 else ''})"
            )
            return mods
        except Exception as exc:
            Debug.log_warning(f"JIT stdlib trace error: {exc}")
            return []

    # ------------------------------------------------------------------
    # Nuitka execution
    # ------------------------------------------------------------------

    def _run_nuitka(
        self,
        cmd: List[str],
        on_progress: Optional[Callable[[str, float], None]],
        cancel_event: Optional[threading.Event] = None,
    ) -> str:
        """Run Nuitka as a subprocess and stream output.  Returns dist dir."""
        env = os.environ.copy()

        # Redirect TEMP / TMP to an ASCII-safe location so MinGW's
        # std::filesystem never encounters non-ASCII characters.
        safe_tmp = os.path.join(self._staging_dir, "_tmp")
        os.makedirs(safe_tmp, exist_ok=True)
        env["TEMP"] = safe_tmp
        env["TMP"] = safe_tmp

        safe_profile = os.path.join(self._staging_dir, "_profile")
        safe_local_appdata = os.path.join(safe_profile, "AppData", "Local")
        safe_roaming_appdata = os.path.join(safe_profile, "AppData", "Roaming")
        for path in (safe_profile, safe_local_appdata, safe_roaming_appdata):
            os.makedirs(path, exist_ok=True)

        env["USERPROFILE"] = safe_profile
        env["HOME"] = safe_profile
        env["LOCALAPPDATA"] = safe_local_appdata
        env["APPDATA"] = safe_roaming_appdata
        if sys.platform == "win32":
            drive, tail = os.path.splitdrive(safe_profile)
            env["HOMEDRIVE"] = drive or "C:"
            env["HOMEPATH"] = tail or "\\"

        # Use a persistent cache directory so Nuitka can reuse compiled C
        # code across builds — this is the single biggest speed win.
        os.makedirs(_NUITKA_CACHE_DIR, exist_ok=True)
        env["NUITKA_CACHE_DIR"] = _NUITKA_CACHE_DIR

        # If we switch away from the current interpreter to a reusable build
        # venv, preserve the current import roots so Nuitka can still resolve
        # the live Infernux package and project-installed dependencies.
        pythonpath_entries: list[str] = []
        existing_pythonpath = env.get("PYTHONPATH", "")
        if existing_pythonpath:
            pythonpath_entries.extend([p for p in existing_pythonpath.split(os.pathsep) if p])
        pythonpath_entries.extend(
            path for path in sys.path
            if path and os.path.isdir(path)
        )
        if pythonpath_entries:
            env["PYTHONPATH"] = os.pathsep.join(_dedupe_paths(pythonpath_entries))

        if sys.platform == "win32":
            env = _ensure_windows_msvc_environment(env)

        import time as _time
        _nuitka_proc_t0 = _time.perf_counter()
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            cwd=self._staging_dir,
        )

        lines_collected: List[str] = []
        try:
            for line in proc.stdout:
                if cancel_event is not None and cancel_event.is_set():
                    raise _BuildCancelled()
                line = line.rstrip()
                lines_collected.append(line)
                if on_progress:
                    # Crude progress: Nuitka logs many lines; we map to 10%–85%
                    pct = min(0.85, 0.10 + len(lines_collected) * 0.001)
                    on_progress(line[-80:] if len(line) > 80 else line, pct)
        except _BuildCancelled:
            proc.kill()
            proc.wait()
            raise

        proc.wait()
        _nuitka_elapsed = _time.perf_counter() - _nuitka_proc_t0
        Debug.log_internal(
            f"  Nuitka subprocess finished in {_nuitka_elapsed:.1f}s  "
            f"({len(lines_collected)} output lines, exit {proc.returncode})"
        )

        if proc.returncode != 0:
            tail = "\n".join(lines_collected[-30:])
            raise RuntimeError(
                f"Nuitka compilation failed (exit code {proc.returncode}).\n"
                f"Last output:\n{tail}"
            )

        # Nuitka places output in <staging_dir>/boot.dist/
        dist_dir = os.path.join(self._staging_dir, "boot.dist")
        if not os.path.isdir(dist_dir):
            raise RuntimeError(
                f"Nuitka dist directory not found: {dist_dir}\n"
                "Compilation may have failed silently."
            )
        return dist_dir

    # ------------------------------------------------------------------
    # Inject native engine libraries
    # ------------------------------------------------------------------

    def _inject_native_libs(self, dist_dir: str):
        """Copy _Infernux.pyd + engine DLLs into the Nuitka dist directory.

        Nuitka won't automatically pick up .pyd files built outside its
        compilation scope (pybind11 extensions), so we inject them into
        the correct package subdirectory so that
        ``from ._Infernux import *`` (relative import in Infernux.lib)
        can find the .pyd, and ``os.add_dll_directory(lib_dir)`` picks
        up the companion DLLs.
        """
        import time as _time
        _inject_t0 = _time.perf_counter()
        import Infernux.lib as _lib
        lib_dir = Path(_lib.__file__).parent

        # Target: <dist>/Infernux/lib/  — mirrors the installed package
        # structure so relative imports work at runtime.
        target_dir = Path(dist_dir) / "Infernux" / "lib"
        target_dir.mkdir(parents=True, exist_ok=True)

        # Also put DLLs in the dist root as a fallback for Windows DLL
        # search (the .exe directory is always searched).
        dist_root = Path(dist_dir)

        # List of native files to inject
        native_files = []
        for f in lib_dir.iterdir():
            if f.is_file() and f.suffix.lower() in (".pyd", ".dll"):
                native_files.append(f)

        for src in native_files:
            # .pyd goes into the package subdir (for relative import)
            dst_pkg = target_dir / src.name
            if not dst_pkg.exists():
                shutil.copy2(src, dst_pkg)
                Debug.log_internal(f"  Injected (lib): {src.name}")

            # DLLs also go into the dist root (for OS DLL search path)
            if src.suffix.lower() == ".dll":
                dst_root = dist_root / src.name
                if not dst_root.exists():
                    shutil.copy2(src, dst_root)
                    Debug.log_internal(f"  Injected (root): {src.name}")

        Debug.log_internal(
            f"  native lib injection total: {_time.perf_counter() - _inject_t0:.2f}s  "
            f"({len(native_files)} files)"
        )

    # ------------------------------------------------------------------
    # Inject raw JIT packages
    # ------------------------------------------------------------------

    def _inject_jit_packages(self, dist_dir: str):
        """Copy raw site-packages into the Nuitka dist for JIT-dependent packages.

        Numba requires Python bytecode at runtime for LLVM JIT compilation.
        Nuitka's C compilation removes bytecode, so these packages must be
        copied as-is from the builder environment's site-packages.
        """
        if not self.raw_copy_packages:
            return

        import time as _time
        _t0 = _time.perf_counter()

        # Discover site-packages directory of the builder Python.
        # In conda environments getsitepackages() returns [env_root,
        # env_root/Lib/site-packages] — we need the one that actually
        # contains installed packages (the "Lib/site-packages" entry).
        result = _run_python(
            self._builder_python,
            ["-c",
             "import site, os, json; "
             "print(json.dumps(site.getsitepackages()))"],
        )
        import json as _json
        candidates = _json.loads(result.stdout.strip())
        site_packages = ""
        for cand in reversed(candidates):
            if os.path.isdir(cand) and os.path.isdir(os.path.join(cand, "numba")):
                site_packages = cand
                break
        if not site_packages:
            # Fallback: pick the last entry (usually Lib/site-packages)
            for cand in reversed(candidates):
                if os.path.isdir(cand):
                    site_packages = cand
                    break
        if not site_packages:
            Debug.log_warning(
                f"Builder site-packages not found in {candidates}"
            )
            return

        Debug.log_internal(
            f"  site-packages resolved: {site_packages}  "
            f"({_time.perf_counter() - _t0:.2f}s)"
        )

        copied: list[str] = []
        dist_root = Path(dist_dir)

        for pkg in sorted(self.raw_copy_packages):
            src = os.path.join(site_packages, pkg)
            if not os.path.isdir(src):
                Debug.log_warning(
                    f"JIT package '{pkg}' not found in {site_packages} — skipping"
                )
                continue

            _pkg_t0 = _time.perf_counter()
            dst = dist_root / pkg
            if dst.exists():
                if sys.platform == "win32":
                    subprocess.run(
                        ["cmd", "/c", "rd", "/s", "/q", str(dst)],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                else:
                    shutil.rmtree(dst)

            # Use robocopy on Windows for significantly faster bulk copy.
            # /XD skips directories that are never needed at runtime,
            # cutting the copy volume substantially (especially numpy).
            # Package-specific strip dirs come from _JIT_STRIP_DIRS.
            xd_dirs = ["__pycache__", "tests", "test"]
            xd_dirs.extend(self._JIT_STRIP_DIRS.get(pkg, []))
            if sys.platform == "win32":
                rc = subprocess.call(
                    ["robocopy", src, str(dst), "/E",
                     "/XD", *xd_dirs,
                     "/XF", "*.pyc", "*.pdb", "*.lib", "*.a",
                     "/NFL", "/NDL", "/NJH", "/NJS", "/NP"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=0x08000000,
                )
                if rc >= 8:
                    Debug.log_warning(
                        f"robocopy failed for '{pkg}' (exit {rc}), "
                        f"falling back to shutil.copytree"
                    )
                    if dst.exists():
                        shutil.rmtree(dst)
                    shutil.copytree(src, dst)
            else:
                shutil.copytree(src, dst)
                # Strip on non-Windows too
                for strip_dir in xd_dirs:
                    strip_path = dst / strip_dir
                    if strip_path.is_dir():
                        shutil.rmtree(strip_path)

            elapsed = _time.perf_counter() - _pkg_t0
            copied.append(f"{pkg} ({elapsed:.1f}s)")

            # Copy companion .libs directory if it exists (e.g. numpy.libs
            # contains OpenBLAS DLLs that numpy's C extensions need).
            libs_name = f"{pkg}.libs"
            libs_src = os.path.join(site_packages, libs_name)
            if os.path.isdir(libs_src):
                libs_dst = dist_root / libs_name
                if libs_dst.exists():
                    if sys.platform == "win32":
                        subprocess.run(
                            ["cmd", "/c", "rd", "/s", "/q", str(libs_dst)],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
                    else:
                        shutil.rmtree(libs_dst)
                if sys.platform == "win32":
                    subprocess.call(
                        ["robocopy", libs_src, str(libs_dst), "/E",
                         "/NFL", "/NDL", "/NJH", "/NJS", "/NP"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        creationflags=0x08000000,
                    )
                else:
                    shutil.copytree(libs_src, libs_dst)
                copied.append(f"{libs_name}")

        if copied:
            Debug.log_internal(
                f"  JIT package injection: {', '.join(copied)}  "
                f"(total {_time.perf_counter() - _t0:.1f}s)"
            )

    # ------------------------------------------------------------------
    # UTF-8 application manifest (Windows)
    # ------------------------------------------------------------------

    # Complete manifest that tells Windows to use UTF-8 as the process's
    # ANSI code page (Windows 10 1903+).  Without this, any path
    # containing non-ASCII characters (e.g. Chinese usernames) causes
    # the C++ engine to fail with "No mapping for the Unicode character
    # exists in the target multi-byte code page".
    _UTF8_MANIFEST = (
        b'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\r\n'
        b'<assembly xmlns="urn:schemas-microsoft-com:asm.v1" manifestVersion="1.0">\r\n'
        b'  <trustInfo xmlns="urn:schemas-microsoft-com:asm.v3">\r\n'
        b'    <security>\r\n'
        b'      <requestedPrivileges>\r\n'
        b'        <requestedExecutionLevel level="asInvoker" uiAccess="false"/>\r\n'
        b'      </requestedPrivileges>\r\n'
        b'    </security>\r\n'
        b'  </trustInfo>\r\n'
        b'  <compatibility xmlns="urn:schemas-microsoft-com:compatibility.v1">\r\n'
        b'    <application>\r\n'
        b'      <supportedOS Id="{8e0f7a12-bfb3-4fe8-b9a5-48fd50a15a9a}"/>\r\n'
        b'    </application>\r\n'
        b'  </compatibility>\r\n'
        b'  <application xmlns="urn:schemas-microsoft-com:asm.v3">\r\n'
        b'    <windowsSettings>\r\n'
        b'      <activeCodePage xmlns="http://schemas.microsoft.com/SMI/2019/WindowsSettings">UTF-8</activeCodePage>\r\n'
        b'      <dpiAware xmlns="http://schemas.microsoft.com/SMI/2005/WindowsSettings">true/pm</dpiAware>\r\n'
        b'      <dpiAwareness xmlns="http://schemas.microsoft.com/SMI/2016/WindowsSettings">permonitorv2,permonitor</dpiAwareness>\r\n'
        b'    </windowsSettings>\r\n'
        b'  </application>\r\n'
        b'</assembly>\r\n'
    )

    def _embed_utf8_manifest(self, dist_dir: str):
        """Embed an application manifest with UTF-8 active code page.

        Uses the Win32 resource-update API so no external tools (mt.exe,
        rc.exe) are required.  Replaces the default Nuitka manifest.
        """
        import ctypes
        from ctypes import wintypes

        exe_path = os.path.join(dist_dir, self.output_filename)
        if not os.path.isfile(exe_path):
            Debug.log_warning(
                f"Cannot embed manifest: EXE not found at {exe_path}"
            )
            return

        k32 = ctypes.windll.kernel32

        # --- open for resource update --------------------------------
        k32.BeginUpdateResourceW.argtypes = [wintypes.LPCWSTR, wintypes.BOOL]
        k32.BeginUpdateResourceW.restype = wintypes.HANDLE
        h = k32.BeginUpdateResourceW(exe_path, False)
        if not h:
            Debug.log_warning(
                f"BeginUpdateResource failed (error {ctypes.GetLastError()})"
            )
            return

        # RT_MANIFEST = 24, CREATEPROCESS_MANIFEST_RESOURCE_ID = 1
        RT_MANIFEST = 24
        MANIFEST_ID = 1
        data = self._UTF8_MANIFEST

        k32.UpdateResourceW.argtypes = [
            wintypes.HANDLE,   # hUpdate
            wintypes.LPVOID,   # lpType  (MAKEINTRESOURCE)
            wintypes.LPVOID,   # lpName  (MAKEINTRESOURCE)
            wintypes.WORD,     # wLanguage
            ctypes.c_char_p,   # lpData
            wintypes.DWORD,    # cb
        ]
        k32.UpdateResourceW.restype = wintypes.BOOL

        ok = k32.UpdateResourceW(h, RT_MANIFEST, MANIFEST_ID, 0, data, len(data))
        if not ok:
            Debug.log_warning(
                f"UpdateResource failed (error {ctypes.GetLastError()})"
            )
            k32.EndUpdateResourceW(h, True)  # discard changes
            return

        k32.EndUpdateResourceW.argtypes = [wintypes.HANDLE, wintypes.BOOL]
        k32.EndUpdateResourceW.restype = wintypes.BOOL
        k32.EndUpdateResourceW(h, False)

        Debug.log_internal("Embedded UTF-8 active-code-page manifest")

    # ------------------------------------------------------------------
    # Code signing (reduces antivirus false positives)
    # ------------------------------------------------------------------

    def _sign_executable(self, dist_dir: str):
        """Sign the built EXE with a self-signed certificate.

        Unsigned executables — especially those compiled with MinGW —
        are far more likely to trigger antivirus false positives because
        they lack an Authenticode signature.  This method creates a
        self-signed code-signing certificate (cached per-machine) and
        applies it to the output EXE using PowerShell's
        ``Set-AuthenticodeSignature``.

        A self-signed certificate won't prevent SmartScreen warnings
        (that requires a purchased EV certificate), but it does help
        with heuristic-based AV scanners that penalise unsigned binaries.
        """
        exe_path = os.path.join(dist_dir, self.output_filename)
        if not os.path.isfile(exe_path):
            return

        # Use PowerShell to: (1) find or create a self-signed code signing
        # cert in CurrentUser\\My, (2) sign the EXE.
        ps_script = r'''
$ErrorActionPreference = "Stop"
$certName = "Infernux Build Signing"
$securityModule = Get-Module -ListAvailable Microsoft.PowerShell.Security | Select-Object -First 1
if (-not $securityModule) {
    Write-Output "UNSUPPORTED:security-module"
    exit 0
}

Import-Module Microsoft.PowerShell.Security -ErrorAction Stop

if (-not (Get-PSDrive -Name Cert -ErrorAction SilentlyContinue)) {
    Write-Output "UNSUPPORTED:cert-drive"
    exit 0
}

$setAuth = Get-Command Set-AuthenticodeSignature -ErrorAction SilentlyContinue
if (-not $setAuth) {
    Write-Output "UNSUPPORTED:set-authenticode"
    exit 0
}

$newSelfSigned = Get-Command New-SelfSignedCertificate -ErrorAction SilentlyContinue

$cert = Get-ChildItem Cert:\CurrentUser\My |
        Where-Object {
            $_.Subject -eq "CN=$certName" -and
            $_.NotAfter -gt (Get-Date) -and
            $_.HasPrivateKey -and
            ($_.EnhancedKeyUsageList | Where-Object { $_.FriendlyName -eq "Code Signing" })
        } |
        Select-Object -First 1

if (-not $cert) {
    if (-not $newSelfSigned) {
        Write-Output "UNSUPPORTED:new-self-signed-certificate"
        exit 0
    }

    $cert = New-SelfSignedCertificate `
        -Subject "CN=$certName" `
        -Type CodeSigningCert `
        -CertStoreLocation Cert:\CurrentUser\My `
        -NotAfter (Get-Date).AddYears(5)
}

$result = Set-AuthenticodeSignature -FilePath $EXE_PATH -Certificate $cert -HashAlgorithm SHA256
if ($null -eq $result) {
    Write-Output "UNSUPPORTED:no-result"
    exit 0
}

Write-Output ("STATUS:" + [string]$result.Status)
if ($result.StatusMessage) {
    Write-Output ("MESSAGE:" + [string]$result.StatusMessage)
}
'''
        ps_script = ps_script.replace("$EXE_PATH", f'"{exe_path}"')
        try:
            r = subprocess.run(
                ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
                 "-Command", ps_script],
                capture_output=True, text=True, timeout=60,
            )
            stdout_lines = [line.strip() for line in (r.stdout or "").splitlines() if line.strip()]
            stderr_text = (r.stderr or "").strip()

            unsupported = next((line for line in stdout_lines if line.startswith("UNSUPPORTED:")), "")
            status_line = next((line for line in stdout_lines if line.startswith("STATUS:")), "")
            message_line = next((line for line in stdout_lines if line.startswith("MESSAGE:")), "")

            if r.returncode != 0:
                details = stderr_text or "\n".join(stdout_lines)
                Debug.log_warning(f"Code signing failed: {details}")
                return

            if unsupported:
                reason = unsupported.split(":", 1)[1]
                Debug.log_internal(f"Code signing skipped: unsupported PowerShell signing environment ({reason})")
                return

            status = status_line.split(":", 1)[1] if status_line else ""
            message = message_line.split(":", 1)[1] if message_line else ""

            if status == "Valid":
                Debug.log_internal("Signed EXE with self-signed certificate")
            else:
                details = message or stderr_text or "\n".join(stdout_lines)
                Debug.log_warning(f"Code signing returned: {status or details}")
        except Exception as exc:
            Debug.log_warning(f"Code signing skipped: {exc}")

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def _cleanup_build_artifacts(self):
        """Remove Nuitka's intermediate .build directory from staging.

        Deletion runs in a background daemon thread so the caller doesn't
        block.  On Windows we use ``rd /s /q`` which is dramatically faster
        than Python's shutil.rmtree (native NTFS batch-delete vs per-file
        unlink syscalls).
        """
        dirs_to_remove: list[str] = []
        build_dir = os.path.join(self._staging_dir, "boot.build")
        if os.path.isdir(build_dir):
            dirs_to_remove.append(build_dir)
        safe_tmp = os.path.join(self._staging_dir, "_tmp")
        if os.path.isdir(safe_tmp):
            dirs_to_remove.append(safe_tmp)
        # Remove the copied boot script (tiny file, do it synchronously)
        staged_script = os.path.join(self._staging_dir, "boot.py")
        if os.path.isfile(staged_script):
            os.remove(staged_script)

        if dirs_to_remove:
            def _bg_remove(paths: list[str]):
                for p in paths:
                    if sys.platform == "win32":
                        # rd /s /q is 5-10x faster than shutil.rmtree on NTFS
                        subprocess.run(
                            ["cmd", "/c", "rd", "/s", "/q", p],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
                    else:
                        shutil.rmtree(p, ignore_errors=True)

            t = threading.Thread(target=_bg_remove, args=(dirs_to_remove,), daemon=True)
            t.start()

    # ------------------------------------------------------------------
    # Icon conversion
    # ------------------------------------------------------------------

    def _ensure_ico(self, icon_path: str) -> Optional[str]:
        """Return a .ico path, converting from PNG/JPG if needed.

        Nuitka's ``--windows-icon-from-ico`` requires a real .ico file.
        If the source is already .ico, return it as-is.  Otherwise
        convert via Pillow (no ImageMagick needed).
        """
        ext = os.path.splitext(icon_path)[1].lower()
        if ext == ".ico":
            return icon_path

        try:
            _ensure_python_packages(self._builder_python, "PIL")
            from PIL import Image
        except ImportError:
            Debug.log_warning(
                "Pillow not installed — skipping icon embedding.  "
                "Install with: pip install Pillow"
            )
            return None

        ico_path = os.path.join(self._staging_dir, "icon.ico")
        try:
            img = Image.open(icon_path)
            # Standard Windows icon sizes
            sizes = [(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)]
            img.save(ico_path, format="ICO", sizes=sizes)
            Debug.log_internal(f"Converted {os.path.basename(icon_path)} → icon.ico")
            return ico_path
        except Exception as exc:
            Debug.log_warning(f"Icon conversion failed: {exc}")
            return None
