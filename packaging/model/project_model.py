import datetime
import os
import sys
import json
import subprocess
import shutil
import glob

from hub_utils import is_frozen, is_project_open
from python_runtime import PythonRuntimeError, PythonRuntimeManager
import logging

# Suppress console windows for all child processes on Windows
_NO_WINDOW: int = 0x08000000 if sys.platform == "win32" else 0


def _popen_kwargs(*, capture_output: bool = False) -> dict:
    """Common subprocess kwargs: suppress console window for child processes.

    When capture_output is True we collect stdout/stderr so the UI can show a
    meaningful failure message instead of hanging indefinitely.
    """
    kw: dict = {"stdin": subprocess.DEVNULL}
    if capture_output:
        kw["stdout"] = subprocess.PIPE
        kw["stderr"] = subprocess.PIPE
        kw["text"] = True
        kw["encoding"] = "utf-8"
        kw["errors"] = "replace"
    else:
        kw["stdout"] = subprocess.DEVNULL
        kw["stderr"] = subprocess.DEVNULL
    if sys.platform == "win32":
        kw["creationflags"] = _NO_WINDOW
    return kw


def _run_hidden(args: list[str], *, timeout: int) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            args,
            check=True,
            timeout=timeout,
            **_popen_kwargs(capture_output=True),
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"Command timed out after {timeout} s.\n{' '.join(args)}"
        ) from exc
    except subprocess.CalledProcessError as exc:
        details = _summarize_output(exc.stderr or exc.stdout)
        raise RuntimeError(
            f"Command failed (exit code {exc.returncode}).\n{' '.join(args)}\n{details}"
        ) from exc


def _summarize_output(output: str) -> str:
    text = (output or "").strip()
    if not text:
        return "No diagnostic output was produced."
    lines = text.splitlines()
    return "\n".join(lines[-20:])


_NATIVE_IMPORT_SMOKE_TEST = (
    "import Infernux.lib\n"
    "print('INFERNUX_NATIVE_IMPORT_OK')\n"
)


def _find_dev_wheel() -> str:
    """Find the Infernux wheel in the dist/ directory next to the engine source.

    Only used in dev mode (non-frozen).
    """
    engine_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    dist_dir = os.path.join(engine_root, "dist")
    wheels = glob.glob(os.path.join(dist_dir, "infernux-*.whl"))
    if wheels:
        wheels.sort(key=os.path.getmtime, reverse=True)
        return wheels[0]
    return ""


class ProjectModel:
    def __init__(self, db, version_manager=None, runtime_manager=None):
        self.db = db
        self.version_manager = version_manager
        self.runtime_manager = runtime_manager or PythonRuntimeManager()

    def add_project(self, name, path):
        return self.db.add_project(name, path)

    def delete_project(self, name):
        base_path = self.db.get_project_path(name)
        project_dir = os.path.join(base_path, name) if base_path else ""

        if project_dir and is_project_open(project_dir):
            raise RuntimeError(
                f"The project is currently open in Infernux and cannot be deleted:\n{project_dir}"
            )

        if project_dir and os.path.exists(project_dir):
            try:
                shutil.rmtree(project_dir)
            except OSError as exc:
                raise RuntimeError(
                    f"Failed to remove the project folder:\n{project_dir}\n{exc}"
                ) from exc

        self.db.delete_project(name)

    
    def init_project_folder(self, project_name: str, project_path: str,
                            engine_version: str = ""):
        project_dir = os.path.join(project_path, project_name)
        os.makedirs(project_dir, exist_ok=True)

        # Create subdirectories
        for subdir in ("ProjectSettings", "Logs", "Library", "Assets"):
            os.makedirs(os.path.join(project_dir, subdir), exist_ok=True)

        # Create a README file in assets
        readme_path = os.path.join(project_dir, "Assets", "README.md")
        with open(readme_path, "w") as f:
            f.write("# Project Assets\n\nThis folder contains all the assets for the project.\n")

        # Create default project requirements
        req_path = os.path.join(project_dir, "ProjectSettings", "requirements.txt")
        if not os.path.isfile(req_path):
            self._copy_bundled_requirements(req_path, engine_version)

        # Create .ini file in project path
        ini_path = os.path.join(project_dir, f"{project_name}.ini")
        now = datetime.datetime.now()
        with open(ini_path, "w", encoding="utf-8") as f:
            f.write("[Project]\n")
            f.write(f"name = {project_name}\n")
            f.write(f"path = {project_dir}\n")
            f.write(f"created_at = {now}\n")
            f.write(f"changed_at = {now}\n")

        # ── Pin engine version ──────────────────────────────────────────
        if engine_version:
            from version_manager import VersionManager
            VersionManager.write_project_version(project_dir, engine_version)

        # ── Create project Python runtime and install Infernux ────────
        runtime_path = os.path.join(project_dir, ".runtime", "python312")
        try:
            self._create_project_runtime(project_dir)
            self._install_infernux_in_runtime(project_dir, engine_version)
        except Exception:
            shutil.rmtree(os.path.join(project_dir, ".runtime"), ignore_errors=True)
            raise

        # ── Create VS Code workspace configuration ─────────────────────
        self._create_vscode_workspace(project_dir)

    # -----------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------

    def _copy_bundled_requirements(self, dest_path: str, engine_version: str) -> None:
        """Copy the default requirements.txt to *dest_path*.

        Resolves the file from the source tree (dev mode) or extracts it
        from the engine wheel, avoiding any ``import Infernux`` in the Hub
        process (which doesn't have the engine package installed).
        """
        import zipfile

        # 1) Dev mode: resolve from the source tree next to this repo
        engine_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        source_req = os.path.join(
            engine_root, "python", "Infernux", "resources", "supports", "requirements.txt",
        )
        if os.path.isfile(source_req):
            shutil.copy2(source_req, dest_path)
            return

        # 2) Extract from the wheel (a zip file)
        wheel = ""
        if engine_version and self.version_manager is not None:
            wheel = self.version_manager.get_wheel_path(engine_version) or ""
        if not wheel and not is_frozen():
            wheel = _find_dev_wheel()
        if wheel and os.path.isfile(wheel):
            try:
                with zipfile.ZipFile(wheel) as zf:
                    for name in zf.namelist():
                        if name.endswith("resources/supports/requirements.txt"):
                            with zf.open(name) as src, open(dest_path, "wb") as dst:
                                shutil.copyfileobj(src, dst)
                            return
            except zipfile.BadZipFile as _exc:
                logging.getLogger(__name__).debug("[Suppressed] %s: %s", type(_exc).__name__, _exc)
                pass

    @staticmethod
    def _get_project_python(project_dir: str) -> str:
        """Return the Python executable for the project.

        In frozen (packaged Hub) mode, each project owns a full Python copy
        under .runtime/python312/.  In dev mode, we use a classic .venv.
        """
        if is_frozen():
            runtime_dir = os.path.join(project_dir, ".runtime", "python312")
            if sys.platform == "win32":
                return os.path.join(runtime_dir, "python.exe")
            return os.path.join(runtime_dir, "bin", "python")
        # Dev mode: classic .venv
        venv_dir = os.path.join(project_dir, ".venv")
        if sys.platform == "win32":
            return os.path.join(venv_dir, "Scripts", "python.exe")
        return os.path.join(venv_dir, "bin", "python")

    def _create_project_runtime(self, project_dir: str) -> None:
        if is_frozen():
            runtime_path = os.path.join(project_dir, ".runtime", "python312")
            try:
                self.runtime_manager.create_project_runtime(runtime_path)
            except PythonRuntimeError as exc:
                raise RuntimeError(str(exc)) from exc
            return

        # Dev mode: create a classic .venv
        venv_path = os.path.join(project_dir, ".venv")
        _run_hidden([sys.executable, "-m", "venv", "--copies", venv_path], timeout=600)

    def _install_infernux_in_runtime(self, project_dir: str, engine_version: str = ""):
        """Install the Infernux wheel into the project's Python environment.

        In frozen (packaged Hub) mode, the wheel is installed into the project's
        full Python copy at .runtime/python312/.
        In dev mode, the wheel is installed into the classic .venv.

        Source builds are intentionally blocked here so project creation never
        falls back to a local C++ compile.
        """
        project_python = ProjectModel._get_project_python(project_dir)
        if not os.path.isfile(project_python):
            raise RuntimeError(
                f"Project Python not found at {project_python}.\n"
                "The project runtime may not have been created correctly."
            )

        wheel = ""

        if engine_version and self.version_manager is not None:
            wheel = self.version_manager.get_wheel_path(engine_version) or ""

        if not wheel and not is_frozen():
            wheel = _find_dev_wheel()

        if not wheel:
            if is_frozen():
                raise RuntimeError(
                    f"No downloaded Infernux wheel was found for version {engine_version or '(unknown)'}.\n"
                    "Open the Installs page and install that engine version first."
                )
            raise RuntimeError(
                "No prebuilt Infernux wheel was found in dist/.\n"
                "Build a wheel first; project creation will not fall back to a source build."
            )

        _PIP_FLAGS = [
            "--no-input",
            "--disable-pip-version-check",
            "--prefer-binary",
            "--only-binary=:all:",
        ]

        _run_hidden(
            [project_python, "-m", "pip", "install", "--force-reinstall", *_PIP_FLAGS, wheel],
            timeout=600,
        )
        ProjectModel.validate_python_runtime(project_python)

    @staticmethod
    def validate_python_runtime(project_python: str) -> None:
        if not os.path.isfile(project_python):
            raise RuntimeError(
                f"Project Python not found at {project_python}.\n"
                "The project runtime may not have been created correctly."
            )

        _run_hidden([project_python, "-c", _NATIVE_IMPORT_SMOKE_TEST], timeout=120)

    @staticmethod
    def validate_project_runtime(project_dir: str) -> None:
        ProjectModel.validate_python_runtime(ProjectModel._get_project_python(project_dir))

    @staticmethod
    def _get_site_packages(project_dir: str) -> str:
        """Return the site-packages directory for the project's Python runtime."""
        if is_frozen():
            runtime_dir = os.path.join(project_dir, ".runtime", "python312")
            if sys.platform == "win32":
                return os.path.join(runtime_dir, "Lib", "site-packages")
            return os.path.join(runtime_dir, "lib", "python3.12", "site-packages")
        venv_dir = os.path.join(project_dir, ".venv")
        if sys.platform == "win32":
            return os.path.join(venv_dir, "Lib", "site-packages")
        return os.path.join(venv_dir, "lib", "python3.12", "site-packages")

    @staticmethod
    def _create_vscode_workspace(project_dir: str):
        """
        Create .vscode/ config so that opening any file inside the project
        uses the correct Python interpreter and gets full Infernux autocompletion.
        """
        vscode_dir = os.path.join(project_dir, ".vscode")
        os.makedirs(vscode_dir, exist_ok=True)

        # ── settings.json ───────────────────────────────────────────────
        project_python = ProjectModel._get_project_python(project_dir)
        site_packages = ProjectModel._get_site_packages(project_dir)
        settings = {
            "python.defaultInterpreterPath": project_python,
            "python.analysis.typeCheckingMode": "basic",
            "python.analysis.autoImportCompletions": True,
            "python.analysis.extraPaths": [site_packages],
            "python.analysis.diagnosticSeverityOverrides": {
                "reportMissingModuleSource": "none",
            },
            "editor.formatOnSave": True,
            "files.exclude": {
                "**/__pycache__": True,
                "**/*.pyc": True,
                "**/*.meta": True,
                ".venv": True,
                ".runtime": True,
                "Library": True,
                "Logs": True,
                "ProjectSettings": True,
            },
        }
        settings_path = os.path.join(vscode_dir, "settings.json")
        with open(settings_path, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=4, ensure_ascii=False)

        # ── extensions.json ─────────────────────────────────────────────
        extensions = {
            "recommendations": [
                "ms-python.python",
                "ms-python.vscode-pylance",
            ]
        }
        extensions_path = os.path.join(vscode_dir, "extensions.json")
        with open(extensions_path, "w", encoding="utf-8") as f:
            json.dump(extensions, f, indent=4, ensure_ascii=False)

        # ── pyrightconfig.json (at project root) ────────────────────────
        # In frozen mode, point Pyright directly at the project runtime Python;
        # in dev mode, use the classic venvPath/venv convention.
        if is_frozen():
            pyright_config = {
                "pythonPath": ProjectModel._get_project_python(project_dir),
                "pythonVersion": "3.12",
                "typeCheckingMode": "basic",
                "reportMissingModuleSource": False,
                "reportWildcardImportFromLibrary": False,
                "extraPaths": [site_packages],
                "include": ["Assets"],
            }
        else:
            pyright_config = {
                "venvPath": ".",
                "venv": ".venv",
                "pythonVersion": "3.12",
                "typeCheckingMode": "basic",
                "reportMissingModuleSource": False,
                "reportWildcardImportFromLibrary": False,
                "extraPaths": [site_packages],
                "include": ["Assets"],
            }
        pyright_path = os.path.join(project_dir, "pyrightconfig.json")
        with open(pyright_path, "w", encoding="utf-8") as f:
            json.dump(pyright_config, f, indent=4, ensure_ascii=False)