import datetime
import os
import sys
import json
import subprocess
import shutil
import glob

from hub_utils import is_frozen
from python_runtime import PythonRuntimeError, PythonRuntimeManager

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


def _find_dev_wheel() -> str:
    """Find the InfEngine wheel in the dist/ directory next to the engine source.

    Only used in dev mode (non-frozen).
    """
    engine_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    dist_dir = os.path.join(engine_root, "dist")
    wheels = glob.glob(os.path.join(dist_dir, "infengine-*.whl"))
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

        # ── Create .venv and install InfEngine ──────────────────────────
        venv_path = os.path.join(project_dir, ".venv")
        try:
            self._create_project_venv(project_dir)
            self._install_infengine_in_venv(project_dir, engine_version)
        except Exception:
            shutil.rmtree(venv_path, ignore_errors=True)
            raise

        # ── Create VS Code workspace configuration ─────────────────────
        self._create_vscode_workspace(project_dir)

    # -----------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _get_venv_python(project_dir: str) -> str:
        """Return the Python executable inside the project's .venv."""
        venv_dir = os.path.join(project_dir, ".venv")
        if sys.platform == "win32":
            return os.path.join(venv_dir, "Scripts", "python.exe")
        return os.path.join(venv_dir, "bin", "python")

    def _create_project_venv(self, project_dir: str) -> None:
        venv_path = os.path.join(project_dir, ".venv")
        if is_frozen():
            try:
                self.runtime_manager.create_venv(venv_path)
            except PythonRuntimeError as exc:
                raise RuntimeError(str(exc)) from exc
            return

        _run_hidden([sys.executable, "-m", "venv", "--copies", venv_path], timeout=600)

    def _install_infengine_in_venv(self, project_dir: str, engine_version: str = ""):
        """Install the InfEngine wheel into the project's .venv.

        In frozen (packaged Hub) mode, the wheel comes from the version
        manager cache (~/.infengine/versions/<ver>/).
        In dev mode, we use a local prebuilt wheel from dist/.

        Source builds are intentionally blocked here so project creation never
        falls back to a local C++ compile.
        """
        venv_python = ProjectModel._get_venv_python(project_dir)
        if not os.path.isfile(venv_python):
            raise RuntimeError(
                f"venv python not found at {venv_python}.\n"
                "The virtual environment may not have been created correctly."
            )

        wheel = ""

        if engine_version and self.version_manager is not None:
            wheel = self.version_manager.get_wheel_path(engine_version) or ""

        if not wheel and not is_frozen():
            wheel = _find_dev_wheel()

        if not wheel:
            if is_frozen():
                raise RuntimeError(
                    f"No downloaded InfEngine wheel was found for version {engine_version or '(unknown)'}.\n"
                    "Open the Installs page and install that engine version first."
                )
            raise RuntimeError(
                "No prebuilt InfEngine wheel was found in dist/.\n"
                "Build a wheel first; project creation will not fall back to a source build."
            )

        _PIP_FLAGS = [
            "--no-input",
            "--disable-pip-version-check",
            "--prefer-binary",
            "--only-binary=:all:",
        ]

        _run_hidden(
            [venv_python, "-m", "pip", "install", "--force-reinstall", *_PIP_FLAGS, wheel],
            timeout=600,
        )

    @staticmethod
    def _create_vscode_workspace(project_dir: str):
        """
        Create .vscode/ config so that opening any file inside the project
        uses the correct Python interpreter and gets full InfEngine autocompletion.
        """
        vscode_dir = os.path.join(project_dir, ".vscode")
        os.makedirs(vscode_dir, exist_ok=True)

        # ── settings.json ───────────────────────────────────────────────
        venv_python = ProjectModel._get_venv_python(project_dir)
        settings = {
            "python.defaultInterpreterPath": venv_python,
            "python.analysis.typeCheckingMode": "basic",
            "python.analysis.autoImportCompletions": True,
            "python.analysis.diagnosticSeverityOverrides": {
                "reportMissingModuleSource": "none",
            },
            "editor.formatOnSave": True,
            "files.exclude": {
                "**/__pycache__": True,
                "**/*.pyc": True,
                "**/*.meta": True,
                ".venv": True,
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
        pyright_config = {
            "venvPath": ".",
            "venv": ".venv",
            "pythonVersion": "3.12",
            "typeCheckingMode": "basic",
            "reportMissingModuleSource": False,
            "reportWildcardImportFromLibrary": False,
            "include": ["Assets"],
        }
        pyright_path = os.path.join(project_dir, "pyrightconfig.json")
        with open(pyright_path, "w", encoding="utf-8") as f:
            json.dump(pyright_config, f, indent=4, ensure_ascii=False)