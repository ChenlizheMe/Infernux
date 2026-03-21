from __future__ import annotations

import os
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Callable, Optional

try:
    import winreg
except ImportError:
    winreg = None

from hub_utils import get_bundle_dir, get_inner_dir, is_frozen


_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0
_RUNTIME_ROOT = Path.home() / ".infengine" / "runtime"
_INSTALLER_NAME = "python-3.12.0-amd64.exe"
_INSTALLER_URL = "https://www.python.org/ftp/python/3.12.0/python-3.12.0-amd64.exe"


class PythonRuntimeError(RuntimeError):
    pass


class PythonRuntimeManager:
    def __init__(self) -> None:
        _RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)

    def installer_path(self) -> str:
        if is_frozen():
            return self.bundled_installer_path()
        return str(_RUNTIME_ROOT / _INSTALLER_NAME)

    def bundled_installer_path(self) -> str:
        return os.path.join(get_bundle_dir(), "_inner", "runtime", _INSTALLER_NAME)

    def private_runtime_root(self) -> str:
        return os.path.join(get_inner_dir(), "python312")

    def private_runtime_python(self) -> str:
        if sys.platform == "win32":
            return os.path.join(self.private_runtime_root(), "python.exe")
        return os.path.join(self.private_runtime_root(), "bin", "python")

    def has_runtime(self) -> bool:
        return bool(self.get_runtime_path())

    def get_runtime_path(self) -> Optional[str]:
        for candidate in self._candidate_paths():
            if self._is_valid_python312(candidate):
                return candidate
        return None

    def ensure_runtime(self) -> str:
        python_exe = self.get_runtime_path()
        if python_exe:
            return python_exe

        installer = self.prepare_installer()
        self.install_runtime(installer)

        python_exe = self.get_runtime_path()
        if python_exe:
            return python_exe

        raise PythonRuntimeError(
            "Python 3.12 installation completed, but python.exe was not detected.\n"
            "Please verify that Python 3.12 was installed successfully."
        )

    def prepare_installer(
        self,
        *,
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> str:
        if is_frozen():
            installer = self.bundled_installer_path()
            if os.path.isfile(installer):
                return installer
            raise PythonRuntimeError(
                "Bundled Python 3.12 installer was not found in the Hub package.\n"
                f"Expected path: {installer}"
            )

        return self.download_installer(on_progress=on_progress)

    def download_installer(
        self,
        *,
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> str:
        installer = Path(self.installer_path())
        if installer.is_file():
            return str(installer)

        tmp_path = installer.with_suffix(installer.suffix + ".tmp")
        req = urllib.request.Request(_INSTALLER_URL)
        req.add_header("User-Agent", "InfEngine-Hub/1.0")

        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                total = int(resp.headers.get("Content-Length", "0") or 0)
                downloaded = 0
                chunk_size = 1024 * 1024
                with open(tmp_path, "wb") as f:
                    while True:
                        chunk = resp.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        if on_progress is not None and total > 0:
                            on_progress(downloaded, total)
        except OSError as exc:
            raise PythonRuntimeError(
                f"Failed to download Python 3.12 installer.\n{exc}"
            ) from exc

        os.replace(tmp_path, installer)
        return str(installer)

    def install_runtime(self, installer_path: str) -> None:
        if sys.platform != "win32":
            raise PythonRuntimeError("Automatic Python installation is only supported on Windows.")

        os.makedirs(self.private_runtime_root(), exist_ok=True)
        install_args = [
            "/quiet",
            f"TargetDir={self.private_runtime_root()}",
            "InstallAllUsers=0",
            "PrependPath=0",
            "Include_test=0",
            "Include_launcher=0",
            "InstallLauncherAllUsers=0",
            "AssociateFiles=0",
            "Shortcuts=0",
            "Include_pip=1",
        ]

        completed = self._run_command(
            [installer_path, *install_args],
            timeout=1800,
        )
        if completed.returncode != 0:
            details = self._summarize_output(completed.stderr or completed.stdout)
            raise PythonRuntimeError(
                "Python 3.12 installer failed.\n"
                f"Exit code: {completed.returncode}\n"
                f"{details}"
            )

    def create_venv(self, venv_path: str) -> str:
        python_exe = self.ensure_runtime()
        completed = self._run_command(
            [python_exe, "-m", "venv", "--copies", venv_path],
            timeout=600,
        )
        if completed.returncode != 0:
            details = self._summarize_output(completed.stderr or completed.stdout)
            raise PythonRuntimeError(
                "Failed to create the project virtual environment.\n"
                f"Exit code: {completed.returncode}\n"
                f"{details}"
            )

        if sys.platform == "win32":
            venv_python = os.path.join(venv_path, "Scripts", "python.exe")
        else:
            venv_python = os.path.join(venv_path, "bin", "python")

        if not os.path.isfile(venv_python):
            raise PythonRuntimeError(
                f"Virtual environment creation finished, but python.exe was not found at {venv_python}."
            )
        return venv_python

    def _candidate_paths(self) -> list[str]:
        candidates: list[str] = []

        private_python = self.private_runtime_python()
        candidates.append(private_python)

        env_candidate = os.environ.get("INFENGINE_PYTHON312")
        if env_candidate:
            candidates.append(env_candidate)

        if is_frozen():
            return self._dedupe_candidates(candidates)

        candidates.extend(self._registry_candidates())

        for root in filter(None, [os.environ.get("ProgramFiles"), os.environ.get("LocalAppData")]):
            if root == os.environ.get("LocalAppData"):
                candidates.append(os.path.join(root, "Programs", "Python", "Python312", "python.exe"))
            else:
                candidates.append(os.path.join(root, "Python312", "python.exe"))

        py_launcher = self._python_from_launcher()
        if py_launcher:
            candidates.append(py_launcher)

        return self._dedupe_candidates(candidates)

    @staticmethod
    def _dedupe_candidates(candidates: list[str]) -> list[str]:
        deduped: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            normalized = os.path.normcase(os.path.abspath(candidate))
            if normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(candidate)
        return deduped

    def _registry_candidates(self) -> list[str]:
        if winreg is None:
            return []

        candidates: list[str] = []
        keys = [
            (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Python\PythonCore\3.12\InstallPath"),
            (winreg.HKEY_CURRENT_USER, r"SOFTWARE\Python\PythonCore\3.12\InstallPath"),
            (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Python\PythonCore\3.12\InstallPath"),
        ]

        for hive, subkey in keys:
            try:
                with winreg.OpenKey(hive, subkey) as key:
                    install_path, _ = winreg.QueryValueEx(key, None)
            except OSError:
                continue

            if install_path:
                candidates.append(os.path.join(install_path, "python.exe"))
        return candidates

    def _python_from_launcher(self) -> Optional[str]:
        completed = self._run_command(
            ["py", "-3.12", "-c", "import sys; print(sys.executable)"],
            timeout=20,
            raise_on_error=False,
        )
        if completed.returncode != 0:
            return None

        value = (completed.stdout or "").strip().splitlines()
        if not value:
            return None
        return value[-1].strip()

    def _is_valid_python312(self, python_exe: str) -> bool:
        if not python_exe or not os.path.isfile(python_exe):
            return False

        completed = self._run_command(
            [python_exe, "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
            timeout=20,
            raise_on_error=False,
        )
        if completed.returncode != 0:
            return False
        return (completed.stdout or "").strip() == "3.12"

    def _run_command(
        self,
        args: list[str],
        *,
        timeout: int,
        raise_on_error: bool = True,
    ) -> subprocess.CompletedProcess:
        kwargs: dict = {
            "stdin": subprocess.DEVNULL,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "text": True,
        }
        if sys.platform == "win32":
            kwargs["creationflags"] = _NO_WINDOW

        try:
            return subprocess.run(args, timeout=timeout, check=raise_on_error, **kwargs)
        except FileNotFoundError as exc:
            if not raise_on_error:
                return subprocess.CompletedProcess(args=args, returncode=1, stdout="", stderr=str(exc))
            raise PythonRuntimeError(
                f"Command not found.\n{' '.join(args)}\n{exc}"
            ) from exc
        except OSError as exc:
            if not raise_on_error:
                return subprocess.CompletedProcess(args=args, returncode=1, stdout="", stderr=str(exc))
            raise PythonRuntimeError(
                f"Failed to execute command.\n{' '.join(args)}\n{exc}"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise PythonRuntimeError(
                f"Command timed out after {timeout} seconds.\n{' '.join(args)}"
            ) from exc
        except subprocess.CalledProcessError as exc:
            details = self._summarize_output(exc.stderr or exc.stdout)
            raise PythonRuntimeError(
                f"Command failed with exit code {exc.returncode}.\n{' '.join(args)}\n{details}"
            ) from exc

    @staticmethod
    def _summarize_output(output: str) -> str:
        text = (output or "").strip()
        if not text:
            return "No diagnostic output was produced."
        lines = text.splitlines()
        return "\n".join(lines[-20:])