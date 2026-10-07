from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Optional

from hub_utils import (
    get_bundle_dir,
    get_hub_shared_data_dir,
    is_frozen,
    merge_child_env_utf8,
)
from hub_utils import remove_directory_tree as _remove_tree
from private_python_runtime import (
    extract_runtime_archive,
    has_runtime_build_support as _has_build_support,
    is_current_private_runtime_root,
    runtime_archive_for_machine,
    verify_runtime_archive,
    runtime_prefix,
    runtime_publication,
)
from python_runtime_catalog import (
    DEFAULT_PYTHON_RUNTIME,
    PythonRuntimeId,
    SUPPORTED_PYTHON_RUNTIMES,
    runtime_release,
)
from runtime_requirements import runtime_modules, runtime_packages
from runtime_script_relocation import RELOCATE_RUNTIME_SCRIPTS


_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0
_RUNTIME_PACKAGES = runtime_packages()
_REQUIRED_RUNTIME_MODULES = runtime_modules()
_RUNTIME_COPY_EXCLUDED_DIRS = {"__pycache__", ".pytest_cache", "test", "tests"}
_RUNTIME_COPY_EXCLUDED_FILE_SUFFIXES = (".pyc", ".pyo")




def _runtime_bundle_name() -> str:
    return "runtime_bundle.zip"


class PythonRuntimeError(RuntimeError):
    pass


def _default_runtime_dir() -> str:
    return os.path.join(get_hub_shared_data_dir(), "Runtimes")


def _emit_status(callback: Optional[Callable[[str], None]], message: str) -> None:
    if callback is not None:
        callback(message)


def _run_command(args: list[str], *, timeout: int, raise_on_error: bool = False) -> subprocess.CompletedProcess:
    kwargs = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
        "env": merge_child_env_utf8({"PYTHONDONTWRITEBYTECODE": "1"}),
    }
    if sys.platform == "win32":
        kwargs["creationflags"] = _NO_WINDOW

    try:
        return subprocess.run(args, timeout=timeout, check=raise_on_error, **kwargs)
    except OSError as exc:
        if not raise_on_error:
            return subprocess.CompletedProcess(args=args, returncode=1, stdout="", stderr=str(exc))
        raise PythonRuntimeError(str(exc)) from exc
    except subprocess.TimeoutExpired as exc:
        raise PythonRuntimeError(f"Command timed out after {timeout} seconds.\n{subprocess.list2cmdline(args)}") from exc
    except subprocess.CalledProcessError as exc:
        details = (exc.stderr or exc.stdout or "").strip()
        raise PythonRuntimeError(
            f"Command failed with exit code {exc.returncode}.\n{subprocess.list2cmdline(args)}\n{details}"
        ) from exc


def _find_python_in_root(root: str) -> Optional[str]:
    if not root or not os.path.isdir(root):
        return None

    direct_candidates = [
        os.path.join(root, "python.exe"),
        os.path.join(root, "Python.exe"),
        os.path.join(root, "bin", "python"),
    ]
    for candidate in direct_candidates:
        if os.path.isfile(candidate):
            return candidate

    return None


def _pth_files(root: str) -> list[str]:
    if not root or not os.path.isdir(root):
        return []
    return [
        os.path.join(root, name)
        for name in os.listdir(root)
        if name.lower().endswith("._pth") and os.path.isfile(os.path.join(root, name))
    ]


def _is_embedded_root(root: str) -> bool:
    return bool(_pth_files(root))


def _is_python_version(
    python_exe: str, runtime: str | PythonRuntimeId
) -> bool:
    if not python_exe or not os.path.isfile(python_exe):
        return False

    runtime_id = PythonRuntimeId.parse(runtime)
    completed = _run_command(
        [python_exe, "-I", "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
        timeout=20,
        raise_on_error=False,
    )
    return (
        completed.returncode == 0
        and (completed.stdout or "").strip() == runtime_id.series
    )


def _fast_copy_threads() -> int:
    raw_value = os.environ.get("INFERNUX_FAST_COPY_THREADS", "16")
    try:
        return max(1, min(128, int(raw_value)))
    except ValueError:
        return 16


def _runtime_artifact_ignore(_directory: str, names: list[str]) -> set[str]:
    ignored: set[str] = set()
    for name in names:
        lower_name = name.lower()
        if lower_name in _RUNTIME_COPY_EXCLUDED_DIRS or lower_name.endswith(_RUNTIME_COPY_EXCLUDED_FILE_SUFFIXES):
            ignored.add(name)
    return ignored


def _copy_tree_fast(src: str, dest: str, *, exclude_runtime_artifacts: bool = False) -> bool:
    if sys.platform != "win32" or not os.path.isdir(src) or shutil.which("robocopy") is None:
        return False

    os.makedirs(dest, exist_ok=True)
    args = [
        "robocopy", src, dest,
        "/E",
        f"/MT:{_fast_copy_threads()}",
        "/R:1", "/W:1",
        "/XJ",
        "/COPY:DAT", "/DCOPY:DAT",
    ]
    if exclude_runtime_artifacts:
        args.extend(["/XD", *_RUNTIME_COPY_EXCLUDED_DIRS, "/XF", "*.pyc", "*.pyo"])
    args.extend(["/NFL", "/NDL", "/NJH", "/NJS", "/NP"])
    completed = subprocess.run(
        args,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=_NO_WINDOW,
        env=merge_child_env_utf8(),
    )
    if completed.returncode < 8:
        return True

    raise PythonRuntimeError(
        f"Failed to copy Python runtime ({src} -> {dest}, exit {completed.returncode}).\n"
        f"{(completed.stderr or '').strip()}"
    )


def _copy_tree(src: str, dest: str) -> None:
    _remove_tree(dest)
    if not _copy_tree_fast(src, dest):
        shutil.copytree(src, dest)


def _copy_project_runtime_tree(src: str, dest: str) -> None:
    if not _copy_tree_fast(src, dest, exclude_runtime_artifacts=True):
        shutil.copytree(src, dest, ignore=_runtime_artifact_ignore)


def _download_file(url: str, dest: str, *, user_agent: str, timeout: int = 120) -> None:
    req = urllib.request.Request(url)
    req.add_header("User-Agent", user_agent)
    with urllib.request.urlopen(req, timeout=timeout) as resp, open(dest, "wb") as f:
        shutil.copyfileobj(resp, f)


class PythonRuntimeManager:
    def __init__(
        self,
        runtime_dir: Optional[str] = None,
        bundle_runtime_dir: Optional[str] = None,
        *,
        default_version: str | PythonRuntimeId = DEFAULT_PYTHON_RUNTIME,
    ) -> None:
        self._runtime_dir = os.path.abspath(runtime_dir) if runtime_dir else _default_runtime_dir()
        os.makedirs(self._runtime_dir, exist_ok=True)
        self._bundle_runtime_dir = os.path.abspath(bundle_runtime_dir) if bundle_runtime_dir else ""
        self._default_runtime = PythonRuntimeId.parse(default_version)
        runtime_release(self._default_runtime)

    @property
    def default_version(self) -> str:
        return self._default_runtime.series

    @staticmethod
    def supported_versions() -> list[str]:
        return [runtime.series for runtime in SUPPORTED_PYTHON_RUNTIMES]

    def _runtime_id(
        self, version: str | PythonRuntimeId | None = None
    ) -> PythonRuntimeId:
        runtime_id = self._default_runtime if version is None else PythonRuntimeId.parse(version)
        runtime_release(runtime_id)
        return runtime_id

    def installed_runtime_dir(self) -> str:
        return self._runtime_dir

    def bundled_runtime_dirs(self) -> list[str]:
        dirs = []
        if self._bundle_runtime_dir:
            dirs.append(self._bundle_runtime_dir)
        dirs.extend([
            os.path.join(get_bundle_dir(), "InfernuxHubData", "runtime"),
            os.path.join(get_bundle_dir(), "runtime"),
            os.path.join(get_bundle_dir(), "_internal", "InfernuxHubData", "runtime"),
            os.path.join(get_bundle_dir(), "_internal", "runtime"),
            os.path.join(get_bundle_dir(), "payload", "InfernuxHubData", "runtime"),
            os.path.join(get_bundle_dir(), "payload", "runtime"),
            os.path.join(get_bundle_dir(), "payload", "_internal", "InfernuxHubData", "runtime"),
            os.path.join(get_bundle_dir(), "payload", "_internal", "runtime"),
        ])
        result: list[str] = []
        seen: set[str] = set()
        for path in dirs:
            norm = os.path.normcase(os.path.abspath(path))
            if norm in seen:
                continue
            seen.add(norm)
            result.append(path)
        return result

    def private_runtime_root(
        self, version: str | PythonRuntimeId | None = None
    ) -> str:
        runtime_id = self._runtime_id(version)
        return os.path.join(self.installed_runtime_dir(), runtime_id.directory_name)

    def private_runtime_python(
        self, version: str | PythonRuntimeId | None = None
    ) -> str:
        runtime_root = self.private_runtime_root(version)
        if sys.platform == "win32":
            return os.path.join(runtime_root, "python.exe")
        return os.path.join(runtime_root, "bin", "python")

    def runtime_archive_path(
        self, version: str | PythonRuntimeId | None = None
    ) -> str:
        runtime_id = self._runtime_id(version)
        return os.path.join(
            self.installed_runtime_dir(),
            runtime_archive_for_machine(runtime=runtime_id).name,
        )

    def bundled_runtime_bundle_paths(self) -> list[str]:
        bundle_name = _runtime_bundle_name()
        return [os.path.join(path, bundle_name) for path in self.bundled_runtime_dirs()]

    def installed_versions(self) -> list[str]:
        return [
            runtime.series
            for runtime in SUPPORTED_PYTHON_RUNTIMES
            if self.get_runtime_path(runtime)
        ]

    def has_runtime(
        self, version: str | PythonRuntimeId | None = None
    ) -> bool:
        return bool(self.get_runtime_path(version))

    def get_runtime_path(
        self, version: str | PythonRuntimeId | None = None
    ) -> Optional[str]:
        runtime_id = self._runtime_id(version)
        roots = [self.private_runtime_root(runtime_id)]
        for root in roots:
            candidate = _find_python_in_root(root)
            if (
                candidate
                and _is_python_version(candidate, runtime_id)
                and not _is_embedded_root(root)
                and is_current_private_runtime_root(root, runtime=runtime_id)
            ):
                return candidate
        return None

    def ensure_runtime(
        self,
        *,
        version: str | PythonRuntimeId | None = None,
        on_status: Optional[Callable[[str], None]] = None,
        allow_frozen_repair: bool = False,
    ) -> str:
        runtime_id = self._runtime_id(version)
        with self._runtime_lock(runtime_id):
            return self._ensure_runtime(runtime_id, on_status=on_status, allow_frozen_repair=allow_frozen_repair)

    @contextmanager
    def _runtime_lock(self, runtime_id: PythonRuntimeId):
        """One writer/copy per ABI; the OS releases ownership if a Hub exits."""
        path = Path(self._runtime_dir) / f".{runtime_id.directory_name}.lock"
        with path.open("a+b") as stream:
            stream.seek(0)
            if sys.platform == "win32":
                import msvcrt
                acquire = lambda: msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                release = lambda: msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                acquire = lambda: fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                release = lambda: fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
            try:
                acquire()
            except OSError as exc:
                raise PythonRuntimeError(f"Python {runtime_id.series} runtime is busy in another Hub operation.") from exc
            try:
                yield
            finally:
                release()

    def _ensure_runtime(self, runtime_id: PythonRuntimeId, *, on_status=None, allow_frozen_repair=False) -> str:
        python_exe = self.get_runtime_path(runtime_id)
        if python_exe:
            runtime_root = runtime_prefix(python_exe)
            has_build_support = _has_build_support(runtime_root, runtime_id)
            has_required_modules = self._has_modules(python_exe, *_REQUIRED_RUNTIME_MODULES)
            if has_build_support and has_required_modules:
                return python_exe
            if is_frozen() and not allow_frozen_repair:
                raise PythonRuntimeError(
                    f"The installed managed Python {runtime_id.series} runtime is missing build support or required packages.\n"
                    "Please reinstall Infernux Hub so the runtime can be prepared during installation."
                )
        return self._provision_managed_runtime(runtime_id, on_status=on_status)

    def create_project_runtime(
        self,
        dest_path: str,
        *,
        version: str | PythonRuntimeId | None = None,
        on_status: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Copy the full managed Python runtime to *dest_path* for a project.

        Each project owns its own complete Python copy so there is no need
        for virtual-environment indirection.
        """
        runtime_id = self._runtime_id(version)
        dest_path = os.path.abspath(dest_path)
        with self._runtime_lock(runtime_id):
            return self._create_project_runtime(dest_path, runtime_id, on_status=on_status)

    def _create_project_runtime(self, dest_path: str, runtime_id: PythonRuntimeId, *, on_status=None) -> str:
        _emit_status(
            on_status, f"Checking managed Python {runtime_id.series} runtime..."
        )
        if not self.has_runtime(runtime_id):
            raise PythonRuntimeError(
                f"Python {runtime_id.series} is not installed in Infernux Hub.\n"
                f"Install Python {runtime_id.series} from the Installs page first."
            )
        self._ensure_runtime(
            runtime_id,
            allow_frozen_repair=is_frozen(),
            on_status=on_status,
        )
        source = self.private_runtime_root(runtime_id)
        if not os.path.isdir(source):
            raise PythonRuntimeError(
                f"The managed Python {runtime_id.series} runtime directory does not exist.\n"
                f"Expected at: {source}"
            )

        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        try:
            if os.path.exists(dest_path):
                raise FileExistsError(dest_path)
            _emit_status(on_status, "Copying Python runtime into the project...")
            with runtime_publication(dest_path, replace_existing=False) as candidate:
                _copy_project_runtime_tree(source, str(candidate))
                python = _find_python_in_root(str(candidate))
                if not python or not _is_python_version(python, runtime_id):
                    raise PythonRuntimeError("The copied project Python could not be started.")
                final_python = os.path.join(dest_path, "python.exe" if sys.platform == "win32" else "bin/python")
                self._relocate_runtime_scripts(python, final_python)
        except OSError as exc:
            raise PythonRuntimeError(
                f"Failed to copy the managed Python runtime to {dest_path}.\n{exc}"
            ) from exc

        if sys.platform == "win32":
            project_python = os.path.join(dest_path, "python.exe")
        else:
            project_python = os.path.join(dest_path, "bin", "python")

        if not os.path.isfile(project_python):
            raise PythonRuntimeError(
                f"Runtime copy finished, but Python was not found at {project_python}."
            )
        return project_python

    def _provision_managed_runtime(
        self,
        version: str | PythonRuntimeId,
        *,
        on_status: Optional[Callable[[str], None]] = None,
    ) -> str:
        runtime_id = self._runtime_id(version)
        bundled_python = self._seed_runtime_from_bundle(
            version=runtime_id, on_status=on_status
        )
        if bundled_python:
            return bundled_python
        return self._extract_runtime_to_root(
            self.private_runtime_root(runtime_id), version=runtime_id, on_status=on_status,
        )

    def _prepare_candidate(self, candidate: Path, runtime_id: PythonRuntimeId,
                           *, on_status=None) -> None:
        python = _find_python_in_root(str(candidate))
        if not python or not _is_python_version(python, runtime_id) or _is_embedded_root(str(candidate)):
            raise PythonRuntimeError(
                f"Private Python {runtime_id.series} extraction completed, but a valid full runtime was not found afterwards."
            )
        self._prepare_managed_runtime(python, runtime_id, on_status=on_status)
        self._relocate_runtime_scripts(python, self.private_runtime_python(runtime_id))

    @staticmethod
    def _relocate_runtime_scripts(python_exe: str, final_python: str) -> None:
        _run_command([python_exe, "-I", "-c", RELOCATE_RUNTIME_SCRIPTS, final_python],
                     timeout=120, raise_on_error=True)

    def _seed_runtime_from_bundle(
        self, *, version: str | PythonRuntimeId | None = None,
        on_status: Optional[Callable[[str], None]] = None,
    ) -> Optional[str]:
        runtime_id = self._runtime_id(version)
        target = self.private_runtime_root(runtime_id)

        def prepare(candidate: Path) -> None:
            if not is_current_private_runtime_root(candidate, runtime=runtime_id):
                raise PythonRuntimeError(f"The bundled Python {runtime_id.series} runtime has an invalid release marker.")
            self._prepare_candidate(candidate, runtime_id, on_status=on_status)

        for source_root in self.bundled_runtime_dirs():
            source = Path(source_root) / runtime_id.directory_name
            if not source.exists():
                continue
            _emit_status(on_status, f"Copying bundled Python {runtime_id.series} runtime...")
            with runtime_publication(target) as candidate:
                _copy_tree(str(source), str(candidate))
                prepare(candidate)
            return self.private_runtime_python(runtime_id)

        for bundle_path in self.bundled_runtime_bundle_paths():
            if not os.path.isfile(bundle_path):
                continue
            prefix = runtime_id.directory_name + "/"
            try:
                with zipfile.ZipFile(bundle_path, "r") as bundle:
                    members = [m for m in bundle.infolist() if m.filename.replace("\\", "/").startswith(prefix)]
                    if not members:
                        continue
                    paths = []
                    seen = set()
                    # Reject ambiguous ZIP identities before writing any member.
                    for member in members:
                        name = member.filename[len(prefix):].removesuffix("/")
                        if not name and member.is_dir():
                            continue
                        parts = name.split("/")
                        if ("\\" in member.filename or ":" in name or not name
                                or any(part in {"", ".", ".."} or part.rstrip(". ") != part for part in parts)):
                            raise PythonRuntimeError(f"Invalid bundled runtime path: {member.filename!r}")
                        key = os.path.normcase(name)
                        if key in seen:
                            raise PythonRuntimeError(f"Duplicate bundled runtime path: {member.filename!r}")
                        seen.add(key)
                        paths.append((member, Path(*parts)))
                    _emit_status(on_status, f"Extracting bundled Python {runtime_id.series} runtime...")
                    with runtime_publication(target) as candidate:
                        candidate.mkdir()
                        for member, relative in paths:
                            path = candidate / relative
                            if member.is_dir():
                                path.mkdir(parents=True, exist_ok=True)
                                continue
                            path.parent.mkdir(parents=True, exist_ok=True)
                            with bundle.open(member) as reader, path.open("wb") as writer:
                                shutil.copyfileobj(reader, writer)
                            if sys.platform != "win32" and member.create_system == 3:
                                os.chmod(path, (member.external_attr >> 16) & 0o777)
                        prepare(candidate)
            except (OSError, zipfile.BadZipFile) as exc:
                raise PythonRuntimeError(f"The bundled Python {runtime_id.series} runtime is invalid.\n{exc}") from exc
            return self.private_runtime_python(runtime_id)
        return None

    def _prepare_managed_runtime(
        self,
        python_exe: str,
        version: str | PythonRuntimeId | None = None,
        *,
        on_status: Optional[Callable[[str], None]] = None,
    ) -> None:
        runtime_id = self._runtime_id(version)
        runtime_root = runtime_prefix(python_exe)
        if _is_embedded_root(runtime_root):
            raise PythonRuntimeError(
                f"Infernux Hub requires a full Python {runtime_id.series} runtime for Nuitka builds, but an embeddable runtime was detected."
            )
        self._ensure_runtime_build_support(
            runtime_root, runtime_id, on_status=on_status
        )
        self._ensure_pip(python_exe, on_status=on_status)
        self._ensure_runtime_packages(
            python_exe, runtime_id, on_status=on_status
        )

    def _ensure_runtime_archive(
        self,
        version: str | PythonRuntimeId | None = None,
        *,
        on_status: Optional[Callable[[str], None]] = None,
    ) -> str:
        runtime_id = self._runtime_id(version)
        archive = runtime_archive_for_machine(runtime=runtime_id)
        archive_path = self.runtime_archive_path(runtime_id)
        os.makedirs(self.installed_runtime_dir(), exist_ok=True)

        if os.path.isfile(archive_path):
            try:
                verify_runtime_archive(archive_path, archive.sha256)
                return archive_path
            except RuntimeError:
                os.remove(archive_path)

        _emit_status(
            on_status,
            f"Downloading isolated Python {runtime_id.series} runtime for {platform.machine()}...",
        )
        tmp_path = archive_path + ".tmp"
        try:
            if os.path.isfile(tmp_path):
                os.remove(tmp_path)
            _download_file(
                archive.url,
                tmp_path,
                user_agent="Infernux-Hub/1.0",
            )
            verify_runtime_archive(tmp_path, archive.sha256)
            os.replace(tmp_path, archive_path)
        except urllib.error.URLError as exc:
            if "unknown url type: https" in str(exc).lower():
                raise PythonRuntimeError(
                    f"Failed to download the private Python {runtime_id.series} runtime because HTTPS support is unavailable in the packaged Hub."
                ) from exc
            raise PythonRuntimeError(
                f"Failed to download the private Python {runtime_id.series} runtime.\n{exc}"
            ) from exc
        except OSError as exc:
            raise PythonRuntimeError(
                f"Failed to download the private Python {runtime_id.series} runtime.\n{exc}"
            ) from exc
        except RuntimeError as exc:
            raise PythonRuntimeError(str(exc)) from exc
        finally:
            if os.path.isfile(tmp_path):
                os.remove(tmp_path)

        return archive_path

    def _extract_runtime_to_root(
        self,
        runtime_root: str,
        *,
        version: str | PythonRuntimeId | None = None,
        on_status: Optional[Callable[[str], None]] = None,
    ) -> str:
        runtime_id = self._runtime_id(version)
        expected_root = os.path.normcase(
            os.path.realpath(self.private_runtime_root(runtime_id))
        )
        requested_root = os.path.normcase(os.path.realpath(runtime_root))
        if requested_root != expected_root:
            raise PythonRuntimeError(
                "Refusing to deploy the private Python runtime outside the Hub-owned "
                f"{runtime_id.directory_name} directory."
            )
        archive_path = self._ensure_runtime_archive(
            runtime_id, on_status=on_status
        )
        os.makedirs(os.path.dirname(runtime_root), exist_ok=True)
        _emit_status(
            on_status,
            f"Extracting private Python {runtime_id.series} runtime...",
        )

        try:
            archive = runtime_archive_for_machine(runtime=runtime_id)
            extract_runtime_archive(
                archive_path,
                runtime_root,
                expected_sha256=archive.sha256,
                runtime=runtime_id,
                validate=lambda candidate: self._prepare_candidate(candidate, runtime_id, on_status=on_status),
            )
        except RuntimeError as exc:
            raise PythonRuntimeError(str(exc)) from exc

        return self.private_runtime_python(runtime_id)

    def reinstall_runtime(
        self,
        version: str | PythonRuntimeId | None = None,
        *,
        on_status: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Replace the Hub-owned runtime from a verified bundled/downloaded archive."""
        runtime_id = self._runtime_id(version)
        with self._runtime_lock(runtime_id):
            return self._provision_managed_runtime(runtime_id, on_status=on_status)

    def _ensure_runtime_build_support(
        self,
        runtime_root: str,
        version: str | PythonRuntimeId | None = None,
        *,
        on_status: Optional[Callable[[str], None]] = None,
    ) -> None:
        runtime_id = self._runtime_id(version)
        if _has_build_support(runtime_root, runtime_id):
            return

        raise PythonRuntimeError(
            f"Managed Python {runtime_id.series} is missing CPython build support files "
            "(Python.h / Python link library).\n"
            "Reinstall Infernux Hub or rebuild the bundled runtime so these files are available."
        )

    def _ensure_pip(self, python_exe: str, *, on_status: Optional[Callable[[str], None]] = None) -> None:
        completed = _run_command([python_exe, "-I", "-m", "pip", "--version"], timeout=60, raise_on_error=False)
        if completed.returncode == 0:
            return

        _emit_status(on_status, "Installing pip into the managed Python runtime...")
        completed = _run_command(
            [python_exe, "-I", "-m", "ensurepip", "--upgrade"],
            timeout=600,
            raise_on_error=False,
        )
        if completed.returncode != 0:
            raise PythonRuntimeError(
                "Failed to install pip into the managed Python runtime.\n"
                f"{(completed.stderr or completed.stdout or '').strip()}"
            )

    def _ensure_runtime_packages(
        self,
        python_exe: str,
        version: str | PythonRuntimeId | None = None,
        *,
        on_status: Optional[Callable[[str], None]] = None,
    ) -> None:
        self._runtime_id(version)
        if self._has_modules(python_exe, *_REQUIRED_RUNTIME_MODULES):
            return

        _emit_status(on_status, "Installing managed runtime support packages...")
        args = [
            python_exe,
            "-I",
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-input",
            "--prefer-binary",
            "--no-compile",
            "--upgrade",
        ]
        args.extend(_RUNTIME_PACKAGES)
        completed = _run_command(args, timeout=1800, raise_on_error=False)
        if completed.returncode != 0:
            raise PythonRuntimeError(
                "Failed to install support packages into the managed Python runtime.\n"
                f"{(completed.stderr or completed.stdout or '').strip()}"
            )

        if not self._has_modules(python_exe, *_REQUIRED_RUNTIME_MODULES):
            raise PythonRuntimeError(
                "Managed Python runtime is still missing required support packages after installation."
            )

    def _has_modules(self, python_exe: str, *module_names: str) -> bool:
        checks = " and ".join(
            [f"importlib.util.find_spec('{module_name}') is not None" for module_name in module_names]
        )
        completed = _run_command(
            [python_exe, "-I", "-c", f"import importlib.util; print(int({checks}))"],
            timeout=30,
            raise_on_error=False,
        )
        return completed.returncode == 0 and (completed.stdout or "").strip() == "1"
