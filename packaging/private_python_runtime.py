from __future__ import annotations

import json
import logging
import os
import platform
import shutil
import sys
import tarfile
import tempfile
import time
from dataclasses import dataclass
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path

from python_runtime_catalog import (
    DEFAULT_PYTHON_RUNTIME,
    PythonRuntimeId,
    SUPPORTED_PYTHON_RUNTIMES,
    runtime_release,
)

_DEFAULT_RELEASE = runtime_release(DEFAULT_PYTHON_RUNTIME)
PYTHON_VERSION = _DEFAULT_RELEASE.patch_version
PYTHON_BUILD_RELEASE = _DEFAULT_RELEASE.build_release
PRIVATE_RUNTIME_MARKER = ".infernux-private-python-runtime.json"
_PRIVATE_RUNTIME_MARKER_KEYS = {
    "owner",
    "kind",
    "python_version",
    "python_series",
    "source_archive",
    "written_at",
}


@dataclass(frozen=True)
class RuntimeArchive:
    name: str
    url: str


def runtime_prefix(python_exe: str) -> str:
    directory = os.path.dirname(python_exe)
    return directory if sys.platform == "win32" else os.path.dirname(directory)


def has_runtime_build_support(root: str, runtime: str | PythonRuntimeId) -> bool:
    runtime_id = PythonRuntimeId.parse(runtime)
    if sys.platform == "win32":
        header = Path(root) / "include" / "Python.h"
        libraries = Path(root) / "libs"
        names = [f"{runtime_id.windows_library_stem}.lib", "python3.lib"]
    else:
        header = Path(root) / "include" / runtime_id.unix_library_stem / "Python.h"
        libraries = Path(root) / "lib"
        stem = f"lib{runtime_id.unix_library_stem}"
        names = [f"{stem}.dylib", "libpython3.dylib"] if sys.platform == "darwin" else [
            f"{stem}.so", f"{stem}.so.1.0", "libpython3.so",
        ]
    return header.is_file() and any((libraries / name).is_file() for name in names)


def runtime_archive_for_machine(
    *,
    system: str | None = None,
    machine: str | None = None,
    runtime: str | PythonRuntimeId = DEFAULT_PYTHON_RUNTIME,
) -> RuntimeArchive:
    release = runtime_release(runtime)
    platform_name = (system or sys.platform).lower()
    architecture = (
        machine
        or platform.machine()
        or os.environ.get("PROCESSOR_ARCHITECTURE")
        or ""
    ).lower()

    if platform_name in {"win32", "windows"}:
        if architecture in {"amd64", "x86_64"}:
            target = "x86_64-pc-windows-msvc"
        elif architecture in {"x86", "i386", "i686"}:
            target = "i686-pc-windows-msvc"
        elif architecture in {"arm64", "aarch64"}:
            target = "aarch64-pc-windows-msvc"
        else:
            raise RuntimeError(
                f"No isolated Infernux Python runtime archive is published for Windows {architecture or 'unknown'}."
            )
    elif platform_name in {"darwin", "macos"}:
        if architecture in {"arm64", "aarch64"}:
            target = "aarch64-apple-darwin"
        elif architecture in {"amd64", "x86_64"}:
            target = "x86_64-apple-darwin"
        else:
            raise RuntimeError(
                f"No isolated Infernux Python runtime archive is published for macOS {architecture or 'unknown'}."
            )
    elif platform_name.startswith("linux"):
        if architecture in {"arm64", "aarch64"}:
            target = "aarch64-unknown-linux-gnu"
        elif architecture in {"amd64", "x86_64"}:
            target = "x86_64-unknown-linux-gnu"
        else:
            raise RuntimeError(
                f"No isolated Infernux Python runtime archive is published for Linux {architecture or 'unknown'}."
            )
    else:
        raise RuntimeError(f"Unsupported platform for the Infernux private Python runtime: {platform_name}")

    name = (
        f"cpython-{release.patch_version}+{release.build_release}-{target}-"
        "install_only.tar.gz"
    )
    return RuntimeArchive(
        name=name,
        url=(
            "https://github.com/astral-sh/python-build-standalone/releases/download/"
            f"{release.build_release}/{name.replace('+', '%2B')}"
        ),
    )


def write_private_runtime_marker(
    runtime_root: str | os.PathLike[str],
    archive_name: str,
    *,
    runtime: str | PythonRuntimeId = DEFAULT_PYTHON_RUNTIME,
) -> None:
    release = runtime_release(runtime)
    root = Path(runtime_root)
    marker = root / PRIVATE_RUNTIME_MARKER
    payload = {
        "owner": "Infernux Hub",
        "kind": "private-python-runtime",
        "python_version": release.patch_version,
        "python_series": release.runtime_id.series,
        "source_archive": archive_name,
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    temporary = marker.with_suffix(marker.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, marker)


def is_private_runtime_root(runtime_root: str | os.PathLike[str]) -> bool:
    marker = Path(runtime_root) / PRIVATE_RUNTIME_MARKER
    if not marker.is_file():
        return False
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        isinstance(payload, dict)
        and set(payload) == _PRIVATE_RUNTIME_MARKER_KEYS
        and payload.get("owner") == "Infernux Hub"
        and payload.get("kind") == "private-python-runtime"
    )


def is_current_private_runtime_root(
    runtime_root: str | os.PathLike[str],
    *,
    runtime: str | PythonRuntimeId = DEFAULT_PYTHON_RUNTIME,
) -> bool:
    marker = Path(runtime_root) / PRIVATE_RUNTIME_MARKER
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
        release = runtime_release(runtime)
        archive = runtime_archive_for_machine(runtime=release.runtime_id)
    except (OSError, json.JSONDecodeError, RuntimeError, ValueError):
        return False
    return (
        isinstance(payload, dict)
        and set(payload) == _PRIVATE_RUNTIME_MARKER_KEYS
        and payload.get("owner") == "Infernux Hub"
        and payload.get("kind") == "private-python-runtime"
        and payload.get("python_version") == release.patch_version
        and payload.get("python_series") == release.runtime_id.series
        and payload.get("source_archive") == archive.name
    )


def _replace_runtime_directory(source: Path, destination: Path) -> None:
    # Windows can briefly retain image/scanner handles after the candidate
    # interpreter exits. Retry the atomic rename, never recopy or delete the
    # live runtime. Persistent permissions/sharing failures still propagate.
    delays = (0.05, 0.1, 0.2, 0.4, 0.8, 0.8)
    for attempt in range(len(delays) + 1):
        try:
            os.replace(source, destination)
            return
        except OSError as exc:
            if (sys.platform != "win32" or getattr(exc, "winerror", None) not in {5, 32, 33}
                    or attempt == len(delays)):
                raise
            time.sleep(delays[attempt])


@contextmanager
def runtime_publication(destination: str | os.PathLike[str], *, replace_existing: bool = True):
    """Prepare an owned candidate, then replace the live tree on successful exit."""
    target = Path(destination).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    extract_root = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.extract-", dir=target.parent)
    )
    backup = extract_root / "previous-runtime"
    committed = False
    try:
        unpacked_runtime = extract_root / "python"
        yield unpacked_runtime
        if not unpacked_runtime.is_dir():
            raise RuntimeError("Runtime publication requires a prepared python/ directory.")

        # Keep the live tree until extraction, its marker, and validation have
        # all succeeded. A failed rename restores the previous tree once;
        # this is transaction rollback, never a second installation attempt.
        if target.exists():
            if not replace_existing:
                raise FileExistsError(str(target))
            _replace_runtime_directory(target, backup)
        try:
            _replace_runtime_directory(unpacked_runtime, target)
        except BaseException:
            if backup.exists():
                try:
                    _replace_runtime_directory(backup, target)
                except OSError as exc:
                    raise RuntimeError(
                        f"Runtime publication and restoration failed; the previous runtime is preserved at {backup}"
                    ) from exc
            raise
        committed = True
    finally:
        # A failed restore must leave its backup available for recovery. Once
        # committed, cleanup failure cannot turn a valid install into failure.
        if committed or not backup.exists():
            try:
                shutil.rmtree(extract_root)
            except OSError as exc:
                logging.getLogger(__name__).warning(
                    "Could not clean runtime extraction directory %s: %s", extract_root, exc,
                )


def extract_runtime_archive(
    archive_path: str | os.PathLike[str],
    destination: str | os.PathLike[str],
    *,
    runtime: str | PythonRuntimeId = DEFAULT_PYTHON_RUNTIME,
    validate: Callable[[Path], None] | None = None,
) -> None:
    archive = Path(archive_path).resolve()
    if not archive.is_file():
        raise RuntimeError(f"Private Python runtime archive not found: {archive}")
    with runtime_publication(destination) as candidate:
        try:
            with tarfile.open(archive, mode="r:gz") as package:
                package.extractall(candidate.parent, filter="data")
        except (tarfile.TarError, OSError) as exc:
            raise RuntimeError(f"Invalid private Python runtime archive: {archive}") from exc
        if not candidate.is_dir():
            raise RuntimeError(
                "Unexpected private Python runtime archive layout: missing the python/ root."
            )
        write_private_runtime_marker(candidate, archive.name, runtime=runtime)
        if validate is not None:
            validate(candidate)


def prune_runtime_staging_cache(
    runtime_cache_root: str | os.PathLike[str],
    *,
    runtime: str | PythonRuntimeId = DEFAULT_PYTHON_RUNTIME,
) -> None:
    """Keep only the target ABI inputs in the packaging runtime cache.

    This function only operates on the explicitly supplied packaging cache,
    never on Hub's managed user-runtime directory. Keep only the pinned
    runtime being staged so unrelated ABI output cannot enter a fresh Hub
    installer.
    """
    root = Path(runtime_cache_root)
    if not root.is_dir():
        return
    target_runtime = PythonRuntimeId.parse(runtime)
    for runtime_id in SUPPORTED_PYTHON_RUNTIMES:
        if runtime_id == target_runtime:
            continue
        stale_runtime = root / runtime_id.directory_name
        if stale_runtime.is_dir():
            shutil.rmtree(stale_runtime)

    # Current Hub runtimes are staged from verified install-only archives.
    # Installers are not inputs to the current install-only archive workflow.
    for pattern in ("python-*.exe", "python-*.pkg", "python-*.msi"):
        for artifact in root.glob(pattern):
            if artifact.is_file():
                artifact.unlink()

    expected_archive = runtime_archive_for_machine(runtime=target_runtime).name
    for artifact in root.glob("cpython-*-install_only.tar.gz*"):
        if artifact.is_file() and artifact.name != expected_archive:
            artifact.unlink()
