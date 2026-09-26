"""Hub-owned Blender authoring tool used by the model importer."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import tarfile
import urllib.request
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable

from hub_utils import get_hub_shared_data_dir


BLENDER_VERSION = "5.2.2"
BLENDER_SERIES = "5.2"
BLENDER_SUPPORT_SCHEMA = "infernux.blender_support"
MANIFEST_NAME = "infernux-blender-support.json"
_RELEASE_ROOT = "https://download.blender.org/release/Blender5.2"
_ASSETS = {
    "linux-x64": (
        "blender-5.2.2-linux-x64.tar.xz",
        383_295_504,
        "84098912789dc450e95697c4184fb8a90acbe5111c2ba4aede3fecb57806a168",
    ),
    "windows-x64": (
        "blender-5.2.2-windows-x64.zip",
        404_453_484,
        "3849d17a682cba006075aaa3f3597ecb5c9c30ec31035b2e092c53e40679b535",
    ),
}


class BlenderSupportError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class BlenderSupportStatus:
    installed: bool
    root: Path
    executable: Path | None = None
    version: str = ""
    error: str = ""


def host_id() -> str:
    machine = platform.machine().casefold()
    if machine not in {"amd64", "x86_64"}:
        raise BlenderSupportError(
            f"Blender authoring support is not published for this architecture: {machine}"
        )
    if os.name == "nt":
        return "windows-x64"
    if os.name == "posix" and platform.system().casefold() == "linux":
        return "linux-x64"
    raise BlenderSupportError(
        f"Blender authoring support is not published for this host: {platform.system()}"
    )


def default_blender_support_root() -> Path:
    return (
        Path(get_hub_shared_data_dir())
        / "AuthoringTools"
        / "Blender"
        / BLENDER_VERSION
        / host_id()
    ).resolve()


def _executable_relative() -> Path:
    return Path("blender.exe" if os.name == "nt" else "blender")


def _portable_member(value: str) -> Path:
    path = PurePosixPath(value.replace("\\", "/"))
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise BlenderSupportError(f"Blender archive member is unsafe: {value}")
    return Path(*path.parts)


def _manifest(root: Path) -> dict[str, object]:
    try:
        document = json.loads((root / MANIFEST_NAME).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BlenderSupportError(f"Blender support manifest is unreadable: {root}") from exc
    expected = {"$schema", "version", "host", "executable", "source"}
    if type(document) is not dict or set(document) != expected:
        raise BlenderSupportError(f"Blender support manifest is invalid: {root}")
    if (
        document["$schema"] != BLENDER_SUPPORT_SCHEMA
        or document["version"] != BLENDER_VERSION
        or document["host"] != host_id()
    ):
        raise BlenderSupportError(f"Blender support does not match this Editor: {root}")
    executable = _portable_member(str(document["executable"]))
    if not (root / executable).is_file():
        raise BlenderSupportError(f"Blender executable is missing: {root / executable}")
    return document


def validate_blender_support(root: Path) -> Path:
    document = _manifest(root.expanduser().resolve())
    return root.expanduser().resolve() / _portable_member(str(document["executable"]))


def blender_support_environment(root: Path) -> dict[str, str]:
    return {"INFERNUX_BLENDER_EXECUTABLE": str(validate_blender_support(root))}


def _extract_archive(archive: Path, destination: Path) -> Path:
    unpack = destination / "unpack"
    unpack.mkdir(parents=True)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                relative = _portable_member(member.filename)
                target = (unpack / relative).resolve()
                if unpack not in target.parents and target != unpack:
                    raise BlenderSupportError(f"Blender archive escapes its root: {member.filename}")
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(member) as reader, target.open("wb") as writer:
                    shutil.copyfileobj(reader, writer)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as bundle:
            bundle.extractall(unpack, filter="data")
    else:
        raise BlenderSupportError(f"Blender archive is invalid: {archive}")
    roots = [entry for entry in unpack.iterdir() if entry.name != "__MACOSX"]
    if len(roots) != 1 or not roots[0].is_dir():
        raise BlenderSupportError("Blender archive must contain one versioned root directory")
    return roots[0]


class BlenderSupportManager:
    def __init__(self, root: str | os.PathLike[str] | None = None) -> None:
        self.root = Path(root).expanduser().resolve() if root else default_blender_support_root()

    def status(self) -> BlenderSupportStatus:
        if not self.root.exists():
            return BlenderSupportStatus(False, self.root)
        try:
            executable = validate_blender_support(self.root)
        except BlenderSupportError as exc:
            return BlenderSupportStatus(False, self.root, error=str(exc))
        return BlenderSupportStatus(True, self.root, executable, BLENDER_VERSION)

    def activate_environment(self) -> bool:
        status = self.status()
        if not status.installed or status.executable is None:
            return False
        os.environ["INFERNUX_BLENDER_EXECUTABLE"] = str(status.executable)
        return True

    def install(self, *, on_progress: Callable[[int, int], None] | None = None) -> str:
        name, expected_size, expected_digest = _ASSETS[host_id()]
        download = Path(get_hub_shared_data_dir()) / "Cache" / "Downloads" / "Blender" / name
        download.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        received = 0
        request = urllib.request.Request(
            f"{_RELEASE_ROOT}/{name}",
            headers={"User-Agent": "Infernux-Hub/1.0"},
        )
        with urllib.request.urlopen(request, timeout=120) as response, download.open("wb") as writer:
            while block := response.read(1024 * 1024):
                writer.write(block)
                digest.update(block)
                received += len(block)
                if on_progress is not None:
                    on_progress(received, expected_size)
        if received != expected_size or digest.hexdigest() != expected_digest:
            download.unlink(missing_ok=True)
            raise BlenderSupportError(
                f"Downloaded Blender {BLENDER_VERSION} archive failed its release integrity check"
            )
        try:
            return self.install_archive(download)
        finally:
            download.unlink(missing_ok=True)

    def install_archive(self, archive: str | os.PathLike[str]) -> str:
        source = Path(archive).expanduser().resolve()
        if not source.is_file():
            raise BlenderSupportError(f"Blender archive does not exist: {source}")
        self.root.parent.mkdir(parents=True, exist_ok=True)
        staging = self.root.parent / f".{self.root.name}.staging-{uuid.uuid4().hex}"
        backup = self.root.parent / f".{self.root.name}.backup-{uuid.uuid4().hex}"
        try:
            staging.mkdir()
            payload = _extract_archive(source, staging)
            executable = payload / _executable_relative()
            if not executable.is_file():
                raise BlenderSupportError(f"Blender archive is missing {_executable_relative()}")
            (payload / MANIFEST_NAME).write_text(
                json.dumps(
                    {
                        "$schema": BLENDER_SUPPORT_SCHEMA,
                        "version": BLENDER_VERSION,
                        "host": host_id(),
                        "executable": _executable_relative().as_posix(),
                        "source": f"{_RELEASE_ROOT}/{source.name}",
                    },
                    indent=2,
                ) + "\n",
                encoding="utf-8",
            )
            validate_blender_support(payload)
            if self.root.exists():
                os.replace(self.root, backup)
            try:
                os.replace(payload, self.root)
            except BaseException:
                if backup.exists() and not self.root.exists():
                    os.replace(backup, self.root)
                raise
            if backup.exists():
                shutil.rmtree(backup)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
            if backup.exists() and self.root.exists():
                shutil.rmtree(backup)
        self.activate_environment()
        return str(self.root)


__all__ = [
    "BLENDER_SERIES",
    "BLENDER_VERSION",
    "BlenderSupportError",
    "BlenderSupportManager",
    "BlenderSupportStatus",
    "blender_support_environment",
    "default_blender_support_root",
    "host_id",
    "validate_blender_support",
]
