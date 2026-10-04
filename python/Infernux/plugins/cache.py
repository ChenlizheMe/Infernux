"""Hub-owned shared storage for downloaded InxPackage versions."""

from __future__ import annotations

import filecmp
import json
import os
import re
import shutil
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping
from urllib.parse import quote

from Infernux.engine.path_utils import resolved_path, same_path
from Infernux.engine.user_data import DATA_ROOT_ENV, get_infernux_data_root

from Infernux.engine.player_package_native import read_entry

from .package import PACKAGE_EXTENSION, PACKAGE_MANIFEST, InxPackage, portable_meta_bytes


_STAGING_NAME = re.compile(r"^[a-z0-9-]+-(\d+)-[a-z0-9_]+$")
PACKAGE_CACHE_ROOT_ENV = "INFERNUX_PACKAGE_CACHE_ROOT"


def _same_authored_package_contents(first: str, second: str) -> bool:
    """Compare version identity without container layout or derived sidecars.

    Legacy packages retain content_hash and other importer observations. Those
    fields are removed at installation by the same sidecar normalization used
    here. Authored GUIDs, source bytes and effective importer settings remain
    immutable, including Mesh readability and texture sampling settings.
    """
    left, right = InxPackage.inspect(first), InxPackage.inspect(second)
    if left.metadata != right.metadata:
        return False
    left_paths = {entry["path"] for entry in left.entries}
    right_paths = {entry["path"] for entry in right.entries}
    if left_paths != right_paths:
        return False
    sidecars = {
        record["meta_archive_path"]: record["logical_path"]
        for record in left.file_records
    }
    for path in sorted(left_paths - {PACKAGE_MANIFEST}):
        left_bytes, right_bytes = read_entry(first, path), read_entry(second, path)
        if path in sidecars:
            left_bytes = portable_meta_bytes(left_bytes, sidecars[path])
            right_bytes = portable_meta_bytes(right_bytes, sidecars[path])
        if left_bytes != right_bytes:
            return False
    return True


def package_cache_root() -> str:
    """Return the Hub library shared by every Editor project."""

    configured = os.environ.get(PACKAGE_CACHE_ROOT_ENV, "").strip()
    if configured:
        return resolved_path(os.path.expandvars(os.path.expanduser(configured)))
    return resolved_path(
        os.path.join(
            os.path.expandvars(os.path.expanduser(os.environ["INFERNUX_SHARED_DATA_ROOT"]))
            if os.environ.get("INFERNUX_SHARED_DATA_ROOT", "").strip()
            else get_infernux_data_root(),
            "Library", "Plugins",
        )
    )


def _process_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))
        if not handle:
            return False
        ctypes.windll.kernel32.CloseHandle(handle)
        return True
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _encoded_segment(value: str, label: str) -> str:
    normalized = str(value).strip()
    if not normalized or normalized in {".", ".."}:
        raise ValueError(f"InxPackage cache {label} is invalid: {value!r}")
    return quote(normalized, safe="-._~")


class SharedPackageCache:
    """Store one package per reference/version without hashing the whole archive."""

    def __init__(
        self,
        root: str | os.PathLike[str] | None = None,
        *,
        staging_root: str | os.PathLike[str] | None = None,
    ) -> None:
        self.root = resolved_path(root or package_cache_root())
        self.package_root = resolved_path(os.path.join(self.root, "packages"))
        self.staging_root = resolved_path(
            staging_root or os.path.join(self.root, ".staging")
        )

    def _prune_staging(self) -> None:
        if not os.path.isdir(self.staging_root):
            return
        current_pid = os.getpid()
        for entry in Path(self.staging_root).iterdir():
            match = _STAGING_NAME.fullmatch(entry.name)
            if not match:
                continue
            owner_pid = int(match.group(1))
            if owner_pid == current_pid or _process_is_alive(owner_pid):
                continue
            if entry.is_dir():
                shutil.rmtree(entry)
            else:
                entry.unlink()

    @contextmanager
    def workspace(self, label: str):
        """Create one process-owned staging directory inside the shared cache."""

        normalized = str(label).strip().casefold().replace("_", "-")
        if not normalized or any(
            character not in "abcdefghijklmnopqrstuvwxyz0123456789-"
            for character in normalized
        ):
            raise ValueError(f"Invalid package staging label: {label!r}")
        os.makedirs(self.staging_root, exist_ok=True)
        self._prune_staging()
        workspace = tempfile.mkdtemp(
            prefix=f"{normalized}-{os.getpid()}-",
            dir=self.staging_root,
        )
        try:
            yield workspace
        finally:
            shutil.rmtree(workspace)

    @staticmethod
    def validate_location(location: str) -> str:
        value = str(location).replace("\\", "/")
        path = PurePosixPath(value)
        if (
            not value
            or any(part in {"", ".", ".."} for part in value.split("/"))
            or path.is_absolute()
            or not path.parts
            or path.parts[0] != "packages"
            or any(part in {"", ".", ".."} for part in path.parts)
            or path.suffix.casefold() != PACKAGE_EXTENSION
        ):
            raise ValueError(f"InxPackage cache location is invalid: {location!r}")
        return path.as_posix()

    def relative_path(self, reference: str, version: str) -> str:
        reference_parts = str(reference).split("/")
        if not reference_parts or any(not part.strip() for part in reference_parts):
            raise ValueError(f"InxPackage cache reference is invalid: {reference!r}")
        encoded_reference = "/".join(
            _encoded_segment(part, "reference") for part in reference_parts
        )
        encoded_version = _encoded_segment(version, "version")
        return f"packages/{encoded_reference}/{encoded_version}{PACKAGE_EXTENSION}"

    def path(self, reference: str, version: str) -> str:
        relative = self.relative_path(reference, version)
        return resolved_path(os.path.join(self.root, *relative.split("/")))

    def resolve(self, reference: str, version: str) -> str:
        destination = self.path(reference, version)
        return destination if os.path.isfile(destination) else ""

    def store(self, source: str, *, reference: str, version: str) -> str:
        source_path = resolved_path(source)
        destination = self.path(reference, version)
        if same_path(source_path, destination):
            return destination

        os.makedirs(os.path.dirname(destination), exist_ok=True)
        temporary = destination + f".tmp.{uuid.uuid4().hex}"
        try:
            shutil.copy2(source_path, temporary)
            try:
                # Publish once, atomically. Installed projects use this archive
                # as their update baseline; another download must not replace it.
                os.link(temporary, destination)
            except FileExistsError:
                if not filecmp.cmp(temporary, destination, shallow=False):
                    message = (
                        f"Plugin version is immutable: {reference}@{version}; "
                        "increment the package version before changing its content"
                    )
                    try:
                        equivalent = _same_authored_package_contents(temporary, destination)
                    except (OSError, ValueError, RuntimeError, KeyError) as exc:
                        raise ValueError(message) from exc
                    if not equivalent:
                        raise ValueError(message) from None
        finally:
            try:
                os.remove(temporary)
            except FileNotFoundError:
                pass
        return destination

    @staticmethod
    def _project_references(project_root: str | os.PathLike[str]) -> set[str]:
        project = Path(resolved_path(project_root))
        if not project.is_dir():
            raise FileNotFoundError(
                f"Cannot clean the package cache while a registered project is unavailable: {project}"
            )
        registry_path = project / "ProjectSettings" / "InxPlugins.json"
        if not registry_path.is_file():
            return set()
        try:
            document = json.loads(registry_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Cannot clean the package cache because a project registry is unreadable: {registry_path}"
            ) from exc
        if (
            not isinstance(document, Mapping)
            or document.get("$schema") != "infernux.plugin_registry"
            or not isinstance(document.get("packages"), list)
            or not isinstance(document.get("installed"), list)
        ):
            raise RuntimeError(
                f"Cannot clean the package cache because a project registry is invalid: {registry_path}"
            )
        locations: set[str] = set()
        for raw in [*document["packages"], *document["installed"]]:
            if not isinstance(raw, Mapping):
                raise RuntimeError(
                    f"Cannot clean the package cache because a package record is invalid: {registry_path}"
                )
            source = raw.get("source")
            if not isinstance(source, Mapping):
                continue
            if str(source.get("cache_scope", "")).casefold() != "hub":
                continue
            try:
                locations.add(
                    SharedPackageCache.validate_location(
                        str(source.get("cache_location", ""))
                    )
                )
            except ValueError as exc:
                raise RuntimeError(
                    f"Cannot clean the package cache because a project reference is invalid: {registry_path}"
                ) from exc
        return locations

    def prune_unreferenced(
        self,
        project_roots: Iterable[str | os.PathLike[str]],
        *,
        dry_run: bool = False,
    ) -> tuple[str, ...]:
        """Remove package versions unused by every explicitly registered project."""

        referenced: set[str] = set()
        for project_root in tuple(project_roots):
            referenced.update(self._project_references(project_root))

        candidates: list[Path] = []
        if os.path.isdir(self.package_root):
            for path in sorted(Path(self.package_root).rglob(f"*{PACKAGE_EXTENSION}")):
                relative = PurePosixPath(path.relative_to(self.root).as_posix()).as_posix()
                if relative not in referenced:
                    candidates.append(path)
        if not dry_run:
            for path in candidates:
                path.unlink()
        return tuple(str(path) for path in candidates)


__all__ = [
    "SharedPackageCache",
    "package_cache_root",
]
