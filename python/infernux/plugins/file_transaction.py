"""File publication shared by package installs and generated package members."""

from __future__ import annotations

import importlib.util
import os
import shutil
import stat
import uuid

from pathlib import Path
from typing import Iterable

from infernux.engine.path_utils import is_path_within, resolved_path, same_path


def prune_package_directories(project_root: str, removed_paths: Iterable[str]) -> None:
    """Remove empty managed ancestors while retaining all surviving contents."""
    boundaries = (os.path.join(project_root, "Packages"), os.path.join(project_root, "Assets", "Plugins"))
    roots = {resolved_path(os.path.dirname(path)) for path in removed_paths if path}
    for root in sorted(roots, key=lambda path: len(Path(path).parts), reverse=True):
        boundary = next((candidate for candidate in boundaries
                         if is_path_within(root, candidate, allow_root=False)), "")
        if not boundary:
            continue
        current = root
        while is_path_within(current, boundary, allow_root=False):
            try:
                os.rmdir(current)
            except OSError:
                break
            current = os.path.dirname(current)


def script_bytecode_paths(source: str) -> tuple[str, ...]:
    """Derived caches of this owned source, never other author files."""
    if not source.endswith(".py"):
        return ()
    directory = os.path.join(os.path.dirname(source), "__pycache__")
    if not os.path.isdir(directory) or os.path.islink(directory):
        return ()
    result = []
    for entry in os.scandir(directory):
        if entry.is_symlink() or not entry.is_file() or not entry.name.endswith(".pyc"):
            continue
        try:
            cached_source = importlib.util.source_from_cache(entry.path)
        except ValueError:
            continue
        if same_path(cached_source, source):
            result.append(entry.path)
    return tuple(sorted(result))


class InstallTransaction:
    def __init__(self, project_root: str) -> None:
        self.project_root = project_root
        self.id = uuid.uuid4().hex
        self.root = os.path.join(project_root, "Cache", "Plugins", ".transactions", self.id)
        self.backups: list[tuple[str, str | None]] = []
        self.committed = False
        os.makedirs(self.root, exist_ok=True)

    def write(self, destination: str, payload: bytes, *, mode: int | None = None) -> None:
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        backup = None
        if os.path.exists(destination):
            backup = os.path.join(self.root, f"backup-{len(self.backups)}")
            shutil.copy2(destination, backup)
            if mode is None:
                mode = stat.S_IMODE(os.stat(destination).st_mode)
        temporary = os.path.join(self.root, f"write-{len(self.backups)}")
        with open(temporary, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if mode is not None:
            os.chmod(temporary, stat.S_IMODE(mode))
        os.replace(temporary, destination)
        self.backups.append((destination, backup))

    def remove(self, destination: str) -> None:
        if not os.path.exists(destination):
            return
        backup = os.path.join(self.root, f"backup-{len(self.backups)}")
        shutil.copy2(destination, backup)
        self.backups.append((destination, backup))
        os.remove(destination)

    def commit(self) -> None:
        self.committed = True
        shutil.rmtree(self.root, ignore_errors=True)

    def rollback(self) -> None:
        if self.committed:
            return
        for destination, backup in reversed(self.backups):
            if backup and os.path.isfile(backup):
                os.replace(backup, destination)
            else:
                try:
                    os.remove(destination)
                except FileNotFoundError:
                    pass
        prune_package_directories(self.project_root, (destination for destination, backup in self.backups if backup is None))
        shutil.rmtree(self.root, ignore_errors=True)
