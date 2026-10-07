"""Resolve and stage installed desktop dependencies without pip or a compiler.

Installed distribution metadata owns the dependency graph and file inventory.
The Player has no site initialization: editable installs and .pth hooks cannot
be silently copied as if they were self-contained runtime packages.
"""
from __future__ import annotations

import csv
from collections import deque
from dataclasses import dataclass
import importlib.metadata as metadata
import io
from pathlib import Path, PurePosixPath
import py_compile
import shutil
from typing import Iterable

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from .path_utils import resolved_path


@dataclass(frozen=True)
class DependencyFile:
    source: Path
    relative: str


@dataclass(frozen=True)
class RuntimeDependency:
    name: str
    version: str
    files: tuple[DependencyFile, ...]
    record_path: str


def _installed_files(distribution: metadata.Distribution) -> tuple[tuple[DependencyFile, ...], str]:
    name = distribution.metadata["Name"]
    record = distribution.read_text("RECORD")
    if record is None:
        raise RuntimeError(f"Player dependency has no installed RECORD inventory: {name}")
    base = Path(resolved_path(distribution.locate_file("")))
    result = []
    record_path = ""
    for row in csv.reader(io.StringIO(record)):
        if not row:
            continue
        relative = PurePosixPath(row[0])
        source = Path(resolved_path(distribution.locate_file(row[0])))
        # pip console wrappers are installation tools, not Player entrypoints.
        if not source.is_relative_to(base):
            if source.parent.name.casefold() in {"scripts", "bin"}:
                continue
            raise RuntimeError(f"Player dependency file is outside its install root: {name}: {row[0]}")
        if relative.is_absolute() or any(part in {"..", "."} for part in relative.parts):
            raise RuntimeError(f"Invalid Player dependency inventory path: {name}: {row[0]}")
        if "__pycache__" in relative.parts or relative.suffix in {".pyc", ".pyo", ".pyi"}:
            continue
        if relative.suffix in {".pth", ".egg-link"}:
            raise RuntimeError(f"Player dependency requires unsupported site initialization: {name}: {relative}")
        if relative.suffix.casefold() in {".exe", ".pdb", ".lib", ".exp", ".a"}:
            if relative.suffix.casefold() == ".exe":
                raise RuntimeError(f"Player dependency contains another executable: {name}: {relative}")
            continue
        if relative.parts[0].casefold() in {"infernux", "stdlib", "modules", "data"}:
            raise RuntimeError(f"Player dependency overlaps the engine runtime: {name}: {relative}")
        if relative.parts[0].endswith(".dist-info"):
            if relative.name == "RECORD":
                record_path = relative.as_posix()
                continue
            # Keep runtime metadata and licenses, never a machine-local install
            # URL/history. RECORD is regenerated for adjacent source-less pyc.
            if relative.name in {"direct_url.json", "INSTALLER", "REQUESTED"}:
                continue
        if not source.is_file():
            raise RuntimeError(f"Player dependency inventory file is missing: {name}: {relative}")
        result.append(DependencyFile(source, relative.as_posix()))
    if not record_path:
        raise RuntimeError(f"Player dependency has no dist-info RECORD entry: {name}")
    return tuple(sorted(result, key=lambda item: item.relative)), record_path


def resolve_runtime_dependencies(
    imports: Iterable[str],
    *,
    requirements: Iterable[str] = (),
    constraints: Iterable[str] = (),
    bundled: Iterable[str] = (),
    forbidden: Iterable[str] = (),
) -> tuple[RuntimeDependency, ...]:
    """Freeze the installed PEP 508 closure, including selected extras/markers.

    Constraints only restrict distributions reached from runtime roots; an
    Editor dependency in an enabled plugin does not become a runtime root.
    """
    bundled_names = {canonicalize_name(name) for name in bundled}
    forbidden_names = {canonicalize_name(name) for name in forbidden}
    providers = metadata.packages_distributions()
    pending = deque(Requirement(value) for value in requirements)
    for module in sorted(set(imports)):
        if canonicalize_name(module) in bundled_names:
            continue
        names = providers.get(module, ())
        if not names:
            raise RuntimeError(
                f"Player import has no installed distribution metadata: {module}. "
                "Install it as a package or include it in project Assets/Packages."
            )
        pending.extend(Requirement(name) for name in sorted(names))
    by_name: dict[str, list[Requirement]] = {}
    for value in constraints:
        requirement = Requirement(value)
        if requirement.marker is None or requirement.marker.evaluate():
            by_name.setdefault(canonicalize_name(requirement.name), []).append(requirement)
    selected: dict[str, metadata.Distribution] = {}
    extras: dict[str, set[str]] = {}
    while pending:
        requirement = pending.popleft()
        if requirement.marker is not None and not requirement.marker.evaluate():
            continue
        name = canonicalize_name(requirement.name)
        if name in forbidden_names:
            raise RuntimeError(f"Player dependency requires the disabled CPU JIT capability: {name}")
        try:
            distribution = selected.get(name) or metadata.distribution(name)
        except metadata.PackageNotFoundError as exc:
            raise RuntimeError(f"Player dependency is not installed: {requirement}") from exc
        restrictions = [requirement, *by_name.get(name, ())]
        for restriction in restrictions:
            if not restriction.specifier.contains(distribution.version, prereleases=True):
                raise RuntimeError(
                    f"Player dependency version conflict: {restriction}; installed {distribution.version}"
                )
            if restriction.url:
                import json

                origin = distribution.read_text("direct_url.json")
                if origin is None or json.loads(origin).get("url") != restriction.url:
                    raise RuntimeError(f"Player dependency source does not satisfy {restriction}")
        # Engine-owned packages come from the exact platform payload. Never
        # overlay the local site's numpy/Numba/engine implementation on it.
        if name in bundled_names:
            continue
        requested_extras = set().union(*(item.extras for item in restrictions))
        if name in selected and requested_extras <= extras[name]:
            continue
        selected[name] = distribution
        extras.setdefault(name, set()).update(requested_extras)
        contexts = {"", *extras[name]}
        for value in distribution.requires or ():
            dependency = Requirement(value)
            if dependency.marker is None or any(
                dependency.marker.evaluate({"extra": extra}) for extra in contexts
            ):
                # Its marker has already been evaluated in the parent's
                # extras context. The child must not evaluate it as extra="".
                dependency.marker = None
                pending.append(dependency)
    result = []
    destinations: dict[str, Path] = {}
    for name, distribution in sorted(selected.items()):
        files, record_path = _installed_files(distribution)
        for item in files:
            relative = item.relative + "c" if item.relative.endswith(".py") else item.relative
            previous = destinations.setdefault(relative.casefold(), item.source)
            if previous != item.source:
                raise RuntimeError(f"Player dependencies claim the same runtime path: {relative}")
        result.append(RuntimeDependency(name, distribution.version, files, record_path))
    return tuple(result)


def stage_runtime_dependencies(
    dependencies: Iterable[RuntimeDependency], destination: Path,
) -> tuple[str, ...]:
    """Write only the frozen inventory, compiling py with portable filenames."""
    destination.mkdir(parents=True, exist_ok=False)
    written: set[str] = set()
    for dependency in dependencies:
        installed = []
        for item in dependency.files:
            relative = item.relative + "c" if item.relative.endswith(".py") else item.relative
            target = destination / relative
            if relative not in written:
                target.parent.mkdir(parents=True, exist_ok=True)
                if item.relative.endswith(".py"):
                    py_compile.compile(
                        str(item.source), cfile=str(target), dfile=item.relative,
                        doraise=True, optimize=2,
                        invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH,
                    )
                else:
                    shutil.copyfile(item.source, target)
                written.add(relative)
            installed.append((relative, "", target.stat().st_size))
        record = destination / dependency.record_path
        record.parent.mkdir(parents=True, exist_ok=True)
        with record.open("w", encoding="utf-8", newline="") as stream:
            csv.writer(stream, lineterminator="\n").writerows(
                [*installed, (dependency.record_path, "", "")]
            )
        written.add(dependency.record_path)
    return tuple(sorted(written))
