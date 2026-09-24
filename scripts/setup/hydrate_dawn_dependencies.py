#!/usr/bin/env python3
"""Hydrate Dawn's pinned dependencies without unreviewed network fallbacks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any


LOCK_SCHEMA = "infernux.dawn_dependencies"
SNAPSHOT_SCHEMA = "infernux.dawn_official_snapshots"
AUDIT_SCHEMA = "infernux.dawn_dependency_audit"
LOCK_VERSION = 1
HEX40 = re.compile(r"[0-9a-f]{40}\Z")
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
TRANSPORTS = {"github", "official_gitiles_snapshot"}


class HydrationError(RuntimeError):
    """Raised when a dependency cannot pass the supply-chain gates."""


@dataclass(frozen=True)
class Dependency:
    path: str
    transport: str
    deps_url: str
    source_url: str
    commit: str
    tree: str


@dataclass(frozen=True)
class DependencyLock:
    path: Path
    dawn_revision: str
    dependencies: tuple[Dependency, ...]
    sha256: str


@dataclass(frozen=True)
class SnapshotArtifact:
    dependency: Dependency
    path: Path
    sha256: str


@dataclass
class PreparedDependency:
    dependency: Dependency
    path: Path | None
    hydration: str
    transport_sha256: str | None = None
    quarantined_path: Path | None = None


def _load_json(path: Path, description: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HydrationError(f"Unable to read {description} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise HydrationError(f"{description} must contain one JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise HydrationError(f"Unable to hash {path}: {exc}") from exc
    return digest.hexdigest()


def _require_string(mapping: dict[str, Any], key: str, context: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise HydrationError(f"{context}.{key} must be a non-empty string")
    return value


def _validate_relative_path(value: str, context: str) -> None:
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or not posix.parts
        or posix.parts[0] != "third_party"
        or any(part in {"", ".", ".."} for part in posix.parts)
        or "\\" in value
    ):
        raise HydrationError(f"{context} is not a safe third_party-relative path: {value!r}")


def load_lock(path: Path) -> DependencyLock:
    payload = _load_json(path, "dependency lock")
    if payload.get("schema") != LOCK_SCHEMA or payload.get("version") != LOCK_VERSION:
        raise HydrationError(f"Unsupported dependency lock schema/version: {path}")

    dawn_revision = _require_string(payload, "dawn_revision", "lock")
    if not HEX40.fullmatch(dawn_revision):
        raise HydrationError("lock.dawn_revision must be a lowercase 40-hex Git commit")
    expected_count = payload.get("dependency_count")
    expected_snapshot_count = payload.get("snapshot_dependency_count")
    if not isinstance(expected_count, int) or expected_count < 1:
        raise HydrationError("lock.dependency_count must be a positive integer")
    if not isinstance(expected_snapshot_count, int) or expected_snapshot_count < 0:
        raise HydrationError("lock.snapshot_dependency_count must be a non-negative integer")

    raw_dependencies = payload.get("dependencies")
    if not isinstance(raw_dependencies, list) or len(raw_dependencies) != expected_count:
        raise HydrationError(
            f"lock.dependencies must contain exactly {expected_count} entries"
        )

    dependencies: list[Dependency] = []
    seen_paths: set[str] = set()
    for index, raw in enumerate(raw_dependencies):
        context = f"lock.dependencies[{index}]"
        if not isinstance(raw, dict):
            raise HydrationError(f"{context} must be an object")
        relative = _require_string(raw, "path", context)
        _validate_relative_path(relative, f"{context}.path")
        if relative in seen_paths:
            raise HydrationError(f"Duplicate dependency path in lock: {relative}")
        seen_paths.add(relative)
        transport = _require_string(raw, "transport", context)
        if transport not in TRANSPORTS:
            raise HydrationError(f"Unsupported transport for {relative}: {transport}")
        deps_url = _require_string(raw, "deps_url", context)
        source_url = _require_string(raw, "source_url", context)
        commit = _require_string(raw, "commit", context)
        tree = _require_string(raw, "tree", context)
        if not HEX40.fullmatch(commit):
            raise HydrationError(f"Invalid commit for {relative}: {commit}")
        if not HEX40.fullmatch(tree):
            raise HydrationError(f"Invalid tree for {relative}: {tree}")
        dependencies.append(
            Dependency(relative, transport, deps_url, source_url, commit, tree)
        )

    sorted_paths = sorted(seen_paths)
    for index, relative in enumerate(sorted_paths):
        prefix = relative + "/"
        if any(candidate.startswith(prefix) for candidate in sorted_paths[index + 1 :]):
            raise HydrationError(f"Overlapping dependency paths are forbidden: {relative}")

    actual_snapshot_count = sum(
        dependency.transport == "official_gitiles_snapshot"
        for dependency in dependencies
    )
    if actual_snapshot_count != expected_snapshot_count:
        raise HydrationError(
            "lock.snapshot_dependency_count does not match the dependency entries"
        )
    return DependencyLock(
        path=path,
        dawn_revision=dawn_revision,
        dependencies=tuple(dependencies),
        sha256=_sha256(path),
    )


def _git(path: Path, *arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    for key in tuple(environment):
        if key.startswith("GIT_CONFIG_") or key in {
            "GIT_DIR",
            "GIT_WORK_TREE",
            "GIT_OBJECT_DIRECTORY",
            "GIT_ALTERNATE_OBJECT_DIRECTORIES",
            "GIT_COMMON_DIR",
            "GIT_INDEX_FILE",
            "GIT_TEMPLATE_DIR",
        }:
            environment.pop(key, None)
    environment.update(
        {
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_OPTIONAL_LOCKS": "0",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
        }
    )
    command = ["git", "-C", str(path), *arguments]
    try:
        return subprocess.run(
            command,
            check=check,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            env=environment,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = ""
        if isinstance(exc, subprocess.CalledProcessError):
            detail = (exc.stderr or exc.stdout or "").strip()
        suffix = f": {detail}" if detail else ""
        raise HydrationError(f"Git command failed for {path}: {' '.join(arguments)}{suffix}") from exc


def verify_repository(path: Path, dependency: Dependency) -> None:
    git_directory = path / ".git"
    if (
        path.is_symlink()
        or not path.is_dir()
        or git_directory.is_symlink()
        or not git_directory.is_dir()
    ):
        raise HydrationError(f"Dependency is not a Git worktree: {dependency.path}")
    remotes = _git(path, "remote").stdout.splitlines()
    if remotes != ["origin"]:
        raise HydrationError(
            f"Dependency must have exactly one origin remote: {dependency.path}: {remotes}"
        )
    origins = _git(path, "remote", "get-url", "--all", "origin").stdout.splitlines()
    if origins != [dependency.source_url]:
        raise HydrationError(
            f"Origin mismatch for {dependency.path}: expected {dependency.source_url!r}, got {origins!r}"
        )
    for relative in ("objects/info/alternates", "objects/info/http-alternates"):
        if (git_directory / relative).exists():
            raise HydrationError(
                f"External Git object alternates are forbidden: {dependency.path}"
            )
    replacements = _git(path, "for-each-ref", "--format=%(refname)", "refs/replace").stdout
    if replacements:
        raise HydrationError(f"Git replacement refs are forbidden: {dependency.path}")
    commit = _git(path, "rev-parse", "--verify", "HEAD^{commit}").stdout.strip()
    if commit != dependency.commit:
        raise HydrationError(
            f"Commit mismatch for {dependency.path}: expected {dependency.commit}, got {commit}"
        )
    tree = _git(path, "rev-parse", "--verify", "HEAD^{tree}").stdout.strip()
    if tree != dependency.tree:
        raise HydrationError(
            f"Tree mismatch for {dependency.path}: expected {dependency.tree}, got {tree}"
        )
    status = _git(path, "status", "--porcelain=v1", "--untracked-files=all").stdout
    if status:
        raise HydrationError(
            f"Dependency worktree is dirty: {dependency.path}: {status.strip()}"
        )
    _git(path, "fsck", "--strict")


def _repository_matches(path: Path, dependency: Dependency) -> bool:
    try:
        verify_repository(path, dependency)
    except HydrationError:
        return False
    return True


def _validate_tar_member(member: tarfile.TarInfo, destination: Path) -> None:
    normalized_name = member.name.replace("\\", "/")
    posix = PurePosixPath(normalized_name)
    windows = PureWindowsPath(member.name)
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or any(part == ".." for part in posix.parts)
    ):
        raise HydrationError(f"Snapshot archive contains an unsafe path: {member.name!r}")
    if member.islnk():
        raise HydrationError(f"Snapshot archive contains a hard link: {member.name!r}")
    if not (member.isfile() or member.isdir() or member.issym()):
        raise HydrationError(f"Snapshot archive contains a special file: {member.name!r}")
    data_filter = getattr(tarfile, "data_filter", None)
    if data_filter is None:
        raise HydrationError(
            "This Python runtime lacks tarfile.data_filter; refusing unsafe extraction"
        )
    try:
        data_filter(member, str(destination))
    except (tarfile.TarError, OSError, ValueError) as exc:
        raise HydrationError(
            f"Snapshot archive member is unsafe: {member.name!r}: {exc}"
        ) from exc


def _inspect_archive(path: Path, extraction_root: Path) -> None:
    try:
        with tarfile.open(path, mode="r:*") as archive:
            members = archive.getmembers()
            if not members:
                raise HydrationError(f"Snapshot archive is empty: {path}")
            for member in members:
                _validate_tar_member(member, extraction_root)
    except HydrationError:
        raise
    except (OSError, tarfile.TarError) as exc:
        raise HydrationError(f"Unable to inspect snapshot archive {path}: {exc}") from exc


def preflight_snapshots(
    snapshot_dir: Path,
    dependency_lock: DependencyLock,
    scratch_root: Path,
) -> dict[str, SnapshotArtifact]:
    manifest_path = snapshot_dir / "manifest.json"
    manifest = _load_json(manifest_path, "snapshot manifest")
    if manifest.get("schema") != SNAPSHOT_SCHEMA or manifest.get("version") != LOCK_VERSION:
        raise HydrationError(f"Unsupported snapshot manifest schema/version: {manifest_path}")
    if manifest.get("dawn_revision") != dependency_lock.dawn_revision:
        raise HydrationError("Snapshot manifest Dawn revision does not match the lock")

    required = {
        dependency.path: dependency
        for dependency in dependency_lock.dependencies
        if dependency.transport == "official_gitiles_snapshot"
    }
    raw_artifacts = manifest.get("dependencies")
    if not isinstance(raw_artifacts, list) or len(raw_artifacts) != len(required):
        raise HydrationError(
            f"Snapshot manifest must contain exactly {len(required)} dependencies"
        )

    artifacts: dict[str, SnapshotArtifact] = {}
    for index, raw in enumerate(raw_artifacts):
        context = f"snapshot.dependencies[{index}]"
        if not isinstance(raw, dict):
            raise HydrationError(f"{context} must be an object")
        relative = _require_string(raw, "path", context)
        dependency = required.get(relative)
        if dependency is None or relative in artifacts:
            raise HydrationError(f"Unexpected or duplicate snapshot dependency: {relative}")
        archive_name = _require_string(raw, "archive", context)
        if (
            Path(archive_name).name != archive_name
            or PureWindowsPath(archive_name).drive
            or archive_name in {".", ".."}
            or not archive_name.endswith(".tar")
        ):
            raise HydrationError(f"Unsafe snapshot archive name: {archive_name!r}")
        expected_sha = _require_string(raw, "sha256", context)
        if not HEX64.fullmatch(expected_sha):
            raise HydrationError(f"Invalid snapshot SHA-256 for {relative}")
        for key, expected in (
            ("source_url", dependency.source_url),
            ("commit", dependency.commit),
            ("tree", dependency.tree),
        ):
            actual = _require_string(raw, key, context)
            if actual != expected:
                raise HydrationError(f"Snapshot {key} mismatch for {relative}")
        archive_path = snapshot_dir / archive_name
        if archive_path.is_symlink() or not archive_path.is_file():
            raise HydrationError(f"Snapshot archive is missing: {archive_path}")
        actual_sha = _sha256(archive_path)
        if actual_sha != expected_sha:
            raise HydrationError(
                f"Snapshot SHA-256 mismatch for {relative}: expected {expected_sha}, got {actual_sha}"
            )
        _inspect_archive(archive_path, scratch_root / f"snapshot-check-{index}")
        artifacts[relative] = SnapshotArtifact(dependency, archive_path, actual_sha)

    missing = sorted(set(required) - set(artifacts))
    if missing:
        raise HydrationError(f"Snapshot manifest is missing dependencies: {', '.join(missing)}")
    return artifacts


def _extract_snapshot(artifact: SnapshotArtifact, destination: Path) -> None:
    if _sha256(artifact.path) != artifact.sha256:
        raise HydrationError(f"Snapshot changed after preflight: {artifact.path}")
    _inspect_archive(artifact.path, destination)
    destination.mkdir(parents=True)
    try:
        with tarfile.open(artifact.path, mode="r:*") as archive:
            archive.extractall(destination, filter="data")
    except (OSError, tarfile.TarError) as exc:
        raise HydrationError(f"Unable to extract snapshot {artifact.path}: {exc}") from exc


def _fetch_github(dependency: Dependency, destination: Path) -> None:
    destination.mkdir(parents=True)
    _git(destination, "init", "--quiet")
    _git(destination, "remote", "add", "origin", dependency.source_url)
    _git(
        destination,
        "-c",
        "protocol.version=2",
        "fetch",
        "--no-tags",
        "--depth=1",
        "origin",
        dependency.commit,
    )
    _git(destination, "-c", "advice.detachedHead=false", "checkout", "--detach", dependency.commit)


def _safe_target(dawn_root: Path, relative: str) -> Path:
    root = dawn_root.resolve()
    target = dawn_root.joinpath(*PurePosixPath(relative).parts)
    resolved_parent = target.parent.resolve()
    try:
        resolved_parent.relative_to(root)
    except ValueError as exc:
        raise HydrationError(f"Dependency target escapes Dawn root: {relative}") from exc
    return target


def _lexists(path: Path) -> bool:
    return os.path.lexists(path)


def _move(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.replace(source, destination)
    except OSError as exc:
        raise HydrationError(f"Unable to move {source} to {destination}: {exc}") from exc


def _rollback(
    applied: list[tuple[PreparedDependency, Path, Path | None]],
) -> list[str]:
    failures: list[str] = []
    for prepared, target, quarantined in reversed(applied):
        try:
            if _lexists(target) and prepared.path is not None:
                _move(target, prepared.path)
            if quarantined is not None and _lexists(quarantined):
                _move(quarantined, target)
        except HydrationError as exc:
            failures.append(str(exc))
    return failures


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _audit_payload(
    dependency_lock: DependencyLock,
    prepared_dependencies: list[PreparedDependency],
) -> dict[str, Any]:
    return {
        "schema": AUDIT_SCHEMA,
        "version": LOCK_VERSION,
        "dawn_revision": dependency_lock.dawn_revision,
        "dependency_lock": {
            "path": str(dependency_lock.path),
            "sha256": dependency_lock.sha256,
        },
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "dependencies": [
            {
                "path": prepared.dependency.path,
                "transport": prepared.dependency.transport,
                "hydration": prepared.hydration,
                "deps_url": prepared.dependency.deps_url,
                "source_url": prepared.dependency.source_url,
                "commit": prepared.dependency.commit,
                "tree": prepared.dependency.tree,
                "transport_sha256": prepared.transport_sha256,
                "quarantined_path": (
                    str(prepared.quarantined_path)
                    if prepared.quarantined_path is not None
                    else None
                ),
                "fsck": "strict",
            }
            for prepared in prepared_dependencies
        ],
    }


def verify_all(dawn_root: Path, dependency_lock: DependencyLock) -> None:
    for dependency in dependency_lock.dependencies:
        verify_repository(_safe_target(dawn_root, dependency.path), dependency)


def hydrate(
    dawn_root: Path,
    dependency_lock: DependencyLock,
    snapshot_dir: Path | None,
    audit_manifest: Path,
) -> None:
    if not dawn_root.is_dir():
        raise HydrationError(f"Dawn root is not a directory: {dawn_root}")
    try:
        audit_manifest.resolve().relative_to(dawn_root.resolve())
    except ValueError:
        pass
    else:
        raise HydrationError("The dependency audit manifest must be outside the Dawn root")
    targets = {
        dependency.path: _safe_target(dawn_root, dependency.path)
        for dependency in dependency_lock.dependencies
    }
    needs_snapshot = any(
        dependency.transport == "official_gitiles_snapshot"
        and not _repository_matches(targets[dependency.path], dependency)
        for dependency in dependency_lock.dependencies
    )

    with tempfile.TemporaryDirectory(
        prefix=".dawn-dependency-hydration-", dir=dawn_root.parent
    ) as temporary_name:
        scratch_root = Path(temporary_name)
        snapshot_artifacts: dict[str, SnapshotArtifact] = {}
        if needs_snapshot:
            if snapshot_dir is None:
                raise HydrationError(
                    "Official Gitiles snapshots are required; set "
                    "INFERNUX_DAWN_OFFICIAL_SNAPSHOT_DIR or pass --snapshot-dir"
                )
            snapshot_artifacts = preflight_snapshots(
                snapshot_dir.resolve(), dependency_lock, scratch_root
            )

        prepared_dependencies: list[PreparedDependency] = []
        for index, dependency in enumerate(dependency_lock.dependencies):
            target = targets[dependency.path]
            if _repository_matches(target, dependency):
                prepared_dependencies.append(
                    PreparedDependency(dependency, None, "cached")
                )
                continue
            prepared_path = scratch_root / f"dependency-{index:02d}"
            if dependency.transport == "github":
                _fetch_github(dependency, prepared_path)
                hydration = "github"
                transport_sha256 = None
            else:
                artifact = snapshot_artifacts[dependency.path]
                _extract_snapshot(artifact, prepared_path)
                hydration = "official_gitiles_snapshot"
                transport_sha256 = artifact.sha256
            verify_repository(prepared_path, dependency)
            prepared_dependencies.append(
                PreparedDependency(
                    dependency,
                    prepared_path,
                    hydration,
                    transport_sha256,
                )
            )

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        quarantine_root = (
            dawn_root.parent
            / ".dawn-dependency-quarantine"
            / f"{dawn_root.name}-{timestamp}-{os.getpid()}"
        )
        applied: list[tuple[PreparedDependency, Path, Path | None]] = []
        try:
            for prepared in prepared_dependencies:
                if prepared.path is None:
                    continue
                target = targets[prepared.dependency.path]
                quarantined: Path | None = None
                if _lexists(target):
                    quarantined = quarantine_root.joinpath(
                        *PurePosixPath(prepared.dependency.path).parts
                    )
                    _move(target, quarantined)
                    prepared.quarantined_path = quarantined
                applied.append((prepared, target, quarantined))
                _move(prepared.path, target)

            verify_all(dawn_root, dependency_lock)
            _atomic_write_json(
                audit_manifest,
                _audit_payload(dependency_lock, prepared_dependencies),
            )
        except Exception as exc:
            rollback_failures = _rollback(applied)
            if rollback_failures:
                raise HydrationError(
                    f"Hydration failed and rollback was incomplete: {exc}; "
                    + "; ".join(rollback_failures)
                ) from exc
            raise


def _arguments(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dawn-root", required=True, type=Path)
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--snapshot-dir", type=Path)
    parser.add_argument("--audit-manifest", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    arguments = parser.parse_args(argv)
    if not arguments.verify_only and arguments.audit_manifest is None:
        parser.error("--audit-manifest is required unless --verify-only is used")
    return arguments


def main(argv: list[str] | None = None) -> int:
    arguments = _arguments(sys.argv[1:] if argv is None else argv)
    try:
        dependency_lock = load_lock(arguments.lock.resolve())
        dawn_root = arguments.dawn_root.resolve()
        if arguments.verify_only:
            verify_all(dawn_root, dependency_lock)
            if arguments.audit_manifest is not None:
                audit_manifest = arguments.audit_manifest.resolve()
                try:
                    audit_manifest.relative_to(dawn_root)
                except ValueError:
                    pass
                else:
                    raise HydrationError(
                        "The dependency audit manifest must be outside the Dawn root"
                    )
                _atomic_write_json(
                    audit_manifest,
                    _audit_payload(
                        dependency_lock,
                        [
                            PreparedDependency(dependency, None, "cached")
                            for dependency in dependency_lock.dependencies
                        ],
                    ),
                )
            print(f"Dawn dependency verification passed: {dawn_root}")
        else:
            hydrate(
                dawn_root,
                dependency_lock,
                arguments.snapshot_dir,
                arguments.audit_manifest.resolve(),
            )
            print(f"Dawn dependency hydration passed: {dawn_root}")
    except HydrationError as exc:
        print(f"Dawn dependency hydration failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
