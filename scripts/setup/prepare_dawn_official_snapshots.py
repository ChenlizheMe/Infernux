#!/usr/bin/env python3
"""Prepare the pinned Dawn Chromium dependencies from official Gitiles repos.

The Dawn lock deliberately does not permit community mirrors for Chromium-vendored
dependencies.  This helper creates the Git-worktree tarballs expected by the
hydrator, and keeps an existing verified cache intact when one is available.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path


SCRIPT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_ROOT))
import hydrate_dawn_dependencies as hydrator  # noqa: E402


def _run_git(path: Path, *arguments: str) -> None:
    environment = os.environ.copy()
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
    subprocess.run(
        ["git", "-C", str(path), *arguments],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        env=environment,
    )


def _archive_name(dependency: hydrator.Dependency) -> str:
    return dependency.path.removeprefix("third_party/").replace("/", "_") + ".tar"


def _clone_worktree(dependency: hydrator.Dependency, destination: Path) -> None:
    destination.mkdir(parents=True)
    _run_git(destination, "init", "--quiet")
    _run_git(destination, "remote", "add", "origin", dependency.source_url)
    _run_git(
        destination,
        "-c",
        "protocol.version=2",
        "fetch",
        "--no-tags",
        "--depth=1",
        "origin",
        dependency.commit,
    )
    _run_git(
        destination,
        "-c",
        "advice.detachedHead=false",
        "checkout",
        "--detach",
        dependency.commit,
    )
    hydrator.verify_repository(destination, dependency)


def _write_archive(worktree: Path, destination: Path) -> None:
    with tarfile.open(destination, mode="w") as archive:
        archive.add(worktree, arcname=".", recursive=True)


def _write_manifest(output: Path, lock: hydrator.DependencyLock, artifacts: list[dict[str, str]]) -> None:
    manifest = {
        "schema": hydrator.SNAPSHOT_SCHEMA,
        "version": hydrator.LOCK_VERSION,
        "dawn_revision": lock.dawn_revision,
        "dependencies": artifacts,
    }
    temporary = output / ".manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output / "manifest.json")


def prepare(lock_path: Path, output: Path) -> None:
    lock = hydrator.load_lock(lock_path.resolve())
    required = [
        dependency
        for dependency in lock.dependencies
        if dependency.transport == "official_gitiles_snapshot"
    ]
    if not required:
        raise RuntimeError("The Dawn lock contains no official Gitiles snapshots")
    output.mkdir(parents=True, exist_ok=True)

    # A cache is usable only when the same lock, source identities, hashes, and
    # safe archive structure all pass the production hydrator's checks.
    manifest = output / "manifest.json"
    if manifest.is_file():
        try:
            with tempfile.TemporaryDirectory(prefix="dawn-snapshot-check-") as scratch:
                hydrator.preflight_snapshots(output, lock, Path(scratch))
            return
        except hydrator.HydrationError:
            manifest.unlink(missing_ok=True)

    with tempfile.TemporaryDirectory(prefix="dawn-official-snapshots-") as scratch_name:
        scratch = Path(scratch_name)
        artifacts: list[dict[str, str]] = []
        for dependency in required:
            archive_name = _archive_name(dependency)
            archive_path = output / archive_name
            with tempfile.TemporaryDirectory(dir=scratch) as worktree_name:
                worktree = Path(worktree_name) / "worktree"
                _clone_worktree(dependency, worktree)
                temporary_file = tempfile.NamedTemporaryFile(
                    dir=output,
                    prefix=f".{archive_name}.",
                    suffix=".tmp",
                    delete=False,
                )
                temporary_archive = Path(temporary_file.name)
                temporary_file.close()
                try:
                    _write_archive(worktree, temporary_archive)
                    os.replace(temporary_archive, archive_path)
                finally:
                    temporary_archive.unlink(missing_ok=True)
            artifacts.append(
                {
                    "path": dependency.path,
                    "archive": archive_name,
                    "sha256": hydrator._sha256(archive_path),
                    "source_url": dependency.source_url,
                    "commit": dependency.commit,
                    "tree": dependency.tree,
                }
            )
        _write_manifest(output, lock, artifacts)
        with tempfile.TemporaryDirectory(prefix="dawn-snapshot-final-check-") as scratch_check:
            hydrator.preflight_snapshots(output, lock, Path(scratch_check))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    try:
        prepare(arguments.lock, arguments.output.resolve())
    except (OSError, RuntimeError, subprocess.CalledProcessError, hydrator.HydrationError) as exc:
        print(f"Dawn official snapshot preparation failed: {exc}", file=sys.stderr)
        return 1
    print(f"Dawn official snapshots ready: {arguments.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
