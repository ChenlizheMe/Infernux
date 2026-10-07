from __future__ import annotations

import hashlib
import json
import stat
import subprocess
import tarfile
from dataclasses import replace
from pathlib import Path

import pytest

from scripts.setup import hydrate_dawn_dependencies as hydrate
from scripts.setup import prepare_dawn_official_snapshots as prepare_snapshots


DAWN_REVISION = "1" * 40


def _git(path: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _repository(path: Path, name: str, origin: str | None = None) -> tuple[str, str]:
    path.mkdir(parents=True)
    _git(path, "init", "--quiet")
    _git(path, "config", "user.name", "Infernux Test")
    _git(path, "config", "user.email", "test@infernux.invalid")
    _git(path, "config", "core.autocrlf", "false")
    _git(path, "config", "core.filemode", "false")
    (path / "payload.txt").write_text(f"{name}\n", encoding="utf-8")
    _git(path, "add", "payload.txt")
    _git(path, "commit", "--quiet", "-m", f"fixture {name}")
    if origin is not None:
        _git(path, "remote", "add", "origin", origin)
    return _git(path, "rev-parse", "HEAD"), _git(path, "rev-parse", "HEAD^{tree}")


def _dependency(
    relative: str,
    transport: str,
    source_url: str,
    commit: str,
    tree: str,
) -> dict[str, str]:
    return {
        "path": relative,
        "transport": transport,
        "deps_url": source_url,
        "source_url": source_url,
        "commit": commit,
        "tree": tree,
    }


def _write_lock(path: Path, dependencies: list[dict[str, str]]) -> Path:
    payload = {
        "schema": hydrate.LOCK_SCHEMA,
        "version": hydrate.LOCK_VERSION,
        "dawn_revision": DAWN_REVISION,
        "dependency_count": len(dependencies),
        "snapshot_dependency_count": sum(
            dependency["transport"] == "official_gitiles_snapshot"
            for dependency in dependencies
        ),
        "dependencies": dependencies,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _snapshot_archive(repository: Path, archive: Path) -> str:
    with tarfile.open(archive, "w") as stream:
        stream.add(repository, arcname=".")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    return digest


def _write_snapshot_manifest(
    snapshot_dir: Path,
    entries: list[tuple[dict[str, str], Path, str]],
) -> None:
    payload = {
        "schema": hydrate.SNAPSHOT_SCHEMA,
        "version": hydrate.LOCK_VERSION,
        "dawn_revision": DAWN_REVISION,
        "dependencies": [
            {
                "path": dependency["path"],
                "archive": archive.name,
                "sha256": sha256,
                "source_url": dependency["source_url"],
                "commit": dependency["commit"],
                "tree": dependency["tree"],
            }
            for dependency, archive, sha256 in entries
        ],
    }
    (snapshot_dir / "manifest.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def test_prepare_official_snapshots_builds_and_reuses_verified_worktree_cache(
    tmp_path: Path,
):
    source = tmp_path / "official-source"
    commit, tree = _repository(source, "official")
    dependency = _dependency(
        "third_party/official",
        "official_gitiles_snapshot",
        str(source.resolve()),
        commit,
        tree,
    )
    lock_path = _write_lock(tmp_path / "lock.json", [dependency])
    output = tmp_path / "snapshots"

    prepare_snapshots.prepare(lock_path, output)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["dependencies"][0]["path"] == dependency["path"]
    assert (output / manifest["dependencies"][0]["archive"]).is_file()
    # A second invocation must use the verified cache instead of replacing it.
    archive = output / manifest["dependencies"][0]["archive"]
    before = archive.stat().st_mtime_ns
    prepare_snapshots.prepare(lock_path, output)
    assert archive.stat().st_mtime_ns == before


def test_hydrate_places_only_fully_verified_dependencies_and_quarantines_bad_cache(
    tmp_path: Path,
):
    direct_source = tmp_path / "direct-source"
    direct_commit, direct_tree = _repository(direct_source, "direct")
    direct = _dependency(
        "third_party/direct/src",
        "github",
        str(direct_source.resolve()),
        direct_commit,
        direct_tree,
    )

    snapshot_url = "https://chromium.googlesource.com/chromium/test/snapshot.git"
    snapshot_source = tmp_path / "snapshot-source"
    snapshot_commit, snapshot_tree = _repository(
        snapshot_source, "snapshot", snapshot_url
    )
    snapshot = _dependency(
        "third_party/snapshot",
        "official_gitiles_snapshot",
        snapshot_url,
        snapshot_commit,
        snapshot_tree,
    )

    lock_path = _write_lock(tmp_path / "lock.json", [direct, snapshot])
    dependency_lock = hydrate.load_lock(lock_path)
    snapshot_dir = tmp_path / "snapshots"
    snapshot_dir.mkdir()
    archive = snapshot_dir / "snapshot.tar"
    archive_sha = _snapshot_archive(snapshot_source, archive)
    _write_snapshot_manifest(snapshot_dir, [(snapshot, archive, archive_sha)])

    dawn_root = tmp_path / "dawn"
    bad_cache = dawn_root / "third_party/direct/src"
    bad_cache.mkdir(parents=True)
    (bad_cache / "keep.txt").write_text("quarantine me", encoding="utf-8")
    audit_path = tmp_path / "audit.json"

    hydrate.hydrate(dawn_root, dependency_lock, snapshot_dir, audit_path)

    hydrate.verify_all(dawn_root, dependency_lock)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["schema"] == hydrate.AUDIT_SCHEMA
    by_path = {item["path"]: item for item in audit["dependencies"]}
    assert by_path[direct["path"]]["hydration"] == "github"
    assert by_path[snapshot["path"]]["transport_sha256"] == archive_sha
    quarantined = Path(by_path[direct["path"]]["quarantined_path"])
    assert (quarantined / "keep.txt").read_text(encoding="utf-8") == "quarantine me"


def test_snapshot_preflight_rejects_traversal_before_any_dependency_is_placed(
    tmp_path: Path,
):
    source = tmp_path / "direct-source"
    commit, tree = _repository(source, "direct")
    direct = _dependency(
        "third_party/direct/src", "github", str(source.resolve()), commit, tree
    )
    snapshot_dependencies: list[dict[str, str]] = []
    manifest_entries: list[tuple[dict[str, str], Path, str]] = []
    snapshot_dir = tmp_path / "snapshots"
    snapshot_dir.mkdir()
    for index in range(2):
        url = f"https://chromium.googlesource.com/chromium/test/snapshot-{index}.git"
        repository = tmp_path / f"snapshot-source-{index}"
        snapshot_commit, snapshot_tree = _repository(repository, f"snapshot-{index}", url)
        dependency = _dependency(
            f"third_party/snapshot-{index}",
            "official_gitiles_snapshot",
            url,
            snapshot_commit,
            snapshot_tree,
        )
        archive = snapshot_dir / f"snapshot-{index}.tar"
        if index == 0:
            archive_sha = _snapshot_archive(repository, archive)
        else:
            with tarfile.open(archive, "w") as stream:
                payload = tmp_path / "escape-payload"
                payload.write_text("escape", encoding="utf-8")
                stream.add(payload, arcname="../escape")
            archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
        snapshot_dependencies.append(dependency)
        manifest_entries.append((dependency, archive, archive_sha))
    _write_snapshot_manifest(snapshot_dir, manifest_entries)

    dependency_lock = hydrate.load_lock(
        _write_lock(tmp_path / "lock.json", [direct, *snapshot_dependencies])
    )
    dawn_root = tmp_path / "dawn"
    marker = dawn_root / "third_party/direct/src/marker.txt"
    marker.parent.mkdir(parents=True)
    marker.write_text("unchanged", encoding="utf-8")

    with pytest.raises(hydrate.HydrationError, match="unsafe path"):
        hydrate.hydrate(dawn_root, dependency_lock, snapshot_dir, tmp_path / "audit.json")

    assert marker.read_text(encoding="utf-8") == "unchanged"
    assert not (tmp_path / "escape").exists()
    assert not (tmp_path / ".dawn-dependency-quarantine").exists()
    assert not (tmp_path / "audit.json").exists()


def test_prepare_failure_leaves_all_existing_targets_in_place(tmp_path: Path):
    valid_source = tmp_path / "valid-source"
    valid_commit, valid_tree = _repository(valid_source, "valid")
    valid = _dependency(
        "third_party/valid",
        "github",
        str(valid_source.resolve()),
        valid_commit,
        valid_tree,
    )
    missing = _dependency(
        "third_party/missing",
        "github",
        str((tmp_path / "does-not-exist").resolve()),
        "2" * 40,
        "3" * 40,
    )
    dependency_lock = hydrate.load_lock(
        _write_lock(tmp_path / "lock.json", [valid, missing])
    )
    dawn_root = tmp_path / "dawn"
    markers: list[Path] = []
    for relative in (valid["path"], missing["path"]):
        marker = dawn_root / relative / "marker.txt"
        marker.parent.mkdir(parents=True)
        marker.write_text(relative, encoding="utf-8")
        markers.append(marker)

    with pytest.raises(hydrate.HydrationError, match="Git command failed"):
        hydrate.hydrate(dawn_root, dependency_lock, None, tmp_path / "audit.json")

    assert [marker.read_text(encoding="utf-8") for marker in markers] == [
        valid["path"],
        missing["path"],
    ]
    assert not (tmp_path / ".dawn-dependency-quarantine").exists()
    assert not (tmp_path / "audit.json").exists()


def test_placement_failure_rolls_back_every_previously_placed_dependency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    dependencies: list[dict[str, str]] = []
    for name in ("first", "second"):
        source = tmp_path / f"{name}-source"
        commit, tree = _repository(source, name)
        dependencies.append(
            _dependency(
                f"third_party/{name}",
                "github",
                str(source.resolve()),
                commit,
                tree,
            )
        )
    dependency_lock = hydrate.load_lock(
        _write_lock(tmp_path / "lock.json", dependencies)
    )
    dawn_root = tmp_path / "dawn"
    dawn_root.mkdir()
    real_move = hydrate._move

    def fail_second_placement(source: Path, destination: Path) -> None:
        if destination == dawn_root / "third_party/second":
            raise hydrate.HydrationError("injected placement failure")
        real_move(source, destination)

    monkeypatch.setattr(hydrate, "_move", fail_second_placement)

    with pytest.raises(hydrate.HydrationError, match="injected placement failure"):
        hydrate.hydrate(dawn_root, dependency_lock, None, tmp_path / "audit.json")

    assert not (dawn_root / "third_party/first").exists()
    assert not (dawn_root / "third_party/second").exists()
    assert not (tmp_path / "audit.json").exists()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("source_url", "https://invalid.example/repository.git", "Origin mismatch"),
        ("commit", "0" * 40, "Commit mismatch"),
        ("tree", "0" * 40, "Tree mismatch"),
    ),
)
def test_repository_verification_rejects_every_identity_mismatch(
    tmp_path: Path,
    field: str,
    value: str,
    message: str,
):
    origin = "https://github.com/example/fixture.git"
    repository = tmp_path / "repository"
    commit, tree = _repository(repository, "identity", origin)
    dependency = hydrate.Dependency(
        "third_party/fixture", "github", origin, origin, commit, tree
    )

    with pytest.raises(hydrate.HydrationError, match=message):
        hydrate.verify_repository(repository, replace(dependency, **{field: value}))


def test_repository_verification_runs_strict_object_check(tmp_path: Path):
    origin = "https://github.com/example/fixture.git"
    repository = tmp_path / "repository"
    commit, tree = _repository(repository, "fsck", origin)
    dependency = hydrate.Dependency(
        "third_party/fixture", "github", origin, origin, commit, tree
    )
    object_path = repository / ".git/objects/aa" / ("b" * 38)
    object_path.parent.mkdir()
    object_path.write_bytes(b"not a valid loose Git object")
    object_path.chmod(stat.S_IREAD)

    with pytest.raises(hydrate.HydrationError, match="Git command failed"):
        hydrate.verify_repository(repository, dependency)
