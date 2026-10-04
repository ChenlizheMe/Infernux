"""Exercise deletion boundaries in disposable repositories, never the checkout."""

from pathlib import Path
import os
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[2]
POWERSHELL = shutil.which("pwsh")
pytestmark = pytest.mark.skipif(POWERSHELL is None, reason="PowerShell is required")


def _git(root, *args):
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)


def _write(root, name, text="generated"):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def repository(tmp_path):
    root = tmp_path / "engine"
    root.mkdir()
    _git(root, "init", "-q")
    _write(root, ".gitmodules", "")
    _write(root, ".gitignore", "dist/\n*.dll\n__pycache__/\n")
    _write(root, "scripts/maintenance/clean_workspace.ps1", (
        ROOT / "scripts/maintenance/clean_workspace.ps1"
    ).read_text(encoding="utf-8"))
    _git(root, "add", ".gitmodules", ".gitignore", "scripts")
    return root


def _clean(root, *args):
    return subprocess.run(
        [POWERSHELL, "-NoProfile", "-File",
         str(root / "scripts/maintenance/clean_workspace.ps1"), *args],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=60,
        cwd=root.parent,
    )


@pytest.mark.parametrize("preview", [True, False], ids=["preview", "delete"])
def test_cleanup_handles_root_and_nested_submodule_outputs(repository, preview):
    release = _write(repository, "dist/releases/current/installer.exe")
    binary = _write(repository, "python/Infernux/lib/generated.dll")
    local_tool = _write(repository, "update_distributions.py")
    tracked_binary = _write(repository, "fixture.dll", "authored fixture")
    local_notes = _write(repository, "dev/collaboration/notes.md", "private research")
    local_binary = _write(repository, "dev/collaboration/example.dll", "private example")
    _git(repository, "add", "-f", "fixture.dll")
    submodule = repository / "external/plugin"
    submodule.mkdir(parents=True)
    _git(submodule, "init", "-q")
    _write(repository, ".gitmodules", '[submodule "plugin"]\npath = external/plugin\n')
    _write(submodule, ".gitignore", "package/editor/infernux_web/player/\n")
    source = _write(submodule, "exporter.py")
    _git(submodule, "add", ".gitignore", "exporter.py")
    payload = _write(submodule, "package/editor/infernux_web/player/player.exe")
    nested = submodule / "external/compiler"
    nested.mkdir(parents=True)
    _git(nested, "init", "-q")
    _write(submodule, ".gitmodules", '[submodule "compiler"]\npath = external/compiler\n')
    _write(nested, ".gitignore", "__pycache__/\n")
    _git(nested, "add", ".gitignore")
    nested_cache = _write(nested, "tests/__pycache__/cached.pyc")
    args = ("-WhatIf",) if preview else ()
    result = _clean(repository, *args)
    assert result.returncode == 0, result.stderr
    for generated in (release, binary, payload, nested_cache):
        assert generated.exists() == preview, generated
    assert source.exists() and local_tool.exists() and tracked_binary.exists()
    assert local_notes.read_text() == "private research"
    assert local_binary.read_text() == "private example"
    # A second invocation must be safe and leave authored data intact.
    repeated = _clean(repository, *args)
    assert repeated.returncode == 0, repeated.stderr
    assert source.read_text() == "generated"
    assert tracked_binary.read_text() == "authored fixture"


@pytest.mark.parametrize("preview", [True, False], ids=["preview", "delete"])
def test_cleanup_rejects_tracked_files_inside_output_directory(repository, preview):
    source = _write(repository, "dist/source.txt", "must survive")
    _git(repository, "add", "-f", "dist/source.txt")
    result = _clean(repository, *(["-WhatIf"] if preview else []))
    assert result.returncode != 0
    assert "Refusing to remove tracked source files" in result.stderr
    assert source.read_text() == "must survive"


def test_cleanup_rejects_submodule_path_outside_workspace(repository):
    outside = _write(repository.parent, "outside/keep.txt", "must survive")
    _write(repository, ".gitmodules", '[submodule "outside"]\npath = ../outside\n')
    result = _clean(repository)
    assert result.returncode != 0
    assert "Submodule outside workspace" in result.stderr
    assert outside.read_text() == "must survive"


@pytest.mark.parametrize("name", ["dist", "dist/linked"])
def test_cleanup_rejects_linked_output_directories(repository, name):
    outside = _write(repository.parent, "outside/keep.txt", "must survive")
    link = repository / name
    link.parent.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        # Junctions need no elevated symlink privilege on Windows.
        subprocess.run(
            [POWERSHELL, "-NoProfile", "-Command",
             "& { param($link, $target) New-Item -ItemType Junction -Path $link -Target $target | Out-Null }",
             str(link), str(outside.parent)], check=True, capture_output=True,
        )
    else:
        link.symlink_to(outside.parent, target_is_directory=True)
    try:
        result = _clean(repository)
        assert result.returncode != 0
        assert "reparse point" in result.stderr
        assert outside.read_text() == "must survive"
        assert link.exists()
    finally:
        # Remove the link itself so pytest's temporary tree cleanup cannot cross it.
        if link.exists():
            if os.name == "nt":
                link.rmdir()
            else:
                link.unlink()
