"""Exercise real cleanup entrypoints only inside an owned temporary tree."""
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from infernux.engine.game_builder import GameBuilder
from infernux.engine import nuitka_builder as nuitka_module
from infernux.engine.nuitka_builder import NuitkaBuilder


KINDS = ("project", "embedded", "bundle", "player", "staging", "intermediates", "package", "package-libs")
NAMES = ("Puzzle&Co", "Puzzle%INX_CLEANUP_SENTINEL%", "Puzzle (team)^!", "中文 合作项目")


@pytest.fixture
def cleanup_entrypoints(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "packaging"))
    modules = {
        "project": importlib.import_module("model.project_model"),
        "embedded": importlib.import_module("embed_runtime_manager"),
        "bundle": importlib.import_module("stage_bundled_python_runtime"),
    }
    # Never execute the old shell-based deletion, even in the baseline run.
    original_run = subprocess.run
    def no_shell_cleanup(args, *positional, **kwargs):
        command = args if isinstance(args, str) else " ".join(map(str, args))
        assert not ("cmd" in command.lower() and any(word in command.lower().split() for word in ("rd", "rmdir"))), command
        return original_run(args, *positional, **kwargs)
    monkeypatch.setattr(subprocess, "run", no_shell_cleanup)
    return modules


def _run_cleanup(kind, target, root, modules, monkeypatch):
    assert target.resolve().is_relative_to(root.resolve()) and target.resolve() != root.resolve()
    if kind in modules:
        modules[kind]._remove_tree(str(target))
    elif kind == "player":
        GameBuilder._remove_directory_tree(str(target))
    elif kind == "staging":
        entry = root / "source.py"
        entry.write_text("pass\n", encoding="utf-8")
        builder = object.__new__(NuitkaBuilder)
        builder._staging_dir = str(target)
        builder.entry_script = str(entry)
        builder._prepare_staging()
        assert list(target.iterdir()) == [target / "boot.py"]
    elif kind == "intermediates":
        builder = object.__new__(NuitkaBuilder)
        builder._staging_dir = str(target.parent)
        # Collect and execute any legacy background work in this thread so
        # assertion failures cannot vanish in a daemon during the baseline.
        deferred = []
        class CapturedThread:
            def __init__(self, *, target, args, daemon):
                deferred.append((target, args))
            def start(self):
                pass
        with monkeypatch.context() as patch:
            patch.setattr(nuitka_module.threading, "Thread", CapturedThread)
            builder._cleanup_build_artifacts()
            for callback, args in deferred:
                callback(*args)
    else:
        site = root / "source-site"
        (site / "fixture").mkdir(parents=True)
        (site / "fixture" / "__init__.py").write_text("value = 42\n", encoding="utf-8")
        (site / "fixture.libs").mkdir()
        (site / "fixture.libs" / "native.dll").write_bytes(b"fixture")
        dist = target.parent
        builder = object.__new__(NuitkaBuilder)
        builder._builder_python = sys.executable
        builder.raw_copy_packages = ["fixture"]
        with monkeypatch.context() as patch:
            patch.setattr(nuitka_module, "_run_python", lambda *a, **k: SimpleNamespace(stdout=json.dumps([str(site)])))
            builder._inject_jit_packages(str(dist))
        assert (dist / "fixture" / "__init__.pyc").is_file()
        assert (dist / "fixture.libs" / "native.dll").read_bytes() == b"fixture"


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("name", NAMES)
def test_cleanup_paths_are_literal_and_preserve_siblings(tmp_path, monkeypatch, cleanup_entrypoints, kind, name):
    monkeypatch.setenv("INX_CLEANUP_SENTINEL", "must-not-expand")
    owner = tmp_path / name
    owner.mkdir()
    leaf = {"intermediates": "boot.build", "package": "fixture", "package-libs": "fixture.libs"}.get(kind, "target")
    target = owner / leaf
    (target / "nested").mkdir(parents=True)
    (target / "nested" / "obsolete.bin").write_bytes(b"old")
    sentinel = owner / "sibling.keep"
    sentinel.write_bytes(b"teammate")
    expansion = tmp_path / name.replace("%INX_CLEANUP_SENTINEL%", "must-not-expand").split("&", 1)[0]
    if expansion != owner:
        expansion.mkdir(exist_ok=True)
        (expansion / "untouched").write_bytes(b"unrelated")
    _run_cleanup(kind, target, tmp_path, cleanup_entrypoints, monkeypatch)
    assert not (target / "nested").exists()
    if kind not in ("staging", "package", "package-libs"):
        assert not target.exists()
    assert sentinel.read_bytes() == b"teammate"
    if expansion != owner:
        assert (expansion / "untouched").read_bytes() == b"unrelated"


@pytest.mark.parametrize("kind", KINDS)
def test_cleanup_failure_stops_the_owning_operation(tmp_path, monkeypatch, cleanup_entrypoints, kind):
    import shutil

    leaf = {"intermediates": "boot.build", "package": "fixture", "package-libs": "fixture.libs"}.get(kind, "target")
    target = tmp_path / "owner" / leaf
    target.mkdir(parents=True)
    original = target / "old.data"
    original.write_bytes(b"retained")
    removed = []
    def denied(path, *args, **kwargs):
        assert Path(path).resolve() == target.resolve()
        removed.append(str(path))
        raise PermissionError("directory in use")
    monkeypatch.setattr(shutil, "rmtree", denied)
    with pytest.raises(PermissionError, match="directory in use"):
        _run_cleanup(kind, target, tmp_path, cleanup_entrypoints, monkeypatch)
    assert len(removed) == 1
    assert original.read_bytes() == b"retained"


@pytest.fixture(params=["hub", "engine"])
def remove_tree(request, cleanup_entrypoints):
    if request.param == "hub":
        return cleanup_entrypoints["project"]._remove_tree
    from infernux.engine.filesystem import remove_directory_tree
    return remove_directory_tree


def test_cleanup_missing_target_is_already_complete(tmp_path, remove_tree):
    remove_tree(tmp_path / "missing")
    assert not (tmp_path / "missing").exists()


@pytest.mark.parametrize("invalid", ["empty", "root"])
def test_cleanup_requires_a_non_root_target(tmp_path, monkeypatch, remove_tree, invalid):
    import shutil

    def never_delete(*args, **kwargs):
        pytest.fail("invalid target reached recursive deletion")
    monkeypatch.setattr(shutil, "rmtree", never_delete)
    with pytest.raises(ValueError):
        remove_tree("" if invalid == "empty" else Path(tmp_path.anchor))


@pytest.mark.skipif(sys.platform != "win32", reason="Windows junction semantics")
@pytest.mark.parametrize("linked_root", [True, False])
def test_cleanup_does_not_follow_windows_junctions(tmp_path, remove_tree, linked_root):
    import _winapi

    owner = tmp_path / "owner"
    owner.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "untouched").write_bytes(b"other owner")
    junction = owner / "linked"
    _winapi.CreateJunction(str(outside), str(junction))
    try:
        if linked_root:
            with pytest.raises(ValueError, match="linked root"):
                remove_tree(junction)
            assert os.path.isjunction(junction)
        else:
            assert owner.resolve().is_relative_to(tmp_path.resolve())
            remove_tree(owner)
            assert not owner.exists()
        assert (outside / "untouched").read_bytes() == b"other owner"
    finally:
        if os.path.isjunction(junction):
            os.rmdir(junction)
