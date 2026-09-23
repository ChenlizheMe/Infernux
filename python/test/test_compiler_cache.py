"""Compiler artifact ownership, bounded storage and live-code lifetime."""

import os

import pytest

from Infernux import jit
from Infernux.application import Application
from Infernux._compiler.cache import compiler_cache_root, prune_cache_files
from Infernux import _jit_cache
from Infernux.engine.project_context import using_project_root


def test_player_gpu_artifacts_are_shipped_while_cpu_cache_remains_writable(tmp_path, monkeypatch):
    project = tmp_path / "project"
    player = tmp_path / "player"
    monkeypatch.setattr(Application, "is_player", staticmethod(lambda: False))
    monkeypatch.setenv("NUMBA_CACHE_DIR", str(tmp_path / "unrelated-global-cache"))
    with using_project_root(str(project)):
        assert compiler_cache_root() == project / "Library/Artifacts/Compute"
        assert _jit_cache.cpu_cache_root() == project / "Library/Artifacts/Compute/CPU"
        monkeypatch.setattr(Application, "is_player", staticmethod(lambda: True))
        monkeypatch.setattr(Application, "persistent_data_path", staticmethod(lambda: str(player)))
        assert compiler_cache_root() == project / "Library/Artifacts/Compute"
        assert _jit_cache.cpu_cache_root() == player / "Cache/Compute/CPU"


def test_standalone_disk_cache_requires_explicit_storage(tmp_path, monkeypatch):
    monkeypatch.setattr(Application, "is_player", staticmethod(lambda: False))
    monkeypatch.delenv("NUMBA_CACHE_DIR", raising=False)
    with using_project_root(None):
        with pytest.raises(RuntimeError, match="active project or explicit"):
            _jit_cache.cpu_cache_root()
        monkeypatch.setenv("NUMBA_CACHE_DIR", str(tmp_path / "tool-cache"))
        assert _jit_cache.cpu_cache_root() == tmp_path / "tool-cache"
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("NUMBA_CACHE_DIR", "intermediate/../tool-cache")
        assert _jit_cache.cpu_cache_root() == tmp_path / "tool-cache"


def test_project_switch_does_not_reuse_previous_project_disk_owner(tmp_path):
    def kernel(value):
        return value + 2

    with using_project_root(str(tmp_path / "first")):
        first = jit.compile(kernel, auto_parallel=False, cache=True)
        assert first(3) == 5
        first_root = _jit_cache.cpu_cache_root()
        original_files = set(first_root.iterdir())
    with using_project_root(str(tmp_path / "second")):
        second = jit.compile(kernel, auto_parallel=False, cache=True)
        assert first._compiled is not second._compiled
        assert second(3.0) == 5.0
        second_root = _jit_cache.cpu_cache_root()
        assert list(second_root.glob("inx-*.nbc"))
    assert set(first_root.iterdir()) == original_files
    assert first(4) == 6


def test_cpu_artifact_eviction_preserves_loaded_code(tmp_path, monkeypatch):
    monkeypatch.setattr(_jit_cache, "_CACHE_FILE_LIMIT", 2)
    with using_project_root(str(tmp_path)):
        functions = []
        for offset in range(3):
            def make_function(amount):
                def kernel(value):
                    return value + amount
                return jit.compile(kernel, auto_parallel=False, cache=True)

            functions.append(make_function(offset))
            assert functions[-1](7) == 7 + offset
        assert len(list(_jit_cache.cpu_cache_root().glob("inx-*.nb[ci]"))) <= 2
        assert [function(7) for function in functions] == [7, 8, 9]


def test_cpu_byte_budget_does_not_break_current_specialization(tmp_path, monkeypatch):
    monkeypatch.setattr(_jit_cache, "_CACHE_BYTE_LIMIT", 1)
    with using_project_root(str(tmp_path)):
        @jit.compile(auto_parallel=False, cache=True)
        def kernel(value):
            return value * 2

        assert kernel(4) == 8
        assert sum(path.stat().st_size for path in _jit_cache.cpu_cache_root().glob("inx-*.nb[ci]")) <= 1
        assert kernel(5) == 10


def test_warm_calls_do_not_scan_disk_or_prune_cache(tmp_path, monkeypatch):
    with using_project_root(str(tmp_path)):
        @jit.compile(auto_parallel=False, cache=True)
        def kernel(value):
            return value + 2

        assert kernel(3) == 5

        def unexpected_prune(*args, **kwargs):
            pytest.fail("warm calls must not perform disk-cache housekeeping")

        monkeypatch.setattr(_jit_cache, "prune_cache_files", unexpected_prune)
        assert kernel(8) == 10


def test_pruning_respects_bytes_and_keeps_unrelated_files_and_directories(tmp_path):
    old = tmp_path / "old.inxgpu"
    new = tmp_path / "new.inxgpu"
    old.write_bytes(b"1234")
    new.write_bytes(b"5678")
    os.utime(old, (1, 1))
    os.utime(new, (2, 2))
    unrelated = tmp_path / "authored.txt"
    unrelated.write_text("keep", encoding="utf-8")
    nested = tmp_path / "directory.inxgpu"
    nested.mkdir()
    (nested / "authored.txt").write_text("keep", encoding="utf-8")
    prune_cache_files(tmp_path, "*.inxgpu", file_limit=128, byte_limit=4)
    assert not old.exists()
    assert new.read_bytes() == b"5678"
    assert unrelated.read_text(encoding="utf-8") == "keep"
    assert (nested / "authored.txt").exists()


def test_unwritable_owned_location_is_not_redirected(tmp_path):
    (tmp_path / "Library").write_bytes(b"not a directory")
    with using_project_root(str(tmp_path)):
        @jit.compile(auto_parallel=False, cache=True)
        def kernel(value):
            return value + 2

        with pytest.raises(OSError):
            kernel(3)
