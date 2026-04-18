"""Tests for Infernux.jit public helpers and startup self-repair."""

from __future__ import annotations

import py_compile

import Infernux.jit as jit
import Infernux._jit_kernels as jit_kernels


class TestEnsureJitRuntime:
    def test_no_install_when_already_available(self, monkeypatch):
        monkeypatch.setattr(jit, "JIT_AVAILABLE", True)

        def _unexpected(*_args, **_kwargs):
            raise AssertionError("should not be called")

        monkeypatch.setattr(jit, "_ensure_pip", _unexpected)
        monkeypatch.setattr(jit, "_install_numba", _unexpected)

        assert jit.ensure_jit_runtime() is True

    def test_respects_disable_env(self, monkeypatch):
        monkeypatch.setattr(jit, "JIT_AVAILABLE", False)
        monkeypatch.setattr(jit, "_has_module", lambda _name: False)
        monkeypatch.setenv("INFERNUX_DISABLE_JIT_AUTOINSTALL", "1")

        called = {"install": False}

        def _install() -> bool:
            called["install"] = True
            return True

        monkeypatch.setattr(jit, "_install_numba", _install)

        assert jit.ensure_jit_runtime() is False
        assert called["install"] is False

    def test_installs_and_reloads_when_missing(self, monkeypatch):
        monkeypatch.setattr(jit, "JIT_AVAILABLE", False)
        monkeypatch.delenv("_INFERNUX_JIT_RUNTIME_CHECKED", raising=False)
        monkeypatch.delenv("INFERNUX_DISABLE_JIT_AUTOINSTALL", raising=False)

        state = {"installed": False}

        def _has_module(_name: str) -> bool:
            return state["installed"]

        def _ensure_pip() -> bool:
            return True

        def _install_numba() -> bool:
            state["installed"] = True
            return True

        def _reload() -> None:
            jit.JIT_AVAILABLE = True

        monkeypatch.setattr(jit, "_has_module", _has_module)
        monkeypatch.setattr(jit, "_ensure_pip", _ensure_pip)
        monkeypatch.setattr(jit, "_install_numba", _install_numba)
        monkeypatch.setattr(jit, "_reload_jit_exports", _reload)

        assert jit.ensure_jit_runtime() is True
        assert state["installed"] is True


class TestAutoParallelNjit:
    @staticmethod
    def _fake_numba_njit(*factory_args, **factory_kwargs):
        def _compile(fn, *, mode: str):
            state = {"calls": 0}

            def _compiled(*args, **kwargs):
                state["calls"] += 1
                if mode == "parallel" and kwargs.pop("_force_parallel_fail", False):
                    raise RuntimeError("parallel failed")
                value = fn(*args, **kwargs)
                if mode == "parallel" and state["calls"] > 1:
                    value += 0
                elif mode == "serial" and state["calls"] > 1:
                    for _ in range(2000):
                        value += 0
                return value

            _compiled.mode = mode
            _compiled.state = state
            return _compiled

        if factory_args and callable(factory_args[0]) and len(factory_args) == 1 and not factory_kwargs:
            return _compile(factory_args[0], mode="serial")

        mode = "parallel" if factory_kwargs.get("parallel") else "serial"

        def _decorator(fn):
            return _compile(fn, mode=mode)

        return _decorator

    def test_auto_parallel_builds_dual_variants(self, monkeypatch):
        monkeypatch.setattr(jit_kernels, "_HAS_NUMBA", True)
        monkeypatch.setattr(jit_kernels, "_NUITKA_COMPILED", False)
        monkeypatch.setattr(jit_kernels, "_real_njit", self._fake_numba_njit)

        @jit_kernels.njit(cache=True, auto_parallel=True)
        def burn(n: int) -> int:
            total = 0
            for i in range(n):
                total += i
            return total

        assert burn.py(5) == 10
        assert getattr(burn, "auto_parallel", False) is True
        assert burn.serial.mode == "serial"
        assert burn.parallel.mode == "parallel"
        assert burn.selected_mode == "parallel"

    def test_warmup_can_pin_serial_variant(self, monkeypatch):
        monkeypatch.setattr(jit_kernels, "_HAS_NUMBA", True)
        monkeypatch.setattr(jit_kernels, "_NUITKA_COMPILED", False)
        monkeypatch.setattr(jit_kernels, "_real_njit", self._fake_numba_njit)
        monkeypatch.setattr(
            jit_kernels,
            "_benchmark_callable",
            lambda fn, *_args, **_kwargs: 1.0 if getattr(fn, "mode", "") == "serial" else 2.0,
        )

        @jit_kernels.njit(cache=True, auto_parallel=True)
        def burn(n: int) -> int:
            total = 0
            for i in range(n):
                total += i
            return total

        jit_kernels.warmup(burn, 100)
        assert burn.selected_mode == "serial"
        assert burn(5) == 10

    def test_warmup_can_pin_parallel_after_compile_cost(self, monkeypatch):
        monkeypatch.setattr(jit_kernels, "_HAS_NUMBA", True)
        monkeypatch.setattr(jit_kernels, "_NUITKA_COMPILED", False)
        monkeypatch.setattr(jit_kernels, "_real_njit", self._fake_numba_njit)
        monkeypatch.setattr(
            jit_kernels,
            "_benchmark_callable",
            lambda fn, *_args, **_kwargs: 2.0 if getattr(fn, "mode", "") == "serial" else 1.0,
        )

        @jit_kernels.njit(cache=True, auto_parallel=True)
        def burn(n: int) -> int:
            total = 0
            for i in range(n):
                total += i
            return total

        jit_kernels.warmup(burn, 100)
        assert burn.selected_mode == "parallel"
        assert burn(5) == 10

    def test_parallel_failure_falls_back_to_serial(self, monkeypatch):
        monkeypatch.setattr(jit_kernels, "_HAS_NUMBA", True)
        monkeypatch.setattr(jit_kernels, "_NUITKA_COMPILED", False)
        monkeypatch.setattr(jit_kernels, "_real_njit", self._fake_numba_njit)

        @jit_kernels.njit(cache=True, auto_parallel=True)
        def burn(n: int, _force_parallel_fail: bool = False) -> int:
            total = 0
            for i in range(n):
                total += i
            return total

        assert burn(5, _force_parallel_fail=True) == 10
        assert burn.selected_mode == "serial"

    def test_try_build_auto_parallel_variant_rewrites_range_to_prange(self, monkeypatch):
        used = {"prange": False}

        def _fake_prange(*args):
            used["prange"] = True
            return range(*args)

        monkeypatch.setattr(jit_kernels, "prange", _fake_prange)

        def burn(n: int) -> int:
            total = 0
            for i in range(n):
                total += i
            return total

        rewritten = jit_kernels._try_build_auto_parallel_variant(burn)
        assert rewritten is not None
        assert rewritten(5) == 10
        assert used["prange"] is True

    def test_try_build_auto_parallel_variant_can_load_prebuilt_sidecar(self, tmp_path, monkeypatch):
        sidecar_py = tmp_path / "stress.autop.py"
        sidecar_py.write_text(
            "def burn(n):\n"
            "    total = 0\n"
            "    for i in prange(n):\n"
            "        total += i\n"
            "    return total\n",
            encoding="utf-8",
        )
        sidecar_pyc = sidecar_py.with_suffix(sidecar_py.suffix + "c")
        py_compile.compile(str(sidecar_py), cfile=str(sidecar_pyc), doraise=True)

        used = {"prange": False}

        def _fake_prange(*args):
            used["prange"] = True
            return range(*args)

        monkeypatch.setattr(jit_kernels, "prange", _fake_prange)
        monkeypatch.setattr(
            jit_kernels,
            "_auto_parallel_sidecar_candidates",
            lambda _fn: [str(sidecar_pyc)],
        )

        def burn(n: int) -> int:
            total = 0
            for i in range(n):
                total += i
            return total

        rewritten = jit_kernels._try_build_auto_parallel_variant(burn)
        assert rewritten is not None
        assert rewritten(5) == 10
        assert used["prange"] is True

    def test_try_build_auto_parallel_variant_rewrites_mult_reduction(self, monkeypatch):
        used = {"prange": False}

        def _fake_prange(*args):
            used["prange"] = True
            return range(*args)

        monkeypatch.setattr(jit_kernels, "prange", _fake_prange)

        def product(n: int) -> int:
            acc = 1
            for i in range(1, n + 1):
                acc *= i
            return acc

        rewritten = jit_kernels._try_build_auto_parallel_variant(product)
        assert rewritten is not None
        assert rewritten(5) == 120
        assert used["prange"] is True

    def test_try_build_auto_parallel_variant_rewrites_indexed_array_store(self, monkeypatch):
        used = {"prange": False}

        def _fake_prange(*args):
            used["prange"] = True
            return range(*args)

        monkeypatch.setattr(jit_kernels, "prange", _fake_prange)

        def fill(arr):
            for i in range(len(arr)):
                arr[i] = i * 2

        rewritten = jit_kernels._try_build_auto_parallel_variant(fill)
        assert rewritten is not None
        data = [0] * 5
        rewritten(data)
        assert data == [0, 2, 4, 6, 8]
        assert used["prange"] is True

    def test_try_build_auto_parallel_variant_allows_continue(self, monkeypatch):
        used = {"prange": False}

        def _fake_prange(*args):
            used["prange"] = True
            return range(*args)

        monkeypatch.setattr(jit_kernels, "prange", _fake_prange)

        def evens(n: int) -> int:
            total = 0
            for i in range(n):
                if i % 2 != 0:
                    continue
                total += i
            return total

        rewritten = jit_kernels._try_build_auto_parallel_variant(evens)
        assert rewritten is not None
        assert rewritten(6) == 6  # 0 + 2 + 4
        assert used["prange"] is True

    def test_build_sidecar_source_handles_mult_reduction(self):
        source = (
            "from Infernux.jit import njit\n"
            "@njit(auto_parallel=True)\n"
            "def product(n):\n"
            "    acc = 1\n"
            "    for i in range(1, n + 1):\n"
            "        acc *= i\n"
            "    return acc\n"
        )
        sidecar = jit_kernels.build_auto_parallel_sidecar_source(source)
        assert sidecar is not None
        assert "prange" in sidecar

    def test_build_sidecar_source_handles_indexed_store(self):
        source = (
            "from Infernux.jit import njit\n"
            "@njit(auto_parallel=True)\n"
            "def fill(arr):\n"
            "    for i in range(len(arr)):\n"
            "        arr[i] = i * 2\n"
        )
        sidecar = jit_kernels.build_auto_parallel_sidecar_source(source)
        assert sidecar is not None
        assert "prange" in sidecar