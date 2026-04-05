"""Tests for Infernux.jit public helpers and startup self-repair."""

from __future__ import annotations

import Infernux.jit as jit


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


class TestPrecompileJit:
    def test_precompile_jit_is_noop(self):
        # precompile_jit() is kept for backward compat; it does nothing.
        jit.precompile_jit()  # should not raise