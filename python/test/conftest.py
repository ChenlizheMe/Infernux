"""Shared fixtures for InfEngine integration tests.

All tests use the real C++ backend (InfEngine.lib). No fake/mock objects.

Session-scoped ``engine`` fixture (autouse) initialises Vulkan + SDL once for
the entire test run — every test executes with the real C++ engine running.
Per-function ``scene`` fixture creates a fresh Scene for each test.
"""
from __future__ import annotations

import os
import tempfile

import pytest

from InfEngine.lib import (
    InfEngine as NativeEngine,
    LogLevel,
    SceneManager,
    Vector3,
    Physics,
    InputManager,
    lib_dir,
)
from InfEngine.resources import resources_path
from InfEngine.input import Input


# ── session-scoped engine (Vulkan + SDL, created once for ALL tests) ─────

@pytest.fixture(scope="session", autouse=True)
def engine():
    """Start the real C++ engine with a tiny off-screen window.

    ``autouse=True`` ensures every test in the suite runs with the engine
    initialised — Vulkan renderer, SDL window, physics world, and input
    subsystem are all live.
    """
    project = tempfile.mkdtemp(prefix="infengine_test_")
    os.makedirs(os.path.join(project, "ProjectSettings"), exist_ok=True)

    eng = NativeEngine(lib_dir)
    eng.set_log_level(LogLevel.Warn)
    eng.init_renderer(64, 64, project, resources_path)
    yield eng
    eng.cleanup()


@pytest.fixture()
def scene(engine):
    """Create a disposable Scene and make it active.  Cleaned up after each test."""
    sm = SceneManager.instance()
    sc = sm.create_scene("pytest_scene")
    sm.set_active_scene(sc)
    yield sc
    # Ensure play mode is stopped (no-op if already stopped)
    if sm.is_playing():
        sm.stop()


# ── per-test C++ rigidbody via scene ─────────────────────────────────────

@pytest.fixture
def cpp_rigidbody(scene):
    """Create a C++ Rigidbody through a real scene GameObject."""
    go = scene.create_game_object("_rb_fixture")
    return go.add_component("Rigidbody")


@pytest.fixture(autouse=True)
def _reset_input_state():
    """Reset Input focus state between every test."""
    Input._game_focused = True
    Input._game_viewport_origin = (0.0, 0.0)
    yield
    Input._game_focused = True
    Input._game_viewport_origin = (0.0, 0.0)
