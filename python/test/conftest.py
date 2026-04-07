"""Shared fixtures for Infernux integration tests.

All tests use the real C++ backend (Infernux.lib). No fake/mock objects.

Session-scoped ``engine`` fixture (autouse) initialises Vulkan + SDL once for
the entire test run — every test executes with the real C++ engine running.
Per-function ``scene`` fixture creates a fresh Scene for each test.
"""
from __future__ import annotations

import glob
import os
import tempfile

import pytest

from Infernux.lib import (
    Infernux as NativeEngine,
    LogLevel,
    SceneManager,
    Vector3,
    Physics,
    InputManager,
    lib_dir,
)
from Infernux.resources import resources_path
from Infernux.input import Input


# ── session-scoped engine (Vulkan + SDL, created once for ALL tests) ─────

@pytest.fixture(scope="session", autouse=True)
def engine():
    """Start the real C++ engine with a tiny off-screen window.

    ``autouse=True`` ensures every test in the suite runs with the engine
    initialised — Vulkan renderer, SDL window, physics world, and input
    subsystem are all live.
    """
    project = tempfile.mkdtemp(prefix="infernux_test_")
    os.makedirs(os.path.join(project, "ProjectSettings"), exist_ok=True)

    eng = NativeEngine(lib_dir)
    eng.set_log_level(LogLevel.Warn)
    eng.init_renderer(64, 64, project, resources_path)
    yield eng
    # NOTE: eng.cleanup() is intentionally omitted.  The native Cleanup()
    # path triggers heap corruption when Jolt bodies are destroyed en-masse
    # during scene teardown (known engine bug).  Since the test process is
    # about to exit, the OS reclaims all resources anyway.


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
    # Unload the scene so Jolt physics bodies are destroyed before the next
    # test creates a new scene.  Without this, stale bodies from previous
    # tests remain in the PhysicsWorld and cause access violations when
    # DispatchContactEvents / ForceAllBodiesToCurrentTransform dereference
    # Collider pointers that belong to the old (inactive) scene.
    sm.unload_scene(sc)


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


def pytest_sessionfinish(session, exitstatus):
    """Clean up after the test session.

    1. Remove all .meta files under the project directory.
    2. Hard-exit the process to avoid native heap corruption during C++
       static-singleton destruction (PhysicsECSStore vs SceneManager ordering).
    """
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for meta in glob.glob(os.path.join(root, "**", "*.meta"), recursive=True):
        try:
            os.remove(meta)
        except OSError:
            pass
    # Redirect faulthandler output to devnull so the spurious "access
    # violation" message from DLL unload during _exit() is suppressed,
    # while still letting faulthandler catch the SEH exception (which
    # preserves exit code 0 instead of 0xC0000409).
    import faulthandler
    _devnull = open(os.devnull, "w")               # noqa: SIM115
    faulthandler.enable(file=_devnull)
    os._exit(exitstatus)
