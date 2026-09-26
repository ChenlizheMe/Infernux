"""Shared fixtures for Infernux integration tests.

All tests use the real C++ backend (Infernux.lib). No fake/mock objects.

Session-scoped ``engine`` fixture (autouse) initialises Vulkan + SDL once for
the entire test run — every test executes with the real C++ engine running.
Per-function ``scene`` fixture creates a fresh Scene for each test.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

# Acceptance helpers live in the repository's scripts package.  Pytest may
# choose ``python/test`` as the import root, so make the repository root
# explicit instead of relying on the caller's working directory.
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

# MCP is a real external InxPackage. Unit tests import its source checkout
# explicitly; production discovers the same code only after package installation
# adds Packages/<reference>/editor to the preload import path.
_MCP_PLUGIN_RUNTIME = (
    Path(__file__).resolve().parents[2]
    / "external"
    / "plugins"
    / "infernux_mcp"
    / "package"
    / "editor"
)
if str(_MCP_PLUGIN_RUNTIME) not in sys.path:
    sys.path.insert(0, str(_MCP_PLUGIN_RUNTIME))

from Infernux.lib import (
    Infernux as NativeEngine,
    LogLevel,
    SceneManager,
    Vector3,
    Physics,
    InputManager,
    NativeRuntimeFrameBarrier,
    lib_dir,
)
from Infernux.resources import resources_path
from Infernux.input import Input
from Infernux.components._component_lifecycle import RuntimeExecutionScheduler
from Infernux.engine.runtime_change_journal import RuntimeFrameBarrier


@pytest.fixture(autouse=True)
def _reset_editor_interaction_state():
    """Prevent process-wide editor interaction state from leaking across tests."""
    from Infernux.engine.interaction import (
        ClipboardService,
        DocumentRegistry,
        EditorInteractionCore,
    )

    previous_core = EditorInteractionCore._instance
    from Infernux.engine.play_mode import PlayModeManager

    # PlayModeManager is a process singleton, but most tests construct a
    # short-lived manager for their own scenario.  Keeping that manager alive
    # makes editor-only preview behavior depend on test order.
    PlayModeManager._instance = None
    registry = DocumentRegistry()
    clipboard = ClipboardService()
    try:
        yield registry
    finally:
        from Infernux.engine.ui.asset_resource_preview import (
            release_all_preview_authoring,
        )
        from Infernux.core.assets import AssetManager
        from Infernux.particle.artifact import ParticleArtifactRegistry

        current_core = EditorInteractionCore._instance
        if current_core is not None and current_core is not previous_core:
            current_core.shutdown()
        release_all_preview_authoring()
        AssetManager.flush()
        ParticleArtifactRegistry.clear()
        # Never publish a test-owned manager into the next test.  The native
        # session fixture does not require a Python PlayModeManager; tests that
        # need one create it explicitly and own it for that test.
        PlayModeManager._instance = None
        EditorInteractionCore._instance = previous_core
        registry.clear()
        clipboard.clear(reason="test_teardown")
        if DocumentRegistry._instance is registry:
            DocumentRegistry._instance = None
        if ClipboardService._instance is clipboard:
            ClipboardService._instance = None


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
    try:
        yield eng
    finally:
        # Full native cleanup. The historical heap corruption here was fixed by
        # (a) SceneManager::Shutdown() destroying all scenes inside Cleanup()
        #     before PhysicsWorld::Shutdown(), and
        # (b) leaking the scene/physics/asset singletons so no engine teardown
        #     ever runs during C++ static destruction.
        # Running cleanup in CI is intentional: it is the regression test for
        # that fix.
        try:
            eng.cleanup()
        finally:
            # The native engine is pointed at this disposable project. Cleaning
            # that directory is sufficient and never touches tracked resources.
            shutil.rmtree(project, ignore_errors=True)


@pytest.fixture(scope="session")
def runtime_scheduler(engine):
    """Install the same shared Python lifecycle owner used by Editor and Player."""
    scheduler = RuntimeExecutionScheduler(name="pytest-native", native_bridge=True)
    manager = SceneManager.instance()
    try:
        yield scheduler
    finally:
        manager.clear_runtime_lifecycle_callbacks()
        scheduler.clear()
        scheduler.unbind_native_bridge()


@pytest.fixture(autouse=True)
def _install_runtime_scheduler_bridge(runtime_scheduler):
    """Restore the authoritative bridge before every native integration test."""
    manager = SceneManager.instance()
    manager.clear_runtime_lifecycle_callbacks()
    runtime_scheduler.end_native_frame()
    manager.set_runtime_lifecycle_callbacks(
        runtime_scheduler.begin_native_frame,
        lambda delta: runtime_scheduler.execute_native_phase("fixed_update", delta),
        lambda delta: runtime_scheduler.execute_native_phase("update", delta),
        lambda delta: runtime_scheduler.execute_native_phase("late_update", delta),
        runtime_scheduler.execute_native_editor_update,
        runtime_scheduler.end_native_frame,
    )
    barrier_map = {
        NativeRuntimeFrameBarrier.TRANSFORM_TO_PHYSICS:
            RuntimeFrameBarrier.TRANSFORM_TO_PHYSICS,
        NativeRuntimeFrameBarrier.PHYSICS_SIMULATION:
            RuntimeFrameBarrier.PHYSICS_SIMULATION,
        NativeRuntimeFrameBarrier.PHYSICS_TO_TRANSFORM:
            RuntimeFrameBarrier.PHYSICS_TO_TRANSFORM,
        NativeRuntimeFrameBarrier.TRANSFORM_RESOLVE:
            RuntimeFrameBarrier.TRANSFORM_RESOLVE,
        NativeRuntimeFrameBarrier.FINAL_TRANSFORM_RESOLVE:
            RuntimeFrameBarrier.FINAL_TRANSFORM_RESOLVE,
        NativeRuntimeFrameBarrier.ANIMATION_TIMELINE:
            RuntimeFrameBarrier.ANIMATION_TIMELINE,
        NativeRuntimeFrameBarrier.RENDER_EXTRACTION:
            RuntimeFrameBarrier.RENDER_EXTRACTION,
        NativeRuntimeFrameBarrier.RENDER_GRAPH:
            RuntimeFrameBarrier.RENDER_GRAPH,
        NativeRuntimeFrameBarrier.SNAPSHOT_PUBLICATION:
            RuntimeFrameBarrier.SNAPSHOT_PUBLICATION,
        NativeRuntimeFrameBarrier.PENDING_DESTROY:
            RuntimeFrameBarrier.PENDING_DESTROY,
    }

    def consume_barrier(native_barrier):
        barrier = barrier_map[native_barrier]
        changes = runtime_scheduler.consume_native_barrier(barrier)
        if changes is not None and barrier == RuntimeFrameBarrier.TRANSFORM_TO_PHYSICS:
            runtime_scheduler.execute_native_phase(
                "physics_pre_step", manager.get_fixed_time_step()
            )
        elif changes is not None and barrier == RuntimeFrameBarrier.PHYSICS_TO_TRANSFORM:
            runtime_scheduler.execute_native_phase(
                "physics_post_step", manager.get_fixed_time_step()
            )

    manager.set_runtime_frame_barrier_callback(consume_barrier)
    runtime_scheduler.bind_native_bridge(manager)
    yield runtime_scheduler


@pytest.fixture()
def scene(engine):
    """Create a disposable World fixture and unload every Scene it creates."""
    sm = SceneManager.instance()
    baseline_scenes = tuple(
        sm.get_scene_at(index) for index in range(int(sm.scene_count))
    )
    baseline_world_ids = {
        int(loaded.world_id) for loaded in baseline_scenes if loaded is not None
    }
    baseline_active = sm.get_active_scene()
    baseline_active_world_id = (
        int(baseline_active.world_id) if baseline_active is not None else 0
    )
    sc = sm.create_scene("pytest_scene")
    sm.set_active_scene(sc)
    yield sc
    # Ensure play mode is stopped (no-op if already stopped)
    if sm.is_playing():
        sm.stop()
    # Additive Scenes are peers in one physics World. Tests using this fixture
    # own every Scene created after entry, including extra round-trip Scenes.
    # Preserve the engine's initial Scene and any explicit outer fixture.
    for index in range(int(sm.scene_count) - 1, -1, -1):
        loaded = sm.get_scene_at(index)
        if loaded is not None and int(loaded.world_id) not in baseline_world_ids:
            sm.unload_scene(loaded)
    remaining = {
        int(sm.get_scene_at(index).world_id): sm.get_scene_at(index)
        for index in range(int(sm.scene_count))
    }
    if baseline_active_world_id in remaining:
        sm.set_active_scene(remaining[baseline_active_world_id])
    # A single-scene commit is allowed to retire a pre-existing Scene. No Scene
    # created by this fixture may survive, regardless of that operation.
    assert set(remaining) <= baseline_world_ids


# ── per-test C++ rigidbody via scene ─────────────────────────────────────

@pytest.fixture
def cpp_rigidbody(scene):
    """Create a C++ Rigidbody through a real scene GameObject."""
    go = scene.create_game_object("_rb_fixture")
    return go.add_component("Rigidbody")


@pytest.fixture(autouse=True)
def _reset_input_state():
    """Reset Input focus state between every test."""
    InputManager.instance().reset_all()
    Input._game_focused = True
    Input._automation_game_input_depth = 0
    Input._game_viewport_origin = (0.0, 0.0)
    yield
    InputManager.instance().reset_all()
    Input._game_focused = True
    Input._automation_game_input_depth = 0
    Input._game_viewport_origin = (0.0, 0.0)


@pytest.fixture(autouse=True)
def _reset_physics_state(engine):
    """Keep process-wide physics settings isolated between tests."""
    earth_gravity = Vector3(0.0, -9.81, 0.0)
    Physics.set_gravity(earth_gravity)
    yield
    Physics.set_gravity(earth_gravity)
