"""Public updates use native phase boundaries and preload cleanup ownership."""

import pytest
import time
from types import SimpleNamespace

from infernux.components import InxComponent
from infernux.lib import Physics, RenderPipelineCallback, SceneManager, Vector3
from infernux.runtime_services import register_update
from infernux.timing import Time
from test_plugin_preload_failures import package_project
from conftest import install_runtime_scheduler_bridge


@pytest.fixture
def updates():
    registrations = []

    def register(callback, **options):
        registration = register_update(callback, **options)
        registrations.append(registration)
        return registration

    try:
        yield register
    finally:
        for registration in registrations:
            registration.close()


@pytest.fixture(autouse=True)
def render_pipeline(engine):
    class Pipeline(RenderPipelineCallback):
        def render(self, context, camera):
            pass

    engine.set_render_pipeline(Pipeline())
    try:
        yield
    finally:
        engine.set_render_pipeline(None)


@pytest.fixture
def clock_world(engine, scene, runtime_scheduler):
    """Use a real native graph/owner while keeping this a native-phase test.

    These tests intentionally use conftest's native Engine, not a Python Engine
    or project scripting session. Reuse its real bridge for the actual owner.
    """
    from infernux.components._component_lifecycle import RuntimeExecutionScheduler
    from infernux.engine.runtime_change_journal import RuntimeChangeJournal
    from infernux.lib import _Infernux

    manager = SceneManager.instance()
    current = []

    def start():
        assert not current and not manager._runtime_world_ids
        identity = int(manager._begin_runtime_world())
        scheduler = RuntimeExecutionScheduler(name="pytest-update-clock", native_bridge=True,
            world_id=identity, change_journal=RuntimeChangeJournal())
        current.append((identity, scheduler))
        # Begin allocates a World; it does not clone the authored scene. This
        # native phase unit starts with the same empty graph as the scene fixture.
        with _Infernux._RuntimeWorldScope(identity):
            runtime_scene = manager.create_scene("pytest_update_clock")
            manager.set_active_scene(runtime_scene)
            assert runtime_scene._runtime_world_id == identity
        assert manager._publish_runtime_world(identity)
        assert manager.get_active_scene() is runtime_scene
        # This is a temporary callback switch, not host shutdown: the author
        # scheduler and its editor registrations remain alive throughout Play.
        install_runtime_scheduler_bridge(scheduler, reset_callbacks=False)
        manager.play()
        assert manager.is_playing()
        return manager

    def stop():
        if not current:
            return
        identity, scheduler = current[0]
        manager.stop()
        scheduler.end_native_frame()
        scheduler.unbind_native_bridge()
        scheduler.clear()
        manager._begin_runtime_world_retirement(identity)
        install_runtime_scheduler_bridge(runtime_scheduler, reset_callbacks=False)
        assert manager._retire_runtime_world_scenes(identity)
        deadline = time.monotonic() + 10
        while not manager._finish_runtime_world_retirement(identity):
            assert time.monotonic() < deadline, "native clock World retirement did not settle"
            engine.tick(0.0)
        current.clear()
        assert not manager._runtime_world_ids

    try:
        yield SimpleNamespace(start=start, stop=stop)
    finally:
        stop()
        Time.time_scale = 1.0


@pytest.fixture
def live_clock(clock_world):
    yield clock_world.start()


@pytest.mark.parametrize("scale,fixed_steps", [(1.0, 2), (0.5, 1), (0.0, 0)])
def test_native_phases_supply_scaled_and_independent_time(engine, scene, live_clock, updates, scale, fixed_steps):
    scene = live_clock.get_active_scene()
    # Resolve Play's cold GPU control publication before measuring script phases.
    # A query is a synchronous consumer, not another simulation frame or step.
    clock = (Time.frame_count, Time.fixed_time, live_clock.fixed_step_index)
    Physics.raycast(Vector3(0, 0, 0), Vector3(0, -1, 0), 1)
    assert (Time.frame_count, Time.fixed_time, live_clock.fixed_step_index) == clock
    observed = []

    class Probe(InxComponent):
        def update(self, delta_time):
            observed.append(("component", delta_time))

    scene.create_game_object("Update order").add_component(Probe)
    updates(lambda dt: observed.append(("fixed", dt)), phase="fixed_update", unscaled=True)
    updates(lambda dt: observed.append(("update", dt)))
    updates(lambda dt: observed.append(("independent", dt)), unscaled=True)
    updates(lambda dt: observed.append(("late", dt)), phase="late_update")
    Time.time_scale = scale
    # This case checks phase/time semantics, not budget-induced continuation.
    # Give the measured logical frame enough Host time for its cold callbacks.
    work = engine.frame_work
    previous_budget = work.owner_ns, work.foreground_reserve_ns
    work.configure(100_000_000, 0)
    try:
        engine.tick(0.04)
    finally:
        work.configure(*previous_budget)
    assert [name for name, _ in observed] == ["fixed"] * fixed_steps + ["component", "update", "independent", "late"]
    assert [dt for name, dt in observed if name == "fixed"] == pytest.approx([0.02 / scale] * fixed_steps if scale else [])
    assert [dt for _, dt in observed[-4:]] == pytest.approx([0.04 * scale, 0.04 * scale, 0.04, 0.04 * scale])


def test_registration_snapshot_and_immediate_close(engine, live_clock, updates):
    observed = []
    added = False

    def first(delta):
        nonlocal added
        observed.append("first")
        if not added:
            added = True
            updates(lambda dt: observed.append("new"), phase="late_update")
            pending.close()

    first_registration = updates(first)
    pending = updates(lambda dt: observed.append("closed"))
    engine.tick(0.01)
    assert observed == ["first"]
    observed.clear()
    engine.tick(0.01)
    assert observed == ["first", "new"]
    first_registration.enabled = False
    observed.clear()
    engine.tick(0.01)
    assert observed == ["new"]


def test_editor_updates_do_not_run_in_play_and_game_updates_do_not_run_after_stop(engine, scene, runtime_scheduler, updates, clock_world):
    observed = []
    editor = updates(lambda dt: observed.append(("editor", dt)), phase="editor_update")
    game = updates(lambda dt: observed.append(("game", dt)))
    manager = SceneManager.instance()
    engine.tick(0.025)
    assert observed == [("editor", pytest.approx(0.025))]
    observed.clear()
    # The Edit registration above verified game callbacks do not run in Edit.
    # The actual Play owner has its own registration/work mirror.
    game.close()
    clock_world.start()
    game = updates(lambda dt: observed.append(("game", dt)))
    engine.tick(0.01)
    assert [name for name, _ in observed] == ["game"]
    observed.clear()
    clock_world.stop()
    assert all(name == "editor" for name, _ in observed)
    observed.clear()  # Real zero-time retirement service may have delivered Edit callbacks.
    engine.tick(0.01)
    assert [name for name, _ in observed] == ["editor"]
    editor.close()
    game.close()
    observed.clear()
    frames = runtime_scheduler.profiler_snapshot()["native_frame_begins"]
    engine.tick(0.01)
    assert observed == []
    assert runtime_scheduler.profiler_snapshot()["native_frame_begins"] == frames


@pytest.mark.parametrize("fail_preload", [False, True])
def test_preload_reload_disable_and_failure_retire_owned_updates(package_project, engine, scene, fail_preload):
    _root, manager, install = package_project
    source = '''from infernux.lifecycle import InxPreload
class Startup(InxPreload):
    def preload(self, context):
        self.calls = 0
        self.registration = context.add_update(self.tick, phase="editor_update")
    def tick(self, delta):
        self.calls += 1
'''
    if fail_preload:
        source = source.replace('    def tick', '        raise RuntimeError("registration-rejected")\n    def tick')
    state = install("test/owned-updates", {"runtime/startup.py": source})
    if fail_preload:
        assert not state.loaded and "registration-rejected" in state.error
        from infernux.runtime_services import _update_snapshot

        assert _update_snapshot()["editor_update"] == ()
        return
    identity = state.lifecycle[0]["identity"]
    first = manager.preloads.states[identity].instance
    engine.tick(0.01)
    assert first.calls == 1
    assert manager.reload("test/owned-updates").loaded
    second = manager.preloads.states[identity].instance
    assert second is not first and not first.registration.enabled
    engine.tick(0.01)
    assert (first.calls, second.calls) == (1, 1)
    manager.set_enabled("test/owned-updates", False)
    engine.tick(0.01)
    assert second.calls == 1 and not second.registration.enabled
