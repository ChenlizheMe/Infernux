"""Public updates use native phase boundaries and preload cleanup ownership."""

import pytest

from infernux.components import InxComponent
from infernux.lib import Physics, RenderPipelineCallback, SceneManager, Vector3
from infernux.runtime_services import register_update
from infernux.timing import Time
from test_plugin_preload_failures import package_project


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
def live_clock(engine, scene):
    manager = SceneManager.instance()
    manager.play()
    try:
        yield manager
    finally:
        manager.stop()
        Time.time_scale = 1.0


@pytest.mark.parametrize("scale,fixed_steps", [(1.0, 2), (0.5, 1), (0.0, 0)])
def test_native_phases_supply_scaled_and_independent_time(engine, scene, live_clock, updates, scale, fixed_steps):
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
    engine.tick(0.04)
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


def test_editor_updates_do_not_run_in_play_and_game_updates_do_not_run_after_stop(engine, scene, runtime_scheduler, updates):
    observed = []
    editor = updates(lambda dt: observed.append(("editor", dt)), phase="editor_update")
    game = updates(lambda dt: observed.append(("game", dt)))
    manager = SceneManager.instance()
    engine.tick(0.025)
    assert observed == [("editor", pytest.approx(0.025))]
    observed.clear()
    manager.play()
    engine.tick(0.01)
    assert [name for name, _ in observed] == ["game"]
    observed.clear()
    manager.stop()
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
