"""Actual Vulkan view submissions after topology and source failures."""
import sys
import time
import textwrap

from infernux.lib import RenderPipelineCallback, Vector3
from infernux.renderstack.render_stack import RenderStack


def test_current_pipeline_failure_never_submits_the_retired_graph(scene, tmp_path, engine):
    from infernux.engine.resources_manager import ResourcesManager
    from infernux.engine.project_context import get_project_root, set_project_root
    from infernux.application import Application
    project = tmp_path / "Project"
    assets = project / "Assets"
    assets.mkdir(parents=True)
    previous_project = get_project_root()
    set_project_root(str(project))
    resources = ResourcesManager(str(project), engine)
    Application._bind_engine(engine, 'editor')
    coordinator = resources._ensure_event_handler().reload
    def write(name, content):
        path = assets / (name + ".py")
        path.write_text(textwrap.dedent(content), encoding="utf-8")
        return str(path)
    source = '''
        from infernux.renderstack.default_forward_pipeline import DefaultForwardPipeline
        calls = []
        class CurrentPipeline(DefaultForwardPipeline):
            name = "Review Current Pipeline"
            def define_topology(self, graph, *, preview=False):
                calls.append("VERSION")
                BODY
    '''
    good_body = "super().define_topology(graph, preview=preview)"
    path = write("NativePipelineSource", source.replace("VERSION", "v1").replace("BODY", good_body))
    coordinator.ensure_ready((path,))
    old_module = sys.modules["NativePipelineSource"]
    stack = RenderStack()
    stack.pipeline_class_name = "Review Current Pipeline"
    camera = scene.create_game_object("Reload submission camera")
    camera.transform.position = Vector3(0, 0, -4)
    camera.add_component("Camera")

    class Host(RenderPipelineCallback):
        def render(self, context, camera):
            stack.render(context, camera)

    host = Host()
    phase = 0
    frames = 0
    pending = None
    observations = []
    failures = []
    deadline = time.monotonic() + 15
    old_revision = None
    old_executions = None
    old_calls = None

    def before_update(_delta):
        nonlocal pending
        try:
            if time.monotonic() > deadline:
                raise AssertionError("Native rendering did not finish its reload phases")
            if pending == "topology":
                write("NativePipelineSource", source.replace("VERSION", "bad-topology").replace(
                    "BODY", "raise ValueError('current topology rejected')"))
                coordinator.ensure_ready((path,))
            elif pending == "syntax":
                write("NativePipelineSource", "def invalid(:\n")
                coordinator.request((path,))
                coordinator.compile_pending()
                coordinator.drain()
                assert coordinator.states[path].status == "failed"
            elif pending == "repair":
                write("NativePipelineSource", source.replace("VERSION", "v3").replace("BODY", good_body))
                coordinator.ensure_ready((path,))
                assert not stack._graph_state.build_failed, (
                    stack._pipeline_source_file, stack._pipeline_module,
                    {key: len(value) for key, value in resources._script_reload_callbacks.items()},
                )
            pending = None
        except BaseException as exc:
            failures.append(exc)
            engine.exit()

    def after_draw():
        nonlocal phase, frames, pending, old_revision, old_executions, old_calls
        try:
            if pending is not None:
                return
            snapshot = engine.renderer_frame_snapshot
            state = stack._graph_state
            executed = snapshot["game_render_graph_current_executed"]
            executions = snapshot["game_render_graph_execution_count"]
            observations.append((phase, executions))
            if phase == 0:
                if not executed:
                    return
                assert state.description is not None and not state.build_failed
                old_revision = state.description.source_revision
                old_executions = executions
                old_calls = tuple(old_module.calls)
            elif phase in (1, 2):
                assert executions == old_executions, (phase, old_executions, snapshot)
                assert state.description is None and state.build_failed
                assert tuple(old_module.calls) == old_calls
            elif phase == 3:
                assert state.description is not None, (state.build_failed, state.pipeline_error, sys.modules['NativePipelineSource'].calls)
                assert state.description.source_revision != old_revision
                assert executed and executions > 0, snapshot
                assert set(sys.modules["NativePipelineSource"].calls) == {"v3"}
                assert tuple(old_module.calls) == old_calls
            frames += 1
            if frames < 3:
                return
            frames = 0
            if phase == 0:
                pending = "topology"
            elif phase == 1:
                assert sys.modules["NativePipelineSource"].calls == ["bad-topology"]
                pending = "syntax"
            elif phase == 2:
                assert "NativePipelineSource" not in sys.modules
                pending = "repair"
            else:
                phase = 4
                engine.exit()
                return
            phase += 1
        except BaseException as exc:
            failures.append(exc)
            engine.exit()

    engine.resize_game_render_target(64, 64)
    engine.set_scene_view_visible(False)
    engine.set_game_camera_enabled(True)
    engine.set_play_mode_rendering(True)
    engine.set_render_pipeline(host)
    try:
        engine.set_pre_scene_update_callback(before_update)
        engine.set_post_draw_callback(after_draw)
        engine.run()
        if failures:
            raise failures[0]
        assert phase == 4, observations
        assert len({count for phase, count in observations if phase in (1, 2)}) == 1
    finally:
        engine.set_pre_scene_update_callback(None)
        engine.set_post_draw_callback(None)
        engine.set_render_pipeline(None)
        engine.set_game_camera_enabled(False)
        engine.set_play_mode_rendering(False)
        engine.set_scene_view_visible(True)
        stack.on_destroy()
        resources.cleanup()
        Application._unbind_engine(engine)
        set_project_root(previous_project)
