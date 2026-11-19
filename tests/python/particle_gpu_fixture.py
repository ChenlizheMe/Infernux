"""Run particle pixel fixtures in the ordinary prepared Play World."""
from contextlib import contextmanager
from time import monotonic
from functools import wraps

from infernux.lib import _Infernux


def finish_transition(state):
    deadline = monotonic() + 30.0
    while state.runner.is_busy:
        assert monotonic() < deadline, state.manager.state
        state.runner.tick()
        state.host.tick(0.0)


@contextmanager
def playing_particle_world(state, width, height, pipeline):
    from infernux.engine.runtime_screen_ui import RuntimeScreenUISubmission
    # A complete World publication includes its immutable empty UI packet too.
    state.host._screen_ui_submission = RuntimeScreenUISubmission(state.host)
    state.host.set_render_pipeline(pipeline)
    assert state.manager.enter_play_mode()
    finish_transition(state)
    assert state.manager.is_playing
    identities = tuple(state.native._runtime_world_ids)
    assert len(identities) == 1
    engine = state.host.get_native_engine()
    # The viewport follows the Play clone, rather than the retained document.
    engine._request_game_screen("particle_gpu_fixture", identities[0], width, height, required=True)
    try:
        with _Infernux._RuntimeWorldScope(identities[0]):
            runtime_scene = state.native.get_active_scene()
            assert runtime_scene is not state.scene
        yield runtime_scene
    finally:
        engine._release_game_screen("particle_gpu_fixture")
        assert state.manager.exit_play_mode()
        finish_transition(state)
        assert state.manager.is_edit_mode
        state.host.set_render_pipeline(None)
        state.host._screen_ui_submission = None


def particle_callback(state):
    # Keep the native owner loop outside a World query/mutation scope.
    world_id, = state.native._runtime_world_ids

    def decorate(callback):
        @wraps(callback)
        def scoped(*args):
            with _Infernux._RuntimeWorldScope(world_id):
                return callback(*args)
        return scoped
    return decorate


def run_particle_frames(state, finished):
    deadline = monotonic() + 60.0
    while not finished():
        assert monotonic() < deadline, state.host.get_native_engine().renderer_frame_snapshot
        # The normal component lifecycle consumes this caller-controlled delta.
        state.host.tick(1.0 / 120.0)
