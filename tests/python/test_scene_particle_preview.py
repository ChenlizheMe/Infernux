from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest

from infernux.components import ParticleSystem
from infernux.engine.ui.scene_view_panel import SceneViewPanel


def _panel() -> SceneViewPanel:
    panel = SceneViewPanel(engine=None)
    panel._play_mode_manager = SimpleNamespace(is_edit_mode=True)
    return panel


def test_scene_particle_preview_follows_primary_selection():
    panel = _panel()
    component = ParticleSystem()
    calls = []
    component.editor_preview_begin = lambda: calls.append("begin") or True
    component.editor_preview_pause = lambda: calls.append("pause") or True
    component.editor_preview_suspend = component.editor_preview_pause
    component.editor_preview_is_playing = lambda: True
    selected = SimpleNamespace(get_py_components=lambda: [component])

    panel._on_particle_preview_selection(selected)
    assert panel._particle_preview_component is component
    assert panel._particle_preview_playing is True
    assert calls == ["begin"]

    panel._on_particle_preview_selection(None)
    assert panel._particle_preview_component is None
    assert calls == ["begin", "pause"]


def test_scene_particle_preview_suspends_when_selected_owner_becomes_inactive():
    panel = _panel()
    calls = []
    hierarchy_reads = []

    class Owner:
        id = 9
        active = True

        @property
        def active_in_hierarchy(self):
            hierarchy_reads.append("tick")
            return self.active

    owner = Owner()
    component = SimpleNamespace(
        game_object=owner,
        editor_preview_pause=lambda: calls.append("pause") or True,
        editor_preview_suspend=lambda: calls.append("suspend") or True,
        editor_preview_update=lambda *_args: calls.append("update") or True,
    )
    panel._particle_preview_component = component
    panel._particle_preview_object = owner
    panel._particle_preview_playing = True

    owner.active = False
    panel._tick_particle_preview(0.016)

    assert calls == ["suspend"]
    assert panel._particle_preview_component is None
    assert panel._particle_preview_object is None
    assert panel._particle_preview_playing is False
    assert hierarchy_reads


def test_scene_particle_preview_never_binds_or_controls_play_mode_component():
    panel = _panel()
    panel._play_mode_manager.is_edit_mode = False
    calls = []
    component = SimpleNamespace(
        editor_preview_begin=lambda: calls.append("begin") or True,
        editor_preview_end=lambda: calls.append("end"),
    )
    selected = SimpleNamespace(get_py_components=lambda: [component])

    panel._on_particle_preview_selection(selected)

    assert panel._particle_preview_component is None
    assert panel._particle_preview_object is None
    assert calls == []


def test_scene_particle_preview_pre_render_ticks_hidden_scene_tab(monkeypatch):
    panel = _panel()
    calls = []
    panel._tick_particle_preview = calls.append
    clock = iter((10.0, 10.025))
    scene_view_module = importlib.import_module("infernux.engine.ui.scene_view_panel")
    monkeypatch.setattr(scene_view_module.time, "monotonic", lambda: next(clock))

    panel._pre_render(None)
    panel._pre_render(None)

    assert calls == pytest.approx([0.0, 0.025])


def test_particle_preview_begin_restarts_every_enabled_emitter_after_rebuild():
    component = ParticleSystem()
    component._editor_preview_play_requested = True
    component._ensure_editor_preview_runtime = lambda: (True, True)
    calls = []
    component.restart = lambda **kwargs: calls.append(kwargs) or True

    assert component.editor_preview_begin() is True
    assert calls == [{"honor_play_on_start": False}]


def test_particle_preview_stop_drops_native_draws_immediately():
    component = ParticleSystem()
    component._batch_id = 1
    component._gpu_controllers = [object()]
    calls = []
    component.stop = lambda: calls.append("stop") or True
    component._remove_native_batch = lambda: calls.append("remove")

    assert component.editor_preview_stop() is True
    assert calls == ["stop", "remove"]
    assert component._editor_preview_play_requested is False
    assert component._playing is False


def test_particle_preview_update_repairs_lost_native_residency():
    component = ParticleSystem()
    component._editor_preview_active = True
    component._editor_preview_play_requested = True
    calls = []
    component._ensure_editor_preview_runtime = lambda: calls.append("ensure") or (True, True)
    component.restart = lambda **kwargs: calls.append(("restart", kwargs)) or True
    component.update = lambda delta: calls.append(("update", delta))
    component._gpu_runtime_resident = lambda: True

    assert component.editor_preview_update(0.25, 2.0) is True
    assert calls == [
        "ensure",
        ("restart", {"honor_play_on_start": False}),
        ("update", 0.5),
    ]


def test_particle_preview_republishes_stale_python_runtime(monkeypatch):
    component = ParticleSystem()
    component._gpu_controllers = [object()]
    component._gpu_emitter_ids = [7]
    component._gpu_emitter_indices = [0]
    native = SimpleNamespace(_gpu_particle_artifact_revision=lambda emitter_id: 1 if emitter_id == 8 else 0)
    monkeypatch.setattr(
        ParticleSystem,
        "_native_engine",
        staticmethod(lambda: native),
    )
    calls = []

    def compile_asset(*, force=False):
        calls.append(
            (
                force,
                list(component._gpu_controllers),
                list(component._gpu_emitter_ids),
            )
        )
        component._gpu_controllers = [object()]
        component._gpu_emitter_ids = [8]
        component._gpu_emitter_indices = [0]
        return True

    component._load_saved_artifact = compile_asset

    assert component._ensure_editor_preview_runtime() == (True, True)
    assert calls == [(True, [], [])]


def test_particle_preview_rebinds_after_play_mode_restores_scene(monkeypatch):
    from infernux.engine.play_mode import PlayModeState

    panel = _panel()
    old_component = SimpleNamespace(editor_preview_end=lambda: None)
    panel._particle_preview_component = old_component
    panel._particle_preview_object = object()
    panel._particle_preview_playing = True
    panel._particle_preview_prepared = True
    calls = []
    monkeypatch.setattr(
        panel,
        "_restore_particle_preview_selection",
        lambda: calls.append("restore"),
    )

    panel._on_particle_preview_play_mode_changed(
        SimpleNamespace(
            old_state=PlayModeState.PLAYING,
            new_state=PlayModeState.EDIT,
        )
    )

    assert panel._particle_preview_component is None
    assert panel._particle_preview_object is None
    assert panel._particle_preview_playing is False
    assert panel._particle_preview_prepared is False
    assert panel._particle_preview_restore_pending is True
    assert calls == []

    deferred_module = importlib.import_module("infernux.engine.deferred_task")
    monkeypatch.setattr(
        deferred_module.DeferredTaskRunner,
        "instance",
        classmethod(lambda _cls: SimpleNamespace(is_busy=False)),
    )
    panel._restore_particle_preview_if_ready()

    assert panel._particle_preview_restore_pending is False
    assert calls == ["restore"]


def test_invalid_particle_preview_wrapper_never_removes_native_batch():
    panel = _panel()
    calls = []
    panel._particle_preview_component = SimpleNamespace(_remove_native_batch=lambda: calls.append("remove"))
    panel._particle_preview_object = object()

    panel._discard_invalid_particle_preview()

    assert panel._particle_preview_component is None
    assert calls == []


def test_particle_validate_defers_definition_rebuild_to_update():
    from infernux.core.asset_ref import ParticleGraphRef

    component = ParticleSystem()
    component.graph = ParticleGraphRef("graph-guid", "Assets/Test.particlegraph")
    component._runtime_definition_signature = component._definition_signature()
    component._runtime_rebuild_pending = False
    calls = []
    component._sync_serialized_instance_overrides = lambda: calls.append("sync")
    component._load_saved_artifact = lambda *, force=False: calls.append(("compile", force)) or True
    component._gpu_controllers = [object()]
    component._particle_metadata = object()
    component._update_gpu_particle_graph = lambda delta: calls.append(("update", delta))

    component.random_seed = 17
    component.on_validate()

    assert component._runtime_rebuild_pending is True
    assert calls == []

    component.update(0.25)

    assert component._runtime_rebuild_pending is False
    assert calls == ["sync", ("compile", True), ("update", 0.25)]


def test_particle_preview_entering_play_forgets_handles_without_runtime_command():
    from infernux.engine.play_mode import PlayModeState

    panel = _panel()
    calls = []
    panel._particle_preview_component = SimpleNamespace(editor_preview_end=lambda: calls.append("end"))
    panel._particle_preview_object = object()
    panel._particle_preview_prepared = True

    panel._on_particle_preview_play_mode_changed(
        SimpleNamespace(
            old_state=PlayModeState.EDIT,
            new_state=PlayModeState.PLAYING,
        )
    )

    assert panel._particle_preview_component is None
    assert panel._particle_preview_object is None
    assert calls == []


@pytest.mark.parametrize("play_on_awake", [False, True])
def test_play_mode_deserialize_initializes_particle_runtime_from_play_on_awake(monkeypatch, play_on_awake):
    from infernux.engine.play_mode import PlayModeManager

    monkeypatch.setattr(
        PlayModeManager,
        "instance",
        classmethod(lambda _cls: SimpleNamespace(is_playing=True)),
    )
    component = ParticleSystem()
    component.play_on_awake = play_on_awake

    component.on_after_deserialize()

    assert component._playing is play_on_awake


def test_particle_replacement_retires_edit_preview_native_batch(monkeypatch):
    from infernux.components.component import InxComponent

    component = ParticleSystem()
    component._gpu_controllers = [object()]
    calls = []
    component._remove_native_batch = lambda: calls.append("remove")
    component._clear_runtime_state = lambda: calls.append("clear")
    monkeypatch.setattr(
        InxComponent,
        "_detach_native_binding_for_replacement",
        lambda _self: calls.append("detach"),
    )

    component._detach_native_binding_for_replacement()

    assert calls == ["remove", "clear", "detach"]
