"""Camera churn must release view bindings while resident particles keep drawing."""
import json

import numpy as np
import pytest

import infernux as inx
from infernux.core.asset_ref import ParticleGraphRef
from infernux.core.assets import AssetManager
from infernux.lib import CameraClearFlags, ConsolePanel, SceneManager, Vector3, vec4f
from infernux.particle import ParticleScriptCompiler
from infernux.runtime_services import install_runtime_service, remove_runtime_service
from tests.gpu.shared_camera_gpu_test import MONITOR, Monitor, Router
from tests.python.test_particle_camera_layers import ParticleLayerPipeline, layer_source


@pytest.mark.parametrize('kind,sort', [
    ('sprite', 'none'), ('sprite', 'back_to_front'), ('ribbon', 'none'),
    ('mesh', 'none'), ('mesh', 'back_to_front'),
])
def test_camera_churn_releases_particle_view_bindings(engine, scene, tmp_path, monkeypatch, kind, sort):
    database = engine.get_asset_database()
    monkeypatch.setattr(AssetManager, '_engine', engine)
    monkeypatch.setattr(AssetManager, '_asset_database', database)
    assets = []

    def import_asset(name, text):
        path = tmp_path / name
        path.write_text(text, encoding='utf-8')
        result = AssetManager.import_asset(str(path), database=database)
        assert result.succeeded, result.error
        assets.append(path)
        return result.guid

    import_asset('Monitor.frag', MONITOR)
    graph_path = tmp_path / 'Resident.particlegraph'
    ParticleScriptCompiler().parse(layer_source(kind, sort)).save(str(graph_path))
    result = AssetManager.import_asset(str(graph_path), database=database)
    assert result.succeeded, result.error
    graph_guid = result.guid
    assets.append(graph_path)
    targets = []
    for index in range(2):
        guid = import_asset(f'View{index}.rendertexture', json.dumps({
            '$type': 'render_texture', 'size': {'width': 64, 'height': 64},
            'format': 'rgba16_sfloat', 'depth_format': 'd32_sfloat', 'samples': 1,
            'filter': 'nearest', 'storage': False, 'sampled_depth': False,
        }))
        targets.append(inx.RenderTexture._from_native(engine._load_render_texture(guid)))

    def camera(index):
        obj = scene.create_game_object(f'Particle View {index}')
        obj.transform.position = Vector3(0, 0, -5)
        result = obj.add_component('Camera')
        result.target_texture = targets[index]
        result.depth = index + 1
        result.culling_mask = 128
        result.clear_flags = CameraClearFlags.SolidColor
        result.background_color = vec4f(0, 0, 0, 0)
        return obj, result

    retained_owner, retained = camera(0)
    transient_owner, transient = camera(1)
    monitor = scene.create_game_object('View Retirement Monitor').add_component('Camera')
    monitor.culling_mask, monitor.depth = 0, -1
    scene.main_camera = monitor
    router = Router()
    router.shared = ParticleLayerPipeline('forward')
    router.monitor = Monitor()
    # The monitor reads only the retained producer. A destroyed camera's target
    # deliberately has no producer, so importing it would invalidate the frame.
    router.monitor.builds, router.monitor.targets = 0, [targets[0], targets[0]]
    router.monitor_id = monitor.component_id
    owner = scene.create_game_object('Resident Particles')
    owner.layer = 7
    component = owner.add_py_component(inx.ParticleSystem())
    manager, console = SceneManager.instance(), ConsolePanel()
    install_runtime_service('gpu-particles', engine)
    frame, changed, ticket, request = 0, 0, None, None
    failures, results = [], []
    initial_batch = retained_view = None
    seen_cameras = {transient.component_id}
    # 16 destroy/recreate cycles, with a live independent camera throughout.
    phase_count = 33
    try:
        engine.set_render_pipeline(router)
        engine.resize_game_render_target(128, 64)
        engine.set_game_camera_enabled(True)
        engine.set_scene_view_visible(False)
        manager.play()
        manager.pause()
        engine.set_play_mode_rendering(True)
        component.graph = ParticleGraphRef(guid=graph_guid)
        component.awake()
        assert component.play(), component.last_compile_error

        def update(_delta):
            component.update(1. / 120.)

        def after_draw():
            nonlocal frame, changed, ticket, request, transient_owner, transient, initial_batch, retained_view
            try:
                frame += 1
                assert frame < 1200, results
                if ticket is None and frame >= changed + 12:
                    ticket = engine.request_render_target_readback(True)
                    request = component.request_gpu_view_diagnostics('game', retained.component_id)
                elif ticket is not None and ticket.done:
                    diagnostic = component.poll_gpu_view_diagnostics('game', request, retained.component_id)
                    if diagnostic['status'] == 'pending':
                        return
                    assert diagnostic['status'] == 'completed' and not diagnostic['error'], diagnostic
                    pixels = ticket.result_numpy()
                    half = pixels.shape[1] // 2
                    counts = [int(np.count_nonzero(pixels[:, side * half:(side + 1) * half, 0] > .9))
                              for side in (0, 1)]
                    runtime = component.runtime_diagnostics()
                    output = diagnostic['outputs'][0]
                    results.append(dict(phase=len(results), present=transient is not None, pixels=counts,
                                        frame=frame, output=output, batch=runtime['batch_id']))
                    (tmp_path / 'particle-view-retirement.json').write_text(json.dumps(results), encoding='utf-8')
                    assert counts[0] > 20 and (transient is None or counts[1] > 20), results
                    assert output['resident_view_binding_count'] == (1 if transient is None else 2), results
                    assert output['visible_count'] == (1 if kind == 'ribbon' else 2), output
                    assert output['sorter_allocated'] == (sort != 'none'), output
                    if initial_batch is None:
                        initial_batch = runtime['batch_id']
                        retained_view = router.shared.views[retained.component_id]
                    assert runtime['batch_id'] == initial_batch
                    assert router.shared.views[retained.component_id] == retained_view
                    assert router.shared.builds == router.monitor.builds == 1
                    assert not [entry for entry in console._get_visible_log_snapshot(1000)
                                if entry['level'] in ('ERROR', 'FATAL', 'WARN', 'WARNING')]
                    if len(results) == phase_count:
                        engine.exit()
                        return
                    if transient is not None:
                        scene.destroy_game_object(transient_owner)
                        transient_owner = transient = None
                    else:
                        transient_owner, transient = camera(1)
                        assert transient.component_id not in seen_cameras
                        seen_cameras.add(transient.component_id)
                    changed, ticket = frame, None
            except BaseException as error:
                failures.append(error)
                engine.exit()

        engine.set_pre_scene_update_callback(update)
        engine.set_post_draw_callback(after_draw)
        engine.run()
        if failures:
            raise failures[0]
        assert len(results) == phase_count and len(seen_cameras) == 17
    finally:
        engine.set_pre_scene_update_callback(None)
        engine.set_post_draw_callback(None)
        component._remove_native_batch()
        engine.set_render_pipeline(None)
        engine.set_play_mode_rendering(False)
        router.dispose()
        manager.stop()
        assert remove_runtime_service('gpu-particles', engine)
        retained.target_texture = None
        if transient is not None:
            transient.target_texture = None
        for path in assets:
            assert database.delete_asset(str(path))
