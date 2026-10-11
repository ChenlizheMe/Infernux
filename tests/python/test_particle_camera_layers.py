"""Two real cameras must filter one resident ParticleSystem independently."""
import json

import numpy as np
import pytest

import infernux as inx
from infernux.components import ParticleSystem
from infernux.core.asset_ref import ParticleGraphRef
from infernux.core.assets import AssetManager
from infernux.lib import CameraClearFlags, ConsolePanel, SceneManager, Vector3, vec4f
from infernux.particle import ParticleScriptCompiler
from infernux.rendergraph import Format
from infernux.renderstack import RenderPipeline
from infernux.runtime_services import install_runtime_service, remove_runtime_service
from tests.gpu.shared_camera_gpu_test import MONITOR, Monitor, Router
from tests.python.test_particle_motion_pass import MOTION_DISPLAY, motion_source


class ParticleLayerPipeline(RenderPipeline):
    name = 'Particle Layer Views'

    def __init__(self, material_pass):
        super().__init__()
        self.material_pass = material_pass
        self.builds = 0
        self.views = {}

    def define_topology(self, graph):
        self.builds += 1
        graph.set_msaa_samples(1)
        color = graph.create_texture('color', camera_target=True)
        depth = graph.create_texture('depth', format=Format.D32_SFLOAT, samples=1)
        graph.add_pass('Particles').write_color(color).write_depth(depth).set_clear(
            color=(0, 0, 0, 0), depth=1,
        ).draw_renderers(material_pass='forward' if self.material_pass == 'motion' else self.material_pass)
        if self.material_pass == 'motion':
            motion = graph.create_texture('motion', format=Format.RG16_SFLOAT, samples=1)
            graph.add_pass('Motion').read(depth).write_color(motion).set_clear(color=(0, 0, 0, 0)).draw_renderers(
                material_pass='motion')
            graph.add_pass('MotionMask').write_color(color).set_texture('motionTex', motion).fullscreen_quad(
                'Particle Motion Readback')
        graph.screen_ui_overlay_section(resources={'color'})
        graph.set_output(color)

    def render_camera(self, context, camera, culling):
        self.views[camera.component_id] = int(context.graph_instance_id)
        super().render_camera(context, camera, culling)


def layer_source(kind, sort):
    source = motion_source(kind).replace('ParticleScript\n', 'ParticleScript, AssetReference\n')
    source = source.replace('"f32", 0.0', '"f32", 0.25')
    arguments = f'sort={sort!r}'
    if kind == 'mesh':
        arguments += ', mesh=AssetReference(guid="builtin-mesh:Cube")'
    return source.replace(f'particles.{kind}()', f'particles.{kind}({arguments})')


@pytest.mark.parametrize('material_pass', ['forward', 'forward_plus', 'motion'])
@pytest.mark.parametrize('kind,sort', [
    ('sprite', 'none'), ('sprite', 'back_to_front'), ('ribbon', 'none'),
    ('mesh', 'none'), ('mesh', 'back_to_front'),
])
def test_particle_camera_masks_and_owner_layer_change_without_restart(
        engine, scene, tmp_path, monkeypatch, kind, sort, material_pass):
    database = engine.get_asset_database()
    monkeypatch.setattr(AssetManager, '_engine', engine)
    monkeypatch.setattr(AssetManager, '_asset_database', database)
    assets = []
    for name, text in [('Monitor.frag', MONITOR), ('Motion.frag', MOTION_DISPLAY)]:
        path = tmp_path / name
        path.write_text(text, encoding='utf-8')
        result = AssetManager.import_asset(str(path), database=database)
        assert result.succeeded, result.error
        assets.append(path)
    graph_path = tmp_path / 'Layer.particlegraph'
    ParticleScriptCompiler().parse(layer_source(kind, sort)).save(str(graph_path))
    result = AssetManager.import_asset(str(graph_path), database=database)
    assert result.succeeded, result.error
    graph_guid = result.guid
    assets.append(graph_path)
    cameras, owners, targets = [], [], []
    for index in range(2):
        path = tmp_path / f'View{index}.rendertexture'
        path.write_text(json.dumps({'$type': 'render_texture', 'size': {'width': 64, 'height': 64},
            'format': 'rgba16_sfloat', 'depth_format': 'd32_sfloat', 'samples': 1,
            'filter': 'nearest', 'storage': False, 'sampled_depth': False}), encoding='utf-8')
        imported = AssetManager.import_asset(str(path), database=database)
        assert imported.succeeded, imported.error
        assets.append(path)
        target = inx.RenderTexture._from_native(engine._load_render_texture(imported.guid))
        targets.append(target)
        owner = scene.create_game_object(f'Particle Camera {index}')
        owner.transform.position = Vector3(0, 0, -5)
        camera = owner.add_component('Camera')
        camera.target_texture = target
        camera.depth = index + 1
        camera.clear_flags = CameraClearFlags.SolidColor
        camera.background_color = vec4f(0, 0, 0, 0)
        cameras.append(camera)
        owners.append(owner)
    monitor = scene.create_game_object('Particle Layer Monitor').add_component('Camera')
    monitor.culling_mask = 0
    monitor.depth = -1
    scene.main_camera = monitor
    router = Router()
    router.shared = ParticleLayerPipeline(material_pass)
    router.monitor = Monitor()
    router.monitor.builds = 0
    router.monitor.targets = targets
    router.monitor_id = monitor.component_id
    owner = scene.create_game_object('Resident Layer Particles')
    owner.layer = 7
    component = owner.add_py_component(ParticleSystem())
    manager, console = SceneManager.instance(), ConsolePanel()
    install_runtime_service('gpu-particles', engine)
    # Both cameras keep rendering throughout. Only masks/owner layer change.
    phases = [
        ('included_left', 128, 512, 7, (True, False)),
        ('mask_zero', 0, 512, 7, (False, False)),
        ('included_right', 512, 128, 7, (False, True)),
        ('included_both', 128, 128, 7, (True, True)),
        ('owner_layer_changed', 128, 256, 8, (False, True)),
        ('changed_layer_left', 256, 0, 8, (True, False)),
        ('owner_layer_restored', 128, 512, 7, (True, False)),
        ('outside_frustum', 128, 128, 7, (False, False)),
        ('views_restored', 128, 128, 7, (True, True)),
    ]
    failures, results = [], []
    frame, changed, ticket = 0, 0, None
    initial_batch = initial_views = None
    view_tickets = []
    simulation_tickets = []

    def apply_phase(index):
        _, first, second, layer, _ = phases[index]
        cameras[0].culling_mask, cameras[1].culling_mask = first, second
        owner.layer = layer
        for camera_owner in owners:
            camera_owner.transform.position = Vector3(1000 if phases[index][0] == 'outside_frustum' else 0, 0, -5)

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
        apply_phase(0)

        def update(_delta):
            component.update(1. / 120.)

        def after_draw():
            nonlocal frame, changed, ticket, initial_batch, initial_views
            try:
                frame += 1
                assert frame < 250, results
                if ticket is None and frame >= changed + 16:
                    ticket = engine.request_render_target_readback(True)
                    view_tickets[:] = [
                        component.request_gpu_view_diagnostics('game', camera.component_id)
                        for camera in cameras
                    ]
                    simulation_tickets[:] = [component.request_gpu_diagnostics(1, 2)]
                elif ticket is not None and ticket.done:
                    culls = [
                        component.poll_gpu_view_diagnostics('game', request, camera.component_id)
                        for camera, request in zip(cameras, view_tickets)
                    ]
                    simulation = component.poll_gpu_diagnostics(simulation_tickets[0])
                    if any(item['status'] == 'pending' for item in culls) or simulation['status'] == 'pending':
                        return
                    pixels = ticket.result_numpy()
                    half = pixels.shape[1] // 2
                    counts = [int(np.count_nonzero(pixels[:, side * half:(side + 1) * half, 0] > .9))
                              for side in (0, 1)]
                    phase = phases[len(results)]
                    diagnostics = component.runtime_diagnostics()
                    results.append(dict(phase=phase[0], pixels=counts, batch=diagnostics['batch_id'],
                        size=list(pixels.shape), views=router.shared.views.copy(),
                        masks=[camera.culling_mask for camera in cameras],
                        layers=[owner.layer, component.game_object.layer],
                        cull=culls, diagnostics=diagnostics, simulation=simulation,
                        targets=[target.guid for target in targets]))
                    (tmp_path / 'particle-camera-layers.json').write_text(json.dumps(results), encoding='utf-8')
                    for count, visible, cull in zip(counts, phase[4], culls):
                        assert (count > 20) if visible else (count == 0), results
                        assert cull['status'] == 'completed' and not cull['error'], cull
                        assert len(cull['outputs']) == 1, cull
                        output = cull['outputs'][0]
                        source_count = 1 if kind == 'ribbon' else 2
                        assert output['source_count'] == source_count
                        assert output['visible_count'] == (source_count if visible else 0), output
                        assert output['sorter_allocated'] == (sort != 'none'), output
                    assert simulation['status'] == 'completed' and not simulation['error'], simulation
                    gpu = simulation['emitters'][0]
                    assert gpu['alive_count'] == gpu['initialized_spawn_count'] == 2, gpu
                    # The freelist places both survivors outside the dense draw
                    # range [0, 2); every surface must follow its index buffer.
                    assert {sample['slot_index'] for sample in gpu['state_samples']} == {6, 7}
                    assert diagnostics['resident'] and diagnostics['playing'], diagnostics
                    if initial_batch is None:
                        initial_batch, initial_views = diagnostics['batch_id'], router.shared.views.copy()
                    assert diagnostics['batch_id'] == initial_batch
                    assert router.shared.views == initial_views and len(initial_views) == 2
                    assert len(set(initial_views.values())) == 2
                    assert router.shared.builds == router.monitor.builds == 1
                    assert not [entry for entry in console._get_visible_log_snapshot(1000)
                                if entry['level'] in ('ERROR', 'FATAL', 'WARN', 'WARNING')]
                    if len(results) == len(phases):
                        engine.exit()
                        return
                    apply_phase(len(results))
                    changed, ticket = frame, None
            except BaseException as error:
                failures.append(error)
                engine.exit()

        engine.set_pre_scene_update_callback(update)
        engine.set_post_draw_callback(after_draw)
        engine.run()
        if failures:
            raise failures[0]
        assert len(results) == len(phases)
    finally:
        engine.set_pre_scene_update_callback(None)
        engine.set_post_draw_callback(None)
        component._remove_native_batch()
        engine.set_render_pipeline(None)
        engine.set_play_mode_rendering(False)
        router.dispose()
        manager.stop()
        assert remove_runtime_service('gpu-particles', engine)
        for camera in cameras:
            camera.target_texture = None
        for path in assets:
            assert database.delete_asset(str(path))
