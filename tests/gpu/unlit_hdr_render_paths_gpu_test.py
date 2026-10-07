"""Check HDR Unlit surface colors across render paths using GPU readback."""

import json
from pathlib import Path
import tempfile

import numpy as np
import infernux as inx
from infernux.core.asset_ref import RenderEffectRef
from infernux.core.assets import AssetManager
from infernux.lib import ConsolePanel, InxMaterial, RenderPipelineCallback, SceneManager, Vector3
from infernux.renderstack import DefaultDeferredPipeline, DefaultForwardPipeline, DefaultForwardPlusPipeline
from infernux.renderstack.render_effect import RenderEffect
from infernux.renderstack.render_effect_asset import EffectAssetReference, RenderEffectAsset


SCALE_SHADER = '''#version 450
ShaderInfo {
 Name "HDR Parity Scale" Hidden On Capabilities [Fullscreen]
 Resources { Texture2D _SourceTex } Inputs { Float2 inUV } Outputs { Float4 outColor }
}
void main() { vec4 c = texture(_SourceTex, inUV); outColor = vec4(c.rgb * 0.125, c.a); }
'''


@inx.renderstack.render_effect_feature('test.hdr_parity.scale', route_policy=inx.renderstack.RoutePolicy.INLINE)
class ScaleEffect(inx.renderstack.FullScreenEffect):
    name = 'HDR Parity Scale'
    injection_point = 'before_post_process'
    modifies = {'color'}

    def setup_passes(self, graph, bus):
        self.apply_single_source_effect(graph, bus, output_name='scaled', pass_name='Scale',
            shader_name='HDR Parity Scale', format=inx.rendergraph.Format.RGBA16_SFLOAT)


class StackHost(RenderPipelineCallback):
    def __init__(self):
        super().__init__()
        self.stack = inx.RenderStack()

    def render(self, context, camera):
        self.stack.render(context, camera)


def main():
    plans = [(DefaultForwardPipeline, 1), (DefaultForwardPipeline, 4),
             (DefaultForwardPlusPipeline, 1), (DefaultForwardPlusPipeline, 4),
             (DefaultDeferredPipeline, 1)]
    phases = ('baseline', 'after_opaque', 'after_camera_ui', 'final', 'restored')
    authored_color = np.array([.4, .2, 2.])
    linear_color = np.where(authored_color <= .04045, authored_color / 12.92,
                            ((authored_color + .055) / 1.055) ** 2.4)
    evidence = []
    with tempfile.TemporaryDirectory(prefix='infernux-unlit-hdr-parity-') as folder:
        project = Path(folder)
        for name in ('Assets', 'Packages', 'ProjectSettings'):
            (project / name).mkdir()
        shader = project / 'Assets/Scale.frag'
        shader.write_text(SCALE_SHADER, encoding='utf-8')
        mesh = project / 'Assets/Quad.obj'
        mesh.write_text('v -1 -1 0\nv 1 -1 0\nv 1 1 0\nv -1 1 0\nvn 0 0 -1\nf 1//1 3//1 2//1\nf 1//1 4//1 3//1\n', encoding='ascii')
        frontend = inx.Engine()
        native = frontend.get_native_engine()
        console = ConsolePanel()
        host = StackHost()
        failures = []
        complete = False
        try:
            frontend.init_renderer(160, 120, str(project))
            frontend.resize_game_render_target(160, 120)
            native.set_scene_view_visible(False)
            native.set_editor_fps_cap(240)
            native.set_editor_idle_fps(0)
            database = frontend.get_asset_database()
            imported_mesh = AssetManager.import_asset(str(mesh), database=database)
            imported_shader = AssetManager.import_asset(str(shader), database=database)
            assert imported_mesh and imported_shader
            scale = RenderEffect(RenderEffectAsset('test.hdr_parity.scale',
                dependencies=(EffectAssetReference(guid=imported_shader.guid),)))
            scene = SceneManager.instance().get_active_scene()
            for obj in scene.get_root_objects():
                scene.destroy_game_object(obj)
            camera = scene.create_game_object('HDR Camera')
            camera.transform.position = Vector3(0, 0, -4)
            camera.add_component('Camera')
            obj = scene.create_game_object('HDR Quad')
            obj.transform.local_scale = Vector3(20, 20, 1)
            renderer = obj.add_component('MeshRenderer')
            renderer.set_mesh_asset_guid(imported_mesh.guid)
            material = InxMaterial.create_default_unlit()
            material.set_color('baseColor', *authored_color, 1.)
            renderer.set_material(0, material)
            plan = phase = frame = changed = 0
            ticket = baseline = None

            def switch_pipeline():
                base, samples = plans[plan]
                pipeline = base()
                pipeline.shadow_resolution = 256
                if base is not DefaultDeferredPipeline:
                    pipeline.msaa_samples = samples
                host.stack._pipeline = pipeline
                host.stack.invalidate_graph()

            switch_pipeline()
            frontend.set_render_pipeline(host)
            native.set_game_camera_enabled(True)

            def after_draw():
                nonlocal plan, phase, frame, changed, ticket, baseline, complete
                try:
                    frame += 1
                    if ticket is None and frame >= changed + 10:
                        ticket = frontend.request_render_target_readback(True)
                    elif ticket is not None and ticket.done:
                        pixels = ticket.result_numpy().copy().astype(np.float32)
                        label = phases[phase]
                        value = np.clip(linear_color * (1. if label in ('baseline', 'restored') else .125), 0., 1.)
                        expected = np.where(value <= .0031308, value * 12.92, 1.055 * value ** (1 / 2.4) - .055)
                        actual = pixels[60, 80, :3]
                        assert not host.stack.effect_compile_errors, host.stack.effect_compile_errors
                        np.testing.assert_allclose(actual, expected, atol=.002, rtol=0,
                            err_msg=f'{plans[plan][0].name}/{label}')
                        if label == 'baseline':
                            baseline = pixels
                        if label == 'restored':
                            np.testing.assert_array_equal(pixels, baseline)
                        issues = [e for e in console._get_visible_log_snapshot(2000)
                                  if e['level'] in ('ERROR', 'FATAL', 'WARN', 'WARNING')]
                        assert not issues, issues
                        evidence.append({'pipeline': plans[plan][0].name, 'samples': plans[plan][1],
                            'phase': label, 'actual': actual.tolist(), 'expected': expected.tolist()})
                        print('PASS ' + json.dumps(evidence[-1]), flush=True)
                        phase += 1
                        host.stack.effect_slots = []
                        if phase == len(phases):
                            plan += 1
                            if plan == len(plans):
                                complete = True
                                native.exit()
                                return
                            phase = 0
                            switch_pipeline()
                        if phases[phase] not in ('baseline', 'restored'):
                            host.stack.add_effect_slot(phases[phase], RenderEffectRef(effect=scale))
                        host.stack.invalidate_graph()
                        changed = frame
                        ticket = None
                    if frame > 600:
                        raise AssertionError('HDR render-path parity audit timed out')
                except BaseException as error:
                    failures.append(error)
                    native.exit()

            native.set_post_draw_callback(after_draw)
            native.run()
            if failures:
                raise failures[0]
            assert complete
        finally:
            frontend.set_render_pipeline(None)
            host.stack.on_destroy()
            native.cleanup()
    print('PASS 25 HDR Unlit render-path phases: ' + json.dumps({'package': inx.__file__, 'phases': evidence}), flush=True)


if __name__ == '__main__':
    main()
