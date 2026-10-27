"""Read actual shader globals from simultaneous Scene, Game and texture views."""
import json
from pathlib import Path
import tempfile

import infernux as inx
import numpy as np
from infernux.core.assets import AssetManager
from infernux.lib import ConsolePanel, SceneManager, Vector3
from infernux.rendergraph import Format
from infernux.renderstack import RenderPipeline
from shared_camera_gpu_test import Monitor, Router


SURFACE = '''#version 450
ShaderInfo {
    Name "View Globals Probe" ShadingModel Unlit Cull Off
    Imports ["Lib Camera"]
    Properties { Float mode = 0 }
}
void surface(out SurfaceData s) {
    s = InitSurfaceData();
    if (material.mode < 0.5)
        s.albedo = vec3(_Globals._ProjectionParams.x,
                        _Globals._ProjectionParams.y / 1000.0,
                        -_Globals._WorldSpaceCameraPos.z / 20.0);
    else if (material.mode < 1.5)
        s.albedo = vec3(_Globals._ScreenParams.xy / 200.0,
                        _Globals._ScreenParams.z * 20.0);
    else
        s.albedo = vec3(getCameraNear(), getCameraFar() / 1000.0,
                        linearEyeDepth(gl_FragCoord.z) / 20.0);
}
'''

POST = '''#version 450
ShaderInfo {
    Name "View Globals Post" Hidden On Capabilities [Fullscreen, CameraMatrices, EngineGlobals]
    Resources { Texture2D sourceTex }
    Inputs { Float2 inUV } Outputs { Float4 outColor }
}
void main() {
    bool ownView = all(lessThan(abs(_Globals._ProjectionParams - ubo.projectionParams), vec4(0.0001)))
        && all(lessThan(abs(_Globals._WorldSpaceCameraPos.xyz - inverse(ubo.view)[3].xyz), vec3(0.0001)))
        && all(lessThan(abs(_Globals._ScreenParams.xy - vec2(textureSize(sourceTex, 0))), vec2(0.01)));
    outColor = ownView ? texture(sourceTex, inUV) : vec4(1, 0, 1, 1);
}
'''

MONITOR = '''#version 450
ShaderInfo {
    Name "Tutorial Shared View Monitor" Hidden On Capabilities [Fullscreen, EngineGlobals]
    Resources { Texture2D leftTex Texture2D rightTex }
    Inputs { Float2 inUV } Outputs { Float4 outColor }
}
void main() {
    bool ownView = all(lessThan(abs(_Globals._ProjectionParams.xy - vec2(0.7, 900)), vec2(0.0001)))
        && all(lessThan(abs(_Globals._WorldSpaceCameraPos.xyz - vec3(0, 0, -14)), vec3(0.0001)))
        && all(lessThan(abs(_Globals._ScreenParams.xy - vec2(160, 120)), vec2(0.01)));
    outColor = !ownView ? vec4(1, 0, 1, 1)
        : inUV.x < 0.5 ? texture(leftTex, vec2(inUV.x * 2, inUV.y))
                       : texture(rightTex, vec2(inUV.x * 2 - 1, inUV.y));
}
'''


class ProbePipeline(RenderPipeline):
    name = 'View Globals Probe'

    def define_topology(self, graph):
        graph.set_msaa_samples(1)
        source = graph.create_texture('source', format=Format.RGBA16_SFLOAT)
        depth = graph.create_texture('depth', format=Format.D32_SFLOAT)
        color = graph.create_texture('color', camera_target=True)
        graph.add_pass('Geometry').write_color(source).write_depth(depth).set_clear(
            color=(0, 0, 0, 1), depth=0).draw_renderers(material_pass='forward')
        graph.add_pass('Check own view').write_color(color).set_texture(
            'sourceTex', source).fullscreen_quad('View Globals Post')
        graph.screen_ui_section(resources={'color'})
        graph.set_output(color)


def main():
    with tempfile.TemporaryDirectory(prefix='infernux-view-globals-') as folder:
        project = Path(folder)
        for directory in ('Assets', 'Packages', 'ProjectSettings'):
            (project / directory).mkdir()
        for name, source in (('Probe.frag', SURFACE), ('Post.frag', POST), ('Monitor.frag', MONITOR)):
            (project / 'Assets' / name).write_text(source, encoding='ascii')

        engine = inx.Engine()
        native = engine.get_native_engine()
        console = ConsolePanel()
        router = Router()
        router.shared = ProbePipeline()
        router.monitor = Monitor()
        router.monitor.builds = 0
        errors = []
        complete = False
        try:
            engine.init_renderer(160, 120, folder)
            engine.resize_scene_render_target(110, 70)
            engine.resize_game_render_target(160, 120)
            native.set_editor_fps_cap(240)
            native.set_editor_idle_fps(0)
            native.set_scene_view_visible(True)
            native.set_show_grid(False)
            editor = native.editor_camera
            editor.restore_state(0, 0, -12, 0, 0, 0, 12, 0, 0)
            editor.near_clip, editor.far_clip = .3, 700
            database = engine.get_asset_database()
            for name in ('Probe.frag', 'Post.frag', 'Monitor.frag'):
                result = AssetManager.import_asset(str(project / 'Assets' / name), database=database)
                assert result, result.error
            scene = SceneManager.instance().get_active_scene()
            for owner in scene.get_root_objects():
                scene.destroy_game_object(owner)
            surface = scene.create_primitive(inx.PrimitiveType.Quad)
            surface.transform.local_scale = Vector3(100, 100, 1)
            material = inx.Material.create_unlit()
            material.frag_shader_name = 'View Globals Probe'
            surface.get_component(inx.MeshRenderer).set_material(0, material)

            cameras, targets = [], []
            for index, (width, height, near, far, z) in enumerate(((53, 29, .1, 100, -5), (97, 61, .5, 250, -8))):
                path = project / f'Assets/View{index}.rendertexture'
                path.write_text(json.dumps({'$type': 'render_texture', 'size': {'width': width, 'height': height},
                                            'format': 'rgba16_sfloat', 'depth_format': 'd32_sfloat',
                                            'samples': 1, 'filter': 'nearest', 'storage': False,
                                            'sampled_depth': False}), encoding='utf-8')
                result = AssetManager.import_asset(str(path), database=database)
                assert result, result.error
                target = inx.RenderTexture.load_by_guid(result.guid)
                targets.append(target)
                owner = scene.create_game_object(f'Camera {index}')
                owner.transform.position = Vector3(0, 0, z)
                camera = owner.add_component(inx.Camera)
                camera.set_clip_planes(near, far)
                camera.target_texture = target
                camera.depth = index
                cameras.append(camera)
            owner = scene.create_game_object('Monitor')
            owner.transform.position = Vector3(0, 0, -14)
            monitor = owner.add_component(inx.Camera)
            monitor.set_clip_planes(.7, 900)
            monitor.culling_mask = 0
            monitor.depth = 5
            scene.main_camera = monitor
            router.monitor_id, router.monitor.targets = monitor.component_id, targets
            engine.set_render_pipeline(router)
            native.set_game_camera_enabled(True)

            phase = frame = changed = 0
            tickets = None
            initial_images = None

            def after_draw():
                nonlocal phase, frame, changed, tickets, complete, initial_images
                try:
                    frame += 1
                    if tickets is None and frame >= changed + 8:
                        tickets = [engine.request_render_target_readback(False), engine.request_render_target_readback(True)]
                    elif tickets is not None and all(ticket.done for ticket in tickets):
                        images = [ticket.result_numpy().copy()[..., :3].astype(float) for ticket in tickets]
                        if phase == 0:
                            initial_images = images
                        elif phase in (4, 5, 6):
                            np.testing.assert_array_equal(images[0], initial_images[0])
                            if phase == 5:
                                assert not np.array_equal(images[1], initial_images[1])
                            else:
                                np.testing.assert_array_equal(images[1], initial_images[1])
                        # RenderTexture outputs stay linear; the display outputs are encoded once.
                        images = [np.where(p <= .04045, p / 12.92, ((p + .055) / 1.055) ** 2.4) for p in images]
                        actual = [np.median(images[0].reshape(-1, 3), axis=0),
                                  np.median(images[1][:, :80].reshape(-1, 3), axis=0),
                                  np.median(images[1][:, 80:].reshape(-1, 3), axis=0)]
                        if phase == 1:
                            expected = [(110/200, 70/200, 20/110), (53/200, 29/200, 20/53), (97/200, 61/200, 20/97)]
                        elif phase == 3:
                            expected = [(.6, .9, .75), (.2, .3, .3), (.5, .25, .4)]
                        else:
                            expected = [(.3, .7, .6), (.1, .1, .25), (.5, .25, .4)]
                        print('View globals', phase, [p.tolist() for p in actual], flush=True)
                        np.testing.assert_allclose(actual, expected, atol=.004)
                        assert not [entry for entry in console._get_visible_log_snapshot(2000)
                                    if entry['level'] in ('ERROR', 'FATAL', 'WARN', 'WARNING')]
                        if phase == 6:
                            complete = True
                            native.exit()
                            return
                        phase += 1
                        material.set_float('mode', phase if phase < 3 else 0)
                        if phase == 3:
                            editor.near_clip, editor.far_clip = .6, 900
                            editor.restore_state(0, 0, -15, 0, 0, 0, 15, 0, 0)
                            cameras[0].set_clip_planes(.2, 300)
                            # Shader position must follow the rendered view, not the object's transform.
                            matrix = cameras[0].view_matrix.copy()
                            matrix[2, 3] = 6
                            cameras[0].view_matrix = matrix
                            cameras[0].depth = 4
                            assert targets[0].resize(79, 47)
                        elif phase == 4:
                            editor.near_clip, editor.far_clip = .3, 700
                            editor.restore_state(0, 0, -12, 0, 0, 0, 12, 0, 0)
                            cameras[0].set_clip_planes(.1, 100)
                            cameras[0].reset_view_matrix()
                            cameras[0].depth = 0
                            assert targets[0].resize(53, 29)
                        elif phase == 5:
                            monitor.dithering = True
                        elif phase == 6:
                            monitor.dithering = False
                        changed, tickets = frame, None
                    if frame > 180:
                        raise AssertionError('View globals audit timed out')
                except BaseException as error:
                    errors.append(error)
                    native.exit()

            native.set_post_draw_callback(after_draw)
            native.set_play_mode_rendering(True)
            native.run()
            if errors:
                raise errors[0]
            assert complete
        finally:
            engine.set_render_pipeline(None)
            router.dispose()
            native.cleanup()
    print('PASS independent Scene/Game/texture shader globals and fullscreen matrices', flush=True)


if __name__ == '__main__':
    main()
