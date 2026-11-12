"""An authored per-light response shares all light traversal in F and F+."""

import argparse
import math
from pathlib import Path
import tempfile

import numpy as np
import infernux as inx
from infernux.lib import ConsolePanel, LightShadows, LightType, SceneManager, Vector3


MODEL = '''import infernux.shader as shader
class IncidentResponse(shader.shadingmodel):
    lighting_contract = "custom_response"
    def shade_light(self, surface: shader.SurfaceData, ctx: shader.ShadingContext, light: shader.LightData) -> shader.vec3:
        return light.color * light.attenuation * light.shadow
    def shade(self, surface: shader.SurfaceData, ctx: shader.ShadingContext) -> shader.vec3:
        return shader.lighting.direct(surface, ctx)
'''
SURFACE = '''import infernux.shader as shader
class Incident(shader.surface):
    shading_model = shader.ref("./incident.shadingmodel.py")
    def surface(self, ctx: shader.SurfaceContext) -> shader.SurfaceData:
        return shader.SurfaceData(normal_ws=ctx.normal_ws)
'''


class ResponsePipeline(inx.renderstack.RenderPipeline):
    name = "Shared Light Response"

    def define_topology(self, graph):
        self.builds += 1
        graph.set_msaa_samples(1)
        color = graph.create_texture("radiance", camera_target=True)
        depth = graph.create_texture("depth", format=inx.rendergraph.Format.D32_SFLOAT, samples=1)
        graph.add_pass("Incident lights").write_color(color).write_depth(depth).set_clear(
            color=(0, 0, 0, 0), depth=0.0).draw_renderers(material_pass=self.material_pass)
        graph.set_output(color)


def attenuation(distance, radius=10.0):
    return max(1.0 - (distance / radius) ** 4, 0.0) ** 2 / (distance * distance + 1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pipeline', choices=('forward', 'forward_plus'), default='forward')
    mode = parser.parse_args().pipeline
    with tempfile.TemporaryDirectory(prefix='infernux-light-response-') as directory:
        root = Path(directory)
        (root / 'Assets').mkdir()
        (root / 'ProjectSettings').mkdir()
        model = root / 'Assets/incident.shadingmodel.py'
        surface = root / 'Assets/incident.frag.py'
        model.write_text(MODEL, encoding='utf8')
        surface.write_text(SURFACE, encoding='utf8')
        frontend, console, pipeline = inx.Engine(), ConsolePanel(), ResponsePipeline()
        pipeline.material_pass, pipeline.builds = mode, 0
        native = frontend.get_native_engine()
        errors, complete = [], False
        try:
            frontend.init_renderer(160, 120, directory)
            frontend.resize_game_render_target(160, 120)
            native.set_scene_view_visible(False)
            native.set_editor_fps_cap(240)
            native.set_editor_idle_fps(0)
            scene = SceneManager.instance().get_active_scene()
            for owner in scene.get_root_objects():
                scene.destroy_game_object(owner)
            camera = scene.create_game_object('相机')
            camera.transform.position = Vector3(0, 0, -5)
            camera.add_component(inx.Camera).field_of_view = 60
            owner = scene.create_game_object('光照响应平面')
            renderer = owner.add_component(inx.MeshRenderer)
            renderer.set_inline_mesh_data(
                np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], np.float32),
                np.array([[0, 0, -1]] * 4, np.float32),
                np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32),
                np.array([0, 2, 1, 0, 3, 2], np.uint32), 'Response Quad')
            renderer.set_material(inx.Material.from_shader(str(surface)))
            lights = []
            for name, kind, color in (('方向光', LightType.Directional, (1, 0, 0)),
                                      ('点光', LightType.Point, (0, 1, 0)),
                                      ('聚光', LightType.Spot, (0, 0, 1)),
                                      ('面光', LightType.Area, (1, 1, 0))):
                source = scene.create_game_object(name)
                source.transform.position = Vector3(0, 0, -3)
                light = source.add_component(inx.Light)
                light.light_type, light.color, light.intensity = kind, color, 0.0
                light.range, light.shadows = 10.0, LightShadows.NoShadows
                if kind == LightType.Area:
                    light.area_size = (1.6, 1.0)
                lights.append(light)
            frontend.set_render_pipeline(pipeline)
            native.set_game_camera_enabled(True)
            # The observed center pixel is half a pixel from the optical axis.
            xy = (math.tan(math.pi / 6) * 5 / 120,) * 2
            point = attenuation(math.sqrt(sum(value * value for value in xy) + 9.0))
            facing = 3 / math.sqrt(sum(value * value for value in xy) + 9.0)
            area = sum(attenuation(math.sqrt((xy[0] - x) ** 2 + (xy[1] - y) ** 2 + 9.0))
                       for x in (-.8, .8) for y in (-.5, .5)) * .25 * facing
            contributions = np.array([[.2, 0, 0], [0, 2 * point, 0], [0, 0, 3 * point],
                                      [4 * area, 4 * area, 0]])
            enabled = ((0, 0, 0, 0), (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0),
                       (0, 0, 0, 1), (1, 1, 1, 1), (1, 0, 0, 1), (1, 1, 1, 1), (1, 1, 1, 1))
            phase = frame = changed = 0
            ticket = None

            def after_draw():
                nonlocal phase, frame, changed, ticket, complete
                try:
                    frame += 1
                    if ticket is None and frame >= changed + 8:
                        ticket = frontend.request_render_target_readback(True)
                    elif ticket is not None and ticket.done:
                        pixels = ticket.result_numpy().astype(np.float32)
                        actual = pixels[60, 80, :3]
                        expected = contributions[np.array(enabled[phase], dtype=bool)].sum(axis=0)
                        if phase == 8:
                            expected *= .5
                        np.testing.assert_allclose(actual, expected, atol=.001, rtol=.003)
                        assert pixels[60, 80, 3] > .99
                        assert pipeline.builds == 1
                        assert not [row for row in console._get_visible_log_snapshot(2000)
                                    if row['level'] in ('ERROR', 'FATAL')]
                        print('LIGHT_RESPONSE', mode, phase, 'actual', actual.tolist(), 'expected', expected.tolist(), flush=True)
                        phase += 1
                        if phase == len(enabled):
                            complete = True
                            native.exit()
                            return
                        for index, light in enumerate(lights):
                            light.intensity = (.2, 2., 3., 4.)[index] if enabled[phase][index] else 0.
                            light.culling_mask = 0xFFFFFFFF
                        if phase == 6:
                            for index in (1, 2):
                                lights[index].intensity = (2., 3.)[index - 1]
                                lights[index].culling_mask = 0
                        if phase == 8:
                            model.write_text(MODEL.replace('return light.color', 'return 0.5 * light.color'), encoding='utf8')
                            assert inx.Shader.reload(str(model))
                        changed, ticket = frame, None
                    if frame > 350:
                        raise AssertionError('Shared response GPU timeout')
                except BaseException as error:
                    errors.append(error)
                    native.exit()

            native.set_post_draw_callback(after_draw)
            native.set_play_mode_rendering(True)
            native.run()
            if errors:
                raise errors[0]
            assert complete
            print('SLANG_LIGHT_RESPONSE_OK', mode, flush=True)
        finally:
            frontend.set_render_pipeline(None)
            pipeline.dispose()
            native.cleanup()


if __name__ == '__main__':
    main()
