"""Consume the actual native World ray page through a fullscreen graph input."""

from argparse import ArgumentParser
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic

import numpy as np

import infernux as inx
from infernux.core.assets import AssetManager
from infernux.lib import SceneManager, Vector3
from infernux.renderstack import RenderPipeline


SHADER = '''import infernux.shader as shader

class WorldRayPage(shader.fullscreen):
    shader_id = "tests/world_ray_geometry"
    page = shader.input("world_ray_geometry", shader.Buffer[shader.uvec4])

    def main(self, uv: shader.vec2) -> shader.vec4:
        info = self.page[1]
        geometry = self.page[0]
        points = geometry.w - geometry.z
        if info.y != 1 or info.z != info.x + 8 or len(self.page) != info.z + 4 * points:
            return shader.vec4(0.0, 0.0, 1.0, 1.0)
        if self.page[len(self.page) - 1].w != 1065353216:
            return shader.vec4(1.0, 0.0, 0.0, 1.0)
        first = self.page[info.x + 3]
        second = self.page[info.x + 7]
        if first.x == second.x and first.y > 0 and second.y > 0 and first.y != second.y:
            a = self.page[info.x].w & 2147483647
            b = self.page[info.x + 4].w & 2147483647
            position = shader.min(a, b)
            if position == 1090519040:
                return shader.vec4(0.0, 1.0, 1.0, 1.0)
            if position != 0:
                return shader.vec4(1.0, 1.0, 0.0, 1.0)
            return shader.vec4(0.0, 1.0, 0.0, 1.0)
        return shader.vec4(1.0, 0.0, 0.0, 1.0)
'''


class RayGeometryGraph(RenderPipeline):
    name = "World Ray Geometry Probe"

    def define_topology(self, graph):
        graph.set_msaa_samples(1)
        color = graph.create_texture("color", camera_target=True)
        page = graph._create_world_ray_geometry()
        assert graph._create_world_ray_geometry() is page
        with graph.add_pass("ReadWorldRayGeometry") as render_pass:
            render_pass.set_buffer("page", page)
            render_pass.write_color(color)
            render_pass.fullscreen_quad("tests/world_ray_geometry")
        graph.set_output(color)


def main():
    parser = ArgumentParser()
    parser.add_argument("--moving", action="store_true")
    args = parser.parse_args()
    with TemporaryDirectory(prefix="infernux-world-ray-geometry-") as directory:
        project = Path(directory)
        for name in ("Assets", "Packages", "ProjectSettings"):
            (project / name).mkdir()
        shader_path = project / "Assets/world_ray_geometry.frag.py"
        shader_path.write_text(SHADER, encoding="ascii")
        mesh_path = project / "Assets/triangle.obj"
        mesh_path.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="ascii")

        frontend = inx.Engine()
        native = frontend.get_native_engine()
        try:
            frontend.init_renderer(64, 64, directory)
            frontend.resize_game_render_target(64, 64)
            native.set_scene_view_visible(False)
            native.set_editor_idle_fps(0)
            native.set_game_camera_enabled(True)
            database = frontend.get_asset_database()
            shader = AssetManager.import_asset(str(shader_path), database=database)
            assert shader, shader.error
            mesh = AssetManager.import_asset(str(mesh_path), database=database)
            assert mesh, mesh.error

            scene = SceneManager.instance().get_active_scene()
            for obj in scene.get_root_objects():
                scene.destroy_game_object(obj)
            camera = scene.create_game_object("Ray page camera")
            camera.transform.position = Vector3(0, 0, -5)
            camera.add_component("Camera")
            triangles = []
            for position in (0, 100):
                obj = scene.create_game_object(f"Asset triangle {position}")
                obj.transform.position = Vector3(position, 0, 0)
                renderer = obj.add_component("MeshRenderer")
                renderer.set_mesh_asset_guid(mesh.guid)
                triangles.append(obj)

            frontend.set_render_pipeline(RayGeometryGraph())
            deadline = monotonic() + 15
            failure = None
            ticket = None
            last_rgb = None
            passed = False
            stage = "static"
            accepted_moving = 0
            moving_deadline = 0.0
            direction = 1

            def after_draw():
                nonlocal failure, ticket, last_rgb, passed, stage, accepted_moving, moving_deadline, direction
                try:
                    if ticket is None and native.game_presentation_ready:
                        ticket = frontend.request_render_target_readback(True)
                    elif ticket is not None and ticket.done:
                        last_rgb = ticket.result_numpy()[32, 32, :3].copy()
                        ticket = None
                        if stage == "static" and np.allclose(last_rgb, (0, 1, 0), atol=.01):
                            if args.moving:
                                moving_deadline = monotonic() + 0.5
                                stage = "moving"
                            else:
                                passed = True
                        elif stage == "moving" and np.allclose(last_rgb, (1, 1, 0), atol=.01):
                            accepted_moving += 1
                        elif stage == "settled" and np.allclose(last_rgb, (0, 1, 1), atol=.01):
                            passed = True
                    if stage == "moving":
                        if monotonic() >= moving_deadline:
                            triangles[0].transform.position = Vector3(8, 0, 0)
                            stage = "settled"
                        else:
                            triangles[0].transform.position = Vector3(direction, 0, 0)
                            direction = 3 - direction
                    if passed:
                        native.exit()
                    elif monotonic() >= deadline:
                        raise AssertionError(f"World ray geometry did not settle during {stage}, pixel {last_rgb}")
                except BaseException as error:
                    failure = error
                    native.exit()

            native.set_post_draw_callback(after_draw)
            native.run()
            if failure is not None:
                raise failure
            assert passed
            if args.moving:
                assert accepted_moving, "Game never presented a moving frozen World"
            print(f"World ray geometry matched on-screen and off-screen instances, moving samples {accepted_moving}")
        finally:
            frontend.set_render_pipeline(None)
            native.cleanup()


if __name__ == "__main__":
    main()
