"""Real Vulkan regression for a compute-resident MeshRenderer vertex stream."""
from pathlib import Path
import tempfile

import numpy as np

import Infernux as inx
from Infernux import Engine
from Infernux.lib import PrimitiveType, SceneManager, Vector3


@inx.compute.kernel
def deform(domain, vertices, delta):
    i = inx.compute.index(domain)
    # A shear changes the surface direction. A uniform translation would leave
    # normals unchanged and could not exercise automatic normal rebuilding.
    vertices[i, 1] = vertices[i, 1] + delta * vertices[i, 0]


def main():
    with tempfile.TemporaryDirectory(prefix="infernux-compute-mesh-") as root:
        project = Path(root)
        (project / "Assets").mkdir()
        (project / "ProjectSettings").mkdir()
        frontend = Engine()
        engine = frontend.get_native_engine()
        failures = []
        frames = 0
        vertex_buffer = None
        seam_buffer = None
        scaled_buffer = None
        scaled_renderer = None
        domain = None
        scaled_domain = None
        unrelated_domain = None
        unrelated_vertices = None
        initial_normals = None
        initial_tangents = None
        try:
            frontend.init_renderer(64, 64, str(project))
            engine.set_editor_fps_cap(240.0)
            engine.set_editor_idle_fps(0.0)
            scene = SceneManager.instance().get_active_scene()
            renderer = scene.create_primitive(PrimitiveType.Sphere, "resident sphere").get_component("MeshRenderer")
            active_vertex_count = renderer.vertex_count
            vertex_buffer = renderer.create_vertex_buffer(
                device="gpu", capacity=active_vertex_count + 64)
            assert vertex_buffer.shape == (active_vertex_count + 64, 23)
            domain = inx.buffer(shape=renderer.vertex_count, dtype=np.int32, device="gpu")
            unrelated_domain = inx.buffer(shape=8, dtype=np.int32, device="gpu")
            unrelated_vertices = inx.buffer(shape=(8, 23), dtype=np.float32, device="gpu")
            initial_vertices = vertex_buffer.get_data().numpy(copy=False)
            initial_normals = initial_vertices[:active_vertex_count, 3:6].copy()
            initial_tangents = initial_vertices[:active_vertex_count, 6:10].copy()
            renderer.set_vertex_buffer(
                vertex_buffer,
                (-2.0, -2.0, -2.0),
                (2.0, 2.0, 2.0),
                space="world",
            )
            assert renderer.vertex_buffer_capacity == active_vertex_count + 64

            # A world-space resident stream bypasses the object's draw matrix,
            # but its conservative bounds remain attached to the Transform
            # pose at binding time. Editor Rect/selection tools and camera
            # culling must therefore follow later authoring changes without a
            # GPU readback.
            np.testing.assert_allclose(
                renderer.get_world_bounds(),
                (-2.0, -2.0, -2.0, 2.0, 2.0, 2.0),
                atol=1.0e-5,
            )
            renderer.game_object.transform.position = Vector3(3.0, 4.0, 5.0)
            renderer.game_object.transform.local_scale = Vector3(2.0, 0.5, 1.5)
            np.testing.assert_allclose(
                renderer.get_world_bounds(),
                (-1.0, 3.0, 2.0, 7.0, 5.0, 8.0),
                atol=1.0e-5,
            )
            renderer.game_object.transform.position = Vector3(0.0, 0.0, 0.0)
            renderer.game_object.transform.local_scale = Vector3(1.0, 1.0, 1.0)

            # Equal positions do not imply shared topology. This folded pair
            # models a UV/hard-edge split and must keep two independent normals.
            seam_renderer = scene.create_game_object("split seam").add_component("MeshRenderer")
            seam_positions = np.array([
                [0, 0, 0], [1, 0, 0], [0, 1, 0],
                [0, 0, 0], [0, 1, 0], [0, 0, 1],
            ], dtype=np.float32)
            seam_uvs = np.array([
                [0, 0], [1, 0], [0, 1],
                [0, 0], [1, 0], [0, 1],
            ], dtype=np.float32)
            seam_indices = np.arange(6, dtype=np.uint32)
            seam_renderer.set_inline_mesh_data(
                seam_positions, None, seam_uvs, seam_indices, "split seam")
            seam_buffer = seam_renderer.create_vertex_buffer(device="gpu")
            seam_domain = inx.buffer(shape=6, dtype=np.int32, device="gpu")
            inx.compute.launch(deform, params=(seam_domain, seam_buffer, 0.0))
            seam_values = seam_buffer.get_data().numpy(copy=False)
            np.testing.assert_allclose(seam_values[0, 3:6], [0, 0, 1], atol=1.0e-5)
            np.testing.assert_allclose(seam_values[3, 3:6], [1, 0, 0], atol=1.0e-5)
            seam_domain.close()

            # Deliberately preserve scaled authored normals while asking the
            # engine to derive only tangents.  The tangent builder must use a
            # unit normal frame without rewriting the authored stream.
            scaled_renderer = scene.create_game_object("scaled authored normals").add_component("MeshRenderer")
            scaled_normals = np.array(
                [[0, 0, 4], [0, 0, 4], [0, 0, 4], [0, 0, 0]], dtype=np.float32)
            scaled_renderer.set_inline_mesh_data(
                np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [2, 2, 2]], dtype=np.float32),
                scaled_normals,
                np.array([[0, 0], [1, 0], [0, 1], [0, 0]], dtype=np.float32),
                np.array([0, 1, 2], dtype=np.uint32),
                "scaled authored normals",
            )
            scaled_buffer = scaled_renderer.create_vertex_buffer(
                device="gpu", auto_normals=False, auto_tangents=True)
            scaled_domain = inx.buffer(shape=4, dtype=np.int32, device="gpu")
            inx.compute.launch(deform, params=(scaled_domain, scaled_buffer, 0.0))
            scaled_values = scaled_buffer.get_data().numpy(copy=False)
            np.testing.assert_allclose(
                scaled_values[:, 3:6], scaled_normals, atol=1.0e-5)
            np.testing.assert_allclose(
                np.linalg.norm(scaled_values[:, 6:9], axis=1), 1.0, atol=1.0e-5)
            np.testing.assert_allclose(
                scaled_values[:3, 6:9] @ np.array([0, 0, 1], dtype=np.float32),
                0.0, atol=1.0e-5,
            )
            np.testing.assert_array_equal(scaled_values[3, 6:10], [1, 0, 0, 1])

            def after_draw():
                nonlocal frames
                try:
                    frames += 1
                    if 32 <= frames < 96:
                        inx.compute.launch(deform, params=(domain, vertex_buffer, 0.005))
                        inx.compute._flush_commands()
                        # Submit unrelated work afterwards. Graphics must wait
                        # for the resident mesh's exact write, not the newest
                        # submission merely because it shares the compute lane.
                        inx.compute.launch(
                            deform, params=(unrelated_domain, unrelated_vertices, 0.0))
                        inx.compute._flush_commands()
                    if frames == 48:
                        frame = engine.renderer_frame_snapshot
                        resident_serial = frame["submission_resident_compute_write_serial"]
                        latest_serial = frame["submission_latest_background_compute_serial"]
                        assert resident_serial > 0, frame
                        assert latest_serial > resident_serial, frame
                    if frames == 128:
                        assert engine.resident_mesh_vertex_buffer_count >= 1
                        values = vertex_buffer.get_data().numpy(copy=False)
                        normals = values[:active_vertex_count, 3:6]
                        tangents = values[:active_vertex_count, 6:10]
                        lengths = np.linalg.norm(normals, axis=1)
                        assert float(np.max(np.abs(normals - initial_normals))) > 1.0e-3
                        assert float(lengths.min()) > 0.99
                        assert float(lengths.max()) < 1.01
                        assert float(np.max(np.abs(tangents - initial_tangents))) > 1.0e-3
                        np.testing.assert_allclose(
                            np.linalg.norm(tangents[:, :3], axis=1), 1.0, atol=1.0e-3)
                        np.testing.assert_allclose(
                            np.sum(normals * tangents[:, :3], axis=1), 0.0, atol=1.0e-3)
                        np.testing.assert_allclose(np.abs(tangents[:, 3]), 1.0, atol=0.0)
                        np.testing.assert_array_equal(
                            values[active_vertex_count:],
                            np.zeros((64, 23), dtype=np.float32),
                        )
                        # MeshRenderer borrows the native allocation through
                        # the engine's shared RHI ownership.  Closing the
                        # author-facing wrapper must not invalidate rendering.
                        vertex_buffer.close()
                        assert renderer.vertex_buffer_capacity == active_vertex_count + 64
                    if frames == 136:
                        assert engine.resident_mesh_vertex_buffer_count >= 1
                        frame = engine.renderer_frame_snapshot
                        assert frame["scene_draw_call_count"] > 0, frame
                        renderer.clear_vertex_buffer()
                        assert renderer.vertex_buffer_capacity == 0
                        engine.exit()
                except BaseException as exc:
                    failures.append(exc)
                    engine.exit()

            engine.set_post_draw_callback(after_draw)
            engine.set_play_mode_rendering(True)
            engine.run()
            if failures:
                raise failures[0]
            assert frames >= 128
        finally:
            if vertex_buffer is not None:
                renderer.clear_vertex_buffer()
            if seam_buffer is not None:
                seam_buffer.close()
            if scaled_renderer is not None:
                scaled_renderer.clear_vertex_buffer()
            if scaled_domain is not None:
                scaled_domain.close()
            if scaled_buffer is not None:
                scaled_buffer.close()
            if unrelated_domain is not None:
                unrelated_domain.close()
            if unrelated_vertices is not None:
                unrelated_vertices.close()
            inx.compute._release_engine_resources()
            engine.cleanup()
    print("INFERNUX_COMPUTE_RESIDENT_MESH_OK")


if __name__ == "__main__":
    main()
