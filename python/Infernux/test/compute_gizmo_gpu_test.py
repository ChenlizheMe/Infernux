"""Real Vulkan regression for GPU-resident component Gizmo line geometry."""
from pathlib import Path
import tempfile

import numpy as np

import Infernux as inx
from Infernux import Engine
from Infernux.gizmos import Gizmos


@inx.compute.kernel
def move_points(domain, positions, amount):
    i = inx.compute.index(domain)
    positions[i, 1] = positions[i, 1] + amount


class ResidentGizmoProbe(inx.InxComponent):
    def on_draw_gizmos(self):
        Gizmos.matrix = self._matrix
        Gizmos.draw_lines(self._positions, self._edges)
        Gizmos.draw_wire_spheres(self._positions, 0.05, segments=8)


def main():
    with tempfile.TemporaryDirectory(prefix="infernux-compute-gizmo-") as root:
        project = Path(root)
        (project / "Assets").mkdir()
        (project / "ProjectSettings").mkdir()
        frontend = Engine()
        engine = frontend.get_native_engine()
        positions = domain = None
        failures = []
        frames = 0
        try:
            frontend.init_renderer(64, 64, str(project))
            engine.resize_scene_render_target(64, 64)
            engine.set_editor_fps_cap(240.0)
            engine.set_editor_idle_fps(0.0)
            initial = np.array(
                [[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.75, 0.0]],
                dtype=np.float32,
            )
            positions = inx.buffer(shape=initial.shape, dtype=np.float32, device="gpu", data=initial)
            domain = inx.buffer(shape=len(initial), dtype=np.int32, device="gpu")
            edges = np.array([[0, 1], [1, 2], [2, 0]], dtype=np.uint32)

            with inx.compute._record_commands():
                Gizmos._begin_frame()
                inx.compute.launch(move_points, params=(domain, positions, 0.25))
                Gizmos.draw_lines(positions, edges)
                Gizmos.draw_wire_spheres(positions, 0.05, segments=8)
                assert len(positions._gizmo_wire_sphere_states) == 1

            line_state = positions._gizmo_line_states[id(edges)]
            line_vertices = line_state.vertices.get_data().numpy(copy=False)
            np.testing.assert_allclose(
                line_vertices[:, :3], initial + (0.0, 0.25, 0.0), atol=1.0e-5,
            )

            translated = list(Gizmos._identity_matrix)
            translated[12:15] = [2.0, 3.0, 4.0]
            with inx.compute._record_commands():
                inx.compute.launch(move_points, params=(domain, positions, 0.5))
                Gizmos._begin_frame()
                Gizmos.matrix = translated
                Gizmos.draw_lines(positions, edges)
                Gizmos.draw_wire_spheres(positions, 0.05, segments=8)
                assert len(positions._gizmo_wire_sphere_states) == 1
                engine.clear_component_cpu_gizmos()
                engine.upload_component_resident_gizmos(Gizmos._get_resident_data())

            line_vertices = line_state.vertices.get_data().numpy(copy=False)
            np.testing.assert_allclose(
                line_vertices[:, :3], initial + (0.0, 0.75, 0.0), atol=1.0e-5,
            )
            descriptors = Gizmos._get_resident_data()
            assert descriptors
            assert descriptors[0][4] == translated
            del descriptors, line_vertices, line_state

            # The render-extraction barrier collects components and replaces
            # each frame's Gizmos. A one-off upload before run() is not a
            # component callback and must not survive that collection.
            probe = ResidentGizmoProbe()
            probe._positions, probe._edges, probe._matrix = positions, edges, translated
            scene = inx.SceneManager.get_active_scene()
            scene.create_game_object('Resident Gizmo Probe').add_py_component(probe)

            def after_draw():
                nonlocal frames
                try:
                    frames += 1
                    if frames == 64:
                        assert engine.resident_mesh_vertex_buffer_count >= 2
                        values = positions.get_data().numpy(copy=False)
                        np.testing.assert_allclose(values[:, 1], initial[:, 1] + 0.75, atol=1.0e-5)
                        engine.exit()
                except BaseException as exc:
                    failures.append(exc)
                    engine.exit()

            engine.set_post_draw_callback(after_draw)
            engine.set_play_mode_rendering(True)
            engine.run()
            if failures:
                raise failures[0]
            assert frames >= 64
        finally:
            if positions is not None:
                positions.close()
            if domain is not None:
                domain.close()
            inx.compute._release_engine_resources()
            engine.cleanup()
    print("INFERNUX_COMPUTE_RESIDENT_GIZMO_OK")


if __name__ == "__main__":
    main()
