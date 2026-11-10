"""Real Vulkan regression for GPU-resident component Gizmo line geometry."""
from pathlib import Path
import tempfile
from time import monotonic

import numpy as np

import infernux as inx
from infernux import Engine
from infernux.engine.cpu_continuation import CpuWorkBudget
from infernux.gizmos import Gizmos
from infernux.lib import _Infernux
from infernux.renderstack import DefaultForwardPipeline


@inx.compute.kernel
def move_points(domain, positions, amount):
    i = inx.compute.index(domain)
    positions[i, 1] = positions[i, 1] + amount


class ResidentGizmoProbe(inx.InxComponent):
    def on_draw_gizmos(self):
        Gizmos.matrix = self._matrix
        Gizmos.draw_line((0, 0, 0), (1, 0, 0))
        Gizmos.matrix = list(self._matrix)
        Gizmos.matrix[12] += .5
        Gizmos.draw_line((0, 0, 0), (0, 1, 0))
        Gizmos.matrix = self._matrix
        Gizmos.color = (1, 0, 0)
        Gizmos.draw_lines(self._positions, self._edges)
        Gizmos.draw_wire_spheres(self._positions, 0.05, segments=8)
        Gizmos.color = (0, 0, 1)
        Gizmos.draw_lines(self._positions, self._edges)
        Gizmos.draw_wire_spheres(self._positions, 0.1, segments=8)


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
        verified = False
        try:
            frontend.init_renderer(64, 64, str(project))
            engine.resize_scene_render_target(64, 64)
            engine.set_scene_view_visible(True)
            engine.set_show_grid(False)
            engine.set_play_fps_cap(30.0)
            frontend.set_render_pipeline(DefaultForwardPipeline())
            # Leave host preparation and device time for both expansion and topology.
            frontend._cpu_work(CpuWorkBudget(owner_ns=10_000_000, gpu_frame_ns=20_000_000))
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
                assert len(positions._gizmo_wire_sphere_pool.states) == 1

            line_state = positions._gizmo_line_pool.states[0]
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
                assert len(positions._gizmo_wire_sphere_pool.states) == 1
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
            for sample_object in scene.get_root_objects():
                scene.destroy_game_object(sample_object)
            scene.create_game_object('Resident Gizmo Probe').add_py_component(probe)
            deadline = monotonic() + 40.0

            def after_frame(_foreground_ready):
                nonlocal frames, verified
                try:
                    frames += 1
                    resident_count = engine.resident_mesh_vertex_buffer_count
                    if frames >= 64 and resident_count >= 4:
                        assert engine.renderer_frame_snapshot['scene_render_graph_execution_count'] > 0
                        assert len(positions._gizmo_line_pool.states) == 2
                        assert len(positions._gizmo_wire_sphere_pool.states) == 2
                        values = positions.get_data().numpy(copy=False)
                        np.testing.assert_allclose(values[:, 1], initial[:, 1] + 0.75, atol=1.0e-5)
                        verified = True
                        engine.exit()
                    elif monotonic() >= deadline:
                        owner = engine.frame_work
                        scene_frames = engine.renderer_frame_snapshot['scene_render_graph_execution_count']
                        raise AssertionError(
                            f"Gizmo publication timed out after {frames} host frames, "
                            f"{engine.submitted_render_frames} submissions, and {scene_frames} Scene executions. "
                            f"Descriptors={len(Gizmos._get_resident_data())}, resident={resident_count}, "
                            f"mesh pending/submitted/completed={engine.pending_mesh_gpu_upload_count}/"
                            f"{engine.submitted_mesh_gpu_upload_count}/{engine.completed_mesh_gpu_upload_count}, "
                            f"remaining host ns={owner.remaining(True)}, "
                            f"remaining GPU ns={owner.resource_remaining(_Infernux._FrameWorkResource.GPU)}, "
                            f"GPU queue gate={owner.gpu_unavailable_reason!r}"
                        )
                except BaseException as exc:
                    failures.append(exc)
                    engine.exit()

            engine.set_end_frame_callback(after_frame)
            engine.set_play_mode_rendering(True)
            engine.run()
            if failures:
                raise failures[0]
            assert verified
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
