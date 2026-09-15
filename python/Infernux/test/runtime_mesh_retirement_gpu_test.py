"""Real Vulkan lifetime test for generation-owned changing inline meshes."""
from pathlib import Path
import tempfile
import argparse

import numpy as np
from Infernux import Engine
from Infernux.lib import PrimitiveType, SceneManager, Vector3


def main(multiple_cameras=False):
    with tempfile.TemporaryDirectory(prefix="infernux-inline-retirement-") as root:
        project = Path(root)
        (project / "Assets").mkdir()
        (project / "ProjectSettings").mkdir()
        frontend = Engine()
        engine = frontend.get_native_engine()
        frames, settled = 0, False
        baseline_count, baseline_bytes = 0, 0
        failures = []
        try:
            frontend.init_renderer(64, 64, str(project))
            engine.set_editor_fps_cap(240.0)
            engine.set_editor_idle_fps(0.0)
            scene = SceneManager.instance().get_active_scene()
            if multiple_cameras:
                for index in range(2):
                    owner = scene.create_game_object(f"game-camera-{index}")
                    camera = owner.add_component("Camera")
                    camera.depth = float(index)
                    owner.transform.position = Vector3(0.0, 2.0, 5.0)
                engine.set_game_camera_enabled(True)
                engine.resize_game_render_target(64, 64)
            objects = [scene.create_primitive(PrimitiveType.Cube, name) for name in ("inline-a", "inline-b")]
            renderers = [obj.get_component("MeshRenderer") for obj in objects]
            positions = np.array([[-1, 0, -1], [-1, 0, 1], [1, 0, -1]], dtype=np.float32)
            normals = np.tile(np.array([[0, 1, 0]], dtype=np.float32), (3, 1))
            uvs = np.zeros((3, 2), dtype=np.float32)
            indices = np.array([0, 1, 2], dtype=np.uint32)
            initial_evictions = engine.mesh_gpu_eviction_count

            def after_draw():
                nonlocal frames, settled, baseline_count, baseline_bytes
                try:
                    frames += 1
                    if frames == 10:
                        baseline = engine.gpu_residency_snapshot
                        baseline_count = baseline["runtime_mesh_entry_count"]
                        baseline_bytes = baseline["runtime_mesh_bytes"]
                    if 10 < frames <= 202:
                        positions[:, 1] = frames * 0.001
                        for renderer in renderers:
                            renderer.set_inline_mesh_data(positions, normals, uvs, indices, "changing triangle")
                    if frames in (220, 240):
                        if multiple_cameras:
                            frame = engine.renderer_frame_snapshot
                            assert frame["game_camera_count"] == 2, frame
                            assert frame["game_render_graph_current_executed"], frame
                            assert len(frame["game_render_view_ids"]) == 2, frame
                            assert frame["game_draw_call_count"] > 0, frame
                        snapshot = engine.gpu_residency_snapshot
                        expected_entries = baseline_count + (2 if frames == 220 else 1)
                        assert snapshot["runtime_mesh_entry_count"] == expected_entries, snapshot
                        assert snapshot["runtime_mesh_bytes"] > baseline_bytes
                        scene.destroy_game_object(objects[0 if frames == 220 else 1])
                    if frames == 260:
                        snapshot = engine.gpu_residency_snapshot
                        assert snapshot["runtime_mesh_entry_count"] == baseline_count, snapshot
                        assert snapshot["runtime_mesh_bytes"] == baseline_bytes
                        assert engine.pending_mesh_gpu_upload_count == 0
                        assert engine.mesh_gpu_eviction_count == initial_evictions
                        settled = True
                        print("runtime mesh retirement: object generations retired; final residency at baseline")
                        engine.exit()
                except BaseException as exc:
                    failures.append(exc)
                    engine.exit()

            engine.set_post_draw_callback(after_draw)
            engine.set_play_mode_rendering(True)
            engine.run()
            if failures:
                raise failures[0]
            assert settled, "Render loop ended before lifetime assertions"
        finally:
            engine.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--multiple-cameras", action="store_true")
    main(multiple_cameras=parser.parse_args().multiple_cameras)
