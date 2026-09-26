"""Real Vulkan regression for public runtime Mesh publication and destruction."""

from pathlib import Path
import tempfile

import numpy as np

from Infernux import Engine, Mesh
from Infernux.lib import SceneManager


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="infernux-runtime-mesh-resource-") as root:
        project = Path(root)
        (project / "Assets").mkdir()
        (project / "ProjectSettings").mkdir()
        frontend = Engine()
        engine = frontend.get_native_engine()
        frames = 0
        destroyed = False
        settled = False
        failures = []
        try:
            frontend.init_renderer(64, 64, str(project))
            engine.set_editor_fps_cap(240.0)
            engine.set_editor_idle_fps(0.0)
            expected_bytes_after_destroy = 0
            positions = np.array(
                [[-1, 0, -1], [-1, 0, 1], [1, 0, -1]], dtype=np.float32
            )
            indices = np.array([0, 1, 2], dtype=np.uint32)
            mesh = Mesh.from_data(positions, indices, name="Runtime GPU Triangle")
            scene = SceneManager.instance().get_active_scene()
            objects = [scene.create_game_object(name) for name in ("runtime-a", "runtime-b")]
            for owner in objects:
                owner.add_component("MeshRenderer").set_mesh_asset_guid(mesh.guid)

            def after_draw() -> None:
                nonlocal frames, destroyed, settled, expected_bytes_after_destroy
                try:
                    frames += 1
                    if frames == 24:
                        record = next(r for r in engine.asset_runtime_records if r.guid == mesh.guid)
                        assert record.gpu_resident_bytes > 0, record.gpu_resident_bytes
                        expected_bytes_after_destroy = (
                            engine.mesh_gpu_resident_bytes - record.gpu_resident_bytes
                        )
                        mesh.destroy()
                        assert all(owner.get_component("MeshRenderer").get_mesh_asset() is None for owner in objects)
                        destroyed = True
                    if frames == 72:
                        record = next((r for r in engine.asset_runtime_records if r.guid == mesh.guid), None)
                        assert destroyed
                        assert engine.pending_mesh_gpu_upload_count == 0
                        assert record is None or record.gpu_resident_bytes == 0
                        assert engine.mesh_gpu_resident_bytes == expected_bytes_after_destroy, (
                            engine.mesh_gpu_resident_bytes,
                            expected_bytes_after_destroy,
                            engine.retired_mesh_gpu_lease_count,
                        )
                        settled = True
                        print("runtime Mesh resource: shared publication and in-flight GPU retirement passed")
                        engine.exit()
                except BaseException as exc:
                    failures.append(exc)
                    engine.exit()

            engine.set_post_draw_callback(after_draw)
            engine.set_play_mode_rendering(True)
            engine.run()
            if failures:
                raise failures[0]
            assert settled, "Render loop ended before runtime Mesh retirement settled"
        finally:
            engine.cleanup()


if __name__ == "__main__":
    main()
