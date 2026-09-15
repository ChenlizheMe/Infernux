"""Real Vulkan regression: superseded mesh generations retire below budget."""
from pathlib import Path
import argparse
import tempfile

from Infernux import Engine
from Infernux.lib import AssetRegistry, PrimitiveType, SceneManager


def main(continuous=False):
    update_count = 192 if continuous else 24
    frame_interval = 1 if continuous else 8
    with tempfile.TemporaryDirectory(prefix="infernux-mesh-publication-") as root:
        project = Path(root)
        (project / "Assets").mkdir()
        (project / "ProjectSettings").mkdir()
        source = project / "Assets" / "surface.obj"
        source.write_text("v -1 0 -1\nv -1 0 1\nv 1 0 -1\nf 1 2 3\n", encoding="ascii")
        frontend = Engine()
        engine = frontend.get_native_engine()
        frames = 0
        published = 0
        settled = False
        failure = []
        peak_pending = 0
        try:
            frontend.init_renderer(64, 64, str(project))
            engine.set_editor_fps_cap(240.0)
            engine.set_editor_idle_fps(0.0)
            registry = AssetRegistry.instance()
            mesh = registry.load_mesh(str(source))
            positions = mesh._particle_sampling_data()["positions"]
            scene = SceneManager.instance().get_active_scene()
            initial_evictions = engine.mesh_gpu_eviction_count
            for name in ("shared-a", "shared-b"):
                go = scene.create_primitive(PrimitiveType.Cube, name)
                go.get_component("MeshRenderer").set_mesh_asset_guid(mesh.guid)

            def after_draw():
                nonlocal frames, published, settled, peak_pending
                try:
                    frames += 1
                    peak_pending = max(peak_pending, engine.pending_mesh_gpu_upload_count)
                    # Continuous mode deliberately supersedes in-flight versions.
                    if frames % frame_interval == 0 and published < update_count:
                        positions[:, 1] = (published % 24) * 0.02
                        registry.update_mesh_positions(mesh.guid, 0, positions)
                        published += 1
                    if frames == 240:
                        record = next(r for r in engine.asset_runtime_records if r.guid == mesh.guid)
                        assert published == update_count
                        assert engine.completed_mesh_gpu_upload_count >= (1 if continuous else update_count)
                        assert engine.pending_mesh_gpu_upload_count == 0
                        assert engine.mesh_gpu_eviction_count == initial_evictions
                        assert record.runtime_version == registry.get_asset_residency(mesh.guid).runtime_version
                        assert record.gpu_resident_bytes > 0
                        assert record.stale_gpu_bytes == 0, record.stale_gpu_bytes
                        settled = True
                        print(f"mesh publication GPU regression: {published} updates; "
                              f"peak pending = {peak_pending}; final pending = 0; stale GPU bytes = 0")
                        engine.exit()
                except BaseException as exc:
                    failure.append(exc)
                    engine.exit()

            engine.set_post_draw_callback(after_draw)
            engine.set_play_mode_rendering(True)
            engine.run()
            if failure:
                raise failure[0]
            assert settled, "Render loop ended before the retirement assertion"
        finally:
            engine.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--continuous", action="store_true")
    main(continuous=parser.parse_args().continuous)
