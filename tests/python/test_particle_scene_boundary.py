"""Real GPU simulation survives render-view invalidation at scene boundaries."""
import pytest

from infernux.components import ParticleSystem
from infernux.core.asset_ref import ParticleGraphRef
from infernux.core.assets import AssetManager
from infernux.lib import SceneManager
from infernux.particle import ParticleScriptCompiler
from infernux.renderstack import RenderStackPipeline
from infernux.runtime_services import install_runtime_service, remove_runtime_service


@pytest.mark.parametrize("capacity", [4095, 4096])
def test_view_boundary_preserves_first_burst_and_resident_particles(engine, scene, tmp_path, capacity):
    database = engine.get_asset_database()
    path = tmp_path / "Boundary.particlegraph"
    ParticleScriptCompiler().parse(f'''
from infernux.particle import EmitterSettings, ParticleBurst, ParticleEmitter, ParticleScript
class Boundary(ParticleScript):
    class Emitter(ParticleEmitter):
        stable_id = "emitter"
        settings = EmitterSettings(capacity={capacity}, spawn_rate=0.0, duration=1000.0, loop=False,
                                   bursts=(ParticleBurst(time=0.0, count=16),))
        def init(self, ctx, particles):
            particles.set_position((0.125, 0.25, 0.375))
            particles.set_lifetime(1000.0)
        def update(self, ctx, particles):
            particles.set_position((0.125, 0.25, 0.375))
        def rendering(self, ctx, particles):
            particles.sprite()
''').save(str(path))
    imported = AssetManager.import_asset(str(path), database=database)
    assert imported.succeeded, imported.error
    scene.create_game_object("Camera").add_component("Camera")
    component = scene.create_game_object("Burst").add_py_component(ParticleSystem())
    manager = SceneManager.instance()
    install_runtime_service("gpu-particles", engine)
    pipeline = RenderStackPipeline()
    errors, snapshots = [], []
    frames, capture_after = 0, 4
    request = None
    try:
        engine.set_render_pipeline(pipeline)
        manager.play()
        manager.pause()
        # Publish before the first render frame, as an ordinary Play entry does.
        component.graph = ParticleGraphRef(guid=imported.guid)
        component.awake()
        assert component.play()

        def update_frame(_delta):
            nonlocal frames, capture_after, request
            frames += 1
            try:
                assert frames <= 240, (snapshots, request)
                if request is not None:
                    result = component.poll_gpu_diagnostics(request)
                    assert result["status"] in {"pending", "completed"}, result
                    if result["status"] == "completed":
                        emitter = result["emitters"][0]
                        assert emitter["alive_count"] == 16, emitter
                        assert emitter["free_count"] == capacity - 16, emitter
                        samples = emitter["state_samples"]
                        assert len(samples) == 8, emitter
                        for sample in samples:
                            assert sample["attributes"]["builtin.position"] == pytest.approx((.125, .25, .375))
                        identities = [(sample["slot_index"], sample["spawn_generation"]) for sample in samples]
                        if snapshots:
                            assert identities == snapshots[0]
                        snapshots.append(identities)
                        request = None
                        capture_after = frames + 4
                        if len(snapshots) == 1:
                            manager.mark_temporal_discontinuity()
                        elif len(snapshots) == 2:
                            engine.set_play_mode_rendering(True)
                        elif len(snapshots) == 3:
                            engine.set_play_mode_rendering(False)
                        else:
                            engine.exit()
                            return
                component.update(1. / 120.)
                if request is None and frames >= capture_after:
                    request = component.request_gpu_diagnostics(1, 8)
            except BaseException as error:
                errors.append(error)
                engine.exit()

        engine.set_pre_scene_update_callback(update_frame)
        engine.run()
        assert not errors, repr(errors)
        assert len(snapshots) == 4
    finally:
        engine.set_pre_scene_update_callback(None)
        engine.set_play_mode_rendering(False)
        component._remove_native_batch()
        engine.set_render_pipeline(None)
        manager.stop()
        assert remove_runtime_service("gpu-particles", engine)
        assert database.delete_asset(str(path))
