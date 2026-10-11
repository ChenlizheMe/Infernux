"""A failed particle surface source recovers through GUID-stable asset editing."""
import json
from pathlib import Path

from PIL import Image

from infernux.components import ParticleSystem
from infernux.core.asset_ref import ParticleGraphRef
from infernux.core.asset_types import TextureImportSettings, TextureCompression, write_texture_import_settings
from infernux.core.assets import AssetManager
from infernux.lib import ConsolePanel, SceneManager, Vector3
from infernux.particle import ParticleScriptCompiler
from infernux.renderstack import RenderStackPipeline
from infernux.runtime_services import install_runtime_service, remove_runtime_service


def source(guid):
    return f'''
from infernux.particle import AssetReference, EmitterSettings, ParticleBurst, ParticleEmitter, ParticleScript
class Recovery(ParticleScript):
    class Emitter(ParticleEmitter):
        stable_id = "emitter"
        settings = EmitterSettings(capacity=8, spawn_rate=0.0, duration=1000.0, loop=False,
                                   bursts=(ParticleBurst(time=0.0, count=1),))
        def init(self, ctx, particles):
            particles.set_position((0.0, 0.0, 0.0))
            particles.set_size(2.0)
            particles.set_lifetime(1000.0)
        def update(self, ctx, particles):
            pass
        def rendering(self, ctx, particles):
            particles.sprite(properties={{"texSampler": AssetReference(guid={guid!r})}})
'''


def create_volume(path, database):
    path.write_text(json.dumps({"$schema": "infernux.vector_field", "dimensions": [2, 1, 1],
        "storage_order": "x_fastest", "bake_basis": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        "vectors": [[1, 0, 0], [0, 1, 0]]}), encoding="utf-8")
    result = AssetManager.import_asset(str(path), database=database)
    assert result.succeeded, result.error
    return result.guid


def replace_with_image(volume, image, database):
    # Correct the source type while retaining its GUID and every consumer.
    Image.new("RGBA", (8, 8), (0, 255, 0, 255)).save(volume, format="PNG")
    volume.rename(image)
    Path(str(volume) + ".meta").rename(Path(str(image) + ".meta"))
    moved = AssetManager.move_asset(str(volume), str(image), database=database)
    assert moved.succeeded, moved.error
    settings = TextureImportSettings(generate_mipmaps=False, compression=TextureCompression.NONE)
    assert write_texture_import_settings(str(image), settings)
    result = AssetManager.reimport_asset(str(image), database=database)
    assert result.succeeded, result.error
    return result.guid


def test_failed_particle_texture_recovers_without_recreating_emitter(engine, scene, tmp_path):
    database = engine.get_asset_database()
    volume, image = tmp_path / "Surface.inxvfield", tmp_path / "Surface.png"
    guid = create_volume(volume, database)
    graph = tmp_path / "Recovery.particlegraph"
    ParticleScriptCompiler().parse(source(guid)).save(str(graph))
    imported = AssetManager.import_asset(str(graph), database=database)
    assert imported.succeeded, imported.error
    camera = scene.create_game_object("Camera")
    camera.transform.position = Vector3(0, 0, -5)
    camera.add_component("Camera")
    component = scene.create_game_object("Particle").add_py_component(ParticleSystem())
    manager, console = SceneManager.instance(), ConsolePanel()
    pipeline = RenderStackPipeline()
    previous_engine = AssetManager._engine
    AssetManager._engine = engine
    install_runtime_service("gpu-particles", engine)
    failures, colors, initial_errors = [], [], []
    resident_id = None
    frame, changed, ticket = 0, 0, None
    try:
        engine.set_render_pipeline(pipeline)
        engine.resize_game_render_target(64, 64)
        engine.set_game_camera_enabled(True)
        manager.play()
        manager.pause()
        component.graph = ParticleGraphRef(guid=imported.guid)
        component.awake()
        assert component.play()

        def update(_delta):
            component.update(1. / 120.)

        def after_draw():
            nonlocal frame, changed, ticket, resident_id
            try:
                frame += 1
                assert frame < 160, colors
                AssetManager.flush_pending_gpu_texture_reloads()
                if ticket is None and frame >= changed + 16:
                    ticket = engine.request_render_target_readback(True)
                elif ticket is not None and ticket.done:
                    pixels = ticket.result_numpy()
                    center = pixels[pixels.shape[0] // 2, pixels.shape[1] // 2, :3].tolist()
                    colors.append(center)
                    (tmp_path / "texture-recovery.json").write_text(json.dumps(colors), encoding="utf-8")
                    if len(colors) == 1:
                        assert min(center) > .7, center
                        messages = console._get_visible_log_snapshot(1000)
                        assert any("expected texture dimension" in str(item) for item in messages), messages
                        initial_errors.extend(item for item in messages if item['level'] in ('ERROR', 'FATAL'))
                        resident_id = component._batch_id
                        assert replace_with_image(volume, image, database) == guid
                        changed, ticket = frame, None
                    else:
                        assert center[1] > .7 and center[0] < .1 and center[2] < .1, center
                        assert component.runtime_diagnostics()["resident"]
                        assert component._batch_id == resident_id
                        errors = [item for item in console._get_visible_log_snapshot(1000)
                                  if item['level'] in ('ERROR', 'FATAL')]
                        assert errors == initial_errors, errors
                        engine.exit()
            except BaseException as error:
                failures.append(error)
                engine.exit()

        engine.set_pre_scene_update_callback(update)
        engine.set_post_draw_callback(after_draw)
        engine.run()
        if failures:
            raise failures[0]
        assert len(colors) == 2
    finally:
        engine.set_pre_scene_update_callback(None)
        engine.set_post_draw_callback(None)
        component._remove_native_batch()
        engine.set_render_pipeline(None)
        manager.stop()
        assert remove_runtime_service("gpu-particles", engine)
        AssetManager._engine = previous_engine
        for path in (graph, volume, image):
            if path.exists():
                assert database.delete_asset(str(path))
