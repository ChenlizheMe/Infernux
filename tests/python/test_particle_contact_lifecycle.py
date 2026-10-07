"""Actual collision shaders, contact lifecycle, continuations and GPU readback."""
import json
import pytest

from infernux.components import ParticleSystem
from infernux.core.asset_ref import ParticleGraphRef
from infernux.core.assets import AssetManager
from infernux.lib import SceneManager
from infernux.particle import ParticleScriptCompiler
from infernux.renderstack import RenderStackPipeline
from infernux.runtime_services import install_runtime_service, remove_runtime_service


def contact_source(capacity, wait):
    continuation = "            ctx.wait_frames(2)\n            particles.add_size(1.0)" if wait else ""
    return f'''
from infernux.particle import EmitterSettings, ParticleBurst, ParticleEmitter, ParticleScript
class Contacts(ParticleScript):
    class Emitter(ParticleEmitter):
        stable_id = "emitter"
        settings = EmitterSettings(capacity={capacity}, spawn_rate=0.0, duration=1000.0, loop=False,
                                   collision_enabled=True,
                                   bursts=(ParticleBurst(time=0.0, count=1, cycles=2, interval=1.0),))
        def init(self, ctx, particles):
            particles.set_position((0.0, 0.0, 0.0))
            particles.set_lifetime(0.8)
            particles.set_size(0.1)
            particles.set_color((0.0, 0.0, 0.0, 1.0))
        def update(self, ctx, particles):
            particles.set_position((0.0, 0.0, 0.0))
        def rendering(self, ctx, particles):
            particles.sprite()
        def collision_enter(self, ctx, particles):
            particles.add_color((1.0, 0.0, 0.0, 0.0))
{continuation}
        def collision_stay(self, ctx, particles):
            particles.add_color((0.0, 1.0, 0.0, 0.0))
        def collision_exit(self, ctx, particles):
            particles.add_color((0.0, 0.0, 1.0, 0.0))
'''


@pytest.mark.parametrize("capacity", [4095, 4096])
@pytest.mark.parametrize("wait", [False, True])
def test_gpu_contacts_survive_multi_contact_deletion_and_slot_reuse(engine, scene, tmp_path, capacity, wait):
    database = engine.get_asset_database()
    path = tmp_path / "Contacts.particlegraph"
    ParticleScriptCompiler().parse(contact_source(capacity, wait)).save(str(path))
    imported = AssetManager.import_asset(str(path), database=database)
    assert imported.succeeded, imported.error
    scene.create_game_object("Camera").add_component("Camera")
    first = scene.create_game_object("First trigger")
    first_box = first.add_component("BoxCollider")
    first_box.is_trigger = True
    second = scene.create_game_object("Second trigger")
    second_box = second.add_component("BoxCollider")
    second_box.is_trigger = True
    second_box.enabled = False
    component = scene.create_game_object("Contacts").add_py_component(ParticleSystem())
    manager = SceneManager.instance()
    install_runtime_service("gpu-particles", engine)
    pipeline = RenderStackPipeline()
    errors, snapshots = [], []
    frames, capture_after = 0, 8
    request = None
    evidence = []
    try:
        engine.set_render_pipeline(pipeline)
        manager.play()
        manager.pause()
        component.graph = ParticleGraphRef(guid=imported.guid)
        component.awake()
        assert component.play()

        def update_frame(_delta):
            nonlocal frames, capture_after, request
            frames += 1
            try:
                assert frames <= 200, (snapshots, request)
                if request is not None:
                    result = component.poll_gpu_diagnostics(request)
                    assert result["status"] in {"pending", "completed"}, result
                    if result["status"] == "completed":
                        evidence.append(dict(frame=frames, result=result, renderer=engine.renderer_frame_snapshot))
                        (tmp_path / "contact-readback.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
                        emitter = result["emitters"][0]
                        assert emitter["alive_count"] == 1 and emitter["free_count"] == capacity - 1, emitter
                        sample, = emitter["state_samples"]
                        if emitter["contact_current_record_count"]:
                            assert emitter["contact_min_particle_index"] == sample["slot_index"], emitter
                            assert emitter["contact_max_particle_index"] == sample["slot_index"], emitter
                        attributes = sample["attributes"]
                        phase = len(snapshots)
                        enter, exits = ((1, 0), (2, 0), (2, 1), (2, 2), (3, 2), (1, 0))[phase]
                        color = attributes["builtin.color"]
                        assert color[0] == enter and color[2] == exits and color[3] == 1., sample
                        assert color[1] > 0, sample
                        assert attributes["builtin.size"] == pytest.approx(.1 + (enter if wait else 0)), sample
                        if snapshots:
                            assert sample["slot_index"] == snapshots[0]["slot_index"]
                            if phase == 5:
                                assert attributes["builtin.id"] != snapshots[0]["attributes"]["builtin.id"]
                            else:
                                assert attributes["builtin.id"] == snapshots[0]["attributes"]["builtin.id"]
                        snapshots.append(sample)
                        request = None
                        capture_after = frames + 8
                        if phase == 0:
                            second_box.enabled = True
                        elif phase == 1:
                            first_box.enabled = False
                        elif phase == 2:
                            scene.destroy_game_object(second)
                        elif phase == 3:
                            first_box.enabled = True
                        elif phase == 4:
                            capture_after = 140
                        else:
                            engine.exit()
                            return
                component.update(1. / 120.)
                if request is None and frames >= capture_after:
                    request = component.request_gpu_diagnostics(1, 1)
            except BaseException as error:
                errors.append(error)
                engine.exit()

        engine.set_pre_scene_update_callback(update_frame)
        engine.run()
        assert not errors, repr(errors)
        assert len(snapshots) == 6
    finally:
        engine.set_pre_scene_update_callback(None)
        component._remove_native_batch()
        engine.set_render_pipeline(None)
        manager.stop()
        assert remove_runtime_service("gpu-particles", engine)
        assert database.delete_asset(str(path))
