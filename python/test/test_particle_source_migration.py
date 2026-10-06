"""Live Python -> native particle bindings survive Scene residency changes."""
import pytest

from infernux.components import ParticleSystem
from infernux.components.ref_wrappers import ComponentRef
from infernux.core.asset_ref import ParticleGraphRef
from infernux.core.assets import AssetManager
from infernux.graph import AssetReference, GraphDocument, GraphLinkRecord, GraphNodeRecord, PortKind, TypeRef, ValueType
from infernux.lib import SceneManager, Vector3
from infernux.particle import EmitterSettings, ParticleBurst, ParticleEmitterAsset, ParticleGraphAsset, ParticleParameter
from infernux.runtime_services import install_runtime_service, remove_runtime_service
from infernux.renderstack import RenderStackPipeline
from test_model_animation_clips import model


@pytest.mark.parametrize("move", ["root", "reparent"])
def test_live_skin_binding_follows_scene_moves_without_republishing(model, scene, engine, monkeypatch, move):
    database, source, model_guid = model
    manager = SceneManager.instance()
    source_scene = manager.create_scene("Particle skin source")
    destination = manager.create_scene("Particle skin destination")
    manager.set_active_scene(scene)
    source_object = source_scene.create_game_object("Skin")
    skin = source_object.add_component("SkinnedMeshRenderer")
    skin.set_source_model_guid(model_guid)
    nodes = (
        GraphNodeRecord("root.update", "particle.root.update"),
        GraphNodeRecord("parameter", "particle.parameter", properties={"parameter": "surface"}),
        GraphNodeRecord("sample", "particle.mesh.sample"),
        GraphNodeRecord("coordinate", "common.constant.vec3", properties={"value": [0.1, 0.2, 0.3]}),
        GraphNodeRecord("position", "particle.attribute.position"),
    )
    update = GraphDocument("particle.update", nodes, (
        GraphLinkRecord("exec", "root.update", "out", "position", "in", PortKind.EXEC),
        GraphLinkRecord("mesh", "parameter", "value", "sample", "mesh", PortKind.VALUE),
        GraphLinkRecord("coordinate", "coordinate", "value", "sample", "sample", PortKind.VALUE),
        GraphLinkRecord("position", "sample", "position", "position", "value", PortKind.VALUE),
    ))
    path = source.with_name("LiveSkin.particlegraph")
    ParticleGraphAsset(
        parameters=(ParticleParameter("surface", "Surface", TypeRef(ValueType.MESH),
                                      AssetReference(guid=model_guid).to_dict()),),
        emitters=(ParticleEmitterAsset(
            stable_id="skin-emitter", update=update,
            settings=EmitterSettings(capacity=8, spawn_rate=0., bursts=(ParticleBurst(0., 1),)),
        ),),
    ).save(str(path))
    imported = AssetManager.import_asset(str(path), database=database)
    assert imported.succeeded, imported.error
    owner = scene.create_game_object("Particles")
    scene.create_game_object("Camera").add_component("Camera")
    component = owner.add_py_component(ParticleSystem())
    install_runtime_service("gpu-particles", engine)
    results, errors = [], []
    frames = 0
    request = None
    capture_after = 3
    pipeline = RenderStackPipeline()
    try:
        engine.set_render_pipeline(pipeline)
        manager.play()
        manager.pause()
        # Let the renderer consume this fixture's initial Scene/Play boundary
        # before publishing the graph whose continuity is under test.
        boundary_frames = 0

        def prepare_frame(_delta):
            nonlocal boundary_frames
            boundary_frames += 1
            if boundary_frames == 3:
                engine.exit()

        engine.set_pre_scene_update_callback(prepare_frame)
        engine.run()
        component.graph = ParticleGraphRef(guid=imported.guid)
        component.awake()
        component.set_parameter("surface", ComponentRef(skin))
        assert component.play()
        controller = component._gpu_controllers[0]
        emitter_id = component._gpu_emitter_ids[0]
        revision = engine._gpu_particle_artifact_revision(emitter_id)

        def reject_republication(*_args, **_kwargs):
            raise AssertionError("A source Scene move must not rebuild or reset its particle graph")

        monkeypatch.setattr(component, "_publish_gpu_particle_graph", reject_republication)

        def update_frame(_delta):
            nonlocal frames, request, capture_after
            frames += 1
            try:
                assert frames <= 240, ("GPU diagnostics did not complete", results, request)
                if request is not None:
                    result = component.poll_gpu_diagnostics(request)
                    assert result["status"] in {"pending", "completed"}, result
                    if result["status"] == "completed":
                        emitter = result["emitters"][0]
                        assert emitter["alive_count"] == 1, emitter
                        sample = emitter["state_samples"][0]
                        assert sample["attributes"]["builtin.position"]
                        results.append(sample)
                        request = None
                        capture_after = frames + 3
                        if len(results) == 1:
                            if move == "root":
                                manager.move_game_object_to_scene(source_object, destination)
                            else:
                                source_object.set_parent(destination.create_game_object("Skin parent"), True)
                            assert source_object.scene is destination
                            source_object.transform.position = Vector3(100., 0., 0.)
                        elif len(results) == 2:
                            manager.dont_destroy_on_load(source_object)
                            manager.prepare_active_scene_replacement()
                            assert source_object.scene is manager.get_runtime_persistent_scene()
                            source_object.transform.position = Vector3(200., 0., 0.)
                        else:
                            engine.exit()
                            return
                component.update(1. / 120.)
                if request is None and frames >= capture_after:
                    request = component.request_gpu_diagnostics(1, 1)
                assert component._gpu_controllers[0] is controller
                assert engine._gpu_particle_artifact_revision(emitter_id) == revision
                assert manager.get_active_scene() is scene
            except BaseException as error:
                errors.append(error)
                engine.exit()

        engine.set_pre_scene_update_callback(update_frame)
        engine.run()
        assert not errors, repr(errors)
        assert len(results) == 3
        baseline = results[0]["attributes"]["builtin.position"]
        for sample, displacement in zip(results, (0., 100., 200.)):
            assert sample["slot_index"] == results[0]["slot_index"]
            assert sample["spawn_generation"] == results[0]["spawn_generation"]
            assert sample["attributes"]["builtin.position"] == pytest.approx(
                (baseline[0] + displacement, baseline[1], baseline[2]), abs=1e-3,
            )
    finally:
        engine.set_pre_scene_update_callback(None)
        engine.set_render_pipeline(None)
        component._remove_native_batch()
        manager.stop()
        assert remove_runtime_service("gpu-particles", engine)
