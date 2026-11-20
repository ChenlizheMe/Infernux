"""Skybox dependencies use the authored Scene schema through publication."""

import pytest
from PIL import Image

from infernux.core.assets import AssetManager
from infernux.lib import ConsolePanel, PrimitiveType, SceneManager, Vector3, _Infernux as native
from tests.gpu.skybox_dependency_case import create_skybox_asset, publish_skybox, observe_skybox


@pytest.fixture
def skybox_asset(engine, tmp_path, monkeypatch, request):
    database = engine.get_asset_database()
    monkeypatch.setattr(AssetManager, "_engine", engine)
    monkeypatch.setattr(AssetManager, "_asset_database", database)
    source, guid = create_skybox_asset(tmp_path, database, getattr(request, 'param', 'procedural'))
    try:
        yield source, guid, database
    finally:
        source.unlink()
        assert database.delete_asset(str(source))
        texture = source.with_suffix('.png')
        if texture.exists():
            texture.unlink()
            assert database.delete_asset(str(texture))


def test_skybox_rejects_existing_asset_of_wrong_type(engine, scene, tmp_path):
    source = tmp_path / "WrongSkybox.png"
    Image.new("RGB", (2, 2), (25, 50, 75)).save(source)
    database = engine.get_asset_database()
    assert database.import_asset(str(source))
    guid = database.get_guid_from_path(str(source))
    try:
        scene.set_environment({"skybox_material_guid": guid})
        with pytest.raises(ValueError, match=r"Scene.environment.skyboxMaterialGuid expects Material.*Texture"):
            native._collect_scene_resource_dependencies(scene.serialize_document())
    finally:
        source.unlink()
        assert database.delete_asset(str(source))


@pytest.mark.parametrize("snapshot", [False, True], ids=["document", "play-snapshot"])
def test_skybox_is_preloaded_before_scene_commit(engine, scene, skybox_asset, snapshot):
    _, guid, database = skybox_asset
    publish_skybox(scene, guid, database, engine, snapshot)


@pytest.mark.parametrize(
    "pipeline_name,snapshot,skybox_asset",
    [("DefaultForwardPipeline", False, "procedural"), ("DefaultDeferredPipeline", True, "procedural"),
     ("DefaultForwardPipeline", False, "panoramic"), ("DefaultDeferredPipeline", True, "panoramic")],
    indirect=["skybox_asset"],
)
def test_published_skybox_draws_authored_material(engine, scene, skybox_asset, snapshot, pipeline_name):
    import infernux.renderstack as pipelines

    _, guid, database = skybox_asset
    manager, console = SceneManager.instance(), ConsolePanel()
    previous = [
        (obj, obj.active_self)
        for index in range(manager.scene_count)
        for obj in manager.get_scene_at(index).get_root_objects()
    ]
    previous_rendering = engine.is_play_mode_rendering()
    pipeline = getattr(pipelines, pipeline_name)()
    results, failures = [], []
    try:
        for obj, _ in previous:
            obj.active = False
        camera_owner = scene.create_game_object("Skybox Witness")
        camera_owner.transform.position = Vector3(0, 0, -5)
        camera_owner.add_component("Camera")
        publish_skybox(scene, guid, database, engine, snapshot)
        scene.main_camera = scene.find("Skybox Witness").get_component("Camera")
        camera = scene.main_camera
        engine.resize_game_render_target(160, 120)
        engine.set_game_camera_enabled(True)
        engine.set_render_pipeline(pipeline)
        engine.set_play_mode_rendering(True)
        frame, ticket, changed = 0, None, 0
        images = []

        def after_draw():
            nonlocal frame, ticket, changed
            try:
                frame += 1
                assert frame < 120
                execution = engine.renderer_frame_snapshot["game_render_graph_execution_count"]
                if ticket is None and engine.game_presentation_ready and execution >= changed + 8:
                    ticket = engine.request_render_target_readback(True)
                elif ticket is not None and ticket.done:
                    pixels = ticket.result_numpy().copy()
                    results.append(observe_skybox(pixels))
                    images.append(pixels)
                    issues = [
                        item
                        for item in console._get_visible_log_snapshot(1000)
                        if item["level"] in ("ERROR", "FATAL", "WARN", "WARNING")
                    ]
                    assert not issues, issues
                    if len(images) == 1:
                        camera.dithering = True
                    elif len(images) == 2:
                        import numpy as np
                        difference = pixels[..., :3] - images[0][..., :3]
                        assert np.any(np.abs(difference) > .001)
                        assert np.abs(difference).max() < .006
                        assert np.abs(difference.mean(axis=(0, 1))).max() < .001
                        camera.dithering = False
                    else:
                        import numpy as np
                        np.testing.assert_array_equal(pixels, images[0])
                        engine.exit()
                    ticket, changed = None, execution
            except BaseException as error:
                failures.append(error)
                engine.exit()

        engine.set_post_draw_callback(after_draw)
        engine.run()
        if failures:
            raise failures[0]
        assert len(results) == 3
    finally:
        engine.set_post_draw_callback(None)
        engine.set_render_pipeline(None)
        engine.set_play_mode_rendering(previous_rendering)
        pipeline.dispose()
        for obj, active in previous:
            obj.active = active


@pytest.mark.parametrize("skybox_asset", ["panoramic"], indirect=True)
def test_panoramic_skybox_lights_a_diffuse_receiver(engine, scene, skybox_asset):
    from infernux.renderstack import DefaultDeferredPipeline

    _, guid, database = skybox_asset
    previous = [(obj, obj.active_self) for obj in scene.get_root_objects()]
    previous_rendering = engine.is_play_mode_rendering()
    pipeline = DefaultDeferredPipeline()
    pipeline.global_illumination = True
    errors = []
    ticket = None
    result = None
    callbacks = 0
    try:
        for obj, _ in previous:
            obj.active = False
        owner = scene.create_game_object("Environment witness camera")
        owner.transform.position = Vector3(0, 1.8, -3.1)
        owner.transform.euler_angles = Vector3(13, 0, 0)
        scene.main_camera = owner.add_component("Camera")
        floor = scene.create_primitive(PrimitiveType.Cube, "Environment receiver")
        floor.transform.position = Vector3(0, -.1, 0)
        floor.transform.local_scale = Vector3(8, .2, 9)
        publish_skybox(scene, guid, database, engine, snapshot=True)
        scene.main_camera = scene.find("Environment witness camera").get_component("Camera")
        engine.resize_game_render_target(320, 180)
        engine.set_game_camera_enabled(True)
        engine.set_render_pipeline(pipeline)
        engine.set_play_mode_rendering(True)
        first = engine.renderer_frame_snapshot["game_render_graph_execution_count"]

        def after_draw():
            nonlocal callbacks, ticket, result
            try:
                callbacks += 1
                assert callbacks < 300, "The environment-lit view did not produce a readback"
                frame = engine.renderer_frame_snapshot
                settled = frame["game_render_graph_execution_count"] >= first + 32
                if ticket is None and engine.game_presentation_ready and settled:
                    assert "Ray Diffuse Irradiance" in frame["game_render_graph_pass_names"]
                    ticket = engine.request_render_target_readback(True)
                elif ticket is not None and ticket.done:
                    result = ticket.result_numpy().astype("float32")
                    engine.exit()
            except BaseException as error:
                errors.append(error)
                engine.exit()

        engine.set_post_draw_callback(after_draw)
        engine.run()
        if errors:
            raise errors[0]
        assert result is not None
        sky = result[5:25, 110:210, :3].mean(axis=(0, 1))
        lit = result[125:155, 110:210, :3].mean(axis=(0, 1))
        assert sky[0] > lit[0] + .03, (sky, lit)
        assert lit[0] > max(lit[1], lit[2]) + .15, lit
    finally:
        engine.set_post_draw_callback(None)
        engine.set_render_pipeline(None)
        engine.set_play_mode_rendering(previous_rendering)
        pipeline.dispose()
        for obj, active in previous:
            obj.active = active
