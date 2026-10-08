"""Skybox dependencies use the authored Scene schema through publication."""
import uuid

import pytest
from PIL import Image

from infernux.core.assets import AssetManager
from infernux.lib import ConsolePanel, SceneManager, Vector3, _Infernux as native
from tests.gpu.skybox_dependency_case import create_skybox_asset, publish_skybox, observe_skybox


@pytest.fixture
def skybox_asset(engine, tmp_path, monkeypatch):
    database = engine.get_asset_database()
    monkeypatch.setattr(AssetManager, '_engine', engine)
    monkeypatch.setattr(AssetManager, '_asset_database', database)
    source, guid = create_skybox_asset(tmp_path,database)
    try:
        yield source, guid, database
    finally:
        source.unlink()
        assert database.delete_asset(str(source))


@pytest.mark.parametrize('snapshot', [False, True], ids=['document','play-snapshot'])
@pytest.mark.parametrize('kind', ['material','missing','empty','mesh-control'])
def test_skybox_typed_dependency_uses_authored_document(scene, skybox_asset, snapshot, kind):
    _, material_guid, _ = skybox_asset
    guid = str(uuid.uuid4()) if kind == 'missing' else material_guid
    if kind in ('material','missing'):
        scene.set_environment({'skybox_material_guid':guid})
    elif kind == 'mesh-control':
        scene.create_game_object('Material Owner').add_component('MeshRenderer').set_materials([guid])
    document = scene.serialize_document()
    assert 'skyboxMaterialGuid' in document['environment']
    assert 'skybox_material_guid' not in document['environment']
    dependencies = (scene._capture_play_mode_snapshot()._resource_dependencies() if snapshot else
                    native._collect_scene_resource_dependencies(document))
    assert list(dependencies).count((guid,'Material')) == (0 if kind == 'empty' else 1)


def test_skybox_rejects_existing_asset_of_wrong_type(engine, scene, tmp_path):
    source = tmp_path / 'WrongSkybox.png'
    Image.new('RGB',(2,2),(25,50,75)).save(source)
    database = engine.get_asset_database()
    assert database.import_asset(str(source))
    guid = database.get_guid_from_path(str(source))
    try:
        scene.set_environment({'skybox_material_guid':guid})
        with pytest.raises(ValueError, match=r'Scene.environment.skyboxMaterialGuid expects Material.*Texture'):
            native._collect_scene_resource_dependencies(scene.serialize_document())
    finally:
        source.unlink()
        assert database.delete_asset(str(source))


@pytest.mark.parametrize('snapshot', [False,True], ids=['document','play-snapshot'])
def test_skybox_is_preloaded_before_scene_commit(engine, scene, skybox_asset, snapshot):
    _, guid, database = skybox_asset
    publish_skybox(scene,guid,database,engine,snapshot)


@pytest.mark.parametrize('snapshot', [False,True], ids=['document','play-snapshot'])
@pytest.mark.parametrize('pipeline_name', ['DefaultForwardPipeline','DefaultForwardPlusPipeline','DefaultDeferredPipeline'])
def test_published_skybox_draws_authored_material(engine, scene, skybox_asset, snapshot, pipeline_name):
    import infernux.renderstack as pipelines
    _, guid, database = skybox_asset
    manager, console = SceneManager.instance(), ConsolePanel()
    previous = [(obj,obj.active_self) for index in range(manager.scene_count)
                for obj in manager.get_scene_at(index).get_root_objects()]
    previous_rendering = engine.is_play_mode_rendering()
    pipeline = getattr(pipelines,pipeline_name)()
    results, failures = [], []
    try:
        for obj,_ in previous:
            obj.active = False
        camera_owner = scene.create_game_object('Skybox Witness')
        camera_owner.transform.position = Vector3(0,0,-5)
        camera_owner.add_component('Camera')
        publish_skybox(scene,guid,database,engine,snapshot)
        scene.main_camera = scene.find('Skybox Witness').get_component('Camera')
        engine.resize_game_render_target(160,120)
        engine.set_game_camera_enabled(True)
        engine.set_render_pipeline(pipeline)
        manager.play()
        manager.pause()
        engine.set_play_mode_rendering(True)
        frame, ticket = 0, None

        def after_draw():
            nonlocal frame,ticket
            try:
                frame += 1
                assert frame < 120
                if ticket is None and frame >= 8:
                    ticket = engine.request_render_target_readback(True)
                elif ticket is not None and ticket.done:
                    results.append(observe_skybox(ticket.result_numpy()))
                    issues = [item for item in console._get_visible_log_snapshot(1000)
                              if item['level'] in ('ERROR','FATAL','WARN','WARNING')]
                    assert not issues, issues
                    engine.exit()
            except BaseException as error:
                failures.append(error)
                engine.exit()

        engine.set_post_draw_callback(after_draw)
        engine.run()
        if failures:
            raise failures[0]
        assert len(results) == 1
    finally:
        engine.set_post_draw_callback(None)
        engine.set_render_pipeline(None)
        manager.stop()
        engine.set_play_mode_rendering(previous_rendering)
        pipeline.dispose()
        for obj,active in previous:
            obj.active = active
