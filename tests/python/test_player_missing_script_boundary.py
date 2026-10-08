"""Player scene publication cannot succeed with authoring repair placeholders."""
import json
import time

import pytest

from infernux.components import InxComponent
from infernux.components._cds_bridge import get_class_info
from infernux.components.missing_script import create_missing_script_component
from infernux.engine.component_restore import preflight_scene_python_components
from infernux.engine.player_scene import PlayerSceneService
from infernux.engine.player_service_graph import PlayerRuntimeAssetCatalog
from infernux.engine.scene_authoring import encode_runtime_scene_artifact
from infernux.lib import SceneManager, _cds_alive_count


class _BoundaryHealthyComponent(InxComponent):
    value: int = 37


class _BoundaryAllocationFailure(InxComponent):
    def __new__(cls):
        raise RuntimeError('intentional component allocation failure')


@pytest.fixture
def missing_script_scene(scene, tmp_path):
    witness = scene.create_game_object('ExistingWorldWitness')
    witness.add_py_component(_BoundaryHealthyComponent())
    baseline = scene.serialize_document()
    candidate = json.loads(json.dumps(baseline))
    candidate['name'] = 'BrokenPlayerScene'
    candidate['objects'][0]['components'].append({
        'type_id': 'python:1d3d7178de9a4aebad69650ed48da0af:2d3d7178de9a4aebad69650ed48da0af:absent_gameplay:MissingGameplay',
        'component_id': candidate['nextComponentId'],
        'enabled': True, 'execution_order': 0, 'data': {'authored_value': 37},
    })
    candidate['nextComponentId'] += 1
    path = tmp_path / 'Content/Unresolved.scene'
    path.parent.mkdir()
    path.write_text(json.dumps(encode_runtime_scene_artifact(candidate)), encoding='utf-8')
    artifact = 'content:missing-script-boundary'
    catalog = PlayerRuntimeAssetCatalog.from_documents(str(tmp_path), {'artifacts': [dict(
        runtime_artifact_id=artifact, runtime_path='Content/Unresolved.scene',
        package='Game_Data/Content.inxpkg', logical_type='scene', asset_guid='broken-scene', dependencies=[])]},
        {'entries': [dict(guid='broken-scene', runtime_path='Assets/Scenes/Unresolved.scene',
                         primary_runtime_artifact_id=artifact, runtime_artifact_ids=[artifact], dependencies=[])]})
    service = PlayerSceneService()
    service.bind_runtime_catalog(catalog)
    try:
        yield scene, witness, baseline, candidate, service
    finally:
        service.cancel_pending_load()
        SceneManager.instance().stop()


@pytest.mark.parametrize('mode', ['initial', 'single', 'additive'])
@pytest.mark.parametrize('failure', ['absent', 'constructor', 'placeholder'])
def test_player_rejects_missing_script_before_publishing_world(missing_script_scene, tmp_path, mode, failure):
    scene, witness, baseline, candidate, service = missing_script_scene
    descriptor = candidate['objects'][0]['components'][-1]
    expected_error = 'MissingGameplay'
    if failure == 'constructor':
        cls = _BoundaryAllocationFailure
        descriptor['type_id'] = (
            f'python:{cls._get_intrinsic_script_guid()}:{cls._get_type_guid()}:'
            f'{cls.__module__}:{cls.__qualname__}'
        )
        expected_error = 'intentional component allocation failure'
        (tmp_path / 'Content/Unresolved.scene').write_text(
            json.dumps(encode_runtime_scene_artifact(candidate)), encoding='utf-8')
    elif failure == 'placeholder':
        _, script_guid, type_guid, module_name, qualified_name = descriptor['type_id'].split(':')
        placeholder = create_missing_script_component(
            type_name='MissingGameplay', script_guid=script_guid, type_guid=type_guid,
            module_name=module_name, qualified_name=qualified_name, fields={},
            error='existing authoring repair placeholder',
        )
        placeholder._call_on_destroy()
    manager = SceneManager.instance()
    if mode != 'initial':
        manager.play()
        baseline = scene.serialize_document()
    component = witness.get_py_component(_BoundaryHealthyComponent)
    class_id = get_class_info(_BoundaryHealthyComponent)[0]
    alive = _cds_alive_count(class_id)
    worlds = {manager.get_scene_at(index).world_id for index in range(manager.scene_count)}
    if mode == 'initial':
        assert service.load_initial('broken-scene') is False
    else:
        assert service.request_prepared_load('broken-scene', mode=mode)
        deadline = time.monotonic() + 5
        while service.is_load_pending and time.monotonic() < deadline:
            service.process_pending_load()
            time.sleep(0.001)
        assert not service.is_load_pending
    assert expected_error in service.last_error
    assert 'objects[0].components[1]' in service.last_error
    assert manager.get_active_scene() is scene
    assert scene.find('ExistingWorldWitness') is witness
    assert witness.get_py_component(_BoundaryHealthyComponent) is component
    assert component.value == 37
    assert _cds_alive_count(class_id) == alive
    assert scene.serialize_document() == baseline
    assert {manager.get_scene_at(index).world_id for index in range(manager.scene_count)} == worlds


def test_editor_preflight_preserves_missing_script_for_repair(missing_script_scene):
    _, _, _, candidate, _ = missing_script_scene
    graph = preflight_scene_python_components(candidate)
    try:
        assert len(graph.components) == 2
        placeholder = graph.components[-1].instance
        assert placeholder._is_broken
        assert placeholder._serialize_fields_document()['authored_value'] == 37
    finally:
        graph.discard()


def test_player_publishes_resolved_script(missing_script_scene, tmp_path):
    scene, _, _, candidate, service = missing_script_scene
    candidate['objects'][0]['components'].pop()
    (tmp_path / 'Content/Unresolved.scene').write_text(
        json.dumps(encode_runtime_scene_artifact(candidate)), encoding='utf-8')
    assert service.load_initial('broken-scene'), service.last_error
    assert service.last_error == ''
    restored = scene.find('ExistingWorldWitness').get_py_component(_BoundaryHealthyComponent)
    assert restored.value == 37
    assert not getattr(restored, '_is_broken', False)
