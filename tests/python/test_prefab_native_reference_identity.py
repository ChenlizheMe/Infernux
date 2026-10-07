"""Native joint references use Prefab source identities, then instance IDs."""
import json

import pytest

from infernux.engine.prefab_manager import instantiate_prefab, save_prefab
from infernux.engine.prefab_overrides import apply_overrides_to_prefab


def _joint_tree(scene, kind):
    root = scene.create_game_object('JointPrefab')
    owner = scene.create_game_object('Owner')
    owner.set_parent(root)
    owner.add_component('Rigidbody').is_kinematic = True
    target = scene.create_game_object('Target')
    target.set_parent(root)
    body = target.add_component('Rigidbody')
    body.is_kinematic = True
    owner.add_component(kind).connected_body = body
    return root


@pytest.mark.parametrize('kind', ['HingeJoint', 'SliderJoint'])
def test_prefab_export_localizes_native_joint_references(scene, tmp_path, kind):
    root = _joint_tree(scene, kind)
    path = tmp_path / 'Joint.prefab'
    assert save_prefab(root, str(path))
    document = json.loads(path.read_text(encoding='utf-8'))['root_object']
    owner, target = document['children']
    joint = next(c for c in owner['components'] if c['type_id'].endswith('.' + kind))
    body = next(c for c in target['components'] if c['type_id'].endswith('.Rigidbody'))
    assert joint['data']['connected_body_component_id'] == body['component_id']


@pytest.mark.parametrize('kind', ['HingeJoint', 'SliderJoint'])
def test_repeated_prefab_instances_and_apply_keep_joint_targets(scene, tmp_path, kind):
    root = _joint_tree(scene, kind)
    path = tmp_path / 'Joint.prefab'
    assert save_prefab(root, str(path))
    first, second = [instantiate_prefab(file_path=str(path), guid='joint-source', scene=scene) for _ in range(2)]
    for instance in (first, second):
        owner, target = instance.get_children()
        assert owner.get_component(kind).connected_body is target.get_component('Rigidbody')
    # A source change must not overwrite an external target authored only on
    # another instance. Its private merge ID never belongs in the source file.
    external = scene.create_game_object('External')
    external_body = external.add_component('Rigidbody')
    second.get_child(0).get_component(kind).connected_body = external_body
    first.get_child(1).get_component('Rigidbody').mass = 3.0
    assert apply_overrides_to_prefab(first, str(path))
    assert first.get_child(0).get_component(kind).connected_body is first.get_child(1).get_component('Rigidbody')
    assert second.get_child(0).get_component(kind).connected_body is external_body


@pytest.mark.parametrize('kind', ['HingeJoint', 'SliderJoint'])
def test_prefab_joint_null_target_and_external_target_save_boundary(scene, tmp_path, kind):
    root = _joint_tree(scene, kind)
    joint = root.get_child(0).get_component(kind)
    joint.connected_body = None
    path = tmp_path / 'Joint.prefab'
    assert save_prefab(root, str(path))
    original = path.read_bytes()
    instance = instantiate_prefab(file_path=str(path), guid='null-joint-source', scene=scene)
    assert instance.get_child(0).get_component(kind).connected_body is None
    external = scene.create_game_object('External')
    joint.connected_body = external.add_component('Rigidbody')
    assert not save_prefab(root, str(path))
    assert path.read_bytes() == original
