"""Geometry publication must retain a joint's authored attachment points."""
import pytest

from infernux.lib import Physics, SceneManager, Vector3


def _step(count):
    for _ in range(count):
        SceneManager.instance().step(1 / 60)


def _mechanism(scene, kind):
    support = scene.create_game_object('Shape edit support')
    support.transform.position = Vector3(0, 2, 0)
    support_body = support.add_component('Rigidbody')
    support_body.is_kinematic = True
    support_body.use_gravity = False
    support.add_component('BoxCollider')
    owner = scene.create_game_object('Shape edit mechanism')
    body = owner.add_component('Rigidbody')
    body.use_gravity = False
    body.drag = body.angular_drag = 0
    owner.add_component('BoxCollider')
    joint = owner.add_component(kind)
    joint.anchor = Vector3(0, 2, 0)
    joint.axis = Vector3(0, 0, 1) if kind == 'HingeJoint' else Vector3(1, 0, 0)
    joint.connected_body = support_body
    manager = SceneManager.instance()
    manager.play()
    manager.pause()
    _step(3)
    return owner, body, joint, support


@pytest.mark.parametrize('kind', ['HingeJoint', 'SliderJoint'])
@pytest.mark.parametrize('side', ['owner', 'support'])
@pytest.mark.parametrize('change', ['center', 'add_shape', 'remove_shape'])
def test_shape_center_of_mass_change_preserves_joint_anchor(scene, kind, side, change):
    owner, body, joint, support = _mechanism(scene, kind)
    edited = owner if side == 'owner' else support
    if change == 'remove_shape':
        sibling = edited.add_component('SphereCollider')
        sibling.center = Vector3(0, .4, .2)
        Physics.sync_transforms()
        _step(60)
    before = owner.transform.transform_point(joint.anchor) - support.transform.position
    if change == 'center':
        edited.get_component('BoxCollider').center = Vector3(0, .4, .2)
    elif change == 'add_shape':
        edited.add_component('SphereCollider').center = Vector3(0, .4, .2)
    else:
        assert edited.remove_component(sibling)
    Physics.sync_transforms()
    _step(120)
    after = owner.transform.transform_point(joint.anchor) - support.transform.position
    # Slider translation along X is unconstrained; Y/Z must remain attached.
    assert (after.y, after.z) == pytest.approx((before.y, before.z), abs=.005)
    if kind == 'HingeJoint':
        assert after.x == pytest.approx(before.x, abs=.005)
