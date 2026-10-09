"""Geometry publication must retain a joint's authored attachment points."""
import pytest

from infernux.lib import Physics, SceneManager, Vector3


def _step(count):
    for _ in range(count):
        SceneManager.instance().step(1 / 60)


def _mechanism(scene, kind, initial_scale=(1, 1, 1), support_scale=(1, 1, 1), support_position=(0, 2, 0)):
    support = scene.create_game_object('Shape edit support')
    support.transform.position = Vector3(*support_position)
    support.transform.local_scale = Vector3(*support_scale)
    support_body = support.add_component('Rigidbody')
    support_body.is_kinematic = True
    support_body.use_gravity = False
    support.add_component('BoxCollider')
    owner = scene.create_game_object('Shape edit mechanism')
    owner.transform.local_scale = Vector3(*initial_scale)
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


@pytest.mark.parametrize('kind', ['HingeJoint', 'SliderJoint'])
@pytest.mark.parametrize('side', ['owner', 'support', 'world'])
@pytest.mark.parametrize('sleeping', [False, True])
@pytest.mark.parametrize('parent_scale', [False, True])
def test_runtime_scale_preserves_each_joint_attachment(scene, kind, side, sleeping, parent_scale):
    owner, body, joint, support = _mechanism(scene, kind)
    # The connection is away from both origins; scaling either end must move
    # that end's local anchor, without changing the other end's attachment.
    joint.connected_body = None
    support.transform.position = Vector3(0, 1, 0)
    Physics.sync_transforms()
    joint.connected_body = support.get_component('Rigidbody')
    _step(90)
    connected_anchor = support.transform.inverse_transform_point(owner.transform.transform_point(joint.anchor))
    if side == 'world':
        joint.connected_body = None
    fixed_anchor = owner.transform.transform_point(joint.anchor)
    edited = support if side == 'support' else owner
    if parent_scale:
        parent = scene.create_game_object('Scale parent')
        edited.transform.set_parent(parent.transform, True)
        edited = parent
    if sleeping:
        body.sleep()
        assert body.is_sleeping()
    for scale in (Vector3(1.3, 1.7, .8), Vector3(.6, .9, 1.2), Vector3(1, 1, 1)):
        edited.transform.local_scale = scale
        Physics.sync_transforms()
        _step(150)
        expected = fixed_anchor if side == 'world' else support.transform.transform_point(connected_anchor)
        error = owner.transform.transform_point(joint.anchor) - expected
        assert (error.y, error.z) == pytest.approx((0, 0), abs=.005)
        if kind == 'HingeJoint':
            assert error.x == pytest.approx(0, abs=.005)


@pytest.mark.parametrize('kind', ['HingeJoint', 'SliderJoint'])
def test_scale_preserves_joint_reference_frame_and_limits(scene, kind):
    owner, body, joint, support = _mechanism(scene, kind)
    joint.use_limits = True
    if kind == 'HingeJoint':
        joint.minimum_angle = -20
        joint.maximum_angle = 20
        body.angular_velocity = Vector3(0, 0, 1.2)
    else:
        joint.minimum_distance = -.5
        joint.maximum_distance = .5
        body.velocity = Vector3(1, 0, 0)
    _step(10)
    field = 'current_angle' if kind == 'HingeJoint' else 'current_position'
    before = getattr(joint, field)
    assert abs(before) > .05
    owner.get_component('BoxCollider').center = Vector3(.1, .2, .3)
    owner.transform.local_scale = Vector3(1.3, 1.7, .8)
    Physics.sync_transforms()
    assert getattr(joint, field) == pytest.approx(before, abs=.001)
    # A discontinuous 1.4 m anchor relocation with an offset COM needs the
    # position solver to settle; this check bounds the converged error.
    _step(600)
    assert abs(getattr(joint, field)) <= (20.5 if kind == 'HingeJoint' else .505)
    error = owner.transform.transform_point(joint.anchor) - support.transform.position
    assert (error.y, error.z) == pytest.approx((0, 0), abs=.005)


@pytest.mark.parametrize('kind', ['HingeJoint', 'SliderJoint'])
@pytest.mark.parametrize('initial_y', [0, -1])
def test_joint_retains_authored_anchor_when_growing_from_collapsed_or_mirrored_scale(scene, kind, initial_y):
    owner, body, joint, support = _mechanism(scene, kind, (1, initial_y, 1))
    before = owner.transform.transform_point(joint.anchor)
    owner.transform.local_scale = Vector3(1, 1, 1)
    Physics.sync_transforms()
    _step(300)
    after = owner.transform.transform_point(joint.anchor)
    assert (after.y, after.z) == pytest.approx((before.y, before.z), abs=.005)


@pytest.mark.parametrize('kind', ['HingeJoint', 'SliderJoint'])
def test_collapsed_connected_body_retains_its_physical_anchor(scene, kind):
    owner, body, joint, support = _mechanism(scene, kind, support_scale=(1, 0, 1), support_position=(0, 1, 0))
    before = owner.transform.transform_point(joint.anchor)
    # The connected anchor is inferred from the actual bound physics point.
    # A collapsed axis has no inverse; expanding it must retain that offset.
    support.transform.local_scale = Vector3(1, 1, 1)
    Physics.sync_transforms()
    _step(300)
    after = owner.transform.transform_point(joint.anchor)
    assert (after.y, after.z) == pytest.approx((before.y, before.z), abs=.005)
