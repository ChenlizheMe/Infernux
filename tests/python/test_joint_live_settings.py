"""Editing a joint's limits or collision filter must not rebind its frames."""
import pytest

from infernux.lib import Vector3
from tests.python.test_joint_shape_publication import _mechanism, _step


@pytest.mark.parametrize('kind', ['HingeJoint', 'SliderJoint'])
@pytest.mark.parametrize('setting', ['use_limits', 'minimum', 'maximum', 'enable_collision'])
def test_live_joint_settings_retain_angle_or_displacement_reference(scene, kind, setting):
    owner, body, joint, support = _mechanism(scene, kind)
    joint.use_limits = True
    if kind == 'HingeJoint':
        joint.minimum_angle = -30
        joint.maximum_angle = 30
        body.angular_velocity = Vector3(0, 0, 1.2)
    else:
        joint.minimum_distance = -1
        joint.maximum_distance = 1
        body.velocity = Vector3(1, 0, 0)
    _step(10)
    field = 'current_angle' if kind == 'HingeJoint' else 'current_position'
    before = getattr(joint, field)
    assert abs(before) > .05
    if setting in ('use_limits', 'enable_collision'):
        setattr(joint, setting, not getattr(joint, setting))
    else:
        limit = setting + ('_angle' if kind == 'HingeJoint' else '_distance')
        setattr(joint, limit, getattr(joint, limit) * .8)
    assert getattr(joint, field) == pytest.approx(before, abs=.0001)
