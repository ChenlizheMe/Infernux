"""Position and rotation commands compose in one kinematic fixed-step pose."""

import math
import pytest

import infernux as inx
from infernux.lib import SceneManager


@pytest.mark.parametrize('order', ['position-first', 'rotation-first'])
def test_kinematic_position_and_rotation_compose_across_fixed_steps(scene, order):
    owner = scene.create_game_object('ComposedKinematicPose')
    owner.transform.position = inx.Vector3(0, 5, 0)
    owner.add_component(inx.SphereCollider)
    body = owner.add_component(inx.Rigidbody)
    body.is_kinematic = True

    class PoseDriver(inx.InxComponent):
        def awake(self):
            self.steps = 0

        def fixed_update(self, delta_time):
            if self.steps == 3:
                return
            self.steps += 1
            position = inx.Vector3(self.steps, 5, 0)
            rotation = inx.quatf(0, math.sin(self.steps * .1), 0, math.cos(self.steps * .1))
            if order == 'position-first':
                body.move_position(position)
                body.move_rotation(rotation)
            else:
                body.move_rotation(rotation)
                body.move_position(position)

    driver = owner.add_component(PoseDriver)
    manager = SceneManager.instance()
    manager.play()
    manager.pause()
    for step in range(1, 4):
        manager.step(.02)
        assert driver.steps == step
        assert tuple(body.position) == pytest.approx((step, 5, 0), abs=.001)
        assert tuple(owner.transform.position) == pytest.approx((step, 5, 0), abs=.001)
        rotation = tuple(body.rotation)
        assert abs(rotation[1]) == pytest.approx(math.sin(step * .1), abs=.001)
        assert abs(rotation[3]) == pytest.approx(math.cos(step * .1), abs=.001)
    manager.step(.02)
    assert tuple(body.position) == pytest.approx((3, 5, 0), abs=.001)
    assert tuple(body.velocity) == pytest.approx((0, 0, 0), abs=.001)
