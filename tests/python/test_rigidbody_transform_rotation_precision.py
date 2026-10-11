"""Small authored rotations and an animation's final pose reach real Jolt bodies."""
import pytest

from infernux.lib import SceneManager, Vector3, quatf


def assert_orientation(actual, expected):
    # Quaternion components reveal sub-degree errors hidden by a rounded dot product.
    sign = 1 if sum(a * b for a, b in zip(actual, expected)) >= 0 else -1
    assert tuple(actual) == pytest.approx(tuple(sign * value for value in expected), abs=2e-6)


@pytest.mark.parametrize('kinematic', [False, True])
@pytest.mark.parametrize('degrees', [0.01, 0.1, 0.6, 1.2])
def test_small_authored_rotation_survives_physics_publication(scene, kinematic, degrees):
    owner = scene.create_game_object('Precise authored rotation')
    owner.add_component('BoxCollider')
    body = owner.add_component('Rigidbody')
    body.is_kinematic = kinematic
    body.use_gravity = False
    manager = SceneManager.instance()
    manager.play()
    manager.pause()
    dt = manager.get_fixed_time_step()
    manager.step(dt)

    for target in (degrees, 0.0):
        owner.transform.local_euler_angles = Vector3(0, target, 0)
        expected = quatf.euler(0, target, 0)
        manager.step(dt)
        assert_orientation(body.rotation, expected)
        assert_orientation(owner.transform.rotation, expected)
        manager.step(dt)
        assert_orientation(owner.transform.rotation, expected)


def test_kinematic_bridge_animation_keeps_its_exact_final_rotation(scene):
    owner = scene.create_game_object('Rotating puzzle bridge')
    owner.transform.local_euler_angles = Vector3(0, 90, 0)
    owner.add_component('BoxCollider')
    body = owner.add_component('Rigidbody')
    body.is_kinematic = True
    body.use_gravity = False
    manager = SceneManager.instance()
    manager.play()
    manager.pause()
    dt = manager.get_fixed_time_step()
    manager.step(dt)
    for index in range(75):
        owner.transform.local_euler_angles = Vector3(0, max(0, 90 - (index + 1) * 1.2), 0)
        manager.step(dt)
    for _ in range(8):
        manager.step(dt)
        assert_orientation(body.rotation, quatf.identity())
        assert_orientation(owner.transform.rotation, quatf.identity())
