"""Sleeping must preserve visible joint motion while retiring bodies at rest."""
import math

import pytest

from infernux.lib import ForceMode, SceneManager, Vector3


def _pendulum(scene, wide=False, mass=1, angle=0, limited=False):
    owner = scene.create_game_object('Sleep accuracy pendulum')
    owner.transform.position = Vector3(0, 3, 0)
    owner.transform.local_scale = Vector3(2.418, 2.27, .35) if wide else Vector3(.35, 1.5, .35)
    owner.transform.euler_angles = Vector3(0, 0, angle)
    owner.add_component('BoxCollider')
    body = owner.add_component('Rigidbody')
    body.mass = mass
    body.angular_drag = .1
    joint = owner.add_component('HingeJoint')
    joint.anchor = Vector3(0, .75, 0)
    joint.axis = Vector3(0, 0, 1)
    joint.use_limits = limited
    joint.minimum_angle = -10
    joint.maximum_angle = 10
    return owner, body, joint


@pytest.mark.parametrize('rate', [30, 60, 120])
@pytest.mark.parametrize('wide', [False, True])
@pytest.mark.parametrize('mass', [1, 20])
def test_free_pendulum_does_not_sleep_at_visible_turning_point(scene, rate, wide, mass):
    owner, body, joint = _pendulum(scene, wide, mass)
    manager = SceneManager.instance()
    previous = manager.get_fixed_time_step()
    dt = 1 / rate
    manager.set_fixed_time_step(dt)
    manager.play()
    manager.pause()
    try:
        manager.step(dt)
        body.angular_velocity = Vector3(0, 0, 1.2)
        observed_angle = 0
        for _ in range(rate * 20):
            manager.step(dt)
            angle = abs(joint.current_angle)
            observed_angle = max(observed_angle, angle)
            assert math.isfinite(angle)
            if body.is_sleeping():
                # This is a visual accuracy bound, not a requirement to keep
                # every microscopic oscillation alive indefinitely.
                assert angle < .5, f'pendulum froze {angle:.3f} degrees away from equilibrium'
                break
        assert observed_angle > 2
    finally:
        manager.set_fixed_time_step(previous)


def test_supported_body_grid_retires_without_disabling_sleep(scene):
    floor = scene.create_game_object('Sleep grid floor')
    floor.transform.position = Vector3(0, -.5, 0)
    floor.transform.local_scale = Vector3(24, 1, 24)
    floor.add_component('BoxCollider')
    bodies = []
    for index in range(64):
        obj = scene.create_game_object(f'Rest control {index}')
        obj.transform.position = Vector3((index % 8 - 3.5) * 1.5, .5, (index // 8 - 3.5) * 1.5)
        obj.add_component('BoxCollider')
        body = obj.add_component('Rigidbody')
        body.mass = 1 if index % 2 else 20
        bodies.append(body)
    manager = SceneManager.instance()
    previous = manager.get_fixed_time_step()
    manager.set_fixed_time_step(1 / 60)
    manager.play()
    manager.pause()
    try:
        for _ in range(180):
            manager.step(1 / 60)
        assert all(body.is_sleeping() for body in bodies)
        before = [body.position for body in bodies]
        bodies[0].add_force(Vector3(0, 1, 0), ForceMode.Impulse)
        for _ in range(6):
            manager.step(1 / 60)
        assert all(body.is_sleeping() for body in bodies[1:])
        assert all((body.position - position).magnitude < .0001
                   for body, position in zip(bodies[1:], before[1:]))
    finally:
        manager.set_fixed_time_step(previous)


@pytest.mark.parametrize('kind', ['pendulum', 'hinge_limit', 'floor_contact'])
def test_stationary_mechanisms_still_sleep_and_wake_on_impulse(scene, kind):
    if kind == 'floor_contact':
        floor = scene.create_game_object('Sleep support floor')
        floor.transform.position = Vector3(0, -.5, 0)
        floor.transform.local_scale = Vector3(10, 1, 10)
        floor.add_component('BoxCollider')
        owner = scene.create_game_object('Sleeping box')
        owner.transform.position = Vector3(0, .5, 0)
        owner.add_component('BoxCollider')
        body = owner.add_component('Rigidbody')
    else:
        owner, body, joint = _pendulum(scene, angle=45 if kind == 'hinge_limit' else 0,
                                      limited=kind == 'hinge_limit')
    manager = SceneManager.instance()
    previous = manager.get_fixed_time_step()
    dt = 1 / 60
    manager.set_fixed_time_step(dt)
    manager.play()
    manager.pause()
    try:
        for _ in range(600):
            manager.step(dt)
            if body.is_sleeping():
                break
        assert body.is_sleeping(), f'{kind} failed to retire after coming to rest'
        before = body.position
        for _ in range(60):
            manager.step(dt)
        assert (body.position - before).magnitude < .0001
        body.add_force(Vector3(0, 1, 0) if kind == 'floor_contact' else Vector3(2, 0, 0), ForceMode.Impulse)
        assert not body.is_sleeping()
        for _ in range(6):
            manager.step(dt)
        assert (body.position - before).magnitude > .0001
    finally:
        manager.set_fixed_time_step(previous)
