"""Collision normals point from the other collider towards the receiver."""

import pytest

from infernux.components import InxComponent
from infernux.lib import SceneManager, Vector3


class ContactNormalProbe(InxComponent):
    def awake(self):
        self.contacts = []

    def on_collision_enter(self, collision):
        self.contacts.append(('enter', collision.contact_normal.to_tuple()))

    def on_collision_stay(self, collision):
        self.contacts.append(('stay', collision.contact_normal.to_tuple()))


@pytest.mark.parametrize('ball_first', [False, True])
def test_floor_contact_normal_matches_receiver_independent_of_body_order(scene, ball_first):
    objects = {}
    for name in (('Ball', 'Floor') if ball_first else ('Floor', 'Ball')):
        owner = scene.create_game_object(name)
        if name == 'Ball':
            owner.transform.position = Vector3(0, 2, 0)
            owner.add_component('SphereCollider')
            owner.add_component('Rigidbody')
        else:
            owner.transform.position = Vector3(0, -.5, 0)
            owner.transform.local_scale = Vector3(20, 1, 20)
            owner.add_component('BoxCollider')
        objects[name] = owner.add_component(ContactNormalProbe)
    manager = SceneManager.instance()
    manager.play()
    manager.pause()
    for _ in range(60):
        manager.step(.02)
    for name, direction in [('Ball', 1), ('Floor', -1)]:
        contacts = objects[name].contacts
        assert {phase for phase, _ in contacts} == {'enter', 'stay'}, (name, contacts)
        for phase, normal in contacts:
            assert normal == pytest.approx((0, direction, 0), abs=.001), (name, phase, normal)
