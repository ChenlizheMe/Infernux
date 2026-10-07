"""Physics callbacks and queries return the canonical public component objects."""

import pytest

import infernux as inx
from infernux.lib import SceneManager, Vector3
from infernux.lib import CollisionInfo, RaycastHit
from infernux.physics import Physics


class PublicPhysicsProbe(inx.InxComponent):
    def awake(self):
        self.events = []

    def on_trigger_enter(self, other):
        self.events.append(('trigger-enter', other))
    def on_trigger_stay(self, other):
        self.events.append(('trigger-stay', other))
    def on_trigger_exit(self, other):
        self.events.append(('trigger-exit', other))
    def on_collision_enter(self, collision):
        self.events.append(('collision-enter', collision.collider))
    def on_collision_stay(self, collision):
        self.events.append(('collision-stay', collision.collider))
    def on_collision_exit(self, collision):
        self.events.append(('collision-exit', collision.collider))


@pytest.mark.parametrize('shape', [inx.BoxCollider, inx.SphereCollider, inx.CapsuleCollider, inx.CylinderCollider])
@pytest.mark.parametrize('phase', ['enter', 'stay', 'exit'])
def test_trigger_bridge_reuses_exact_public_collider(scene, shape, phase):
    owner = scene.create_game_object('CallbackCollider')
    expected = owner.add_component(shape)
    listener = scene.create_game_object('CallbackListener').add_component(PublicPhysicsProbe)
    getattr(listener, '_call_on_trigger_' + phase)(expected._require_cpp_component())
    name, received = listener.events[-1]
    assert name == 'trigger-' + phase
    assert received is expected is owner.get_component(shape)
    assert isinstance(received, inx.Collider)
    assert received.game_object is owner


def test_real_contact_callbacks_reuse_public_collider_through_all_phases(scene):
    ground = scene.create_game_object('CanonicalGround')
    ground.transform.position = Vector3(0, -.5, 0)
    ground.transform.local_scale = Vector3(20, 1, 20)
    expected = ground.add_component(inx.BoxCollider)
    ball = scene.create_game_object('CanonicalBall')
    ball.transform.position = Vector3(0, 2, 0)
    ball.add_component(inx.SphereCollider)
    body = ball.add_component(inx.Rigidbody)
    listener = ball.add_component(PublicPhysicsProbe)
    manager = SceneManager.instance()
    manager.play()
    manager.pause()
    for _ in range(100):
        manager.step(.02)
    body.velocity = Vector3(0, 15, 0)
    for _ in range(15):
        manager.step(.02)
    events = [(name, collider) for name, collider in listener.events if name.startswith('collision-')]
    assert {name for name, _ in events} == {'collision-enter', 'collision-stay', 'collision-exit'}
    assert all(collider is expected for _, collider in events)
    assert all(isinstance(collider, inx.Collider) for _, collider in events)


@pytest.mark.parametrize('query', ['raycast', 'sphere_cast', 'box_cast', 'capsule_cast', 'raycast_all'])
def test_query_hit_exposes_same_public_collider_as_component_lookup(scene, query):
    owner = scene.create_game_object('CanonicalQueryTarget')
    owner.transform.position = Vector3(0, 0, 5)
    expected = owner.add_component(inx.BoxCollider)
    calls = {
        'raycast': lambda: Physics.raycast((0, 0, 0), (0, 0, 1), 10),
        'sphere_cast': lambda: Physics.sphere_cast((0, 0, 0), .2, (0, 0, 1), 10),
        'box_cast': lambda: Physics.box_cast((0, 0, 0), (.2, .2, .2), (0, 0, 1), max_distance=10),
        'capsule_cast': lambda: Physics.capsule_cast((0, -.2, 0), (0, .2, 0), .2, (0, 0, 1), 10),
        'raycast_all': lambda: Physics.raycast_all((0, 0, 0), (0, 0, 1), 10),
    }
    result = calls[query]()
    hits = result if query == 'raycast_all' else [result]
    assert len(hits) == 1 and hits[0] is not None
    assert hits[0].collider is expected
    assert isinstance(hits[0].collider, inx.Collider)
    assert hits[0].collider.game_object is owner


@pytest.mark.parametrize('query', ['overlap_sphere', 'overlap_box', 'overlap_capsule'])
def test_overlap_queries_return_public_components_with_stable_identity(scene, query):
    owner = scene.create_game_object('CanonicalOverlapTarget')
    expected = owner.add_component(inx.BoxCollider)
    calls = {
        'overlap_sphere': lambda: Physics.overlap_sphere((0, 0, 0), 1),
        'overlap_box': lambda: Physics.overlap_box((0, 0, 0), (1, 1, 1)),
        'overlap_capsule': lambda: Physics.overlap_capsule((0, -.5, 0), (0, .5, 0), 1),
    }
    assert calls[query]() == [expected]
    assert calls[query]()[0] is expected


@pytest.mark.parametrize('phase', ['enter', 'stay', 'exit'])
def test_trigger_component_identity_does_not_select_first_same_type_collider(scene, phase):
    owner = scene.create_game_object('CompoundCallbackOwner')
    first = owner.add_component(inx.BoxCollider)
    second = owner.add_component(inx.BoxCollider)
    listener = scene.create_game_object('CompoundCallbackListener').add_component(PublicPhysicsProbe)
    getattr(listener, '_call_on_trigger_' + phase)(second._require_cpp_component())
    assert listener.events[-1][1] is second
    assert listener.events[-1][1] is not first
    assert owner.get_components(inx.BoxCollider) == [first, second]


@pytest.mark.parametrize('value_type', [CollisionInfo, RaycastHit])
def test_empty_native_value_has_no_public_collider(value_type):
    assert value_type().collider is None
