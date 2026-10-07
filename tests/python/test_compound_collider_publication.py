"""Compound membership publishes matching shape, sensor state and query identity."""
import pytest

from infernux.lib import Physics, SceneManager, Vector3


def cast(x, triggers=True):
    return Physics.raycast(Vector3(x, 3, 0), Vector3(0, -1, 0), 8, 0xffffffff, triggers)


def compound(scene, active_index=0):
    owner = scene.create_game_object("Compound membership")
    colliders = [owner.add_component(name) for name in ("BoxCollider", "SphereCollider", "CapsuleCollider")]
    for index, collider in enumerate(colliders):
        collider.center = Vector3(index * 6, 0, 0)
        collider.enabled = index == active_index
    return owner, colliders


@pytest.mark.parametrize("active_index", [0, 1, 2])
@pytest.mark.parametrize("published", [False, True])
@pytest.mark.parametrize("operation", ["remove", "disable"])
def test_last_active_member_retires_geometry_and_reenable_uses_live_identity(scene, active_index, published, operation):
    owner, colliders = compound(scene, active_index)
    if published:
        assert cast(active_index * 6) is not None
    target = colliders[active_index]
    if operation == "remove":
        assert owner.remove_component(target)
    else:
        target.enabled = False
    assert all(cast(index * 6) is None for index in range(3))
    for index, sibling in enumerate(colliders):
        if index == active_index:
            continue
        sibling.enabled = True
        hit = cast(index * 6)
        assert hit is not None
        assert hit.collider.component_id == sibling.component_id
        assert cast(active_index * 6) is None
        sibling.enabled = False
        assert cast(index * 6) is None


@pytest.mark.parametrize("solid_first", [False, True])
@pytest.mark.parametrize("operation", ["enable", "add", "trigger_property"])
def test_trigger_only_to_mixed_compound_accepts_solid_queries(scene, solid_first, operation):
    owner = scene.create_game_object("Sensor publication")
    if solid_first:
        solid = owner.add_component("BoxCollider")
        sensor = owner.add_component("SphereCollider")
    else:
        sensor = owner.add_component("SphereCollider")
        solid = owner.add_component("BoxCollider")
    sensor.center, sensor.is_trigger = Vector3(6, 0, 0), True
    if operation == "add":
        assert owner.remove_component(solid)
    elif operation == "trigger_property":
        solid.is_trigger = True
    else:
        solid.enabled = False
    assert cast(6) is not None and cast(6, False) is None
    for iteration in range(3):
        if operation == "add":
            solid = owner.add_component("BoxCollider")
        elif operation == "trigger_property":
            solid.is_trigger = False
        else:
            solid.enabled = True
        hit = cast(0, False)
        assert hit is not None, (operation, iteration)
        assert hit.collider.component_id == solid.component_id
        assert cast(6, False) is None
        assert cast(6).collider.component_id == sensor.component_id
        if operation == "add":
            assert owner.remove_component(solid)
        elif operation == "trigger_property":
            solid.is_trigger = True
        else:
            solid.enabled = False
        assert cast(0, False) is None
        assert cast(6, False) is None


@pytest.mark.parametrize("operation", ["enable_solid", "remove_last_solid", "disable_last_solid"])
def test_compound_membership_controls_actual_falling_body(scene, operation):
    ground = scene.create_game_object("Mutable ground")
    floor = ground.add_component("BoxCollider")
    floor.size = Vector3(8, 1, 8)
    sensor = ground.add_component("SphereCollider")
    sensor.center, sensor.is_trigger = Vector3(6, 0, 0), True
    if operation == "enable_solid":
        floor.enabled = False
        assert cast(6) is not None and cast(0) is None
        floor.enabled = True
    else:
        sensor.enabled = False
        assert cast(0) is not None
        if operation == "remove_last_solid":
            assert ground.remove_component(floor)
        else:
            floor.enabled = False
    ball = scene.create_game_object("Falling body")
    ball.transform.position = Vector3(0, 3, 0)
    rigidbody = ball.add_component("Rigidbody")
    ball.add_component("SphereCollider")
    manager = SceneManager.instance()
    manager.play()
    manager.pause()
    for _ in range(90):
        manager.step(1 / 60)
    if operation == "enable_solid":
        assert rigidbody.position.y == pytest.approx(1, abs=.08)
    else:
        assert rigidbody.position.y < -3


@pytest.mark.parametrize("operation", ["remove", "disable", "deactivate", "destroy_owner", "move_shape", "remove_member", "keep"])
def test_sleeping_body_wakes_when_last_support_shape_retires(scene, operation):
    ground = scene.create_game_object("Retiring support")
    floor = ground.add_component("BoxCollider")
    floor.size = Vector3(8, 1, 8)
    disabled = ground.add_component("SphereCollider")
    disabled.center = Vector3(6, 0, 0)
    disabled.enabled = False
    ball = scene.create_game_object("Sleeping ball")
    ball.transform.position = Vector3(0, 2, 0)
    body = ball.add_component("Rigidbody")
    ball.add_component("SphereCollider")
    manager = SceneManager.instance()
    manager.play()
    manager.pause()
    for _ in range(240):
        manager.step(1 / 60)
    assert body.position.y == pytest.approx(1, abs=.08)
    body.sleep()
    assert body.is_sleeping()
    if operation == "remove_member":
        disabled.enabled = True
        assert ground.remove_component(floor)
    elif operation == "remove":
        assert ground.remove_component(floor)
    elif operation == "disable":
        floor.enabled = False
    elif operation == "deactivate":
        ground.active = False
    elif operation == "destroy_owner":
        scene.destroy_game_object(ground)
        scene.process_pending_destroys()
    elif operation == "move_shape":
        floor.center = Vector3(12, 0, 0)
    for _ in range(60):
        manager.step(1 / 60)
    if operation == "keep":
        assert body.position.y == pytest.approx(1, abs=.08)
        assert body.is_sleeping()
    else:
        assert body.position.y < -2


def test_expanding_shape_wakes_sleepers_in_new_bounds(scene):
    ground = scene.create_game_object("Expanded geometry")
    shape = ground.add_component("BoxCollider")
    shape.center = Vector3(-12, 0, 0)
    assert cast(-12) is not None
    ball = scene.create_game_object("Sleeping obstacle")
    body = ball.add_component("Rigidbody")
    body.use_gravity = False
    ball.add_component("SphereCollider")
    manager = SceneManager.instance()
    manager.play()
    manager.pause()
    manager.step(1 / 60)
    body.sleep()
    assert body.is_sleeping()
    shape.center = Vector3(0, 0, 0)
    assert not body.is_sleeping()


@pytest.mark.parametrize("query", ["raycast_all", "sphere_cast", "overlap_sphere"])
def test_mixed_member_filtering_is_consistent_across_queries(scene, query):
    owner = scene.create_game_object("Query compound")
    solid = owner.add_component("BoxCollider")
    sensor = owner.add_component("SphereCollider")
    sensor.center, sensor.is_trigger = Vector3(6, 0, 0), True
    solid.enabled = False
    assert cast(6) is not None

    def ids(x, triggers):
        if query == "raycast_all":
            hits = Physics.raycast_all(Vector3(x, 3, 0), Vector3(0, -1, 0), 8, 0xffffffff, triggers)
            return [hit.collider.component_id for hit in hits]
        if query == "sphere_cast":
            hit = Physics.sphere_cast(Vector3(x, 3, 0), .1, Vector3(0, -1, 0), 8, 0xffffffff, triggers)
            return [] if hit is None else [hit.collider.component_id]
        return [collider.component_id for collider in Physics.overlap_sphere(Vector3(x, 0, 0), .2, 0xffffffff, triggers)]

    for _ in range(3):
        solid.enabled = True
        assert ids(0, False) == [solid.component_id]
        assert ids(6, False) == []
        assert ids(6, True) == [sensor.component_id]
        solid.enabled = False
        assert ids(0, True) == []
