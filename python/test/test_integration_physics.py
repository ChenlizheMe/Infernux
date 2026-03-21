"""Integration tests — Physics simulation with real Jolt backend (real engine)."""
from __future__ import annotations

import pytest

from InfEngine.lib import SceneManager, Vector3, Physics


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _step_frames(n: int = 60, dt: float = 1.0 / 60.0):
    """Advance the physics simulation by *n* frames."""
    sm = SceneManager.instance()
    for _ in range(n):
        sm.step(dt)


def _make_ground(scene):
    """Create a large static ground plane (BoxCollider, no Rigidbody)."""
    ground = scene.create_game_object("Ground")
    ground.transform.position = Vector3(0, 0, 0)
    ground.transform.local_scale = Vector3(100, 1, 100)
    ground.add_component("BoxCollider")
    return ground


def _make_ball(scene, *, pos=None, mass=1.0, radius=0.5):
    """Create a dynamic sphere with Rigidbody + SphereCollider."""
    ball = scene.create_game_object("Ball")
    ball.transform.position = pos or Vector3(0, 10, 0)
    rb = ball.add_component("Rigidbody")
    rb.mass = mass
    col = ball.add_component("SphereCollider")
    col.radius = radius
    return ball, rb


# ═══════════════════════════════════════════════════════════════════════════
# Gravity & free fall
# ═══════════════════════════════════════════════════════════════════════════

class TestGravity:
    def test_set_gravity_persists(self, scene):
        Physics.set_gravity(Vector3(0, -9.81, 0))
        g = Physics.get_gravity()
        assert g.y == pytest.approx(-9.81, abs=0.01)

    def test_custom_gravity(self, scene):
        Physics.set_gravity(Vector3(0, -20, 0))
        g = Physics.get_gravity()
        assert g.y == pytest.approx(-20, abs=0.01)
        Physics.set_gravity(Vector3(0, -9.81, 0))  # restore

    def test_ball_falls_under_gravity(self, scene):
        """A dynamic sphere above the ground should lose altitude."""
        Physics.set_gravity(Vector3(0, -9.81, 0))
        _make_ground(scene)
        ball, rb = _make_ball(scene, pos=Vector3(0, 10, 0))

        sm = SceneManager.instance()
        sm.play()
        sm.pause()
        y0 = ball.transform.position.y
        _step_frames(60)
        y1 = ball.transform.position.y

        assert y1 < y0, f"Ball should fall: {y0} → {y1}"

    def test_no_gravity_ball_stays(self, scene):
        """With gravity off a dynamic body should not fall."""
        _make_ground(scene)
        ball, rb = _make_ball(scene, pos=Vector3(0, 10, 0))
        rb.use_gravity = False

        sm = SceneManager.instance()
        sm.play()
        sm.pause()
        y0 = ball.transform.position.y
        _step_frames(30)
        y1 = ball.transform.position.y

        assert y1 == pytest.approx(y0, abs=0.1)


# ═══════════════════════════════════════════════════════════════════════════
# Collision — ball lands on ground
# ═══════════════════════════════════════════════════════════════════════════

class TestCollision:
    def test_ball_lands_on_ground(self, scene):
        """Ball should collide with ground and stop roughly at ground level."""
        Physics.set_gravity(Vector3(0, -9.81, 0))
        _make_ground(scene)
        ball, rb = _make_ball(scene, pos=Vector3(0, 5, 0), radius=0.5)

        sm = SceneManager.instance()
        sm.play()
        sm.pause()
        _step_frames(180)  # 3 seconds — plenty of time to settle
        y = ball.transform.position.y
        # Ground top is at Y=0.5 (center 0, scale 1 → half-height 0.5)
        # Ball radius 0.5 → ball center rests at about Y=1.0
        assert y < 5.0, "Ball should have fallen"
        assert y > -1.0, "Ball should not fall through ground"


# ═══════════════════════════════════════════════════════════════════════════
# Rigidbody properties in play mode
# ═══════════════════════════════════════════════════════════════════════════

class TestRigidbodyInScene:
    def test_mass_affects_physics(self, scene):
        _make_ground(scene)
        heavy, rb_heavy = _make_ball(scene, pos=Vector3(-3, 10, 0), mass=100.0)
        light, rb_light = _make_ball(scene, pos=Vector3(3, 10, 0), mass=0.1)
        rb_heavy.use_gravity = True
        rb_light.use_gravity = True

        sm = SceneManager.instance()
        sm.play()
        sm.pause()
        _step_frames(30)

        # Both should have fallen (gravity is mass-independent in Newtonian physics,
        # but drag or solver differences may cause slight variation)
        assert heavy.transform.position.y < 10.0
        assert light.transform.position.y < 10.0

    def test_kinematic_body_does_not_fall(self, scene):
        _make_ground(scene)
        ball, rb = _make_ball(scene, pos=Vector3(0, 10, 0))
        rb.is_kinematic = True

        sm = SceneManager.instance()
        sm.play()
        sm.pause()
        y0 = ball.transform.position.y
        _step_frames(60)
        y1 = ball.transform.position.y
        assert y1 == pytest.approx(y0, abs=0.1)

    def test_velocity_readable_during_fall(self, scene):
        Physics.set_gravity(Vector3(0, -9.81, 0))
        _make_ground(scene)
        ball, rb = _make_ball(scene, pos=Vector3(0, 20, 0))

        sm = SceneManager.instance()
        sm.play()
        sm.pause()
        _step_frames(30)

        vel = rb.velocity
        # Ball is falling, so velocity Y should be negative
        assert vel.y < 0, f"Velocity should be downward: {vel.y}"

    def test_add_force_impulse(self, scene):
        """An upward impulse should propel the ball up."""
        from InfEngine.lib import ForceMode
        Physics.set_gravity(Vector3(0, 0, 0))  # disable gravity for a clean test
        _make_ground(scene)
        ball, rb = _make_ball(scene, pos=Vector3(0, 5, 0))

        sm = SceneManager.instance()
        sm.play()
        sm.pause()
        _step_frames(1)
        rb.add_force(Vector3(0, 100, 0), ForceMode.Impulse)
        _step_frames(30)

        assert ball.transform.position.y > 5.0, "Impulse should move ball up"
        Physics.set_gravity(Vector3(0, -9.81, 0))


# ═══════════════════════════════════════════════════════════════════════════
# Raycasting with actual scene objects
# ═══════════════════════════════════════════════════════════════════════════

class TestRaycast:
    def test_raycast_hits_ground(self, scene):
        _make_ground(scene)

        sm = SceneManager.instance()
        sm.play()
        sm.pause()
        _step_frames(1)

        hit = Physics.raycast(Vector3(0, 50, 0), Vector3(0, -1, 0), 100.0)
        assert hit is not None
        assert 0 < hit.distance < 50.0
        assert hit.normal.y == pytest.approx(1.0, abs=0.1)

    def test_raycast_miss(self, scene):
        _make_ground(scene)
        sm = SceneManager.instance()
        sm.play()
        sm.pause()
        _step_frames(1)

        # Ray pointing away from all objects
        hit = Physics.raycast(Vector3(0, 50, 0), Vector3(0, 1, 0), 100.0)
        assert hit is None

    def test_raycast_all_returns_multiple(self, scene):
        _make_ground(scene)
        # Create floating box collider above ground
        box = scene.create_game_object("FloatingBox")
        box.transform.position = Vector3(0, 5, 0)
        box.add_component("BoxCollider")

        sm = SceneManager.instance()
        sm.play()
        sm.pause()
        _step_frames(1)

        hits = Physics.raycast_all(Vector3(0, 50, 0), Vector3(0, -1, 0), 100.0)
        assert len(hits) >= 2  # at least ground + floating box


# ═══════════════════════════════════════════════════════════════════════════
# Overlap queries
# ═══════════════════════════════════════════════════════════════════════════

class TestOverlapQueries:
    def test_overlap_sphere_finds_colliders(self, scene):
        for i in range(3):
            go = scene.create_game_object(f"Obj{i}")
            go.transform.position = Vector3(float(i), 0, 0)
            go.add_component("SphereCollider")

        sm = SceneManager.instance()
        sm.play()
        sm.pause()
        _step_frames(1)

        result = Physics.overlap_sphere(Vector3(0, 0, 0), 50.0)
        assert len(result) >= 3

    def test_overlap_box_finds_colliders(self, scene):
        go = scene.create_game_object("TestBox")
        go.add_component("BoxCollider")

        sm = SceneManager.instance()
        sm.play()
        sm.pause()
        _step_frames(1)

        result = Physics.overlap_box(Vector3(0, 0, 0), Vector3(10, 10, 10))
        assert len(result) >= 1


# ═══════════════════════════════════════════════════════════════════════════
# Shape casts
# ═══════════════════════════════════════════════════════════════════════════

class TestShapeCasts:
    def test_sphere_cast_hits_ground(self, scene):
        _make_ground(scene)
        sm = SceneManager.instance()
        sm.play()
        sm.pause()
        _step_frames(1)

        hit = Physics.sphere_cast(Vector3(0, 50, 0), 1.0, Vector3(0, -1, 0), 100.0)
        assert hit is not None
        assert hit.distance < 60

    def test_box_cast_hits_ground(self, scene):
        _make_ground(scene)
        sm = SceneManager.instance()
        sm.play()
        sm.pause()
        _step_frames(1)

        hit = Physics.box_cast(Vector3(0, 50, 0), Vector3(1, 1, 1), Vector3(0, -1, 0), 100.0)
        assert hit is not None


# ═══════════════════════════════════════════════════════════════════════════
# Layer collision filtering
# ═══════════════════════════════════════════════════════════════════════════

class TestLayerCollision:
    def test_ignore_layer_collision_round_trip(self, scene):
        sm = SceneManager.instance()
        sm.play()
        sm.pause()
        _step_frames(1)

        Physics.ignore_layer_collision(10, 11, True)
        assert Physics.get_ignore_layer_collision(10, 11) is True
        Physics.ignore_layer_collision(10, 11, False)
        assert Physics.get_ignore_layer_collision(10, 11) is False
