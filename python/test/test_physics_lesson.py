"""Run the learning scene against the real Jolt solver and callback bridge."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from Infernux.components import InxComponent
from Infernux.lib import RigidbodyConstraints, SceneManager, Vector3


@pytest.mark.parametrize("legacy_force", [True, False], ids=["old-3N", "documented-force"])
def test_locked_sphere_lesson_reaches_sensor_only_above_sliding_friction(scene, legacy_force):
    lesson = Path(__file__).resolve().parents[2] / "docs/learn/gameplay-physics-events.md"
    source = lesson.read_text(encoding="utf-8")
    forces = re.findall(r"add_force\(inx.Vector3\(([\d.]+), 0\.0, 0\.0\)\)", source)
    assert len(forces) == 2 and forces[0] == forces[1], "English and Chinese examples must agree"
    force = 3.0 if legacy_force else float(forces[0])
    class LessonProbe(InxComponent):
        def awake(self):
            self.events = []

        def start(self):
            self.body = self.game_object.get_component("Rigidbody")

        def fixed_update(self, fixed_delta_time):
            self.body.add_force(Vector3(force, 0, 0))

        def on_collision_enter(self, other):
            self.events.append(("collision_enter", other.game_object.name))

        def on_collision_stay(self, other):
            self.events.append(("collision_stay", other.game_object.name))

        def on_collision_exit(self, other):
            self.events.append(("collision_exit", other.game_object.name))

        def on_trigger_enter(self, other):
            self.events.append(("trigger_enter", other.game_object.name))

        def on_trigger_stay(self, other):
            self.events.append(("trigger_stay", other.game_object.name))

        def on_trigger_exit(self, other):
            self.events.append(("trigger_exit", other.game_object.name))

    ground = scene.create_game_object("Ground")
    ground.transform.position = Vector3(0, -0.5, 0)
    ground.transform.local_scale = Vector3(12, 1, 4)
    ground.add_component("BoxCollider")
    sensor = scene.create_game_object("Sensor")
    sensor.transform.position = Vector3(2, 0.75, 0)
    sensor.transform.local_scale = Vector3(1, 1.5, 4)
    sensor.add_component("BoxCollider").is_trigger = True
    probe = scene.create_game_object("Probe")
    probe.transform.position = Vector3(-4, 2, 0)
    probe.add_component("SphereCollider")
    body = probe.add_component("Rigidbody")
    body.mass = 1.0
    body.drag = 0.0
    body.constraints = int(RigidbodyConstraints.FreezePositionZ) | int(RigidbodyConstraints.FreezeRotation)
    script = probe.add_component(LessonProbe)
    manager = SceneManager.instance()
    manager.play()
    manager.pause()
    for _ in range(250):
        manager.step(0.02)
    assert ("collision_enter", "Ground") in script.events
    assert ("collision_stay", "Ground") in script.events
    if legacy_force:
        assert body.position.x < 0
        assert abs(body.velocity.x) < 0.1
        assert not any(phase.startswith("trigger") for phase, _ in script.events)
    else:
        assert ("collision_exit", "Ground") in script.events
        phases = [phase for phase, name in script.events if name == "Sensor"]
        assert phases[0] == "trigger_enter"
        assert "trigger_stay" in phases
        assert phases[-1] == "trigger_exit"
