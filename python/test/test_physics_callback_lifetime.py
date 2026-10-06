"""Real Jolt contact dispatch under component mutations, isolated against native UAF."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("kind", ["trigger", "collision"])
@pytest.mark.parametrize("phase", ["enter", "stay", "exit"])
@pytest.mark.parametrize("action", [
    "remove_sibling", "remove_peer", "disable_sibling", "disable_peer",
    "deactivate_owner", "remove_self", "remove_own_collider",
    "remove_peer_collider", "replace_sibling", "append_peer", "remove_self_throw",
])
def test_contact_dispatch_resolves_live_receivers(tmp_path, kind, phase, action):
    import infernux

    result = subprocess.run(
        [sys.executable, "-X", "utf8", "-B", str(Path(__file__).resolve()),
         str(tmp_path), kind, phase, action],
        env={**os.environ, "PYTHONPATH": str(Path(infernux.__file__).resolve().parent.parent)},
        capture_output=True, text=True, encoding="utf-8", timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    if action == "remove_self_throw":
        assert "ContactProbe.on_" + kind + "_" + phase in result.stdout + result.stderr
        assert "CONTACT_SELF_REMOVE_EXPECTED" in result.stdout + result.stderr
    evidence = json.loads((tmp_path / "contact-evidence.json").read_text(encoding="utf-8"))
    assert evidence["status"] == "passed"


@pytest.mark.parametrize("phase", ["enter", "stay", "exit"])
@pytest.mark.parametrize("action", ["keep", "remove", "replace", "ignore_box", "ignore_sphere", "ignore_both"])
@pytest.mark.parametrize("kind", ["trigger", "collision"])
def test_contact_batch_retains_compound_member_identity(tmp_path, phase, action, kind):
    import infernux

    result = subprocess.run(
        [sys.executable, "-X", "utf8", "-B", str(Path(__file__).resolve()),
         str(tmp_path), "compound_" + kind, phase, action],
        env={**os.environ, "PYTHONPATH": str(Path(infernux.__file__).resolve().parent.parent)},
        capture_output=True, text=True, encoding="utf-8", timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads((tmp_path / "contact-evidence.json").read_text(encoding="utf-8"))["status"] == "passed"


def exercise_contact_dispatch(project, kind, phase, action):
    from infernux.components import InxComponent
    from infernux.engine.engine import Engine
    from infernux.engine.preferences_store import PreferencesStore
    from infernux.lib import LogLevel, Physics, RuntimeMode, SceneManager, Vector3

    if sys.platform == "win32":
        import ctypes
        # A regression must fail this worker, never wait for a crash dialog.
        ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002)

    (project / "Assets").mkdir(parents=True)
    (project / "ProjectSettings").mkdir()
    PreferencesStore()._path = str(project / "preferences.json")
    engine = Engine(LogLevel.Warn, RuntimeMode.Headless)
    trace, errors = [], []
    mutated = []
    objects, colliders, probes = {}, {}, {}
    removed_handles = []
    evidence_path = project / "contact-evidence.json"

    def save(status):
        evidence_path.write_text(json.dumps(dict(
            status=status, kind=kind, phase=phase, action=action,
            trace=trace, errors=errors, mutated=mutated,
        ), indent=2), encoding="utf-8")

    class ContactProbe(InxComponent):
        side = ""
        role = ""

        def receive(self, received_kind, received_phase, contact):
            if received_kind != kind or received_phase != phase:
                return
            try:
                # Access the payload before mutating the graph. Every delivered
                # callback must receive a still-live peer collider/object.
                other = contact.game_object
                assert other.name in ("ContactStatic", "ContactMover")
                trace.append([self.side, self.role])
                if self.role != "mutator" or mutated:
                    return
                side, peer = self.side, "b" if self.side == "a" else "a"
                mutated.append(side)
                save("mutating")
                if action in ("remove_sibling", "replace_sibling", "remove_peer"):
                    target_side = peer if action == "remove_peer" else side
                    victim = probes[target_side, "victim"]
                    old_handle = victim._cpp_component.handle
                    removed_handles.append(old_handle)
                    assert objects[target_side].remove_component(victim)
                    if action == "replace_sibling":
                        replacement = objects[side].add_component(ContactProbe)
                        replacement.side, replacement.role = side, "replacement"
                        # IDs may be restored by document publication. A new
                        # generation must never inherit the queued callback.
                        replacement._cpp_component._set_component_id(old_handle.id)
                        assert scene.resolve_component(old_handle) is None
                        assert scene.resolve_component(replacement._cpp_component.handle) is not None
                elif action in ("disable_sibling", "disable_peer"):
                    target_side = peer if action == "disable_peer" else side
                    probes[target_side, "victim"].enabled = False
                elif action == "deactivate_owner":
                    objects[side].active = False
                elif action in ("remove_self", "remove_self_throw"):
                    removed_handles.append(self._cpp_component.handle)
                    assert objects[side].remove_component(self)
                elif action in ("remove_own_collider", "remove_peer_collider"):
                    target_side = peer if action == "remove_peer_collider" else side
                    removed_handles.append(colliders[target_side].handle)
                    assert objects[target_side].remove_component(colliders[target_side])
                elif action == "append_peer":
                    added = objects[peer].add_component(ContactProbe)
                    added.side, added.role = peer, "added"
                save("mutated")
            except Exception as error:
                errors.append(repr(error))
                save("callback-error")

        def on_trigger_enter(self, other):
            self.receive("trigger", "enter", other)

        def on_trigger_stay(self, other):
            self.receive("trigger", "stay", other)

        def on_trigger_exit(self, other):
            self.receive("trigger", "exit", other)

        def on_collision_enter(self, collision):
            self.receive("collision", "enter", collision)

        def on_collision_stay(self, collision):
            self.receive("collision", "stay", collision)

        def on_collision_exit(self, collision):
            self.receive("collision", "exit", collision)

    if action == "remove_self_throw":
        # Exercise the native exception boundary too: ordinary user exceptions
        # are caught by _safe_lifecycle_call before they reach PyComponentProxy.
        def throwing_entry(self, contact):
            before = len(mutated)
            self.receive(kind, phase, contact)
            if len(mutated) != before:
                raise RuntimeError("CONTACT_SELF_REMOVE_EXPECTED")

        setattr(ContactProbe, "_call_on_" + kind + "_" + phase, throwing_entry)

    try:
        engine.init_headless(str(project))
        manager = SceneManager.instance()
        scene = manager.get_active_scene()
        objects["a"] = scene.create_game_object("ContactStatic")
        objects["b"] = scene.create_game_object("ContactMover")
        colliders["a"] = objects["a"].add_component("BoxCollider")
        colliders["a"].size = Vector3(4, 4, 4)
        colliders["a"].is_trigger = kind == "trigger"
        mover = objects["b"]
        mover.transform.position = Vector3(8, 0, 0)
        colliders["b"] = mover.add_component("SphereCollider")
        body = mover.add_component("Rigidbody")
        body.is_kinematic = kind == "trigger"
        body.use_gravity = False
        for side, game_object in objects.items():
            for role in ("mutator", "victim", "survivor"):
                probe = game_object.add_component(ContactProbe)
                probe.side, probe.role = side, role
                probes[side, role] = probe
        manager.play()
        manager.pause()
        manager.step()
        mover.transform.position = Vector3(0, 2.25, 0)
        Physics.sync_transforms()
        manager.step()
        if phase != "enter":
            manager.step()
        if phase == "exit":
            mover.transform.position = Vector3(8, 0, 0)
            Physics.sync_transforms()
            manager.step()
        assert not errors, errors
        assert len(mutated) == 1, trace
        side = mutated[0]
        peer = "b" if side == "a" else "a"
        assert trace[0] == [side, "mutator"]
        assert not any(role in ("replacement", "added") for _, role in trace), trace
        for handle in removed_handles:
            assert scene.resolve_component(handle) is None
        if action in ("deactivate_owner", "remove_own_collider", "remove_peer_collider"):
            assert trace == [[side, "mutator"]], trace
        else:
            expected = [[which, role] for which in (side, peer)
                        for role in ("mutator", "victim", "survivor")]
            if action in ("remove_sibling", "replace_sibling", "disable_sibling"):
                expected.remove([side, "victim"])
            elif action in ("remove_peer", "disable_peer"):
                expected.remove([peer, "victim"])
            assert trace == expected, (trace, expected)
        save("passed")
    finally:
        engine.exit()


def exercise_compound_batch(project, phase, action, kind):
    from infernux.components import InxComponent
    from infernux.engine.engine import Engine
    from infernux.engine.preferences_store import PreferencesStore
    from infernux.lib import LogLevel, Physics, RuntimeMode, SceneManager, Vector3

    (project / "Assets").mkdir(parents=True)
    (project / "ProjectSettings").mkdir()
    PreferencesStore()._path = str(project / "preferences.json")
    engine = Engine(LogLevel.Warn, RuntimeMode.Headless)
    seen, errors = [], []
    members = {}
    removed = []

    class CompoundProbe(InxComponent):
        def receive(self, received_phase, other):
            if received_phase != phase:
                return
            try:
                handle = other._cpp_component.handle
                seen.append(handle.id)
                if len(seen) != 1 or action not in ("remove", "replace"):
                    return
                target_id = next(identity for identity in members if identity != handle.id)
                victim = members[target_id]
                removed.append(victim.handle)
                victim_type = type(victim).__name__
                assert compound.remove_component(victim)
                if action == "replace":
                    replacement = compound.add_component(victim_type)
                    replacement.is_trigger = kind == "trigger"
                    replacement._set_component_id(target_id)
                assert scene.resolve_component(removed[0]) is None
            except Exception as error:
                errors.append(repr(error))

        def on_trigger_enter(self, other):
            self.receive("enter", other)

        def on_trigger_stay(self, other):
            self.receive("stay", other)

        def on_trigger_exit(self, other):
            self.receive("exit", other)

        def on_collision_enter(self, collision):
            self.receive("enter", collision.collider)

        def on_collision_stay(self, collision):
            self.receive("stay", collision.collider)

        def on_collision_exit(self, collision):
            self.receive("exit", collision.collider)

    try:
        engine.init_headless(str(project))
        manager = SceneManager.instance()
        scene = manager.get_active_scene()
        compound = scene.create_game_object("Compound")
        box = compound.add_component("BoxCollider")
        box.center = Vector3(-2, 0, 0)
        box.is_trigger = kind == "trigger"
        sphere = compound.add_component("SphereCollider")
        sphere.center = Vector3(2, 0, 0)
        sphere.is_trigger = kind == "trigger"
        members = {member.component_id: member for member in (box, sphere)}
        mover = scene.create_game_object("Mover")
        mover.transform.position = Vector3(20, 0, 0)
        mover_collider = mover.add_component("BoxCollider")
        mover_collider.size = Vector3(8, 1, 1)
        ignored = set()
        for member, selected in ((box, "ignore_box"), (sphere, "ignore_sphere")):
            if action in (selected, "ignore_both"):
                Physics.ignore_collision(member, mover_collider)
                assert Physics.get_ignore_collision(member, mover_collider)
                ignored.add(member.component_id)
        body = mover.add_component("Rigidbody")
        body.is_kinematic = kind == "trigger"
        body.use_gravity = False
        mover.add_component(CompoundProbe)
        manager.play()
        manager.pause()
        manager.step()
        mover.transform.position = Vector3(0, 0.75 if kind == "collision" else 0, 0)
        Physics.sync_transforms()
        manager.step()
        if phase != "enter":
            manager.step()
        if phase == "exit":
            mover.transform.position = Vector3(20, 0, 0)
            Physics.sync_transforms()
            manager.step()
        assert not errors, errors
        if action == "keep" or action.startswith("ignore_"):
            expected = set(members) - ignored
            assert len(seen) == len(expected) and set(seen) == expected, (seen, expected)
        else:
            assert len(seen) == 1 and len(removed) == 1, seen
        (project / "contact-evidence.json").write_text(json.dumps(dict(
            status="passed", kind=kind, phase=phase, action=action, seen=seen,
            removed=[item.id for item in removed], errors=errors,
        ), indent=2), encoding="utf-8")
    finally:
        engine.exit()


if __name__ == "__main__":
    if sys.argv[2].startswith("compound_"):
        exercise_compound_batch(Path(sys.argv[1]), *sys.argv[3:5], sys.argv[2].removeprefix("compound_"))
    else:
        exercise_contact_dispatch(Path(sys.argv[1]), *sys.argv[2:5])
