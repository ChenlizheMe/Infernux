"""Cleanup callbacks run while their native owner graph still exists."""

import pytest

from infernux.components import InxComponent, serialized_field
from infernux.instantiate import Destroy
from infernux.lib import SceneManager, Vector3


@pytest.mark.parametrize("operation", ["component", "object", "scene"])
def test_destroy_callback_can_read_owner_transform_and_native_sibling(scene, operation):
    events = []

    class CleanupProbe(InxComponent):
        value: float = serialized_field(default=3.0)

        def on_destroy(self):
            owner = self.game_object
            events.append((
                owner.id,
                owner.name,
                self.transform.position.x,
                self.value,
                owner.get_component("BoxCollider").game_object.id,
                owner.scene.find_by_id(owner.id).id,
            ))

    owner = scene.create_game_object("CleanupOwner")
    owner.add_component("BoxCollider")
    component = owner.add_component(CleanupProbe)
    owner_id = owner.id
    if operation == "component":
        component.destroy()
    elif operation == "object":
        Destroy(owner)
        scene.process_pending_destroys()
    else:
        SceneManager.instance().unload_scene(scene)

    assert events == [(owner_id, "CleanupOwner", 0.0, 3.0, owner_id, owner_id)]
    assert component._is_destroyed
    assert component._cpp_component is None
    assert component._cds_slot is None
    with pytest.raises(RuntimeError, match="not bound to a live GameObject"):
        _ = component.game_object


def test_destroyed_hierarchy_keeps_each_owner_available_for_its_cleanup(scene):
    events = []

    class CleanupProbe(InxComponent):
        def on_destroy(self):
            events.append((self.game_object.id, self.game_object.name, self.transform.local_position.x))

    root = scene.create_game_object("CleanupRoot")
    child = scene.create_game_object("CleanupChild")
    child.set_parent(root, False)
    root.add_component(CleanupProbe)
    child.add_component(CleanupProbe)
    expected = {(root.id, "CleanupRoot", 0.0), (child.id, "CleanupChild", 0.0)}
    Destroy(root)
    scene.process_pending_destroys()
    assert set(events) == expected
    assert len(events) == 2


def test_reentrant_cleanup_requests_do_not_repeat_callbacks_or_lose_owner(scene):
    events = []

    class CleanupProbe(InxComponent):
        def on_destroy(self):
            owner = self.game_object
            self.destroy()
            Destroy(owner)
            events.append((self.game_object.id, owner.name))

    owner = scene.create_game_object("ReentrantCleanup")
    component = owner.add_component(CleanupProbe)
    owner_id = owner.id
    Destroy(owner)
    scene.process_pending_destroys()
    scene.process_pending_destroys()
    assert events == [(owner_id, "ReentrantCleanup")]
    assert component._is_destroyed


def test_throwing_cleanup_still_releases_native_binding_and_fields(scene):
    events = []

    class CleanupProbe(InxComponent):
        value: float = serialized_field(default=1.0)

        def on_destroy(self):
            events.append(self.game_object.name)
            raise RuntimeError("intentional cleanup failure")

    owner = scene.create_game_object("ThrowingCleanup")
    component = owner.add_component(CleanupProbe)
    component.destroy()
    assert events == ["ThrowingCleanup"]
    assert component._is_destroyed
    assert component._cpp_component is None
    assert component._native_handle is None
    assert component._native_scene is None
    assert component._cds_slot is None
    assert owner.get_component(CleanupProbe) is None


def test_coroutine_finalizers_can_read_owner_but_cannot_schedule_new_work(scene):
    events = []

    def unexpected_work():
        events.append("unexpected coroutine ran")
        yield None

    class CleanupProbe(InxComponent):
        def work(self):
            try:
                yield None
            finally:
                events.append(("finally", self.game_object.name))
                with pytest.raises(RuntimeError, match="cannot start a coroutine"):
                    self.start_coroutine(unexpected_work())

        def on_destroy(self):
            events.append(("destroy", self.game_object.name))
            with pytest.raises(RuntimeError, match="cannot start a coroutine"):
                self.start_coroutine(unexpected_work())

    owner = scene.create_game_object("CoroutineCleanup")
    component = owner.add_component(CleanupProbe)
    coroutine = component.start_coroutine(component.work())
    component.destroy()

    assert events == [("finally", "CoroutineCleanup"), ("destroy", "CoroutineCleanup")]
    assert coroutine.is_finished
    assert component._coroutine_scheduler is None
    assert component.__dict__["_runtime_coroutine_scheduler"] is None
    with pytest.raises(RuntimeError, match="cannot start a coroutine"):
        component.start_coroutine(unexpected_work())
    assert len(events) == 2


@pytest.mark.parametrize("snapshot", [False, True])
def test_retained_world_cleanup_reads_original_owner_without_binding_to_replacement(scene, snapshot):
    events = []

    class CleanupProbe(InxComponent):
        def on_disable(self):
            events.append(("disable", self.game_object.name, self.transform.position.x))

        def on_destroy(self):
            events.append(("destroy", self.game_object.name, self.transform.position.x))

    owner = scene.create_game_object("RetainedCleanup")
    component = owner.add_component(CleanupProbe)
    document = scene._capture_play_mode_snapshot() if snapshot else scene.serialize_document()
    owner.transform.position = Vector3(7.0, 0.0, 0.0)
    commit = scene._commit_play_mode_snapshot_retaining_world if snapshot else scene._commit_document_retaining_world
    token = commit(document)
    assert token is not None
    replacement = scene.find_by_id(owner.id)
    assert replacement is not owner
    assert replacement.transform.position.x == 0.0
    token.finalize()

    assert events == [("disable", "RetainedCleanup", 7.0), ("destroy", "RetainedCleanup", 7.0)]
    assert component._is_destroyed
    assert component._native_handle is None
    assert "_native_cleanup_binding" not in component.__dict__
    assert scene.find_by_id(replacement.id) is replacement
    assert replacement.transform.position.x == 0.0
    with pytest.raises(RuntimeError, match="not bound to a live GameObject"):
        _ = component.game_object


def test_throwing_coroutine_finalizer_does_not_skip_destroy_callback_or_leave_binding(scene):
    events = []

    class CleanupProbe(InxComponent):
        def work(self):
            try:
                yield None
            finally:
                events.append(("finally", self.game_object.name))
                raise RuntimeError("intentional coroutine cleanup failure")

        def on_destroy(self):
            events.append(("destroy", self.game_object.name))

    component = scene.create_game_object("ThrowingCoroutineCleanup").add_component(CleanupProbe)
    coroutine = component.start_coroutine(component.work())
    component.destroy()
    assert events == [("finally", "ThrowingCoroutineCleanup"), ("destroy", "ThrowingCoroutineCleanup")]
    assert coroutine.is_finished
    assert component._is_destroyed
    assert component._coroutine_scheduler is None
    assert component._native_handle is None
    assert "_native_cleanup_binding" not in component.__dict__
