"""Committed Play domain handoff retires old state after publication."""

import pytest

from infernux.components import InxComponent, serialized_field
from infernux.engine.component_restore import replace_scene_python_components_for_play
from infernux.lib import Vector3


class DomainCleanupProbe(InxComponent):
    value: int = serialized_field(default=3)
    _events = []
    _reject_publication = False
    _throw_cleanup = False
    _throw_coroutine = False

    def on_after_deserialize(self):
        if type(self)._reject_publication:
            raise RuntimeError("intentional Play publication failure")

    def work(self):
        try:
            yield None
        finally:
            type(self)._events.append(("coroutine", self.game_object.name, self.value))
            if type(self)._throw_coroutine:
                raise RuntimeError("intentional domain coroutine cleanup failure")

    def on_destroy(self):
        owner = self.game_object
        type(self)._events.append(("destroy", owner.name, self.transform.position.x,
                                  self.value, owner.get_component("BoxCollider").game_object.id))
        self.value = 99
        self.destroy()
        if type(self)._throw_cleanup:
            raise RuntimeError("intentional domain cleanup failure")


@pytest.fixture(autouse=True)
def reset_domain_probe():
    DomainCleanupProbe._events = []
    DomainCleanupProbe._reject_publication = False
    DomainCleanupProbe._throw_cleanup = False
    DomainCleanupProbe._throw_coroutine = False
    yield
    DomainCleanupProbe._reject_publication = False
    DomainCleanupProbe._throw_cleanup = False
    DomainCleanupProbe._throw_coroutine = False


@pytest.mark.parametrize("coroutine", [False, True])
def test_committed_domain_cleanup_keeps_old_owner_and_fields_without_touching_new_instance(scene, coroutine):
    owner = scene.create_game_object("PlayDomainOwner")
    owner.transform.position = Vector3(7, 0, 0)
    owner.add_component("BoxCollider")
    previous = owner.add_component(DomainCleanupProbe)
    previous.value = 9
    owner_id = owner.id
    running = previous.start_coroutine(previous.work()) if coroutine else None
    assert previous._awake_called
    snapshot = scene._capture_play_mode_snapshot()

    assert replace_scene_python_components_for_play(scene, snapshot)

    current = owner.get_component(DomainCleanupProbe)
    assert current is not previous
    assert current.value == 9
    assert current.game_object is owner
    assert not current._is_destroyed
    assert not current._awake_called
    expected = [("destroy", "PlayDomainOwner", 7.0, 9, owner_id)]
    if coroutine:
        expected.insert(0, ("coroutine", "PlayDomainOwner", 9))
        assert running.is_finished
    assert DomainCleanupProbe._events == expected
    assert previous._is_destroyed
    assert previous._cds_slot is None
    assert previous._cpp_component is None
    assert previous._coroutine_scheduler is None
    with pytest.raises(RuntimeError, match="not bound to a live GameObject"):
        _ = previous.game_object


def test_failed_domain_publication_restores_exact_fields_and_live_coroutines_without_cleanup(scene):
    owners = [scene.create_game_object(name) for name in ("RollbackFirst", "RollbackSecond")]
    previous = []
    for index, owner in enumerate(owners):
        owner.add_component("BoxCollider")
        component = owner.add_component(DomainCleanupProbe)
        component.value = 9 + index
        previous.append(component)
    slots = [component._cds_slot for component in previous]
    running = [component.start_coroutine(component.work()) for component in previous]
    snapshot = scene._capture_play_mode_snapshot()
    DomainCleanupProbe._reject_publication = True
    try:
        with pytest.raises(RuntimeError, match="intentional Play publication failure"):
            replace_scene_python_components_for_play(scene, snapshot)
    finally:
        DomainCleanupProbe._reject_publication = False

    assert [owner.get_component(DomainCleanupProbe) for owner in owners] == previous
    assert [component.value for component in previous] == [9, 10]
    assert [component._cds_slot for component in previous] == slots
    assert all(not component._is_destroyed and component._awake_called for component in previous)
    assert all(not handle.is_finished for handle in running)
    assert all(component._coroutine_scheduler.count == 1 for component in previous)
    assert DomainCleanupProbe._events == []


def test_throwing_domain_cleanup_still_releases_old_fields_after_successful_publication(scene):
    owner = scene.create_game_object("ThrowingDomainCleanup")
    owner.add_component("BoxCollider")
    previous = owner.add_component(DomainCleanupProbe)
    previous.value = 9
    snapshot = scene._capture_play_mode_snapshot()
    DomainCleanupProbe._throw_cleanup = True
    try:
        assert replace_scene_python_components_for_play(scene, snapshot)
    finally:
        DomainCleanupProbe._throw_cleanup = False

    assert DomainCleanupProbe._events == [("destroy", "ThrowingDomainCleanup", 0.0, 9, owner.id)]
    assert previous._is_destroyed
    assert previous._cds_slot is None
    assert previous._coroutine_scheduler is None
    assert owner.get_component(DomainCleanupProbe).value == 9


def test_never_awakened_edit_domain_is_released_without_destroy_callback(scene):
    owner = scene.create_game_object("InactiveDomainCleanup")
    owner.active = False
    owner.add_component("BoxCollider")
    previous = owner.add_component(DomainCleanupProbe)
    previous.value = 9
    snapshot = scene._capture_play_mode_snapshot()
    assert not previous._awake_called

    assert replace_scene_python_components_for_play(scene, snapshot)

    assert DomainCleanupProbe._events == []
    assert previous._cds_slot is None
    assert previous._is_destroyed
    assert owner.get_component(DomainCleanupProbe).value == 9


def test_throwing_domain_coroutine_cleanup_still_runs_destroy_and_releases_fields(scene):
    owner = scene.create_game_object("ThrowingDomainCoroutine")
    owner.add_component("BoxCollider")
    previous = owner.add_component(DomainCleanupProbe)
    previous.value = 9
    running = previous.start_coroutine(previous.work())
    snapshot = scene._capture_play_mode_snapshot()
    DomainCleanupProbe._throw_coroutine = True
    try:
        assert replace_scene_python_components_for_play(scene, snapshot)
    finally:
        DomainCleanupProbe._throw_coroutine = False

    assert DomainCleanupProbe._events == [
        ("coroutine", "ThrowingDomainCoroutine", 9),
        ("destroy", "ThrowingDomainCoroutine", 0.0, 9, owner.id),
    ]
    assert running.is_finished
    assert previous._is_destroyed
    assert previous._cds_slot is None
    assert previous._coroutine_scheduler is None
    assert owner.get_component(DomainCleanupProbe).value == 9
