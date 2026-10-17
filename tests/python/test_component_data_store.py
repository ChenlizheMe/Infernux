from __future__ import annotations

import gc
import weakref

import numpy as np
import pytest

from infernux import lib
from infernux.batch import batch_read, batch_write, create_batch_handle, create_scene_batch_handle
from infernux.components import InxComponent, serialized_field
from infernux.components._cds_bridge import get_class_id


@pytest.mark.parametrize("native_storage", [False, True])
def test_batch_write_preserves_field_ranges_and_publishes_changes(native_storage):
    from infernux.components._component_lifecycle import RuntimeExecutionScheduler
    from infernux.engine.runtime_change_journal import (
        RuntimeChangeDomain,
        RuntimeChangeJournal,
        RuntimeFieldKey,
    )

    class Base(InxComponent):
        _uses_component_data_store = native_storage
        speed: float = serialized_field(default=1.0, range=(0.0, 5.0))

    class Derived(Base):
        pass

    journal = RuntimeChangeJournal()
    scheduler = RuntimeExecutionScheduler(name="batch-values", change_journal=journal)
    components = [Derived(), Derived()]
    try:
        cursor = journal.create_cursor("batch-values")
        batch_write(components, np.array([2.0, 3.0]), "speed")
        assert not journal.consume(cursor).changed  # Unregistered candidates stay private.
        for index, component in enumerate(components):
            component._registered_go_id = index + 1
        source = np.array([-2.0, 8.0, 100.0])  # Extra rows are allowed by the API.
        batch_write(components, source, "speed")
        np.testing.assert_array_equal(source, [-2.0, 8.0, 100.0])
        np.testing.assert_array_equal(batch_read(components, "speed"), [0.0, 5.0])
        batch_changes = journal.consume(cursor)
        assert batch_changes.for_domain(RuntimeChangeDomain.COMPONENT_FIELD).fields == frozenset(
            RuntimeFieldKey(Derived._type_guid_, component.component_id, "speed") for component in components
        )
        for component, value in zip(components, source):
            component.speed = float(value)
        scalar_changes = journal.consume(cursor)
        assert batch_changes.changes == scalar_changes.changes
    finally:
        for component in components:
            component._registered_go_id = None
            component._call_on_destroy()
        scheduler.clear()


def test_batch_rejects_mixed_classes_before_accessing_colliding_slots():
    class First(InxComponent):
        value: float = 1.0

    class Second(InxComponent):
        value: float = 2.0

    first, second = First(), Second()
    try:
        assert first._cds_class_id != second._cds_class_id
        assert first._cds_slot == second._cds_slot
        with pytest.raises(TypeError, match="same concrete type"):
            batch_read([first, second], "value")
        with pytest.raises(TypeError, match="same concrete type"):
            batch_write([first, second], np.array([31.0, 42.0]), "value")
        assert (first.value, second.value) == (1.0, 2.0)
    finally:
        first._call_on_destroy()
        second._call_on_destroy()


def test_inherited_candidate_publication_and_rollback_preserve_live_parent():
    from infernux.components._cds_bridge import prepare_schema_publication
    from infernux.components._component_registration import candidate_component_registration_scope

    class Base(InxComponent):
        speed: float = 2.0

    parent = Base()
    publication = None
    try:
        parent.speed = 11.0
        parent_binding = Base.speed._cds_class_id
        with candidate_component_registration_scope():

            class Candidate(Base):
                count: int = 3

        assert Candidate.speed is not Base.speed
        assert Candidate.speed.metadata is Base.speed.metadata
        assert Candidate.speed._cds_class_id is None
        assert get_class_id(Candidate) is None
        publication = prepare_schema_publication([Candidate])
        entry = publication.entry(Candidate)
        assert set(entry.field_map) == {"speed", "count"}
        slot = publication.allocate_slot(Candidate)
        publication.set_value(Candidate, "speed", slot, 23.0)
        publication.set_value(Candidate, "count", slot, 5)
        publication.seal()
        assert Candidate.speed._cds_class_id is None
        publication.commit()
        field_id, type_code = entry.field_map["speed"]
        assert Candidate.speed._cds_class_id == entry.class_id
        assert lib._cds_get(entry.class_id, field_id, slot, type_code) == 23.0
        assert Base.speed._cds_class_id == parent_binding
        assert batch_read([parent], "speed")[0] == 11.0

        publication.rollback()
        assert Candidate.speed._cds_class_id is None
        assert Base.speed._cds_class_id == parent_binding
        assert batch_read([parent], "speed")[0] == parent.speed == 11.0
    finally:
        if publication is not None and not publication.rolled_back:
            publication.rollback()
        parent._call_on_destroy()


def test_cds_generational_handles_reject_stale_access():
    class_id = lib._cds_register_class("python.tests:GenerationalHandle")
    field_id = lib._cds_register_field(class_id, "value", 0)
    first = lib._cds_alloc(class_id)
    assert len(first) == 2
    lib._cds_set(class_id, field_id, first, 0, 4.5)
    assert lib._cds_get(class_id, field_id, first, 0) == pytest.approx(4.5)
    assert lib._cds_is_alive(class_id, first) is True

    lib._cds_free(class_id, first)
    assert lib._cds_is_alive(class_id, first) is False
    replacement = lib._cds_alloc(class_id)
    assert replacement[0] == first[0]
    assert replacement[1] != first[1]
    with pytest.raises(RuntimeError, match="stale or invalid"):
        lib._cds_get(class_id, field_id, first, 0)
    with pytest.raises(RuntimeError, match="stale or invalid"):
        lib._cds_free(class_id, first)
    lib._cds_free(class_id, replacement)


def test_transform_batch_handle_rejects_destroyed_transform(scene):
    game_object = scene.create_game_object("batch_handle_target")
    handle = create_batch_handle([game_object.transform])
    assert batch_read(handle, "local_position").shape == (1, 3)

    scene.destroy_game_object(game_object)
    scene.process_pending_destroys()
    with pytest.raises(RuntimeError, match="stale transform"):
        batch_read(handle, "local_position")


def test_transform_batch_handle_compacts_stale_transforms_with_mask(scene):
    first = scene.create_game_object("compact_stale")
    second = scene.create_game_object("compact_live")
    handle = create_batch_handle([first.transform, second.transform], mode="compact")

    scene.destroy_game_object(first)
    scene.process_pending_destroys()

    values, mask = batch_read(handle, "local_position")
    assert values.shape == (1, 3)
    np.testing.assert_array_equal(mask, np.asarray([False, True]))

    write_mask = batch_write(
        handle,
        np.asarray([[90.0, 90.0, 90.0], [4.0, 5.0, 6.0]], dtype=np.float32),
        "local_position",
    )
    np.testing.assert_array_equal(write_mask, mask)
    np.testing.assert_allclose(
        batch_read([second.transform], "local_position"),
        np.asarray([[4.0, 5.0, 6.0]], dtype=np.float32),
    )

    with pytest.raises(ValueError, match="mode must be"):
        create_batch_handle([second.transform], mode="lenient")


def test_scene_batch_handle_filters_in_native_code(scene):
    first = scene.create_game_object("WaveCube_0")
    second = scene.create_game_object("Other")
    third = scene.create_game_object("WaveCube_1")
    first.transform.position = lib.Vector3(1.0, 2.0, 3.0)
    second.transform.position = lib.Vector3(4.0, 5.0, 6.0)
    third.transform.position = lib.Vector3(7.0, 8.0, 9.0)

    handle = create_scene_batch_handle(scene, name_prefix="WaveCube_")
    positions = batch_read(handle, "position")

    assert len(handle) == 2
    np.testing.assert_allclose(
        positions,
        np.asarray([[1.0, 2.0, 3.0], [7.0, 8.0, 9.0]], dtype=np.float32),
    )


def test_script_replacement_detach_releases_the_old_numeric_slot():
    class ReplacedComponent(InxComponent):
        value: float = 1.0

    component = ReplacedComponent()
    class_id = component._cds_class_id
    slot = component._cds_slot
    assert class_id is not None and slot is not None
    assert lib._cds_is_alive(class_id, slot) is True

    component._detach_native_binding_for_replacement()

    assert component._cds_slot is None
    assert lib._cds_is_alive(class_id, slot) is False


def test_unbound_component_collection_releases_its_numeric_slot():
    class UnboundComponent(InxComponent):
        value: float = 1.0

    component = UnboundComponent()
    class_id, slot = component._cds_class_id, component._cds_slot
    witness = weakref.ref(component)
    assert lib._cds_is_alive(class_id, slot)
    del component
    gc.collect()
    assert witness() is None
    assert not lib._cds_is_alive(class_id, slot)


@pytest.mark.parametrize("native_storage", [False, True])
def test_batch_values_reserve_configure_save_and_reopen(tmp_path, native_storage):
    import json

    class Values(InxComponent):
        _uses_component_data_store = native_storage
        count: int = serialized_field(default=1, range=(0, 5))
        active: bool = False
        position: lib.Vector3 = lib.Vector3(1, 2, 3)

    if native_storage:
        Values.reserve_instances(257)
        class_id = get_class_id(Values)
        assert lib._cds_capacity(class_id) >= 257 and lib._cds_alive_count(class_id) == 0
        with pytest.raises(ValueError, match="non-negative integer"):
            Values.reserve_instances(-1)
    components = [Values(), Values()]
    restored = None
    try:
        batch_write(components, np.array([-1.8, 7.5]), "count")
        assert [value.count for value in components] == [0, 5]
        with pytest.raises(ValueError):
            batch_write(components, np.array([3.0, float("nan")]), "count")
        assert [value.count for value in components] == [0, 5]
        batch_write(components, np.array([0, 256]), "active")
        flags = batch_read(components, "active")
        assert flags.dtype == np.dtype(bool)
        np.testing.assert_array_equal(flags, [False, True])
        assert components[0].active is False and components[1].active is True
        with pytest.raises(ValueError, match="shape"):
            batch_write(components, np.zeros((2, 2), dtype=np.float32), "position")
        np.testing.assert_array_equal(batch_read(components, "position"), [[1, 2, 3], [1, 2, 3]])
        source = np.arange(12, dtype=np.float32).reshape(2, 6)[:, ::2]
        batch_write(components, source, "position")
        np.testing.assert_array_equal(batch_read(components, "position"), source)
        path = tmp_path / "batch-values.json"
        document = components[1]._serialize_fields_document()
        path.write_text(json.dumps(document), encoding="utf-8")
        restored = Values()
        restored._deserialize_fields_document(json.loads(path.read_text(encoding="utf-8")))
        assert restored._serialize_fields_document() == document
        assert restored.count == 5 and restored.active is True
        np.testing.assert_array_equal(batch_read([restored], "position"), source[1:2])
    finally:
        if restored is not None:
            restored._call_on_destroy()
        for component in components:
            component._call_on_destroy()
