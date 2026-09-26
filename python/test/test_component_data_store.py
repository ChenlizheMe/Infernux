from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from Infernux import lib
from Infernux.batch import batch_read, batch_write, create_batch_handle, create_scene_batch_handle
from Infernux.components import InxComponent
from Infernux.components._cds_bridge import get_class_id


def test_inherited_numeric_fields_share_the_declared_native_layout():
    from Infernux.components._cds_bridge import get_class_info

    class Base(InxComponent):
        speed: float = 2.0

    parent = Base()

    class Derived(Base):
        count: int = 3

    class Sibling(Base):
        offset: float = 4.0

    children = [Derived(), Derived()]
    sibling = Sibling()
    try:
        class_id, fields = get_class_info(Derived)
        assert set(fields) == {"speed", "count"}
        field_id, type_code = fields["speed"]
        assert lib._cds_get(class_id, field_id, children[0]._cds_slot, type_code) == 2.0
        batch_write(children, np.asarray([7.0, 8.0], dtype=np.float32), "speed")
        assert [child.speed for child in children] == [7.0, 8.0]
        children[0].speed = 9.0
        np.testing.assert_array_equal(batch_read(children, "speed"), [9.0, 8.0])
        assert children[0]._serialize_fields_document()["speed"] == 9.0
        assert parent.speed == 2.0 and sibling.speed == 2.0
        parent.speed = 11.0
        assert batch_read([parent], "speed")[0] == 11.0
        assert "speed" not in Derived._serialized_fields_  # Still declared by Base.
        assert Base.speed is not Derived.speed and Base.speed is not Sibling.speed
    finally:
        for value in [parent, sibling, *children]:
            value._call_on_destroy()


def test_repeated_layout_registration_binds_new_descriptors():
    from Infernux.components._cds_bridge import get_class_info

    class Repeated(InxComponent):
        value: float = 2.0

    old = Repeated()
    old_class = Repeated

    class Repeated(InxComponent):
        value: float = 3.0

    new = Repeated()
    try:
        assert get_class_id(old_class) == get_class_id(Repeated)
        class_id, fields = get_class_info(Repeated)
        field_id, type_code = fields["value"]
        assert lib._cds_get(class_id, field_id, new._cds_slot, type_code) == 3.0
        batch_write([new], np.asarray([8.0], dtype=np.float32), "value")
        assert new.value == 8.0 and old.value == 2.0
    finally:
        old._call_on_destroy()
        new._call_on_destroy()


def test_inherited_candidate_publication_and_rollback_preserve_live_parent():
    from Infernux.components._cds_bridge import prepare_schema_publication
    from Infernux.components._component_registration import candidate_component_registration_scope

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


def test_deep_inheritance_and_numeric_override_use_concrete_layouts():
    from Infernux.components._cds_bridge import get_class_info

    class Base(InxComponent):
        value: float = 2.5

    class Middle(Base):
        pass

    class Leaf(Middle):
        pass

    class Override(Middle):
        value: int = 7

    instances = [Base(), Middle(), Leaf(), Override()]
    try:
        for instance, expected in zip(instances, [2.5, 2.5, 2.5, 7]):
            assert instance.value == expected
            assert batch_read([instance], "value")[0] == expected
        assert len({get_class_id(type(instance)) for instance in instances}) == 4
        assert get_class_info(Leaf)[1]["value"][1] != get_class_info(Override)[1]["value"][1]
        instances[2].value = 9.5
        assert batch_read([instances[2]], "value")[0] == 9.5
        assert [instance.value for instance in instances] == [2.5, 2.5, 9.5, 7]
    finally:
        for instance in instances:
            instance._call_on_destroy()


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


def test_cds_batch_validates_handle_and_data_shapes():
    class_id = lib._cds_register_class("python.tests:BatchValidation")
    field_id = lib._cds_register_field(class_id, "position", 4)
    handles = [lib._cds_alloc(class_id), lib._cds_alloc(class_id)]
    handle_array = np.asarray(handles, dtype=np.uint32)
    values = np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)

    lib._cds_batch_scatter(class_id, field_id, 4, handle_array, values)
    np.testing.assert_array_equal(
        lib._cds_batch_gather(class_id, field_id, 4, handle_array), values
    )

    with pytest.raises(ValueError, match=r"shape \(N, 2\)"):
        lib._cds_batch_gather(
            class_id, field_id, 4, np.asarray([handles[0][0]], dtype=np.uint32)
        )
    with pytest.raises(ValueError, match=r"shape \(N, 3\)"):
        lib._cds_batch_scatter(
            class_id, field_id, 4, handle_array, np.zeros((2, 2), dtype=np.float32)
        )
    with pytest.raises(ValueError, match="field type mismatch"):
        lib._cds_batch_gather(class_id, field_id, 0, handle_array)

    for handle in handles:
        lib._cds_free(class_id, handle)


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


def test_transform_store_bumps_serial_when_a_world_matrix_becomes_dirty():
    source = (
        Path(__file__).resolve().parents[2]
        / "cpp/infernux/function/scene/TransformECSStore.cpp"
    ).read_text(encoding="utf-8")
    marker = source.index("void TransformECSStore::MarkWorldMatrixDirty")
    body = source[marker : source.index("void TransformECSStore::SetCachedWorldPosition")]
    assert "newlyDirty" in body
    assert "++m_globalTransformSerial" in body


def test_transform_batch_write_bumps_global_transform_serial(scene):
    first = scene.create_game_object("serial_a")
    second = scene.create_game_object("serial_b")
    manager = lib.SceneManager.instance()
    before = int(manager.get_global_transform_serial())

    batch_write(
        [first.transform, second.transform],
        np.asarray([[0.0, 15.0, 0.0], [0.0, 45.0, 0.0]], dtype=np.float32),
        "local_euler_angles",
    )

    after = int(manager.get_global_transform_serial())
    assert after > before
    np.testing.assert_allclose(
        batch_read([first.transform, second.transform], "local_euler_angles"),
        np.asarray([[0.0, 15.0, 0.0], [0.0, 45.0, 0.0]], dtype=np.float32),
        atol=1e-4,
    )


def test_component_class_can_reserve_numeric_storage():
    class ReservedComponent(InxComponent):
        value: float = 0.0

    ReservedComponent.reserve_instances(257)
    class_id = get_class_id(ReservedComponent)
    assert lib._cds_capacity(class_id) >= 257
    assert lib._cds_alive_count(class_id) == 0

    with pytest.raises(ValueError, match="non-negative integer"):
        ReservedComponent.reserve_instances(-1)


def test_component_cleanup_tolerates_an_already_released_slot():
    class CleanupComponent(InxComponent):
        value: float = 1.0

    component = CleanupComponent()
    class_id = component._cds_class_id
    slot = component._cds_slot
    assert class_id is not None and slot is not None

    lib._cds_free(class_id, slot)
    component._call_on_destroy()

    assert component._cds_slot is None
    assert component._cds_class_id is None


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


def test_reserve_rejects_component_without_numeric_storage():
    class TextOnlyComponent(InxComponent):
        label: str = "text"

    with pytest.raises(TypeError, match="no CDS-backed numeric fields"):
        TextOnlyComponent.reserve_instances(10)


def test_component_layout_revision_gets_distinct_storage():
    class ReloadedComponent(InxComponent):
        value: float = 0.0

    first_class_id = get_class_id(ReloadedComponent)

    class ReloadedComponent(InxComponent):
        value: int = 0

    second_class_id = get_class_id(ReloadedComponent)
    assert first_class_id != second_class_id
