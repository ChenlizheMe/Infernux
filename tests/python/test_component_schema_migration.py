from __future__ import annotations

from typing import Annotated

import pytest

from infernux.components import FormerlySerializedAs, InxComponent, serialized_field
from infernux.components._cds_migration import (
    FieldSchemaMigrationError,
    build_class_schema_migration,
    prepare_instance_values,
)
from infernux.components._component_registration import (
    candidate_component_registration_scope,
)


def _component(name: str, fields: dict[str, object]) -> type:
    namespace = {"__module__": f"schema_migration_{name}", **fields}
    with candidate_component_registration_scope():
        return type(name, (InxComponent,), namespace)


def test_stable_id_rename_preserves_value_without_a_runtime_name_alias():
    target = _component("StableRenameProbe", {"speed": serialized_field(1.0, field_id="movement")})
    candidate = _component("StableRenameProbe", {"velocity": serialized_field(2.0, field_id="movement")})
    instance = target()
    instance.speed = 9.0
    migration = build_class_schema_migration(target, candidate)
    assert migration.changed
    assert migration.removed_fields == ()
    assert prepare_instance_values(migration, (instance,))[instance] == {"velocity": 9.0}
    assert not hasattr(candidate(), "speed")


def test_changed_id_does_not_reuse_value_just_because_name_matches():
    target = _component("ChangedIdentityProbe", {"value": serialized_field(1, field_id="old")})
    candidate = _component("ChangedIdentityProbe", {"value": serialized_field(2, field_id="new")})
    instance = target()
    instance.value = 9
    migration = build_class_schema_migration(target, candidate)
    assert migration.changed
    assert migration.removed_fields == ("value",)
    assert prepare_instance_values(migration, (instance,))[instance] == {"value": 2}


def test_id_matching_handles_attribute_name_swap_without_swapping_meanings():
    target = _component(
        "IdentitySwapProbe",
        {
            "first": serialized_field(1, field_id="a"),
            "second": serialized_field(2, field_id="b"),
        },
    )
    candidate = _component(
        "IdentitySwapProbe",
        {
            "first": serialized_field(10, field_id="b"),
            "second": serialized_field(20, field_id="a"),
        },
    )
    instance = target()
    instance.first, instance.second = 7, 8
    migration = build_class_schema_migration(target, candidate)
    assert migration.changed
    assert prepare_instance_values(migration, (instance,))[instance] == {"first": 8, "second": 7}


def test_retired_cds_slot_waits_for_old_runtime_epoch_then_drains(monkeypatch):
    import gc

    from infernux.components import _cds_bridge
    from infernux.engine.runtime_dispatch import (
        current_runtime_epoch,
        ensure_runtime_dispatch_types,
        publish_runtime_dispatch_epoch,
    )

    target = _component(
        "EpochRetirementProbe",
        {"__annotations__": {"value": float}, "value": 1.0},
    )
    ensure_runtime_dispatch_types((target,))
    old_epoch = current_runtime_epoch()
    class_id = 9001
    slot = (3, 7)
    freed = []

    class _Probe:
        def __init__(self):
            self.alive = {slot}

        def _cds_alloc(self, candidate_class_id):
            sentinel = (9, 11)
            self.alive.add(sentinel)
            return sentinel

        def _cds_is_alive(self, candidate_class_id, candidate_slot):
            return candidate_class_id == class_id and tuple(candidate_slot) in self.alive

        def _cds_free(self, candidate_class_id, candidate_slot):
            candidate_slot = tuple(candidate_slot)
            self.alive.discard(candidate_slot)
            freed.append((candidate_class_id, candidate_slot))

    key = _cds_bridge._class_key(target)
    previous = _cds_bridge._class_registry.get(key)
    _cds_bridge._class_registry[key] = (class_id, {})
    monkeypatch.setattr(_cds_bridge, "_lib", _Probe())
    try:
        reference = _cds_bridge._retain_layout_epoch(class_id, target)
        assert reference is not None
        _cds_bridge.release_slot(target, slot, class_id)
        assert freed == []

        publication = publish_runtime_dispatch_epoch((), retired_types=(target,))
        publication.commit()
        del publication
        del old_epoch
        gc.collect()
        assert freed == [(class_id, slot), (class_id, (9, 11))]
    finally:
        _cds_bridge._retired_layouts.pop(class_id, None)
        if previous is None:
            _cds_bridge._class_registry.pop(key, None)
        else:
            _cds_bridge._class_registry[key] = previous
