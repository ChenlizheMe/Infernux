"""Runtime change journal: field tracking is paid for only by subscribers."""

from infernux.engine.runtime_change_journal import (
    RuntimeChangeDomain,
    RuntimeChangeJournal,
    RuntimeFieldKey,
)


def test_structural_cursor_does_not_enable_field_tracking():
    journal = RuntimeChangeJournal()
    cursor = journal.create_cursor("structural", domains=("scene_topology",))

    journal.publish_component_field_value("Probe", 7, "speed")

    assert not journal.consume(cursor).changed


def test_field_consumer_enables_tracking_and_receives_keys():
    journal = RuntimeChangeJournal()
    cursor = journal.create_cursor("inspector")

    journal.publish_component_field_value("Probe", 7, "speed")
    changes = journal.consume(cursor)

    assert changes.for_domain(RuntimeChangeDomain.COMPONENT_FIELD).fields == frozenset(
        {RuntimeFieldKey("Probe", 7, "speed")}
    )
    assert changes.for_domain(RuntimeChangeDomain.SCRIPT_VALUE).stable_ids == frozenset({7})


def test_batch_fields_coalesce_inside_one_transaction():
    journal = RuntimeChangeJournal()
    cursor = journal.create_cursor("inspector")

    with journal.transaction():
        journal.publish_component_fields("Probe", (1, 2, 1), "speed")
        journal.publish_component_field_value("Probe", 3, "speed")

    changes = journal.consume(cursor)

    assert changes.for_domain(RuntimeChangeDomain.COMPONENT_FIELD).fields == frozenset(
        {
            RuntimeFieldKey("Probe", 1, "speed"),
            RuntimeFieldKey("Probe", 2, "speed"),
            RuntimeFieldKey("Probe", 3, "speed"),
        }
    )
    assert changes.for_domain(RuntimeChangeDomain.SCRIPT_VALUE).stable_ids == frozenset(
        {1, 2, 3}
    )

    lagging = journal.create_cursor("lagging", domains=("script_value",))
    for start in range(0, 42, 7):
        for component_id in range(start, start + 7):
            journal.publish_component_field_value("Probe", component_id, "speed")
            journal.flush()
        changes = journal.consume(cursor)
        assert changes.for_domain(RuntimeChangeDomain.COMPONENT_FIELD).fields == frozenset(
            RuntimeFieldKey("Probe", component_id, "speed")
            for component_id in range(start, start + 7)
        )
        assert changes.for_domain(RuntimeChangeDomain.SCRIPT_VALUE).stable_ids == frozenset(
            range(start, start + 7)
        )
    changes = journal.consume(lagging)
    assert changes.full_resync
    assert changes.for_domain(RuntimeChangeDomain.SCRIPT_VALUE).broad


def test_scheduler_cursor_reports_structural_domains_only():
    from infernux.components._component_lifecycle import RuntimeExecutionScheduler

    journal = RuntimeChangeJournal()
    scheduler = RuntimeExecutionScheduler(name="structural-only", change_journal=journal, world_id=0)
    try:
        journal.publish_component_field_value("Probe", 1, "speed")
        journal.publish(
            RuntimeChangeDomain.COMPONENT_STRUCTURE,
            stable_id="Probe",
        )
        changes = journal.consume(scheduler._change_cursor)

        assert RuntimeChangeDomain.COMPONENT_STRUCTURE in changes.changes
        assert RuntimeChangeDomain.COMPONENT_FIELD not in changes.changes
    finally:
        scheduler.clear()


def test_component_notifications_follow_owner_even_after_native_invalidation():
    from types import SimpleNamespace
    from infernux.components._component_lifecycle import (
        ComponentLifecycleMixin,
        RuntimeExecutionScheduler,
        notify_runtime_component_added,
        notify_runtime_component_removed,
        notify_runtime_component_changed,
        notify_runtime_component_value_changed,
    )

    class Probe(ComponentLifecycleMixin):
        _type_guid_ = "shared-owner-notification-probe"
        _component_id = 7
        _native_generation = 1

        def __init__(self, owner):
            self._cpp_component = SimpleNamespace(game_object=SimpleNamespace(
                scene=SimpleNamespace(_runtime_world_id=owner)))
            self._registered_go_id = 1

    first_id, second_id = 10**15 + 1, 10**15 + 2
    journals = [RuntimeChangeJournal() for _ in range(3)]
    schedulers = [RuntimeExecutionScheduler(world_id=owner, change_journal=journal)
                  for owner, journal in zip((first_id, second_id, 0), journals)]
    cursors = [journal.create_cursor("owner-notifications") for journal in journals]
    first, second = Probe(first_id), Probe(second_id)
    try:
        notify_runtime_component_added(first)
        notify_runtime_component_added(second)
        for journal, cursor in zip(journals, cursors):
            journal.consume(cursor)

        # Same canonical type and component ID must never alias journals.
        notify_runtime_component_value_changed(first, "speed")
        assert journals[0].consume(cursors[0]).changed
        assert not journals[1].consume(cursors[1]).changed
        assert not journals[2].consume(cursors[2]).changed
        notify_runtime_component_changed(first)
        assert journals[0].consume(cursors[0]).changed
        assert not journals[1].consume(cursors[1]).changed

        first.__dict__["_runtime_coroutine_scheduler"] = SimpleNamespace(count=1)
        RuntimeExecutionScheduler._notify_component_runtime_work(first)
        assert id(first) in schedulers[0]._work_components
        assert id(first) not in schedulers[1]._work_components
        assert id(first) not in schedulers[2]._work_components

        # _invalidate_native_binding clears these before the remove callback.
        first._cpp_component = None
        first._registered_go_id = None
        notify_runtime_component_removed(first)
        assert not schedulers[0]._components
        assert id(first) not in schedulers[0]._work_components
        assert journals[0].consume(cursors[0]).changed
        assert not journals[1].consume(cursors[1]).changed
        assert not journals[2].consume(cursors[2]).changed

        # Explicit unbound users are isolated in domain zero.
        unbound = Probe(0)
        unbound._cpp_component = None
        unbound._registered_go_id = None
        schedulers[2].register_component(unbound)
        for journal, cursor in zip(journals, cursors):
            journal.consume(cursor)
        notify_runtime_component_value_changed(unbound, "speed")
        assert not journals[0].consume(cursors[0]).changed
        assert not journals[1].consume(cursors[1]).changed
        assert journals[2].consume(cursors[2]).changed

        # First native binding after direct unbound registration must retire
        # that old membership even though _registered_go_id had no real owner.
        unbound._cpp_component = SimpleNamespace(game_object=SimpleNamespace(
            scene=SimpleNamespace(_runtime_world_id=second_id)))
        unbound._registered_go_id = 1
        notify_runtime_component_added(unbound)
        assert not schedulers[2]._components
        assert id(unbound) in schedulers[1]._component_tokens
        assert not journals[0].consume(cursors[0]).changed
        assert journals[1].consume(cursors[1]).changed
        assert journals[2].consume(cursors[2]).changed
    finally:
        for scheduler in schedulers:
            scheduler.clear()


def test_batch_component_notifications_group_real_owners_and_keep_candidates_private():
    from types import SimpleNamespace
    from infernux.components._component_lifecycle import (
        ComponentLifecycleMixin,
        RuntimeExecutionScheduler,
        notify_runtime_component_added,
        notify_runtime_component_values,
    )

    class Probe(ComponentLifecycleMixin):
        _type_guid_ = "shared-batch-owner-probe"
        _native_generation = 1

        def __init__(self, owner, component_id, registered=True):
            self._component_id = component_id
            self._registered_go_id = 1 if registered else None
            self._cpp_component = SimpleNamespace(game_object=SimpleNamespace(
                scene=SimpleNamespace(_runtime_world_id=owner)))

    owners = (10**15 + 3, 10**15 + 4)
    journals = [RuntimeChangeJournal(), RuntimeChangeJournal()]
    schedulers = [RuntimeExecutionScheduler(world_id=owner, change_journal=journal)
                  for owner, journal in zip(owners, journals)]
    cursors = [journal.create_cursor("batch-owners") for journal in journals]
    components = (Probe(owners[0], 7), Probe(owners[1], 7),
                  Probe(owners[0], 8), Probe(owners[0], 9, registered=False))
    try:
        for component in components[:3]:
            notify_runtime_component_added(component)
        for journal, cursor in zip(journals, cursors):
            journal.consume(cursor)
        notify_runtime_component_values(components, "speed")
        expected = ({7, 8}, {7})
        for journal, cursor, ids in zip(journals, cursors, expected):
            changes = journal.consume(cursor)
            assert changes.for_domain(RuntimeChangeDomain.COMPONENT_FIELD).fields == frozenset(
                RuntimeFieldKey(Probe._type_guid_, identity, "speed") for identity in ids)
    finally:
        for scheduler in schedulers:
            scheduler.clear()


def test_owner_notification_bucket_does_not_retain_retired_schedulers():
    import gc
    import weakref
    from infernux.components._component_lifecycle import RuntimeExecutionScheduler

    owner = 10**15 + 5
    scheduler = RuntimeExecutionScheduler(world_id=owner, change_journal=RuntimeChangeJournal())
    reference = weakref.ref(scheduler)
    assert owner in RuntimeExecutionScheduler._schedulers_by_world
    del scheduler
    gc.collect()
    assert reference() is None
    assert owner not in RuntimeExecutionScheduler._schedulers_by_world


def test_unbound_scheduler_default_journal_does_not_use_ambient_world():
    from unittest.mock import patch
    from infernux.components._component_lifecycle import RuntimeExecutionScheduler

    with patch("infernux.engine.runtime_change_journal.runtime_change_journal",
               side_effect=AssertionError("unbound scheduler borrowed ambient journal")):
        first = RuntimeExecutionScheduler(world_id=0)
        second = RuntimeExecutionScheduler(world_id=0)
    try:
        assert first.change_journal is not second.change_journal
    finally:
        first.clear()
        second.clear()
