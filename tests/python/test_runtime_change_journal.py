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


def test_scheduler_cursor_reports_structural_domains_only():
    from infernux.components._component_lifecycle import RuntimeExecutionScheduler

    journal = RuntimeChangeJournal()
    scheduler = RuntimeExecutionScheduler(name="structural-only", change_journal=journal)
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
