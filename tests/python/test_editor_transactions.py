from __future__ import annotations

import pytest

from infernux.engine.interaction import EditorTransaction, TransactionRollbackError
from infernux.engine.undo import CompoundCommand, UndoCommand, UndoManager


class _AppendCommand(UndoCommand):
    """Order-sensitive command: earlier elements are later commands' parents."""

    marks_dirty = False

    def __init__(self, state, value, calls):
        super().__init__(value)
        self.state = state
        self.value = value
        self.calls = calls
        self.fail_execute = False
        self.fail_undo = False
        self.disposed = 0

    def execute(self):
        self.calls.append(("execute", self.value))
        if self.fail_execute:
            raise ValueError(f"cannot execute {self.value}")
        self.state.append(self.value)

    def undo(self):
        self.calls.append(("undo", self.value))
        if self.fail_undo:
            raise OSError(f"cannot undo {self.value}")
        assert self.state[-1] == self.value
        self.state.pop()

    def dispose(self):
        self.disposed += 1


def _applied_transaction():
    state, calls = [], []
    transaction = EditorTransaction("Parent and children")
    commands = [_AppendCommand(state, value, calls) for value in ("parent", "child", "leaf")]
    for command in commands:
        transaction.execute(command)
    calls.clear()
    return transaction, commands, state, calls


def test_rollback_retains_failed_command_and_dependencies_for_explicit_retry():
    transaction, commands, state, calls = _applied_transaction()
    commands[1].fail_undo = True

    with pytest.raises(TransactionRollbackError, match="2 command") as failure:
        transaction.rollback()

    assert failure.value.transaction is transaction
    assert failure.value.command is commands[1]
    assert isinstance(failure.value.__cause__, OSError)
    assert state == ["parent", "child"]
    assert calls == [("undo", "leaf"), ("undo", "child")]
    assert transaction.command_count == 2
    assert transaction.is_closed and transaction.is_rollback_pending
    assert [command.disposed for command in commands] == [0, 0, 1]
    for action in (transaction.commit, lambda: transaction.execute(commands[2]),
                   lambda: transaction.record_applied(commands[2])):
        with pytest.raises(RuntimeError, match="closed"):
            action()

    commands[1].fail_undo = False
    transaction.rollback()
    assert state == []
    assert calls == [("undo", "leaf"), ("undo", "child"), ("undo", "child"), ("undo", "parent")]
    assert transaction.command_count == 0
    assert transaction.is_closed and not transaction.is_rollback_pending
    assert transaction.rollback_error is None
    assert [command.disposed for command in commands] == [1, 1, 1]
    transaction.rollback()
    assert len(calls) == 4


def test_execute_preserves_original_error_and_never_undos_failed_execute():
    transaction, commands, state, calls = _applied_transaction()
    commands[1].fail_undo = True
    failed = _AppendCommand(state, "never applied", calls)
    failed.fail_execute = True

    with pytest.raises(ValueError, match="cannot execute never applied") as failure:
        transaction.execute(failed)

    recovery = failure.value.__cause__
    assert isinstance(recovery, TransactionRollbackError)
    assert recovery.transaction is transaction
    assert isinstance(recovery.__cause__, OSError)
    assert "rollback also failed" in failure.value.__notes__[0]
    assert ("undo", "never applied") not in calls
    assert state == ["parent", "child"]
    commands[1].fail_undo = False
    recovery.transaction.rollback()
    assert state == []


def test_context_manager_does_not_implicitly_retry_failed_execute_recovery():
    manager = UndoManager()
    state, calls = [], []
    first = _AppendCommand(state, "parent", calls)
    first.fail_undo = True
    failed = _AppendCommand(state, "never applied", calls)
    failed.fail_execute = True

    with pytest.raises(ValueError) as failure:
        with manager.transaction("Failing edit") as transaction:
            transaction.execute(first)
            transaction.execute(failed)

    assert calls == [("execute", "parent"), ("execute", "never applied"), ("undo", "parent")]
    assert not manager.can_undo and not manager.is_executing
    recovery = failure.value.__cause__
    assert isinstance(recovery, TransactionRollbackError)
    assert recovery.transaction.is_rollback_pending
    first.fail_undo = False
    recovery.transaction.rollback()
    assert state == []
    assert calls[-1] == ("undo", "parent")


def test_context_body_error_remains_primary_and_recovery_is_retained_after_exit():
    manager = UndoManager()
    state, calls = [], []
    command = _AppendCommand(state, "parent", calls)
    command.fail_undo = True
    body_error = LookupError("body rejected candidate")

    with pytest.raises(LookupError) as failure:
        with manager.transaction("Edit") as transaction:
            transaction.execute(command)
            raise body_error

    assert failure.value is body_error
    recovery = failure.value.__cause__
    assert isinstance(recovery, TransactionRollbackError)
    command.fail_undo = False
    recovery.transaction.rollback()
    assert state == []
    assert manager.action_journal.cursor == 0


def test_swallowed_execute_error_cannot_publish_partial_transaction():
    manager = UndoManager()
    state, calls = [], []
    command = _AppendCommand(state, "parent", calls)
    command.fail_undo = True
    failed = _AppendCommand(state, "never applied", calls)
    failed.fail_execute = True

    with pytest.raises(TransactionRollbackError) as failure:
        with manager.transaction("Edit") as transaction:
            transaction.execute(command)
            with pytest.raises(ValueError):
                transaction.execute(failed)

    assert not manager.can_undo
    command.fail_undo = False
    failure.value.transaction.rollback()
    assert state == []


def test_swallowed_execute_error_with_successful_abort_still_cannot_commit():
    manager = UndoManager()
    state, calls = [], []
    failed = _AppendCommand(state, "never applied", calls)
    failed.fail_execute = True

    with pytest.raises(RuntimeError, match="closed"):
        with manager.transaction("Edit") as transaction:
            transaction.execute(_AppendCommand(state, "parent", calls))
            with pytest.raises(ValueError):
                transaction.execute(failed)

    assert state == []
    assert not manager.can_undo


def test_context_abort_keeps_undo_callbacks_inside_suppression():
    manager = UndoManager()
    observed = []

    class Command(_AppendCommand):
        def undo(self):
            observed.append(manager.is_executing)
            super().undo()

    with pytest.raises(ValueError, match="body failed"):
        with manager.transaction("Edit") as transaction:
            transaction.execute(Command([], "parent", []))
            raise ValueError("body failed")

    assert observed == [True]
    assert not manager.is_executing and not manager.can_undo


def test_recovery_after_context_exit_keeps_callback_history_suppressed():
    manager = UndoManager()
    state, calls, observed = [], [], []

    class Command(_AppendCommand):
        def undo(self):
            observed.append(manager.is_executing)
            # Editor mutation observers use this guard before recording the
            # changes that their linked callbacks make during replay.
            if not manager.is_executing:
                manager.record(_AppendCommand([], "linked callback", []))
            super().undo()

    command = Command(state, "parent", calls)
    command.fail_undo = True
    with pytest.raises(LookupError) as failure:
        with manager.transaction("Edit") as transaction:
            transaction.execute(command)
            raise LookupError("body failed")

    assert not manager.is_executing
    command.fail_undo = False
    failure.value.__cause__.transaction.rollback()

    assert observed == [True, True]
    assert state == []
    assert manager.action_journal.entries == ()
    assert not manager.is_executing


def test_normal_transaction_transfers_ownership_to_one_replayable_compound():
    manager = UndoManager()
    state, calls = [], []
    commands = [_AppendCommand(state, value, calls) for value in ("parent", "child")]
    with manager.transaction("Edit") as transaction:
        transaction.execute(commands[0])
        commands[1].execute()
        transaction.record_applied(commands[1])

    assert transaction.is_closed and transaction.command_count == 0
    assert len(manager.action_journal.entries) == 1
    assert isinstance(manager.action_journal.entries[0].action, CompoundCommand)
    transaction.rollback()  # ownership was transferred, not aborted
    assert state == ["parent", "child"]
    manager.undo()
    assert state == []
    manager.redo()
    assert state == ["parent", "child"]
    manager.clear()
    assert [command.disposed for command in commands] == [1, 1]


def test_compound_undo_compensates_before_journal_retry():
    manager = UndoManager()
    state, calls = [], []
    commands = [_AppendCommand(state, value, calls) for value in ("parent", "child", "leaf")]
    with manager.transaction("Edit") as transaction:
        for command in commands:
            transaction.execute(command)
    commands[1].fail_undo = True
    calls.clear()

    manager.undo()

    assert state == ["parent", "child", "leaf"]
    assert calls == [("undo", "leaf"), ("undo", "child"), ("execute", "leaf")]
    assert manager.action_journal.cursor == 1 and manager.can_undo
    commands[1].fail_undo = False
    calls.clear()
    manager.undo()
    assert state == []
    assert calls == [("undo", "leaf"), ("undo", "child"), ("undo", "parent")]
    assert manager.action_journal.cursor == 0 and manager.can_redo
