"""Editor transaction builder with dependency-safe, retryable rollback."""

from __future__ import annotations

import uuid
from contextlib import nullcontext


class TransactionRollbackError(RuntimeError):
    """An abort stopped at a command that could not be undone.

    ``transaction.rollback()`` explicitly retries the remaining commands. The
    exception retains the transaction even when its context manager has exited.
    """

    def __init__(self, transaction: EditorTransaction, command) -> None:
        self.transaction = transaction
        self.command = command
        command_name = str(getattr(command, "description", "") or type(command).__name__)
        super().__init__(
            f"Rollback of editor transaction '{transaction.description}' stopped at "
            f"'{command_name}'; {transaction.command_count} command(s) still need undo. "
            "Retry with this exception's transaction.rollback()."
        )


class EditorTransaction:
    """Execute related commands and publish one journal action.

    Commands must provide the strong exception guarantee: a failing execute or
    undo leaves that command's state unchanged. A failed execute is never blindly
    undone. A failed rollback retains the failed command and its predecessors;
    earlier commands may be dependencies and cannot safely be undone past it.
    """

    def __init__(self, description: str, transaction_id: str = "", *, rollback_scope=None) -> None:
        self.description = str(description or "Transaction")
        self.transaction_id = str(transaction_id or uuid.uuid4().hex)
        self._commands: list = []
        self._state = "open"
        self._rollback_error: TransactionRollbackError | None = None
        # A manager-created transaction must retain replay suppression even
        # when recovery is explicitly retried after its with block has exited.
        self._rollback_scope = rollback_scope or nullcontext

    @property
    def command_count(self) -> int:
        """Commands still owned by this builder (including pending rollback)."""
        return len(self._commands)

    @property
    def is_closed(self) -> bool:
        """Whether execute, record_applied and commit are no longer allowed."""
        return self._state != "open"

    @property
    def is_rollback_pending(self) -> bool:
        return self._state == "rolling_back"

    @property
    def rollback_error(self) -> TransactionRollbackError | None:
        """Last failed recovery attempt, cleared once rollback succeeds."""
        return self._rollback_error

    def execute(self, command) -> None:
        self._require_open()
        try:
            command.execute()
        except Exception as error:
            self._rollback_after_error(error)
            raise
        self._commands.append(command)

    def record_applied(self, command) -> None:
        self._require_open()
        self._commands.append(command)

    def rollback(self) -> None:
        """Undo in reverse dependency order, preserving progress on failure.

        Calling this again after TransactionRollbackError is an explicit retry;
        successfully undone commands are never undone a second time.
        """
        if self._state in {"committed", "rolled_back"}:
            return
        with self._rollback_scope():
            self._rollback_commands()

    def _rollback_commands(self) -> None:
        self._state = "rolling_back"
        while self._commands:
            command = self._commands[-1]
            try:
                command.undo()
            except Exception as error:
                self._rollback_error = TransactionRollbackError(self, command)
                raise self._rollback_error from error
            self._commands.pop()
            # These commands will never enter history. Release snapshots only
            # after their undo succeeded; a failed command still needs them.
            from .action_journal import EditorActionJournal

            EditorActionJournal._dispose_action(command)
        self._state = "rolled_back"
        self._rollback_error = None

    def _rollback_after_error(self, error: Exception) -> None:
        try:
            self.rollback()
        except TransactionRollbackError as recovery_error:
            error.add_note(
                "Editor transaction rollback also failed. The chained "
                "TransactionRollbackError retains the transaction for explicit retry."
            )
            raise error from recovery_error

    def commit(self):
        self._require_open()
        if not self._commands:
            self._state = "committed"
            return None
        from infernux.engine.undo._base import CompoundCommand

        command = CompoundCommand(self._commands, self.description)
        self._commands.clear()
        self._state = "committed"
        return command

    def _require_open(self) -> None:
        if self.is_closed:
            raise RuntimeError("editor transaction is already closed")
