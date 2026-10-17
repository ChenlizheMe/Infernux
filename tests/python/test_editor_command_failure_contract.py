from __future__ import annotations


from infernux.engine.interaction import (
    CommandStatus,
    EditorCommand,
    EditorCommandRegistry,
)


def _raise(message: str):
    raise RuntimeError(message)


def test_execute_failure_has_one_structured_error_surface():
    registry = EditorCommandRegistry()
    registry.register(
        EditorCommand(
            "probe.execute",
            lambda _context: _raise("handler failed"),
        )
    )

    result = registry.execute("probe.execute")

    assert result.status is CommandStatus.FAILED
    assert result.message == "handler failed"
