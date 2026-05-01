"""Transactional MCP tools for long-horizon agent edits."""

from __future__ import annotations

from Infernux.mcp.project_tools import transactions
from Infernux.mcp.tools.common import ok, register_tool_metadata


def register_transaction_tools(mcp, project_path: str) -> None:
    _register_metadata()

    @mcp.tool(name="transaction.begin")
    def transaction_begin(label: str = "") -> dict:
        """Start a best-effort MCP transaction for generated file/asset edits."""
        return ok(transactions.begin(project_path, label=label))

    @mcp.tool(name="transaction.status")
    def transaction_status() -> dict:
        """Return the active or last MCP transaction status."""
        return ok(transactions.status())

    @mcp.tool(name="transaction.commit")
    def transaction_commit() -> dict:
        """Commit the active MCP transaction."""
        return ok(transactions.commit())

    @mcp.tool(name="transaction.rollback")
    def transaction_rollback() -> dict:
        """Rollback tracked file/asset mutations from the active transaction."""
        return ok(transactions.rollback())


def _register_metadata() -> None:
    metadata = {
        "transaction.begin": "Start a best-effort transaction for long-horizon MCP mutations.",
        "transaction.status": "Inspect active/last MCP transaction state.",
        "transaction.commit": "Accept tracked MCP mutations.",
        "transaction.rollback": "Restore tracked paths from the active MCP transaction.",
    }
    for name, summary in metadata.items():
        register_tool_metadata(
            name,
            summary=summary,
            side_effects=["Reads or mutates MCP transaction bookkeeping under .infernux/mcp_transactions."],
            recovery=["Use transaction.status before rollback; use asset.list to inspect generated paths after rollback."],
            concepts={"MCP Transaction": "A best-effort recovery boundary for multi-step agent edits."},
            next_suggested_tools=["transaction.status", "mcp.trace.current"],
        )
