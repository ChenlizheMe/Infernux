"""MCP management tools for project-defined agent tools."""

from __future__ import annotations

from Infernux.mcp.project_tools.registry import get_project_tool_registry
from Infernux.mcp.project_tools.trace import (
    clear_session_log,
    current_trace,
    last_trace,
    list_traces,
    read_session_log,
    session_log_info,
    start_trace,
    stop_trace,
)
from Infernux.mcp.capabilities import feature_enabled
from Infernux.mcp.tools.common import main_thread, ok, register_tool_metadata


def register_project_tool_management(mcp, project_path: str) -> None:
    _register_metadata()
    registry = get_project_tool_registry(project_path)
    registry.configure(project_path)

    @mcp.tool(name="project_tools.list")
    def project_tools_list() -> dict:
        """List project-defined MCP tools and load reports."""
        return ok(registry.list_tools())

    @mcp.tool(name="project_tools.reload")
    def project_tools_reload() -> dict:
        """Rediscover project-defined MCP tools from Assets/AgentTools."""
        return main_thread("project_tools.reload", registry.reload)

    @mcp.tool(name="project_tools.validate")
    def project_tools_validate(path: str = "") -> dict:
        """Validate one project tool file or every discovered tool file."""
        return main_thread("project_tools.validate", lambda: registry.validate(path))

    @mcp.tool(name="project_tools.explain")
    def project_tools_explain(name: str) -> dict:
        """Explain one project-defined MCP tool."""
        return main_thread("project_tools.explain", lambda: registry.explain(name))

    @mcp.tool(name="project_tools.audit")
    def project_tools_audit(limit: int = 100) -> dict:
        """Return recent project tool load/call audit events."""
        return ok(registry.audit(limit))

    if feature_enabled("trace_recorder"):
        @mcp.tool(name="mcp.trace.start")
        def mcp_trace_start(task: str = "") -> dict:
            """Start recording MCP tool call trace metadata."""
            return ok(start_trace(project_path, task))

        @mcp.tool(name="mcp.trace.stop")
        def mcp_trace_stop(save: bool = True) -> dict:
            """Stop the active MCP trace and optionally save it to .infernux/mcp_traces."""
            return ok(stop_trace(project_path, save=save))

        @mcp.tool(name="mcp.trace.current")
        def mcp_trace_current() -> dict:
            """Return the currently active MCP trace."""
            return ok(current_trace())

        @mcp.tool(name="mcp.trace.last")
        def mcp_trace_last() -> dict:
            """Return the last stopped MCP trace."""
            return ok(last_trace())

        @mcp.tool(name="mcp.trace.list")
        def mcp_trace_list(limit: int = 50) -> dict:
            """List saved MCP trace files."""
            return ok({"traces": list_traces(project_path, limit=limit)})

    if feature_enabled("session_call_log"):
        @mcp.tool(name="mcp.session_log.info")
        def mcp_session_log_info() -> dict:
            """Return the current per-editor-session MCP call log path and size."""
            return ok(session_log_info(project_path))

        @mcp.tool(name="mcp.session_log.read")
        def mcp_session_log_read(limit: int = 200) -> dict:
            """Read recent MCP calls from the current editor session log."""
            return ok(read_session_log(project_path, limit=limit))

        @mcp.tool(name="mcp.session_log.clear")
        def mcp_session_log_clear() -> dict:
            """Clear the current editor session MCP call log."""
            return ok(clear_session_log(project_path))


def register_project_defined_tools(mcp, project_path: str) -> dict:
    if not feature_enabled("project_defined_tools"):
        return {"registered": [], "disabled": True}
    registry = get_project_tool_registry(project_path)
    registry.configure(project_path)
    registry.discover()
    return registry.register_with_mcp(mcp)


def _register_metadata() -> None:
    for name, summary in {
        "project_tools.list": "List project-defined MCP tools.",
        "project_tools.reload": "Rediscover and register project-defined MCP tools.",
        "project_tools.validate": "Validate project tool loadability and schema quality.",
        "project_tools.explain": "Explain one project-defined MCP tool.",
        "project_tools.audit": "Return project tool load/call audit events.",
        "mcp.trace.start": "Start an MCP tool-call trace.",
        "mcp.trace.stop": "Stop and optionally save the active MCP trace.",
        "mcp.trace.current": "Return the active MCP trace.",
        "mcp.trace.last": "Return the last stopped MCP trace.",
        "mcp.trace.list": "List saved MCP trace files.",
        "mcp.session_log.info": "Return current session MCP call log file info.",
        "mcp.session_log.read": "Read recent current-session MCP call log entries.",
        "mcp.session_log.clear": "Clear the current-session MCP call log.",
    }.items():
        register_tool_metadata(name, summary=summary)

