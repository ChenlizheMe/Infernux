"""Console MCP tools."""

from __future__ import annotations

from Infernux.mcp.tools.common import main_thread


def register_console_tools(mcp) -> None:
    @mcp.tool(name="console.read")
    def console_read(limit: int = 100, levels: list[str] | None = None) -> dict:
        """Read recent DebugConsole entries."""

        def _read():
            from Infernux.debug import DebugConsole
            allowed = {str(level).upper() for level in (levels or [])}
            entries = []
            for entry in DebugConsole.instance().get_entries()[-max(int(limit), 1):]:
                level = getattr(entry.log_type, "name", str(entry.log_type)).upper()
                if allowed and level not in allowed:
                    continue
                entries.append({
                    "time": entry.get_formatted_time(),
                    "level": level,
                    "message": entry.message,
                    "source_file": entry.source_file,
                    "source_line": entry.source_line,
                })
            return {"entries": entries}

        return main_thread("console.read", _read)
