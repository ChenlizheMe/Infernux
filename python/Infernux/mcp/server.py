"""Embedded HTTP MCP server for Infernux Editor."""

from __future__ import annotations

import threading
from typing import Optional

from Infernux.debug import Debug

HOST = "127.0.0.1"
PORT = 9713

_server_thread: Optional[threading.Thread] = None
_server = None
_project_path = ""


def start_server(project_path: str, *, host: str = HOST, port: int = PORT) -> bool:
    """Start the embedded HTTP MCP server if it is not already running."""
    global _server_thread, _server, _project_path

    if _server_thread is not None and _server_thread.is_alive():
        return True

    try:
        FastMCP = _import_fastmcp()
    except Exception as exc:
        Debug.log_warning(
            "Infernux MCP disabled: install PyPI packages 'mcp' and 'fastmcp' to enable "
            f"the embedded HTTP server ({exc})."
        )
        return False

    _project_path = project_path
    _server = FastMCP("Infernux Editor")
    from Infernux.mcp.tools import register_all_tools
    register_all_tools(_server, project_path)

    def _run() -> None:
        last_error = None
        try:
            for transport in ("http", "streamable-http"):
                try:
                    _server.run(transport=transport, host=host, port=int(port))
                    return
                except Exception as exc:
                    last_error = exc
                    if transport == "streamable-http":
                        raise
        except Exception as exc:
            Debug.log_error(f"Infernux MCP HTTP server stopped: {exc or last_error}")

    _server_thread = threading.Thread(target=_run, name="InfernuxMCPHTTP", daemon=True)
    _server_thread.start()
    Debug.log_internal(f"Infernux MCP HTTP server starting at http://{host}:{port}/mcp")
    return True


def stop_server() -> None:
    """Best-effort stop hook for editor shutdown."""
    global _server
    server = _server
    _server = None
    for method_name in ("stop", "shutdown", "close"):
        method = getattr(server, method_name, None)
        if callable(method):
            try:
                method()
            except Exception as exc:
                Debug.log_suppressed(f"Infernux.mcp.stop_server.{method_name}", exc)
            break


def is_running() -> bool:
    return _server_thread is not None and _server_thread.is_alive()


def endpoint_url() -> str:
    return f"http://{HOST}:{PORT}/mcp"


def _import_fastmcp():
    try:
        from fastmcp import FastMCP
        return FastMCP
    except Exception as first:
        try:
            from mcp.server.fastmcp import FastMCP
            return FastMCP
        except Exception as second:
            raise ImportError(
                "Need PyPI packages 'mcp' and 'fastmcp' (see ProjectSettings/requirements.txt). "
                f"Primary import failed: {first!r}; fallback failed: {second!r}"
            ) from second
