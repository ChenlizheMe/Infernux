"""Embedded MCP integration for Infernux Editor."""

from Infernux.mcp.capabilities import current_config
from Infernux.mcp.server import endpoint_url, is_running, start_server, stop_server

__all__ = ["current_config", "endpoint_url", "is_running", "start_server", "stop_server"]
