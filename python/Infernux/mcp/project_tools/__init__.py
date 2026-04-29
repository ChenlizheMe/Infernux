"""Project-defined MCP tool extension API."""

from .decorators import AgentToolMetadata, InxAgentToolset, agent_action, agent_tool
from .registry import ProjectToolRegistry, get_project_tool_registry

__all__ = [
    "AgentToolMetadata",
    "InxAgentToolset",
    "ProjectToolRegistry",
    "agent_action",
    "agent_tool",
    "get_project_tool_registry",
]

