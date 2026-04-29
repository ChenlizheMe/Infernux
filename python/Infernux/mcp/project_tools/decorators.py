"""Decorators for project-defined MCP tools."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class AgentToolMetadata:
    name: str
    summary: str = ""
    tags: list[str] = field(default_factory=list)
    generated: bool = False
    source_trace: str = ""
    source_traces: list[str] = field(default_factory=list)
    validation: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


def agent_tool(
    *,
    name: str,
    summary: str = "",
    tags: list[str] | tuple[str, ...] | None = None,
    generated: bool = False,
    source_trace: str = "",
    source_traces: list[str] | tuple[str, ...] | None = None,
    validation: str = "",
    **extra: Any,
) -> Callable:
    """Mark a function as a project-defined MCP tool.

    Project tools are ordinary project Python code. This decorator only adds
    metadata so the MCP registry can discover and expose the function.
    """
    if not name or not str(name).strip():
        raise ValueError("agent_tool requires a non-empty name.")

    def _decorate(fn: Callable) -> Callable:
        setattr(fn, "__inx_agent_tool__", AgentToolMetadata(
            name=str(name).strip(),
            summary=str(summary or ""),
            tags=[str(tag) for tag in (tags or [])],
            generated=bool(generated),
            source_trace=str(source_trace or ""),
            source_traces=[str(item) for item in (source_traces or [])],
            validation=str(validation or ""),
            extra=dict(extra),
        ))
        return fn

    return _decorate


def agent_action(
    *,
    name: str = "",
    summary: str = "",
    tags: list[str] | tuple[str, ...] | None = None,
    generated: bool = False,
    source_trace: str = "",
    source_traces: list[str] | tuple[str, ...] | None = None,
    validation: str = "",
    **extra: Any,
) -> Callable:
    """Mark an ``InxAgentToolset`` method as a project MCP action."""

    def _decorate(fn: Callable) -> Callable:
        setattr(fn, "__inx_agent_action__", AgentToolMetadata(
            name=str(name or "").strip(),
            summary=str(summary or ""),
            tags=[str(tag) for tag in (tags or [])],
            generated=bool(generated),
            source_trace=str(source_trace or ""),
            source_traces=[str(item) for item in (source_traces or [])],
            validation=str(validation or ""),
            extra=dict(extra),
        ))
        return fn

    return _decorate


class InxAgentToolset:
    """Base class for grouping project MCP tools under one namespace."""

    namespace = "project"
    tags: list[str] = []

