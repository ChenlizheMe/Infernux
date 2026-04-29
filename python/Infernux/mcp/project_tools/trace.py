"""Minimal trace recorder for MCP tool calls."""

from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any


_active_trace: dict[str, Any] | None = None
_last_trace: dict[str, Any] | None = None


def start_trace(project_path: str, task: str = "") -> dict[str, Any]:
    global _active_trace
    _active_trace = {
        "trace_id": f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}",
        "task": str(task or ""),
        "started_at": time.time(),
        "steps": [],
    }
    return current_trace()


def stop_trace(project_path: str, save: bool = True) -> dict[str, Any]:
    global _active_trace, _last_trace
    if _active_trace is None:
        return {"active": False, "trace": None, "saved_path": ""}
    trace = dict(_active_trace)
    trace["ended_at"] = time.time()
    trace["elapsed_seconds"] = max(0.0, trace["ended_at"] - float(trace.get("started_at", trace["ended_at"])))
    saved_path = ""
    if save:
        saved_path = _save_trace(project_path, trace)
    _last_trace = trace
    _active_trace = None
    return {"active": False, "trace": trace, "saved_path": saved_path}


def current_trace() -> dict[str, Any]:
    return {"active": _active_trace is not None, "trace": _active_trace}


def last_trace() -> dict[str, Any]:
    return {"trace": _last_trace}


def record_tool_call(
    name: str,
    *,
    ok: bool,
    elapsed_ms: float = 0.0,
    arguments: dict[str, Any] | None = None,
    error: str = "",
) -> None:
    if _active_trace is None:
        return
    step = {
        "index": len(_active_trace["steps"]),
        "tool": str(name),
        "ok": bool(ok),
        "elapsed_ms": round(float(elapsed_ms), 3),
    }
    if arguments:
        step["arguments"] = _jsonable_summary(arguments)
    if error:
        step["error"] = str(error)
    _active_trace["steps"].append(step)


def list_traces(project_path: str, limit: int = 50) -> list[dict[str, Any]]:
    trace_dir = _trace_dir(project_path)
    if not os.path.isdir(trace_dir):
        return []
    entries = []
    for name in sorted(os.listdir(trace_dir), reverse=True):
        if not name.endswith(".json"):
            continue
        path = os.path.join(trace_dir, name)
        entries.append({
            "file": os.path.relpath(path, project_path).replace("\\", "/"),
            "name": name,
            "size": os.path.getsize(path),
        })
        if len(entries) >= max(int(limit), 1):
            break
    return entries


def _save_trace(project_path: str, trace: dict[str, Any]) -> str:
    trace_dir = _trace_dir(project_path)
    os.makedirs(trace_dir, exist_ok=True)
    file_path = os.path.join(trace_dir, f"{trace['trace_id']}.json")
    with open(file_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(trace, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return os.path.relpath(file_path, project_path).replace("\\", "/")


def _trace_dir(project_path: str) -> str:
    return os.path.join(os.path.abspath(project_path), ".infernux", "mcp_traces")


def _jsonable_summary(value: Any, *, max_string: int = 240) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value if len(value) <= max_string else value[:max_string] + "...<truncated>"
    if isinstance(value, dict):
        return {str(k): _jsonable_summary(v, max_string=max_string) for k, v in list(value.items())[:40]}
    if isinstance(value, (list, tuple)):
        items = list(value)
        summarized = [_jsonable_summary(v, max_string=max_string) for v in items[:40]]
        if len(items) > 40:
            summarized.append(f"...<{len(items) - 40} more>")
        return summarized
    return str(value)

