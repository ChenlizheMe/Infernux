"""Configurable capability gates for the Infernux MCP layer."""

from __future__ import annotations

import copy
import json
import os
from typing import Any


CONFIG_REL_PATH = os.path.join("ProjectSettings", "mcp_capabilities.json")

DEFAULT_CAPABILITY_CONFIG: dict[str, Any] = {
    "enabled": True,
    "profile": "research_full",
    "write_default_config_on_bootstrap": True,
    "features": {
        "self_description": True,
        "executable_contracts": True,
        "runtime_observation": True,
        "batch_execution": True,
        "transactions": True,
        "trace_recorder": True,
        "session_call_log": True,
        "trace_to_tool_evolution": True,
        "project_defined_tools": True,
        "semantic_workflows": True,
        "safety_validation": True,
        "discovery_files": True,
    },
    "tool_groups": {
        "docs": True,
        "api": True,
        "project": True,
        "editor": True,
        "scene": True,
        "hierarchy": True,
        "asset": True,
        "material": True,
        "renderstack": True,
        "console": True,
        "camera": True,
        "runtime": True,
        "ui": True,
        "project_tool_management": True,
        "project_defined_tools": True,
        "research": True,
        "transactions": True,
    },
    "disabled_tools": [],
    "limits": {
        "main_thread_timeout_ms": 30000,
        "project_tool_timeout_ms": 60000,
        "trace_argument_max_string": 240,
        "session_log_result_max_string": 480,
        "batch_max_steps": 100,
        "transaction_max_tracked_paths": 1000,
    },
    "contracts": {
        "require_summary": True,
        "require_recovery": True,
        "require_side_effects_for_mutations": True,
        "grade_threshold": 0.70,
    },
    "evolution": {
        "suggest_project_tools_from_failed_traces": True,
        "suggest_project_tools_from_repeated_sequences": True,
        "min_repeated_sequence_length": 3,
        "generated_tool_root": "Assets/AgentTools",
    },
}

_CURRENT_CONFIG: dict[str, Any] = copy.deepcopy(DEFAULT_CAPABILITY_CONFIG)
_PROJECT_PATH = ""


def configure(project_path: str, *, write_default: bool = True) -> dict[str, Any]:
    """Load, merge, and optionally materialize project MCP capability config."""
    global _CURRENT_CONFIG, _PROJECT_PATH
    _PROJECT_PATH = os.path.abspath(project_path or "")
    config = load_capability_config(_PROJECT_PATH)
    _CURRENT_CONFIG = config
    if write_default and bool(config.get("write_default_config_on_bootstrap", True)):
        write_default_config(_PROJECT_PATH)
    return copy.deepcopy(_CURRENT_CONFIG)


def current_config() -> dict[str, Any]:
    return copy.deepcopy(_CURRENT_CONFIG)


def project_path() -> str:
    return _PROJECT_PATH


def config_path(project_path: str | None = None) -> str:
    root = os.path.abspath(project_path or _PROJECT_PATH or "")
    return os.path.join(root, CONFIG_REL_PATH)


def load_capability_config(project_path: str) -> dict[str, Any]:
    config = copy.deepcopy(DEFAULT_CAPABILITY_CONFIG)
    path = config_path(project_path)
    if not path or not os.path.isfile(path):
        return config
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            _deep_merge(config, data)
    except Exception:
        return config
    return config


def write_default_config(project_path: str | None = None) -> str:
    """Write a complete default-on config file if it does not exist yet."""
    path = config_path(project_path)
    if not path:
        return ""
    if os.path.isfile(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(DEFAULT_CAPABILITY_CONFIG, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return path


def save_config(config: dict[str, Any] | None = None, project_path: str | None = None) -> str:
    global _CURRENT_CONFIG
    if config is not None:
        _CURRENT_CONFIG = _normalized_config(config)
    path = config_path(project_path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(_CURRENT_CONFIG, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return path


def is_enabled() -> bool:
    return bool(_CURRENT_CONFIG.get("enabled", True))


def feature_enabled(name: str) -> bool:
    return bool((_CURRENT_CONFIG.get("features") or {}).get(name, True))


def tool_group_enabled(name: str) -> bool:
    return bool((_CURRENT_CONFIG.get("tool_groups") or {}).get(name, True))


def tool_enabled(name: str) -> bool:
    return str(name) not in set(str(item) for item in _CURRENT_CONFIG.get("disabled_tools", []))


def limit(name: str, default: Any = None) -> Any:
    return (_CURRENT_CONFIG.get("limits") or {}).get(name, default)


def set_feature(name: str, enabled: bool) -> dict[str, Any]:
    _CURRENT_CONFIG.setdefault("features", {})[str(name)] = bool(enabled)
    return current_config()


def set_tool_group(name: str, enabled: bool) -> dict[str, Any]:
    _CURRENT_CONFIG.setdefault("tool_groups", {})[str(name)] = bool(enabled)
    return current_config()


def set_tool_enabled(name: str, enabled: bool) -> dict[str, Any]:
    disabled = set(str(item) for item in _CURRENT_CONFIG.get("disabled_tools", []))
    if enabled:
        disabled.discard(str(name))
    else:
        disabled.add(str(name))
    _CURRENT_CONFIG["disabled_tools"] = sorted(disabled)
    return current_config()


def _normalized_config(config: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(DEFAULT_CAPABILITY_CONFIG)
    _deep_merge(merged, config if isinstance(config, dict) else {})
    return merged


def _deep_merge(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_merge(target[key], value)
        else:
            target[key] = value
