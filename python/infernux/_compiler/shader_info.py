"""Literal GPU asset identity, visibility and inherited render-state defaults."""

from __future__ import annotations

import ast
import math
from pathlib import Path

from .diagnostics import parse_gpu_module
from .slang_ast import SlangEmitter, qualified


INFO_DEFAULTS = {
    "shader_id": None, "hidden": False, "surface_type": "opaque", "render_queue": -1,
    "cull_mode": "back", "depth_write": None, "depth_test": "", "blend": "",
    "pass_tag": "", "stencil": "", "alpha_clip": None,
    "cast_shadows": True, "receive_shadows": True,
}
_CHOICES = {
    "surface_type": {"opaque", "transparent"}, "cull_mode": {"back", "front", "none"},
    "depth_test": {"", "off", "never", "less", "equal", "less_equal", "greater", "not_equal", "greater_equal", "always"},
    "blend": {"", "off", "alpha", "additive", "premultiply", "multiply"},
}


def info_field(statement):
    target = (statement.targets[0] if isinstance(statement, ast.Assign) and len(statement.targets) == 1
              else statement.target if isinstance(statement, ast.AnnAssign) else None)
    return qualified(target) if isinstance(target, ast.Name) else ""


def read_shader_info(source, source_path):
    from infernux._source_kinds import source_extension

    suffix = source_extension(source_path)
    result = dict(INFO_DEFAULTS, shader_id=Path(source_path).name[:-len(suffix)])
    owner = SlangEmitter(source, source_path)
    classes = [node for node in parse_gpu_module(source, source_path).body if isinstance(node, ast.ClassDef)]
    declared = set()
    for declaration in classes:
        for statement in declaration.body:
            field = info_field(statement)
            if field not in INFO_DEFAULTS:
                continue
            if field in declared:
                owner.error(statement, "Duplicate shader declaration: " + field, code="author.metadata")
            try:
                value = ast.literal_eval(statement.value)
            except (ValueError, TypeError):
                owner.error(statement, "Shader metadata uses literal values", code="author.metadata")
            if field in _CHOICES:
                valid = isinstance(value, str) and value in _CHOICES[field]
            elif field in {"hidden", "cast_shadows", "receive_shadows"}:
                valid = type(value) is bool
            elif field == "depth_write":
                valid = value is None or type(value) is bool
            elif field == "alpha_clip":
                valid = value is None or (type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1)
            elif field == "render_queue":
                valid = type(value) is int and 0 <= value <= 2147483647
            else:
                valid = isinstance(value, str)
            if field == "shader_id":
                valid = valid and bool(value.strip()) and "\\" not in value and all(value.split("/"))
            if not valid:
                owner.error(statement, "Invalid shader declaration: " + field, code="author.metadata")
            result[field] = value
            declared.add(field)
    if result["surface_type"] == "transparent":
        for field, value in (("blend", "alpha"), ("depth_write", False), ("render_queue", 3000)):
            if field not in declared:
                result[field] = value
    return result
