"""Registered GPU stages and generated graphics shader sources."""

from __future__ import annotations

import pytest

from infernux.core.shader import Shader
from infernux.engine.ui.project_file_ops import create_shader


def test_shader_control_api_rejects_unregistered_stages():
    for operation in (
        Shader.is_loaded,
        Shader.reload,
    ):
        with pytest.raises(ValueError, match="registered GPU source stage"):
            operation("parallel", "invalid_stage")


def test_shader_file_creation_accepts_graphics_stages(tmp_path):
    for stage in ("vert", "frag"):
        ok, error = create_shader(str(tmp_path), f"Stage_{stage}", stage)
        assert ok, error
        path = tmp_path / f"stage_{stage}.{stage}.py"
        source = path.read_text(encoding="utf-8")
        if stage == "frag":
            from infernux._compiler.shader import parse_shader
            definition = parse_shader(source, str(path), "created_fragment")
            assert definition.surface and definition.name == "StageFrag"
            assert definition.info["shader_id"] == "project/stage_frag"
            assert {"base_color", "albedo_map"} <= {field.name for field in definition.parameters}
        else:
            import json
            from infernux._compiler.shader import _prepare_source
            _, _, schema, _ = _prepare_source(source, str(path), "created_vertex", None)
            assert json.loads(schema)["domain"] == "vertex"
            assert json.loads(schema)["info"]["shader_id"] == "project/stage_vert"
        assert "@" not in source
