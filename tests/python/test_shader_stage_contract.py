"""Shader stage surface is graphics-only and rejects compute explicitly."""

from __future__ import annotations

from infernux.core.shader import Shader
from infernux.engine.ui.project_file_ops import create_shader


def test_shader_control_api_rejects_compute():
    for operation in (
        lambda: Shader.is_loaded("parallel", "compute"),
        lambda: Shader.reload("parallel", "compute"),
    ):
        try:
            operation()
        except ValueError as exc:
            assert "shader_type must be 'vertex' or 'fragment'" in str(exc)
        else:
            raise AssertionError("compute shader stage was accepted")


def test_shader_file_creation_accepts_graphics_stages(tmp_path):
    for stage in ("vert", "frag"):
        ok, error = create_shader(str(tmp_path), f"Stage_{stage}", stage)
        assert ok, error
        path = tmp_path / ("stage_frag.frag.py" if stage == "frag" else "Stage_vert.vert")
        source = path.read_text(encoding="utf-8")
        if stage == "frag":
            from infernux._compiler.shader import parse_shader
            definition = parse_shader(source, str(path), "created_fragment")
            assert definition.surface and definition.name == "StageFrag"
            assert definition.info["shader_id"] == "project/stage_frag"
            assert {"base_color", "albedo_map"} <= {field.name for field in definition.parameters}
        else:
            assert "ShaderInfo {" in source and 'Name "Stage Vert"' in source
        assert "@" not in source
