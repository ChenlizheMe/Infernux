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
            assert "external parallel backend" in str(exc)
        else:
            raise AssertionError("compute shader stage was accepted")


def test_shader_file_creation_accepts_graphics_stages(tmp_path):
    for stage in ("vert", "frag"):
        ok, error = create_shader(str(tmp_path), f"Stage_{stage}", stage)
        assert ok, error
        source = (tmp_path / f"Stage_{stage}.{stage}").read_text(encoding="utf-8")
        assert "ShaderInfo {" in source
        assert f'Name "Stage {stage.title()}"' in source
        assert "@" not in source
