"""GPU author source is compiled statically, independently of the host script domain."""

import json

import pytest

from infernux._compiler.shader import parse_shader
from infernux._source_kinds import GPU_SOURCE_SUFFIXES, is_gpu_source, is_host_python_source, source_extension
from infernux.engine.project_context import get_script_module_name, is_project_script


SOURCE = '''import infernux.shader as shader

class Tint(shader.fullscreen):
    strength = shader.parameter.float(default=0.5, range=(0.0, 1.0), label="强度")
    color = shader.parameter.color(default=(0.2, 0.4, 0.8, 1.0), hdr=True)
    active = shader.parameter.bool(default=True)

    def main(self, uv: shader.vec2) -> shader.vec4:
        wave = shader.sin(uv.x * 3.0) * self.strength
        rgb = self.color.rgb * wave
        return shader.vec4(rgb if self.active else shader.vec3(0.0), self.color.a)
'''


def test_static_fullscreen_schema_and_both_targets():
    definition = parse_shader(SOURCE, "测试.frag.py", "tint")
    assert [field.name for field in definition.parameters] == ["strength", "color", "active"]
    assert definition.parameters[0].range == (0.0, 1.0)
    assert definition.parameters[0].label == "强度"
    assert definition.parameters[1].hdr
    assert definition.parameters[2].default is True
    compiled = definition.compile()
    for target in compiled["targets"]:
        assert {(entry["name"], entry["stage"]) for entry in target["entries"]} == {
            ("vertex_main", "vertex"), ("fragment_main", "fragment"),
        }
        reflection = json.loads(target["reflection"])
        block = reflection["parameters"][0]["type"]["elementVarLayout"]["type"]
        assert [field["name"] for field in block["fields"]] == ["strength", "color", "active"]
        assert block["fields"][-1]["type"]["scalarType"] == "uint32"


def test_shader_files_never_enter_host_discovery(tmp_path):
    root = str(tmp_path)
    for suffix in GPU_SOURCE_SUFFIXES:
        path = tmp_path / "Assets" / ("物体" + suffix.upper())
        assert is_gpu_source(path)
        assert source_extension(path) == suffix
        assert not is_host_python_source(path)
        assert not is_project_script(str(path), root)
        assert get_script_module_name(str(path), root) is None
    assert is_host_python_source("物体.py")
    assert not is_host_python_source("物体.particle.py")
    assert not is_gpu_source("物体.frag.py.txt")


def test_static_source_does_not_execute_host_code(tmp_path):
    sentinel = tmp_path / "executed"
    statement = f'__import__("pathlib").Path({str(sentinel)!r}).touch()\n'
    with pytest.raises(SyntaxError) as failure:
        parse_shader(statement + SOURCE, "副作用.frag.py", "side_effect")
    assert failure.value.filename == "副作用.frag.py"
    assert failure.value.lineno == 1
    assert not sentinel.exists()


def test_generated_diagnostics_keep_author_path_and_line():
    definition = parse_shader(SOURCE.replace("uv.x", "uv.z"), "错误.frag.py", "invalid_swizzle")
    with pytest.raises(RuntimeError, match=r"错误.frag.py:9:"):
        definition.compile()
