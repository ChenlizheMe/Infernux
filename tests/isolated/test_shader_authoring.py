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


def test_mesh_fragment_uses_native_geometry_and_material_domain():
    source = '''import infernux.shader as shader

class Grid(shader.frag):
    tint = shader.material.color(default=(0.8, 0.3, 0.1, 1.0), hdr=True, label="颜色")
    scale = shader.material.float(default=2.0, range=(0.0, 4.0))

    def main(self, ctx: shader.mesh_input) -> shader.vec4:
        stripe = shader.step(0.5, shader.frac(ctx.position_ws.x * self.scale))
        return shader.vec4(self.tint.rgb * stripe, self.tint.a)
'''
    definition = parse_shader(source, "网格.frag.py", "mesh_grid")
    assert definition.domain == "mesh"
    for target in definition.compile()["targets"]:
        assert {(entry["name"], entry["stage"]) for entry in target["entries"]} == {
            ("vertex_main", "vertex"), ("fragment_main", "fragment"),
        }
        locations = {(resource["set"], resource["binding"]): resource for resource in target["resources"]}
        assert {(0, 0), (1, 5), (2, 1), (2, 2), (2, 3)} <= locations.keys()
        fields = locations[0, 0]["value"]["fields"]
        assert [field["name"] for field in fields] == ["tint", "scale"]


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


@pytest.mark.parametrize("source,code", [
    ("import numpy as np\n" + SOURCE, "author.import"),
    (SOURCE.replace("parameter.float(default=0.5", "parameter.int(default=0.5"), "author.parameter"),
    (SOURCE.replace("parameter.bool(default=True)", "parameter.bool(default=1)"), "author.parameter"),
])
def test_author_diagnostics_are_structured_before_native_compilation(source, code):
    with pytest.raises(SyntaxError) as failure:
        parse_shader(source, "静态检查.frag.py", "checked")
    diagnostic = failure.value.diagnostics[0]
    assert diagnostic["code"] == code
    assert diagnostic["phase"] == "author" and diagnostic["severity"] == "error"
    assert diagnostic["path"] == "静态检查.frag.py" and diagnostic["line"] > 0
    assert diagnostic["column"] > 0 and diagnostic["end_line"] >= diagnostic["line"]


def test_native_error_records_keep_author_statement_ranges_and_recover():
    definition = parse_shader(SOURCE.replace("uv.x", "uv.z"), "错误.frag.py", "invalid_swizzle")
    with pytest.raises(RuntimeError) as failure:
        definition.compile()
    errors = [item for item in failure.value.diagnostics if item["severity"] == "error"]
    assert errors and errors[0]["path"] == "错误.frag.py" and errors[0]["line"] == 9
    assert (errors[0]["column"], errors[0]["end_line"], errors[0]["end_column"]) == (9, 9, len(SOURCE.splitlines()[8]) + 1)
    assert errors[0]["phase"] == "compile" and errors[0]["code"].startswith("E")
    assert parse_shader(SOURCE, "错误.frag.py", "invalid_swizzle").compile()["targets"]


def test_packaged_standard_module_is_found_by_native_loader():
    from infernux.lib import _Infernux

    # The Windows extension loader uses extended-length DLL paths. Slang must
    # find its module before enforcing the upstream experimental-module gate.
    source = 'import slang.numerics; [shader("compute")][numthreads(1,1,1)] void main() {}'
    with pytest.raises(RuntimeError, match="experimental module"):
        _Infernux._compile_shader_program(source, "standard.compute.py", "standard")


def test_static_shader_library_dependencies_and_typed_helpers(tmp_path):
    (tmp_path / "curves.shaderlib.py").write_text('''import infernux.shader as shader

def pulse(value: shader.f32) -> shader.f32:
    return shader.smoothstep(0.0, 1.0, value)
''', encoding="utf-8")
    palette = tmp_path / "palette.shaderlib.py"
    palette.write_text('''import infernux.shader as shader
from .curves import pulse

def tint(rgb: shader.vec3, value: shader.f32) -> shader.vec3:
    return rgb * pulse(value)
''', encoding="utf-8")
    source = '''import infernux.shader as shader
from . import palette as pal

class Painted(shader.fullscreen):
    strength = shader.parameter.float(default=0.5)

    def paint(self, uv: shader.vec2) -> shader.vec3:
        return pal.tint(shader.vec3(uv, 1.0), self.strength)

    def main(self, uv: shader.vec2) -> shader.vec4:
        return shader.vec4(self.paint(uv), 1.0)
'''
    path = str(tmp_path / "painted.frag.py")
    definition = parse_shader(source, path, "painted")
    assert [item[1] for item in definition.libraries if item[1].endswith(".shaderlib.py")] == [str(tmp_path / "curves.shaderlib.py"), str(palette)]
    previous = definition.compile()
    for target in previous["targets"]:
        assert len(target["entries"]) == 2
    palette.write_text(palette.read_text(encoding="utf-8").replace("rgb * pulse(value)", "rgb * pulse(value) * 0.5"), encoding="utf-8")
    edited = parse_shader(source, path, "painted").compile()
    assert edited["targets"][0]["entries"][1]["code"] != previous["targets"][0]["entries"][1]["code"]

    # A dependent library reports its original statement, rather than columns
    # from the generated Slang expression.
    palette.write_text(palette.read_text(encoding="utf-8").replace("return rgb", "return rgb.w"), encoding="utf-8")
    with pytest.raises(RuntimeError) as library_failure:
        parse_shader(source, path, "painted").compile()
    error = next(item for item in library_failure.value.diagnostics if item["severity"] == "error")
    assert (error["path"], error["line"], error["column"]) == (str(palette), 5, 5)

    # A source cycle is diagnosed at the import, before any host code can run.
    (tmp_path / "curves.shaderlib.py").write_text('''import infernux.shader as shader
from .palette import tint
''', encoding="utf-8")
    with pytest.raises(SyntaxError, match="import cycle") as failure:
        parse_shader(source, path, "painted")
    assert failure.value.filename == str(tmp_path / "curves.shaderlib.py")
    assert failure.value.lineno == 2


def test_typed_image_port_passes_through_a_library_without_binding_numbers(tmp_path):
    (tmp_path / "sampling.shaderlib.py").write_text('''import infernux.shader as shader

def read_color(image: shader.Color2D, uv: shader.vec2) -> shader.vec4:
    return shader.sample_level(image, uv, 0.0)
''', encoding="utf-8")
    source = '''import infernux.shader as shader
from .sampling import read_color
from infernux.shader.lib.color import luminance

class Grayscale(shader.fullscreen):
    image = shader.input("scene", shader.Color2D, domain="scene_linear_hdr")
    offset: shader.vec2 = shader.parameter.vec2(default=(0.0, 0.0))
    strength = shader.parameter.float(default=0.75, range=(0.0, 1.0))

    def main(self, uv: shader.vec2) -> shader.vec4:
        color = read_color(self.image, uv + self.offset)
        value = luminance(color.rgb)
        return shader.vec4(shader.lerp(color.rgb, shader.vec3(value), self.strength), color.a)
'''
    definition = parse_shader(source, str(tmp_path / "grayscale.frag.py"), "grayscale")
    assert definition.inputs[0].semantic == "scene"
    assert definition.inputs[0].domain == "scene_linear_hdr"
    assert definition.parameters[0].kind == "vec2"
    for target in definition.compile()["targets"]:
        vertex, fragment = target["entries"]
        assert not vertex["bindings"]
        assert len(fragment["bindings"]) == 3  # image, sampler, parameter block
        image, sampler, parameters = target["resources"]
        assert image["name"] == "inx_image.texture"
        assert sampler["name"] == "inx_image.sampler"
        assert image["binding"] != sampler["binding"]
        offset, strength = parameters["value"]["fields"]
        assert (offset["name"], offset["columns"], offset["offset"]) == ("offset", 2, 0)
        assert (strength["name"], strength["offset"]) == ("strength", 8)
