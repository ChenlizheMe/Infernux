"""A vertex asset and its surface share parameters and the real target layouts."""

import json

import pytest

from infernux._compiler.shader import _prepare_source, parse_shader


VERTEX = '''import infernux.shader as shader
from .wave_math import wave
class Wave(shader.vert):
    shader_id = "tests/wave"
    amplitude = shader.material.float(default=0.2, range=(0.0, 1.0))
    displacement = shader.material.texture_2d(default="black")
    def deform(self, vertex: shader.VertexData, ctx: shader.VertexContext) -> shader.VertexData:
        height = wave(vertex.position_os.x, ctx.time) * self.amplitude
        texel = shader.sample_level(self.displacement, vertex.uv0, 0.0)
        vertex.position_os.y += height + texel.r
        return vertex
'''

SURFACE = '''import infernux.shader as shader
class Paint(shader.surface):
    tint = shader.material.color(default=(0.3, 0.5, 0.7, 1.0))
    amplitude = shader.material.float(default=0.2, range=(0.0, 1.0))
    def surface(self, ctx: shader.SurfaceContext) -> shader.SurfaceData:
        return shader.SurfaceData(albedo=self.tint.rgb * self.amplitude)
'''


def test_vertex_and_surface_link_shared_fields_and_compile_both_targets(tmp_path):
    vertex = tmp_path / "wave.vert.py"
    vertex.write_text(VERTEX, encoding="utf8")
    (tmp_path / "wave_math.shaderlib.py").write_text('''import infernux.shader as shader
def wave(position: float, time: float) -> float:
    return shader.sin(position * 4.0 + time * 2.0)
''', encoding="utf8")
    definition = parse_shader(SURFACE, str(tmp_path / "paint.frag.py"), "paint", vertex_path=str(vertex))
    assert definition.vertex_source == str(vertex.resolve())
    assert next(field for field in definition.parameters if field.name == "amplitude").stages == ("vertex", "fragment")
    assert next(field for field in definition.parameters if field.name == "displacement").stages == ("vertex",)
    for target in definition.compile()["targets"]:
        entries = {entry["name"]: entry for entry in target["entries"]}
        assert {"vertex_main", "vertex_shadow", "vertex_motion", "fragment_gbuffer"} <= entries.keys()
        assert [2, 0] in entries["vertex_motion"]["bindings"]
    source, libraries, schema, ranges = _prepare_source(VERTEX, str(vertex), "wave", None)
    assert json.loads(schema)["domain"] == "vertex"
    from infernux.lib import _Infernux
    compilation = _Infernux._compile_shader_program(source, str(vertex), "wave", libraries, ("spirv", "wgsl"), source_ranges=ranges)
    assert all(not target["entries"] for target in compilation["targets"])


def test_vertex_field_conflicts_and_stage_errors_locate_original_source(tmp_path):
    vertex = tmp_path / "wave.vert.py"
    vertex.write_text(VERTEX.replace('from .wave_math import wave\n', '').replace('wave(vertex.position_os.x, ctx.time)', 'ctx.time'), encoding="utf8")
    with pytest.raises(SyntaxError, match="disagree on material field") as error:
        parse_shader(SURFACE.replace("default=0.2", "default=0.3"), str(tmp_path / "paint.frag.py"), "paint", vertex_path=str(vertex))
    assert error.value.filename == str(tmp_path / "paint.frag.py")
    assert error.value.lineno == 2
    vertex.write_text(vertex.read_text(encoding="utf8").replace("ctx.time", "ctx.bad_field"), encoding="utf8")
    with pytest.raises(SyntaxError, match="Unknown shader stage input") as error:
        parse_shader(SURFACE, str(tmp_path / "paint.frag.py"), "paint", vertex_path=str(vertex))
    assert error.value.filename == str(vertex.resolve())


def test_fragment_does_not_own_the_material_vertex_selection(tmp_path):
    definition = parse_shader(SURFACE, str(tmp_path / "paint.frag.py"), "paint")
    assert definition.vertex_source is None
    coupled = SURFACE.replace('(shader.surface):', '(shader.surface):\n    vertex_shader = shader.ref("./wave.vert.py")')
    with pytest.raises(SyntaxError, match="Choose the vertex shader on the material"):
        parse_shader(coupled, str(tmp_path / "paint.frag.py"), "paint")


def test_vertex_implicit_lod_is_reported_at_the_python_call(tmp_path):
    vertex = tmp_path / "implicit.vert.py"
    source = VERTEX.replace('from .wave_math import wave\n', '').replace('wave(vertex.position_os.x, ctx.time)', 'ctx.time')
    source = source.replace('shader.sample_level(self.displacement, vertex.uv0, 0.0)', 'shader.sample(self.displacement, vertex.uv0)')
    vertex.write_text(source, encoding="utf-8")
    with pytest.raises(SyntaxError, match="fragment stage.*implicit_lod") as error:
        parse_shader(SURFACE, str(tmp_path / "paint.frag.py"), "paint", vertex_path=str(vertex))
    assert error.value.filename == str(vertex)
    assert error.value.lineno == next(index + 1 for index, line in enumerate(source.splitlines()) if 'texel =' in line)
    assert error.value.diagnostics[0]["code"] == "author.stage"


def test_vertex_fragment_effects_follow_helpers_without_restricting_the_whole_library(tmp_path):
    (tmp_path / "leaf.shaderlib.py").write_text('''import infernux.shader as gpu
def edge(value: float) -> float:
    return gpu.fwidth(value)
def pure(value: float) -> float:
    return value * 2.0
''', encoding="utf-8")
    layer = tmp_path / "layer.shaderlib.py"
    layer_source = '''import infernux.shader as gpu
from .leaf import edge, pure
def fragment_only(value: float) -> float:
    return edge(value)
def shared(value: float) -> float:
    return pure(value)
'''
    layer.write_text(layer_source, encoding="utf-8")
    source = '''import infernux.shader as shader
from .layer import fragment_only as height
class Displace(shader.vert):
    def offset(self, position: float) -> float:
        return height(position)
    def deform(self, vertex: shader.VertexData, ctx: shader.VertexContext) -> shader.VertexData:
        vertex.position_os.y += self.offset(vertex.position_os.x)
        return vertex
'''
    vertex = tmp_path / "derived.vert.py"
    vertex.write_text(source, encoding="utf-8")
    with pytest.raises(SyntaxError, match="self.offset requires the fragment stage.*derivatives") as error:
        parse_shader(SURFACE, str(tmp_path / "paint.frag.py"), "paint", vertex_path=str(vertex))
    assert (error.value.filename, error.value.lineno) == (str(vertex), 7)
    vertex.write_text(source.replace('fragment_only as height', 'shared as height'), encoding="utf-8")
    compiled = parse_shader(SURFACE, str(tmp_path / "paint.frag.py"), "paint", vertex_path=str(vertex)).compile()
    assert {target["target"] for target in compiled["targets"]} == {"spirv", "wgsl"}


def test_library_schema_exposes_per_function_stage_requirements(tmp_path):
    path = tmp_path / "sampling.shaderlib.py"
    source = '''import infernux.shader as shader
def smooth(value: float) -> float:
    return shader.fwidth(value)
def sample(image: shader.Color2D, uv: shader.vec2) -> shader.vec4:
    return shader.sample(image, uv)
def load(image: shader.Color2D, uv: shader.vec2) -> shader.vec4:
    return shader.sample_level(image, uv, 0.0)
'''
    _, _, schema, _ = _prepare_source(source, str(path), "sampling", None)
    functions = {item["name"]: item["requires"] for item in json.loads(schema)["functions"]}
    assert functions == {"smooth": ["derivatives"], "sample": ["implicit_lod"], "load": []}


def test_vertex_fields_accept_boolean_unsigned_and_three_by_three_matrices(tmp_path):
    vertex = tmp_path / "typed.vert.py"
    vertex.write_text('''import infernux.shader as shader
class Typed(shader.vert):
    enabled = shader.material.bool(default=True)
    marker = shader.material.uint(default=4275878552)
    basis = shader.material.mat3(default=((1.0, 0.2, 0.0), (0.1, 0.9, 0.0), (0.0, 0.0, 1.0)))
    def deform(self, vertex: shader.VertexData, ctx: shader.VertexContext) -> shader.VertexData:
        if self.enabled and self.marker == 4275878552:
            vertex.position_os = self.basis @ vertex.position_os
        return vertex
''', encoding="utf-8")
    definition = parse_shader(SURFACE, str(tmp_path / "paint.frag.py"), "paint", vertex_path=str(vertex))
    fields = {field.name: field for field in definition.parameters}
    assert {fields[name].kind for name in ("enabled", "marker", "basis")} == {"bool", "uint", "mat3"}
    assert all(fields[name].stages == ("vertex",) for name in ("enabled", "marker", "basis"))
    for target in definition.compile()["targets"]:
        entries = {entry["name"] for entry in target["entries"]}
        assert {"vertex_main", "vertex_shadow", "vertex_motion", "fragment_main"} <= entries


@pytest.mark.parametrize("kind,suffix", [("vert", ".vert.py"), ("frag", ".frag.py"),
                                        ("shaderlib", ".shaderlib.py"), ("shadingmodel", ".shadingmodel.py")])
def test_project_gpu_templates_compile_without_host_execution(tmp_path, kind, suffix):
    from infernux.engine.ui.project_file_ops import create_shader
    from infernux.lib import _Infernux
    created = create_shader(str(tmp_path), "CommonMath", kind)
    assert created.success, created.detail
    path = tmp_path / ("common_math" + suffix)
    assert created.created_path == str(path)
    source = path.read_text(encoding="utf-8")
    generated, libraries, schema, ranges = _prepare_source(source, str(path), "template", None)
    result = _Infernux._compile_shader_program(generated, str(path), "template", libraries, ("spirv", "wgsl"), source_ranges=ranges)
    assert {target["target"] for target in result["targets"]} == {"spirv", "wgsl"}
    assert not (tmp_path / "common_math.glsl").exists()
