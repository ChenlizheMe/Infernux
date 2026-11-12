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
