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


@pytest.mark.parametrize("expression,kind", [
    ("shader.sin(1) + shader.sqrt(2) + 1 / 2", "float"),
    ("shader.sin(shader.ivec3(1, 2, 3))", "float3"),
    ("shader.ivec3(1, 3, -5) / 2", "float3"),
    ("shader.sign(shader.vec3(-1.0, 0.0, 1.0))", "int3"),
    ("shader.dot(shader.ivec3(1, 2, 3), shader.ivec3(3, 2, 1))", "int"),
    ("shader.frame_time() + shader.frame_delta_time()", "float"),
])
def test_intrinsic_results_keep_their_native_types(expression, kind):
    result = "shader.vec4(value, 1.0)" if kind.endswith("3") else "shader.vec4(value, 0.0, 0.0, 1.0)"
    source = f'''import infernux.shader as shader
class Math(shader.fullscreen):
    def main(self, uv: shader.vec2) -> shader.vec4:
        value = {expression}
        return {result}
'''
    definition = parse_shader(source, "数学.frag.py", "intrinsic_math")
    assert f"{kind} value;" in definition.generated_source
    assert {target["target"] for target in definition.compile()["targets"]} == {"spirv", "wgsl"}


def test_frame_globals_are_optional_and_follow_library_dependencies(tmp_path):
    ordinary = parse_shader(SOURCE, "ordinary.frag.py", "ordinary")
    assert "import inx_frame;" not in ordinary.generated_source
    assert all(resource["name"] != "inx_globals" for target in ordinary.compile()["targets"]
               for resource in target["resources"])
    (tmp_path / "clock.shaderlib.py").write_text('''import infernux.shader as gpu
def time() -> float:
    return gpu.frame_time() + gpu.frame_delta_time()
''', encoding="utf-8")
    source = '''import infernux.shader as shader
from .clock import time
class Clock(shader.fullscreen):
    def main(self, uv: shader.vec2) -> shader.vec4:
        return shader.vec4(time(), uv.x, uv.y, 1.0)
'''
    definition = parse_shader(source, str(tmp_path / "clock.frag.py"), "clock")
    assert "import inx_frame;" not in definition.generated_source
    for target in definition.compile()["targets"]:
        globals = next(resource for resource in target["resources"] if resource["name"] == "inx_globals")
        assert (globals["set"], globals["binding"], globals["value"]["byte_size"]) == (2, 0, 128)
        assert [field["offset"] for field in globals["value"]["fields"]] == list(range(0, 128, 16))
    with pytest.raises(SyntaxError, match="takes no arguments"):
        parse_shader(source.replace("time()", "shader.frame_time(1.0)"), str(tmp_path / "invalid.frag.py"), "invalid")


def test_mesh_fragment_uses_native_geometry_and_material_domain():
    source = '''import infernux.shader as shader

class Grid(shader.frag):
    alpha_clip = 0.3
    tint = shader.material.color(default=(0.8, 0.3, 0.1, 1.0), hdr=True, label="颜色")
    scale = shader.material.float(default=2.0, range=(0.0, 4.0))
    image = shader.material.texture_2d(default="white", label="纹理")
    detail = shader.material.texture_2d(default="normal")

    def describe(self, normal: shader.vec3) -> shader.SurfaceData:
        return shader.SurfaceData(albedo=self.tint.rgb, normal_ws=normal, alpha=self.tint.a)

    def main(self, ctx: shader.mesh_input) -> shader.vec4:
        stripe = shader.step(0.5, shader.frac(ctx.position_ws.x * self.scale))
        image = shader.sample(self.image, ctx.uv0)
        detail = shader.sample_level(self.detail, ctx.uv0, 0.0)
        surface = self.describe(ctx.normal_ws)
        return shader.vec4(surface.albedo * stripe, surface.alpha) * image * detail
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
        assert [field["name"] for field in fields] == ["tint", "scale", "_alpha_clip_threshold"]
        assert [(locations[0, index]["name"], locations[0, index]["type"]) for index in range(1, 5)] == [
            ("inx_image.texture", 2), ("inx_image.sampler", 4),
            ("inx_detail.texture", 2), ("inx_detail.sampler", 4),
        ]
    texture_only = '''import infernux.shader as shader
class Sample(shader.frag):
    image = shader.material.texture_2d()
    def main(self, ctx: shader.mesh_input) -> shader.vec4:
        return shader.sample(self.image, ctx.uv0)
'''
    for target in parse_shader(texture_only, "纹理.frag.py", "texture_only").compile()["targets"]:
        parameters = next(resource for resource in target["resources"] if resource["set"] == 0 and resource["type"] == 0)
        assert [field["name"] for field in parameters["value"]["fields"]] == ["_alpha_clip_threshold"]


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


def test_each_temporal_input_has_engine_owned_validity_and_typed_library_calls(tmp_path):
    (tmp_path / "history.shaderlib.py").write_text('''import infernux.shader as shader
def sample(image: shader.History2D, uv: shader.vec2) -> shader.vec4:
    if image.valid:
        return shader.sample_level(image, uv, 0.0)
    return shader.vec4(0.0)
''', encoding="utf-8")
    source = '''import infernux.shader as shader
from .history import sample
class Temporal(shader.fullscreen):
    first = shader.input("previous_color", shader.History2D)
    second = shader.input("previous_moments", shader.History2D)
    gain = shader.parameter.float(default=0.5)
    def main(self, uv: shader.vec2) -> shader.vec4:
        return sample(self.first, uv) * self.gain + sample(self.second, uv)
'''
    definition = parse_shader(source, str(tmp_path / "temporal.frag.py"), "temporal")
    assert [field.name for field in definition.parameters] == ["gain"]
    assert [field.kind for field in definition.inputs] == ["History2D", "History2D"]
    for target in definition.compile()["targets"]:
        resources = target["resources"]
        parameters = next(resource for resource in resources if resource["name"] == "inx_parameters")
        assert parameters["value"]["byte_size"] == 16
        assert [(field["name"], field["offset"]) for field in parameters["value"]["fields"]] == [
            ("gain", 0), ("_inx_history_valid_first", 4), ("_inx_history_valid_second", 8)]
        assert [resource["name"] for resource in resources if ".texture" in resource["name"]] == [
            "inx_first.texture", "inx_second.texture"]
    with pytest.raises(SyntaxError, match="reserved for engine state"):
        parse_shader(source.replace("gain =", "_inx_history_valid_first ="),
                     str(tmp_path / "invalid.frag.py"), "invalid")


def test_active_resource_tree_owns_builtin_models_and_libraries(tmp_path, monkeypatch):
    from pathlib import Path
    from infernux.resources import get_package_resources_path

    root = tmp_path / "resources"
    library = root / "shaders" / "lib"
    library.mkdir(parents=True)
    pbr = Path(get_package_resources_path()) / "shaders" / "pbr.shadingmodel.py"
    (root / "shaders" / pbr.name).write_text(pbr.read_text(encoding="utf8"), encoding="utf8")
    (library / "active.shaderlib.py").write_text('''import infernux.shader as shader
def color() -> shader.vec3:
    return shader.vec3(0.2, 0.4, 0.6)
''', encoding="utf8")
    monkeypatch.setattr("infernux.resources.resources_path", str(root))
    source = '''import infernux.shader as shader
from infernux.shader.lib.active import color
class Paint(shader.surface):
    def surface(self, ctx: shader.SurfaceContext) -> shader.SurfaceData:
        return shader.SurfaceData(albedo=color())
'''
    definition = parse_shader(source, str(tmp_path / "paint.frag.py"), "active_resources")
    assert definition.deferred_model == "PBR"
    paths = {path for _, path, _ in definition.libraries}
    assert str((root / "shaders" / pbr.name).resolve()) in paths
    assert str((library / "active.shaderlib.py").resolve()) in paths
    assert len(definition.compile()["targets"]) == 2


def test_shader_identity_visibility_and_render_defaults_are_literal_metadata():
    from infernux._compiler.shader import _prepare_source

    source = '''import infernux.shader as shader
class Paint(shader.surface):
    shader_id = "example/materials/paint"
    hidden = True
    surface_type = "transparent"
    cull_mode = "none"
    alpha_clip = 0.25
    cast_shadows = False
    receive_shadows = False
    def surface(self, ctx: shader.SurfaceContext) -> shader.SurfaceData:
        return shader.SurfaceData(albedo=shader.vec3(0.8), alpha=0.6)
'''
    definition = parse_shader(source, "different_file_name.frag.py", "metadata_surface")
    assert definition.info["shader_id"] == "example/materials/paint" and definition.info["hidden"]
    assert definition.info["render_queue"] == 3000 and definition.info["depth_write"] is False
    assert definition.info["blend"] == "alpha" and definition.info["alpha_clip"] == .25
    assert "static const bool INX_RECEIVE_SHADOWS = false;" in definition.generated_source
    assert not {"shader_id", "hidden", "alpha_clip", "render_queue"} & {field.name for field in definition.parameters}
    prepared, libraries, metadata, ranges = _prepare_source(
        source, "different_file_name.frag.py", "metadata", lambda path: pytest.fail("Metadata reads no dependencies"),
        metadata_only=True)
    assert not prepared and not libraries and not ranges and json.loads(metadata) == definition.info
    with pytest.raises(SyntaxError, match="Invalid shader declaration: hidden"):
        parse_shader(source.replace("hidden = True", "hidden = 1"), "bad.frag.py", "bad_metadata")


def test_unlit_model_retains_geometry_passes_without_lighting_resources(tmp_path):
    from infernux._compiler.shading_model import default_shading_model

    model = default_shading_model().with_name("unlit.shadingmodel.py")
    source = f'''import infernux.shader as shader
class Painted(shader.surface):
    shading_model = shader.ref({model.as_posix()!r})
    tint = shader.material.color(default=(0.2, 0.4, 0.8, 1.0))
    def surface(self, ctx: shader.SurfaceContext) -> shader.SurfaceData:
        return shader.SurfaceData(albedo=self.tint.rgb, emission=shader.vec3(0.1), alpha=self.tint.a)
'''
    definition = parse_shader(source, str(tmp_path / "painted.frag.py"), "painted_unlit")
    assert definition.shading_model == definition.deferred_model == "Unlit"
    assert str(model) in {path for _, path, _ in definition.libraries}
    for target in definition.compile()["targets"]:
        entries = {entry["name"]: {tuple(binding) for binding in entry["bindings"]} for entry in target["entries"]}
        assert "fragment_gbuffer" in entries
        assert not any(binding[0] == 1 for binding in entries["fragment_main"])
        assert not any(binding[0] == 1 for binding in entries["fragment_forward_plus"])


def test_custom_shading_model_has_typed_context_imports_and_specialized_lights(tmp_path):
    from infernux._compiler.shader import _prepare_source

    (tmp_path / "bands.shaderlib.py").write_text('''import infernux.shader as shader
def band(value: float, threshold: float) -> float:
    return shader.mix(0.25, 1.0, shader.step(threshold, value))
''', encoding="utf-8")
    model = tmp_path / "band.shadingmodel.py"
    model.write_text('''import infernux.shader as shader
from .bands import band
class Band(shader.shadingmodel):
    lighting_contract = "custom_response"
    def response(self, value: float, threshold: float) -> float:
        return band(value, threshold)
    def shade_light(self, surface: shader.SurfaceData, ctx: shader.ShadingContext, light: shader.LightData) -> shader.vec3:
        weight = self.response(shader.max(0.0, shader.dot(surface.normal_ws, light.direction)), surface.shading_param0)
        direct = surface.albedo * light.color * light.attenuation * light.shadow * weight
        return direct
    def shade(self, surface: shader.SurfaceData, ctx: shader.ShadingContext) -> shader.vec3:
        ambient = shader.lighting.ambient(ctx, surface.normal_ws) * surface.albedo * surface.occlusion
        return shader.lighting.direct(surface, ctx) + ambient + surface.emission
''', encoding="utf-8")
    source = '''import infernux.shader as shader
class Painted(shader.surface):
    shading_model = shader.ref("./band.shadingmodel.py")
    def surface(self, ctx: shader.SurfaceContext) -> shader.SurfaceData:
        return shader.SurfaceData(albedo=shader.vec3(0.5), shading_param0=0.55)
'''
    definition = parse_shader(source, str(tmp_path / "painted.frag.py"), "painted_band")
    assert definition.shading_model == "Band" and definition.deferred_model is None
    for target in definition.compile()["targets"]:
        entries = {entry["name"]: {tuple(binding) for binding in entry["bindings"]} for entry in target["entries"]}
        assert "fragment_gbuffer" not in entries
        assert not {(1, 1), (1, 2), (1, 3)} & entries["fragment_main"]
        assert (1, 1) in entries["fragment_forward_plus"]
    _, libraries, schema, _ = _prepare_source(model.read_text(encoding="utf-8"), str(model), "band", None)
    assert json.loads(schema)["lighting_contract"] == "custom_response"
    assert str(tmp_path / "bands.shaderlib.py") in {path for _, path, _ in libraries}
    original = model.read_text(encoding="utf-8")
    model.write_text(original.replace("def shade_light(", "def other_light("), encoding="utf-8")
    with pytest.raises(SyntaxError, match="lighting.direct requires shade_light"):
        parse_shader(source, str(tmp_path / "painted.frag.py"), "missing_response")
    model.write_text(original, encoding="utf-8")
    model.write_text(model.read_text(encoding="utf-8").replace("ctx, surface.normal_ws", "ctx.uv0, surface.normal_ws"), encoding="utf-8")
    with pytest.raises(SyntaxError, match="Unknown shader stage input: uv0"):
        parse_shader(source, str(tmp_path / "painted.frag.py"), "invalid_band")


def test_builtin_toon_uses_shared_light_response_and_surface_passes():
    from infernux._compiler.shading_model import default_shading_model

    model = default_shading_model().with_name("toon.shadingmodel.py")
    path = model.with_name("toon.frag.py")
    source = path.read_text(encoding="utf8").replace('shader.ref("infernux/shading/toon")',
                                                    f'shader.ref({str(model)!r})')
    definition = parse_shader(source, str(path), "builtin_toon")
    assert definition.shading_model == definition.deferred_model == "Toon"
    assert definition.info["shader_id"] == "infernux/toon"
    fields = {parameter.name for parameter in definition.parameters}
    assert {"diffuse_threshold", "band_softness", "normal_map"} <= fields
    for target in definition.compile()["targets"]:
        entries = {entry["name"] for entry in target["entries"]}
        assert {"fragment_main", "fragment_forward_plus", "fragment_gbuffer", "fragment_depth", "vertex_shadow"} <= entries


def test_surface_specializes_light_resources_and_preserves_context():
    source = '''import infernux.shader as shader
class Painted(shader.surface):
    tint = shader.material.color(default=(0.2, 0.4, 0.8, 1.0))
    def surface(self, ctx: shader.SurfaceContext) -> shader.SurfaceData:
        return shader.SurfaceData(albedo=self.tint.rgb, normal_ws=ctx.normal_ws, alpha=self.tint.a)
'''
    definition = parse_shader(source, "表面.frag.py", "painted_surface")
    assert definition.surface and definition.domain == "mesh"
    assert definition.parameters[-1].name == "_alpha_clip_threshold" and definition.parameters[-1].internal
    for target in definition.compile()["targets"]:
        entries = {entry["name"]: {tuple(binding) for binding in entry["bindings"]} for entry in target["entries"]}
        assert (2, 4) in entries["vertex_main"]  # Identity, layer and receive-shadow flags.
        assert {(1, 5), (1, 7), (2, 4)} <= entries["vertex_shadow"]
        assert (1, 7) not in entries["vertex_main"]
        assert {(1, 0), (1, 4), (1, 6)} <= entries["fragment_main"]
        assert not {(1, 1), (1, 2), (1, 3)} & entries["fragment_main"]
        assert {(1, 1), (1, 2), (1, 3)} <= entries["fragment_forward_plus"]
        assert {"vertex_motion", "fragment_gbuffer", "fragment_depth", "fragment_picking", "fragment_motion",
                "fragment_normal", "fragment_base_color"} <= entries.keys()
        # Geometry outputs retain the authored surface and alpha decision,
        # without linking light lists or shadow resources into their stages.
        for name in ("fragment_gbuffer", "fragment_depth", "fragment_picking", "fragment_motion",
                     "fragment_normal", "fragment_base_color"):
            assert not any(binding[0] == 1 for binding in entries[name])


def test_depth_and_integer_ports_keep_their_value_types_and_reject_filtered_ids():
    source = '''import infernux.shader as shader
class ReadBuffers(shader.fullscreen):
    depth = shader.input("depth", shader.Depth2D)
    identity = shader.input("object", shader.UInt2D)
    def main(self, ctx: shader.fragment_input) -> shader.vec4:
        pixel = shader.ivec2(ctx.position.xy)
        z = shader.load(self.depth, pixel, 0)
        identity = shader.load(self.identity, pixel, 0)
        size = shader.dimensions(self.identity, 0)
        return shader.vec4(z, float(identity.x), float(size.x), 1.0)
'''
    for target in parse_shader(source, "场景缓冲.frag.py", "read_buffers").compile()["targets"]:
        images = {resource["name"]: resource for resource in target["resources"]}
        assert not images["inx_depth.texture"]["depth"]
        assert images["inx_depth.texture"]["sample_type"] == 1  # Raw D32 and R32 both bind as unfilterable float.
        assert images["inx_depth.sampler"]["sampler_type"] == 1  # Nearest sampling on every backend.
        assert images["inx_identity.texture"]["sample_type"] == 3  # Uint, never a filtered float image.
        assert "inx_identity.sampler" not in images
    with pytest.raises(SyntaxError, match="pixel coordinates"):
        parse_shader(source.replace("shader.load(self.identity, pixel, 0)", "shader.sample(self.identity, ctx.uv)"),
                     "场景缓冲.frag.py", "read_buffers")


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
