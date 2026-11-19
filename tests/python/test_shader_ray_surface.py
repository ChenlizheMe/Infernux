"""The same authored PBR Surface compiles at raster and exact secondary hits."""

import json
from pathlib import Path

import pytest

from infernux._compiler.shader import _prepare_source
from infernux._compiler.shader_ray_surface import optional_ray_surface
from infernux.lib import _Infernux as native


SURFACE = '''import infernux.shader as shader
from .marker import marker
class Custom(shader.surface):
    tint = shader.material.color()
    energy = shader.material.float(default=2.0)
    mask = shader.material.texture_2d()
    def surface(self, ctx: shader.SurfaceContext) -> shader.SurfaceData:
        sampled = shader.sample_level(self.mask, ctx.uv0, 0.0)
        return shader.SurfaceData(albedo=marker(self.tint.rgb * sampled.rgb),
                                  emission=marker(self.tint.rgb) * self.energy,
                                  normal_ws=ctx.normal_ws)
'''
LIBRARY = '''import infernux.shader as shader

def marker(value: shader.vec3) -> shader.vec3:
    return value * shader.vec3(0.25, 0.5, 1.0)
'''


def sources(tmp_path):
    path = tmp_path / "custom.frag.py"
    path.write_text(SURFACE, encoding="utf-8")
    (tmp_path / "marker.shaderlib.py").write_text(LIBRARY, encoding="utf-8")
    return path, lambda value: Path(value).read_text(encoding="utf-8")


def test_surface_replay_and_history_readers_track_the_same_depth_layout(tmp_path):
    from infernux._compiler.shader import parse_shader
    from infernux._compiler.shader_library import builtin_shader_root

    surface, read = sources(tmp_path)
    root = builtin_shader_root()
    depth_layout = (root / "lib/ray_surface_depth.shaderlib.py").resolve()
    for path in (surface, root / "lit.frag.py", root / "ray_primary_surface.frag.py",
                 root / "ray_path_temporal.frag.py", root / "ray_reconstruct.frag.py"):
        reads = set()

        def tracked_read(location):
            reads.add(Path(location).resolve())
            return read(location)

        definition = parse_shader(read(path), str(path), "depth_layout_test", reader=tracked_read)
        assert depth_layout in reads, path.name
        assert depth_layout in {Path(source) for _, source, _ in definition.libraries}, path.name


@pytest.mark.parametrize("target", ["spirv", "wgsl"])
def test_secondary_surface_has_reflected_payload_without_extra_active_buffer(tmp_path, target):
    path, read = sources(tmp_path)
    adapter, diagnostic = optional_ray_surface(SURFACE, str(path), "test_secondary", read)
    assert diagnostic is None
    root = '''import test_secondary; import inx_geometry;
[[vk::binding(0, 0)]] ByteAddressBuffer input;
[[vk::binding(1, 0)]] RWStructuredBuffer<float4> output;
[shader("compute")] [numthreads(1, 1, 1)]
void main() {
    InxMeshInput ctx = (InxMeshInput)0; ctx.normal_ws = float3(0, 0, 1);
    let value = test_secondary_evaluate(ctx, test_secondary_load(input, 0));
    output[0] = float4(value.albedo + value.emission, value.alpha);
}'''
    compiled = native._compile_shader_program(
        root, "<secondary-surface-test>", "secondary_surface_test",
        [*adapter.libraries, (adapter.module_name, adapter.source_path, adapter.generated_source)], [target])
    code, = compiled["targets"]
    payload = next(item for item in code["resources"] if item["name"] == adapter.manifest["schema_resource"])
    raw = next(item for item in code["resources"] if item["name"] == "input")
    assert raw["element_stride"] == 4
    assert payload["element_stride"] == 64
    assert all(binding[0] != 3 for binding in code["entries"][0]["bindings"])
    assert adapter.manifest["may_emit"]


def test_screen_derivative_retains_raster_author_and_located_secondary_diagnostic(tmp_path):
    path, read = sources(tmp_path)
    source = SURFACE.replace("marker(self.tint.rgb) * self.energy", "shader.vec3(shader.ddx(ctx.uv0.x))")
    _, _, encoded, _ = _prepare_source(source, str(path), "surface_test", read)
    schema = json.loads(encoded)
    assert schema["surface"]
    assert schema["ray_surface"] is None
    diagnostic = schema["ray_surface_diagnostic"]
    assert diagnostic["path"] == str(path)
    assert diagnostic["line"] == 10
    assert diagnostic["severity"] == "warning"
    assert "derivatives" in diagnostic["message"]


def test_nonemissive_surface_does_not_claim_emitter_support(tmp_path):
    path, read = sources(tmp_path)
    source = SURFACE.replace("emission=marker(self.tint.rgb) * self.energy,", "")
    adapter, diagnostic = optional_ray_surface(source, str(path), "surface_test", read)
    assert diagnostic is None
    assert not adapter.manifest["may_emit"]


def test_invalid_unused_project_surface_does_not_break_unrelated_ray_program(tmp_path):
    from infernux._compiler.shader import parse_shader
    path = tmp_path / "broken.frag.py"
    path.write_text("class Broken(\n", encoding="utf-8")
    source = """import infernux.shader as shader
class Query(shader.fullscreen):
    def main(self, ctx: shader.fragment_input) -> shader.vec4:
        hit = shader.ray.trace_closest(shader.vec3(0.0), shader.vec3(0.0, 0.0, 1.0),
                                      0.0, 10.0, shader.uint(0xffffffff), 0.0)
        return shader.vec4(hit.surface.albedo, shader.float(hit.available))
"""
    definition = parse_shader(source, str(tmp_path / "query.frag.py"), "unrelated_query",
                              ray_surfaces=(("broken_guid", str(path)),))
    assert definition.ray_surfaces == ()
    assert definition.compile(targets=("spirv",))["targets"]


def test_project_catalog_is_lazy_and_frozen_only_for_world_material_consumers(tmp_path):
    surface = tmp_path / "custom.frag.py"
    surface.write_text("""import infernux.shader as shader
class Custom(shader.surface):
    def surface(self, ctx: shader.SurfaceContext) -> shader.SurfaceData:
        return shader.SurfaceData(albedo=shader.vec3(0.5))
""", encoding="utf-8")
    calls = []
    def catalog():
        calls.append(True)
        return (("custom_guid", str(surface)),)
    ordinary = """import infernux.shader as shader
class Ordinary(shader.fullscreen):
    def main(self) -> shader.vec4:
        return shader.vec4(0.5)
"""
    _, _, encoded, _ = _prepare_source(ordinary, str(tmp_path / "ordinary.frag.py"), "ordinary", None,
                                      ray_surfaces=catalog)
    assert calls == []
    assert json.loads(encoded)["ray_surface_catalog"] is None
    query = """import infernux.shader as shader
class Query(shader.fullscreen):
    def main(self) -> shader.vec4:
        hit = shader.ray.trace_closest(shader.vec3(0.0), shader.vec3(0.0, 0.0, 1.0),
                                      0.0, 10.0, shader.uint(0xffffffff), 0.0)
        return shader.vec4(hit.surface.albedo, shader.float(hit.available))
"""
    _, _, encoded, _ = _prepare_source(query, str(tmp_path / "query.frag.py"), "query", None,
                                      ray_surfaces=catalog)
    assert calls == [True]
    assert json.loads(encoded)["ray_surface_catalog"] == ["custom_guid"]


def test_positional_surface_constructor_never_proves_zero_emission():
    import ast
    from infernux._compiler.shader_ray_surface import _may_emit
    main = ast.parse("def surface():\n    return shader.SurfaceData(shader.vec3(1.0))\n").body[0]
    assert _may_emit(main)
