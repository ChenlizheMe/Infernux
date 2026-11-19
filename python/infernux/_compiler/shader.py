"""Static GPU author fields, typed inputs and functions lowered to native Slang."""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path

from .shader_library import LibraryGraph, Module, ShaderEmitter, declarations, engine_imports, uses_surface_lighting
from .shader_types import RESOURCE_BINDINGS, RESOURCE_MODULE, RESOURCE_TYPES
from .shading_model import default_shading_model, read_shading_model
from .shader_info import INFO_DEFAULTS, info_field, read_shader_info
from .shader_parameters import ParameterAuthor, ShaderInput, ShaderParameter, resource_symbol
from .vertex import read_vertex


_RAY_LIT_MODULE = "inx_ray_lit"
_WORLD_RAY_INPUTS = tuple(ShaderInput(name, name, kind) for name, kind in (
    ("world_ray_geometry", "WorldBuffer"),
    ("world_ray_meshes", "WorldBuffer"),
    ("ray_lit_materials", "WorldBuffer"),
    ("ray_color_pages", "Color2DArray"),
    ("ray_data_pages", "Color2DArray"),
))
_TRANSPARENT_PARTITIONS = (("transparent_behind", "false"), ("transparent_front", "true"))
_TRANSPARENT_LAYER = """
[[vk::binding(8, 1)]] Texture2D<uint4> inx_transparent_layer;
void inx_partition_transparent<let front : bool>(float4 position, float depth)
{
    const uint4 layer = inx_transparent_layer.Load(int3(int2(position.xy), 0));
    const bool in_front = layer.x != 0u && depth >= asfloat(layer.w);
    if (in_front != front)
        discard;
}
"""


@dataclass(frozen=True)
class ShaderDefinition:
    name: str
    source_path: str
    module_name: str
    parameters: tuple[ShaderParameter, ...]
    generated_source: str
    libraries: tuple[tuple[str, str, str], ...] = ()
    inputs: tuple[ShaderInput, ...] = ()
    source_ranges: tuple = ()
    domain: str = "fullscreen"
    surface: bool = False
    shading_model: str | None = None
    deferred_model: str | None = None
    info: dict | None = None
    vertex_source: str | None = None
    thin_sheet: bool = False
    color_outputs: tuple[str, ...] = ()
    ray_surfaces: tuple = ()
    ray_surface_catalog: tuple | None = None

    def compile(self, targets=("spirv", "wgsl")):
        from infernux.lib import _Infernux
        return _Infernux._compile_shader_program(
            self.generated_source, self.source_path, self.module_name, self.libraries, targets,
            source_ranges=self.source_ranges,
            input_kinds=(
                tuple((field.name, field.kind) for field in self.inputs)
                + tuple((field.name, field.kind) for field in self.parameters if field.kind == "texture_2d")
            ),
            root_constants=self.domain in {"screen_ui", "world_ui"},
        )


class _Author(ParameterAuthor):
    uses_surface_sampling = False
    ray_entry = None
    ray_material_type = "InxRayLitMaterial"
    ray_sampling_type = "InxRayLitSampling"
    ray_entry_visibility = ""
    color_outputs = ()
    tuple_outputs = ()

    _surface_sampling_operations = {
        "_surface_sample": ("sample", "float4", ("InxColor2D", "float2")),
        "_surface_differentials": ("differentials", "InxSurfaceDifferentials", ("InxMeshInput", "float2", "float2")),
    }

    def _surface_sampling_operation(self, node):
        name = self.qualified(node.func)
        operation = self.symbol(name)
        return self._surface_sampling_operations.get(operation) if operation != name else None

    def infer(self, node):
        if isinstance(node, ast.Call) and (operation := self._surface_sampling_operation(node)) is not None:
            return operation[1]
        return super().infer(node)

    def visit_Call(self, node):
        operation = self._surface_sampling_operation(node)
        if operation is None:
            return super().visit_Call(node)
        if not self.is_surface:
            self.error(node, "Material sampling context is available in the Surface entry", code="author.stage")
        method, _, arguments = operation
        if node.keywords or tuple(self.infer(arg) for arg in node.args) != arguments:
            self.error(node, "Material sampling arguments must match the typed Surface context", code="author.type")
        self.uses_surface_sampling = True
        return f"inx_surface_sampling.{method}(" + ", ".join(self.visit(arg) for arg in node.args) + ")"

    def visit_Attribute(self, node):
        if self.ray_entry and isinstance(node.value, ast.Name) and node.value.id == "self":
            field = self.fields.get(node.attr)
            if isinstance(field, ShaderParameter):
                if field.kind == "texture_2d":
                    return f"inx_ray_material.{field.name}"
                value = f"inx_ray_material.values.{field.name}"
                return f"({value} != 0u)" if field.kind == "bool" else value
        return super().visit_Attribute(node)

    def visit_Return(self, node):
        if not self.tuple_outputs:
            return super().visit_Return(node)
        if not isinstance(node.value, ast.Tuple) or len(node.value.elts) != len(self.tuple_outputs):
            self.error(node, f"Fullscreen main returns a tuple of {len(self.tuple_outputs)} color outputs",
                       code="author.output")
        for value, kind in zip(node.value.elts, self.tuple_outputs):
            if self.infer(value) != kind:
                self.error(value, f"Fullscreen output requires {kind}", code="author.type")
        self.line("return {" + ", ".join(self.visit(value) for value in node.value.elts) + "};")

    def function(self, main):
        if main.decorator_list or main.args.vararg or main.args.kwarg or main.args.kwonlyargs or main.args.defaults:
            self.error(main, "Shader entry parameters have a fixed typed signature")
        arguments = main.args.args
        if (self.is_surface or self.domain in {"screen_ui", "world_ui"}) and len(arguments) != 2:
            self.error(main, "This shader entry accepts self and its typed context")
        if not arguments or arguments[0].arg != "self":
            self.error(main, "Shader main is an instance method")
        if len(arguments) > 2:
            self.error(main, "Shader main accepts self and one typed stage input")
        signature = ""
        if len(arguments) == 2:
            uv = arguments[1]
            kind = self.annotation(uv.annotation) if uv.annotation is not None else None
            expected = ({"InxUIContext"} if self.domain == "screen_ui" else {"InxWorldUIContext"} if self.domain == "world_ui" else {"InxMeshInput"} if self.domain == "mesh" else {"InxParticleInput"} if self.domain == "particle"
                        else {"float2", "InxFragmentInput", "InxMultisampleInput"})
            if kind not in expected:
                self.error(uv, {"mesh": "Mesh input is shader.mesh_input",
                                "particle": "Particle input is shader.ParticleContext",
                                "screen_ui": "Screen UI input is shader.UIContext", "world_ui": "World UI input is shader.WorldUIContext"}.get(
                                    self.domain, "Fullscreen input is shader.vec2, shader.fragment_input or shader.multisample_input"))
            signature = kind + " " + self.bind_value(uv.arg) + (" : TEXCOORD0" if kind == "float2" else "")
            self.types[uv.arg] = kind
        tuple_return = (isinstance(main.returns, ast.Subscript)
                        and self.qualified(main.returns.value) == "tuple")
        if tuple_return:
            if self.domain != "fullscreen":
                self.error(main.returns, "Tuple color outputs belong to shader.fullscreen", code="author.stage")
            elements = main.returns.slice.elts if isinstance(main.returns.slice, ast.Tuple) else [main.returns.slice]
            if not 1 <= len(elements) <= 8:
                self.error(main.returns, "Fullscreen tuples declare between one and eight color outputs",
                           code="author.output")
            kinds = tuple(self.annotation(element) for element in elements)
            if any(kind not in {"float4", "uint4"} for kind in kinds):
                self.error(main.returns, "Fullscreen tuple elements are shader.vec4 or shader.uvec4",
                           code="author.output")
            self.tuple_outputs = self.color_outputs = kinds
            result = "InxFullscreenOutput"
        else:
            result = self.annotation(main.returns) if main.returns is not None else None
        expected = {"InxSurfaceData"} if self.is_surface else {"float4", "InxFragmentOutput"}
        if self.domain == "fullscreen":
            expected.update(("uint4", "InxFullscreenOutput"))
            if not tuple_return:
                self.color_outputs = ("float4" if result == "InxFragmentOutput" else result,)
        if result not in expected:
            permitted = "shader.vec4, shader.uvec4 or shader.FragmentOutput" if self.domain == "fullscreen" else "shader.vec4 or shader.FragmentOutput"
            self.error(main, "Surface returns shader.SurfaceData" if self.is_surface else f"Fragment main returns {permitted}")
        self.aliases = (self.alias,)
        if self.domain == "particle":
            self.lighting_mode = "tiled_lighting"
        self.declare_locals(main.body, parameters=(field.arg for field in arguments[1:]))
        self.local_declarations()
        self.block(main.body)
        semantic = " : SV_Target0" if result in {"float4", "uint4"} else ""
        if self.domain in {"mesh", "particle"} and not self.is_surface:
            argument = self.value_names[arguments[1].arg] if len(arguments) == 2 else ""
            alpha = "color.color.a" if result == "InxFragmentOutput" else "color.a"
            entry = "inx_evaluate_particle" if self.domain == "particle" else "inx_evaluate_fragment"
            threshold = "max(inx_parameters._alpha_clip_threshold, 0.0001)" if self.domain == "particle" else "inx_parameters._alpha_clip_threshold"
            evaluation_signature = signature or ("InxParticleInput inx_input" if self.domain == "particle" else "")
            coverage = ["    if (!(" + (argument or "inx_input") + ".alpha > 0.0)) discard;"] if self.domain == "particle" else []
            inline = ["[ForceInline]"] if self.domain == "particle" else []
            specialization = "<let tiled_lighting : bool>" if self.domain == "particle" else ""
            arguments_specialization = "<tiled_lighting>" if self.domain == "particle" else ""
            evaluator = [self.source_directive(main), *inline,
                    result + " inx_author_fragment" + specialization + "(" + signature + ")",
                    "{", *self.lines, "}", *inline,
                    result + " " + entry + specialization + "(" + evaluation_signature + ")", "{", *coverage,
                    "    " + result + " color = inx_author_fragment" + arguments_specialization + "(" + argument + ");",
                    "    clip(" + alpha + " - " + threshold + ");", "    return color;", "}"]
            if self.domain == "mesh":
                evaluator += ['[shader("fragment")]', f"{result} fragment_main({signature}){semantic}",
                              "{", f"    return {entry}({argument});", "}"]
                depth = "value.depth" if result == "InxFragmentOutput" else "input.position.z"
                value = entry + "(input)" if argument else entry + "()"
                for name, front in _TRANSPARENT_PARTITIONS:
                    evaluator.append(_partitioned_entry(
                        name, result, "InxMeshInput input", semantic, value, front, depth))
            return evaluator
        if self.domain in {"screen_ui", "world_ui"}:
            return [self.source_directive(main), result + " inx_author_ui(" + signature + ")", "{", *self.lines, "}"]
        if self.ray_entry:
            sampling = f", {self.ray_sampling_type} inx_surface_sampling" if self.ray_sampling_type else ""
            declaration = (f"{self.ray_entry_visibility}InxSurfaceData {self.ray_entry}({signature}, {self.ray_material_type} inx_ray_material"
                           f"{sampling})")
        elif self.is_surface:
            sampling = "<Sampling : InxSurfaceSampling>" if self.uses_surface_sampling else ""
            sampling_argument = ", Sampling inx_surface_sampling" if self.uses_surface_sampling else ""
            declaration = f"InxSurfaceData inx_author_surface{sampling}({signature}{sampling_argument})"
        else:
            declaration = result + " fragment_main(" + signature + ")" + semantic
        outputs = (["struct InxFullscreenOutput {",
                    *(f"    {kind} target_{slot} : SV_Target{slot};" for slot, kind in enumerate(self.tuple_outputs)),
                    "};"] if tuple_return else [])
        return outputs + ([self.source_directive(main)] if self.is_surface else ['[shader("fragment")]', self.source_directive(main)]) + [
            declaration, "{", *self.lines, "}"]



def _partitioned_entry(name, result, signature, semantic, value, front, depth, coverage=""):
    return f"""
[shader("fragment")]
{result} fragment_{name}({signature}){semantic}
{{
    {result} value = {value};
    {coverage}
    inx_partition_transparent<{front}>(input.position, {depth});
    return value;
}}
"""


def _particle_passes(result):
    call = "inx_evaluate_particle<false>(input)"
    depth = result == "InxFragmentOutput"
    color_depth = "float depth : SV_Depth;" if depth else ""
    picking_result = "InxParticlePickingOutput" if depth else "uint2"
    motion_result = "InxParticleMotionOutput" if depth else "float2"
    outputs = ("struct InxParticlePickingOutput { uint2 identity : SV_Target0; " + color_depth + " };\n"
               "struct InxParticleMotionOutput { float2 motion : SV_Target0; " + color_depth + " };\n") if depth else ""
    picking_value = ("InxParticlePickingOutput output; output.identity = inx_particle_owner_id(); "
                     "output.depth = value.depth; return output;") if depth else "return inx_particle_owner_id();"
    motion_value = ("InxParticleMotionOutput output; output.motion = input.motion_vector; "
                    "output.depth = value.depth; return output;") if depth else "return input.motion_vector;"
    color = "value.color" if depth else "value"
    color_passes = []
    for prefix, tiled in (("fragment", "false"), ("fragment_forward_plus", "true")):
        color_call = f"inx_evaluate_particle<{tiled}>(input)"
        entry = "fragment_main" if tiled == "false" else prefix
        color_passes.append(f"""
[shader("fragment")]
{result} {entry}(InxParticleInput input){"" if depth else " : SV_Target0"} {{ return {color_call}; }}
[shader("fragment")]
{result} {prefix}_soft_depth(InxParticleInput input){"" if depth else " : SV_Target0"} {{
    {result} value = {color_call};
    {color} = inx_particle_apply_soft_coverage({color}, inx_particle_soft_coverage<true>(input));
    return value;
}}
[shader("fragment")]
{result} {prefix}_soft_color(InxParticleInput input){"" if depth else " : SV_Target0"} {{
    {result} value = {color_call};
    {color} = inx_particle_apply_soft_coverage({color}, inx_particle_soft_coverage<false>(input));
    return value;
}}
""")
    signature = "InxParticleInput input"
    semantic = "" if depth else " : SV_Target0"
    layer_depth = "value.depth" if depth else "input.position.z"
    for name, front in _TRANSPARENT_PARTITIONS:
        for suffix, sample in (("", ""), ("_soft_depth", "true"), ("_soft_color", "false")):
            coverage = (
                f"{color} = inx_particle_apply_soft_coverage({color}, inx_particle_soft_coverage<{sample}>(input));"
                if sample else ""
            )
            color_passes.append(_partitioned_entry(
                name + suffix, result, signature, semantic, "inx_evaluate_particle<true>(input)",
                front, layer_depth, coverage))
    return outputs + "".join(color_passes) + f"""
[shader("fragment")]
{picking_result} fragment_picking(InxParticleInput input){"" if depth else " : SV_Target0"} {{
    {result} value = {call}; {picking_value}
}}
[shader("fragment")]
{motion_result} fragment_motion(InxParticleInput input){"" if depth else " : SV_Target0"} {{
    {result} value = {call}; {motion_value}
}}
[shader("fragment")]
{"float" if depth else "void"} fragment_depth(InxParticleInput input){" : SV_Depth" if depth else ""} {{
    {result} value = {call}; {"return value.depth;" if depth else ""}
}}
"""


_FULLSCREEN_VERTEX = """
struct FullscreenVertex { float4 position : SV_Position; float2 uv : TEXCOORD0; };
[shader("vertex")]
FullscreenVertex vertex_main(uint id : SV_VulkanVertexID)
{
    float2 uv = float2((id << 1) & 2, id & 2);
    FullscreenVertex result;
    result.position = float4(uv * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
    result.uv = uv;
    return result;
}
"""

_MESH_RESOURCES = """
[[vk::binding(1, 2)]] StructuredBuffer<InxModelMatrix> inx_models;
[[vk::binding(2, 2)]] StructuredBuffer<InxSkinState> inx_skins;
[[vk::binding(3, 2)]] StructuredBuffer<InxModelMatrix> inx_bones;
[[vk::binding(4, 2)]] StructuredBuffer<InxInstanceData> inx_instances;
[[vk::binding(5, 2)]] StructuredBuffer<InxModelMatrix> inx_previous_models;
[[vk::binding(6, 2)]] StructuredBuffer<InxWorldBounds> inx_pose_bounds;
float4x4 inx_current_model(uint instance)
{
    InxInstanceData data = inx_instances[instance];
    return inx_models[(data.object_flags & 4u) != 0u ? data.pose_indices.x : instance].value;
}
float4x4 inx_previous_model(InxInstanceData data)
{
    return (data.object_flags & 4u) != 0u ? inx_previous_models[data.pose_indices.y].value : data.previous_model;
}
bool inx_bounds_halfspace(float4 plane, float3 center, float3 extent)
{
    return dot(plane.xyz, center) + plane.w + dot(abs(plane.xyz), extent) >= 0.0;
}
void inx_clip_device_bounds(inout InxMeshInput result, uint instance, bool ignore_near = false)
{
    InxInstanceData data = inx_instances[instance];
    if ((data.object_flags & 4u) == 0u) return;
    InxWorldBounds bounds = inx_pose_bounds[data.pose_indices.z];
    float3 center = (bounds.minimum.xyz + bounds.maximum.xyz) * 0.5;
    float3 extent = (bounds.maximum.xyz - bounds.minimum.xyz) * 0.5;
    float4x4 clip = mul(inx_view.projection, inx_view.view);
    bool visible = inx_bounds_halfspace(clip[3] + clip[0], center, extent)
        && inx_bounds_halfspace(clip[3] - clip[0], center, extent)
        && inx_bounds_halfspace(clip[3] + clip[1], center, extent)
        && inx_bounds_halfspace(clip[3] - clip[1], center, extent)
        && inx_bounds_halfspace(clip[2], center, extent)
        && (ignore_near || inx_bounds_halfspace(clip[3] - clip[2], center, extent));
    if (!visible) result.position = float4(2.0, 2.0, 2.0, 1.0);
}
"""

_VERTEX_DEFORM = """
InxVertexInput inx_apply_vertex(InxVertexInput input, uint instance, float time, float4x4 model)
{
    InxVertexContext context;
    context.time = time;
    context.delta_time = inx_globals.time.w;
    context.object_id = inx_instances[instance].object_id;
    context.layer_mask = inx_instances[instance].layer_mask;
    context.model_matrix = model;
    InxVertexData vertex = inx_vertex_author::deform(inx_vertex_data(input), context);
    input.position = vertex.position_os;
    input.normal = vertex.normal_os;
    input.tangent = vertex.tangent_os;
    input.color = vertex.color;
    input.uv0 = vertex.uv0;
    input.uv1 = vertex.uv1;
    return input;
}
"""


_VERTEX_PROJECT = """
float4 inx_project_vertex(InxMeshInput vertex)
{
    InxProjectionContext context;
    context.view_matrix = inx_view.view;
    context.projection_matrix = inx_canonical_projection(inx_view.projection);
    return inx_vertex_author::project(vertex, context);
}
"""


def _mesh_vertex(*, deform=False, project=False):
    deformation = ("input = inx_apply_vertex(input, instance, inx_globals.time.x, inx_current_model(instance));"
                   if deform else "")
    projection = "result.position = inx_project_vertex(result);" if project else ""
    return _MESH_RESOURCES + f'''
InxMeshInput inx_mesh_vertex(InxVertexInput input, uint instance)
{{
    {deformation}
    input = inx_skin_vertex(input, inx_skins[instance], inx_bones);
    return inx_transform_vertex(input, inx_current_model(instance), inx_view.view, inx_view.projection, inx_instances[instance]);
}}
{_VERTEX_PROJECT if project else ""}
[shader("vertex")]
InxMeshInput vertex_main(InxVertexInput input, uint instance : SV_VulkanInstanceID)
{{
    InxMeshInput result = inx_mesh_vertex(input, instance);
    {projection}
    inx_clip_device_bounds(result, instance);
    return result;
}}
'''


_SURFACE_PASSES = """
[[vk::binding(7, 1)]] ConstantBuffer<InxShadowParameters> inx_shadow_parameters;
[shader("vertex")]
InxMeshInput vertex_shadow(InxVertexInput input, uint instance : SV_VulkanInstanceID)
{
    InxMeshInput result = inx_mesh_vertex(input, instance);
    if (input.bone_indices.w == 0x4C494E45u && input.bone_weights.z > 0.0)
    {
        float4 light = inx_shadow_parameters.light_vector;
        float3 toward_light = light.w < 0.5 ? light.xyz : light.xyz - result.position_ws;
        result.position_ws -= normalize(toward_light) * (2.0 * abs(input.bone_weights.x) * input.bone_weights.z);
        result.position = mul(inx_view.projection, mul(inx_view.view, float4(result.position_ws, 1.0)));
        result.position.y = -result.position.y;
    }
    if (inx_shadow_parameters.light_vector.w < 0.5)
        result.position.z = min(result.position.z, result.position.w);
    result.view_depth = 0.0;
    inx_clip_device_bounds(result, instance, inx_shadow_parameters.light_vector.w < 0.5);
    float4 p = result.position;
    float4 viewport = inx_shadow_parameters.atlas_viewport;
    result.motion_current_clip = float4(float2(p.x, -p.y) * viewport.xy + p.w * viewport.zw, p.zw);
    return result;
}
[shader("vertex")]
InxMeshInput vertex_motion(InxVertexInput input, uint instance : SV_VulkanInstanceID)
{
    InxMeshInput result = inx_mesh_vertex(input, instance);
    InxInstanceData history = inx_instances[instance];
    result.motion_current_clip = float4(result.position.x, -result.position.y, result.position.zw);
    result.motion_previous_clip = result.motion_current_clip;
    if ((history.object_flags & 1u) != 0u)
    {
        InxSkinState skin = inx_skins[instance];
        skin.bone_offset = skin.previous_bone_offset;
        float3 previous_local = inx_skin_vertex(input, skin, inx_bones).position;
        float4 previous_clip = mul(inx_view.previous_view_projection, mul(inx_previous_model(history), float4(previous_local, 1.0)));
        result.motion_previous_clip = previous_clip;
    }
    inx_clip_device_bounds(result, instance);
    return result;
}
[shader("fragment")]
float4 fragment_main(InxMeshInput input, bool front_facing : SV_IsFrontFace) : SV_Target0
{
    return inx_shade_surface<false>(input, front_facing);
}
[shader("fragment")]
float4 fragment_forward_plus(InxMeshInput input, bool front_facing : SV_IsFrontFace) : SV_Target0
{
    return inx_shade_surface<true>(input, front_facing);
}
[shader("fragment")]
void fragment_depth(InxMeshInput input, bool front_facing : SV_IsFrontFace)
{
    inx_evaluate_surface(input, front_facing);
}
[shader("fragment")]
float fragment_shadow(InxMeshInput input, bool front_facing : SV_IsFrontFace) : SV_Depth
{
    // Evaluate the projected caster plane at the atlas texel center, even on grazing triangles.
    float3 projected = input.motion_current_clip.xyz / input.motion_current_clip.w;
    float3 plane = cross(ddx(projected), ddy(projected));
    float depth = plane.z != 0.0
        ? projected.z - dot(plane.xy, input.position.xy - projected.xy) / plane.z : input.position.z;
    inx_evaluate_surface(input, front_facing);
    return saturate(depth);
}
[shader("fragment")]
uint2 fragment_picking(InxMeshInput input, bool front_facing : SV_IsFrontFace) : SV_Target0
{
    inx_evaluate_surface(input, front_facing);
    return input.object_id;
}
[shader("fragment")]
float2 fragment_motion(InxMeshInput input, bool front_facing : SV_IsFrontFace) : SV_Target0
{
    inx_evaluate_surface(input, front_facing);
    return inx_mesh_has_motion_history(input)
        ? inx_mesh_motion(input) - inx_view_jitter_delta_uv() : float2(0.0);
}
[shader("fragment")]
float4 fragment_normal(InxMeshInput input, bool front_facing : SV_IsFrontFace) : SV_Target0
{
    return float4(inx_evaluate_surface(input, front_facing).normal_ws * 0.5 + 0.5, 1.0);
}
[shader("fragment")]
float4 fragment_base_color(InxMeshInput input, bool front_facing : SV_IsFrontFace) : SV_Target0
{
    InxSurfaceData surface = inx_evaluate_surface(input, front_facing);
    return float4(surface.albedo, surface.alpha);
}
"""


_SURFACE_SAMPLING = """
struct InxSurfaceDifferentials
{
    float3 position_dx;
    float3 position_dy;
    float2 uv_dx;
    float2 uv_dy;
};
interface InxSurfaceSampling
{
    float4 sample(InxColor2D image, float2 uv);
    InxSurfaceDifferentials differentials(InxMeshInput input, float2 uv, float2 uv_scale);
}
struct InxRasterSurfaceSampling : InxSurfaceSampling
{
    float4 sample(InxColor2D image, float2 uv)
    {
        return image.texture.Sample(image.sampler, uv);
    }
    InxSurfaceDifferentials differentials(InxMeshInput input, float2 uv, float2 uv_scale)
    {
        return {ddx(input.position_ws), ddy(input.position_ws), ddx(uv), ddy(uv)};
    }
};
struct InxSecondarySurfaceSampling : InxSurfaceSampling
{
    float lod;
    // Corresponding world-position and raw UV1 edges of the hit triangle.
    float3 edge_u_ws;
    float3 edge_v_ws;
    float2 edge_u_uv1;
    float2 edge_v_uv1;

    float4 sample(InxColor2D image, float2 uv)
    {
        return image.texture.SampleLevel(image.sampler, uv, lod);
    }
    InxSurfaceDifferentials differentials(InxMeshInput input, float2 uv, float2 uv_scale)
    {
        // Orient the cotangent frame by the shading normal, independent of the triangle winding.
        float orientation = dot(cross(edge_u_ws, edge_v_ws), input.normal_ws) < 0.0 ? -1.0 : 1.0;
        return {edge_u_ws, edge_v_ws * orientation, edge_u_uv1 * uv_scale, edge_v_uv1 * (uv_scale * orientation)};
    }
};
"""


_SURFACE_COVERAGE = """
bool inx_surface_covered(InxSurfaceData surface, float threshold)
{
    return !(threshold > 0.0 && surface.alpha < threshold);
}
"""


def _surface_evaluation(uses_surface_sampling):
    raster_arguments = "input, InxRasterSurfaceSampling()" if uses_surface_sampling else "input"
    source = _SURFACE_COVERAGE + f"""
InxSurfaceData inx_evaluate_surface(InxMeshInput input, bool front_facing)
{{
    InxSurfaceData surface = inx_prepare_surface(inx_author_surface({raster_arguments}), input, front_facing);
    if (!inx_surface_covered(surface, inx_parameters._alpha_clip_threshold)) discard;
    return surface;
}}
"""
    if uses_surface_sampling:
        source += """
bool inx_evaluate_secondary_surface(InxMeshInput input, bool front_facing, InxSecondarySurfaceSampling sampling,
                                    out InxSurfaceData surface)
{
    surface = inx_prepare_surface(inx_author_surface(input, sampling), input, front_facing);
    return inx_surface_covered(surface, inx_parameters._alpha_clip_threshold);
}
"""
    return source


_RAY_TEXTURE_PAGES = """
public Texture2DArray<float4> ray_color_pages;
public Texture2DArray<float4> ray_data_pages;
SamplerState ray_color_sampler;
SamplerState ray_data_sampler;

public float4 inx_ray_sample_texture(uint texture_ref, float2 uv, float lod)
{
    float3 coordinates = float3(uv, float(texture_ref >> 3u));
    switch (texture_ref & 7u)
    {
    case 0u: return ray_color_pages.SampleLevel(ray_color_sampler, coordinates, lod);
    case 1u: return ray_data_pages.SampleLevel(ray_data_sampler, coordinates, lod);
    case 2u: return float4(1.0);
    case 3u: return float4(0.0, 0.0, 0.0, 1.0);
    case 4u: return float4(128.0 / 255.0, 128.0 / 255.0, 1.0, 1.0);
    default: return float4(asfloat(0x7fc00000u | texture_ref));
    }
}
"""
_RAY_TEXTURE_MODULE = ("inx_ray_texture_pages", "<infernux/shader/ray-texture-pages>", _RAY_TEXTURE_PAGES)

_RAY_LIT_SAMPLING = """
struct InxRayLitSampling
{
    InxSecondarySurfaceSampling footprint;
    float4 sample(uint texture_ref, float2 uv)
    {
        return inx_ray_sample_texture(texture_ref, uv, footprint.lod);
    }
    InxSurfaceDifferentials differentials(InxMeshInput input, float2 uv, float2 uv_scale)
    {
        return footprint.differentials(input, uv, uv_scale);
    }
};
"""


_RAY_LIT_EVALUATION = """
// Matches RayMaterialParticipation in the material table producer.
static const uint inx_ray_material_lit = 1u;
static const uint inx_ray_material_excluded = 2u;
static const uint inx_ray_material_unlit = 3u;

bool inx_ray_has_surface(uint kind)
{
    return kind == inx_ray_material_lit || kind == inx_ray_material_unlit;
}

float3 inx_ray_unlit_radiance(RaySurfaceHit hit, InxSurfaceData surface)
{
    float3 geometric = hit.front_face ? hit.geometric_normal_ws : -hit.geometric_normal_ws;
    let context = inx_shading_context(hit.position_ws, geometric, 0.0, float2(0.0), 0xffffffffu, false, 0.0);
    return inx_ray_unlit_response::shade<false>(surface, context);
}

bool inx_evaluate_ray_lit(InxMeshInput input, bool front_facing, InxSecondarySurfaceSampling footprint,
                          out InxSurfaceData surface)
{
    InxRayLitSampling sampling = {footprint};
    surface = inx_ray_author_surface(input, sampling);
    surface = inx_prepare_surface(surface, input, front_facing);
    return inx_surface_covered(surface, inx_ray_alpha_clip(input.surface_id));
}

bool inx_evaluate_ray_lit_hit(RaySurfaceHit attributes, uint surface_id, float lod, out InxSurfaceData surface)
{
    InxMeshInput input = (InxMeshInput)0;
    input.surface_id = surface_id;
    input.position_ws = attributes.position_ws;
    input.normal_ws = attributes.normal_ws;
    input.tangent_ws = attributes.tangent_ws;
    input.color = attributes.color.rgb;
    input.uv0 = attributes.uv0;
    input.uv1 = attributes.uv1;
    input.line_color = float4(1.0);
    InxSecondarySurfaceSampling footprint = {lod, attributes.edge_u_ws, attributes.edge_v_ws,
                                             attributes.edge_u_uv1, attributes.edge_v_uv1};
    return inx_evaluate_ray_lit(input, attributes.front_face, footprint, surface);
}

struct InxRayLitHitPolicy : RayHitPolicy
{
    float lod;

    bool accepts(RayPage page, TraceRay ray, RayHit candidate)
    {
        uint participation = inx_ray_participation(candidate.surface_id);
        if (!inx_ray_has_surface(participation))
            return participation != inx_ray_material_excluded;
        if (!(inx_ray_alpha_clip(candidate.surface_id) > 0.0))
            return true;
        InxSurfaceData surface;
        return inx_evaluate_ray_lit_hit(ray_surface_at_hit(page, ray, candidate), candidate.surface_id, lod, surface);
    }
};

RayHit inx_trace_lit_closest(RayPage page, TraceRay ray, uint layer_mask, float lod)
{
    InxRayLitHitPolicy policy = {lod};
    return trace_closest(page, ray, layer_mask, policy);
}

bool inx_trace_lit_visibility(RayPage page, TraceRay ray, uint layer_mask, float lod)
{
    InxRayLitHitPolicy policy = {lod};
    return trace_visibility(page, ray, layer_mask, policy);
}

public InxRaySurfaceHit inx_world_ray_closest(float3 origin_ws, float3 direction_ws, float min_distance,
                                            float max_distance, uint layer_mask, float lod)
{
    TraceRay ray = {origin_ws, min_distance, direction_ws, max_distance};
    RayPage page = ray_page();
    InxRaySurfaceHit result = (InxRaySurfaceHit)0;
    result.distance = max_distance;
    result.available = page.status == ray_page_ready;
    if (!result.available)
        return result;
    RayHit hit = inx_trace_lit_closest(page, ray, layer_mask, lod);
    if (hit.triangle_index != ray_invalid_index)
    {
        if (!inx_ray_has_surface(inx_ray_participation(hit.surface_id)))
        {
            result.available = false;
            return result;
        }
        result.found = true;
        result.distance = hit.distance;
        RaySurfaceHit attributes = ray_surface_at_hit(page, ray, hit);
        result.position_ws = attributes.position_ws;
        result.surface_id = hit.surface_id;
        inx_evaluate_ray_lit_hit(attributes, hit.surface_id, lod, result.surface);
    }
    return result;
}

public bool inx_world_ray_visibility(float3 origin_ws, float3 direction_ws, float min_distance,
                                     float max_distance, uint layer_mask, float lod)
{
    TraceRay ray = {origin_ws, min_distance, direction_ws, max_distance};
    return inx_trace_lit_visibility(ray_page(), ray, layer_mask, lod);
}

float2 inx_ray_primary_uniform(uint index, uint pixel_seed)
{
    // Sobol dimension 3 (x^2 + x + 1, initial numerators 1, 3) stratifies
    // camera phase. These masks invert its leading three bits for this block.
    // Only the order changes. A surface visible at a single jitter phase
    // still receives directions from the entire hemisphere over time.
    uint phase = (countbits(index & 0x79e79e78u) & 1u)
               | ((countbits(index & 0xa28a28a0u) & 1u) << 1u)
               | ((countbits(index & 0x14514510u) & 1u) << 2u);
    index ^= phase;
    // Two base-2 Sobol dimensions. The second direction matrix is Pascal's triangle mod 2.
    uint x = reversebits(index);
    uint y = x;
    y ^= (y & 0x55555555u) << 1u;
    y ^= (y & 0x33333333u) << 2u;
    y ^= (y & 0x0f0f0f0fu) << 4u;
    y ^= (y & 0x00ff00ffu) << 8u;
    y ^= y << 16u;
    uint2 shift = uint2(inx_ray_sample_hash(pixel_seed), inx_ray_sample_hash(pixel_seed + 1u));
    return float2((uint2(x, y) ^ shift) >> 8u) * (1.0 / 16777216.0);
}

float inx_ray_offset(float position, float normal)
{
    if (abs(position) < 1.0 / 32.0)
        return position + normal * (1.0 / 65536.0);
    int offset = int(256.0 * normal);
    return asfloat(asint(position) + (position < 0.0 ? -offset : offset));
}

float3 inx_ray_origin(float3 position, float3 normal)
{
    return float3(inx_ray_offset(position.x, normal.x), inx_ray_offset(position.y, normal.y),
                  inx_ray_offset(position.z, normal.z));
}

float inx_ray_round_outside(float position, float displacement)
{
    if (displacement == 0.0)
        return position;
    if (position == 0.0)
        return asfloat(displacement > 0.0 ? 1u : 0x80000001u);
    uint bits = asuint(position);
    return asfloat((position > 0.0) == (displacement > 0.0) ? bits + 1u : bits - 1u);
}

float3 inx_ray_origin(float3 position, float3 normal, float3 error)
{
    float3 displacement = normal * dot(abs(normal), error);
    float3 outside = position + displacement;
    return float3(inx_ray_round_outside(outside.x, displacement.x),
                  inx_ray_round_outside(outside.y, displacement.y),
                  inx_ray_round_outside(outside.z, displacement.z));
}

struct InxRayShadowHitPolicy : RayHitPolicy
{
    InxBuffer_uint4 records;
    InxRayLitHitPolicy material;

    bool accepts(RayPage page, TraceRay ray, RayHit candidate)
    {
        return inx_buffer_load(records, candidate.surface_id * 3u).w != 0u &&
               material.accepts(page, ray, candidate);
    }
};

TraceRay inx_ray_light_visibility(float3 origin, InxTransportLight light)
{
    if (light.directional)
        return {origin, 0.0, light.sample.direction, 3.402823e38};
    float3 to_light = light.endpoint - origin;
    if (dot(to_light, light.sample.direction) <= 0.0)
        return {origin, 0.0, light.sample.direction, 0.0};
    float distance = length(to_light);
    // Leave the point/spot/rectangle endpoint outside the occlusion interval.
    return {origin, 0.0, to_light / distance, asfloat(asuint(distance) - 1u)};
}

float3 inx_ray_direct_radiance(RayPage page, RaySurfaceHit attributes, InxSurfaceData surface,
                               uint4 metadata, InxBuffer_uint4 records, float3 outgoing, uint seed)
{
    uint2 counts = inx_transport_light_counts();
    uint count = counts.x + counts.y;
    if (count == 0u)
        return float3(0.0);
    float3 geometric = attributes.front_face ? attributes.geometric_normal_ws : -attributes.geometric_normal_ws;
    InxPbrLightResponse response = inx_secondary_light_response(surface, surface.normal_ws, outgoing);
    uint strata = min(count, 4u);
    float3 radiance = float3(0.0);
    float3 total = float3(0.0);
    float3 orientation = float3(0.0);
    uint3 selected = uint3(ray_invalid_index);
    for (uint i = 0u; i < strata; ++i)
    {
        // Each stratum estimates its own sum. Only the shadow-dependent part needs a ray.
        uint stream = inx_ray_sample_hash(seed + i);
        float sample_u = float(inx_ray_sample_hash(stream ^ 0x91e10da5u) >> 8u) * (1.0 / 16777216.0);
        InxTransportLightSelection candidate = inx_sample_transport_light(
            count, strata, i, sample_u, attributes.position_ws, geometric, surface.normal_ws, metadata.x);
        uint index = candidate.index;
        uint corner = inx_ray_sample_hash(stream ^ 0x68bc21ebu) & 3u;
        InxTransportLight light = inx_transport_light(index, attributes.position_ws, corner);
        if ((light.layer_mask & metadata.x) == 0u || light.sample.attenuation <= 0.0 ||
            dot(geometric, light.sample.direction) <= 0.0 || dot(surface.normal_ws, light.sample.direction) <= 0.0)
            continue;
        float3 contribution = candidate.inverse_pdf * response.evaluate(light.sample);
        float strength = metadata.z != 0u ? light.shadow_strength : 0.0;
        radiance += (1.0 - strength) * contribution;
        float3 shadowed = strength * contribution;
        float3 weight = abs(shadowed);
        total += weight;
        // A shared random number correlates RGB without changing each channel's selection PDF.
        float replace_u = float(inx_ray_sample_hash(stream ^ 0xc2b2ae35u) >> 8u) * (1.0 / 16777216.0);
        uint key = (index << 2u) | corner;
        for (uint channel = 0u; channel < 3u; ++channel)
        {
            if (weight[channel] > replace_u * total[channel])
            {
                selected[channel] = key;
                orientation[channel] = sign(shadowed[channel]);
            }
        }
    }
    float3 origin = inx_ray_origin(attributes.position_ws, geometric, attributes.position_error_ws);
    InxRayLitHitPolicy material = {0.0};
    InxRayShadowHitPolicy caster = {records, material};
    // Channels that selected the same light endpoint share one visibility query.
    for (uint channel = 0u; channel < 3u; ++channel)
    {
        uint key = selected[channel];
        if (key == ray_invalid_index)
            continue;
        InxTransportLight light = inx_transport_light(key >> 2u, attributes.position_ws, key & 3u);
        TraceRay shadow_ray = inx_ray_light_visibility(origin, light);
        bool visible = shadow_ray.maximum <= 0.0 || trace_visibility(page, shadow_ray, light.layer_mask, caster);
        for (uint match = channel; match < 3u; ++match)
        {
            if (selected[match] == key)
            {
                if (visible)
                    radiance[match] += total[match] * orientation[match];
                selected[match] = ray_invalid_index;
            }
        }
    }
    return radiance;
}

struct InxRayTransportTerm
{
    float3 radiance;
    bool available;
    float distance;
};

// Proposal selection only. Both lobes are evaluated with the same mixture PDF below.
float inx_ray_diffuse_probability(InxSurfaceData surface, RayGgxMaterial material, float nov)
{
    const float3 luminance = float3(0.2126, 0.7152, 0.0722);
    float diffuse = dot(saturate(surface.albedo), luminance) * (1.0 - saturate(surface.metallic));
    float grazing = 1.0 - saturate(nov);
    float grazing2 = grazing * grazing;
    float glossy = lerp(dot(material.f0, luminance), material.f90, grazing2 * grazing2 * grazing) * material.strength;
    if (diffuse == 0.0) return 0.0;
    if (glossy == 0.0) return 1.0;
    // Keep both proposals represented for mixed materials and coloured incident light.
    return clamp(diffuse / (diffuse + glossy), 0.05, 0.95);
}

RayReflectedLightLobe inx_ray_reflected_light(float4 reflector, uint seed)
{
    RayReflectedLightLobe guide = (RayReflectedLightLobe)0;
    // Narrow lobes need a fresh light connection. Broad lobes share the existing world-cache budget.
    float importance = 1.0 - smoothstep(0.04, 0.16, reflector.w);
    if (reflector.w <= 0.0 || float(inx_ray_sample_hash(seed ^ 0x103bf8d3u) >> 8u) * (1.0 / 16777216.0) >= importance)
        return guide;
    uint count = inx_transport_light_counts().x;
    if (count > 0u)
    {
        uint index = inx_ray_sample_hash(seed ^ 0x459e1d09u) % count;
        let light = inx_transport_light(index, float3(0.0), 0u);
        if (light.layer_mask != 0u && dot(reflector.xyz, light.sample.direction) > 0.0)
            return {reflector.xyz, reflector.w, light.sample.direction};
    }
    return guide;
}

float3 inx_ray_secondary_weight(InxSurfaceData surface, float3 geometric, float3 outgoing, uint seed, uint sample_index,
                                 out RayDirectionProposal proposal, RayReflectedLightLobe guide)
{
    float3 normal = normalize(surface.normal_ws);
    RayGgxMaterial material = ray_lit_ggx(surface.albedo, surface.metallic, surface.smoothness,
                                          0.0, surface.specular_highlights);
    float diffuse_probability = inx_ray_diffuse_probability(surface, material, dot(normal, outgoing));
    if (guide.alpha > 0.0)
        proposal = ray_sample_reflected_light(ray_reflected_uniform3(seed, sample_index), normal, geometric, outgoing,
                                              material, diffuse_probability, guide);
    else
        proposal = ray_sample_continuous(inx_ray_uniform3(seed), normal, geometric, outgoing, material, diffuse_probability);
    if (!proposal.active)
        return float3(0.0);
    return inx_secondary_projected_bsdf(surface, normal, outgoing, proposal.direction) / proposal.pdf;
}

struct InxRayBounce
{
    RayHit hit;
    RaySurfaceHit attributes;
    InxSurfaceData surface;
    float3 outgoing;
    float3 throughput;
};

float inx_ray_mis_weight(float own_pdf, float other_pdf)
{
    if (!(own_pdf > 0.0))
        return 0.0;
    float relative = other_pdf / own_pdf;
    return 1.0 / (1.0 + relative * relative);
}

struct InxRayEmitterSample
{
    RaySurfaceHit attributes;
    float3 direction;
    float3 emission;
    float pdf;
    bool active;
};

InxRayEmitterSample inx_ray_emitter_sample(RayPage page, float3 position, float3 uniform, float maximum)
{
    InxRayEmitterSample sample = (InxRayEmitterSample)0;
    let selection = ray_sample_emissive(page, uniform);
    if (!selection.active)
        return sample;
    TraceRay ray = {position, 0.0, float3(0.0), maximum};
    sample.attributes = ray_surface_at_hit(page, ray, selection.hit);
    float3 delta = sample.attributes.position_ws - position;
    float distance = length(delta);
    if (!(distance > 0.0 && distance <= maximum))
        return sample;
    sample.direction = delta / distance;
    sample.attributes.front_face = dot(sample.direction, sample.attributes.geometric_normal_ws) < 0.0;
    sample.pdf = ray_emissive_solid_angle_pdf(selection.pmf, sample.attributes, position);
    if (!(sample.pdf > 0.0))
        return sample;
    InxSurfaceData surface;
    if (!inx_evaluate_ray_lit_hit(sample.attributes, selection.hit.surface_id, 0.0, surface))
        return sample;
    sample.emission = inx_ray_participation(selection.hit.surface_id) == inx_ray_material_unlit
        ? inx_ray_unlit_radiance(sample.attributes, surface) : surface.emission;
    sample.active = true;
    return sample;
}

bool inx_ray_emitter_visible(RayPage page, float3 origin, InxRayEmitterSample sample)
{
    float3 normal = sample.attributes.front_face ? sample.attributes.geometric_normal_ws
                                                : -sample.attributes.geometric_normal_ws;
    float3 target = inx_ray_origin(sample.attributes.position_ws, normal, sample.attributes.position_error_ws);
    float3 delta = target - origin;
    float distance = length(delta);
    if (!(distance > 0.0))
        return false;
    TraceRay visibility = {origin, 0.0, delta / distance, asfloat(asuint(distance) - 1u)};
    return inx_trace_lit_visibility(page, visibility, 0xffffffffu, 0.0);
}

float3 inx_ray_emissive_direct(RayPage page, RaySurfaceHit attributes, InxSurfaceData surface,
    float3 outgoing, float3 geometric, uint seed, float maximum, RayReflectedLightLobe guide)
{
    let sample = inx_ray_emitter_sample(page, attributes.position_ws, inx_ray_uniform3(seed ^ 0x39aa145du), maximum);
    if (!sample.active || dot(geometric, sample.direction) <= 0.0)
        return float3(0.0);
    float3 normal = normalize(surface.normal_ws);
    RayGgxMaterial material = ray_lit_ggx(surface.albedo, surface.metallic, surface.smoothness,
                                          0.0, surface.specular_highlights);
    float probability = inx_ray_diffuse_probability(surface, material, dot(normal, outgoing));
    float bsdf_pdf = ray_reflected_light_pdf(normal, geometric, outgoing, sample.direction, material,
                                              probability, guide);
    float3 response = inx_secondary_projected_bsdf(surface, normal, outgoing, sample.direction);
    if (!inx_ray_emitter_visible(page, inx_ray_origin(attributes.position_ws, geometric,
                                                     attributes.position_error_ws), sample))
        return float3(0.0);
    return sample.emission * response * (inx_ray_mis_weight(sample.pdf, bsdf_pdf) / sample.pdf);
}

// D owns analytic NEE and MIS-combined emissive/BSDF endpoint samples.
// It contains no further reflected scattering at the endpoint.
InxRayTransportTerm inx_ray_direct_sample(RayPage page, RaySurfaceHit attributes, InxSurfaceData surface,
    uint4 metadata, InxBuffer_uint4 records, float3 outgoing, uint seed, float maximum, float light_roughness,
    RayReflectedLightLobe guide, uint sample_index, out InxRayBounce bounce)
{
    bounce = (InxRayBounce)0;
    bounce.hit.triangle_index = ray_invalid_index;
    InxSurfaceData light_surface = surface;
    light_surface.smoothness = min(surface.smoothness, 1.0 - light_roughness);
    float3 radiance = inx_ray_direct_radiance(page, attributes, light_surface, metadata, records, outgoing, seed);
    float3 geometric = attributes.front_face ? attributes.geometric_normal_ws : -attributes.geometric_normal_ws;
    radiance += inx_ray_emissive_direct(page, attributes, surface, outgoing, geometric, seed, maximum, guide);
    RayDirectionProposal proposal;
    float3 weight = inx_ray_secondary_weight(surface, geometric, outgoing, seed ^ 0xb5297a4du, sample_index, proposal, guide);
    if (!proposal.active || max(weight.r, max(weight.g, weight.b)) <= 0.0)
        return {radiance, true, 0.0};
    TraceRay endpoint = {inx_ray_origin(attributes.position_ws, geometric, attributes.position_error_ws),
                         0.0, proposal.direction, maximum};
    RayHit hit = inx_trace_lit_closest(page, endpoint, 0xffffffffu, 0.0);
    if (hit.triangle_index == ray_invalid_index)
        return {radiance + weight * inx_environment_radiance(endpoint.direction), true, 0.0};
    uint kind = inx_ray_participation(hit.surface_id);
    if (!inx_ray_has_surface(kind))
        return {float3(0.0), false, hit.distance};
    bounce.attributes = ray_surface_at_hit(page, endpoint, hit);
    inx_evaluate_ray_lit_hit(bounce.attributes, hit.surface_id, 0.0, bounce.surface);
    float emission_weight = inx_ray_mis_weight(proposal.pdf,
        ray_emissive_pdf(hit, bounce.attributes, attributes.position_ws));
    if (kind == inx_ray_material_unlit)
        return {radiance + emission_weight * weight * inx_ray_unlit_radiance(bounce.attributes, bounce.surface), true, hit.distance};
    bounce.hit = hit;
    bounce.outgoing = -endpoint.direction;
    bounce.throughput = weight;
    return {radiance + emission_weight * weight * bounce.surface.emission, true, hit.distance};
}

InxRayTransportTerm inx_ray_direct_fresh(RayPage page, RaySurfaceHit attributes, InxSurfaceData surface,
                                          uint4 metadata, InxBuffer_uint4 records, float3 outgoing,
                                          uint seed, float maximum, float light_roughness = 0.0)
{
    InxRayBounce endpoint;
    return inx_ray_direct_sample(page, attributes, surface, metadata, records, outgoing, seed, maximum,
                                 light_roughness, (RayReflectedLightLobe)0, 0u, endpoint);
}

// Fixed additional depth one: sample reflected D(z), excluding Le(z) and an environment miss.
// No previous I1 value participates in this integrand.
InxRayTransportTerm inx_ray_tail_sample(RayPage page, RaySurfaceHit attributes, InxSurfaceData surface,
                                         InxBuffer_uint4 records, float3 outgoing, uint seed, float maximum)
{
    float3 geometric = attributes.front_face ? attributes.geometric_normal_ws : -attributes.geometric_normal_ws;
    RayDirectionProposal proposal;
    float3 weight = inx_ray_secondary_weight(surface, geometric, outgoing, seed ^ 0x1b56c4e9u, 0u,
                                             proposal, (RayReflectedLightLobe)0);
    if (!proposal.active || max(weight.r, max(weight.g, weight.b)) <= 0.0)
        return {float3(0.0), true, 0.0};
    TraceRay ray = {inx_ray_origin(attributes.position_ws, geometric, attributes.position_error_ws),
                    0.0, proposal.direction, maximum};
    RayHit hit = inx_trace_lit_closest(page, ray, 0xffffffffu, 0.0);
    if (hit.triangle_index == ray_invalid_index)
        return {float3(0.0), true, 0.0};
    uint kind = inx_ray_participation(hit.surface_id);
    if (kind == inx_ray_material_unlit)
        return {float3(0.0), true, hit.distance};
    if (kind != inx_ray_material_lit)
        return {float3(0.0), false, hit.distance};
    RaySurfaceHit next_attributes = ray_surface_at_hit(page, ray, hit);
    InxSurfaceData next_surface;
    inx_evaluate_ray_lit_hit(next_attributes, hit.surface_id, 0.0, next_surface);
    InxRayTransportTerm next = inx_ray_direct_fresh(page, next_attributes, next_surface,
        inx_buffer_load(records, hit.surface_id * 3u), records, -ray.direction,
        inx_ray_sample_hash(seed ^ 0x68e31da4u), maximum);
    return {weight * next.radiance, next.available, hit.distance};
}

public float4 inx_ray_radiance_parts(float3 origin, float3 direction, InxBuffer_uint4 records,
                                      uint seed, int part, float maximum)
{
    RayPage page = ray_page();
    if (page.status != ray_page_ready)
        return float4(0.0);
    TraceRay ray = {origin, 0.0, direction, maximum};
    RayHit hit = inx_trace_lit_closest(page, ray, 0xffffffffu, 0.0);
    if (hit.triangle_index == ray_invalid_index)
        return float4(part == 0 ? inx_environment_radiance(direction) : float3(0.0), 1.0);
    uint kind = inx_ray_participation(hit.surface_id);
    if (!inx_ray_has_surface(kind))
        return float4(0.0);
    RaySurfaceHit attributes = ray_surface_at_hit(page, ray, hit);
    InxSurfaceData surface;
    inx_evaluate_ray_lit_hit(attributes, hit.surface_id, 0.0, surface);
    if (part == 4)
        return float4(float(hit.surface_id), float(hit.primitive), hit.distance, 1.0);
    if (kind == inx_ray_material_unlit)
        return float4(part <= 1 ? inx_ray_unlit_radiance(attributes, surface) : float3(0.0), 1.0);
    if (part == 1)
        return float4(surface.emission, 1.0);
    InxRayTransportTerm term;
    if (part == 3)
        term = inx_ray_tail_sample(page, attributes, surface, records, -direction, seed, maximum);
    else
        term = inx_ray_direct_fresh(page, attributes, surface, inx_buffer_load(records, hit.surface_id * 3u),
                                     records, -direction, seed, maximum);
    return float4(term.radiance + (part == 0 ? surface.emission : float3(0.0)), term.available ? 1.0 : 0.0);
}

InxRaySignal inx_ray_signal_output(float3 signal, uint2 witness)
{
    return {float4(signal, 1.0), witness};
}

float inx_ray_tail_cone(InxSurfaceData surface)
{
    RayGgxMaterial scattering = ray_lit_ggx(surface.albedo, surface.metallic, surface.smoothness,
                                            0.0, surface.specular_highlights);
    if (scattering.strength > 0.0 && scattering.alpha < 0.1)
        return 0.0;
    float cone = scattering.strength > 0.0 ? min(0.06, 0.2 * scattering.alpha) : 0.06;
    // Request admission and lookup use exactly the same packed angular tolerance.
    return unpackHalf2x16ToFloat(packHalf2x16(float2(cone, 0.0))).x;
}

float3 inx_ray_cached_tail(RayPage page, RayHit hit, RaySurfaceHit attributes, InxBuffer_uint4 cache,
                           float3 outgoing, float cone)
{
    uint4 header = inx_buffer_load(cache, 0u);
    if (header.x == 0u)
        return float3(0.0);
    uint4 metric = inx_buffer_load(cache, 1u);
    float cell_size = asfloat(metric.x);
    int3 cell = ray_tail_cell(attributes.position_ws, cell_size);
    uint direction = ray_tail_direction(outgoing, metric.y);
    uint hash = ray_tail_hash(cell, 0u, hit.surface_id, hit.primitive, direction);
    float3 normal = attributes.front_face ? attributes.geometric_normal_ws : -attributes.geometric_normal_ws;
    for (uint probe = 0u; probe < 4u; ++probe)
    {
        uint slot = (hash + probe) % header.x;
        uint base = 2u + slot * 8u;
        uint4 key = inx_buffer_load(cache, base);
        uint4 identity = inx_buffer_load(cache, base + 1u);
        if (!ray_tail_same(key, identity, cell, hit.surface_id, hit.primitive, header.y))
            continue;
        uint4 anchor = inx_buffer_load(cache, base + 2u);
        uint4 wo = inx_buffer_load(cache, base + 3u);
        uint4 ng = inx_buffer_load(cache, base + 6u);
        if (wo.w != direction || anchor.w == 0u ||
            !ray_tail_representative(asfloat(anchor.xyz), asfloat(wo.xyz), asfloat(ng.xyz),
                                     attributes.position_ws, outgoing, normal, cell_size, cone))
            continue;
        uint4 exact = inx_buffer_load(cache, base + 7u);
        if (exact.x != hit.instance_index)
            continue;
        uint4 bary = inx_buffer_load(cache, base + 4u);
        TraceRay at_anchor = {asfloat(anchor.xyz), 0.0, -asfloat(wo.xyz), 0.0};
        RayHit anchor_hit = {0.0, asfloat(bary.xy), exact.x, exact.y, hit.primitive, hit.surface_id};
        RaySurfaceHit anchor_surface = ray_surface_at_hit(page, at_anchor, anchor_hit);
        float3 origin = inx_ray_origin(attributes.position_ws, normal, attributes.position_error_ws);
        float3 endpoint = inx_ray_origin(anchor_surface.position_ws, asfloat(ng.xyz), anchor_surface.position_error_ws);
        float3 link = endpoint - origin;
        float distance = length(link);
        if (distance > 0.0)
        {
            TraceRay visibility = {origin, 0.0, link / distance, asfloat(asuint(distance) - 1u)};
            if (!inx_trace_lit_visibility(page, visibility, 0xffffffffu, 0.0))
                continue;
        }
        return asfloat(inx_buffer_load(cache, base + 5u).xyz);
    }
    return float3(0.0);
}

InxRaySignal inx_ray_radiance(RayPage page, TraceRay ray, InxBuffer_uint4 records, InxBuffer_uint4 cache,
                              uint seed, float3 weight, float light_roughness, float4 reflector, uint sample_index,
                              float emission_pdf = 0.0, float3 emitter_origin = float3(0.0))
{
    RayHit hit = inx_trace_lit_closest(page, ray, 0xffffffffu, 0.0);
    if (hit.triangle_index == ray_invalid_index)
        return inx_ray_signal_output(inx_environment_radiance(ray.direction) * weight, uint2(0u));
    uint kind = inx_ray_participation(hit.surface_id);
    if (!inx_ray_has_surface(kind))
        return (InxRaySignal)0;
    RaySurfaceHit attributes = ray_surface_at_hit(page, ray, hit);
    InxSurfaceData secondary;
    inx_evaluate_ray_lit_hit(attributes, hit.surface_id, 0.0, secondary);
    // A delta ray has no competing continuous area-sampling path.
    float emission_weight = emission_pdf > 0.0 ? inx_ray_mis_weight(emission_pdf,
        ray_emissive_pdf(hit, attributes, emitter_origin)) : 1.0;
    if (kind == inx_ray_material_unlit)
        return inx_ray_signal_output(emission_weight * weight * inx_ray_unlit_radiance(attributes, secondary),
                                     uint2(hit.surface_id, asuint(hit.distance)));
    float maximum = asfloat(inx_buffer_load(cache, 1u).z);
    InxRayBounce bounce;
    let guide = inx_ray_reflected_light(reflector, seed);
    let direct = inx_ray_direct_sample(page, attributes, secondary, inx_buffer_load(records, hit.surface_id * 3u),
        records, -ray.direction, seed, maximum, light_roughness, guide, sample_index, bounce);
    if (!direct.available)
        return (InxRaySignal)0;
    float3 tail = float3(0.0);
    float cone = guide.alpha > 0.0 ? 0.0 : inx_ray_tail_cone(secondary);
    if (cone > 0.0)
        tail = inx_ray_cached_tail(page, hit, attributes, cache, -ray.direction, cone);
    else if (bounce.hit.triangle_index != ray_invalid_index)
    {
        // Reuse the endpoint already traced for emission. Only reflected D(z) belongs to the tail.
        let next = inx_ray_direct_fresh(page, bounce.attributes, bounce.surface,
            inx_buffer_load(records, bounce.hit.surface_id * 3u), records, bounce.outgoing,
            inx_ray_sample_hash(seed ^ 0x68e31da4u), maximum);
        if (next.available)
            tail = bounce.throughput * next.radiance;
    }
    return inx_ray_signal_output((emission_weight * secondary.emission + direct.radiance + tail) * weight,
                                  uint2(hit.surface_id, asuint(hit.distance)));
}

public float4 inx_ray_thin_reflection(float3 position_ws, float3 normal_ws, InxBuffer_uint4 records,
                                     float2 pixel_position, InxBuffer_uint4 cache)
{
    RayPage page = ray_page();
    if (page.status != ray_page_ready)
        return float4(0.0);
    float3 toward_eye = normalize(inx_camera_toward_eye(position_ws));
    float3 normal = dot(normal_ws, toward_eye) < 0.0 ? -normal_ws : normal_ws;
    TraceRay ray = {inx_ray_origin(position_ws, normal), 0.0, reflect(-toward_eye, normal),
                    inx_view.projection_params.y};
    uint2 pixel = uint2(pixel_position);
    uint seed = inx_ray_sample_hash(pixel.x ^ inx_ray_sample_hash(pixel.y) ^ inx_view_sample_index());
    return inx_ray_radiance(page, ray, records, cache, seed, float3(1.0), 0.0, float4(0.0), 0u).radiance;
}

uint2 inx_ray_path_key(float2 position, bool specular, bool temporal)
{
    uint2 pixel = uint2(position);
    uint seed = inx_ray_sample_hash(pixel.x ^ inx_ray_sample_hash(pixel.y));
    return uint2(specular ? inx_ray_sample_hash(seed) : seed, temporal ? inx_view_sample_index() : 0u);
}

RayDirectionSample inx_ray_primary_direction(InxSurfaceData surface, InxShadingContext context,
                                              float3 geometric_normal, bool specular, bool temporal, uint2 key,
                                              out uint seed, out float4 reflector)
{
    seed = key.x;
    float2 uniform = inx_ray_primary_uniform(key.y, seed);
    if (temporal)
        seed ^= inx_ray_sample_hash(key.y);
    float3 shading_normal = normalize(surface.normal_ws);
    reflector = float4(0.0);
    RayDirectionSample sample;
    if (specular)
    {
        float3 outgoing = normalize(inx_camera_toward_eye(context.position_ws));
        RayGgxMaterial material = ray_lit_ggx(surface.albedo, surface.metallic, surface.smoothness,
                                              context.normal_variance, surface.specular_highlights);
        reflector = float4(shading_normal, material.alpha);
        sample = ray_sample_ggx(uniform, shading_normal, geometric_normal, outgoing, material);
    }
    else
        sample = ray_sample_irradiance(uniform, shading_normal, geometric_normal);
    return sample;
}

float3 inx_ray_primary_emissive(RayPage page, InxSurfaceData surface, InxShadingContext context,
    float3 geometric, bool specular, uint seed, uint index)
{
    float3 normal = normalize(surface.normal_ws);
    float3 outgoing = normalize(inx_camera_toward_eye(context.position_ws));
    RayGgxMaterial material = ray_lit_ggx(surface.albedo, surface.metallic, surface.smoothness,
                                          context.normal_variance, surface.specular_highlights);
    if (specular && material.alpha == 0.0)
        return float3(0.0);
    let sample = inx_ray_emitter_sample(page, context.position_ws,
        ray_reflected_uniform3(seed ^ 0xa9d53e47u, index), inx_view.projection_params.y);
    if (!sample.active || dot(geometric, sample.direction) <= 0.0)
        return float3(0.0);
    float cosine = max(dot(normal, sample.direction), 0.0);
    float bsdf_pdf = cosine * 0.31830988618379067154;
    float3 response = float3(cosine);
    if (specular)
    {
        let terms = ray_evaluate_ggx(normal, geometric, outgoing, sample.direction, material);
        bsdf_pdf = terms.pdf;
        response = terms.projected_bsdf;
    }
    if (!inx_ray_emitter_visible(page, inx_ray_origin(context.position_ws, geometric), sample))
        return float3(0.0);
    return sample.emission * response * (inx_ray_mis_weight(sample.pdf, bsdf_pdf) / sample.pdf);
}

public InxRaySignal inx_ray_replay_path(InxSurfaceData surface, InxShadingContext context, float3 geometric_normal,
                                        InxBuffer_uint4 records, uint model_id, bool specular, bool temporal,
                                        InxBuffer_uint4 cache, uint2 key)
{
    if (model_id != INX_PBR_MODEL_ID)
        return (InxRaySignal)0;
    uint seed;
    float4 reflector;
    RayDirectionSample sample = inx_ray_primary_direction(surface, context, geometric_normal, specular,
                                                          temporal, key, seed, reflector);
    RayPage page = ray_page();
    if (page.status != ray_page_ready)
        return (InxRaySignal)0;
    uint index = key.y;
    float3 emission = inx_ray_primary_emissive(page, surface, context, geometric_normal, specular, seed, index);
    if (!sample.active)
        return inx_ray_signal_output(emission, uint2(ray_invalid_index, 0u));
    TraceRay ray = {inx_ray_origin(context.position_ws, geometric_normal), 0.0, sample.direction, inx_view.projection_params.y};
    // One of N directions represents solid angle 1/(N*pdf). Project the cap's
    // half-angle onto microfacet slopes for the next analytic light connection.
    // alpha = tan(theta/2), perceptual roughness = sqrt(alpha). Emission and
    // BSDF-sampled endpoints retain the authored material and its matching PDF.
    // Match the footprint to the reconstruction budget. After geometry changes,
    // at most eight reflected samples can contribute to the displayed history.
    bool steady_visibility = world_ray_geometry[1].y == inx_previous_ray_geometry_serial();
    float sample_count = steady_visibility ? (specular ? 96.0 : 64.0) : (specular ? 8.0 : 4.0);
    float cap = max(4.0 * 3.14159265359 * sample_count * sample.pdf - 1.0, 1.0);
    float light_roughness = temporal && !sample.delta ? rsqrt(sqrt(cap)) : 0.0;
    InxRaySignal signal = inx_ray_radiance(page, ray, records, cache, seed, sample.weight, light_roughness, reflector,
                                           index, sample.delta ? 0.0 : sample.pdf, context.position_ws);
    signal.radiance.rgb += emission;
    return signal;
}

public InxRaySignal inx_ray_indirect_signal(InxSurfaceData surface, InxShadingContext context, float3 geometric_normal,
                                            InxBuffer_uint4 records, uint model_id, bool specular, bool temporal,
                                            InxBuffer_uint4 cache)
{
    return inx_ray_replay_path(surface, context, geometric_normal, records, model_id, specular, temporal, cache,
                               inx_ray_path_key(context.pixel_position, specular, temporal));
}

// A positive target leaves support for directions that were black before a lighting or visibility change.
float inx_ray_path_target(float3 radiance)
{
    float3 magnitude = abs(radiance);
    float peak = max(magnitude.r, max(magnitude.g, magnitude.b));
    return 0.125 + peak / (1.0 + peak);
}

public InxRayPathEstimate inx_ray_path_temporal(InxSurfaceData surface, InxShadingContext context, float3 normal,
    InxBuffer_uint4 records, uint model_id, bool specular, InxBuffer_uint4 cache, uint4 history)
{
    uint2 key = inx_ray_path_key(context.pixel_position, specular, true);
    uint represented = min(history.w, 4u);
    uint count = 1u + represented;
    uint2 selected = key;
    float selected_target = 1.0;
    float total = 0.0;
    float3 estimate = float3(0.0);
    InxRaySignal result = (InxRaySignal)0;
    // Use one traversal call site for the bounded stream on both the native and browser GPU compilers.
    for (uint i = 0u; i < (represented == 0u ? 1u : 2u); ++i)
    {
        uint2 candidate = i == 0u ? key : history.xy;
        InxRaySignal sample = inx_ray_replay_path(surface, context, normal, records, model_id, specular, true,
                                                  cache, candidate);
        if (i == 0u)
        {
            result = sample;
            if (result.radiance.w <= 0.0)
                return {result, uint4(0)};
        }
        float mass = i == 0u ? 1.0 : float(represented) * asfloat(history.z);
        float target = inx_ray_path_target(sample.radiance.rgb);
        float score = mass * target;
        total += score;
        uint bits = inx_ray_sample_hash(key.x ^ inx_ray_sample_hash(key.y) ^ ((i + 1u) * 0xb92367d5u));
        if (float(bits >> 8u) * (1.0 / 16777216.0) * total < score)
        {
            selected = candidate;
            selected_target = target;
        }
        estimate += mass * sample.radiance.rgb;
    }
    // Only proposal keys recur. Keep the hit of the fresh ray available to filters and project code.
    result.radiance.rgb = estimate / float(count);
    return {result, uint4(selected, asuint(total / (float(count) * selected_target)), count)};
}

public int2 inx_ray_path_neighbor(uint2 cell)
{
    // A shifted 2x2 tile is paired by an involution. Both pixels read the same temporal publication.
    uint shift = inx_ray_sample_hash(inx_view_sample_index() ^ 0x39faa78du);
    uint2 q = cell + uint2(shift & 1u, (shift >> 1u) & 1u);
    uint2 tile = q >> 1u;
    uint axis = inx_ray_sample_hash(tile.x ^ inx_ray_sample_hash(tile.y ^ shift)) & 3u;
    uint2 mask = axis == 0u ? uint2(1, 0) : (axis == 1u ? uint2(0, 1) : uint2(1, 1));
    return int2(cell) + int2(q ^ mask) - int2(q);
}

public uint2 inx_ray_tail_pixel(uint index, uint2 extent)
{
    uint seed = inx_ray_sample_hash(index ^ inx_ray_sample_hash(inx_view_sample_index()));
    return uint2(seed % extent.x, inx_ray_sample_hash(seed) % extent.y);
}

InxRayTailRequest inx_ray_tail_request(RayPage page, TraceRay ray)
{
    InxRayTailRequest request = (InxRayTailRequest)0;
    if (page.status != ray_page_ready)
        return request;
    RayHit hit = inx_trace_lit_closest(page, ray, 0xffffffffu, 0.0);
    if (hit.triangle_index == ray_invalid_index ||
        inx_ray_participation(hit.surface_id) != inx_ray_material_lit)
        return request;
    RaySurfaceHit attributes = ray_surface_at_hit(page, ray, hit);
    InxSurfaceData surface;
    inx_evaluate_ray_lit_hit(attributes, hit.surface_id, 0.0, surface);
    float cone = inx_ray_tail_cone(surface);
    if (cone == 0.0)
        return request;
    float3 geometric = attributes.front_face ? attributes.geometric_normal_ws : -attributes.geometric_normal_ws;
    request.identity = uint4(hit.instance_index, hit.triangle_index, asuint(hit.barycentrics));
    request.anchor = uint4(asuint(attributes.position_ws), packHalf2x16(float2(0.0, cone)) | 1u);
    request.outgoing = uint4(asuint(-ray.direction), hit.surface_id);
    request.normal = uint4(asuint(geometric), hit.primitive);
    return request;
}

public InxRayTailRequest inx_ray_tail_primary_request(InxSurfaceData surface, InxShadingContext context,
    float3 geometric_normal, uint model_id, bool specular, bool temporal)
{
    if (model_id != INX_PBR_MODEL_ID)
        return (InxRayTailRequest)0;
    uint seed;
    float4 reflector;
    uint2 key = inx_ray_path_key(context.pixel_position, specular, temporal);
    RayDirectionSample sample = inx_ray_primary_direction(surface, context, geometric_normal, specular,
                                                          temporal, key, seed, reflector);
    if (!sample.active)
        return (InxRayTailRequest)0;
    TraceRay ray = {inx_ray_origin(context.position_ws, geometric_normal), 0.0, sample.direction,
                    inx_view.projection_params.y};
    return inx_ray_tail_request(ray_page(), ray);
}

public InxRayTailRequest inx_ray_tail_thin_request(float3 position, float3 normal)
{
    float3 outgoing = normalize(inx_camera_toward_eye(position));
    normal = dot(normal, outgoing) < 0.0 ? -normal : normal;
    TraceRay ray = {inx_ray_origin(position, normal), 0.0, reflect(-outgoing, normal), inx_view.projection_params.y};
    return inx_ray_tail_request(ray_page(), ray);
}

public float4 inx_ray_tail_estimate(InxBuffer_uint4 requests, InxBuffer_uint4 cache,
                                     InxBuffer_uint4 records, uint index)
{
    // The native owner supplies one stable representative and one complete control header.
    uint4 identity = inx_buffer_load(requests, 4u + index * 4u);
    uint4 anchor = inx_buffer_load(requests, 5u + index * 4u);
    if ((anchor.w & 1u) == 0u)
        return float4(0.0, 0.0, 0.0, -1.0);
    uint4 outgoing = inx_buffer_load(requests, 6u + index * 4u);
    uint4 normal = inx_buffer_load(requests, 7u + index * 4u);
    RayPage page = ray_page();
    if (page.status != ray_page_ready)
        return float4(0.0, 0.0, 0.0, -1.0);
    uint slot = anchor.w >> 2u;
    uint count = 0u;
    if ((anchor.w & 2u) == 0u)
        count = inx_buffer_load(cache, 4u + slot * 8u).w;
    float3 wo = asfloat(outgoing.xyz);
    TraceRay at_anchor = {asfloat(anchor.xyz), 0.0, -wo, 0.0};
    RayHit hit = {0.0, asfloat(identity.zw), identity.x, identity.y, normal.w, outgoing.w};
    RaySurfaceHit attributes = ray_surface_at_hit(page, at_anchor, hit);
    InxSurfaceData surface;
    inx_evaluate_ray_lit_hit(attributes, hit.surface_id, 0.0, surface);
    uint4 world = inx_buffer_load(requests, 3u);
    uint seed = inx_ray_sample_hash(slot ^ inx_ray_sample_hash(count) ^ world.x ^ inx_ray_sample_hash(world.y));
    InxRayTransportTerm result = inx_ray_tail_sample(page, attributes, surface, records, wo, seed, asfloat(world.z));
    return float4(result.radiance, result.available ? result.distance : -1.0);
}
"""


_SURFACE_LIGHTING = """
float4 inx_shade_surface<let tiled_lighting : bool>(InxMeshInput input, bool front_facing)
{
    InxShadingContext context = inx_shading_context(input);
    InxSurfaceData surface = inx_evaluate_surface(input, front_facing);
    context.normal_variance = inx_normal_variance(surface.normal_ws);
    context.receives_shadows = context.receives_shadows && INX_RECEIVE_SHADOWS;
    return float4(inx_shading_model::shade<tiled_lighting>(surface, context), surface.alpha);
}
"""

_SURFACE_GBUFFER = """
[shader("fragment")]
InxGBufferOutput fragment_gbuffer(InxMeshInput input, bool front_facing : SV_IsFrontFace)
{
    return inx_pack_gbuffer<INX_GBUFFER_PBR>(inx_evaluate_surface(input, front_facing), input.surface_id);
}

// Raster depth determines coverage. A ray starts on the original triangle:
// subpixel snapping can place its unprojected depth on either side of that plane.
float inx_ray_receiver_depth(InxMeshInput input, float3 plane_normal)
{
    float2 ndc = input.position.xy * inx_globals.screen_params.zw * 2.0 - 1.0;
    float4 point = mul(inx_view.inverse_view_projection, float4(ndc, input.position.z, 1.0));
    float4 step_z = mul(inx_view.inverse_view_projection, float4(0.0, 0.0, 1.0, 0.0));
    float residual = dot(plane_normal, input.position_ws * point.w - point.xyz);
    float slope = dot(plane_normal, step_z.xyz - input.position_ws * step_z.w);
    return slope != 0.0 ? saturate(input.position.z + residual / slope) : input.position.z;
}

[shader("fragment")]
uint4 fragment_ray_surface(InxMeshInput input, bool front_facing : SV_IsFrontFace) : SV_Target0
{
    // Form the plane before coverage can discard a lane in this derivative quad.
    float3 normal = cross(ddx(input.position_ws), ddy(input.position_ws));
    inx_evaluate_surface(input, front_facing);
    float length_squared = dot(normal, normal);
    if (!(length_squared > 0.0))
        return uint4(0u);
    float3 toward_eye = inx_camera_toward_eye(input.position_ws);
    normal *= (dot(normal, toward_eye) < 0.0 ? -1.0 : 1.0) * rsqrt(length_squared);
    float2 encoded = inx_encode_surface_normal(normal);
    uint depth_and_motion = $depth::pack_ray_surface_depth(input.position.z, inx_ray_receiver_depth(input, normal),
                                                          input.motion_previous_clip, inx_mesh_has_motion_history(input));
    return uint4(input.surface_id, packHalf2x16(encoded), packHalf2x16(inx_mesh_motion(input)), depth_and_motion);
}
"""

_SURFACE_THIN = """
struct InxThinSurfaceOutput
{
    uint4 layer : SV_Target0;
    uint4 optics : SV_Target1;
};
[shader("fragment")]
InxThinSurfaceOutput fragment_thin_surface(InxMeshInput input, bool front_facing : SV_IsFrontFace)
{
    // Evaluate the plane over the whole quad before authored coverage discards.
    float3 plane = cross(ddx(input.position_ws), ddy(input.position_ws));
    float plane_squared = dot(plane, plane);
    InxSurfaceData surface = inx_evaluate_surface(input, front_facing);
    if (!(surface.alpha > 0.0 && plane_squared > 0.0))
        discard;
    float3 toward_eye = inx_camera_toward_eye(input.position_ws);
    float3 normal = plane * ((dot(plane, toward_eye) < 0.0 ? -1.0 : 1.0) * rsqrt(plane_squared));
    InxThinSheetData sheet = thin_sheet(input);
    float ior = clamp(sheet.ior, 0.0001, 65504.0);
    float3 optical_depth = min(max(sheet.absorption, float3(0.0)) * max(sheet.thickness, 0.0), float3(65504.0));
    float3 emission = clamp(surface.emission, float3(0.0), float3(65504.0));
    InxThinSurfaceOutput result;
    result.layer = uint4(input.surface_id, packHalf2x16(inx_encode_surface_normal(normal)),
                         packHalf2x16(inx_mesh_motion(input)), asuint(input.position.z));
    result.optics = uint4(packHalf2x16(float2(ior, optical_depth.r)),
                          packHalf2x16(optical_depth.gb), packHalf2x16(emission.rg),
                          packHalf2x16(float2(emission.b, saturate(surface.alpha))));
    return result;
}
"""


@dataclass
class _FragmentSource:
    author: _Author
    graph: LibraryGraph
    declaration: ast.ClassDef
    main: ast.FunctionDef
    helpers: list
    shading_model: tuple | None
    dependencies: tuple
    thin: ast.FunctionDef | None
    info: dict


def _read_fragment(source, source_path, reader=None, graph=None, *, author_type=_Author, ray_surfaces=()):
    info = read_shader_info(source, source_path)
    author = author_type(source, source_path)
    author.alias, imports, body = declarations(source, source_path)
    author.aliases = (author.alias,)
    if len(body) != 1 or not isinstance(body[0], ast.ClassDef):
        author.error(ast.Constant(lineno=1, col_offset=0), "Declare one shader class")
    declaration = body[0]
    bases = {author.alias + ".fullscreen": "fullscreen", author.alias + ".frag": "mesh",
             author.alias + ".surface": "mesh", author.alias + ".particle": "particle",
             author.alias + ".screen_ui": "screen_ui", author.alias + ".world_ui": "world_ui"}
    if declaration.decorator_list or len(declaration.bases) != 1 or author.qualified(declaration.bases[0]) not in bases:
        author.error(declaration, "Declare a fullscreen, mesh, particle, screen UI or world UI shader")
    author.domain = bases[author.qualified(declaration.bases[0])]
    graph = graph if graph is not None else LibraryGraph(reader, allow_world_rays=author.domain == "fullscreen", ray_surfaces=ray_surfaces)
    author.functions, dependencies = graph.imports(imports, author)
    author.lighting_mode = "true" if author.domain == "fullscreen" else "false"
    author.is_surface = author.qualified(declaration.bases[0]) == author.alias + ".surface"
    entry_name = "surface" if author.is_surface else "main"
    main, helpers, shading_model = None, [], None
    for statement in declaration.body:
        if info_field(statement) in INFO_DEFAULTS:
            continue
        if (author.domain == "mesh" and isinstance(statement, ast.Assign) and len(statement.targets) == 1
                and author.qualified(statement.targets[0]) == "shading_model"):
            if shading_model is not None or not author.is_surface:
                author.error(statement, "Declare one shading model on a Surface", code="author.reference")
            shading_model = (graph.reference(statement.value, author, stage="shadingmodel", suffix=".shadingmodel.py"), statement)
        elif isinstance(statement, ast.Assign) and any(author.qualified(target) == "vertex_shader" for target in statement.targets):
            author.error(statement, "Choose the vertex shader on the material; fragment authors do not select a vertex asset", code="author.reference")
        elif isinstance(statement, (ast.Assign, ast.AnnAssign)):
            author.parameter(statement)
        elif isinstance(statement, ast.FunctionDef) and statement.name == entry_name and main is None:
            main = statement
        elif isinstance(statement, ast.FunctionDef) and statement.name != entry_name:
            author.functions["self." + statement.name] = author.signature(statement, statement.name, method=True)
            helpers.append(statement)
        elif isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant) and isinstance(statement.value.value, str):
            pass
        else:
            author.error(statement, "Expected a shader parameter or typed method")
    if main is None:
        author.error(declaration, "Shader class needs a " + entry_name + " entry")
    thin = next((method for method in helpers if method.name == "thin_sheet"), None)
    if thin is not None:
        signature = author.functions["self.thin_sheet"]
        if not author.is_surface or info["surface_type"] != "transparent":
            author.error(thin, "thin_sheet belongs to a transparent shader.surface", code="author.stage")
        if signature.result != "InxThinSheetData" or signature.arguments != ("InxMeshInput",):
            author.error(thin, "thin_sheet accepts a SurfaceContext and returns ThinSheetData", code="author.type")
    if uses_surface_lighting(body, author.alias, author.functions):
        for helper in helpers:
            name = "self." + helper.name
            author.functions[name] = replace(author.functions[name], lighting_specialization=True)
    if author.domain in {"mesh", "particle"}:
        author.fields["_alpha_clip_threshold"] = ShaderParameter("_alpha_clip_threshold", "float", 0.0, "", internal=True)
        author.field_nodes["_alpha_clip_threshold"] = declaration
        author.field_sources["_alpha_clip_threshold"] = (author, declaration)
    if author.is_surface and shading_model is None:
        shading_model = (default_shading_model(), declaration)
    return _FragmentSource(author, graph, declaration, main, helpers, shading_model, dependencies, thin, info)


def parse_shader(source: str, source_path: str, module_name: str, *, reader=None, vertex_path=None, ray_surfaces=()) -> ShaderDefinition:
    fragment = _read_fragment(source, source_path, reader, ray_surfaces=ray_surfaces)
    author, graph, declaration = fragment.author, fragment.graph, fragment.declaration
    model = None
    if fragment.shading_model:
        model_path, location = fragment.shading_model
        model = read_shading_model(model_path, graph, author, location)
    vertex = None
    if vertex_path is not None:
        if author.domain != "mesh":
            author.error(declaration, "A material vertex can only be paired with a mesh fragment", code="author.reference")
        location = declaration
        vertex = read_vertex(vertex_path, graph, author, location)
        if author.is_surface and "self.project" in vertex.author.functions:
            author.error(location, "A projected vertex pairs with shader.frag; shader.surface shares engine projection across geometry passes",
                         code="author.reference")
        for name, field in vertex.author.fields.items():
            previous = author.fields.get(name)
            if previous is not None:
                if replace(previous, stages=field.stages) != field:
                    author.error(location, "Vertex and fragment declarations disagree on material field: " + name, code="author.parameter")
                field = replace(previous, stages=("vertex", "fragment"))
            author.fields[name] = field
            if previous is None:
                author.field_nodes[name] = vertex.author.field_nodes[name]
                author.field_sources[name] = vertex.author.field_sources[name]
            author.types["self." + name] = vertex.author.types["self." + name]
    parameters = tuple(field for field in author.fields.values() if isinstance(field, ShaderParameter))
    inputs = tuple(field for field in author.fields.values() if isinstance(field, ShaderInput))
    if author.domain in {"mesh", "particle", "screen_ui", "world_ui"}:
        if inputs:
            author.error(declaration, "Material image resources are declared as material properties")
    dependencies = tuple(dict.fromkeys((*fragment.dependencies, *(vertex.dependencies if vertex else ()))))
    code = ["import " + RESOURCE_MODULE[0] + ";", "import inx_geometry;", *("import " + name + ";" for name in dependencies)]
    code.extend(engine_imports((declaration,), author.alias, graph=graph, owner=author))
    if graph.ray_lit is not None:
        for field in _WORLD_RAY_INPUTS:
            if field.name in author.fields:
                author.error(author.field_nodes[field.name], "World ray input name is reserved: " + field.name,
                             code="author.parameter")
    if vertex or author.is_surface:
        code.append("import inx_frame;")
    if author.domain == "mesh":
        code.append("import inx_camera;")
    elif author.domain == "particle":
        code.append("import inx_particle;")
    code = list(dict.fromkeys(code))
    if author.domain in {"mesh", "particle"}:
        code.append(_TRANSPARENT_LAYER)
    if author.is_surface:
        code.insert(0, "import inx_surface_model;")
    for field in inputs:
        code.extend([author.source_directive(author.field_nodes[field.name]),
                     ("ByteAddressBuffer" if field.kind == "Buffer" else RESOURCE_BINDINGS[field.kind]) + " " + resource_symbol(field.name) + ";"])
    code.extend(author.parameter_declarations(parameters))
    vertex_ranges = ()
    if vertex:
        vertex_code, vertex_ranges = vertex.emit()
        code.extend([vertex_code, _VERTEX_DEFORM])
        code.append(_mesh_vertex(deform=True, project="self.project" in vertex.author.functions))
    else:
        if author.domain == "mesh":
            code.append(_mesh_vertex())
        elif author.domain == "fullscreen":
            code.append(_FULLSCREEN_VERTEX)
    for helper in fragment.helpers:
        emitter = _Author(source, source_path)
        emitter.alias = author.alias
        emitter.aliases = (author.alias,)
        emitter.fields = author.fields
        emitter.domain = author.domain
        emitter.types = author.types.copy()
        emitter.functions = author.functions
        code.extend(emitter.emit_function(helper, author.functions["self." + helper.name], method=True))
        author.source_ranges.extend(emitter.source_ranges)
    entry = author.function(fragment.main)
    if author.uses_surface_sampling:
        code.append(_SURFACE_SAMPLING)
    code.extend(entry)
    if author.domain in {"screen_ui", "world_ui"}:
        from infernux.lib import _Infernux
        code.append(_Infernux._shader_ui_author_source(
            author.domain == "world_ui", author.annotation(fragment.main.returns) == "InxFragmentOutput"))
    if author.domain == "particle":
        code.append(_particle_passes(author.annotation(fragment.main.returns)))
    if author.is_surface:
        code.append("static const bool INX_RECEIVE_SHADOWS = " + ("true" if fragment.info["receive_shadows"] else "false") + ";")
        code.append(_surface_evaluation(author.uses_surface_sampling))
        code.append(_SURFACE_LIGHTING)
        passes = _SURFACE_PASSES
        if vertex:
            passes = passes.replace("float3 previous_local = inx_skin_vertex(input, skin, inx_bones).position;",
                                    "InxVertexInput previous = inx_apply_vertex(input, instance, inx_globals.frame_params.w, inx_previous_model(history));\n"
                                    "        float3 previous_local = inx_skin_vertex(previous, skin, inx_bones).position;")
        code.append(passes)
        for name, front in _TRANSPARENT_PARTITIONS:
            code.append(_partitioned_entry(
                name, "float4", "InxMeshInput input, bool front_facing : SV_IsFrontFace", " : SV_Target0",
                "inx_shade_surface<true>(input, front_facing)", front,
                "input.position.z"))
        if model.deferred_model:
            depth_layout = graph.load_reference("infernux.shader.lib.ray_surface_depth", 0, author, declaration)
            code.insert(0, "import " + depth_layout.name + ";")
            pbr = "true" if model.deferred_model == "PBR" else "false"
            code.append(f"static const bool INX_GBUFFER_PBR = {pbr};")
            code.append(_SURFACE_GBUFFER.replace("$depth", depth_layout.name))
        if fragment.thin is not None:
            code.append(_SURFACE_THIN)
    libraries = graph.sources()
    if model:
        libraries = (*libraries, ("inx_surface_model", model.path, model.source))
    return ShaderDefinition(declaration.name, source_path, module_name, parameters,
                            "\n".join(code), libraries, (*inputs, *_WORLD_RAY_INPUTS) if graph.ray_lit else inputs,
                            tuple(dict.fromkeys((*graph.ranges(), *(model.source_ranges if model else ()), *vertex_ranges,
                                                 *(vertex.author.source_ranges if vertex else ()), *author.source_ranges))),
                            author.domain, author.is_surface, model.name if model else None, model.deferred_model if model else None,
                            fragment.info, vertex.path if vertex else None, fragment.thin is not None, author.color_outputs,
                            getattr(graph.ray_lit, "manifests", ()),
                            graph.ray_surfaces if graph.ray_lit else None)


def read_shader(path: Path, module_name: str) -> ShaderDefinition:
    return parse_shader(path.read_text(encoding="utf-8-sig"), str(path), module_name)


def _prepare_ray_lit(reader=None, *, _graph=None):
    """Compile the raster-authored surfaces against the shared ray material table."""
    graph = _graph if _graph is not None else LibraryGraph(reader)
    entries, dependencies, ranges = [], [], []
    graph.ordered.append(Module(*_RAY_TEXTURE_MODULE, {}))
    fragments = {}
    for name in ("lit", "unlit"):
        path = default_shading_model().with_name(name + ".frag.py")
        source = graph.reader(str(path))
        if source is None:
            raise FileNotFoundError(path)
        fragment = _read_fragment(source.removeprefix("\ufeff"), str(path), graph=graph)
        author = fragment.author
        author.ray_entry = "inx_ray_author_" + name
        entries.extend(author.function(fragment.main))
        dependencies.extend(fragment.dependencies)
        ranges.extend(author.source_ranges)
        fragments[name] = fragment

    unlit = fragments["unlit"]
    model_path, location = unlit.shading_model
    response_name = "inx_ray_unlit_response"
    model = read_shading_model(model_path, graph, unlit.author, location, namespace=response_name)
    graph.ordered.append(Module(response_name, model.path, model.source, {}, model.source_ranges))
    dependencies.append(response_name)
    lit = fragments["lit"].author
    parameters = tuple(lit.fields.values())
    values = [field for field in parameters if field.kind != "texture_2d"]
    textures = [field for field in parameters if field.kind == "texture_2d"]
    from .shader_ray_surface import optional_ray_surface, parameter_slot_declarations
    adapters, manifests = [], []
    for guid, path in graph.ray_surfaces:
        source = graph.reader(path)
        if source is None:
            raise FileNotFoundError(path)
        module_name = "inx_ray_surface_" + guid.replace("-", "_")
        adapter, diagnostic = optional_ray_surface(source, path, module_name, graph.reader,
                                                   schema_binding=len(adapters) + 1)
        if adapter is None:
            continue
        adapters.append(adapter)
        manifests.append(dict(adapter.manifest, guid=guid, entry_index=len(adapters) + 1))
        existing = {RESOURCE_MODULE[0], *(module.name for module in graph.ordered)}
        for name, library_path, library_source in adapter.libraries:
            if name not in existing:
                graph.ordered.append(Module(name, library_path, library_source, {}, ()))
                existing.add(name)
        graph.ordered.append(Module(adapter.module_name, adapter.source_path, adapter.generated_source, {},
                                    adapter.source_ranges))
    dispatch = ["InxSurfaceData inx_ray_author_surface(InxMeshInput input, InxRayLitSampling sampling)", "{",
                "    uint4 header = inx_ray_material_header(input.surface_id);", "    switch (header.z)", "    {",
                "    case 0u: return inx_ray_author_lit(input, inx_ray_builtin_material(input.surface_id), sampling);",
                "    case 1u: return inx_ray_author_unlit(input, inx_ray_builtin_material(input.surface_id), sampling);"]
    for index, adapter in enumerate(adapters, 2):
        dispatch.append("    case " + str(index) + "u: return " + adapter.manifest["entry"] +
                        "(input, " + adapter.module_name + "_load(ray_lit_materials, header.y));")
    dispatch += ["    }", "    return (InxSurfaceData)0;", "}"]
    code = ["import " + RESOURCE_MODULE[0] + ";", "import inx_geometry;", "import inx_ray_traversal;",
            "import inx_ray_bsdf;", "import inx_ray_tail_cache;", "import inx_ray_texture_pages;", "import inx_camera;", "import inx_lighting;", "import inx_frame;",
            *("import " + name + ";" for name in dict.fromkeys(dependencies)),
            *("import " + adapter.module_name + ";" for adapter in adapters),
            *parameter_slot_declarations(values, "InxRayBuiltin"),
            "struct ShaderParameters {", *lit.parameter_field_declarations(values), "};",
            "struct InxRayBuiltinValues {",
            *("    InxRayBuiltin_Slot_" + field.kind + " " + field.name + ";" for field in values), "};",
            "struct InxRayBuiltinPayload { InxRayBuiltinValues values; uint participation;",
            *("    uint " + field.name + ";" for field in textures), "};",
            "struct InxRayLitMaterial {", "    ShaderParameters values;", "    uint participation;",
            *("    uint " + field.name + ";" for field in textures), "};",
            "[[vk::binding(0, 3)]] StructuredBuffer<InxRayBuiltinPayload> inx_ray_schema_builtin;",
            "ByteAddressBuffer ray_lit_materials;",
            "uint4 inx_ray_material_header(uint id) { return ray_lit_materials.Load<uint4>(id * 16u); }",
            "uint inx_ray_participation(uint id) { return inx_ray_material_header(id).x; }",
            "float inx_ray_alpha_clip(uint id) { return asfloat(inx_ray_material_header(id).w); }",
            "InxRayLitMaterial inx_ray_builtin_material(uint id) {",
            "    let payload = ray_lit_materials.Load<InxRayBuiltinPayload>(inx_ray_material_header(id).y);",
            "    InxRayLitMaterial result = (InxRayLitMaterial)0;",
            *("    result.values." + field.name + " = payload.values." + field.name + ".value;" for field in values),
            *("    result." + field.name + " = payload." + field.name + ";" for field in textures),
            "    return result;", "}",
            _SURFACE_SAMPLING, _RAY_LIT_SAMPLING, *entries, *dispatch, _SURFACE_COVERAGE, _RAY_LIT_EVALUATION]
    return ShaderDefinition("Lit", lit.path, _RAY_LIT_MODULE, parameters, "\n".join(code), graph.sources(),
                            source_ranges=(*graph.ranges(), *ranges), domain="mesh", surface=True, info=fragments["lit"].info,
                            ray_surfaces=tuple(manifests))


def _world_ray_module(graph, owner, location):
    if not graph.allow_world_rays:
        owner.error(location, "World ray queries require a graphics or graph compute program", code="author.stage")
    if graph.ray_lit is None:
        # Materials keep the graphics author contract. Their dependencies share
        # this graph's module identities; only the calling numerical library is
        # constrained to compute operations.
        numerical_context = graph.compute_autodiff
        graph.compute_autodiff = None
        try:
            lit = _prepare_ray_lit(_graph=graph)
        finally:
            graph.compute_autodiff = numerical_context
        library_paths = {module.path for module in graph.ordered}
        ranges = tuple(span for span in lit.source_ranges if span[0] not in library_paths)
        graph.ray_lit = Module(lit.module_name, lit.source_path, lit.generated_source, {}, ranges)
        graph.ray_lit.manifests = lit.ray_surfaces
        graph.ordered.append(graph.ray_lit)
    return graph.ray_lit


def _prepare_source(source, source_path, module_name, reader, *, metadata_only=False, vertex_source=None, ray_surfaces=()):
    """Return a plain native preparation snapshot; no Python object enters GPU work."""
    from infernux._source_kinds import source_extension

    source = source.removeprefix("\ufeff")
    if metadata_only:
        return "", (), json.dumps(read_shader_info(source, source_path), ensure_ascii=False), ()
    if source_extension(source_path) == ".vert.py":
        graph = LibraryGraph(reader)
        owner = ShaderEmitter(source, source_path)
        vertex = read_vertex(source_path, graph, owner, ast.Constant(value=None, lineno=1, col_offset=0), source=source)
        parameters = tuple(vertex.author.fields.values())
        code, ranges = vertex.emit()
        prefix = ["import " + RESOURCE_MODULE[0] + ";", "import inx_geometry;",
                  *engine_imports(vertex.methods, vertex.author.alias, graph=graph, owner=vertex.author),
                  *("import " + name + ";" for name in vertex.dependencies),
                  *vertex.author.parameter_declarations(parameters)]
        schema = {"name": vertex.name, "domain": "vertex", "info": vertex.info,
                  "parameters": [asdict(field) for field in parameters], "inputs": []}
        return "\n".join([*prefix, code]), graph.sources(), json.dumps(schema, ensure_ascii=False), (*graph.ranges(), *vertex.author.source_ranges, *ranges)
    if source_extension(source_path) == ".shadingmodel.py":
        graph = LibraryGraph(reader)
        owner = ShaderEmitter(source, source_path)
        model = read_shading_model(source_path, graph, owner, ast.Constant(value=None, lineno=1, col_offset=0), source=source)
        schema = {"name": model.name, "domain": "shadingmodel", "lighting_contract": model.contract,
                  "info": model.info,
                  "parameters": [], "inputs": []}
        return model.source, graph.sources(), json.dumps(schema, ensure_ascii=False), (*graph.ranges(), *model.source_ranges)
    if source_extension(source_path) == ".shaderlib.py":
        graph = LibraryGraph(reader, allow_world_rays=True)
        owner = ShaderEmitter(source, source_path)
        module = graph.load(Path(source_path), owner, ast.Constant(value=None, lineno=1, col_offset=0), source=source)
        schema = {"name": Path(source_path).name[:-len(".shaderlib.py")], "domain": "library",
                  "info": read_shader_info(source, source_path),
                  "parameters": [], "inputs": [],
                  "functions": [{"name": name, "return_type": function.result, "arguments": function.arguments,
                                 "requires": function.effects, "signature": function.source_signature,
                                 "documentation": function.documentation}
                                for name, function in module.functions.items()]}
        libraries = tuple(item for item in graph.sources() if item[0] != module.name)
        return module.source, libraries, json.dumps(schema, ensure_ascii=False), graph.ranges()
    vertex_path = None
    if vertex_source is not None:
        vertex_path, vertex_text = vertex_source
        source_reader = reader
        def reader(path):
            if Path(path) == Path(vertex_path):
                return vertex_text
            return source_reader(path) if source_reader is not None else Path(path).read_text(encoding="utf-8-sig")
    definition = parse_shader(source.removeprefix("\ufeff"), source_path, module_name, reader=reader, vertex_path=vertex_path, ray_surfaces=ray_surfaces)
    schema = {"name": definition.name, "domain": definition.domain, "surface": definition.surface,
              "thin_sheet": definition.thin_sheet,
              "color_outputs": definition.color_outputs,
              "info": definition.info,
              "shading_model": definition.shading_model, "deferred_model": definition.deferred_model,
              "vertex_source": definition.vertex_source,
              "parameters": [asdict(field) for field in definition.parameters],
              "inputs": [asdict(field) for field in definition.inputs],
              "ray_surfaces": definition.ray_surfaces,
              "ray_surface_catalog": None if definition.ray_surface_catalog is None else
                                     [guid for guid, _ in definition.ray_surface_catalog]}
    if definition.surface and definition.info["shader_id"] not in {"infernux/lit", "infernux/unlit"}:
        from .shader_ray_surface import optional_ray_surface
        adapter, diagnostic = optional_ray_surface(source, source_path, module_name + "_secondary", reader)
        schema["ray_surface"] = adapter.manifest if adapter else None
        schema["ray_surface_diagnostic"] = diagnostic
    return definition.generated_source, definition.libraries, json.dumps(schema, ensure_ascii=False), definition.source_ranges
