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
from .shader_parameters import ParameterAuthor, ShaderInput, ShaderParameter
from .vertex import read_vertex


_RAY_LIT_MODULE = "inx_ray_lit"
_WORLD_RAY_INPUTS = tuple(ShaderInput(name, name, kind) for name, kind in (
    ("world_ray_geometry", "WorldBuffer"),
    ("world_ray_meshes", "WorldBuffer"),
    ("ray_lit_materials", "WorldBuffer"),
    ("ray_texture_descriptors", "WorldBuffer"),
    ("ray_color_pages", "Color2DArray"),
    ("ray_data_pages", "Color2DArray"),
))


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
    ray_lit = False

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
        if self.ray_lit and isinstance(node.value, ast.Name) and node.value.id == "self":
            field = self.fields.get(node.attr)
            if isinstance(field, ShaderParameter):
                if field.kind == "texture_2d":
                    return f"inx_ray_material.{field.name}"
                value = f"inx_ray_material.values.{field.name}"
                return f"({value} != 0u)" if field.kind == "bool" else value
        return super().visit_Attribute(node)

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
            signature = kind + " " + uv.arg + (" : TEXCOORD0" if kind == "float2" else "")
            self.types[uv.arg] = kind
        result = self.annotation(main.returns) if main.returns is not None else None
        expected = {"InxSurfaceData"} if self.is_surface else {"float4", "InxFragmentOutput"}
        if result not in expected:
            self.error(main, "Surface returns shader.SurfaceData" if self.is_surface else
                       "Fragment main returns shader.vec4 or shader.FragmentOutput")
        self.aliases = (self.alias,)
        if self.domain == "particle":
            self.lighting_mode = "tiled_lighting"
        self.declare_locals(main.body, parameters=(field.arg for field in arguments[1:]))
        self.local_declarations()
        self.block(main.body)
        semantic = " : SV_Target0" if result == "float4" else ""
        if self.domain in {"mesh", "particle"} and not self.is_surface:
            argument = arguments[1].arg if len(arguments) == 2 else ""
            alpha = "color.color.a" if result == "InxFragmentOutput" else "color.a"
            entry = "inx_evaluate_particle" if self.domain == "particle" else "fragment_main"
            marker = [] if self.domain == "particle" else ['[shader("fragment")]']
            threshold = "max(inx_parameters._alpha_clip_threshold, 0.0001)" if self.domain == "particle" else "inx_parameters._alpha_clip_threshold"
            evaluation_signature = signature or ("InxParticleInput inx_input" if self.domain == "particle" else "")
            coverage = ["    if (!(" + (argument or "inx_input") + ".alpha > 0.0)) discard;"] if self.domain == "particle" else []
            inline = ["[ForceInline]"] if self.domain == "particle" else []
            specialization = "<let tiled_lighting : bool>" if self.domain == "particle" else ""
            arguments_specialization = "<tiled_lighting>" if self.domain == "particle" else ""
            return [self.source_directive(main), *inline, result + " inx_author_fragment" + specialization + "(" + signature + ")",
                    "{", *self.lines, "}", *marker, *inline,
                    result + " " + entry + specialization + "(" + evaluation_signature + ")" + (semantic if marker else ""), "{", *coverage,
                    "    " + result + " color = inx_author_fragment" + arguments_specialization + "(" + argument + ");",
                    "    clip(" + alpha + " - " + threshold + ");", "    return color;", "}"]
        if self.domain in {"screen_ui", "world_ui"}:
            return [self.source_directive(main), result + " inx_author_ui(" + signature + ")", "{", *self.lines, "}"]
        if self.ray_lit:
            declaration = (f"InxSurfaceData inx_author_surface({signature}, InxRayLitMaterial inx_ray_material, "
                           "InxRayLitSampling inx_surface_sampling)")
        elif self.is_surface:
            sampling = "<Sampling : InxSurfaceSampling>" if self.uses_surface_sampling else ""
            sampling_argument = ", Sampling inx_surface_sampling" if self.uses_surface_sampling else ""
            declaration = f"InxSurfaceData inx_author_surface{sampling}({signature}{sampling_argument})"
        else:
            declaration = result + " fragment_main(" + signature + ")" + semantic
        return ([self.source_directive(main)] if self.is_surface else ['[shader("fragment")]', self.source_directive(main)]) + [
            declaration, "{", *self.lines, "}"]



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
    return result;
}
[shader("vertex")]
InxMeshInput vertex_motion(InxVertexInput input, uint instance : SV_VulkanInstanceID)
{
    InxMeshInput result = inx_mesh_vertex(input, instance);
    InxInstanceData history = inx_instances[instance];
    if ((history.object_flags & 1u) != 0u)
    {
        InxSkinState skin = inx_skins[instance];
        skin.bone_offset = skin.previous_bone_offset;
        float3 previous_local = inx_skin_vertex(input, skin, inx_bones).position;
        float4 previous_clip = mul(inx_view.previous_view_projection, mul(inx_previous_model(history), float4(previous_local, 1.0)));
        result.motion_vector = (result.position.xy / max(abs(result.position.w), 1e-6)
            - float2(previous_clip.x, -previous_clip.y) / max(abs(previous_clip.w), 1e-6)) * 0.5;
    }
    inx_clip_device_bounds(result, instance);
    return result;
}
[shader("fragment")]
float4 fragment_main(InxMeshInput input, bool front_facing : SV_IsFrontFace) : SV_Target0
{
    return inx_shade_surface<false>(inx_evaluate_surface(input, front_facing), input);
}
[shader("fragment")]
float4 fragment_forward_plus(InxMeshInput input, bool front_facing : SV_IsFrontFace) : SV_Target0
{
    return inx_shade_surface<true>(inx_evaluate_surface(input, front_facing), input);
}
[shader("fragment")]
void fragment_depth(InxMeshInput input, bool front_facing : SV_IsFrontFace)
{
    inx_evaluate_surface(input, front_facing);
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
    return input.motion_vector;
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


_RAY_LIT_SAMPLING = """
struct InxRayTextureDescriptor
{
    uint page_class;
    uint layer;
};
StructuredBuffer<InxRayTextureDescriptor> ray_texture_descriptors;
Texture2DArray<float4> ray_color_pages;
Texture2DArray<float4> ray_data_pages;
SamplerState ray_color_sampler;
SamplerState ray_data_sampler;

struct InxRayLitSampling
{
    InxSecondarySurfaceSampling footprint;

    float4 sample(uint descriptor_index, float2 uv)
    {
        InxRayTextureDescriptor descriptor = ray_texture_descriptors[descriptor_index];
        float3 coordinates = float3(uv, float(descriptor.layer));
        switch (descriptor.page_class)
        {
        case 0u:
            return ray_color_pages.SampleLevel(ray_color_sampler, coordinates, footprint.lod);
        case 1u:
            return ray_data_pages.SampleLevel(ray_data_sampler, coordinates, footprint.lod);
        case 2u:
            return float4(1.0);
        case 3u:
            return float4(0.0, 0.0, 0.0, 1.0);
        case 4u:
            return float4(128.0 / 255.0, 128.0 / 255.0, 1.0, 1.0);
        default:
            // Keep the invalid sample a runtime expression in WGSL.
            return float4(asfloat(0x7fc00000u | descriptor.page_class));
        }
    }
    InxSurfaceDifferentials differentials(InxMeshInput input, float2 uv, float2 uv_scale)
    {
        return footprint.differentials(input, uv, uv_scale);
    }
};
"""


_RAY_LIT_EVALUATION = """
bool inx_evaluate_ray_lit(InxMeshInput input, bool front_facing, InxSecondarySurfaceSampling footprint,
                          out InxSurfaceData surface)
{
    InxRayLitMaterial material = ray_lit_materials[input.surface_id];
    InxRayLitSampling sampling = {footprint};
    surface = inx_prepare_surface(inx_author_surface(input, material, sampling), input, front_facing);
    return inx_surface_covered(surface, material.values._alpha_clip_threshold);
}

bool inx_evaluate_ray_lit_hit(RayPage page, TraceRay ray, RayHit hit, float lod, out InxSurfaceData surface)
{
    RaySurfaceHit attributes = ray_surface_at_hit(page, ray, hit);
    InxMeshInput input = (InxMeshInput)0;
    input.surface_id = hit.surface_id;
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
        if (!(ray_lit_materials[candidate.surface_id].values._alpha_clip_threshold > 0.0))
            return true;
        InxSurfaceData surface;
        return inx_evaluate_ray_lit_hit(page, ray, candidate, lod, surface);
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
        result.found = true;
        result.distance = hit.distance;
        result.position_ws = ray.origin + hit.distance * ray.direction;
        result.surface_id = hit.surface_id;
        inx_evaluate_ray_lit_hit(page, ray, hit, lod, result.surface);
    }
    return result;
}

public bool inx_world_ray_visibility(float3 origin_ws, float3 direction_ws, float min_distance,
                                     float max_distance, uint layer_mask, float lod)
{
    TraceRay ray = {origin_ws, min_distance, direction_ws, max_distance};
    return inx_trace_lit_visibility(ray_page(), ray, layer_mask, lod);
}
"""


_SURFACE_LIGHTING = """
float4 inx_shade_surface<let tiled_lighting : bool>(InxSurfaceData surface, InxMeshInput input)
{
    InxShadingContext context = inx_shading_context(input);
    context.normal_variance = inx_normal_variance(surface.normal_ws);
    context.receives_shadows = context.receives_shadows && INX_RECEIVE_SHADOWS;
    return float4(inx_shading_model::shade<tiled_lighting>(surface, context), surface.alpha);
}
"""

_SURFACE_GBUFFER = """
[shader("fragment")]
InxGBufferOutput fragment_gbuffer(InxMeshInput input, bool front_facing : SV_IsFrontFace)
{
    return inx_pack_gbuffer(inx_evaluate_surface(input, front_facing), input.surface_id);
}

[shader("fragment")]
uint2 fragment_ray_surface(InxMeshInput input, bool front_facing : SV_IsFrontFace) : SV_Target0
{
    // Form the plane before coverage can discard a lane in this derivative quad.
    float3 normal = cross(ddx(input.position_ws), ddy(input.position_ws));
    inx_evaluate_surface(input, front_facing);
    float length_squared = dot(normal, normal);
    if (!(length_squared > 0.0))
        return uint2(0u);
    // At fixed screen XY, increasing reversed clip depth moves toward the camera.
    float4 depth_direction = mul(inx_view.inverse_view_projection, float4(0.0, 0.0, 1.0, 0.0));
    float3 toward_eye = depth_direction.xyz - input.position_ws * depth_direction.w;
    normal *= (dot(normal, toward_eye) < 0.0 ? -1.0 : 1.0) * rsqrt(length_squared);
    float2 encoded = inx_encode_surface_normal(normal);
    return uint2(input.surface_id, f32tof16(encoded.x) | (f32tof16(encoded.y) << 16u));
}
"""


def parse_shader(source: str, source_path: str, module_name: str, *, reader=None, vertex_path=None,
                 _graph=None) -> ShaderDefinition:
    info = read_shader_info(source, source_path)
    author = _Author(source, source_path)
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
    graph = _graph if _graph is not None else LibraryGraph(reader, allow_world_rays=author.domain == "fullscreen")
    author.functions, dependencies = graph.imports(imports, author)
    author.lighting_mode = "true" if author.domain == "fullscreen" else "false"
    author.is_surface = author.qualified(declaration.bases[0]) == author.alias + ".surface"
    author.ray_lit = module_name == _RAY_LIT_MODULE
    entry_name = "surface" if author.is_surface else "main"
    main, helpers, references = None, [], {}
    for statement in declaration.body:
        if info_field(statement) in INFO_DEFAULTS:
            continue
        if (author.domain == "mesh" and isinstance(statement, ast.Assign) and len(statement.targets) == 1
                and author.qualified(statement.targets[0]) == "shading_model"):
            name = author.qualified(statement.targets[0])
            if name in references or not author.is_surface:
                author.error(statement, "Declare one shading model on a Surface", code="author.reference")
            references[name] = (graph.reference(statement.value, author, stage="shadingmodel", suffix=".shadingmodel.py"), statement)
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
    if uses_surface_lighting(body, author.alias, author.functions):
        for helper in helpers:
            name = "self." + helper.name
            author.functions[name] = replace(author.functions[name], lighting_specialization=True)
    if author.domain in {"mesh", "particle"}:
        author.fields["_alpha_clip_threshold"] = ShaderParameter("_alpha_clip_threshold", "float", 0.0, "", internal=True)
        author.field_nodes["_alpha_clip_threshold"] = declaration
        author.field_sources["_alpha_clip_threshold"] = (author, declaration)
    if author.ray_lit:
        if not author.is_surface or helpers or references or vertex_path is not None:
            author.error(declaration, "The ray Lit entry requires a Surface body and shared ShaderLib helpers",
                         code="author.stage")
        parameters = tuple(field for field in author.fields.values() if isinstance(field, ShaderParameter))
        values = [field for field in parameters if field.kind != "texture_2d"]
        textures = [field for field in parameters if field.kind == "texture_2d"]
        entry = author.function(main)
        if not author.uses_surface_sampling:
            author.error(main, "The ray Lit entry requires explicit material sampling", code="author.stage")
        code = ["import " + RESOURCE_MODULE[0] + ";", "import inx_geometry;", "import inx_ray_traversal;",
                *("import " + name + ";" for name in dependencies),
                "struct ShaderParameters {", *author.parameter_field_declarations(values), "};",
                "struct InxRayLitMaterial {", "    ShaderParameters values;",
                *("    uint " + field.name + ";" for field in textures), "};",
                "StructuredBuffer<InxRayLitMaterial> ray_lit_materials;",
                _SURFACE_SAMPLING, _RAY_LIT_SAMPLING, *entry, _SURFACE_COVERAGE, _RAY_LIT_EVALUATION]
        return ShaderDefinition(declaration.name, source_path, module_name, parameters, "\n".join(code), graph.sources(),
                                source_ranges=(*graph.ranges(), *author.source_ranges), domain=author.domain,
                                surface=author.is_surface, info=info)
    model = None
    if author.is_surface:
        model_path, location = references.get("shading_model", (default_shading_model(), declaration))
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
    dependencies = tuple(dict.fromkeys((*dependencies, *(vertex.dependencies if vertex else ()))))
    code = ["import " + RESOURCE_MODULE[0] + ";", "import inx_geometry;", *("import " + name + ";" for name in dependencies)]
    code.extend(engine_imports(body, author.alias, graph=graph, owner=author))
    if graph.ray_lit is not None:
        for field in _WORLD_RAY_INPUTS:
            if field.name in author.fields:
                author.error(author.field_nodes[field.name], "World ray input name is reserved: " + field.name,
                             code="author.parameter")
    if vertex:
        code.append("import inx_frame;")
    if author.domain == "mesh":
        code.append("import inx_camera;")
    elif author.domain == "particle":
        code.append("import inx_particle;")
    code = list(dict.fromkeys(code))
    if author.is_surface:
        code.insert(0, "import inx_surface_model;")
    for field in inputs:
        code.extend([author.source_directive(author.field_nodes[field.name]),
                     ("ByteAddressBuffer" if field.kind == "Buffer" else RESOURCE_BINDINGS[field.kind]) + " inx_" + field.name + ";"])
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
    for helper in helpers:
        emitter = _Author(source, source_path)
        emitter.alias = author.alias
        emitter.aliases = (author.alias,)
        emitter.fields = author.fields
        emitter.domain = author.domain
        emitter.types = author.types.copy()
        emitter.functions = author.functions
        code.extend(emitter.emit_function(helper, author.functions["self." + helper.name], method=True))
        author.source_ranges.extend(emitter.source_ranges)
    entry = author.function(main)
    if author.uses_surface_sampling:
        code.append(_SURFACE_SAMPLING)
    code.extend(entry)
    if author.domain in {"screen_ui", "world_ui"}:
        from infernux.lib import _Infernux
        code.append(_Infernux._shader_ui_author_source(
            author.domain == "world_ui", author.annotation(main.returns) == "InxFragmentOutput"))
    if author.domain == "particle":
        code.append(_particle_passes(author.annotation(main.returns)))
    if author.is_surface:
        code.append("static const bool INX_RECEIVE_SHADOWS = " + ("true" if info["receive_shadows"] else "false") + ";")
        code.append(_surface_evaluation(author.uses_surface_sampling))
        code.append(_SURFACE_LIGHTING)
        passes = _SURFACE_PASSES
        if vertex:
            passes = passes.replace("float3 previous_local = inx_skin_vertex(input, skin, inx_bones).position;",
                                    "InxVertexInput previous = inx_apply_vertex(input, instance, inx_globals.frame_params.w, inx_previous_model(history));\n"
                                    "        float3 previous_local = inx_skin_vertex(previous, skin, inx_bones).position;")
        code.append(passes)
        if model.deferred_model:
            code.append(_SURFACE_GBUFFER)
    libraries = graph.sources()
    if model:
        libraries = (*libraries, ("inx_surface_model", model.path, model.source))
    return ShaderDefinition(declaration.name, source_path, module_name, parameters,
                            "\n".join(code), libraries, (*inputs, *_WORLD_RAY_INPUTS) if graph.ray_lit else inputs,
                            tuple(dict.fromkeys((*graph.ranges(), *(model.source_ranges if model else ()), *vertex_ranges,
                                                 *(vertex.author.source_ranges if vertex else ()), *author.source_ranges))),
                            author.domain, author.is_surface, model.name if model else None, model.deferred_model if model else None,
                            info, vertex.path if vertex else None)


def read_shader(path: Path, module_name: str) -> ShaderDefinition:
    return parse_shader(path.read_text(encoding="utf-8-sig"), str(path), module_name)


def _prepare_ray_lit(reader=None, *, _graph=None):
    """Prepare the built-in Lit module for full-layer pages with one sampler per page class."""
    path = default_shading_model().with_name("lit.frag.py")
    graph = _graph if _graph is not None else LibraryGraph(reader)
    source = graph.reader(str(path))
    if source is None:
        raise FileNotFoundError(path)
    return parse_shader(source.removeprefix("\ufeff"), str(path), _RAY_LIT_MODULE, _graph=graph)


def _world_ray_module(graph, owner, location):
    if not graph.allow_world_rays:
        owner.error(location, "World ray queries require a fullscreen shader", code="author.stage")
    if graph.ray_lit is None:
        lit = _prepare_ray_lit(_graph=graph)
        ranges = tuple(span for span in lit.source_ranges if span[0] == lit.source_path)
        graph.ray_lit = Module(lit.module_name, lit.source_path, lit.generated_source, {}, ranges)
        graph.ordered.append(graph.ray_lit)
    return graph.ray_lit


def _prepare_source(source, source_path, module_name, reader, *, metadata_only=False, vertex_source=None):
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
    definition = parse_shader(source.removeprefix("\ufeff"), source_path, module_name, reader=reader, vertex_path=vertex_path)
    schema = {"name": definition.name, "domain": definition.domain, "surface": definition.surface,
              "info": definition.info,
              "shading_model": definition.shading_model, "deferred_model": definition.deferred_model,
              "vertex_source": definition.vertex_source,
              "parameters": [asdict(field) for field in definition.parameters],
              "inputs": [asdict(field) for field in definition.inputs]}
    return definition.generated_source, definition.libraries, json.dumps(schema, ensure_ascii=False), definition.source_ranges
