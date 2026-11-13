"""Static GPU author fields, typed inputs and functions lowered to native Slang."""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path

from .shader_library import LibraryGraph, ShaderEmitter, declarations, frame_imports
from .shader_types import RESOURCE_BINDINGS, RESOURCE_MODULE, RESOURCE_TYPES
from .shading_model import default_shading_model, read_shading_model
from .shader_info import INFO_DEFAULTS, info_field, read_shader_info
from .shader_parameters import ParameterAuthor, ShaderInput, ShaderParameter
from .vertex import read_vertex


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
            input_kinds=tuple((field.name, field.kind) for field in self.inputs),
        )


class _Author(ParameterAuthor):
    def function(self, main):
        if main.decorator_list or main.args.vararg or main.args.kwarg or main.args.kwonlyargs or main.args.defaults:
            self.error(main, "Shader entry parameters have a fixed typed signature")
        arguments = main.args.args
        if self.is_surface and len(arguments) != 2:
            self.error(main, "Surface accepts self and a typed surface context")
        if not arguments or arguments[0].arg != "self":
            self.error(main, "Shader main is an instance method")
        if len(arguments) > 2:
            self.error(main, "Shader main accepts self and one typed stage input")
        signature = ""
        if len(arguments) == 2:
            uv = arguments[1]
            kind = self.annotation(uv.annotation) if uv.annotation is not None else None
            expected = {"InxMeshInput"} if self.domain == "mesh" else {"float2", "InxFragmentInput"}
            if kind not in expected:
                self.error(uv, "Mesh input is shader.mesh_input" if self.domain == "mesh" else
                           "Fullscreen input is shader.vec2 or shader.fragment_input")
            signature = kind + " " + uv.arg + (" : TEXCOORD0" if kind == "float2" else "")
            self.types[uv.arg] = kind
        result = "InxSurfaceData" if self.is_surface else "float4"
        if main.returns is None or self.annotation(main.returns) != result:
            self.error(main, "Surface returns shader.SurfaceData" if self.is_surface else "Fragment main returns shader.vec4")
        self.aliases = (self.alias,)
        self.declare_locals(main.body)
        self.local_declarations()
        self.block(main.body)
        if self.domain == "mesh" and not self.is_surface:
            argument = arguments[1].arg if len(arguments) == 2 else ""
            return [self.source_directive(main), "float4 inx_author_fragment(" + signature + ")",
                    "{", *self.lines, "}", '[shader("fragment")]',
                    "float4 fragment_main(" + signature + ") : SV_Target0", "{",
                    "    float4 color = inx_author_fragment(" + argument + ");",
                    "    clip(color.a - inx_parameters._alpha_clip_threshold);", "    return color;", "}"]
        declaration = ("InxSurfaceData inx_author_surface(" + signature + ")" if self.is_surface else
                       "float4 fragment_main(" + signature + ") : SV_Target0")
        return ([self.source_directive(main)] if self.is_surface else ['[shader("fragment")]', self.source_directive(main)]) + [
            declaration, "{", *self.lines, "}"]


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
[[vk::binding(5, 1)]] ConstantBuffer<InxCameraData> inx_view;
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
    context.projection_matrix = inx_view.projection;
    context.projection_matrix[1] = -context.projection_matrix[1];
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
InxSurfaceData inx_evaluate_surface(InxMeshInput input, bool front_facing)
{
    InxSurfaceData surface = inx_prepare_surface(inx_author_surface(input), input, front_facing);
    if (inx_parameters._alpha_clip_threshold > 0.0 && surface.alpha < inx_parameters._alpha_clip_threshold) discard;
    return surface;
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


_SURFACE_LIGHTING = """
float4 inx_shade_surface<let tiled_lighting : bool>(InxSurfaceData surface, InxMeshInput input)
{
    InxShadingContext context = inx_shading_context(input);
    context.receives_shadows = context.receives_shadows && INX_RECEIVE_SHADOWS;
    return float4(inx_shading_model::shade<tiled_lighting>(surface, context), surface.alpha);
}
"""

_SURFACE_GBUFFER = """
[shader("fragment")]
InxGBufferOutput fragment_gbuffer(InxMeshInput input, bool front_facing : SV_IsFrontFace)
{
    return inx_pack_gbuffer(inx_evaluate_surface(input, front_facing), input, INX_MODEL_ID, INX_RECEIVE_SHADOWS);
}
"""


def parse_shader(source: str, source_path: str, module_name: str, *, reader=None, vertex_path=None) -> ShaderDefinition:
    info = read_shader_info(source, source_path)
    author = _Author(source, source_path)
    author.alias, imports, body = declarations(source, source_path)
    author.aliases = (author.alias,)
    graph = LibraryGraph(reader)
    author.functions, dependencies = graph.imports(imports, author)
    if len(body) != 1 or not isinstance(body[0], ast.ClassDef):
        author.error(ast.Constant(lineno=1, col_offset=0), "Declare one shader class")
    declaration = body[0]
    bases = {author.alias + ".fullscreen": "fullscreen", author.alias + ".frag": "mesh", author.alias + ".surface": "mesh"}
    if declaration.decorator_list or len(declaration.bases) != 1 or author.qualified(declaration.bases[0]) not in bases:
        author.error(declaration, "Declare shader.fullscreen, shader.frag or shader.surface")
    author.domain = bases[author.qualified(declaration.bases[0])]
    author.is_surface = author.qualified(declaration.bases[0]) == author.alias + ".surface"
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
    if author.domain == "mesh":
        author.fields["_alpha_clip_threshold"] = ShaderParameter("_alpha_clip_threshold", "float", 0.0, "", internal=True)
        author.field_nodes["_alpha_clip_threshold"] = declaration
        author.field_sources["_alpha_clip_threshold"] = (author, declaration)
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
    if author.domain == "mesh":
        for field in parameters:
            if field.kind in {"bool", "uint", "mat3"}:
                author.error(author.field_nodes[field.name], "This material value type is not supported by the material owner yet",
                             code="author.parameter")
        if inputs:
            author.error(declaration, "Mesh image resources are declared as material properties")
    dependencies = tuple(dict.fromkeys((*dependencies, *(vertex.dependencies if vertex else ()))))
    code = ["import " + RESOURCE_MODULE[0] + ";", "import inx_geometry;", *("import " + name + ";" for name in dependencies)]
    code.extend(("import inx_frame;",) if vertex else frame_imports(body, author.alias))
    if author.is_surface:
        code.insert(0, "import inx_surface_model;")
    for field in inputs:
        code.extend([author.source_directive(author.field_nodes[field.name]),
                     RESOURCE_BINDINGS[field.kind] + " inx_" + field.name + ";"])
    code.extend(author.parameter_declarations(parameters))
    vertex_ranges = ()
    if vertex:
        vertex_code, vertex_ranges = vertex.emit()
        code.extend([vertex_code, _VERTEX_DEFORM])
        code.append(_mesh_vertex(deform=True, project="self.project" in vertex.author.functions))
    else:
        code.append(_mesh_vertex() if author.domain == "mesh" else _FULLSCREEN_VERTEX)
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
    code.extend(author.function(main))
    if author.is_surface:
        code.append("static const bool INX_RECEIVE_SHADOWS = " + ("true" if info["receive_shadows"] else "false") + ";")
        code.append(_SURFACE_LIGHTING)
        passes = _SURFACE_PASSES
        if vertex:
            passes = passes.replace("float3 previous_local = inx_skin_vertex(input, skin, inx_bones).position;",
                                    "InxVertexInput previous = inx_apply_vertex(input, instance, inx_globals.frame_params.w, inx_previous_model(history));\n"
                                    "        float3 previous_local = inx_skin_vertex(previous, skin, inx_bones).position;")
        code.append(passes)
        if model.deferred_model:
            code.append(_SURFACE_GBUFFER.replace("INX_MODEL_ID", "INX_" + model.deferred_model.upper() + "_MODEL_ID"))
    libraries = graph.sources()
    if model:
        libraries = (*libraries, ("inx_surface_model", model.path, model.source))
    return ShaderDefinition(declaration.name, source_path, module_name, parameters,
                            "\n".join(code), libraries, inputs,
                            tuple(dict.fromkeys((*graph.ranges(), *(model.source_ranges if model else ()), *vertex_ranges,
                                                 *(vertex.author.source_ranges if vertex else ()), *author.source_ranges))),
                            author.domain, author.is_surface, model.name if model else None, model.deferred_model if model else None,
                            info, vertex.path if vertex else None)


def read_shader(path: Path, module_name: str) -> ShaderDefinition:
    return parse_shader(path.read_text(encoding="utf-8-sig"), str(path), module_name)


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
                  *frame_imports(vertex.methods, vertex.author.alias),
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
        graph = LibraryGraph(reader)
        owner = ShaderEmitter(source, source_path)
        module = graph.load(Path(source_path), owner, ast.Constant(lineno=1, col_offset=0), source=source)
        schema = {"name": Path(source_path).name[:-len(".shaderlib.py")], "domain": "library",
                  "info": read_shader_info(source, source_path),
                  "parameters": [], "inputs": [],
                  "functions": [{"name": name, "return_type": function.result, "arguments": function.arguments}
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
