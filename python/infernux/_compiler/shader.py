"""Static GPU author fields, typed inputs and functions lowered to native Slang."""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
import json
from pathlib import Path

from .shader_library import LibraryGraph, ShaderEmitter, declarations
from .shader_types import RESOURCE_MODULE, RESOURCE_TYPES


@dataclass(frozen=True)
class ShaderParameter:
    name: str
    kind: str
    default: object
    label: str
    range: tuple[float, float] | None = None
    hdr: bool = False
    internal: bool = False


@dataclass(frozen=True)
class ShaderInput:
    name: str
    semantic: str
    kind: str
    domain: str | None = None


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

    def compile(self, targets=("spirv", "wgsl")):
        from infernux.lib import _Infernux
        return _Infernux._compile_shader_program(
            self.generated_source, self.source_path, self.module_name, self.libraries, targets,
            source_ranges=self.source_ranges,
        )


_PARAMETER_TYPES = {"float": "float", "int": "int", "uint": "uint", "bool": "uint", "color": "float4",
                    "vec2": "float2", "vec3": "float3", "vec4": "float4", "mat3": "float3x3", "mat4": "float4x4"}
_DEFAULTS = {"float": 0.0, "int": 0, "uint": 0, "bool": False, "color": (1.0, 1.0, 1.0, 1.0),
             "texture_2d": "white",
             "vec2": (0.0, 0.0), "vec3": (0.0, 0.0, 0.0), "vec4": (0.0, 0.0, 0.0, 0.0),
             "mat3": tuple(tuple(float(row == column) for column in range(3)) for row in range(3)),
             "mat4": tuple(tuple(float(row == column) for column in range(4)) for row in range(4))}


class _Author(ShaderEmitter):
    def __init__(self, source: str, path: str):
        super().__init__(source, path)
        self.source = source
        self.alias = ""
        self.fields = {}
        self.field_nodes = {}
        self.domain = "fullscreen"
        self.is_surface = False

    def qualified(self, node):
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return self.qualified(node.value) + "." + node.attr
        self.error(node, "Expected a static shader declaration name")

    def literal(self, node):
        if isinstance(node, ast.Call) and self.qualified(node.func) in {
            self.alias + ".vec2", self.alias + ".vec3", self.alias + ".vec4",
        }:
            dimension = int(self.qualified(node.func)[-1])
            values = (self.literal(value) for value in node.args)
            components = tuple(component for value in values for component in (value if isinstance(value, tuple) else (value,)))
            if len(components) == 1:
                components *= dimension
            if len(components) != dimension:
                self.error(node, "Vector defaults must fill every component")
            return components
        try:
            return ast.literal_eval(node)
        except (ValueError, TypeError):
            self.error(node, "Shader parameter defaults and presentation metadata must be literal values")

    def parameter(self, statement):
        targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
        if len(targets) != 1 or not isinstance(targets[0], ast.Name):
            self.error(statement, "Declare one named shader parameter per field")
        call = statement.value
        if not isinstance(call, ast.Call):
            self.error(statement, "Expected shader.parameter declaration")
        declaration = self.qualified(call.func)
        name = targets[0].id
        self.field_nodes[name] = statement
        if declaration == self.alias + ".input":
            if len(call.args) != 2:
                self.error(call, "shader.input names a semantic and a typed GPU resource")
            kind = self.qualified(call.args[1]).removeprefix(self.alias + ".")
            if kind not in RESOURCE_TYPES:
                self.error(call.args[1], "Unknown GPU input resource type")
            metadata = {keyword.arg: self.literal(keyword.value) for keyword in call.keywords}
            if metadata.keys() - {"domain"}:
                self.error(call, "GPU image inputs declare a color domain")
            self.fields[name] = ShaderInput(name, self.literal(call.args[0]), kind, metadata.get("domain"))
            self.types["self." + name] = RESOURCE_TYPES[kind]
            return
        prefix = self.alias + (".material." if self.domain == "mesh" else ".parameter.")
        if not declaration.startswith(prefix) or declaration[len(prefix):] not in {*_PARAMETER_TYPES, "texture_2d"}:
            self.error(call, "Expected " + prefix + " followed by a supported value type")
        kind = declaration[len(prefix):]
        values = {keyword.arg: self.literal(keyword.value) for keyword in call.keywords}
        if call.args:
            self.error(call, "Declare the parameter's default with default=...")
        unknown = values.keys() - {"default", "label", "range", "hdr"}
        if unknown:
            self.error(call, "Unknown shader parameter metadata: " + ", ".join(sorted(unknown)))
        value = ShaderParameter(name, kind, values.get("default", _DEFAULTS[kind]), values.get("label", name),
                                values.get("range"), values.get("hdr", False))
        self.validate_default(value, call)
        self.fields[name] = value
        self.types["self." + name] = (RESOURCE_TYPES["Color2D"] if kind == "texture_2d" else
                                     "bool" if kind == "bool" else _PARAMETER_TYPES[kind])

    def validate_default(self, field, node):
        value, kind = field.default, field.kind
        numeric = lambda item: type(item) in (int, float)
        valid = True
        if kind == "texture_2d":
            if self.domain != "mesh":
                self.error(node, "Texture asset properties belong to mesh materials; fullscreen images use shader.input",
                           code="author.parameter")
            valid = value in ("white", "black", "normal")
        elif kind == "float":
            valid = numeric(value)
        elif kind in {"int", "uint"}:
            minimum, maximum = (-2**31, 2**31 - 1) if kind == "int" else (0, 2**32 - 1)
            valid = type(value) is int and minimum <= value <= maximum
        elif kind == "bool":
            valid = type(value) is bool
        elif kind == "color" or kind.startswith("vec"):
            count = 4 if kind == "color" else int(kind[-1])
            valid = isinstance(value, (tuple, list)) and len(value) == count and all(map(numeric, value))
        elif kind.startswith("mat"):
            count = int(kind[-1])
            valid = (isinstance(value, (tuple, list)) and len(value) == count
                     and all(isinstance(row, (tuple, list)) and len(row) == count and all(map(numeric, row)) for row in value))
        if not valid:
            self.error(node, f"Default of {field.name!r} must match its declared {kind} type and shape", code="author.parameter")
        if field.range is not None:
            valid = (kind in {"float", "int", "uint"} and isinstance(field.range, (tuple, list))
                     and len(field.range) == 2 and all(map(numeric, field.range)) and field.range[0] <= field.range[1])
            if not valid:
                self.error(node, "A slider range is an ordered pair for a scalar numeric parameter", code="author.parameter")

    def visit_Attribute(self, node):
        if isinstance(node.value, ast.Name) and node.value.id == "self":
            field = self.fields.get(node.attr)
            if isinstance(field, ShaderInput) or (isinstance(field, ShaderParameter) and field.kind == "texture_2d"):
                return "inx_" + node.attr
            if field is None:
                self.error(node, "Unknown shader parameter: " + node.attr)
            name = "inx_parameters." + node.attr
            return "(" + name + " != 0u)" if field.kind == "bool" else name
        return super().visit_Attribute(node)

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

_MESH_VERTEX = """
[[vk::binding(5, 1)]] ConstantBuffer<InxCameraData> inx_view;
[[vk::binding(1, 2)]] StructuredBuffer<InxModelMatrix> inx_models;
[[vk::binding(2, 2)]] StructuredBuffer<InxSkinState> inx_skins;
[[vk::binding(3, 2)]] StructuredBuffer<InxModelMatrix> inx_bones;
[[vk::binding(4, 2)]] StructuredBuffer<InxInstanceData> inx_instances;
InxMeshInput inx_mesh_vertex(InxVertexInput input, uint instance)
{
    input = inx_skin_vertex(input, inx_skins[instance], inx_bones);
    return inx_transform_vertex(input, inx_models[instance].value, inx_view.view, inx_view.projection, inx_instances[instance]);
}
[shader("vertex")]
InxMeshInput vertex_main(InxVertexInput input, uint instance : SV_VulkanInstanceID)
{
    return inx_mesh_vertex(input, instance);
}
"""


_SURFACE_PASSES = """
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
        float4 previous_clip = mul(inx_view.previous_view_projection, mul(history.previous_model, float4(previous_local, 1.0)));
        result.motion_vector = (result.position.xy / max(abs(result.position.w), 1e-6)
            - float2(previous_clip.x, -previous_clip.y) / max(abs(previous_clip.w), 1e-6)) * 0.5;
    }
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
    return inx_shade_pbr<false>(inx_evaluate_surface(input, front_facing), input);
}
[shader("fragment")]
float4 fragment_forward_plus(InxMeshInput input, bool front_facing : SV_IsFrontFace) : SV_Target0
{
    return inx_shade_pbr<true>(inx_evaluate_surface(input, front_facing), input);
}
[shader("fragment")]
InxGBufferOutput fragment_gbuffer(InxMeshInput input, bool front_facing : SV_IsFrontFace)
{
    return inx_pack_gbuffer(inx_evaluate_surface(input, front_facing), input);
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


def parse_shader(source: str, source_path: str, module_name: str, *, reader=None) -> ShaderDefinition:
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
    main, helpers = None, []
    for statement in declaration.body:
        if isinstance(statement, (ast.Assign, ast.AnnAssign)):
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
    if author.is_surface:
        author.fields["_alpha_clip_threshold"] = ShaderParameter("_alpha_clip_threshold", "float", 0.0, "", internal=True)
        author.field_nodes["_alpha_clip_threshold"] = declaration
    parameters = tuple(field for field in author.fields.values() if isinstance(field, ShaderParameter))
    inputs = tuple(field for field in author.fields.values() if isinstance(field, ShaderInput))
    if author.domain == "mesh":
        for field in parameters:
            if field.kind in {"bool", "uint", "mat3"}:
                author.error(author.field_nodes[field.name], "This material value type is not supported by the material owner yet",
                             code="author.parameter")
        if inputs:
            author.error(declaration, "Mesh image resources are declared as material properties")
    code = ["import " + RESOURCE_MODULE[0] + ";", "import inx_geometry;", *("import " + name + ";" for name in dependencies),
            _MESH_VERTEX if author.domain == "mesh" else _FULLSCREEN_VERTEX]
    if author.is_surface:
        code.insert(0, "import inx_lighting;")
    for field in inputs:
        code.extend([author.source_directive(author.field_nodes[field.name]),
                     RESOURCE_TYPES[field.kind] + " inx_" + field.name + ";"])
    textures = tuple(field for field in parameters if field.kind == "texture_2d")
    for index, field in enumerate(textures):
        code.extend([author.source_directive(author.field_nodes[field.name]),
                     f"[[vk::binding({1 + 2 * index}, 0)]] InxColor2D inx_{field.name};"])
    uniforms = tuple(field for field in parameters if field.kind != "texture_2d")
    if uniforms:
        code.append("struct ShaderParameters {")
        for field in uniforms:
            code.extend([author.source_directive(author.field_nodes[field.name]),
                         "    " + _PARAMETER_TYPES[field.kind] + " " + field.name + ";"])
        binding = "[[vk::binding(0, 0)]] " if author.domain == "mesh" else ""
        code.extend(["};", binding + "ConstantBuffer<ShaderParameters> inx_parameters;"])
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
        code.append(_SURFACE_PASSES)
    return ShaderDefinition(declaration.name, source_path, module_name, parameters,
                            "\n".join(code), graph.sources(), inputs,
                            tuple(dict.fromkeys((*graph.ranges(), *author.source_ranges))), author.domain, author.is_surface)


def read_shader(path: Path, module_name: str) -> ShaderDefinition:
    return parse_shader(path.read_text(encoding="utf-8-sig"), str(path), module_name)


def _prepare_source(source, source_path, module_name, reader):
    """Return a plain native preparation snapshot; no Python object enters GPU work."""
    from infernux._source_kinds import source_extension

    if source_extension(source_path) == ".shaderlib.py":
        graph = LibraryGraph(reader)
        owner = ShaderEmitter(source, source_path)
        module = graph.load(Path(source_path), owner, ast.Constant(lineno=1, col_offset=0), source=source)
        schema = {"name": Path(source_path).name[:-len(".shaderlib.py")], "domain": "library",
                  "parameters": [], "inputs": [],
                  "functions": [{"name": name, "return_type": function.result, "arguments": function.arguments}
                                for name, function in module.functions.items()]}
        libraries = tuple(item for item in graph.sources() if item[0] != module.name)
        return module.source, libraries, json.dumps(schema, ensure_ascii=False), graph.ranges()
    definition = parse_shader(source.removeprefix("\ufeff"), source_path, module_name, reader=reader)
    schema = {"name": definition.name, "domain": definition.domain, "surface": definition.surface,
              "parameters": [asdict(field) for field in definition.parameters],
              "inputs": [asdict(field) for field in definition.inputs]}
    return definition.generated_source, definition.libraries, json.dumps(schema, ensure_ascii=False), definition.source_ranges
