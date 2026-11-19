"""Shared matrix authoring follows logical row/column shapes, without host imports."""

import ast

import pytest

from infernux._compiler.shader import parse_shader
from infernux._compiler.slang_ast import SlangEmitter


def expression_type(expression):
    node = ast.parse(expression, mode="eval").body
    emitter = SlangEmitter(expression, "matrix.frag.py")
    kind = emitter.infer(node)
    emitter.visit(node)
    return kind, emitter.required_modules


@pytest.mark.parametrize("expression,kind", [
    ("shader.inverse(shader.mat2(2, 1, 3, 4))", "float2x2"),
    ("shader.inverse(shader.mat3(1))", "float3x3"),
    ("shader.inverse(shader.mat4(1))", "float4x4"),
    ("shader.outer_product(shader.vec2(2, 3), shader.vec3(5, 7, 11))", "float2x3"),
    ("shader.transpose(shader.outer_product(shader.vec2(2, 3), shader.vec4(5)))", "float4x2"),
    ("shader.matrix_comp_mult(shader.mat3(2), shader.mat3(3))", "float3x3"),
])
def test_matrix_shapes_and_optional_math_module(expression, kind):
    assert expression_type(expression) == (kind, {"inx_math"})
    assert expression_type("shader.sin(1.0)")[1] == set()


@pytest.mark.parametrize("expression,message", [
    ("shader.inverse(shader.vec3(1))", "float mat2, mat3 or mat4"),
    ("shader.inverse(shader.outer_product(shader.vec2(1), shader.vec3(1)))", "float mat2, mat3 or mat4"),
    ("shader.inverse(shader.mat2(1), shader.mat2(1))", "1 positional"),
    ("shader.matrix_comp_mult(shader.mat2(1), shader.mat3(1))", "same shape"),
    ("shader.outer_product(shader.ivec2(1), shader.vec2(1))", "two float vectors"),
])
def test_invalid_matrix_calls_are_authored_type_errors(expression, message):
    with pytest.raises(SyntaxError, match=message):
        expression_type(expression)


def test_matrix_math_compiles_for_both_targets_without_resources():
    definition = parse_shader('''import infernux.shader as shader
class MatrixMath(shader.fullscreen):
    def main(self, ctx: shader.fragment_input) -> shader.vec4:
        a = shader.mat2(2, 1, 3, 4)
        b = shader.mat3(2, 1, 3, 0, 4, 2, 1, -1, 2)
        c = shader.mat4(2, 1, 3, 1, 0, 4, 2, 3, 1, -1, 2, 2, -1, 2, 0, 4)
        x = a @ shader.inverse(a)
        y = shader.inverse(b) @ b
        z = c @ shader.inverse(c)
        outer = shader.outer_product(shader.vec2(2, 3), shader.vec3(5, 7, 11))
        comp = shader.matrix_comp_mult(outer, outer)
        return shader.vec4(x[0][0], y[1][1], z[2][2], comp[1][2])
''', "matrix.frag.py", "matrix_math")
    assert "float2x3 outer;" in definition.generated_source
    assert "import inx_math;" in definition.generated_source
    compiled = definition.compile()
    assert {target["target"] for target in compiled["targets"]} == {"spirv", "wgsl"}
    assert not compiled["diagnostics"]
    assert all(not target["resources"] for target in compiled["targets"])


@pytest.mark.parametrize("route", ["helper", "library"])
def test_compute_math_imports_follow_helpers_and_shaderlibs(tmp_path, route):
    import importlib.util
    import numpy as np
    from infernux._compiler.slang import compile_kernel
    from infernux.engine.project_context import using_project_root
    from infernux.engine.build.compute_aot import _gpu_buffer_descriptor

    (tmp_path / "matrix.shaderlib.py").write_text('''import infernux.shader as shader
def first(value: float) -> float:
    m = shader.mat2(value, 1, 2, 3)
    return shader.inverse(m)[0][0]
''', encoding="utf8")
    source = '''from infernux import compute
import infernux.shader as shader
@compute.function
def local_inverse(value):
    return shader.inverse(shader.mat2(value, 1, 2, 3))[0][0]
matrices = compute.library(".matrix")
@compute.kernel
def solve(values, output):
    i = compute.index(values)
    output[i] = SELECTED(values[i])
'''.replace("SELECTED", "local_inverse" if route == "helper" else "matrices.first")
    path = tmp_path / "matrix_work.py"
    path.write_text(source, encoding="utf8")
    spec = importlib.util.spec_from_file_location("matrix_work_" + route, path)
    author = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(author)
    params = tuple(_gpu_buffer_descriptor((4,), np.float32) for _ in range(2))
    with using_project_root(tmp_path):
        for target in ("spirv", "wgsl"):
            artifact = compile_kernel(author.solve.function, params, target=target)
            assert artifact.target == target and len(artifact.tasks) == 1
