"""The shared Slang compiler must see the complete submitted source."""

import pytest

from infernux.lib import _Infernux as native


@pytest.mark.parametrize(
    "stage,tail",
    [
        ("vertex", "plain"),
        ("compute", "nested"),
        ("fragment", "comment_brace"),
        ("fragment", "invalid_statement"),
        ("compute", "unterminated_comment"),
    ],
)
def test_slang_compiles_the_authored_tail(stage, tail):
    bodies = {
        "vertex": '[shader("vertex")] float4 main() : SV_Position { return float4(0, 0, 0, 1); }\n',
        "fragment": '[shader("fragment")] float4 main() : SV_Target0 { return float4(1); }\n',
        "compute": '[shader("compute")] [numthreads(1,1,1)] void main() {}\n',
    }
    body = bodies[stage]
    if tail in ("conditional", "missing_endif"):
        body = "#if 1\n" + body + ("#endif\n" if tail == "conditional" else "")
    elif tail == "nested":
        body = "#if 1\n#if 1\n" + body + "#endif\n#endif\n"
    elif tail == "comment_brace":
        body += "/* a valid trailing comment with } and { in its text */\n"
    elif tail == "declaration":
        body += "static const float tailConstant = 2.0;\n"
    elif tail == "invalid_statement":
        body += "this is not valid Slang;\n"
    elif tail == "unterminated_comment":
        body += "/* a comment that never ends\n"
    label = f"boundary-{stage}-{tail}.slang"
    if tail in ("invalid_statement", "unterminated_comment", "missing_endif"):
        with pytest.raises(native.ShaderCompilationError) as failure:
            native._compile_shader_program(body, label, "boundary", targets=["spirv"])
        assert failure.value.diagnostics and any(item["path"] == label for item in failure.value.diagnostics)
    else:
        compiled = native._compile_shader_program(body, label, "boundary", targets=["spirv"])
        assert bytes(compiled["targets"][0]["entries"][0]["code"])[:4] == b"\x03\x02\x23\x07"
