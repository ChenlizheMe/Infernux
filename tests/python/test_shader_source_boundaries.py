"""The production GLSL compiler must see the complete submitted source."""
import pytest

from infernux.lib import _Infernux as native


@pytest.mark.parametrize('stage', ['vertex', 'fragment', 'compute'])
@pytest.mark.parametrize('tail', ['plain', 'conditional', 'nested', 'comment_brace', 'declaration',
                                 'invalid_statement', 'unterminated_comment', 'missing_endif'])
def test_glsl_compiles_the_authored_tail(stage, tail):
    bodies = {
        'vertex': 'void main() { gl_Position = vec4(0, 0, 0, 1); }\n',
        'fragment': 'layout(location=0) out vec4 color;\nvoid main() { color = vec4(1); }\n',
        'compute': 'layout(local_size_x=1) in;\nvoid main() {}\n',
    }
    body = bodies[stage]
    if tail in ('conditional', 'missing_endif'):
        body = '#if 1\n' + body + ('#endif\n' if tail == 'conditional' else '')
    elif tail == 'nested':
        body = '#if 1\n#if 1\n' + body + '#endif\n#endif\n'
    elif tail == 'comment_brace':
        body += '/* a valid trailing comment with } and { in its text */\n'
    elif tail == 'declaration':
        body += 'const float tailConstant = 2.0;\n'
    elif tail == 'invalid_statement':
        body += 'this is not valid GLSL;\n'
    elif tail == 'unterminated_comment':
        body += '/* a comment that never ends\n'
    source = '#version 450\n' + body
    compile_batch = native._compile_compute_glsl_batch if stage == 'compute' else native._compile_graphics_glsl_batch
    if tail in ('invalid_statement', 'unterminated_comment', 'missing_endif'):
        with pytest.raises(RuntimeError, match='GLSL AOT failed'):
            compile_batch({stage: source}, f'boundary-{stage}-{tail}')
    else:
        compiled = compile_batch({stage: source}, f'boundary-{stage}-{tail}')
        assert bytes(compiled[stage])[:4] == b'\x03\x02\x23\x07'
