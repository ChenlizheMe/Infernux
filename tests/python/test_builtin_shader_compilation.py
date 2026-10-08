"""Compile authored builtin stages, including their shared library imports."""
from pathlib import Path
import re

import pytest

from infernux.lib import _Infernux as native


SHADERS = Path(__file__).parents[2] / 'python/infernux/resources/shaders'
STAGES = sorted(path for path in SHADERS.iterdir()
                if path.suffix in {'.vert', '.frag'}
                and re.search(r'Capabilities\s*\[[^\]]*\b(?:Fullscreen|Standalone)\b',
                              path.read_text(encoding='utf-8')))


@pytest.mark.parametrize('path', STAGES, ids=lambda path:path.name)
def test_builtin_authored_stage_compiles(engine, path):
    generated = native._prepare_authored_shader_glsl(path.read_text(encoding='utf-8'), str(path))
    stage = 'vertex' if path.suffix == '.vert' else 'fragment'
    spirv = native._compile_graphics_glsl_batch({stage:generated}, str(path))[stage]
    assert spirv[:4] == b'\x03\x02\x23\x07'
    assert len(spirv) >= 20 and len(spirv) % 4 == 0
