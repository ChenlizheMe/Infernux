"""Load both packaged extensions and exercise the compiler without site-packages."""

import importlib.util
import os
from pathlib import Path
import sys


def load_extension(directory, name):
    path, = directory.glob(f'{name}.*.pyd')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


directory = Path(sys.argv[1]).resolve()
with os.add_dll_directory(str(directory)):
    load_extension(directory, '_InfernuxBootstrap')
    native = load_extension(directory, '_Infernux')
    program = native._compile_shader_program(
        '[shader("compute")] [numthreads(1,1,1)] void main() {}',
        'wheel_check.slang', 'wheel_check', targets=['spirv', 'wgsl'],
    )
    code = {target['target']: target['entries'][0]['code'] for target in program['targets']}
    if not code['spirv'].startswith(b'\x03\x02\x23\x07') or b'@compute' not in code['wgsl']:
        raise RuntimeError('The packaged Slang compiler did not emit both requested compute shaders')
    print(f'Packaged Slang {native._slang_version()} compiled SPIR-V and WGSL')
