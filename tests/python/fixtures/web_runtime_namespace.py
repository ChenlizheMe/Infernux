"""Cold Web API assembly with real Python/native peers and no browser services.

The empty host stands only for service registration. No host operation or
renderer is invoked; browser execution is a separate acceptance boundary.
"""
import importlib
import ast
import json
import os
from pathlib import Path
import sys
from types import ModuleType
from typing import Any

root = Path(__file__).resolve().parents[3]
bootstrap_path = root / 'external/plugins/infernux_web/native/bootstrap.py'
os.environ['INFERNUX_WEB_RUNTIME'] = '1'
os.environ['INFERNUX_WEB_CPU_COMPUTE'] = '1'
# The top-level bootstrap mounts MEMFS assets and registers actual GPU shaders.
# Execute the complete, unchanged assembler functions with their real peers.
tree = ast.parse(bootstrap_path.read_text(encoding='utf-8'))
definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
               and node.name in {'_package_namespace', '_install_platform_runtime_api'}]
assert len(definitions) == 2
bootstrap = ModuleType('web_runtime_bootstrap')
bootstrap.__dict__.update(os=os, sys=sys, ModuleType=ModuleType, Any=Any,
                          _player_python=str(root / 'python'), _runtime_api_installed=False)
exec(compile(ast.Module(definitions, type_ignores=[]), str(bootstrap_path), 'exec'), bootstrap.__dict__)
bootstrap._package_namespace('infernux', str(root / 'python/infernux'))
native = importlib.import_module('infernux.lib._Infernux')
sys.modules['_InfernuxWebHost'] = ModuleType('_InfernuxWebHost')
bootstrap._install_platform_runtime_api(native)
package = sys.modules['infernux']
assert not any(name in vars(package) for name in ('compute', 'jit', 'buffer', 'Buffer'))
name, mode = sys.argv[1:3]
if name in ('compute', 'jit', 'input', 'ui', 'physics', 'resources'):
    value = (importlib.import_module('infernux.' + name) if mode == 'direct'
             else getattr(package, name))
    assert value is getattr(package, name) is importlib.import_module('infernux.' + name)
    assert value.__name__ == 'infernux.' + name
    if name == 'jit':
        assert value.JIT_AVAILABLE is False
    if name == 'input':
        assert value.InputActionMap.standard_gameplay() is not None
        assert name in package.__all__
    if name == 'ui':
        assert value.UICanvas is not None
        assert name in package.__all__
elif name in ('buffer', 'Buffer'):
    value = getattr(package, name)
    compute = importlib.import_module('infernux.compute')
    assert value is getattr(compute, name) is getattr(package, name)
elif name in ('warmup', 'JIT_AVAILABLE'):
    value = getattr(package, name)
    assert value is getattr(importlib.import_module('infernux.jit'), name)
else:
    try:
        getattr(package, 'not_an_api')
    except AttributeError as error:
        assert str(error) == "module 'infernux' has no attribute 'not_an_api'"
    else:
        raise AssertionError('Unknown public attribute was accepted')

assert not any(key == 'Infernux' or key.startswith('Infernux.') for key in sys.modules)
assert 'infernux.engine.engine' not in sys.modules
bootstrap._install_platform_runtime_api(native)
assert sys.modules['infernux'] is package, 'Repeated setup replaced the published package'
print(json.dumps(dict(passed=True, requested=name, mode=mode, native=native.__file__)))
