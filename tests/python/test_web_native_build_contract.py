"""Web release builds must consume the current engine source inventory."""
from pathlib import Path
import re


def test_web_native_engine_sources_resolve():
    root = Path(__file__).resolve().parents[2]
    source = (root / 'external/plugins/infernux_web/native/CMakeLists.txt').read_text(encoding='utf-8')
    paths = {path for path in re.findall(r'"\$\{INFERNUX_ENGINE_SOURCE_ROOT\}/(cpp/[^"$]+)"', source)
             if '*' not in path}  # Recursive glob expressions are evaluated by CMake itself.
    assert paths, 'The Web runtime must explicitly declare its engine inputs'
    missing = sorted(path for path in paths if not list(root.glob(path)))
    assert not missing, f'Web native build references absent engine sources: {missing}'
