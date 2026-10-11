"""Ordinary component scripts do not require desktop pipeline callbacks."""
import os
from pathlib import Path
import subprocess
import sys


def test_component_script_load_does_not_import_desktop_pipeline(tmp_path):
    source = tmp_path / 'Gameplay.py'
    source.write_text('from infernux import InxComponent\nclass Gameplay(InxComponent):\n    pass\n', encoding='utf-8')
    script = """
import importlib.abc
import sys
class NoDesktopPipeline(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'infernux.renderstack.render_pipeline':
            raise ModuleNotFoundError('Desktop pipeline callbacks are unavailable')
sys.meta_path.insert(0, NoDesktopPipeline())
import infernux.lib as native_api
del native_api.EngineConfig
del native_api.RenderPipelineCallback
from infernux.components.script_loader import load_all_components_from_file
types = load_all_components_from_file(sys.argv[1], register=False)
assert len(types) == 1 and types[0].__name__ == 'Gameplay', types
assert 'infernux.renderstack.render_pipeline' not in sys.modules
from infernux.renderstack import get_render_effect_feature
assert get_render_effect_feature('infernux.post.bloom') is not None
"""
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run([sys.executable, '-c', script, str(source)], cwd=tmp_path,
                            env=dict(os.environ, PYTHONPATH=str(root / 'python')),
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def test_pipeline_exports_keep_canonical_identity():
    import infernux.renderstack as stack
    from infernux.renderstack.render_pipeline import RenderPipeline
    from infernux.renderstack.default_forward_pipeline import DefaultForwardPipeline

    assert stack.RenderPipeline is RenderPipeline
    assert stack.DefaultForwardPipeline is DefaultForwardPipeline
    assert stack.RenderPipeline is stack.RenderPipeline
    assert 'RenderPipeline' in dir(stack)
    assert 'Default Forward' in stack.discover_pipelines()
