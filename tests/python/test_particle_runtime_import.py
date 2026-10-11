"""Cooked particle readers must not require an editor background executor."""
import os
from pathlib import Path
import subprocess
import sys


def test_particle_runtime_import_without_thread_pool(tmp_path):
    script = """
import importlib.abc
import sys
class NoThreadPool(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'concurrent.futures.thread':
            raise ModuleNotFoundError('Thread pools are unavailable on this runtime')
sys.meta_path.insert(0, NoThreadPool())
from infernux.particle import ParticleArtifactRegistry, particle_artifact_filename
from infernux.components.particle_system import ParticleSystem
assert ParticleArtifactRegistry._save_executor is None
assert 'concurrent.futures.thread' not in sys.modules
assert particle_artifact_filename('test-guid') == 'test-guid.inxparticle'
ParticleArtifactRegistry.clear()
assert ParticleArtifactRegistry.get('missing-guid') is None
print('particle runtime readers imported without a thread-pool backend')
"""
    environment = dict(os.environ)
    root = Path(__file__).resolve().parents[2]
    environment['PYTHONPATH'] = str(root / 'python')
    result = subprocess.run([sys.executable, '-c', script], cwd=tmp_path, env=environment,
                            text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
