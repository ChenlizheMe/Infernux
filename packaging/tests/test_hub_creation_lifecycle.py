import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize('action', ['escape', 'close', 'reject', 'accept', 'done', 'reentry', 'quit', 'hub-quit'])
@pytest.mark.parametrize('outcome', ['success', 'failure'])
def test_creation_owns_its_progress_thread_and_result(action, outcome, tmp_path):
    packaging = Path(__file__).resolve().parents[1]
    env = dict(os.environ, QT_QPA_PLATFORM='offscreen', PYTHONUTF8='1',
               INFERNUX_DATA_ROOT=str(tmp_path / 'hub-data'),
               INFERNUX_SHARED_DATA_ROOT=str(tmp_path / 'hub-shared'))
    env['PYTHONPATH'] = os.pathsep.join((str(packaging), str(packaging.parent / 'python')))
    result = subprocess.run([sys.executable, str(Path(__file__).with_name('hub_creation_probe.py')),
                             action, outcome, str(tmp_path)], env=env,
                            capture_output=True, text=True, encoding='utf-8', timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'CREATION_LIFECYCLE_OK' in result.stdout
