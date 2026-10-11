import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize('repeats', [1, 3])
@pytest.mark.parametrize('outcome', ['success', 'failure', 'cancelled', 'update', 'versions'])
def test_language_refresh_preserves_hub_services_and_tasks(repeats, outcome, tmp_path):
    packaging = (Path(__file__).resolve().parents[2] / "packaging")
    env = dict(os.environ, QT_QPA_PLATFORM='offscreen', PYTHONUTF8='1',
               INFERNUX_DATA_ROOT=str(tmp_path / 'hub-data'),
               INFERNUX_SHARED_DATA_ROOT=str(tmp_path / 'hub-shared'))
    env['PYTHONPATH'] = os.pathsep.join((str(packaging), str(packaging.parent / 'python')))
    result = subprocess.run([sys.executable, str(Path(__file__).with_name('hub_language_probe.py')),
                             str(repeats), outcome, str(tmp_path)], env=env, capture_output=True,
                            text=True, encoding='utf-8', timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'LANGUAGE_LIFECYCLE_OK' in result.stdout
