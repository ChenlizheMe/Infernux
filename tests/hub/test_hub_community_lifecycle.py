import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize('action,response', [
    ('quit', 'success'), ('quit', 'failure'), ('quit', 'silent'),
    ('destroy', 'silent'), ('complete', 'success'), ('complete', 'failure'),
    ('complete', 'malformed'), ('complete', 'silent'), ('complete', 'trickle'),
])
def test_community_request_has_bounded_window_lifetime(action, response, tmp_path):
    packaging = (Path(__file__).resolve().parents[2] / "packaging")
    env = dict(os.environ, QT_QPA_PLATFORM='offscreen', PYTHONUTF8='1',
               INFERNUX_DATA_ROOT=str(tmp_path / 'hub-data'),
               INFERNUX_SHARED_DATA_ROOT=str(tmp_path / 'hub-shared'))
    env['PYTHONPATH'] = os.pathsep.join((str(packaging), str(packaging.parent / 'python')))
    result = subprocess.run([sys.executable, str(Path(__file__).with_name('hub_community_probe.py')),
                             action, response], env=env, capture_output=True,
                            text=True, encoding='utf-8', timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'COMMUNITY_LIFECYCLE_OK' in result.stdout
