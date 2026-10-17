"""Each run has its own real Headless engine, isolated from the Vulkan fixture."""
from pathlib import Path
import json
import os
import subprocess
import sys

import pytest
from tests.isolated import engine_python_path


@pytest.mark.parametrize('drive', ['tick', 'run'])
@pytest.mark.parametrize('case', ['steps', 'asset_save', 'scene_save', 'scene_open'])
def test_headless_frames_advance_authoring_once(tmp_path, drive, case):
    worker = Path(__file__).resolve().parents[1] / 'fixtures/headless_frame_maintenance.py'
    project = tmp_path / 'HeadlessLoop'
    env = dict(os.environ, PYTHONPATH=engine_python_path())
    result = subprocess.run([sys.executable, str(worker), case, drive, str(project)],
                            env=env, capture_output=True, text=True, encoding='utf-8', timeout=40)
    assert result.returncode == 0, result.stdout + result.stderr
    evidence = json.loads((project / 'result.json').read_text(encoding='utf-8'))
    assert evidence['passed'] and evidence['case'] == case and evidence['drive'] == drive
