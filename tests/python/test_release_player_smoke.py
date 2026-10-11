"""The Release gate must reject invalid gameplay and empty rendering evidence."""
from copy import deepcopy

import numpy as np
from PIL import Image
import pytest

from tests.acceptance.release_player_smoke import require_release_manifest, verify_evidence


@pytest.mark.parametrize('debug,flavor,control', [
    (True, 'PlayerDevelopment', 'enabled'),
    (False, 'PlayerRelease', 'enabled'),
    (False, 'PlayerDevelopment', 'disabled'),
])
def test_release_gate_rejects_non_release_policy(debug, flavor, control):
    with pytest.raises(RuntimeError, match='PlayerRelease'):
        require_release_manifest(dict(debug_build=debug, runtime_contract=dict(
            product=dict(flavor=flavor), runtime_policy=dict(player_control=control))))


@pytest.fixture
def evidence(tmp_path):
    pixels = np.tile(np.arange(64, dtype=np.uint8)[None, :, None], (64, 1, 3))
    pixels[:8, :8] = (0, 255, 0)
    capture = tmp_path / 'scene.png'
    Image.fromarray(pixels).save(capture)
    data = dict(status='passed', fixed_steps=90, initial_position=[0, .5, 0],
        final_position=[0, .5, 1], trail_points=8, cpu_jit_result=84,
        renderer=dict(submission_ready=True),
        package_text='Package resource reached UIText on every Player target.',
        managed_guid='8b7148eba8303c90b0589315c16f7cba')
    verify_evidence(data, capture)
    return data, capture


@pytest.mark.parametrize('position', [[0, .5, float('nan')], [0, .5, float('inf')], [0, -1, 1], [0, .5, .1]])
def test_release_gate_rejects_invalid_physics(evidence, position):
    data, capture = evidence
    data = deepcopy(data)
    data['final_position'] = position
    with pytest.raises(RuntimeError, match='physics'):
        verify_evidence(data, capture)


@pytest.mark.parametrize('kind', ['flat', 'no-ui'])
def test_release_gate_rejects_missing_rendering(evidence, kind):
    data, capture = evidence
    pixels = np.zeros((64, 64, 3), dtype=np.uint8)
    if kind == 'no-ui':
        pixels[:] = np.arange(64, dtype=np.uint8)[None, :, None]
    Image.fromarray(pixels).save(capture)
    with pytest.raises(RuntimeError, match='screenshot'):
        verify_evidence(data, capture)
