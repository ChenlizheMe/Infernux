"""Evidence capture must preserve the original numerical failure and images."""
import json
import struct
import zlib

import numpy as np
import pytest

from tests.gpu.surface_discard_diagnostics import SurfaceDiscardEvidence, coverage_metrics


def images_with_shadow(count):
    baseline = np.full((20, 20, 4), .5, dtype=np.float32)
    masked = baseline.copy()
    masked[:5, :, :3] = (1., 0., 1.)
    masked[5:, :, 0].flat[:count] = .3
    solid = masked.copy()
    solid[:10, :, :3] = (1., 0., 1.)
    return dict(baseline=baseline, masked_on=masked, solid=solid)


@pytest.mark.parametrize('shadow', [0, 50, 51, 80])
def test_metrics_retain_shadow_boundary(shadow):
    metrics = coverage_metrics(images_with_shadow(shadow))
    assert metrics['visible'] == 100
    assert metrics['solid'] == 200
    assert metrics['shadow'] == shadow
    assert (metrics['shadow'] > 50) == (shadow >= 51)


def test_partial_stages_and_empty_outside_are_supported():
    assert coverage_metrics({}) == {}
    pixels = np.ones((2, 2, 4), dtype=np.float32)
    pixels[..., 1] = 0
    metrics = coverage_metrics(dict(masked_on=pixels, baseline=pixels))
    assert metrics == dict(visible=4, shadow=0, outside_nonfinite=0)
    assert coverage_metrics(dict(masked_on=pixels), depth=True) == dict(visible=4)


def test_failure_exports_lossless_images_and_frame_metadata(tmp_path):
    evidence = SurfaceDiscardEvidence(tmp_path, kind='direct', pipeline='forward', samples=1)
    images = images_with_shadow(0)
    images['baseline'][0, 0, 0] = .12345679
    images['baseline'][19, 19, 3] = np.nan
    for step, pixels in images.items():
        evidence.record(step, pixels, dict(execution=8, passes=['ShadowCasterPass']))
    # A later change to the readback buffer must not mutate retained evidence.
    images['masked_on'][0, 0, 0] = .123
    error = AssertionError('Separated receiver shadow is missing')
    evidence.export_failure(error, [dict(level='ERROR', message='example')])
    with np.load(tmp_path / 'images.npz', allow_pickle=False) as saved:
        assert saved['masked_on'][0, 0, 0] == 1.
        np.testing.assert_array_equal(saved['baseline'], images['baseline'])
        assert saved['baseline'].dtype == np.float32
    report = json.loads((tmp_path / 'evidence.json').read_text(encoding='utf-8'))
    assert report['metrics']['shadow'] == 0
    assert report['images']['baseline']['nonfinite'] == 1
    assert report['frames'][0]['passes'] == ['ShadowCasterPass']
    assert report['logs'][0]['level'] == 'ERROR'
    preview = (tmp_path / 'baseline.png').read_bytes()
    assert preview.startswith(b'\x89PNG\r\n\x1a\n')
    # Decode the emitted PNG instead of merely checking that a file exists.
    offset, compressed = 8, b''
    while offset < len(preview):
        length = struct.unpack('>I', preview[offset:offset + 4])[0]
        kind = preview[offset + 4:offset + 8]
        data = preview[offset + 8:offset + 8 + length]
        crc = struct.unpack('>I', preview[offset + 8 + length:offset + 12 + length])[0]
        assert crc == zlib.crc32(kind + data)
        if kind == b'IDAT':
            compressed += data
        offset += 12 + length
    rows = np.frombuffer(zlib.decompress(compressed), dtype=np.uint8).reshape(20, 61)
    assert not rows[:, 0].any()  # PNG filter type: None
    np.testing.assert_array_equal(rows[0, 1:4], (31, 127, 127))
    assert str(tmp_path) in error.__notes__[0]


def test_empty_capture_can_be_exported(tmp_path):
    evidence = SurfaceDiscardEvidence(tmp_path, kind='direct', pipeline='forward', samples=1)
    evidence.export_failure(RuntimeError('setup failed'), [])
    report = json.loads((tmp_path / 'evidence.json').read_text(encoding='utf-8'))
    assert report['images'] == {}
    assert report['metrics'] == {}


def test_export_failure_does_not_replace_original_error(tmp_path):
    blocked = tmp_path / 'not-a-directory'
    blocked.write_text('block directory creation', encoding='utf-8')
    evidence = SurfaceDiscardEvidence(blocked, kind='direct', pipeline='forward', samples=1)
    original = AssertionError('original GPU assertion')
    with pytest.raises(AssertionError) as caught:
        try:
            raise original
        except AssertionError as error:
            evidence.export_failure(error, [])
            raise
    assert caught.value is original
    assert 'export failed' in original.__notes__[0]


def test_parameter_directories_do_not_overwrite_each_other(tmp_path):
    for kind in ('direct', 'helper'):
        evidence = SurfaceDiscardEvidence(tmp_path / kind, kind=kind, pipeline='forward', samples=1)
        evidence.export_failure(AssertionError(kind), [])
    for kind in ('direct', 'helper'):
        report = json.loads((tmp_path / kind / 'evidence.json').read_text(encoding='utf-8'))
        assert report['kind'] == kind
