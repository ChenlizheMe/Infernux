"""Incomplete image sources never replace a complete imported texture."""
from pathlib import Path
import struct
import zlib

import pytest

from infernux.lib import _Infernux as native


def _png():
    def chunk(kind, payload):
        return struct.pack('>I', len(payload)) + kind + payload + struct.pack('>I', zlib.crc32(kind + payload))
    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', 2, 1, 8, 6, 0, 0, 0))
            + chunk(b'IDAT', zlib.compress(bytes([0, 255, 0, 0, 255, 0, 0, 255, 255]))) + chunk(b'IEND', b''))


VALID = [
    ('png', _png(), bytes([255, 0, 0, 255, 0, 0, 255, 255])),
    ('p3', b'P3\n2 1\n255\n255 0 0 0 0 255\n', bytes([255, 0, 0, 255, 0, 0, 255, 255])),
    ('p2', b'P2\n2 1\n15\n15 5\n', bytes([255, 255, 255, 255, 85, 85, 85, 255])),
    ('p3_comments', b'P3\n# comment\n2 1\n15\n15 0 # pixel\n0 0 0 15\n', bytes([255, 0, 0, 255, 0, 0, 255, 255])),
    ('p3_16', b'P3\n2 1\n65535\n65535 0 0 0 32768 65535\n', bytes([255, 0, 0, 255, 0, 128, 255, 255])),
    ('p6', b'P6\n2 1\n255\n' + bytes([255, 0, 0, 0, 0, 255]), bytes([255, 0, 0, 255, 0, 0, 255, 255])),
    ('p5_space', b'P5\n2 1\n255\n' + bytes([32, 35]), bytes([32, 32, 32, 255, 35, 35, 35, 255])),
    ('p6_scale', b'P6\n2 1\n15\n' + bytes([15, 0, 0, 0, 5, 15]), bytes([255, 0, 0, 255, 0, 85, 255, 255])),
    ('p5_16', b'P5\n2 1\n1023\n' + bytes([3, 255, 2, 0]), bytes([255, 255, 255, 255, 128, 128, 128, 255])),
    ('p6_16', b'P6\n2 1\n65535\n' + bytes([255, 255, 0, 0, 0, 0, 0, 0, 128, 0, 255, 255]), bytes([255, 0, 0, 255, 0, 128, 255, 255])),
]
INVALID = [
    ('p3_impossible_size', b'P3\n65535 65535\n255\n0\n'),
    ('p3_partial', b'P3\n2 1\n255\n255 0 0 0 0\n'),
    ('p2_partial', b'P2\n2 1\n255\n255\n'),
    ('p3_token', b'P3\n2 1\n255\n255 0 0 0 0 invalid\n'),
    ('p2_token', b'P2\n2 1\n255\n255 invalid\n'),
    ('p3_negative', b'P3\n1 1\n255\n255 -1 0\n'),
    ('p3_above_max', b'P3\n1 1\n15\n16 0 0\n'),
    ('p2_above_max', b'P2\n1 1\n15\n16\n'),
    ('p5_partial', b'P5\n2 1\n255\n\xff'),
    ('p6_partial', b'P6\n1 1\n255\n\xff\x00'),
    ('p5_16_partial', b'P5\n2 1\n1023\n\x03\xff\x02'),
    ('p6_above_max', b'P6\n1 1\n15\n\x10\x00\x00'),
    ('p5_above_max', b'P5\n1 1\n15\n\x10'),
    ('p3_zero_width', b'P3\n0 1\n255\n0 0 0'),
    ('p3_zero_max', b'P3\n1 1\n0\n0 0 0'),
    ('p3_bad_header', b'P3\n1 not-height\n255\n0 0 0'),
    ('empty', b''),
]


@pytest.mark.parametrize('entry', ['memory', 'file'])
@pytest.mark.parametrize('case,payload,pixels', VALID + [(c, p, None) for c, p in INVALID],
                         ids=[c for c, *_ in VALID + INVALID])
def test_texture_source_complete_or_invalid(tmp_path, entry, case, payload, pixels):
    if entry == 'memory':
        texture = native.TextureLoader.load_from_memory(payload, case)
    else:
        path = tmp_path / f'{case}.ppm'
        path.write_bytes(payload)
        texture = native.TextureLoader.load_from_file(str(path), case)
        assert Path(texture.source_path) == path
    assert texture.name == case
    assert texture.is_valid() is (pixels is not None)
    if pixels is None:
        assert texture.get_pixels() == b''
        assert (texture.width, texture.height, texture.get_size_bytes()) == (0, 0, 0)
    else:
        assert (texture.width, texture.height, texture.channels) == (2, 1, 4)
        assert texture.get_pixels() == pixels


@pytest.mark.parametrize('case,payload', INVALID + [(c, p) for c, p, _ in VALID],
                         ids=[c for c, *_ in INVALID + VALID])
def test_texture_reimport_publishes_only_complete_source(engine, case, payload):
    registry = native.AssetRegistry.instance()
    database = registry.get_asset_database()
    source = Path(database.assets_root) / f'decode-publication-{case}.ppm'
    source.write_bytes(b'P3\n2 1\n255\n0 255 0 0 255 0\n')
    initial = database.import_asset(str(source))
    assert initial, initial.error
    guid = initial.guid
    artifact = Path(database.get_runtime_artifact_path(guid, native.ResourceType.Texture))
    before = artifact.read_bytes()
    try:
        source.write_bytes(payload)
        result = database.reimport_asset(str(source))
        assert bool(result) is (case in {c for c, *_ in VALID}), result.error
        assert database.get_guid_from_path(str(source)) == guid
        after_path = Path(database.get_runtime_artifact_path(guid, native.ResourceType.Texture))
        if not result:
            assert result.error
            assert after_path == artifact
            assert after_path.read_bytes() == before
            # A corrected save is accepted on the same asset without deleting its meta.
            source.write_bytes(VALID[0][1])
            corrected = database.reimport_asset(str(source))
            assert corrected, corrected.error
            assert corrected.guid == guid
        else:
            assert after_path.read_bytes() != before
    finally:
        registry.invalidate_asset(guid)
        database.delete_asset(str(source))
