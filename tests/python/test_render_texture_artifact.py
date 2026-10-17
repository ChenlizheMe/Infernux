"""One descriptor contract for CPU import/cook and graphical allocation."""

import json

import pytest

from infernux.lib import _Infernux as native


def document():
    return {
        "$type": "render_texture",
        "size": {"width": 641, "height": 401},
        "format": "rgba8_unorm",
        "depth_format": "undefined",
        "samples": 1,
        "filter": "linear",
        "storage": False,
        "sampled_depth": False,
    }


@pytest.mark.parametrize("relative", [False, True])
def test_native_cpu_description_roundtrip_without_engine(relative):
    source = document()
    if relative:
        source.update(
            size={"scale": [0.5, 0.25]},
            format="rgba16_sfloat",
            depth_format="d32_sfloat",
            samples=4,
            sampled_depth=True,
        )
    description = native._render_texture_description_from_json(json.dumps(source))
    payload = native._encode_render_texture_artifact(description, "0123456789abcdef")
    assert payload.startswith(b"INXRTEX1\x04\x03\x02\x01")
    assert payload[16:32] == b"0123456789abcdef"
    assert (
        json.loads(native._render_texture_description_to_json(native._decode_render_texture_artifact(payload)))
        == source
    )


def test_binary_truncation_corruption_and_trailing_payload_are_rejected():
    description = native._render_texture_description_from_json(json.dumps(document()))
    payload = native._encode_render_texture_artifact(description, "0123456789abcdef")
    for size in range(len(payload)):
        with pytest.raises((ValueError, RuntimeError)):
            native._decode_render_texture_artifact(payload[:size])
    for bad in (payload + b"\0", b"broken!!" + payload[8:], payload[:12] + b"\xff" * 4 + payload[16:]):
        with pytest.raises((ValueError, RuntimeError)):
            native._decode_render_texture_artifact(bad)
