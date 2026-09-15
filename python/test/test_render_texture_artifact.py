"""One descriptor contract for CPU import/cook and graphical allocation."""
import json

import pytest

from Infernux.lib import _Infernux as native


def document():
    return {"$type": "render_texture", "schema_version": 1,
            "size": {"width": 641, "height": 401}, "format": "rgba8_unorm",
            "depth_format": "undefined", "samples": 1, "filter": "linear",
            "storage": False, "sampled_depth": False}


@pytest.mark.parametrize("relative", [False, True])
def test_native_cpu_description_roundtrip_without_engine(relative):
    source = document()
    if relative:
        source.update(size={"scale": [0.5, 0.25]}, format="rgba16_sfloat",
                      depth_format="d32_sfloat", samples=4, sampled_depth=True)
    description = native._render_texture_description_from_json(json.dumps(source))
    payload = native._encode_render_texture_artifact(description, "0123456789abcdef")
    assert payload.startswith(b"INXRTEX1\x04\x03\x02\x01")
    assert payload[16:32] == b"0123456789abcdef"
    assert json.loads(native._render_texture_description_to_json(
        native._decode_render_texture_artifact(payload))) == source


def test_inspector_attachment_choices_come_from_native_authoring_contract():
    colors = native._render_texture_format_names(False)
    depths = native._render_texture_format_names(True)
    assert 'undefined' in depths and not set(colors).intersection(depths)
    for color in colors:
        for depth in depths:
            source = document() | {'format': color, 'depth_format': depth}
            descriptor = native._render_texture_description_from_json(json.dumps(source))
            assert json.loads(native._render_texture_description_to_json(descriptor)) == source


@pytest.mark.parametrize("patch", [
    {"samples": 3}, {"samples": True}, {"samples": 256}, {"schema_version": 2},
    {"size": {"width": 0, "height": 1}}, {"size": {"width": 1.5, "height": 3}},
    {"size": {"scale": [True, 1]}}, {"size": {"scale": [1e-100, 1]}},
    {"size": {"scale": [1e100, 1]}}, {"size": {"scale": [-1, 1]}},
    {"size": {"width": 1, "height": 3, "scale": [1, 1]}}, {"extra": "typo"},
    {"format": "undefined"}, {"format": "d32_sfloat"}, {"depth_format": "rgba8_unorm"},
    {"sampled_depth": True}, {"storage": 1}, {"filter": "automatic"},
])
def test_authoring_errors_are_rejected_by_native_contract(patch):
    source = document() | patch
    with pytest.raises((ValueError, RuntimeError)):
        native._render_texture_description_from_json(json.dumps(source))


def test_binary_truncation_corruption_and_trailing_payload_are_rejected():
    description = native._render_texture_description_from_json(json.dumps(document()))
    payload = native._encode_render_texture_artifact(description, "0123456789abcdef")
    for size in range(len(payload)):
        with pytest.raises((ValueError, RuntimeError)):
            native._decode_render_texture_artifact(payload[:size])
    for bad in (payload + b"\0", b"broken!!" + payload[8:], payload[:12] + b"\xff" * 4 + payload[16:]):
        with pytest.raises((ValueError, RuntimeError)):
            native._decode_render_texture_artifact(bad)


def test_project_creation_is_camera_ready_and_never_overwrites(tmp_path):
    from Infernux.engine.ui.project_file_ops import create_render_texture
    from Infernux.host.asset_operations import _CREATE_KINDS

    assert _CREATE_KINDS["render_texture"] == ".rendertexture"
    assert create_render_texture(str(tmp_path), "Monitor") == (True, "")
    path = tmp_path / "Monitor.rendertexture"
    description = native._render_texture_description_from_json(path.read_text(encoding="utf-8"))
    assert description.width == description.height == 256
    assert description.depth_format == native.PixelFormat.D32_SFLOAT
    assert description.color_format == native.PixelFormat.RGBA8_UNORM
    assert native._RenderTextureDesc().depth_format == native.PixelFormat.UNDEFINED
    original = path.read_bytes()
    assert not create_render_texture(str(tmp_path), "Monitor.rendertexture")[0]
    assert path.read_bytes() == original
    assert not create_render_texture(str(tmp_path), ".rendertexture")[0]
