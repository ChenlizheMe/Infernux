"""Public authoring writers reject invalid candidates before touching durable files."""
import json

import pytest


@pytest.mark.parametrize("kind,field,value", [
    ("clip", "fps", -1.0), ("clip", "fps", float("nan")),
    ("timeline", "duration", -1.0), ("timeline", "duration", float("inf")),
    ("fsm", "default_state", "missing"), ("fsm", "mode", "unsupported"),
    ("fsm", "entry_position", [float("nan"), 0.0]),
])
def test_animation_save_rejects_invalid_snapshot_without_replacing_file(tmp_path, kind, field, value):
    from infernux.core.animation_clip import AnimationClip
    from infernux.core.animation_timeline import AnimationTimeline
    from infernux.core.anim_state_machine import AnimStateMachine
    cls, suffix = {
        "clip": (AnimationClip, ".animclip2d"),
        "timeline": (AnimationTimeline, ".animtimeline"),
        "fsm": (AnimStateMachine, ".animfsm"),
    }[kind]
    asset = cls(name="Valid")
    path = tmp_path / ("Valid" + suffix)
    assert asset.save(str(path))
    baseline = path.read_bytes()
    assert cls.load(str(path)) is not None
    setattr(asset, field, value)
    assert not asset.save(str(path))
    assert path.read_bytes() == baseline
    assert cls.load(str(path)) is not None


def _audio_sidecar(tmp_path):
    from infernux.core.asset_types import AudioImportSettings, write_audio_import_settings
    path = str(tmp_path / "Sound.wav")
    sidecar = tmp_path / "Sound.wav.meta"
    sidecar.write_text(json.dumps({"metadata": {"guid": {"type": "string", "value": "a" * 32}}}), encoding="utf-8")
    assert write_audio_import_settings(path, AudioImportSettings(quality=0.25))
    return path, sidecar


@pytest.mark.parametrize("quality", [float("nan"), float("inf"), -0.5, 1.5, True])
def test_audio_writer_rejects_invalid_quality_before_meta_write(tmp_path, quality):
    from infernux.core.asset_types import AudioImportSettings, write_audio_import_settings, read_audio_import_settings
    path, sidecar = _audio_sidecar(tmp_path)
    baseline = sidecar.read_bytes()
    with pytest.raises((TypeError, ValueError)):
        write_audio_import_settings(path, AudioImportSettings(quality=quality))
    assert sidecar.read_bytes() == baseline
    assert read_audio_import_settings(path).quality == 0.25


@pytest.mark.parametrize("updates", [
    {"quality": float("nan")}, {"quality": float("inf")},
    {"nested": {"value": float("nan")}},
    {"a_valid": 1, "overflow": 2**32 + 4},
    {"": "invalid empty field name"},
])
def test_meta_writer_validates_complete_updated_document(tmp_path, updates):
    from infernux.core.asset_types import write_meta_fields
    path, sidecar = _audio_sidecar(tmp_path)
    baseline = sidecar.read_bytes()
    assert not write_meta_fields(path, updates)
    assert sidecar.read_bytes() == baseline


def test_valid_audio_update_preserves_guid_and_reopens(tmp_path):
    from infernux.core.asset_types import AudioImportSettings, write_audio_import_settings, read_audio_import_settings
    path, sidecar = _audio_sidecar(tmp_path)
    candidate = AudioImportSettings(quality=0.75, force_mono=True, load_type="streaming")
    assert write_audio_import_settings(path, candidate)
    assert read_audio_import_settings(path) == candidate
    assert json.loads(sidecar.read_text(encoding="utf-8"))["metadata"]["guid"]["value"] == "a" * 32
