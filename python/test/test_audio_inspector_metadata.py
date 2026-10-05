"""Inspector source metadata uses the same codec headers as playback."""
from __future__ import annotations

import shutil
import wave
from pathlib import Path

import pytest

from infernux.engine.ui.asset_details_renderer import _load_audio


@pytest.mark.parametrize("extension", ["ogg", "mp3", "flac"])
def test_inspector_reports_encoded_audio_source_metadata(tmp_path, extension):
    fixture = Path(__file__).resolve().parents[2] / f"cpp/tests/fixtures/audio/stream_probe.{extension}"
    target = tmp_path / f"维京音乐.{extension}"
    shutil.copyfile(fixture, target)
    settings, info = _load_audio(str(target))
    assert settings.load_type == "decompress_on_load"
    assert info["channels"] == 2
    assert info["sample_rate"] == 44100
    assert 2.9 < info["duration"] < 3.2
    assert info["sample_count"] > 127000
    assert info["duration"] == pytest.approx(info["sample_count"] / info["sample_rate"])


def test_inspector_reports_wav_header_without_resident_pcm(tmp_path):
    target = tmp_path / "source.wav"
    with wave.open(str(target), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(22050)
        stream.writeframes(b"\0\0" * 22050)
    _, info = _load_audio(str(target))
    assert info == {"channels":1, "sample_rate":22050, "sample_count":22050, "duration":1.0}


@pytest.mark.parametrize("extension", ["wav", "ogg", "mp3", "flac"])
def test_invalid_audio_header_reports_failure_instead_of_zero_metadata(tmp_path, extension):
    target = tmp_path / f"broken.{extension}"
    target.write_bytes(b"not an audio stream")
    with pytest.raises(RuntimeError):
        _load_audio(str(target))
