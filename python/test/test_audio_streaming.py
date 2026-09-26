"""Real native voices decode the same encoded assets in resident/streaming modes."""
import json
import shutil
import time
from pathlib import Path

import pytest

from Infernux.lib import AudioClip, AudioEngine


def wait_for(predicate):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.005)
    assert predicate()


@pytest.mark.parametrize("extension", ["ogg", "flac", "mp3"])
@pytest.mark.parametrize("mode", ["decompress_on_load", "streaming"])
@pytest.mark.parametrize("mono", [False, True])
def test_compressed_voice_seek_pause_loop_and_independent_cursors(scene, tmp_path, extension, mode, mono):
    path = tmp_path / f"长音乐.{extension}"
    fixture = Path(__file__).resolve().parents[2] / f"cpp/tests/fixtures/audio/stream_probe.{extension}"
    shutil.copyfile(fixture, path)
    Path(str(path) + '.meta').write_text(json.dumps({'metadata': {
        'load_type': {'type': 'string', 'value': mode},
        'force_mono': {'type': 'bool', 'value': mono},
    }}), encoding='utf-8')
    clip = AudioClip()
    assert clip.load_from_file(str(path))
    assert clip.is_streaming == (mode == 'streaming')
    assert 2.9 < clip.duration < 3.2 and clip.channels == (1 if mono else 2) and clip.sample_rate == 44100
    owner = scene.create_game_object('Codec playback probe')
    source = owner.add_component('AudioSource')._require_cpp_component()
    audio = AudioEngine.instance()
    audio.master_volume = 1
    audio.resume_all()
    source.play_on_awake = False
    source.spatial_blend = 0
    source.loop = True
    source.track_count = 2
    source.set_track_clip(0, clip)
    source.set_track_clip(1, clip)
    try:
        source.play(0)
        wait_for(lambda: audio.output_peak > .02 and source.get_track_time(0) > .05)
        source.pause(0)
        paused = source.get_track_time(0)
        time.sleep(.025)
        assert source.get_track_time(0) == paused
        source.set_track_time(0, 2.95)
        source.un_pause(0)
        wait_for(lambda: .05 < source.get_track_time(0) < 1)
        source.set_track_time(1, 1.5)
        source.play(1)
        wait_for(lambda: source.get_track_time(1) > 1.55)
        assert source.get_track_time(0) < 1
        source.set_track_time(0, 1.2)
        wait_for(lambda: 1.25 < source.get_track_time(0) < 2)
    finally:
        source.stop_all()
        scene.destroy_game_object(owner)
    before = audio.output_time
    wait_for(lambda: audio.output_time > before + .025)  # Device stays alive when silent.
