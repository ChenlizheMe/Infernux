"""Device sessions must not retire scene identities or retain destroyed streams."""
import json
import time
import wave

import pytest

from infernux.lib import AudioClip, AudioEngine


def wait_for(predicate):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.005)
    assert predicate()


def make_clip(tmp_path):
    path = tmp_path / "device.wav"
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(44100)
        stream.writeframes(b"\0\0" * 44100)
    clip = AudioClip()
    assert clip.load_from_file(str(path))
    return clip


def test_device_restarts_preserve_listener_selection_and_standby_ownership(scene):
    audio = AudioEngine.instance()
    owners = [scene.create_game_object(name) for name in ("first listener", "standby listener")]
    listeners = [owner.add_component("AudioListener") for owner in owners]
    try:
        for _ in range(2):
            assert audio.active_listener_game_object_id == owners[0].id
            audio.shutdown()
            audio.initialize()
            assert audio.is_initialized
            assert audio.active_listener_game_object_id == owners[0].id
        listeners[0].enabled = False
        assert audio.active_listener_game_object_id == owners[1].id
        audio.shutdown()
        audio.initialize()
        assert audio.active_listener_game_object_id == owners[1].id
        scene._remove_game_object_immediately(owners[1])
        assert audio.active_listener_game_object_id == 0
    finally:
        if not audio.is_initialized:
            audio.initialize()


def test_listener_lifetime_still_updates_while_device_is_closed(scene):
    audio = AudioEngine.instance()
    owners = [scene.create_game_object(name) for name in ("retiring listener", "surviving listener")]
    for owner in owners:
        owner.add_component("AudioListener")
    try:
        audio.shutdown()
        scene._remove_game_object_immediately(owners[0])
        assert audio.active_listener_game_object_id == owners[1].id
        audio.initialize()
        assert audio.is_initialized
        assert audio.active_listener_game_object_id == owners[1].id
        audio.shutdown()
        scene._remove_game_object_immediately(owners[1])
        audio.initialize()
        assert audio.is_initialized
        assert audio.active_listener_game_object_id == 0
    finally:
        if not audio.is_initialized:
            audio.initialize()


@pytest.mark.parametrize("cancel", [None, "stop", "pause", "stop_all", "play", "disable_autoplay", "deserialize_autoplay"])
def test_deferred_start_is_consumed_once_and_respects_explicit_controls(scene, tmp_path, cancel):
    audio = AudioEngine.instance()
    owner = scene.create_game_object("deferred startup")
    source = owner.add_component("AudioSource")._require_cpp_component()
    source.set_track_clip(0, make_clip(tmp_path))
    source.loop = True
    try:
        audio.shutdown()
        scene.start()
        assert not source.is_track_playing(0)
        if cancel == "disable_autoplay":
            source.play_on_awake = False
        elif cancel == "deserialize_autoplay":
            clip = source.get_track_clip(0)
            document = json.loads(source.serialize())
            document["play_on_awake"] = False
            assert source.deserialize(json.dumps(document))
            source.set_track_clip(0, clip)
        elif cancel is not None:
            getattr(source, cancel)()
        audio.initialize()
        assert audio.is_initialized
        assert source.is_track_playing(0) == (cancel is None)
        if cancel is None:
            wait_for(lambda: source.get_track_time(0) > .02)
        source.stop_all()
        for _ in range(2):
            audio.shutdown()
            audio.initialize()
            assert audio.is_initialized
            assert not source.is_track_playing(0)
    finally:
        source.stop_all()
        if not audio.is_initialized:
            audio.initialize()


def test_start_does_not_restart_a_manually_paused_track(scene, tmp_path):
    owner = scene.create_game_object("paused before Start")
    source = owner.add_component("AudioSource")._require_cpp_component()
    source.set_track_clip(0, make_clip(tmp_path))
    source.loop = True
    source.play()
    source.pause()
    source.set_track_time(0, .25)
    scene.start()
    assert source.is_track_paused(0)
    assert source.get_track_time(0) == .25


@pytest.mark.parametrize("inactive_owner", [False, True])
def test_deferred_start_waits_for_reactivation_after_device_initializes(scene, tmp_path, inactive_owner):
    audio = AudioEngine.instance()
    owner = scene.create_game_object("inactive deferred startup")
    component = owner.add_component("AudioSource")
    source = component._require_cpp_component()
    source.set_track_clip(0, make_clip(tmp_path))
    source.loop = True
    try:
        audio.shutdown()
        scene.start()
        if inactive_owner:
            owner.active = False
        else:
            component.enabled = False
        audio.initialize()
        assert audio.is_initialized
        assert not source.is_track_playing(0)
        if inactive_owner:
            owner.active = True
        else:
            component.enabled = True
        assert source.is_track_playing(0)
        wait_for(lambda: source.get_track_time(0) > .02)
    finally:
        source.stop_all()
        if not audio.is_initialized:
            audio.initialize()


def test_device_initialization_does_not_start_an_editor_source_before_start(scene, tmp_path):
    audio = AudioEngine.instance()
    owner = scene.create_game_object("not started")
    source = owner.add_component("AudioSource")._require_cpp_component()
    source.set_track_clip(0, make_clip(tmp_path))
    source.loop = True
    try:
        audio.shutdown()
        audio.initialize()
        assert audio.is_initialized
        assert not source.is_track_playing(0)
        scene.start()
        assert source.is_track_playing(0)
    finally:
        source.stop_all()
        if not audio.is_initialized:
            audio.initialize()


@pytest.mark.parametrize("disabled", [False, True])
def test_device_shutdown_invalidates_all_owned_track_and_one_shot_handles(scene, tmp_path, disabled):
    audio = AudioEngine.instance()
    owner = scene.create_game_object("owned streams")
    component = owner.add_component("AudioSource")
    source = component._require_cpp_component()
    clip = make_clip(tmp_path)
    source.play_on_awake = False
    source.loop = True
    source.track_count = 2
    source.one_shot_pool_size = 1
    for index in range(2):
        source.set_track_clip(index, clip)
    try:
        for _ in range(2):
            component.enabled = True
            source.play(0)
            source.play(1)
            source.play_one_shot(clip)
            source.pause(0)
            if disabled:
                component.enabled = False
            assert source.is_track_paused(0)
            assert source.is_track_paused(1) == disabled
            assert source.is_track_playing(1) == (not disabled)
            audio.shutdown()
            # Stop here on the old implementation, while the device is closed.
            # Never exercise its known dangling SDL pointer after reinitializing.
            assert not source.is_track_playing(0)
            assert not source.is_track_playing(1)
            assert not source.is_track_paused(0)
            assert not source.is_track_paused(1)
            audio.initialize()
            assert audio.is_initialized
            component.enabled = True
            assert not source.is_track_playing(0)
            assert not source.is_track_playing(1)
            source.play_one_shot(clip)
            source.play(1)
            wait_for(lambda: source.get_track_time(1) > .02)
            source.stop_all()
    finally:
        # This also makes baseline failure safe: when the device is closed,
        # StopAll clears references without calling SDL on stale handles.
        source.stop_all()
        if not audio.is_initialized:
            audio.initialize()
