"""Preview transport never owns scene time or import settings."""
import pytest
import struct
import json
from types import SimpleNamespace

from Infernux.engine.ui.model_animation_preview import AnimationPreviewTransport, render_animation_transport


def test_transport_plays_with_wall_time_and_speed():
    state = AnimationPreviewTransport(playing=True, last_time=2.0, speed=2.0)
    state.advance(2.25, 1.0)
    assert state.seconds == pytest.approx(.5)
    state.advance(2.6, 1.0)
    assert state.seconds == pytest.approx(.2)


def test_pause_and_scrub_preserve_end_pose():
    state = AnimationPreviewTransport(playing=True)
    state.seek(8.0, 3.0)
    state.advance(200.0, 3.0)
    assert state.seconds == 3.0 and not state.playing
    state.seek(-1.0, 3.0)
    assert state.seconds == 0.0


def test_select_identity_not_label_or_order_and_reset_new_clip():
    state = AnimationPreviewTransport(clip_id="id", seconds=.5, playing=True)
    state.select("id")
    assert state.seconds == .5 and state.playing
    state.select("other")
    assert state.seconds == 0.0 and not state.playing and state.last_time is None


def test_reimport_shorter_or_zero_duration_clamps_transport():
    state = AnimationPreviewTransport(seconds=8.0)
    state.advance(10.0, 2.0)
    assert state.seconds == 2.0
    state.playing = True
    state.advance(11.0, 0.0)
    assert state.seconds == 0.0 and not state.playing


class TransportControls:
    def __init__(self, *, edited=False, value=.75, clicked=False):
        self.edited = edited
        self.value = value
        self.clicked = clicked
        self.buttons = []

    def button(self, label):
        self.buttons.append(label)
        return self.clicked

    def drag_float(self, label, value, *args):
        self.last_control_is_time = "model_preview_time" in label
        if "model_preview_time" in label and self.edited:
            return self.value
        return struct.unpack("f", struct.pack("f", value))[0]

    def is_item_edited(self):
        return self.edited and self.last_control_is_time

    def drag_int(self, label, value, *args):
        self.last_control_is_time = False
        return value

    def record_semantic_item(self, *args, **kwargs):
        pass

    def same_line(self):
        pass

    def label(self, value):
        pass


def test_widget_float32_round_trip_does_not_stop_playback(monkeypatch):
    monkeypatch.setattr("Infernux.engine.ui.model_animation_preview.time.perf_counter", lambda: 10.123456789)
    state = AnimationPreviewTransport(playing=True, last_time=10.0)
    controls = TransportControls()
    render_animation_transport(controls, state, 2.0)
    assert state.playing
    assert state.seconds == pytest.approx(.123456789)
    assert state.last_time == 10.123456789


def test_actual_scrub_stops_playback_even_when_value_is_unchanged(monkeypatch):
    monkeypatch.setattr("Infernux.engine.ui.model_animation_preview.time.perf_counter", lambda: 10.0)
    state = AnimationPreviewTransport(seconds=.75, playing=True, last_time=10.0)
    render_animation_transport(TransportControls(edited=True), state, 2.0)
    assert not state.playing and state.seconds == .75 and state.last_time is None


def test_play_pause_uses_stable_widget_identity(monkeypatch):
    monkeypatch.setattr("Infernux.engine.ui.model_animation_preview.time.perf_counter", lambda: 10.0)
    state = AnimationPreviewTransport()
    controls = TransportControls(clicked=True)
    render_animation_transport(controls, state, 2.0)
    assert state.playing
    render_animation_transport(controls, state, 2.0)
    assert not state.playing
    assert controls.buttons[0].split("###")[1] == controls.buttons[1].split("###")[1]


def test_frame_grid_and_end_pose_do_not_change_clip_duration():
    state = AnimationPreviewTransport(frame_rate=24)
    state.seek_frame(12, .8333333333333334)
    assert state.seconds == .5 and state.frame(.8333333333333334) == (12, 20)
    state.seek_frame(20, .81)
    assert state.seconds == .81 and not state.playing
    assert state.frame(.81) == (20, 20)
    state.frame_rate = 60
    assert state.seconds == .81


@pytest.mark.parametrize("take", ["stable-id", "Walk"])
def test_child_and_standalone_clip_preview_uses_published_owner(monkeypatch, take):
    from Infernux.engine.ui import model_animation_preview as preview
    record = {"id": "stable-id", "name": "Walk", "duration": 2.0}
    monkeypatch.setattr("Infernux.core.asset_types.read_asset_metadata",
                        lambda path: {"model_animations": json.dumps([record])})
    monkeypatch.setattr("Infernux.core.assets.AssetManager.preview_dependency_signature", lambda path: 1)
    calls = []
    monkeypatch.setattr(preview, "render_model_animation_preview",
                        lambda *args, **kwargs: calls.append((args, kwargs)))
    state = SimpleNamespace(file_path="model.fbx::subanim:stable-id", extra={})
    preview.render_clip_animation_preview(None, None, state, "model.fbx", take)
    assert calls == [((None, None, state, [record]), {"source_path": "model.fbx"})]


def test_missing_clip_does_not_preview_a_different_take(monkeypatch):
    from Infernux.engine.ui import model_animation_preview as preview
    monkeypatch.setattr("Infernux.core.asset_types.read_asset_metadata",
                        lambda path: {"model_animations": "[]"})
    monkeypatch.setattr("Infernux.core.assets.AssetManager.preview_dependency_signature", lambda path: 1)
    messages = []
    ctx = SimpleNamespace(text_wrapped=messages.append)
    preview.render_clip_animation_preview(ctx, None, SimpleNamespace(extra={}), "model.fbx", "deleted")
    assert len(messages) == 1


def test_clip_preview_source_cache_follows_published_dependency_revision(monkeypatch):
    from Infernux.engine.ui import model_animation_preview as preview
    revision = [1]
    reads = []
    monkeypatch.setattr("Infernux.core.assets.AssetManager.preview_dependency_signature", lambda path: revision[0])
    def read(path):
        reads.append(path)
        return {"model_animations": "[]"}
    monkeypatch.setattr("Infernux.core.asset_types.read_asset_metadata", read)
    ctx, state = SimpleNamespace(text_wrapped=lambda _: None), SimpleNamespace(extra={})
    for _ in range(10):
        preview.render_clip_animation_preview(ctx, None, state, "model.fbx", "deleted")
    assert reads == ["model.fbx"]
    revision[0] += 1
    preview.render_clip_animation_preview(ctx, None, state, "model.fbx", "deleted")
    assert reads == ["model.fbx", "model.fbx"]
