"""Preview transport never owns scene time or import settings."""

import pytest
import struct
import json
from types import SimpleNamespace

from infernux.engine.ui.model_animation_preview import AnimationPreviewTransport


class TransportControls:
    def __init__(self, *, edited=False, value=0.75, clicked=False):
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


@pytest.mark.parametrize("take", ["stable-id", "Walk"])
def test_child_and_standalone_clip_preview_uses_published_owner(monkeypatch, take):
    from infernux.engine.ui import model_animation_preview as preview

    record = {"id": "stable-id", "name": "Walk", "duration": 2.0}
    monkeypatch.setattr(
        "infernux.core.asset_types.read_asset_metadata", lambda path: {"model_animations": json.dumps([record])}
    )
    monkeypatch.setattr("infernux.core.assets.AssetManager.preview_dependency_signature", lambda path: 1)
    calls = []
    monkeypatch.setattr(preview, "render_model_animation_preview", lambda *args, **kwargs: calls.append((args, kwargs)))
    state = SimpleNamespace(file_path="model.fbx::subanim:stable-id", extra={})
    preview.render_clip_animation_preview(None, None, state, "model.fbx", take)
    assert calls == [((None, None, state, [record]), {"source_path": "model.fbx"})]


def test_missing_clip_does_not_preview_a_different_take(monkeypatch):
    from infernux.engine.ui import model_animation_preview as preview

    monkeypatch.setattr("infernux.core.asset_types.read_asset_metadata", lambda path: {"model_animations": "[]"})
    monkeypatch.setattr("infernux.core.assets.AssetManager.preview_dependency_signature", lambda path: 1)
    messages = []
    ctx = SimpleNamespace(text_wrapped=messages.append)
    preview.render_clip_animation_preview(ctx, None, SimpleNamespace(extra={}), "model.fbx", "deleted")
    assert len(messages) == 1


def test_clip_preview_source_cache_follows_published_dependency_revision(monkeypatch):
    from infernux.engine.ui import model_animation_preview as preview

    revision = [1]
    reads = []
    monkeypatch.setattr("infernux.core.assets.AssetManager.preview_dependency_signature", lambda path: revision[0])

    def read(path):
        reads.append(path)
        return {"model_animations": "[]"}

    monkeypatch.setattr("infernux.core.asset_types.read_asset_metadata", read)
    ctx, state = SimpleNamespace(text_wrapped=lambda _: None), SimpleNamespace(extra={})
    for _ in range(10):
        preview.render_clip_animation_preview(ctx, None, state, "model.fbx", "deleted")
    assert reads == ["model.fbx"]
    revision[0] += 1
    preview.render_clip_animation_preview(ctx, None, state, "model.fbx", "deleted")
    assert reads == ["model.fbx", "model.fbx"]


def test_preview_transport_plays_scrubs_reimports_and_selects_by_identity():
    state = AnimationPreviewTransport(clip_id="id", playing=True, last_time=2.0, speed=2.0)
    state.advance(2.25, 1.0)
    assert state.seconds == pytest.approx(0.5)
    state.advance(2.6, 1.0)
    assert state.seconds == pytest.approx(0.2)
    state.select("id")
    assert state.seconds == pytest.approx(0.2) and state.playing
    state.seek(8.0, 3.0)
    state.advance(200.0, 3.0)
    assert state.seconds == 3.0 and not state.playing
    state.seek(-1.0, 3.0)
    assert state.seconds == 0.0
    state.frame_rate = 24
    state.seek_frame(12, 0.8333333333333334)
    assert state.seconds == 0.5 and state.frame(0.8333333333333334) == (12, 20)
    state.seek_frame(20, 0.81)
    assert state.seconds == 0.81 and state.frame(0.81) == (20, 20) and not state.playing
    state.frame_rate = 60
    assert state.seconds == 0.81
    state.advance(201.0, 0.5)
    assert state.seconds == 0.5
    state.playing = True
    state.advance(202.0, 0.0)
    assert state.seconds == 0.0 and not state.playing
    state.seconds, state.playing = 0.5, True
    state.select("other")
    assert state.seconds == 0.0 and not state.playing and state.last_time is None
