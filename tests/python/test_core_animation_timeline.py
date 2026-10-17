"""Public animation authoring workflows and invalid document boundaries."""

from __future__ import annotations


import pytest


from infernux.core.animation_timeline import (
    AnimationTimeline,
    TimelineKeyframe,
    INTERP_CONSTANT,
    INTERP_LINEAR,
    INTERP_EASE_IN_OUT,
    APPLY_ABSOLUTE,
)


def test_sample_zero_span_coincident_keys():
    tl = AnimationTimeline(
        keyframes=[
            TimelineKeyframe(time=1.0, position=[1, 0, 0]),
            TimelineKeyframe(time=1.0, position=[2, 0, 0]),
        ]
    )
    result = tl.sample(1.0)
    assert result is not None  # must not divide by zero


def test_timeline_rejects_duplicate_keyframe_ids():
    first = TimelineKeyframe(stable_id="same")
    second = TimelineKeyframe(stable_id="same")
    timeline = AnimationTimeline(keyframes=[first, second])

    with pytest.raises(ValueError, match="must be unique"):
        timeline.to_dict()


@pytest.mark.parametrize(
    "interp,fraction", [(INTERP_CONSTANT, 0.0), (INTERP_LINEAR, 0.25), (INTERP_EASE_IN_OUT, 0.15625)]
)
def test_authored_timeline_samples_all_transform_channels_and_boundaries(interp, fraction):
    first = TimelineKeyframe(time=0.0, position=[0, 0, 0], rotation=[0, 0, 0], scale=[1, 1, 1])
    last = TimelineKeyframe(time=2.0, position=[10, 0, 0], rotation=[0, 90, 0], scale=[3, 3, 3], interp=interp)
    timeline = AnimationTimeline(keyframes=[last, first])
    position, rotation, scale = timeline.sample(0.5)
    assert position == pytest.approx([10 * fraction, 0, 0])
    assert rotation == pytest.approx([0, 90 * fraction, 0])
    assert scale == pytest.approx([1 + 2 * fraction] * 3)
    assert timeline.sample(-1.0)[0] == [0.0, 0.0, 0.0]
    assert timeline.sample(3.0)[0] == [10.0, 0.0, 0.0]
    assert timeline.keyframes == [last, first]
    position[0] = 999
    assert first.position == [0, 0, 0] and last.position == [10, 0, 0]


def test_timeline_document_and_file_preserve_stable_keyframe_identity(tmp_path):
    first = TimelineKeyframe(time=0.0, position=[1, 2, 3])
    last = TimelineKeyframe(time=3.5, position=[7, 8, 9], rotation=[10, 20, 30], scale=[2, 2, 2])
    timeline = AnimationTimeline(name="Spin", duration=3.5, apply_mode=APPLY_ABSOLUTE, keyframes=[first, last])
    assert not timeline.save()
    document = timeline.to_dict()
    assert "file_path" not in document
    restored = AnimationTimeline.from_dict(document)
    assert restored.to_dict() == document
    assert restored.find_keyframe(first.stable_id).position == [1.0, 2.0, 3.0]
    assert restored.find_keyframe("missing") is None
    path = str(tmp_path / "Spin.animtimeline")
    assert timeline.save(path)
    loaded = AnimationTimeline.load(path)
    assert loaded.to_dict() == document
    loaded.keyframes[0].position[0] = 100
    assert first.position == [1, 2, 3]


@pytest.mark.parametrize("contents", [None])
def test_timeline_load_rejects_unreadable_documents(tmp_path, contents):
    path = tmp_path / "Invalid.animtimeline"
    if contents is not None:
        path.write_text(contents, encoding="utf-8")
    assert AnimationTimeline.load(str(path)) is None
