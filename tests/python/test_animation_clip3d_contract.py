from __future__ import annotations

import json

import pytest

from infernux.core.animation_clip3d import (
    AnimationClip3D,
    embedded_take_descriptors,
)


def _document(guid: str) -> dict:
    return {
        "name": "Walk",
        "source_model_guid": guid,
        "take_name": "Walk",
        "bind_pose_bone_names": [],
        "duration_hint": 1.0,
        "default_loop": True,
        "apply_root_motion": False,
        "reference_pose": "bind_pose",
        "curves": [],
        "events": [],
        "bone_mask": [],
    }


def test_animation_clip_save_rejects_invalid_runtime_state(tmp_path) -> None:
    clip = AnimationClip3D(source_model_guid="model-guid")

    with pytest.raises(ValueError, match="source_model_guid"):
        clip.save(str(tmp_path / "invalid.animclip3d"))


def test_embedded_take_descriptors_require_current_published_clip_table() -> None:
    current = [
        {
            "id": "source-57616c6b",
            "guid": "b" * 32,
            "name": "Walk",
            "duration": 1.0,
        }
    ]
    assert embedded_take_descriptors({"model_animations": json.dumps(current)}) == current
    assert (
        embedded_take_descriptors(
            {
                "animation_names_csv": "Walk,Run",
                "animation_count": 2,
            }
        )
        == []
    )


def test_animation_clip_imported_curves_events_and_mask_round_trip() -> None:
    from infernux.core.animation_clip3d import ImportedFloatCurve
    from infernux.core.animation_event import AnimationEvent

    clip = AnimationClip3D(
        curves=[ImportedFloatCurve("Speed", ((0.0, 2.0), (1.0, 6.0)))],
        events=[AnimationEvent(0.5, "step", "L", 1.0)],
        bone_mask=["Spine", "Arm"],
    )
    restored = AnimationClip3D.from_dict(clip.to_dict())

    assert restored.sample_curve("Speed", 0.25) == pytest.approx(3.0)
    assert restored.events[0].function == "step"
    assert restored.bone_mask == ["Spine", "Arm"]
