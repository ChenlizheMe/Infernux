from __future__ import annotations

import json

import pytest

from Infernux.core.animation_clip3d import (
    AnimationClip3D,
    embedded_take_descriptors,
    is_asset_guid_string,
    resolve_disk_path_for_guid_string,
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


def test_animation_clip_uses_current_asset_guid_contract(tmp_path) -> None:
    guid = "a" * 32
    model = tmp_path / "Robot.fbx"
    model.write_bytes(b"model")

    class AssetDatabase:
        calls: list[str] = []

        @classmethod
        def get_path_from_guid(cls, value: str) -> str:
            cls.calls.append(value)
            return str(model)

    assert is_asset_guid_string(guid)
    assert resolve_disk_path_for_guid_string(AssetDatabase(), guid) == str(model)
    assert AssetDatabase.calls == [guid]
    assert AnimationClip3D.from_dict(_document(guid)).source_model_guid == guid


@pytest.mark.parametrize(
    "guid",
    [
        "A" * 32,
        "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
        "model-guid",
        "a" * 31,
    ],
)
def test_animation_clip_rejects_non_current_guid_forms(guid: str) -> None:
    assert not is_asset_guid_string(guid)
    with pytest.raises(ValueError, match="source_model_guid"):
        AnimationClip3D.from_dict(_document(guid))


def test_animation_clip_does_not_query_invalid_guid_forms() -> None:
    class AssetDatabase:
        @staticmethod
        def get_path_from_guid(_value: str) -> str:
            raise AssertionError("invalid GUID must not reach AssetDatabase")

    assert resolve_disk_path_for_guid_string(AssetDatabase(), "A" * 32) is None


def test_animation_clip_save_rejects_invalid_runtime_state(tmp_path) -> None:
    clip = AnimationClip3D(source_model_guid="model-guid")

    with pytest.raises(ValueError, match="source_model_guid"):
        clip.save(str(tmp_path / "invalid.animclip3d"))


def test_animation_clip_ignores_obsolete_fields_without_preserving_them() -> None:
    document = _document("a" * 32)
    document["source_model_path"] = "Assets/Models/obsolete.fbx"
    document["stable_clip_index"] = 3
    restored = AnimationClip3D.from_dict(document)
    assert not hasattr(restored, "source_model_path")
    assert "source_model_path" not in restored.to_dict()
    assert "stable_clip_index" not in restored.to_dict()


def test_embedded_take_descriptors_require_current_published_clip_table() -> None:
    current = [{
        "id": "source-57616c6b",
        "guid": "b" * 32,
        "name": "Walk",
        "duration": 1.0,
    }]
    assert embedded_take_descriptors({"model_animations": json.dumps(current)}) == current
    assert embedded_take_descriptors({
        "animation_names_csv": "Walk,Run",
        "animation_count": 2,
    }) == []


def test_animation_clip_imported_curves_events_and_mask_round_trip() -> None:
    from Infernux.core.animation_clip3d import ImportedFloatCurve
    from Infernux.core.animation_event import AnimationEvent

    clip = AnimationClip3D(
        curves=[ImportedFloatCurve("Speed", ((0.0, 2.0), (1.0, 6.0)))],
        events=[AnimationEvent(0.5, "step", "L", 1.0)],
        bone_mask=["Spine", "Arm"],
    )
    restored = AnimationClip3D.from_dict(clip.to_dict())

    assert restored.sample_curve("Speed", 0.25) == pytest.approx(3.0)
    assert restored.events[0].function == "step"
    assert restored.bone_mask == ["Spine", "Arm"]
