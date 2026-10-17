"""Imported animation clips share Apply, stable references and binary playback."""

import json
import time
from pathlib import Path

import pytest

from infernux.core.animation_clip3d import AnimationClip3D, embedded_take_descriptors
from infernux.core.asset_types import read_mesh_import_settings, read_meta_file
from infernux.core.assets import AssetManager
from infernux.lib import AssetRegistry
from model_test_support import remove_model_test_folder


def spec(name="Middle", identity="a" * 32):
    return {"id": identity, "name": name, "source_take": "Take", "start": 0.25, "end": 0.75}


@pytest.fixture
def model(engine, tmp_path, monkeypatch):
    database = engine.get_asset_database()
    monkeypatch.setattr(AssetManager, "_engine", engine)
    monkeypatch.setattr(AssetManager, "_asset_database", database)
    source = Path(database.assets_root) / tmp_path.name / "Clips.fbx"
    source.parent.mkdir()
    source.write_bytes(
        (
            Path(__file__).resolve().parents[2] / "external/assimp/test/models/FBX/animation_with_skeleton.fbx"
        ).read_bytes()
    )
    imported = AssetManager.import_asset(str(source), database=database)
    assert imported, imported.error
    try:
        yield database, source, imported.guid
    finally:
        remove_model_test_folder(database, source.parent)


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_clip_apply_rename_reorder_and_failure_are_identity_safe(model, mode):
    database, source, guid = model
    registry = AssetRegistry.instance()
    settings = read_mesh_import_settings(str(source))
    metadata = read_meta_file(str(source))
    source_inventory = metadata["source_animations"]
    take = json.loads(source_inventory)[0]
    seconds = take["duration"]
    first = spec() | {"source_take": take["name"], "start": seconds * 0.25, "end": seconds * 0.75}
    second = spec("Opening", "b" * 32) | {"source_take": take["name"], "start": 0, "end": seconds * 0.25}
    source_descriptor = embedded_take_descriptors(metadata)[0]
    source_clip = AnimationClip3D.from_embedded_take_virtual_path(f"{source}::subanim:{source_descriptor['id']}")
    assert source_clip and source_clip.name == take["name"]
    assert AnimationClip3D.from_embedded_take_virtual_path(f"{source}::subanim:0") is None
    settings.custom_animation_clips = True
    settings.animation_clips = [first, second]

    def apply():
        if mode == "sync":
            return AssetManager.reimport_asset(str(source), import_settings=settings.to_dict(), database=database)
        owner = AssetManager.begin_model_reimport(str(source), settings)
        until = time.monotonic() + 30
        while (result := AssetManager.poll_model_reimport(owner)) is None:
            assert time.monotonic() < until
            time.sleep(0.002)
        return result

    result = apply()
    assert result, result.error
    virtual = f"{source}::subanim:{first['id']}"
    clip = AnimationClip3D.from_embedded_take_virtual_path(virtual)
    assert clip and clip.name == "Middle" and clip.take_name == first["id"]
    assert clip.duration_hint == pytest.approx(seconds * 0.5)
    assert clip.source_model_guid == guid
    assert AnimationClip3D.from_embedded_take_virtual_path(f"{source}::subanim:0") is None
    registry.invalidate_asset(guid)
    mesh = registry.load_mesh(str(source))
    assert mesh.skinned_animation_names == ["Middle", "Opening"]
    settings.animation_clips.reverse()
    settings.animation_clips[1]["name"] = "Renamed"
    result = apply()
    assert result, result.error
    clip = AnimationClip3D.from_embedded_take_virtual_path(virtual)
    assert clip.name == "Renamed" and clip.take_name == first["id"]
    assert read_meta_file(str(source))["source_animations"] == source_inventory
    before = Path(str(source) + ".meta").read_bytes()
    settings.animation_clips[0]["end"] = seconds + 1
    result = apply()
    assert not result and "exceeds source take duration" in result.error
    assert Path(str(source) + ".meta").read_bytes() == before
    assert mesh.skinned_animation_names == ["Opening", "Renamed"]
    assert AssetManager.reimport_asset(str(source), database=database)
    assert AnimationClip3D.from_embedded_take_virtual_path(virtual).take_name == first["id"]
    settings.animation_clips = []
    assert apply()
    assert registry.load_mesh(str(source)).skinned_animation_names == []
    assert embedded_take_descriptors(read_meta_file(str(source))) == []


def test_clip_extras_reject_missing_clip_without_partial_publication(model):
    database, source, _guid = model
    before = Path(str(source) + ".meta").read_bytes()
    settings = read_mesh_import_settings(str(source))
    settings.animation_clip_extras = [
        {
            "clip_id": "missing",
            "curves": [],
            "events": [],
            "bone_mask": [],
        }
    ]

    result = AssetManager.reimport_asset(str(source), import_settings=settings.to_dict(), database=database)

    assert not result and "missing clip" in result.error
    assert Path(str(source) + ".meta").read_bytes() == before


def test_model_clip_import_publishes_playback_settings_curves_events_and_mask(model):
    database, source, _guid = model
    metadata = read_meta_file(str(source))
    descriptor = embedded_take_descriptors(metadata)[0]
    bone = str(metadata["bone_names_csv"]).split(",")[0].strip()
    assert bone
    settings = read_mesh_import_settings(str(source))
    settings.animation_loop_time = False
    settings.animation_apply_root_motion = True
    settings.animation_reference_pose = "first_frame"
    settings.animation_clip_extras = [
        {
            "clip_id": descriptor["id"],
            "curves": [
                {
                    "name": "FootPlant",
                    "keys": [{"time_normalized": 0.0, "value": 0.0}, {"time_normalized": 1.0, "value": 1.0}],
                }
            ],
            "events": [{"time_normalized": 0.5, "function": "footstep", "string_arg": "L", "number_arg": 2.0}],
            "bone_mask": [bone],
        }
    ]
    result = AssetManager.reimport_asset(str(source), import_settings=settings.to_dict(), database=database)
    assert result, result.error
    published = embedded_take_descriptors(read_meta_file(str(source)))[0]
    clip = AnimationClip3D.from_embedded_take_virtual_path(f"{source}::subanim:{published['id']}")
    assert clip is not None
    assert clip.default_loop is published["default_loop"] is False
    assert clip.apply_root_motion is published["apply_root_motion"] is True
    assert clip.reference_pose == published["reference_pose"] == "first_frame"
    assert clip.sample_curve("FootPlant", 0.25) == pytest.approx(0.25)
    assert clip.events[0].to_dict() == {
        "time_normalized": 0.5,
        "function": "footstep",
        "string_arg": "L",
        "number_arg": 2.0,
    }
    assert clip.bone_mask == published["bone_mask"] == [bone]
    assert published["curves"][0]["name"] == "FootPlant" and published["events"][0]["function"] == "footstep"
