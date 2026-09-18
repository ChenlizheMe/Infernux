"""Imported animation clips share Apply, stable references and binary playback."""
import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from Infernux.core.animation_clip3d import AnimationClip3D, embedded_take_descriptors
from Infernux.core.asset_types import MeshImportSettings, read_mesh_import_settings, read_meta_file
from Infernux.core.assets import AssetManager
from Infernux.lib import AssetRegistry


def spec(name="Middle", identity="a" * 32):
    return {"id": identity, "name": name, "source_take": "Take", "start": .25, "end": .75}


@pytest.mark.parametrize("change", [
    {"id": "bad"}, {"name": ""}, {"source_take": ""}, {"start": -1},
    {"end": .1}, {"end": float("inf")}, {"start": True}, {"other": 1},
])
def test_invalid_clip_specs_are_not_accepted(change):
    settings = MeshImportSettings().to_dict()
    settings["animation_clips"] = [spec() | change]
    with pytest.raises(ValueError, match="animation_clips"):
        MeshImportSettings.from_dict(settings)


def test_clip_settings_copy_is_detached_and_old_metadata_keeps_source_mode():
    settings = MeshImportSettings(custom_animation_clips=True, animation_clips=[spec()])
    clone = settings.copy()
    clone.animation_clips[0]["name"] = "Other"
    assert settings.animation_clips[0]["name"] == "Middle"
    document = settings.to_dict()
    del document["custom_animation_clips"], document["animation_clips"]
    restored = MeshImportSettings.from_dict(document)
    assert not restored.custom_animation_clips and restored.animation_clips == []


@pytest.mark.parametrize("durations", [[0], [0, 2], [3]])
def test_add_clip_uses_a_nonzero_duration_source(monkeypatch, durations):
    from Infernux.engine.ui import asset_details_renderer as renderer

    sources = [{"name": f"Take {i}", "duration": duration} for i, duration in enumerate(durations)]
    settings = MeshImportSettings(custom_animation_clips=True)
    state = SimpleNamespace(settings=settings, meta={"source_animations": json.dumps(sources)})
    buttons = []

    def button(label):
        buttons.append(label)
        return True

    ctx = SimpleNamespace(text_wrapped=lambda _: None, button=button,
                          record_semantic_item=lambda *_: None)
    monkeypatch.setattr(renderer, "_edit_import_settings", lambda state, key, edit, label: edit(state.settings))
    renderer._render_model_animation_clips(ctx, state)
    timed_sources = [source for source in sources if source["duration"] > 0]
    assert len(settings.animation_clips) == len(buttons) == bool(timed_sources)
    if timed_sources:
        assert settings.animation_clips[0]["source_take"] == timed_sources[0]["name"]
        assert settings.animation_clips[0]["end"] == timed_sources[0]["duration"]
        assert MeshImportSettings.from_dict(settings.to_dict()) == settings


@pytest.fixture
def model(engine, tmp_path, monkeypatch):
    database = engine.get_asset_database()
    monkeypatch.setattr(AssetManager, "_engine", engine)
    monkeypatch.setattr(AssetManager, "_asset_database", database)
    source = Path(database.assets_root) / tmp_path.name / "Clips.fbx"
    source.parent.mkdir()
    source.write_bytes((Path(__file__).resolve().parents[2] /
                        "external/assimp/test/models/FBX/animation_with_skeleton.fbx").read_bytes())
    imported = AssetManager.import_asset(str(source), database=database)
    assert imported, imported.error
    yield database, source, imported.guid
    AssetRegistry.instance().invalidate_asset(imported.guid)
    database.delete_asset(str(source))


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_clip_apply_rename_reorder_and_failure_are_identity_safe(model, mode):
    database, source, guid = model
    registry = AssetRegistry.instance()
    settings = read_mesh_import_settings(str(source))
    metadata = read_meta_file(str(source))
    source_inventory = metadata["source_animations"]
    take = json.loads(source_inventory)[0]
    seconds = take["duration"]
    first = spec() | {"source_take": take["name"], "start": seconds * .25, "end": seconds * .75}
    second = spec("Opening", "b" * 32) | {"source_take": take["name"], "start": 0, "end": seconds * .25}
    source_clip = AnimationClip3D.from_embedded_take_virtual_path(f"{source}::subanim:0")
    assert source_clip and source_clip.name == take["name"]
    settings.custom_animation_clips = True
    settings.animation_clips = [first, second]

    def apply():
        if mode == "sync":
            return AssetManager.reimport_asset(str(source), import_settings=settings.to_dict(), database=database)
        owner = AssetManager.begin_model_reimport(str(source), settings)
        until = time.monotonic() + 30
        while (result := AssetManager.poll_model_reimport(owner)) is None:
            assert time.monotonic() < until
            time.sleep(.002)
        return result

    result = apply()
    assert result, result.error
    virtual = f"{source}::subanim:{first['id']}"
    clip = AnimationClip3D.from_embedded_take_virtual_path(virtual)
    assert clip and clip.name == "Middle" and clip.take_name == first["id"]
    assert clip.duration_hint == pytest.approx(seconds * .5)
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


def test_stable_source_clip_reference_survives_inventory_reordering(tmp_path, monkeypatch):
    from Infernux.core import asset_types
    source = tmp_path / "clips.gltf"
    source.write_text("{}")
    metadata = {"guid": "c" * 32, "animation_names_csv": "Second,First",
                "model_animations": json.dumps([
                    {"id": "source-" + name.encode().hex(), "name": name, "duration": 1}
                    for name in ("Second", "First")])}
    monkeypatch.setattr(asset_types, "read_meta_file", lambda _: metadata)
    monkeypatch.setattr(asset_types, "read_meta_guid", lambda _: "c" * 32)
    reference = str(source) + "::subanim:source-" + b"First".hex()
    clip = AnimationClip3D.from_embedded_take_virtual_path(reference)
    assert clip.name == "First"
    metadata["model_animations"] = json.dumps(list(reversed(json.loads(metadata["model_animations"]))))
    assert AnimationClip3D.from_embedded_take_virtual_path(reference).take_name == clip.take_name
