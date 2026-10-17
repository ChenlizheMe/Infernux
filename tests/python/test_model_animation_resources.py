"""Model clip GUIDs connect the Project picker, FSM and source-free cooked documents."""

from test_model_animation_clips import model as model
import json
from pathlib import Path

import pytest

from infernux.components.skeletal_animator import SkeletalAnimator
from infernux.components.fields import get_raw_field_value
from infernux.core.anim_state_machine import AnimClipEntry
from infernux.core.animation_clip3d import AnimationClip3D
from infernux.core.asset_ref import AnimationClip3DRef
from infernux.core.asset_types import read_meta_file, read_mesh_import_settings
from infernux.core.assets import AssetManager
from infernux.engine.game_builder import GameBuilder
from infernux.engine.runtime_artifact_catalog import load_asset_index, source_fingerprint, build_catalog
from infernux.engine.ui.animfsm_editor_panel import AnimFSMEditorPanel
from infernux.lib import AssetDependencyGraph
from test_model_animation_clips import spec


def output(source):
    return json.loads(read_meta_file(str(source))["model_animations"])[0]


def test_imported_clip_guid_picker_fsm_and_cook(model, tmp_path, monkeypatch, scene):
    database, source, model_guid = model
    record = output(source)
    guid = record["guid"]
    virtual = str(source) + "::subanim:" + record["id"]
    assert guid != model_guid and database.get_guid_from_path(virtual) == guid
    assert model_guid in AssetDependencyGraph.instance().get_dependencies(guid)
    assert not Path(virtual).exists()
    from infernux.engine.interaction import AssetReferenceCatalog
    monkeypatch.setattr("infernux.engine.project_context.get_project_root", lambda: str(Path(database.assets_root).parent))
    catalog = AssetReferenceCatalog()
    assert any(database.get_guid_from_path(path) == guid for _name, path in catalog.items("AnimationClip3D", ""))

    panel = AnimFSMEditorPanel()
    panel._fsm.mode = "3d"
    state = panel._fsm.add_state("Imported")
    panel._assign_clip_to_state(state, virtual, record_undo=False)
    assert state.clip_guid == guid
    item = panel._group_item_for_path(virtual)
    assert get_raw_field_value(item, "clip").guid == guid
    from infernux.engine.ui.animfsm_editor_panel import _ClipGroupItem
    from infernux.engine.ui.inspector_utils import serialized_field_label
    from infernux.components.fields import get_serialized_fields
    from infernux.engine.i18n import t
    assert get_raw_field_value(_ClipGroupItem(), "clip").guid == ""
    assert serialized_field_label("weight", get_serialized_fields(_ClipGroupItem)["weight"]) == t("animfsm_editor.clip_weight")
    state.clips = [AnimClipEntry(guid)]
    animator = SkeletalAnimator()
    animator._clip_cache = {}
    def path_loader(*_args, **_kwargs):
        raise AssertionError("published clip GUID loading must not re-enter path/sidecar loading")

    AssetManager.invalidate(guid)
    with monkeypatch.context() as loading:
        loading.setattr(AnimationClip3D, "load", path_loader)
        loading.setattr("infernux.core.asset_types.read_asset_metadata", path_loader)
        clip = animator._resolve_clip(state)
        assert animator._resolve_group_clips(state) == [clip]
        assert AnimationClip3DRef(guid=guid).resolve() is clip
        assert AssetManager.load_by_guid(guid) is clip
        assert AnimationClip3D.from_embedded_take_virtual_path(virtual) is clip
        assert AnimationClip3D.from_embedded_take_virtual_path(model_guid + "::subanim:" + record["id"]) is clip
        assert AnimationClip3D.from_embedded_take_virtual_path(virtual + "-missing") is None
    assert clip.take_name == record["id"] and clip.source_model_guid == model_guid

    render_source = source.with_name("Render.fbx")
    render_source.write_bytes(source.read_bytes())
    imported_render = AssetManager.import_asset(str(render_source), database=database)
    assert imported_render, imported_render.error
    renderer = scene.create_game_object("Cross-model clip").add_component("SkinnedMeshRenderer")
    renderer.set_source_model_guid(imported_render.guid)
    assert renderer.get_animation_duration_seconds(clip.take_name, model_guid) == pytest.approx(record["duration"])
    renderer.submit_animation_pose(
        clip.take_name, 0.25, 0.5, animation_source_guid=model_guid,
    )
    assert renderer.animation_source_guid == model_guid
    renderer.submit_pose_stack([{"take_name": clip.take_name, "source_model_guid": model_guid, "time": 0.25}])
    assert renderer.active_take_name == clip.take_name
    assert not AssetManager.reimport_asset(virtual, database=database)

    database.flush_derived_index()
    root = Path(database.assets_root).parent
    entries = {entry["guid"]: entry for entry in load_asset_index(root)}
    entry = entries[guid]
    assert entry["read_only"] and entry["dependencies"] == [model_guid]
    assert source_fingerprint(root, entry)["content_hash"] == entry["content_hash"]
    builder = GameBuilder(str(root), str(tmp_path / "Player"))
    builder._cooked_asset_entries = {guid: entry}
    builder._runtime_artifact_bindings = {}
    builder._runtime_artifact_source_paths = set()
    data = tmp_path / "Player" / "Data"
    builder.freeze_asset_index_entries(list(entries.values()))
    monkeypatch.setattr(builder, "_collect_library_asset_entries", lambda _, **kwargs: {guid: entry})
    builder._copy_cooked_assets(str(data))
    assert not (data / "Assets").exists()
    builder._stage_library_runtime_documents(str(data))
    runtime = f"Library/Artifacts/Document/{guid}.inxdoc"
    cooked = AnimationClip3D.load(str(data / runtime))
    assert cooked.take_name == record["id"] and cooked.source_model_guid == model_guid
    assert not hasattr(cooked, "source_model_path")
    assert not (data / "Assets").exists()
    catalog = build_catalog(
        [
            {
                "package": "Content.inxpkg",
                "runtime_path": runtime,
                "bytes": (data / runtime).stat().st_size,
                "payload": (data / runtime).read_bytes(),
                "asset_binding": builder._runtime_artifact_bindings[runtime],
            },
            {
                "package": "Content.inxpkg",
                "runtime_path": f"Library/Artifacts/Mesh/{model_guid}.inxmesh",
                "bytes": 4,
                "asset_binding": {"source_guid": model_guid, "source_path": "Assets/Clips.fbx", "dependencies": []},
            },
        ],
        player_host={"executable": "Game.exe"},
        package_records=[],
    )
    by_path = {item["runtime_path"]: item for item in catalog["artifacts"]}
    assert by_path[runtime]["dependencies"] == [
        by_path[f"Library/Artifacts/Mesh/{model_guid}.inxmesh"]["runtime_artifact_id"]
    ]
    assert by_path[runtime]["unresolved_dependencies"] == []
    builder._cooked_asset_entries[model_guid] = entries[model_guid]
    mesh_runtime = f"Library/Artifacts/Mesh/{model_guid}.inxmesh"
    builder._runtime_artifact_bindings[mesh_runtime] = {
        "source_guid": model_guid,
        "dependencies": [],
    }
    builder._write_runtime_asset_records(str(data.parent))
    runtime_records = json.loads((data / "Library/RuntimeAssetRecords.json").read_text(encoding="utf-8"))
    model_record = next(item for item in runtime_records["entries"] if item["guid"] == model_guid)
    assert "model_animations" not in model_record["metadata"]["metadata"]
    assert "model_textures" not in model_record["metadata"]["metadata"]
    assert "model_animation_identities" not in model_record["metadata"]["metadata"]
    # Cooking must not mutate the frozen authoring catalog.
    assert "model_animations" in entries[model_guid]["metadata"]["metadata"]


def test_owned_clip_identity_survives_rename_mode_toggle_refresh_and_move(model):
    database, source, _ = model
    original = output(source)
    settings = read_mesh_import_settings(str(source))
    settings.custom_animation_clips = True
    settings.animation_clips = [spec() | {"source_take": original["name"], "start": 0, "end": original["duration"]}]
    assert AssetManager.reimport_asset(str(source), import_settings=settings.to_dict(), database=database)
    first = output(source)
    reference = AnimationClip3DRef(guid=first["guid"])
    original_clip = reference.resolve()
    assert original_clip.name == "Middle"
    settings.animation_clips[0]["name"] = "Renamed"
    assert AssetManager.reimport_asset(str(source), import_settings=settings.to_dict(), database=database)
    assert output(source)["guid"] == first["guid"]
    renamed_clip = reference.resolve()
    assert renamed_clip is not original_clip and renamed_clip.name == "Renamed"
    assert original_clip.name == "Middle"
    settings.custom_animation_clips = False
    assert AssetManager.reimport_asset(str(source), import_settings=settings.to_dict(), database=database)
    assert not database.get_path_from_guid(first["guid"])
    assert reference.resolve() is None
    assert output(source)["guid"] == original["guid"]
    settings.custom_animation_clips = True
    assert AssetManager.reimport_asset(str(source), import_settings=settings.to_dict(), database=database)
    assert output(source)["guid"] == first["guid"]
    database.refresh()
    assert reference.resolve().name == "Renamed"
    moved = source.with_name("Moved.fbx")
    source.rename(moved)
    try:
        assert database.move_asset(str(source), str(moved))
        owner, marker, identity = database.get_path_from_guid(first["guid"]).partition("::subanim:")
        assert Path(owner).samefile(moved)
        assert marker and identity == first["id"]
        assert reference.resolve().name == "Renamed"
    finally:
        moved.rename(source)
        assert database.move_asset(str(moved), str(source))
