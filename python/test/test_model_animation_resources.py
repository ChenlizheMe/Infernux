"""Model clip GUIDs connect the Project picker, FSM and source-free cooked documents."""
import json
from pathlib import Path

from Infernux.components.skeletal_animator import SkeletalAnimator
from Infernux.core.animation_clip3d import AnimationClip3D
from Infernux.core.asset_ref import AnimationClip3DRef
from Infernux.core.asset_types import read_meta_file, read_mesh_import_settings
from Infernux.core.assets import AssetManager
from Infernux.engine.game_builder import GameBuilder
from Infernux.engine.runtime_artifact_catalog import load_asset_index, source_fingerprint, build_catalog
from Infernux.engine.ui.animfsm_editor_panel import AnimFSMEditorPanel
from Infernux.lib import AssetDependencyGraph
from test_model_animation_clips import model, spec


def output(source):
    return json.loads(read_meta_file(str(source))["model_animations"])[0]


def test_imported_clip_guid_picker_fsm_and_cook(model, tmp_path, monkeypatch):
    database, source, model_guid = model
    record = output(source)
    guid = record["guid"]
    virtual = str(source) + "::subanim:" + record["id"]
    assert guid != model_guid and database.get_guid_from_path(virtual) == guid
    assert model_guid in AssetDependencyGraph.instance().get_dependencies(guid)
    assert not Path(virtual).exists()

    panel = AnimFSMEditorPanel()
    panel._fsm.mode = "3d"
    state = panel._fsm.add_state("Imported")
    panel._assign_clip_to_state(state, virtual, record_undo=False)
    assert state.clip_guid == guid
    panel._assign_clip_b_to_state(state, virtual, record_undo=False)
    assert state.clip_b_guid == guid
    animator = SkeletalAnimator()
    animator._clip_cache = {}
    clip = animator._resolve_clip(state)
    assert clip.take_name == record["id"] and clip.source_model_guid == model_guid
    assert animator._resolve_clip_b(state).to_dict() == clip.to_dict()
    assert AnimationClip3DRef(guid=guid).resolve().to_dict() == clip.to_dict()
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
    monkeypatch.setattr(builder, "_collect_library_asset_entries", lambda _: {guid: entry})
    builder._copy_cooked_assets(str(data))
    assert not (data / "Assets").exists()
    builder._stage_library_runtime_documents(str(data))
    runtime = f"Library/Artifacts/Document/{guid}.animclip3d"
    cooked = AnimationClip3D.load(str(data / runtime))
    assert cooked.take_name == record["id"] and cooked.source_model_guid == model_guid
    assert not hasattr(cooked, "source_model_path")
    assert not (data / "Assets").exists()
    catalog = build_catalog([
        {"package": "Content.inxpkg", "runtime_path": runtime,
         "bytes": (data / runtime).stat().st_size, "payload": (data / runtime).read_bytes(),
         "asset_binding": builder._runtime_artifact_bindings[runtime]},
        {"package": "Content.inxpkg", "runtime_path": f"Library/Artifacts/Mesh/{model_guid}.inxmesh",
         "bytes": 4, "asset_binding": {"source_guid": model_guid,
             "source_path": "Assets/Clips.fbx", "dependencies": []}},
    ], player_host={"executable": "Game.exe"}, package_records=[])
    by_path = {item["runtime_path"]: item for item in catalog["artifacts"]}
    assert by_path[runtime]["dependencies"] == [by_path[f"Library/Artifacts/Mesh/{model_guid}.inxmesh"]["runtime_artifact_id"]]
    assert by_path[runtime]["unresolved_dependencies"] == []
    builder._cooked_asset_entries[model_guid] = entries[model_guid]
    mesh_runtime = f"Library/Artifacts/Mesh/{model_guid}.inxmesh"
    builder._runtime_artifact_bindings[mesh_runtime] = {
        "source_guid": model_guid, "dependencies": [],
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
    settings.animation_clips = [spec() | {"source_take": original["name"],
                                          "start": 0, "end": original["duration"]}]
    assert AssetManager.reimport_asset(str(source), import_settings=settings.to_dict(), database=database)
    first = output(source)
    reference = AnimationClip3DRef(guid=first["guid"])
    assert reference.resolve().name == "Middle"
    settings.animation_clips[0]["name"] = "Renamed"
    assert AssetManager.reimport_asset(str(source), import_settings=settings.to_dict(), database=database)
    assert output(source)["guid"] == first["guid"]
    assert reference.resolve().name == "Renamed"
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
        assert database.get_path_from_guid(first["guid"]).replace("\\", "/").startswith(moved.as_posix())
        assert reference.resolve().name == "Renamed"
    finally:
        moved.rename(source)
        assert database.move_asset(str(moved), str(source))


def test_clip_picker_returns_owned_guid(model, monkeypatch):
    _, source, _ = model
    from Infernux.engine.interaction import asset_reference_catalog
    monkeypatch.setattr(asset_reference_catalog, "items", lambda *_: [("Model", str(source))])
    _, value = AnimFSMEditorPanel._embedded_clip3d_picker_items("")[0]
    assert value["guid"] == output(source)["guid"]
    assert value["path_hint"].startswith(str(source) + "::subanim:")


def test_clip_inspector_uses_child_identity_not_parent_model(model):
    database, source, model_guid = model
    from Infernux.engine.ui import asset_details_renderer as ui
    record = output(source)
    ui._ensure_categories()
    state = ui._State()
    path = database.get_path_from_guid(record["guid"])
    assert state.load(path, "animclip3d", ui._categories["animclip3d"])
    assert state.meta["guid"] == record["guid"] != model_guid
    assert state.meta["resource_name"] == record["name"]
    assert state.settings.take_name == record["id"]
