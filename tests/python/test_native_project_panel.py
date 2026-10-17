"""Project browsing, selection, and imported asset authoring workflows."""

import json
from pathlib import Path

import pytest

from infernux.lib import InxMaterial, ProjectPanel
from infernux.engine.ui.project_file_ops import (
    SCRIPT_TEMPLATE, create_material, create_physic_material,
    create_prefab_from_gameobject, create_render_effect, create_render_effect_group,
    do_rename,
)
from infernux.renderstack.render_effect_asset import (
    RenderEffectAsset, RenderEffectGroupAsset, parse_render_effect_document,
)


def test_new_script_template_creates_a_component():
    namespace = {}
    exec(SCRIPT_TEMPLATE.format(class_name="ExampleComponent"), namespace)
    assert issubclass(namespace["ExampleComponent"], namespace["inx"].InxComponent)


def test_explicit_rename_target_replaces_empty_or_stale_selection():
    panel = ProjectPanel()
    target = "C:/Project/Assets/NewMaterial.mat"
    assert panel.can_rename_selected_asset(target)
    assert panel.begin_rename_selected_asset(target)
    panel.set_selected_file("C:/Project/Assets/Previous.mat", False)
    # A stale, nonexistent path must not become an authoritative selection.
    assert not panel.has_selected_assets()
    assert panel.can_rename_selected_asset(target)
    assert panel.begin_rename_selected_asset(target)


def test_asset_creation_writes_current_documents_and_imports_guids(engine, tmp_path):
    database = engine.get_asset_database()
    directory = Path(database.assets_root) / tmp_path.name
    directory.mkdir()
    result = create_material(str(directory), "NewMaterial", database)
    ok, error = result
    assert ok, error
    material_path = directory / "NewMaterial.mat"
    document = json.loads(material_path.read_text(encoding="utf-8"))
    assert document["shaders"]["vertex"]["shader_id"] == "Standard"
    assert document["shaders"]["fragment"]["shader_id"] == "Lit"
    assert document["name"] == "NewMaterial" and document["builtin"] is False
    material = InxMaterial()
    assert material.deserialize_document(document)
    assert material.name == "NewMaterial"
    assert database.get_guid_from_path(str(material_path))

    ok, error = create_physic_material(str(directory), "Ice", database)
    assert ok, error
    physic_path = directory / "Ice.physicMaterial"
    assert json.loads(physic_path.read_text(encoding="utf-8")) == pytest.approx({
        "friction": 0.6, "bounciness": 0.0, "friction_combine": 0, "bounce_combine": 0,
    })
    assert database.get_guid_from_path(str(physic_path))

    ok, error = create_render_effect(
        str(directory), "Pixels", "infernux.route.pixelation", database
    )
    assert ok, error
    effect_path = directory / "Pixels.effect"
    effect = parse_render_effect_document(effect_path.read_text(encoding="utf-8"))
    assert isinstance(effect, RenderEffectAsset)
    assert effect.feature_type == "infernux.route.pixelation" and effect.parameters == {}
    assert database.get_guid_from_path(str(effect_path))
    ok, error = create_render_effect_group(str(directory), "Post", database)
    assert ok, error
    group_path = directory / "Post.effectgroup"
    group = parse_render_effect_document(group_path.read_text(encoding="utf-8"))
    assert isinstance(group, RenderEffectGroupAsset) and group.entries == ()
    assert database.get_guid_from_path(str(group_path))


def test_create_prefab_links_source_to_the_imported_saved_asset(engine, scene, tmp_path):
    database = engine.get_asset_database()
    directory = Path(database.assets_root) / tmp_path.name
    directory.mkdir()
    source = scene.create_game_object("CheckpointGate")
    result = create_prefab_from_gameobject(source, str(directory), database)
    ok, path = result
    assert ok
    assert path == str(directory / "CheckpointGate.prefab") == result.created_path
    assert Path(path).is_file()
    assert source.prefab_guid == database.get_guid_from_path(path)


def test_browsing_stays_in_worktree_and_selection_does_not_navigate(tmp_path):
    first = tmp_path / "Assets" / "First"
    second = tmp_path / "Assets" / "Second"
    library = tmp_path / "Library" / "Resources" / "shaders"
    for directory in (first, second, library):
        directory.mkdir(parents=True)
    asset = second / "Selected.mat"
    asset.write_text("{}", encoding="utf-8")
    panel = ProjectPanel()
    assert not panel.set_current_path("")
    assert panel.get_current_path() == ""
    panel.set_root_path(str(tmp_path))
    for directory in (tmp_path, first.parent, library, first):
        assert panel.can_navigate_to_path(str(directory))
        assert panel.set_current_path(str(directory))
        assert panel.get_current_path() == str(directory)
    assert not panel.can_navigate_to_path(str(tmp_path.parent))
    assert not panel.set_current_path(str(tmp_path.parent))
    panel.set_selected_file(str(asset), False)
    assert panel.get_current_path() == str(first)


def test_external_drop_is_forwarded_as_one_global_import_command(tmp_path):
    assets = tmp_path / "Assets"
    assets.mkdir()
    panel = ProjectPanel()
    panel.set_root_path(str(tmp_path))
    panel.set_current_path(str(assets))
    calls = []
    panel.execute_command = lambda command, source, argument: calls.append(
        (command, source, json.loads(argument))
    ) or True
    panel.receive_dropped_files(["C:/External/One.png", "C:/External/Two.png"])
    assert calls == [("asset.import_external", "drag_drop", {
        "paths": ["C:/External/One.png", "C:/External/Two.png"],
        "destination": str(assets),
    })]


def test_multi_selection_publishes_one_snapshot_and_replay_is_silent(tmp_path):
    first, second = tmp_path / "First.mat", tmp_path / "Second.mat"
    first.write_text("{}", encoding="utf-8")
    second.write_text("{}", encoding="utf-8")
    panel = ProjectPanel()
    panel.set_root_path(str(tmp_path))
    received = []
    panel.on_selection_changed = lambda paths, primary: received.append((list(paths), primary))
    panel.set_selected_files([str(first), str(second)], str(first))
    assert panel.has_selected_assets() and not panel.can_rename_selected_asset()
    panel.clear_selection(False)
    panel.set_selected_file(str(second), False)
    assert panel.can_rename_selected_asset()
    assert received == [([str(first), str(second)], str(first))]


def test_folder_rename_preserves_nested_asset_bytes_and_guids(engine, tmp_path):
    database = engine.get_asset_database()
    source = Path(database.assets_root) / tmp_path.name / "OldFolder"
    nested = source / "Nested"
    nested.mkdir(parents=True)
    ok, error = create_material(str(source), "Root", database)
    assert ok, error
    ok, error = create_material(str(nested), "Child", database)
    assert ok, error
    relative_paths = (Path("Root.mat"), Path("Nested/Child.mat"))
    originals = {
        path: ((source / path).read_bytes(), database.get_guid_from_path(str(source / path)))
        for path in relative_paths
    }
    destination = Path(do_rename(str(source), "RenamedFolder", database))
    assert destination == source.with_name("RenamedFolder").resolve()
    assert not source.exists()
    for relative, (content, guid) in originals.items():
        path = destination / relative
        assert path.read_bytes() == content
        assert guid and database.get_guid_from_path(str(path)) == guid
