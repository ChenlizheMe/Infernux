"""Project directory clicks share selection projection without resource GUIDs."""
from types import SimpleNamespace

import pytest

from infernux.engine import _bootstrap_selection as bridge
from infernux.engine.interaction import (
    AssetMutation, AssetMutationKind, SelectionDomain, SelectionService,
    SelectionTarget,
)
from infernux.engine.path_utils import path_key
from infernux.lib import ProjectPanel


@pytest.fixture
def directory_selection(tmp_path, monkeypatch):
    folder = tmp_path / "Assets" / "Folder 中文"
    folder.mkdir(parents=True)
    asset = tmp_path / "Assets" / "File.txt"
    asset.write_text("test", encoding="utf-8")
    monkeypatch.setattr(bridge, "_selection_asset_database", lambda: SimpleNamespace(
        get_guid_from_path=lambda path: "file-guid" if path_key(path) == path_key(asset) else "",
        get_path_from_guid=lambda guid: str(asset) if guid == "file-guid" else "",
    ))
    return folder, asset


def test_directory_click_roundtrips_into_native_project_selection(directory_selection):
    folder, _asset = directory_selection
    selection = SelectionService()
    panel = ProjectPanel()
    panel.set_root_path(str(folder.parent.parent))
    bootstrap = bridge.BootstrapSelectionMixin()
    bootstrap._on_project_selection_changed((str(folder),), str(folder))
    target = selection.snapshot.primary
    assert target is not None
    assert target.domain is SelectionDomain.ASSET
    assert target.sub_kind == "directory"
    projected = bridge._project_path_for_target(target)
    assert path_key(projected) == path_key(folder)
    received = []
    panel.on_selection_changed = lambda paths, primary: received.append((paths, primary))
    panel.set_selected_files([projected], projected, True)
    assert len(received) == 1
    assert [path_key(path) for path in received[0][0]] == [path_key(folder)]
    assert path_key(received[0][1]) == path_key(folder)


def test_directory_and_resource_share_multiselection(directory_selection):
    folder, asset = directory_selection
    selection = SelectionService()
    bridge.BootstrapSelectionMixin()._on_project_selection_changed(
        (str(folder), str(asset)), str(folder),
    )
    assert len(selection.snapshot.targets) == 2
    assert selection.snapshot.primary.sub_kind == "directory"
    assert SelectionTarget.asset("file-guid") in selection.snapshot.targets


def test_missing_unregistered_file_is_not_a_directory_target(directory_selection):
    folder, _asset = directory_selection
    assert bridge._project_selection_target(str(folder / "missing.py")) is None


@pytest.mark.parametrize("kind", [AssetMutationKind.MOVED, AssetMutationKind.DELETED])
def test_directory_selection_tracks_parent_mutation(directory_selection, kind):
    folder, asset = directory_selection
    nested = folder / "Nested"
    nested.mkdir()
    selection = SelectionService()
    primary = SelectionTarget.project_directory(str(nested))
    resource = SelectionTarget.asset("file-guid")
    selection.replace((resource, primary), owner_id="project", primary=primary)
    changes = []
    selection.add_listener(changes.append)
    destination = folder.parent / "Renamed" if kind is AssetMutationKind.MOVED else ""
    mutation = AssetMutation(kind, str(folder), str(destination))
    bridge.BootstrapSelectionMixin()._on_asset_selection_source_changed(mutation)
    assert resource in selection.snapshot.targets
    if kind is AssetMutationKind.MOVED:
        assert selection.snapshot.primary == SelectionTarget.project_directory(str(destination / "Nested"))
    else:
        assert selection.snapshot.targets == (resource,)
    assert len(changes) == 1 and not changes[0].record_history


def test_empty_directory_move_and_delete_publish_selection_changes(tmp_path):
    from infernux.engine.interaction import AssetMutationService, DocumentRegistry
    from infernux.engine.ui.project_file_ops import move_paths_batch, delete_item

    folder = tmp_path / "Empty"
    folder.mkdir()
    destination = tmp_path / "Renamed"
    selection = SelectionService()
    bus = AssetMutationService(DocumentRegistry(), selection)
    bus.add_observer(bridge.BootstrapSelectionMixin()._on_asset_selection_source_changed)
    selection.select(SelectionTarget.project_directory(str(folder)), owner_id="project")
    database = SimpleNamespace(get_guid_from_path=lambda _path: "")
    assert move_paths_batch(((str(folder), str(destination)),), database) == (str(destination),)
    assert selection.snapshot.primary == SelectionTarget.project_directory(str(destination))
    assert move_paths_batch(((str(destination), str(folder)),), database) == (str(folder),)
    assert selection.snapshot.primary == SelectionTarget.project_directory(str(folder))
    assert delete_item(str(folder), database)
    assert selection.snapshot.is_empty
    assert not tuple(tmp_path.glob("*.meta"))


def test_project_commands_select_empty_directory_without_database(tmp_path):
    from infernux.engine.interaction.project_assets import ProjectAssetCommandService

    folder = tmp_path / "Empty"
    folder.mkdir()
    selection = SelectionService()
    ProjectAssetCommandService(selection)._select_project_paths((str(folder),))
    assert selection.snapshot.primary == SelectionTarget.project_directory(str(folder))


def test_directory_selection_drives_project_keyboard_commands(tmp_path):
    from infernux.engine.interaction import CommandContext, CommandSource, FocusService
    from infernux.engine.ui.core_panel_interactions import project_panel_interaction

    folder = tmp_path / "Folder"
    folder.mkdir()
    selection = SelectionService()
    selection.select(SelectionTarget.project_directory(str(folder)), owner_id="project")
    context = CommandContext(CommandSource.SHORTCUT, FocusService().snapshot, selection.snapshot)
    calls = []
    interactions = SimpleNamespace(
        can_copy=lambda paths: bool(paths),
        copy=lambda paths, *, cut: calls.append(("copy", tuple(paths), cut)) or True,
        request_delete=lambda paths, *, origin: calls.append(("delete", tuple(paths))) or True,
    )
    navigation = SimpleNamespace(
        locate=lambda target, **kwargs: calls.append(("locate", target)) or True,
    )
    panel = SimpleNamespace(
        begin_rename_selected_asset=lambda path: calls.append(("rename", path)) or True,
        can_rename_selected_asset=lambda path: bool(path),
        can_navigate_to_path=lambda path: bool(path),
        get_current_path=lambda: str(tmp_path),
        set_current_path=lambda path: True,
        get_folder_expanded_paths=lambda: [],
        set_folder_expanded_paths=lambda paths: None,
        get_model_expanded_paths=lambda: [],
        set_model_expanded_paths=lambda paths: None,
        request_search_focus=lambda: None,
    )
    descriptor = project_panel_interaction(interactions, navigation, object())
    adapter = descriptor.adapter_factory(panel)
    for command_id in ("edit.copy", "edit.cut", "edit.delete", "edit.rename", "project.locate_asset"):
        handler = adapter.handler(command_id)
        assert handler.can_execute(context)
        assert handler.execute(context)
    path = path_key(folder)
    assert calls == [
        ("copy", (path,), False), ("copy", (path,), True),
        ("delete", (path,)), ("rename", path),
        ("locate", SelectionTarget.project_directory(str(folder))),
    ]
