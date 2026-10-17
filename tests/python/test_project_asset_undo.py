from __future__ import annotations

import os
import shutil
import wave

import pytest

from infernux.lib import InxMaterial
from infernux.engine.asset_creation import AssetCreationResult
from infernux.engine.interaction import EditorActionJournal
from infernux.engine.undo import (
    ProjectAssetCreateCommand,
    ProjectPrefabCreateCommand,
    ProjectAssetCopyCommand,
    ProjectAssetDeleteCommand,
    ProjectAssetMoveCommand,
    ProjectAssetPasteCommand,
    ProjectAssetRenameCommand,
    UndoManager,
)


def _filesystem_move(source: str, destination: str, _database):
    os.replace(source, destination)
    return destination


def _filesystem_delete(path: str, _database):
    if os.path.isdir(path):
        shutil.rmtree(path)
    else:
        os.remove(path)
    meta = path + ".meta"
    if os.path.exists(meta):
        os.remove(meta)
    return True


def _successful_import(_path: str, _database):
    return True


def _filesystem_copy(source: str, destination: str, _database):
    shutil.copy2(source, destination)
    return destination


def test_project_asset_create_redo_refuses_external_destination(tmp_path):
    assets = tmp_path / "Assets"
    assets.mkdir()
    asset = assets / "Created.mat"

    def create():
        asset.write_text("created", encoding="utf-8")
        return AssetCreationResult(True, "", str(asset))

    command = ProjectAssetCreateCommand(
        str(assets),
        create,
        project_root=str(tmp_path),
        backup_root=str(tmp_path / "Library" / "EditorUndo"),
        delete_fn=_filesystem_delete,
        import_fn=_successful_import,
    )
    command.execute()
    command.undo()
    asset.write_text("external", encoding="utf-8")

    with pytest.raises(RuntimeError, match="occupied by an external change"):
        command.redo()
    assert asset.read_text(encoding="utf-8") == "external"
    command.dispose()


def test_project_prefab_create_owns_asset_and_source_linkage(tmp_path):
    assets = tmp_path / "Assets"
    assets.mkdir()
    prefab = assets / "Created.prefab"
    linkage = {"guid": "", "is_root": False}

    def create():
        prefab.write_text("prefab", encoding="utf-8")
        linkage.update(guid="prefab-guid", is_root=True)
        return AssetCreationResult(True, str(prefab), str(prefab))

    asset_command = ProjectAssetCreateCommand(
        str(assets),
        create,
        project_root=str(tmp_path),
        backup_root=str(tmp_path / "Library" / "EditorUndo"),
        delete_fn=_filesystem_delete,
        import_fn=_successful_import,
    )
    command = ProjectPrefabCreateCommand(
        asset_command,
        lambda: dict(linkage),
        lambda snapshot: linkage.update(snapshot),
    )

    command.execute()
    assert prefab.exists()
    assert linkage == {"guid": "prefab-guid", "is_root": True}

    command.undo()
    assert not prefab.exists()
    assert linkage == {"guid": "", "is_root": False}

    command.redo()
    assert prefab.exists()
    assert linkage == {"guid": "prefab-guid", "is_root": True}
    command.dispose()


def test_project_asset_rename_command_replays_both_directions(tmp_path):
    old_path = tmp_path / "Old.mat"
    new_path = tmp_path / "New.mat"
    old_path.write_text("material", encoding="utf-8")
    changed = []
    command = ProjectAssetRenameCommand(
        str(old_path),
        str(new_path),
        move_fn=_filesystem_move,
        on_changed=lambda: changed.append(True),
    )

    command.execute()
    assert new_path.read_text(encoding="utf-8") == "material"
    assert not old_path.exists()

    command.undo()
    assert old_path.read_text(encoding="utf-8") == "material"
    assert not new_path.exists()

    command.redo()
    assert new_path.exists()
    assert changed == [True, True, True]


def test_project_asset_rename_refuses_to_overwrite_external_change(tmp_path):
    old_path = tmp_path / "Old.mat"
    new_path = tmp_path / "New.mat"
    old_path.write_text("original", encoding="utf-8")
    new_path.write_text("external", encoding="utf-8")
    command = ProjectAssetRenameCommand(
        str(old_path),
        str(new_path),
        move_fn=_filesystem_move,
    )

    with pytest.raises(RuntimeError, match="external change"):
        command.execute()

    assert old_path.read_text(encoding="utf-8") == "original"
    assert new_path.read_text(encoding="utf-8") == "external"


def test_failed_asset_rename_undo_keeps_global_cursor_and_files(tmp_path):
    old_path = tmp_path / "Old.mat"
    new_path = tmp_path / "New.mat"
    old_path.write_text("original", encoding="utf-8")
    command = ProjectAssetRenameCommand(
        str(old_path),
        str(new_path),
        move_fn=_filesystem_move,
    )
    command.execute()
    old_path.write_text("external", encoding="utf-8")
    manager = UndoManager(EditorActionJournal())
    manager.record(command)

    manager.undo()

    assert manager.action_journal.cursor == 1
    assert old_path.read_text(encoding="utf-8") == "external"
    assert new_path.read_text(encoding="utf-8") == "original"


def test_project_asset_delete_rolls_back_batch_failure(tmp_path):
    assets = tmp_path / "Assets"
    assets.mkdir()
    first = assets / "A.mat"
    second = assets / "B.mat"
    first.write_text("A", encoding="utf-8")
    second.write_text("B", encoding="utf-8")

    def _fail_after_one(path: str, database):
        if os.path.basename(path) == "A.mat":
            return False
        return _filesystem_delete(path, database)

    command = ProjectAssetDeleteCommand(
        [str(first), str(second)],
        project_root=str(tmp_path),
        delete_fn=_fail_after_one,
        import_fn=_successful_import,
    )

    with pytest.raises(RuntimeError, match="asset delete failed"):
        command.execute()

    assert first.read_text(encoding="utf-8") == "A"
    assert second.read_text(encoding="utf-8") == "B"
    command.dispose()


def test_failed_asset_delete_undo_keeps_cursor_and_external_file(tmp_path):
    asset = tmp_path / "Assets" / "Original.mat"
    asset.parent.mkdir()
    asset.write_text("original", encoding="utf-8")
    command = ProjectAssetDeleteCommand(
        [str(asset)],
        project_root=str(tmp_path),
        delete_fn=_filesystem_delete,
        import_fn=_successful_import,
    )
    manager = UndoManager(EditorActionJournal())
    assert manager.execute(command)
    asset.write_text("external", encoding="utf-8")

    manager.undo()

    assert manager.action_journal.cursor == 1
    assert asset.read_text(encoding="utf-8") == "external"
    manager.clear()


def test_project_asset_delete_restores_real_asset_database_guid(tmp_path, engine):
    import json

    assets = tmp_path / "Assets"
    assets.mkdir()
    database = engine.get_asset_database()
    asset = assets / "Stable.effect"
    asset.write_text(
        json.dumps(
            {
                "$schema": "infernux.render_effect",
                "feature_type": "infernux.post.bloom",
                "parameters": {},
                "dependencies": [],
            }
        ),
        encoding="utf-8",
    )
    guid = database.import_asset(str(asset)).guid
    assert guid

    content = asset.read_bytes()
    meta = asset.with_name(asset.name + ".meta")
    metadata = meta.read_bytes()
    backup_root = tmp_path / "Library" / "EditorUndo"
    changed = []
    command = ProjectAssetDeleteCommand(
        [str(asset)],
        project_root=str(tmp_path),
        backup_root=str(backup_root),
        asset_database=database,
        on_deleted=lambda: changed.append("deleted"),
        on_restored=lambda: changed.append("restored"),
    )

    command.execute()
    assert not database.contains_guid(guid)
    assert not asset.exists() and not meta.exists()
    command.undo()

    assert database.get_guid_from_path(str(asset)) == guid
    assert database.get_path_from_guid(guid)
    assert asset.read_bytes() == content and meta.read_bytes() == metadata
    command.redo()
    assert not database.contains_guid(guid)
    assert not asset.exists() and not meta.exists()
    assert changed == ["deleted", "restored", "deleted"]
    command.dispose()
    assert not any(backup_root.glob("asset-delete-*"))


def test_project_asset_delete_restores_live_material_reference(tmp_path, engine, scene):
    assets = tmp_path / "Assets"
    assets.mkdir()
    database = engine.get_asset_database()
    asset = assets / "LiveReference.mat"
    material = InxMaterial.create_default_unlit()
    material.name = "LiveReference"
    assert material.save_to(str(asset))
    guid = database.import_asset(str(asset)).guid
    assert guid

    owner = scene.create_game_object("MaterialOwner")
    renderer = owner.add_component("MeshRenderer")
    renderer.set_material(0, guid)
    assert renderer.get_material_guids() == [guid]
    assert renderer.get_material(0) is not None

    command = ProjectAssetDeleteCommand(
        [str(asset)],
        project_root=str(tmp_path),
        asset_database=database,
    )
    command.execute()

    assert renderer.get_material_guids() == [guid]
    assert renderer.get_material(0) is None

    command.undo()
    assert database.get_guid_from_path(str(asset)) == guid
    assert renderer.get_material_guids() == [guid]
    assert renderer.get_material(0) is not None
    command.dispose()


def test_project_asset_delete_restores_live_audio_reference(tmp_path, engine, scene):
    assets = tmp_path / "Assets"
    assets.mkdir()
    database = engine.get_asset_database()
    asset = assets / "LiveReference.wav"
    with wave.open(str(asset), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(8000)
        stream.writeframes(b"\x00\x00" * 80)
    guid = database.import_asset(str(asset)).guid
    assert guid

    owner = scene.create_game_object("AudioOwner")
    source = owner.add_component("AudioSource")
    source.set_track_clip_by_guid(0, guid)
    assert source.get_track_clip_guid(0) == guid
    assert source.get_track_clip(0) is not None

    command = ProjectAssetDeleteCommand(
        [str(asset)],
        project_root=str(tmp_path),
        asset_database=database,
    )
    command.execute()

    assert source.get_track_clip_guid(0) == guid
    assert source.get_track_clip(0) is None

    command.undo()
    assert database.get_guid_from_path(str(asset)) == guid
    assert source.get_track_clip_guid(0) == guid
    assert source.get_track_clip(0) is not None
    command.dispose()


def test_project_asset_move_rolls_back_a_partial_failure(tmp_path):
    source = tmp_path / "Source.mat"
    destination = tmp_path / "Destination.mat"
    source.write_text("material", encoding="utf-8")
    calls = []

    def _move_then_fail(old: str, new: str, _database):
        calls.append((old, new))
        os.replace(old, new)
        if len(calls) == 1:
            raise RuntimeError("registry notification failed")
        return new

    command = ProjectAssetMoveCommand(
        str(source),
        str(destination),
        move_fn=_move_then_fail,
    )

    with pytest.raises(RuntimeError, match="registry notification failed"):
        command.execute()

    assert source.read_text(encoding="utf-8") == "material"
    assert not destination.exists()
    assert calls == [
        (str(source), str(destination)),
        (str(destination), str(source)),
    ]


def test_project_asset_copy_preserves_generated_identity_across_replay(tmp_path):
    source = tmp_path / "Source.mat"
    destination = tmp_path / "Copied.mat"
    source.write_text("material", encoding="utf-8")

    def _copy_with_identity(old: str, new: str, _database):
        shutil.copy2(old, new)
        with open(new + ".meta", "w", encoding="utf-8") as stream:
            stream.write('{"guid":"copied-guid"}')
        return new

    command = ProjectAssetCopyCommand(
        str(source),
        str(destination),
        project_root=str(tmp_path),
        copy_fn=_copy_with_identity,
        delete_fn=_filesystem_delete,
        import_fn=_successful_import,
    )

    command.execute()
    command.undo()
    assert not destination.exists()

    command.redo()
    assert destination.read_text(encoding="utf-8") == "material"
    assert (tmp_path / "Copied.mat.meta").read_text(encoding="utf-8") == ('{"guid":"copied-guid"}')
    command.dispose()


def test_project_asset_copy_rolls_back_failed_initial_copy(tmp_path):
    source = tmp_path / "Source.mat"
    destination = tmp_path / "Copied.mat"
    source.write_text("material", encoding="utf-8")

    def _partial_copy(old: str, new: str, _database):
        shutil.copy2(old, new)
        raise RuntimeError("import failed")

    command = ProjectAssetCopyCommand(
        str(source),
        str(destination),
        project_root=str(tmp_path),
        copy_fn=_partial_copy,
        delete_fn=_filesystem_delete,
        import_fn=_successful_import,
    )

    with pytest.raises(RuntimeError, match="import failed"):
        command.execute()

    assert source.exists()
    assert not destination.exists()
    command.dispose()


def test_project_asset_paste_is_one_global_action(tmp_path):
    first = tmp_path / "First.mat"
    second = tmp_path / "Second.mat"
    target = tmp_path / "Target"
    first.write_text("first", encoding="utf-8")
    second.write_text("second", encoding="utf-8")
    target.mkdir()
    destinations = [target / first.name, target / second.name]
    selected = []
    command = ProjectAssetPasteCommand(
        [
            ProjectAssetMoveCommand(str(first), str(destinations[0]), move_fn=_filesystem_move),
            ProjectAssetMoveCommand(str(second), str(destinations[1]), move_fn=_filesystem_move),
        ],
        [str(path) for path in destinations],
        on_applied=lambda paths: selected.append(list(paths)),
    )
    manager = UndoManager(EditorActionJournal())

    assert manager.execute(command)
    assert len(manager.action_journal.entries) == 1
    assert selected == [[str(path) for path in destinations]]
    assert all(path.exists() for path in destinations)

    manager.undo()
    assert first.exists() and second.exists()
    assert not any(path.exists() for path in destinations)

    manager.redo()
    assert all(path.exists() for path in destinations)
    manager.clear()


def test_project_asset_paste_rolls_back_when_selection_commit_fails(tmp_path):
    source = tmp_path / "Source.mat"
    destination = tmp_path / "Destination.mat"
    source.write_text("material", encoding="utf-8")
    command = ProjectAssetPasteCommand(
        [ProjectAssetMoveCommand(str(source), str(destination), move_fn=_filesystem_move)],
        [str(destination)],
        on_applied=lambda _paths: (_ for _ in ()).throw(RuntimeError("selection failed")),
    )

    with pytest.raises(RuntimeError, match="selection failed"):
        command.execute()

    assert source.exists()
    assert not destination.exists()
