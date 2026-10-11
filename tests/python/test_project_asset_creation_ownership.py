"""Creation history must never claim files published by a concurrent checkout."""
from __future__ import annotations

from threading import Event, Thread
from pathlib import Path

import pytest

from infernux.engine.ui import project_file_ops
from infernux.engine.asset_creation import AssetCreationResult
from infernux.engine.undo import ProjectAssetCreateCommand
from infernux.core.data_asset import DataAsset
from infernux.components import serialized_field


class OwnershipData(DataAsset):
    __serialized_type_id__ = "tests.creation.explicit_ownership"
    count: int = serialized_field(default=7)


@pytest.mark.parametrize("outcome", ["success", "rejected", "exception", "partial_failure"])
def test_creation_preserves_an_unrelated_concurrent_directory(engine, tmp_path, outcome):
    database = engine.get_asset_database()
    assets = Path(database.assets_root) / ("concurrent_ownership_" + tmp_path.name)
    assets.mkdir()
    external = assets / "TeammateLevel"
    owned = assets / "MyLevel"
    publish = Event()
    complete = Event()

    def checkout():
        assert publish.wait(10)
        external.mkdir()
        (external / "scene-data.txt").write_text("teammate revision", encoding="utf-8")
        complete.set()

    worker = Thread(target=checkout)
    worker.start()

    def creator():
        result = (project_file_ops.create_folder(str(assets), owned.name)
                  if outcome in {"success", "partial_failure"}
                  else AssetCreationResult(False, "intentional creation rejection"))
        publish.set()
        assert complete.wait(10)
        if outcome == "exception":
            raise RuntimeError("intentional creation exception")
        if outcome == "partial_failure":
            return AssetCreationResult(False, "intentional creation failure after publication", result.created_path)
        return result

    command = ProjectAssetCreateCommand(
        str(assets), creator, project_root=database.project_root, asset_database=database,
        backup_root=str(tmp_path / "Library" / "EditorUndo"),
    )
    error = None
    try:
        try:
            command.execute()
        except RuntimeError as exc:
            error = exc
        assert external.is_dir(), "creation removed the teammate's directory"
        assert (external / "scene-data.txt").read_text(encoding="utf-8") == "teammate revision"
        if outcome == "success":
            assert error is None, str(error)
            assert command.created_path == str(owned.resolve())
            command.undo()
            assert not owned.exists()
            assert external.is_dir()
            command.redo()
            assert owned.is_dir()
            assert (external / "scene-data.txt").read_text(encoding="utf-8") == "teammate revision"
        else:
            assert error is not None and "intentional creation" in str(error)
            assert not owned.exists()
    finally:
        publish.set()
        worker.join(timeout=10)
        command.dispose()
        database.delete_asset(str(assets))


@pytest.mark.parametrize("target", ["outside", "directory_itself", "sidecar"])
def test_creation_rejects_ownership_outside_one_project_item(tmp_path, target):
    assets = tmp_path / "Assets"
    assets.mkdir()
    path = {"outside": tmp_path / "Outside.txt", "directory_itself": assets,
            "sidecar": assets / "Foreign.mat.meta"}[target]
    if path is not assets:
        path.write_text("external data", encoding="utf-8")
    command = ProjectAssetCreateCommand(str(assets), lambda: AssetCreationResult(True, "", str(path)))
    with pytest.raises(RuntimeError, match="outside its Project directory"):
        command.execute()
    assert path.exists()
    if path is not assets:
        assert path.read_text(encoding="utf-8") == "external data"


def test_failed_creation_cleanup_reports_failure_without_deleting_other_paths(tmp_path):
    assets = tmp_path / "Assets"
    assets.mkdir()
    owned = assets / "Created"
    owned.mkdir()
    external = assets / "Teammate"
    external.mkdir()
    deletions = []

    def reject_cleanup(path, _database):
        deletions.append(path)
        return False

    command = ProjectAssetCreateCommand(
        str(assets), lambda: AssetCreationResult(False, "import rejected", str(owned)),
        delete_fn=reject_cleanup,
    )
    with pytest.raises(RuntimeError, match="owned item could not be removed"):
        command.execute()
    assert deletions == [str(owned.resolve())]
    assert owned.is_dir() and external.is_dir()


_CREATORS = [
    "folder", "script", "vertex", "fragment", "scene", "material", "physic_material",
    "render_texture", "data_asset", "prefab", "animclip", "animclip3d", "animfsm",
    "particlegraph", "render_effect", "render_effect_group", "animtimeline", "timelinefsm",
]


@pytest.mark.parametrize("kind,failure", [
    (kind, failure) for failure in ("none", "before_import", "after_import")
    for kind in _CREATORS if kind != "folder" or failure == "none"
])
def test_builtin_creator_owns_exact_path_through_import_and_history(engine, scene, tmp_path, monkeypatch, kind, failure):
    database = engine.get_asset_database()
    directory = Path(database.assets_root) / ("ownership_" + tmp_path.name)
    directory.mkdir()
    external = directory / "Teammate"
    original_import = project_file_ops._import_new_asset

    def reject_import(path, owner):
        if failure == "after_import":
            original_import(path, owner)
        external.mkdir()
        (external / "Keep.txt").write_text("external", encoding="utf-8")
        raise RuntimeError("intentional import rejection")

    if failure != "none":
        monkeypatch.setattr(project_file_ops, "_import_new_asset", reject_import)

    def creator():
        if kind == "folder":
            result = project_file_ops.create_folder(str(directory), "Created")
        elif kind in {"vertex", "fragment"}:
            result = project_file_ops.create_shader(str(directory), "Created", "vert" if kind == "vertex" else "frag", database)
        elif kind == "data_asset":
            result = project_file_ops.create_data_asset(str(directory), "Created", OwnershipData.__serialized_type_id__, database)
        elif kind == "prefab":
            result = project_file_ops.create_prefab_from_gameobject(scene.create_game_object("Created"), str(directory), database)
        elif kind == "render_effect":
            result = project_file_ops.create_render_effect(str(directory), "Created", "infernux.post.bloom", database)
        else:
            result = getattr(project_file_ops, "create_" + kind)(str(directory), "Created", database)
        assert isinstance(result, AssetCreationResult)
        assert result.created_path, (kind, result)
        if failure == "none":
            external.mkdir()
            (external / "Keep.txt").write_text("external", encoding="utf-8")
        return result

    command = ProjectAssetCreateCommand(
        str(directory), creator, project_root=database.project_root,
        backup_root=str(tmp_path / "History"), asset_database=database,
    )
    try:
        if failure != "none":
            with pytest.raises(RuntimeError, match="intentional import rejection"):
                command.execute()
            assert not Path(command.result.created_path).exists()
            assert list(directory.iterdir()) == [external]
        else:
            command.execute()
            path = Path(command.created_path)
            assert path.parent == directory
            content = path.read_bytes() if kind != "folder" else None
            sidecar = Path(str(path) + ".meta")
            meta = sidecar.read_bytes() if kind != "folder" else None
            for _ in range(2):
                command.undo()
                assert not path.exists()
                command.redo()
                assert path.exists()
                if kind != "folder":
                    assert path.read_bytes() == content and sidecar.read_bytes() == meta
            command.undo()
        assert (external / "Keep.txt").read_text(encoding="utf-8") == "external"
    finally:
        command.dispose()


def test_folder_creation_does_not_implicitly_own_missing_ancestors(tmp_path):
    result = project_file_ops.create_folder(str(tmp_path / "Missing"), "Child")
    assert not result.success and not result.created_path
    assert not (tmp_path / "Missing").exists()


@pytest.mark.parametrize("name", [".meta", "Hidden.meta", "Folder.META"])
def test_folder_creation_rejects_reserved_sidecar_names_without_mutation(tmp_path, name):
    result = project_file_ops.create_folder(str(tmp_path), name)
    assert not result.success and not result.created_path
    assert list(tmp_path.iterdir()) == []
