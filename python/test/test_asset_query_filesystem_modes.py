from __future__ import annotations

import os
from pathlib import Path

import pytest

from Infernux.application import Application
from Infernux.core.assets import AssetFile, AssetManager
from Infernux.core.material import Material
from Infernux.core.sandbox_files import SandboxPath


class _Database:
    def __init__(self, entries: dict[str, str]):
        self.entries = dict(entries)

    def get_all_guids(self):
        return list(self.entries)

    def get_path_from_guid(self, guid):
        return self.entries.get(guid, "")

    def get_guid_from_path(self, path):
        key = os.path.normcase(os.path.abspath(path))
        for guid, candidate in self.entries.items():
            if os.path.normcase(os.path.abspath(candidate)) == key:
                return guid
        return ""


def _editor(monkeypatch, root: Path) -> None:
    monkeypatch.setattr(Application, "is_editor", staticmethod(lambda: True))
    monkeypatch.setattr(Application, "is_player", staticmethod(lambda: False))
    monkeypatch.setattr(Application, "data_path", staticmethod(lambda: str(root)))


def test_editor_asset_path_is_converted_to_guid_before_load(monkeypatch, tmp_path):
    assets = tmp_path / "Assets"
    material = assets / "Materials" / "Gold.mat"
    material.parent.mkdir(parents=True)
    material.write_text("{}", encoding="utf-8")
    _editor(monkeypatch, tmp_path)
    monkeypatch.setattr(AssetManager, "_asset_database", _Database({"material-guid": str(material)}))
    observed = []
    monkeypatch.setattr(
        AssetManager,
        "load_by_guid",
        classmethod(lambda cls, guid, asset_type=None: observed.append(guid) or object()),
    )

    assert AssetManager.load("Assets/Materials/Gold.mat") is not None
    assert observed == ["material-guid"]


def test_editor_managed_load_rejects_absolute_paths_even_below_assets(
    monkeypatch, tmp_path
):
    material = tmp_path / "Assets" / "Materials" / "Gold.mat"
    material.parent.mkdir(parents=True)
    material.write_text("{}", encoding="utf-8")
    _editor(monkeypatch, tmp_path)
    monkeypatch.setattr(
        AssetManager, "_asset_database", _Database({"material-guid": str(material)})
    )

    with pytest.raises(ValueError, match="stay below Assets"):
        AssetManager.load(str(material))


@pytest.mark.parametrize(
    "query",
    [
        "Gold.mat",
        "*.mat",
        ".",
        "..",
        "Materials/Gold.mat",
        "Assets//Materials/Gold.mat",
        "Assets/./Materials/Gold.mat",
        "Assets/Materials/../Gold.mat",
        "/Assets/Materials/Gold.mat",
        "C:/Assets/Materials/Gold.mat",
        r"C:\Assets\Materials\Gold.mat",
        r"\\server\share\Assets\Gold.mat",
    ],
)
def test_managed_api_rejects_non_author_paths_without_legacy_fallback(
    monkeypatch, tmp_path, query
):
    (tmp_path / "Assets").mkdir()
    _editor(monkeypatch, tmp_path)
    monkeypatch.setattr(AssetManager, "_asset_database", _Database({}))

    with pytest.raises(ValueError):
        AssetManager.load(query)
    with pytest.raises(ValueError):
        AssetManager.find_assets(query)


def test_editor_managed_find_accepts_a_guid_and_still_returns_a_file_handle(
    monkeypatch, tmp_path
):
    material = tmp_path / "Assets" / "Materials" / "Gold.mat"
    material.parent.mkdir(parents=True)
    material.write_text("{}", encoding="utf-8")
    _editor(monkeypatch, tmp_path)
    monkeypatch.setattr(
        AssetManager, "_asset_database", _Database({"material-guid": str(material)})
    )

    files = AssetManager.find_assets("material-guid")
    assert files == [AssetFile("material-guid")]
    assert files[0].read_text() == "{}"


def test_editor_managed_file_read_rejects_catalog_entries_outside_assets(
    monkeypatch, tmp_path
):
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    (tmp_path / "Assets").mkdir()
    _editor(monkeypatch, tmp_path)
    monkeypatch.setattr(
        AssetManager, "_asset_database", _Database({"outside-guid": str(outside)})
    )

    handle = AssetFile("outside-guid")
    with pytest.raises(PermissionError, match="outside Assets"):
        handle.read_bytes()


def test_editor_managed_file_read_rejects_asset_symlink_escape(monkeypatch, tmp_path):
    assets = tmp_path / "Assets"
    assets.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    link = assets / "linked.txt"
    try:
        link.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"file links are unavailable: {exc}")
    _editor(monkeypatch, tmp_path)
    monkeypatch.setattr(
        AssetManager, "_asset_database", _Database({"linked-guid": str(link)})
    )

    with pytest.raises(PermissionError, match="outside Assets"):
        AssetFile("linked-guid").read_text()


def test_editor_project_context_resolves_assets_through_guid_identity(
    monkeypatch, tmp_path
):
    import Infernux.engine.project_context as project_context

    material = tmp_path / "Assets" / "Materials" / "Gold.mat"
    material.parent.mkdir(parents=True)
    material.write_text("{}", encoding="utf-8")
    _editor(monkeypatch, tmp_path)
    database = _Database({"material-guid": str(material)})
    monkeypatch.setattr(AssetManager, "_asset_database", database)
    observed = []
    original = database.get_guid_from_path
    database.get_guid_from_path = lambda path: observed.append(path) or original(path)

    assert project_context.resolve_asset_path("Assets/Materials/Gold.mat") == str(material)
    assert observed


def test_managed_find_returns_guid_identity_and_player_uses_frozen_path_query(
    monkeypatch, tmp_path
):
    import Infernux.engine.project_context as project_context

    material = tmp_path / "Assets" / "Materials" / "Gold.mat"
    material.parent.mkdir(parents=True)
    material.write_text("{}", encoding="utf-8")
    directory = material.parent / "Directory.mat"
    directory.mkdir()
    nested = material.parent / "Nested" / "Nested.mat"
    nested.parent.mkdir()
    nested.write_text("{}", encoding="utf-8")
    database = _Database(
        {
            "material-guid": str(material),
            "directory-guid": str(directory),
            "nested-guid": str(nested),
        }
    )
    monkeypatch.setattr(AssetManager, "_asset_database", database)
    _editor(monkeypatch, tmp_path)

    matches = AssetManager.find_assets("Assets/Materials/*.mat")
    assert matches == [AssetFile("material-guid")]
    assert not hasattr(matches[0], "guid")
    assert "material-guid" not in repr(matches[0])
    assert AssetManager.find_assets("Assets/Materials") == [
        AssetFile("material-guid")
    ]
    assert AssetManager.find_assets("Assets/Materials/**/*.mat") == [
        AssetFile("material-guid"),
        AssetFile("nested-guid"),
    ]
    with pytest.raises(ValueError, match="GUID or full Assets"):
        AssetManager.find_assets("Gold.mat")

    monkeypatch.setattr(Application, "is_editor", staticmethod(lambda: False))
    monkeypatch.setattr(Application, "is_player", staticmethod(lambda: True))
    monkeypatch.setattr(
        project_context,
        "_runtime_asset_query",
        lambda pattern: ("material-guid",)
        if pattern in {
            "Assets/Materials/*.mat",
            "Assets/Materials/Gold.mat",
            "Assets/Materials",
            "material-guid",
        } else (),
    )
    monkeypatch.setattr(
        project_context,
        "_runtime_asset_resolver",
        lambda guid: str(material) if guid == "material-guid" else None,
    )
    monkeypatch.setattr(
        project_context,
        "_runtime_asset_extension_resolver",
        lambda guid: ".mat" if guid == "material-guid" else "",
    )
    class ForbiddenPlayerDatabase:
        def __getattribute__(self, name):
            if name == "__class__":
                return type(self)
            raise AssertionError(f"Player managed queries touched AssetDatabase.{name}")

    monkeypatch.setattr(AssetManager, "_asset_database", ForbiddenPlayerDatabase())
    monkeypatch.setattr(
        os,
        "walk",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("managed Player queries must not walk the filesystem")
        ),
    )
    monkeypatch.setattr(
        os.path,
        "isfile",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("Player managed queries must not inspect source files")
        ),
    )
    assert AssetManager.find_assets("Assets/Materials/*.mat") == [
        AssetFile("material-guid")
    ]
    assert AssetManager.find_assets("Assets/Materials") == [
        AssetFile("material-guid")
    ]
    assert AssetManager.find_assets(
        "Assets/Materials/*.mat", asset_type=Material
    ) == [AssetFile("material-guid")]
    assert AssetManager.find_assets("material-guid") == [AssetFile("material-guid")]
    with pytest.raises(ValueError, match="GUID or full Assets"):
        AssetManager.find_assets("Gold.mat")
    observed = []
    monkeypatch.setattr(
        AssetManager,
        "load_by_guid",
        classmethod(lambda cls, guid, asset_type=None: observed.append(guid) or object()),
    )
    assert AssetManager.find_assets("material-guid")[0].load() is not None
    assert AssetManager.load("Assets/Materials/Gold.mat") is not None
    assert observed == ["material-guid", "material-guid"]


def test_raw_filesystem_handle_reads_writes_lists_and_deletes_inside_editor_assets(
    monkeypatch, tmp_path
):
    assets = tmp_path / "Assets"
    mods = assets / "Mods"
    mods.mkdir(parents=True)
    (mods / "Directory.json").mkdir()
    (mods / "a.json").write_text("a", encoding="utf-8")
    nested = mods / "Nested" / "nested.json"
    nested.parent.mkdir()
    nested.write_text("nested", encoding="utf-8")
    _editor(monkeypatch, tmp_path)

    matches = AssetManager.find_assets("Mods/*.json", raw_filesystem=True)
    assert [item.relative_path for item in matches] == ["Mods/a.json"]
    assert [
        item.relative_path
        for item in AssetManager.find_assets("Mods", raw_filesystem=True)
    ] == ["Mods/a.json"]
    assert [
        item.relative_path
        for item in AssetManager.find_assets("Mods/**/*.json", raw_filesystem=True)
    ] == ["Mods/a.json", "Mods/Nested/nested.json"]
    assert matches[0].read_text() == "a"

    created = AssetManager.load("Mods/b.json", raw_filesystem=True)
    assert isinstance(created, SandboxPath)
    assert created.write_text("bravo") == 5
    assert created.read_bytes() == b"bravo"
    created.delete()
    assert not (mods / "b.json").exists()


def test_player_load_resolves_path_to_guid_without_asset_database(
    monkeypatch, tmp_path
):
    import Infernux.engine.project_context as project_context

    cooked = tmp_path / "Library" / "Artifacts" / "Document" / "material.mat"
    cooked.parent.mkdir(parents=True)
    cooked.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(Application, "is_editor", staticmethod(lambda: False))
    monkeypatch.setattr(Application, "is_player", staticmethod(lambda: True))
    monkeypatch.setattr(
        project_context,
        "_runtime_asset_query",
        lambda path: ("material-guid",)
        if path == "Assets/Materials/Gold.mat" else (),
    )
    monkeypatch.setattr(
        project_context,
        "_runtime_asset_resolver",
        lambda guid: str(cooked) if guid == "material-guid" else None,
    )
    class ForbiddenPlayerDatabase:
        def __getattribute__(self, name):
            if name == "__class__":
                return type(self)
            raise AssertionError(f"Player managed load touched AssetDatabase.{name}")

    monkeypatch.setattr(AssetManager, "_asset_database", ForbiddenPlayerDatabase())
    AssetManager._cache.clear()
    AssetManager._texture_cache.clear()
    observed = []
    loaded = object()
    monkeypatch.setattr(
        AssetManager,
        "_load_by_type",
        classmethod(
            lambda cls, path, asset_type: observed.append((path, asset_type)) or loaded
        ),
    )

    assert AssetManager.load("Assets/Materials/Gold.mat") is loaded
    assert observed == [(str(cooked), Material)]


def test_player_managed_file_reads_cooked_guid_payload(monkeypatch, tmp_path):
    import Infernux.engine.project_context as project_context

    cooked = tmp_path / "Data" / "Documents" / "config.json"
    cooked.parent.mkdir(parents=True)
    cooked.write_text('{"quality":"high"}', encoding="utf-8")
    monkeypatch.setattr(Application, "is_editor", staticmethod(lambda: False))
    monkeypatch.setattr(Application, "is_player", staticmethod(lambda: True))
    monkeypatch.setattr(
        project_context,
        "_runtime_asset_resolver",
        lambda guid: str(cooked) if guid == "config-guid" else None,
    )

    handle = AssetFile("config-guid")
    assert handle.read_text() == '{"quality":"high"}'
    assert "config-guid" not in repr(handle)


@pytest.mark.parametrize("path", ["../outside.txt", "Mods/../../outside.txt"])
def test_raw_filesystem_rejects_parent_traversal(monkeypatch, tmp_path, path):
    (tmp_path / "Assets").mkdir()
    _editor(monkeypatch, tmp_path)

    with pytest.raises(ValueError, match="cannot contain"):
        AssetManager.load(path, raw_filesystem=True)


@pytest.mark.parametrize(
    "path",
    [
        "/system/secret.txt",
        "C:/system/secret.txt",
        r"C:\system\secret.txt",
        r"\\server\share\secret.txt",
    ],
)
def test_raw_filesystem_rejects_cross_platform_absolute_paths(
    monkeypatch, tmp_path, path
):
    (tmp_path / "Assets").mkdir()
    _editor(monkeypatch, tmp_path)

    with pytest.raises(ValueError, match="must be relative"):
        AssetManager.load(path, raw_filesystem=True)


def test_raw_filesystem_load_rejects_directories(monkeypatch, tmp_path):
    assets = tmp_path / "Assets"
    (assets / "Mods").mkdir(parents=True)
    _editor(monkeypatch, tmp_path)

    with pytest.raises(IsADirectoryError):
        AssetManager.load("Mods", raw_filesystem=True)


def test_sandbox_path_constructor_cannot_select_an_arbitrary_root(
    monkeypatch, tmp_path
):
    assets = tmp_path / "Assets"
    assets.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("secret", encoding="utf-8")
    _editor(monkeypatch, tmp_path)

    with pytest.raises(TypeError):
        SandboxPath(str(outside), "secret.txt")

    with pytest.raises(ValueError, match="must be relative"):
        SandboxPath(str(outside / "secret.txt"))


def test_raw_filesystem_handle_cannot_outlive_its_active_project_root(
    monkeypatch, tmp_path
):
    first = tmp_path / "First"
    second = tmp_path / "Second"
    (first / "Assets").mkdir(parents=True)
    (second / "Assets").mkdir(parents=True)
    (first / "Assets" / "secret.txt").write_text("secret", encoding="utf-8")
    active = {"root": first}
    monkeypatch.setattr(Application, "is_editor", staticmethod(lambda: True))
    monkeypatch.setattr(Application, "is_player", staticmethod(lambda: False))
    monkeypatch.setattr(
        Application, "data_path", staticmethod(lambda: str(active["root"]))
    )
    handle = AssetManager.load("secret.txt", raw_filesystem=True)
    active["root"] = second

    with pytest.raises(PermissionError, match="different sandbox root"):
        handle.read_text()


def test_raw_filesystem_handle_rejects_root_replaced_at_same_path(
    monkeypatch, tmp_path
):
    project = tmp_path / "Project"
    assets = project / "Assets"
    assets.mkdir(parents=True)
    target = assets / "config.txt"
    target.write_text("original", encoding="utf-8")
    _editor(monkeypatch, project)
    handle = AssetManager.load("config.txt", raw_filesystem=True)

    archived = project / "Assets-before-replacement"
    assets.rename(archived)
    assets.mkdir()
    (assets / "config.txt").write_text("replacement", encoding="utf-8")

    with pytest.raises(PermissionError, match="different sandbox root"):
        handle.read_text()


def test_player_managed_resolution_never_falls_back_to_project_disk(
    monkeypatch, tmp_path
):
    import Infernux.engine.project_context as project_context

    source = tmp_path / "Assets" / "Data" / "source.txt"
    source.parent.mkdir(parents=True)
    source.write_text("source", encoding="utf-8")
    monkeypatch.setattr(Application, "is_editor", staticmethod(lambda: False))
    monkeypatch.setattr(Application, "is_player", staticmethod(lambda: True))
    monkeypatch.setattr(project_context, "_project_root", str(tmp_path))
    monkeypatch.setattr(project_context, "_runtime_asset_resolver", None)
    monkeypatch.setattr(project_context, "_runtime_asset_query", None)

    with pytest.raises(RuntimeError, match="query is not configured"):
        project_context.resolve_asset_path("Assets/Data/source.txt")


def test_raw_filesystem_does_not_traverse_symlink(monkeypatch, tmp_path):
    assets = tmp_path / "Assets"
    assets.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("secret", encoding="utf-8")
    link = assets / "Mods"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory links are unavailable: {exc}")
    _editor(monkeypatch, tmp_path)

    with pytest.raises(PermissionError, match="links or junctions"):
        AssetManager.load("Mods/secret.txt", raw_filesystem=True)
    assert AssetManager.find_assets("Mods/*.txt", raw_filesystem=True) == []


@pytest.mark.parametrize("operation", ["read", "write", "delete"])
def test_raw_filesystem_handle_rejects_leaf_replaced_by_symlink(
    monkeypatch, tmp_path, operation
):
    assets = tmp_path / "Assets"
    assets.mkdir()
    inside = assets / "config.txt"
    inside.write_text("inside", encoding="utf-8")
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    _editor(monkeypatch, tmp_path)
    handle = AssetManager.load("config.txt", raw_filesystem=True)
    inside.unlink()
    try:
        inside.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"file links are unavailable: {exc}")

    with pytest.raises(PermissionError, match="links or junctions"):
        if operation == "read":
            handle.read_text()
        elif operation == "write":
            handle.write_text("changed")
        else:
            handle.delete()
    assert outside.read_text(encoding="utf-8") == "outside"


@pytest.mark.parametrize("operation", ["read", "write", "delete"])
def test_raw_filesystem_handle_rechecks_redirect_before_every_io(
    monkeypatch, tmp_path, operation
):
    import Infernux.core.sandbox_files as sandbox_files

    assets = tmp_path / "Assets"
    assets.mkdir()
    target = assets / "config.txt"
    target.write_text("inside", encoding="utf-8")
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    _editor(monkeypatch, tmp_path)
    handle = AssetManager.load("config.txt", raw_filesystem=True)

    if os.name == "nt":
        original = sandbox_files._has_reparse_point
        monkeypatch.setattr(
            sandbox_files,
            "_has_reparse_point",
            lambda path: os.path.normcase(os.path.abspath(path))
            == os.path.normcase(os.path.abspath(target))
            or original(path),
        )
    else:
        original_open = os.open
        redirected = False

        def redirect_before_open(path, flags, mode=0o777, *, dir_fd=None):
            nonlocal redirected
            if not redirected and path == "config.txt" and dir_fd is not None:
                redirected = True
                target.unlink()
                target.symlink_to(outside)
            return original_open(path, flags, mode, dir_fd=dir_fd)

        monkeypatch.setattr(os, "open", redirect_before_open)

    with pytest.raises(PermissionError, match="links or junctions"):
        if operation == "read":
            handle.read_text()
        elif operation == "write":
            handle.write_text("changed")
        else:
            handle.delete()
    if os.name == "nt":
        assert target.read_text(encoding="utf-8") == "inside"
    else:
        assert redirected
    assert outside.read_text(encoding="utf-8") == "outside"


def test_raw_filesystem_query_ignores_a_redirected_subtree(monkeypatch, tmp_path):
    import Infernux.core.sandbox_files as sandbox_files

    assets = tmp_path / "Assets"
    mods = assets / "Mods"
    mods.mkdir(parents=True)
    (mods / "secret.txt").write_text("secret", encoding="utf-8")
    _editor(monkeypatch, tmp_path)
    original = sandbox_files._has_reparse_point
    monkeypatch.setattr(
        sandbox_files,
        "_has_reparse_point",
        lambda path: os.path.normcase(os.path.abspath(path))
        == os.path.normcase(os.path.abspath(mods))
        or original(path),
    )

    assert AssetManager.find_assets("Mods/*.txt", raw_filesystem=True) == []
    with pytest.raises(PermissionError, match="links or junctions"):
        AssetManager.load("Mods/secret.txt", raw_filesystem=True)


@pytest.mark.skipif(os.name == "nt", reason="POSIX dirfd confinement contract")
def test_raw_filesystem_delete_stays_on_open_parent_when_path_is_redirected(
    monkeypatch, tmp_path
):
    assets = tmp_path / "Assets"
    mods = assets / "Mods"
    mods.mkdir(parents=True)
    target = mods / "config.txt"
    target.write_text("inside", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    outside_target = outside / "config.txt"
    outside_target.write_text("outside", encoding="utf-8")
    _editor(monkeypatch, tmp_path)
    handle = AssetManager.load("Mods/config.txt", raw_filesystem=True)
    original_unlink = os.unlink
    redirected = False

    def redirect_then_unlink(path, *, dir_fd=None):
        nonlocal redirected
        if not redirected and path == "config.txt" and dir_fd is not None:
            redirected = True
            mods.rename(assets / "Mods-safe")
            mods.symlink_to(outside, target_is_directory=True)
        return original_unlink(path, dir_fd=dir_fd)

    monkeypatch.setattr(os, "unlink", redirect_then_unlink)
    handle.delete()

    assert redirected
    assert outside_target.read_text(encoding="utf-8") == "outside"
    assert not (assets / "Mods-safe" / "config.txt").exists()


@pytest.mark.skipif(os.name == "nt", reason="POSIX dirfd confinement contract")
def test_raw_filesystem_create_stays_on_open_parent_when_path_is_redirected(
    monkeypatch, tmp_path
):
    assets = tmp_path / "Assets"
    mods = assets / "Mods"
    mods.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    _editor(monkeypatch, tmp_path)
    handle = AssetManager.load("Mods/config.txt", raw_filesystem=True)
    original_open = os.open
    redirected = False

    def redirect_then_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal redirected
        if not redirected and path == "config.txt" and dir_fd is not None:
            redirected = True
            mods.rename(assets / "Mods-safe")
            mods.symlink_to(outside, target_is_directory=True)
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", redirect_then_open)
    assert handle.write_text("inside") == len("inside")

    assert redirected
    assert not (outside / "config.txt").exists()
    assert (assets / "Mods-safe" / "config.txt").read_text(encoding="utf-8") == "inside"


@pytest.mark.skipif(os.name != "nt", reason="Windows handle confinement contract")
def test_windows_raw_filesystem_delete_rejects_leaf_redirected_during_open(
    monkeypatch, tmp_path
):
    import Infernux.core.sandbox_files as sandbox_files

    assets = tmp_path / "Assets"
    assets.mkdir()
    target = assets / "config.txt"
    target.write_text("inside", encoding="utf-8")
    outside_target = tmp_path / "outside.txt"
    outside_target.write_text("outside", encoding="utf-8")
    _editor(monkeypatch, tmp_path)
    handle = AssetManager.load("config.txt", raw_filesystem=True)
    original_open = sandbox_files._windows_create_handle
    redirected = False

    def redirect_before_open(path, desired_access, disposition):
        nonlocal redirected
        if not redirected:
            redirected = True
            target.unlink()
            try:
                target.symlink_to(outside_target)
            except OSError as exc:
                pytest.skip(f"file links are unavailable: {exc}")
        return original_open(path, desired_access, disposition)

    monkeypatch.setattr(
        sandbox_files, "_windows_create_handle", redirect_before_open
    )
    with pytest.raises(PermissionError, match="links or junctions"):
        handle.delete()

    assert redirected
    assert outside_target.read_text(encoding="utf-8") == "outside"


@pytest.mark.skipif(os.name != "nt", reason="Windows handle confinement contract")
def test_windows_raw_filesystem_create_stays_on_open_parent_when_parent_moves(
    monkeypatch, tmp_path
):
    import Infernux.core.sandbox_files as sandbox_files

    assets = tmp_path / "Assets"
    mods = assets / "Mods"
    mods.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    _editor(monkeypatch, tmp_path)
    handle = AssetManager.load("Mods/config.txt", raw_filesystem=True)
    original_open = sandbox_files._windows_create_handle
    redirected = False

    def redirect_before_open(path, desired_access, disposition):
        nonlocal redirected
        if not redirected and disposition == 1:
            redirected = True
            mods.rename(assets / "Mods-safe")
            try:
                mods.symlink_to(outside, target_is_directory=True)
            except OSError as exc:
                pytest.skip(f"directory links are unavailable: {exc}")
        return original_open(path, desired_access, disposition)

    monkeypatch.setattr(
        sandbox_files, "_windows_create_handle", redirect_before_open
    )
    with pytest.raises(PermissionError, match="outside its root"):
        handle.write_text("inside")

    assert redirected
    assert not (outside / "config.txt").exists()
    assert not (assets / "Mods-safe" / "config.txt").exists()


def test_player_raw_filesystem_is_rooted_beside_executable(monkeypatch, tmp_path):
    executable = tmp_path / "Player.exe"
    executable.write_bytes(b"")
    (tmp_path / "Mods").mkdir()
    (tmp_path / "Mods" / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(Application, "is_editor", staticmethod(lambda: False))
    monkeypatch.setattr(Application, "is_player", staticmethod(lambda: True))
    monkeypatch.setattr(
        Application,
        "_player_executable_directory",
        staticmethod(lambda: str(executable.parent)),
    )

    handle = AssetManager.load("Mods/config.json", raw_filesystem=True)

    assert handle.read_text() == "{}"
    assert [item.relative_path for item in AssetManager.find_assets(
        "Mods/*.json", raw_filesystem=True
    )] == ["Mods/config.json"]


def test_web_raw_filesystem_uses_isolated_persistent_loose_root(
    monkeypatch, tmp_path
):
    cooked_root = tmp_path / "cooked-player"
    cooked_mods = cooked_root / "Mods"
    cooked_mods.mkdir(parents=True)
    (cooked_mods / "config.json").write_text(
        '{"source":"cooked"}', encoding="utf-8"
    )
    persistent_root = tmp_path / "browser-persistent"
    loose_root = persistent_root / "loose"
    loose_mods = loose_root / "Mods"
    loose_mods.mkdir(parents=True)
    (loose_mods / "config.json").write_text(
        '{"source":"loose"}', encoding="utf-8"
    )
    monkeypatch.setattr(Application, "is_editor", staticmethod(lambda: False))
    monkeypatch.setattr(Application, "is_player", staticmethod(lambda: True))
    monkeypatch.setattr(
        Application,
        "persistent_data_path",
        staticmethod(lambda: str(persistent_root)),
    )
    monkeypatch.setattr(
        Application,
        "_player_executable_directory",
        staticmethod(lambda: str(cooked_root)),
    )
    monkeypatch.setenv("INFERNUX_WEB_RUNTIME", "1")

    handle = AssetManager.load("Mods/config.json", raw_filesystem=True)
    assert handle.read_text() == '{"source":"loose"}'
    handle.write_text('{"source":"changed"}')

    assert (loose_mods / "config.json").read_text(encoding="utf-8") == (
        '{"source":"changed"}'
    )
    assert (cooked_mods / "config.json").read_text(encoding="utf-8") == (
        '{"source":"cooked"}'
    )
