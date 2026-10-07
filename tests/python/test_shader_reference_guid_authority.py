"""A stored shader GUID cannot be silently reassigned by its display name."""

from types import SimpleNamespace

import pytest

import infernux.lib as lib
from infernux.engine.ui import inspector_shader_utils as shaders


@pytest.mark.parametrize("database_available", [False, True])
def test_missing_shader_guid_preserves_reference_without_name_lookup(monkeypatch, database_available):
    database = SimpleNamespace(get_path_from_guid=lambda _guid: "") if database_available else None
    monkeypatch.setattr(lib, "AssetRegistry", SimpleNamespace(
        instance=lambda: SimpleNamespace(get_asset_database=lambda: database),
    ))
    looked_up = []
    monkeypatch.setattr(shaders, "get_shader_file_path",
                        lambda shader_id, ext: looked_up.append((shader_id, ext)) or "")
    reference = shaders.make_shader_reference({
        "guid": "missing-project-shader", "shader_id": "Standard",
        "path_hint": "Assets/Shaders/Deleted.vert",
    }, ".vert")
    assert reference == {"guid": "missing-project-shader", "shader_id": "Standard", "path_hint": ""}
    assert not looked_up


def test_missing_shader_guid_cannot_bind_builtin_with_same_name(monkeypatch, tmp_path):
    builtin = tmp_path / "Library" / "Resources" / "shaders" / "standard.vert"
    builtin.parent.mkdir(parents=True)
    builtin.write_text('ShaderInfo { Name "Standard" }', encoding="utf-8")
    database = SimpleNamespace(get_path_from_guid=lambda _guid: "")
    monkeypatch.setattr(lib, "AssetRegistry", SimpleNamespace(
        instance=lambda: SimpleNamespace(get_asset_database=lambda: database),
    ))
    monkeypatch.setattr(shaders, "get_shader_file_path", lambda *_args: str(builtin))
    monkeypatch.setattr(shaders, "_read_compiled_shader_metadata",
                        lambda _path: {"shader_id": "Standard", "guid": "builtin-guid"})
    reference = shaders.make_shader_reference({"guid": "missing-project-shader", "shader_id": "Standard"}, ".vert")
    assert reference == {"guid": "missing-project-shader", "shader_id": "Standard", "path_hint": ""}


def test_restored_shader_guid_uses_current_name_instead_of_stored_name(monkeypatch):
    database = SimpleNamespace(get_path_from_guid=lambda guid: "Assets/Shaders/Renamed.vert" if guid == "project-guid" else "")
    monkeypatch.setattr(lib, "AssetRegistry", SimpleNamespace(
        instance=lambda: SimpleNamespace(get_asset_database=lambda: database),
    ))
    monkeypatch.setattr(shaders, "_read_compiled_shader_metadata",
                        lambda _path: {"shader_id": "Renamed", "guid": "project-guid"})
    reference = shaders.make_shader_reference({"guid": "project-guid", "shader_id": "Old"}, ".vert")
    assert reference == {"guid": "project-guid", "shader_id": "Renamed", "path_hint": "Assets/Shaders/Renamed.vert"}


def test_inspector_does_not_ping_same_named_shader_for_missing_guid(monkeypatch):
    from infernux.engine.ui import inspector_material, _inspector_references

    database = SimpleNamespace(get_path_from_guid=lambda _guid: "")
    monkeypatch.setattr(lib, "AssetRegistry", SimpleNamespace(
        instance=lambda: SimpleNamespace(get_asset_database=lambda: database),
    ))
    monkeypatch.setattr(shaders, "get_shader_file_path", lambda *_args: "Library/Resources/shaders/standard.vert")
    pings = []
    monkeypatch.setattr(_inspector_references, "ping_asset_in_project", pings.append)
    reference = {"guid": "missing-project-shader", "shader_id": "Standard"}
    assert inspector_material._shader_reference_path(reference, ".vert") == ""
    inspector_material._ping_shader_reference(reference, ".vert")
    assert not pings
