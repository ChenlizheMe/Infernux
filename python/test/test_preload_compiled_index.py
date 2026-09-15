"""Compiled lifecycle discovery indexes one registry snapshot per refresh."""
from types import SimpleNamespace

from Infernux.plugins import preload


def test_compiled_catalog_indexes_files_once_and_refreshes_updates(tmp_path, monkeypatch):
    paths = [tmp_path / f"module_{index}.pyc" for index in range(12)]
    for path in paths:
        path.touch()
    records = [{"compiled_path_hint": path.name, "preload_declarations": []} for path in paths]
    reads = []
    registry = SimpleNamespace(installed=lambda: reads.append(True) or ({"files": records},))
    manager = preload.PreloadManager(str(tmp_path), runtime=True, registry=registry)
    monkeypatch.setattr(manager, "_source_paths", lambda: iter(map(str, paths)))
    monkeypatch.setattr(manager, "_package_for_path", lambda path: "")
    monkeypatch.setattr(preload, "_is_editor_source", lambda *args: False)
    modules = []
    monkeypatch.setattr(preload, "_module_name", lambda path, root: modules.append(path) or "example")
    manager._refresh_declaration_catalog()
    assert len(reads) == 1
    assert modules == []
    assert manager._catalog_declarations() == []
    records[0]["preload_declarations"] = [{"name": "Ready", "bases": ["InxPreload"]}]
    paths[0].write_bytes(b"changed")
    manager._refresh_declaration_catalog()
    assert len(reads) == 2
    assert modules == [str(paths[0])]
    assert [item.name for item in manager._catalog_declarations()] == ["Ready"]
