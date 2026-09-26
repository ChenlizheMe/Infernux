"""GUID ownership at Prefab editor/runtime entry points."""

from __future__ import annotations

import json
from types import SimpleNamespace

from Infernux.engine import prefab_manager
from Infernux.engine.interaction.scene_objects import SceneObjectCommandService


def _prefab_document(canvas_name: str = "") -> dict:
    document = {
        "root_object": {
            "local_id": 1,
            "name": "Widget",
            "active": True,
            "is_static": False,
            "tag": "",
            "layer": 0,
            "transform": {},
            "components": [],
            "children": [],
        },
        "next_local_id": 2,
        "next_component_id": 1,
    }
    if canvas_name:
        document["source_canvas_name"] = canvas_name
    return document


class _AssetDatabase:
    def __init__(self, paths_by_guid: dict[str, str], guids_by_path: dict[str, str]):
        self.paths_by_guid = paths_by_guid
        self.guids_by_path = guids_by_path

    def get_guid_from_path(self, path):
        return self.guids_by_path.get(str(path), "")

    def get_path_from_guid(self, guid):
        return self.paths_by_guid.get(str(guid), "")


def test_prefab_canvas_path_boundary_uses_guid_canonical_path(tmp_path):
    stale = tmp_path / "old.prefab"
    canonical = tmp_path / "moved.prefab"
    stale.write_text(json.dumps(_prefab_document("WrongCanvas")), encoding="utf-8")
    canonical.write_text(json.dumps(_prefab_document("CurrentCanvas")), encoding="utf-8")
    database = _AssetDatabase(
        {"prefab-guid": str(canonical)},
        {str(stale): "prefab-guid"},
    )

    assert prefab_manager.read_prefab_source_canvas(
        file_path=str(stale), asset_database=database,
    ) == "CurrentCanvas"


def test_prefab_canvas_ignores_unregistered_path(tmp_path):
    legacy = tmp_path / "legacy.prefab"
    legacy.write_text(json.dumps(_prefab_document("LegacyCanvas")), encoding="utf-8")

    assert prefab_manager.read_prefab_source_canvas(
        file_path=str(legacy), asset_database=_AssetDatabase({}, {}),
    ) == ""


def test_prefab_instantiation_path_boundary_caches_by_resolved_guid(tmp_path, monkeypatch):
    stale = tmp_path / "old.prefab"
    canonical = tmp_path / "moved.prefab"
    stale.write_text("stale", encoding="utf-8")
    canonical.write_text("current", encoding="utf-8")
    database = _AssetDatabase(
        {"prefab-guid": str(canonical)},
        {str(stale): "prefab-guid"},
    )
    calls = []

    def load(path, guid, asset_database):
        calls.append((path, guid, asset_database))
        return None

    monkeypatch.setattr(prefab_manager, "_get_cached_prefab_template", load)

    assert prefab_manager.instantiate_prefab(
        file_path=str(stale), scene=object(), asset_database=database,
    ) is None
    assert calls == [(str(canonical), "prefab-guid", database)]


def test_unregistered_prefab_path_is_never_a_cache_identity(tmp_path):
    path = tmp_path / "authoring.prefab"
    path.write_text(json.dumps(_prefab_document()), encoding="utf-8")
    prefab_manager._PREFAB_TEMPLATE_CACHE.clear()

    assert prefab_manager._get_cached_prefab_template(str(path), "") is not None
    assert prefab_manager._PREFAB_TEMPLATE_CACHE == {}


def test_scene_prefab_path_is_resolved_before_runtime_calls(monkeypatch):
    from Infernux.engine.undo import UndoManager

    database = _AssetDatabase(
        {"prefab-guid": "Assets/Moved.prefab"},
        {"Assets/Old.prefab": "prefab-guid"},
    )
    registry = SimpleNamespace(get_asset_database=lambda: database)
    monkeypatch.setattr(
        "Infernux.lib.AssetRegistry.instance", staticmethod(lambda: registry),
    )

    class Selection:
        snapshot = "before"

        def select_scene_object(self, *_args, **_kwargs):
            self.snapshot = "after"

        def apply_snapshot(self, snapshot, **_kwargs):
            self.snapshot = snapshot

    class Scene:
        @staticmethod
        def find_by_id(_identity):
            return None

        @staticmethod
        def get_root_objects():
            return []

    manager = SimpleNamespace(record=lambda _command: True)
    monkeypatch.setattr(
        UndoManager,
        "instance",
        staticmethod(lambda: manager),
    )
    calls = []
    monkeypatch.setattr(
        prefab_manager,
        "read_prefab_source_canvas",
        lambda **kwargs: calls.append(("canvas", kwargs)) or "",
    )
    created = SimpleNamespace(id=41)
    monkeypatch.setattr(
        prefab_manager,
        "instantiate_prefab",
        lambda **kwargs: calls.append(("instantiate", kwargs)) or created,
    )

    service = object.__new__(SceneObjectCommandService)
    service._selection = Selection()
    service._clipboard = None
    service._active_scene = lambda: Scene()
    service.can_external_drop = lambda *_args, **_kwargs: True

    assert service.instantiate_prefab_object("Assets/Old.prefab") is created
    assert calls[0][0] == "canvas"
    assert calls[0][1]["guid"] == "prefab-guid"
    assert "file_path" not in calls[0][1]
    assert calls[1][0] == "instantiate"
    assert calls[1][1]["guid"] == "prefab-guid"
    assert "file_path" not in calls[1][1]
