from __future__ import annotations

import infernux.lib as native_lib
from infernux.core.assets import AssetManager
import pytest


def test_release_engine_clears_native_and_project_lifetime_state(monkeypatch):
    engine = object()
    monkeypatch.setattr(AssetManager, "_engine", engine)
    monkeypatch.setattr(AssetManager, "_asset_database", object())
    monkeypatch.setattr(AssetManager, "_registry", object())
    monkeypatch.setattr(AssetManager, "_cache", {"asset": object()})
    monkeypatch.setattr(AssetManager, "_texture_cache", {"texture": object()})
    monkeypatch.setattr(AssetManager, "_pending_gpu_texture_reloads", {"texture": "path"})
    monkeypatch.setattr(AssetManager, "_scheduled_saves", {"asset": object()})
    monkeypatch.setattr(AssetManager, "_pending_document_write_records", {"asset": [object()]})
    monkeypatch.setattr(AssetManager, "_meta_write_suppression", {"asset": 1.0})

    AssetManager.release_engine(engine)

    assert AssetManager._engine is None
    assert AssetManager._asset_database is None
    assert AssetManager._registry is None
    assert AssetManager._cache == {}
    assert AssetManager._texture_cache == {}
    assert AssetManager._pending_gpu_texture_reloads == {}
    assert AssetManager._scheduled_saves == {}
    assert AssetManager._pending_document_write_records == {}
    assert AssetManager._meta_write_suppression == {}


def test_release_engine_does_not_clear_a_newer_engine(monkeypatch):
    previous = object()
    current = object()
    database = object()
    monkeypatch.setattr(AssetManager, "_engine", current)
    monkeypatch.setattr(AssetManager, "_asset_database", database)

    AssetManager.release_engine(previous)

    assert AssetManager._engine is current
    assert AssetManager._asset_database is database


def test_asset_reference_does_not_reuse_retired_play_mode_database(monkeypatch):
    from types import SimpleNamespace
    from infernux.core.asset_ref import AssetRefBase
    from infernux.engine.play_mode import PlayModeManager
    from infernux.engine.interaction import AssetMutationService

    class RetiredDatabase:
        def get_path_from_guid(self, _guid):
            raise AssertionError("retired native database must never be queried")

    reference = AssetRefBase(guid="a" * 32, path_hint="Assets/Original.inxdata")
    monkeypatch.setattr(AssetMutationService, "instance", classmethod(lambda cls: None))
    monkeypatch.setattr(PlayModeManager, "_instance", SimpleNamespace(_asset_database=RetiredDatabase()))
    monkeypatch.setattr(AssetManager, "_asset_database", None)
    assert reference.path_hint == "Assets/Original.inxdata"


def test_registry_resolution_failure_is_not_suppressed(monkeypatch):
    class BrokenRegistry:
        @staticmethod
        def instance():
            raise RuntimeError("registry failed")

    monkeypatch.setattr(native_lib, "AssetRegistry", BrokenRegistry)

    with pytest.raises(RuntimeError, match="registry failed"):
        AssetManager._resolve_registry()
