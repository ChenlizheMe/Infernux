from __future__ import annotations

import json

import pytest

from infernux.engine.preferences_store import PreferencesStore


def _store_at(path) -> PreferencesStore:
    store = object.__new__(PreferencesStore)
    store._path = str(path)
    return store


def test_invalid_preferences_json_is_not_treated_as_empty(tmp_path):
    path = tmp_path / "preferences.json"
    path.write_text("{\n", encoding="utf-8")
    store = _store_at(path)

    with pytest.raises(json.JSONDecodeError):
        store.load()


def test_set_preserves_unowned_preferences(tmp_path):
    path = tmp_path / "preferences.json"
    path.write_text('{"language": "zh"}\n', encoding="utf-8")
    store = _store_at(path)

    store.set("preferred_ide", "vscode")

    assert json.loads(path.read_text(encoding="utf-8")) == {
        "language": "zh",
        "preferred_ide": "vscode",
    }


def test_save_failure_reaches_the_caller(tmp_path, monkeypatch):
    store = _store_at(tmp_path / "preferences.json")

    def fail_write(_path, _text):
        raise OSError("disk rejected write")

    monkeypatch.setattr(
        "infernux.core.document_store.write_document_text",
        fail_write,
    )

    with pytest.raises(OSError, match="disk rejected write"):
        store.save({"language": "zh"})
