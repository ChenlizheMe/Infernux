from __future__ import annotations

import json

import pytest

from infernux.engine import i18n


def test_locale_loader_rejects_malformed_json(tmp_path, monkeypatch):
    (tmp_path / "broken.json").write_text("{", encoding="utf-8")
    monkeypatch.setattr(i18n, "_LOCALES_DIR", str(tmp_path))
    i18n._tables.pop("broken", None)

    with pytest.raises(json.JSONDecodeError):
        i18n._load_locale_table("broken")


def test_locale_loader_requires_string_map(tmp_path, monkeypatch):
    (tmp_path / "broken.json").write_text('{"label": 42}', encoding="utf-8")
    monkeypatch.setattr(i18n, "_LOCALES_DIR", str(tmp_path))
    i18n._tables.pop("broken", None)

    with pytest.raises(ValueError, match="must map strings to strings"):
        i18n._load_locale_table("broken")


def test_translation_preserves_dynamic_undeclared_label_key(monkeypatch):
    monkeypatch.setattr(i18n, "_current_locale", "en")
    monkeypatch.setattr(i18n, "_tables", {"en": {"known": "Known"}})

    assert i18n.t("dynamic.property") == "dynamic.property"


def test_plugin_translation_catalog_is_owned_atomic_and_locale_complete(monkeypatch):
    monkeypatch.setattr(i18n, "_current_locale", "en")
    monkeypatch.setattr(i18n, "_tables", {"en": {}, "zh": {}})
    monkeypatch.setattr(i18n, "_translation_owners", {})
    monkeypatch.setattr(i18n, "_contributed_tables", {})
    catalog = {
        "$schema": "infernux.editor_translations",
        "locales": {
            "en": {"your_studio.example.menu": "Example"},
            "zh": {"your_studio.example.menu": "示例"},
        },
    }

    i18n.register_translation_catalog("package:your-studio/example", catalog)
    assert i18n.has_translation("your_studio.example.menu")
    assert i18n.t("your_studio.example.menu") == "Example"
    i18n._current_locale = "zh"
    assert i18n.t("your_studio.example.menu") == "示例"
    assert i18n.unregister_translation_catalog("package:your-studio/example")
    assert i18n.t("your_studio.example.menu") == "your_studio.example.menu"


def test_plugin_translation_catalog_rejects_partial_locales_without_mutation(monkeypatch):
    monkeypatch.setattr(i18n, "_tables", {"en": {}, "zh": {}})
    monkeypatch.setattr(i18n, "_translation_owners", {})
    monkeypatch.setattr(i18n, "_contributed_tables", {})

    with pytest.raises(ValueError, match="exactly match"):
        i18n.register_translation_catalog(
            "package:broken",
            {
                "$schema": "infernux.editor_translations",
                "locales": {"en": {"broken.menu": "Broken"}},
            },
        )
    assert i18n._translation_owners == {}


def test_set_locale_rejects_unknown_locale(monkeypatch):
    monkeypatch.setattr(i18n, "_tables", {"en": {}})

    with pytest.raises(ValueError, match="unsupported locale: unknown"):
        i18n.set_locale("unknown")


@pytest.mark.parametrize("locale", ["en", "zh"])
def test_light_native_schema_labels_are_translated(locale):
    from infernux.field_schema import get_native_field_schemas

    table = i18n._load_locale_table(locale)
    keys = set()
    for field in get_native_field_schemas("native:infernux.Light"):
        attrs = field.attributes
        keys.update(attrs[key] for key in ("display_name_key", "tooltip", "header") if key in attrs)
        if "enum" in attrs:
            keys.update(attrs["enum"]["labels"])
    assert keys and all(key.startswith("light.") for key in keys)
    assert not {key for key in keys if not table.get(key) or table[key] == key}
