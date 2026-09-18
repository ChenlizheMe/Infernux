from __future__ import annotations

import json

import pytest

from Infernux.engine import i18n


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


def test_set_locale_rejects_unknown_locale(monkeypatch):
    monkeypatch.setattr(i18n, "_tables", {"en": {}})

    with pytest.raises(ValueError, match="unsupported locale: unknown"):
        i18n.set_locale("unknown")


@pytest.mark.parametrize("locale", ["en", "zh"])
def test_light_native_schema_labels_are_translated(locale):
    from Infernux.field_schema import get_native_field_schemas

    table = i18n._load_locale_table(locale)
    keys = set()
    for field in get_native_field_schemas("native:infernux.Light"):
        attrs = field.attributes
        keys.update(attrs[key] for key in ("display_name_key", "tooltip", "header") if key in attrs)
        if "enum" in attrs:
            keys.update(attrs["enum"]["labels"])
    assert keys and all(key.startswith("light.") for key in keys)
    assert not {key for key in keys if not table.get(key) or table[key] == key}
