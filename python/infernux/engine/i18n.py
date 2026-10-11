"""
Internationalization (i18n) for the Infernux editor.

Provides a simple key-based translation system with two supported locales:
``"en"`` (English) and ``"zh"`` (Simplified Chinese).

Translation strings are stored in external JSON files under ``locales/``:

- ``locales/en.json``
- ``locales/zh.json``

Usage::

    from infernux.engine.i18n import t

    label = t("menu.project")        # "Project" or "项目"
    label = t("menu.preferences")    # "Preferences" or "偏好设置"

The active locale is persisted to the Hub-owned Editor preferences file
so it survives across sessions.
"""

from __future__ import annotations

import json
import os

from infernux.engine.preferences_store import PreferencesStore

# ---------------------------------------------------------------------------
# Locale state
# ---------------------------------------------------------------------------

_current_locale: str = "zh"

# ---------------------------------------------------------------------------
# Translation tables — loaded from locales/*.json at module init
# ---------------------------------------------------------------------------

_LOCALES_DIR = os.path.join(os.path.dirname(__file__), "locales")

_tables: dict[str, dict[str, str]] = {}
_translation_owners: dict[str, dict[str, dict[str, str]]] = {}
_contributed_tables: dict[str, dict[str, str]] = {}

_store = PreferencesStore()


def _load_locale_table(locale: str) -> dict[str, str]:
    """Load and cache a single locale JSON file."""
    if locale in _tables:
        return _tables[locale]
    path = os.path.join(_LOCALES_DIR, f"{locale}.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in data.items()
    ):
        raise ValueError(f"locale table must map strings to strings: {path}")
    _tables[locale] = data
    return data


def _load_all_locales() -> None:
    """Pre-load all discovered locale files."""
    for name in sorted(os.listdir(_LOCALES_DIR)):
        if name.endswith(".json"):
            _load_locale_table(name[:-5])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def t(key: str) -> str:
    """Return the translated string for *key*, or *key* when undeclared."""
    return _tables[_current_locale].get(
        key, _contributed_tables.get(_current_locale, {}).get(key, key)
    )


def has_translation(key: str) -> bool:
    """Return whether *key* is explicitly declared by the active locale."""
    return key in _tables[_current_locale] or key in _contributed_tables.get(
        _current_locale, {}
    )


def register_translation_catalog(
    owner: str, document: dict[str, object]
) -> None:
    """Atomically publish one Editor plugin's ``editor/translations.json``."""
    identity, candidate = validate_translation_catalog(owner, document)

    replacement = dict(_translation_owners)
    replacement[identity] = candidate
    rebuilt = {locale: {} for locale in _tables}
    for catalog in replacement.values():
        for locale, entries in catalog.items():
            rebuilt[locale].update(entries)
    _translation_owners.clear()
    _translation_owners.update(replacement)
    _contributed_tables.clear()
    _contributed_tables.update(rebuilt)


def validate_translation_catalog(
    owner: str, document: dict[str, object]
) -> tuple[str, dict[str, dict[str, str]]]:
    """Validate a plugin catalog without changing the published locale tables."""
    identity = str(owner or "").strip()
    if not identity:
        raise ValueError("translation catalog owner cannot be empty")
    if set(document) != {"$schema", "locales"}:
        raise ValueError("translation catalog must contain exactly '$schema' and 'locales'")
    if document["$schema"] != "infernux.editor_translations":
        raise ValueError("translation catalog has an invalid $schema")
    locales = document["locales"]
    if not isinstance(locales, dict) or set(locales) != set(_tables):
        raise ValueError(
            "translation catalog locales must exactly match the Editor locales: "
            + ", ".join(sorted(_tables))
        )

    candidate: dict[str, dict[str, str]] = {}
    expected_keys: set[str] | None = None
    for locale in sorted(_tables):
        entries = locales[locale]
        if not isinstance(entries, dict) or any(
            not isinstance(key, str)
            or not key
            or "." not in key
            or not isinstance(value, str)
            or not value
            for key, value in entries.items()
        ):
            raise ValueError(
                f"translation catalog locale '{locale}' must map namespaced keys to non-empty strings"
            )
        keys = set(entries)
        if expected_keys is None:
            expected_keys = keys
        elif keys != expected_keys:
            raise ValueError("translation catalog locales must declare identical key sets")
        candidate[locale] = dict(entries)

    for locale, entries in candidate.items():
        for key in entries:
            if key in _tables[locale]:
                raise ValueError(f"plugin translation conflicts with engine key: {key}")
            for other_owner, other_catalog in _translation_owners.items():
                if other_owner != identity and key in other_catalog[locale]:
                    raise ValueError(
                        f"plugin translation key '{key}' is already owned by {other_owner}"
                    )

    return identity, candidate


def unregister_translation_catalog(owner: str) -> bool:
    """Remove every translated key owned by one Editor plugin package."""
    identity = str(owner or "").strip()
    if identity not in _translation_owners:
        return False
    del _translation_owners[identity]
    rebuilt = {locale: {} for locale in _tables}
    for catalog in _translation_owners.values():
        for locale, entries in catalog.items():
            rebuilt[locale].update(entries)
    _contributed_tables.clear()
    _contributed_tables.update(rebuilt)
    return True


def get_locale() -> str:
    """Return the current locale code (``"en"`` or ``"zh"``)."""
    return _current_locale


def set_locale(locale: str) -> None:
    """Set the active locale and persist to disk."""
    global _current_locale
    if locale not in _tables:
        raise ValueError(f"unsupported locale: {locale}")
    _current_locale = locale
    _save_preference()


# ---------------------------------------------------------------------------
# Persistence — Hub-owned Editor preferences
# ---------------------------------------------------------------------------

def _load_preference() -> None:
    """Load the locale from the preferences file."""
    global _current_locale
    locale = _store.get("language", "zh")
    if locale not in _tables:
        raise ValueError(f"unsupported locale in preferences: {locale}")
    _current_locale = locale


def _save_preference() -> None:
    """Save the current locale to the preferences file."""
    _store.set("language", _current_locale)

# ---------------------------------------------------------------------------
# Module init — load locale files, then restore persisted preference
# ---------------------------------------------------------------------------

_load_all_locales()
_load_preference()
