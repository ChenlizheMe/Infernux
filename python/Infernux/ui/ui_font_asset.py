"""GUID-owned project font references used by UI text renderers.

Font fields persist an imported asset GUID. The Editor resolves that GUID from
its live database, while Player resolves it directly to the cooked primary
payload in the frozen runtime catalog. A stale ``path_hint`` is never promoted
to a second resource identity.
"""

from __future__ import annotations


def _font_reference_guid(reference) -> str:
    if reference is None:
        return ""
    from Infernux.core.asset_ref import AssetRefBase, get_asset_type_for_ref

    if not isinstance(reference, AssetRefBase):
        raise TypeError("UI font fields require a Font asset reference")
    if get_asset_type_for_ref(reference) != "Font":
        raise TypeError("UI font fields only accept Font assets")
    return str(reference.guid or "").strip()


def ui_font_reference_path(reference) -> str:
    """Resolve one Font reference by GUID through the active asset catalog."""
    guid = _font_reference_guid(reference)
    if not guid:
        return ""

    from Infernux.application import Application

    if Application.is_player():
        # Player managed resources are already identified by GUID. Resolve the
        # cooked payload directly; do not round-trip through a source alias or
        # let ``path_hint`` become a second runtime identity.
        from Infernux.engine.project_context import resolve_runtime_asset_guid

        cooked_path = str(resolve_runtime_asset_guid(guid) or "")
        if not cooked_path:
            raise FileNotFoundError(
                f"Font asset GUID is absent from the Player catalog: {guid}"
            )
        return cooked_path

    from Infernux.core.assets import AssetManager

    database = AssetManager.require_asset_database()
    authored_path = str(database.get_path_from_guid(guid) or "").strip()
    if not authored_path:
        raise FileNotFoundError(f"Font asset GUID is unavailable: {guid}")
    import os
    if not os.path.isfile(authored_path):
        raise FileNotFoundError(f"Font asset GUID is unavailable: {guid}")
    return authored_path


def ui_font_references(component, primary_field: str = "font",
                       fallback_field: str = "fallback_fonts"):
    """Return raw GUID references without triggering eager asset loading."""
    from Infernux.components.fields import get_raw_field_value

    primary = get_raw_field_value(component, primary_field)
    fallbacks = list(get_raw_field_value(component, fallback_field) or ())
    return primary, fallbacks


def ui_font_paths(component, primary_field: str = "font",
                  fallback_field: str = "fallback_fonts") -> tuple[str, list[str]]:
    """Resolve the component's primary and ordered fallback font chain."""
    primary, fallbacks = ui_font_references(
        component, primary_field, fallback_field,
    )
    primary_path = ui_font_reference_path(primary)
    fallback_paths = [ui_font_reference_path(reference) for reference in fallbacks]
    return primary_path, [path for path in fallback_paths if path]


def ui_font_signature(component, primary_field: str = "font",
                      fallback_field: str = "fallback_fonts") -> tuple[str, tuple[str, ...]]:
    """Return the GUID-only identity used by text layout caches."""
    primary, fallbacks = ui_font_references(
        component, primary_field, fallback_field,
    )
    return (
        _font_reference_guid(primary),
        tuple(_font_reference_guid(reference) for reference in fallbacks),
    )


__all__ = [
    "ui_font_paths",
    "ui_font_reference_path",
    "ui_font_references",
    "ui_font_signature",
]
