"""asset_inspector — generic asset inspector driven by category definitions.

Usage::

    from Infernux.engine.ui.asset_inspector import (
        WidgetType, FieldDef, AssetCategoryDef,
        render_asset_inspector, invalidate,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, List, Optional, Tuple

from Infernux.lib import InxGUIContext


class WidgetType(Enum):
    """Inspector widget type for an editable field."""

    CHECKBOX = "checkbox"
    COMBO = "combo"
    FLOAT = "float"


@dataclass
class FieldDef:
    """Definition of a single editable field in the asset inspector."""

    key: str
    label: str
    field_type: WidgetType
    combo_entries: list = ...
    float_speed: float = 0.1
    float_range: Optional[Tuple[float, float]] = None


@dataclass
class AssetCategoryDef:
    """Declarative definition for an asset category's inspector layout."""

    display_name: str
    access_mode: object
    load_fn: Optional[Callable] = None
    refresh_fn: Optional[Callable] = None
    editable_fields: List[FieldDef] = ...
    extra_meta_keys: List[str] = ...
    custom_header_fn: Optional[Callable] = None
    custom_body_fn: Optional[Callable] = None
    autosave_debounce: float = 0.35
    show_header: bool = True


def render_asset_inspector(
    ctx: InxGUIContext,
    panel: object,
    file_path: str,
    category: str,
) -> None:
    """Single entry point for all asset inspectors."""
    ...

def invalidate() -> None:
    """Reset all inspector state (called on selection change)."""
    ...

def invalidate_asset(path: str) -> None:
    """Clear inspector cache if *path* is the currently inspected asset."""
    ...
