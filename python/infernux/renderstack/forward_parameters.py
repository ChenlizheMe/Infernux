"""Default Forward authoring parameters shared by native render backends."""
from __future__ import annotations

from enum import IntEnum

from infernux.components.fields import serialized_field
from infernux.renderstack._serialized_field_mixin import SerializedFieldCollectorMixin


class MSAASamples(IntEnum):
    """Anti-aliasing sample count."""
    OFF = 1
    X2 = 2
    X4 = 4
    X8 = 8


class DefaultForwardParameters(SerializedFieldCollectorMixin):
    """Typed parameters; no desktop render callbacks or graph allocation."""
    _reserved_attrs_ = frozenset({"name"})
    _render_stack = None
    name = "Default Forward"

    shadow_resolution: int = serialized_field(
        default=4096,
        range=(256, 8192),
        slider=False,
        tooltip="Shadow map resolution (width & height)",
        header="Shadows",
    )
    msaa_samples: MSAASamples = serialized_field(
        default=MSAASamples.X4,
        enum_labels=["X1 (Off)", "X2", "X4", "X8"],
        tooltip="Anti-aliasing sample count (X1 disables multisample anti-aliasing)",
        header="Anti-Aliasing",
    )
