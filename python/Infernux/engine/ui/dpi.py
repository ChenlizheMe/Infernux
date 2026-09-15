"""Strict DPI helpers shared by Editor presentation code."""

from __future__ import annotations

from math import isfinite


def editor_dpi_scale(ctx) -> float:
    """Return authored UI units to native window units for this monitor.

    This is display scale / pixel density, not the framebuffer ratio. Pointer
    events, ImGui bounds and these scaled metrics all use the same window units.
    """
    scale = float(ctx.get_dpi_scale())
    if not isfinite(scale) or scale <= 0.0:
        raise RuntimeError(f"Editor reported an invalid display scale: {scale!r}")
    return scale


def scaled_editor_metric(ctx, value: float) -> float:
    """Convert an authored 100%-DPI Editor metric to the active monitor."""
    return float(value) * editor_dpi_scale(ctx)
