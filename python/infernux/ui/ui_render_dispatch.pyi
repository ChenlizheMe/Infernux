"""Type stubs for infernux.ui.ui_render_dispatch — unified UI element rendering dispatch."""

from __future__ import annotations

from typing import Any, Callable, Optional, Tuple


def register_ui_renderer(component_cls_name: str, backend: str, fn: Callable) -> None:
    """Register a renderer function for a UI component type.

    Args:
        component_cls_name: E.g. ``"UIText"``, ``"UIImage"``, ``"UIButton"``.
        backend: ``"editor"`` (ImGui draw-list) or ``"runtime"`` (GPU ScreenUI).
        fn: Callable with backend-specific signature.
    """
    ...


def get_ui_renderer(component_cls_name: str, backend: str) -> Optional[Callable]:
    """Look up a registered renderer by component class name and backend."""
    ...


def dispatch(elem: Any, backend: str, **kwargs: Any) -> bool:
    """Dispatch rendering of *elem* to the registered handler.

    Walks the MRO to find a matching renderer.

    Args:
        elem: The UI element to render.
        backend: ``"editor"`` or ``"runtime"``.
        **kwargs: Backend-specific keyword arguments forwarded to the renderer.

    Returns:
        ``True`` if a handler was found and called, ``False`` otherwise.
    """
    ...

def runtime_ui_revision(
    scene: Any, canvases: Any, width: int, height: int,
    texture_generation: int = 0,
    world_elements: Any = (), persistent_scene: Any = None,
    runtime_scenes: Any = None,
) -> int: ...


def extract_common(elem: Any) -> dict:
    """Extract shared visual attributes from any ``InxUIScreenComponent``.

    Returns a dict with keys: ``color``, ``opacity``, ``rotation``,
    ``mirror_h``, ``mirror_v``, ``corner_radius``.
    """
    ...


def text_align_to_float(align_h: Any, align_v: Any) -> Tuple[float, float]:
    """Convert ``TextAlignH`` / ``TextAlignV`` enums to ``(0.0 / 0.5 / 1.0)`` floats."""
    ...
