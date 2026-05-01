"""Type stubs for Infernux.renderstack.render_stack_pipeline — engine bridge to RenderStack."""

from __future__ import annotations

from typing import Any, ClassVar

from Infernux.renderstack.render_pipeline import RenderPipeline


class RenderStackPipeline(RenderPipeline):
    """Engine-level entry point bridge to RenderStack.

    When the engine calls ``render()``, this class:

    1. Finds the scene's ``RenderStack`` component.
    2. If found, delegates to ``RenderStack.render()``.
    3. If not found, falls back to plain pipeline rendering (no pass injection).

    Usage::

        context.set_render_pipeline(RenderStackPipeline())
    """

    name: ClassVar[str]
    """Display/internal pipeline name: ``"_RenderStackBridge"``."""

    def __init__(self) -> None: ...
    def render(self, context: Any, cameras: Any) -> None: ...
