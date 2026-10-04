"""Scene component that drives a public RuntimeAcceptance session."""

from __future__ import annotations

from infernux.acceptance import RuntimeAcceptance
from infernux.application import Application
from infernux.components.component import InxComponent
from infernux.components.decorators import add_component_menu, disallow_multiple
from infernux.debug import Debug


@add_component_menu("Testing/Runtime Acceptance Runner")
@disallow_multiple
class RuntimeAcceptanceRunner(InxComponent):
    """Bootstrap a process-owned manifest in Editor Play or Standalone.

    Put this component in a small bootstrap scene, include that scene and every
    manifest scene in BuildSettings, then enter Play. Assertion components call
    ``RuntimeAcceptance.pass_current()`` or ``RuntimeAcceptance.fail_current()``;
    the process session continues after this bootstrap scene is unloaded.
    """

    manifest_path: str = "Assets/Acceptance/RenderingVfxAcceptance.json"
    result_path: str = ""

    def start(self):
        try:
            if Application.is_editor():
                # Editor acceptance validates the same Game render path as a
                # Player. Focus once at bootstrap, without fighting later user
                # interaction while the scene sequence is running.
                from infernux.engine.ui.closable_panel import ClosablePanel

                ClosablePanel.focus_panel_by_id("game_view")
            RuntimeAcceptance.begin(self.manifest_path, self.result_path)
        except Exception as exc:
            Debug.log_error(f"[RuntimeAcceptance] failed to start: {type(exc).__name__}: {exc}")

    def update(self, delta_time: float):
        pass


__all__ = ["RuntimeAcceptanceRunner"]
