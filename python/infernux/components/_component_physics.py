"""ComponentPhysicsMixin — extracted from InxComponent."""
from __future__ import annotations



from infernux.lib import _wrap_native_builtin_component
from infernux.engine.runtime_dispatch import current_runtime_epoch


class ComponentPhysicsMixin:
    """ComponentPhysicsMixin method group for InxComponent."""

    def _dispatch_runtime_event(self, method_name: str, *args) -> None:
        """Dispatch one native/editor event against one published epoch."""
        if not self.enabled:
            return
        self._safe_lifecycle_call(
            method_name,
            *args,
            epoch=current_runtime_epoch(),
        )

    def _call_on_collision_enter(self, collision):
        """Internal: Trigger on_collision_enter lifecycle."""
        self._dispatch_runtime_event("on_collision_enter", collision)

    def _call_on_collision_stay(self, collision):
        """Internal: Trigger on_collision_stay lifecycle."""
        self._dispatch_runtime_event("on_collision_stay", collision)

    def _call_on_collision_exit(self, collision):
        """Internal: Trigger on_collision_exit lifecycle."""
        self._dispatch_runtime_event("on_collision_exit", collision)

    def _call_on_trigger_enter(self, other):
        """Internal: Trigger on_trigger_enter lifecycle."""
        self._dispatch_runtime_event("on_trigger_enter", _wrap_native_builtin_component(other))

    def _call_on_trigger_stay(self, other):
        """Internal: Trigger on_trigger_stay lifecycle."""
        self._dispatch_runtime_event("on_trigger_stay", _wrap_native_builtin_component(other))

    def _call_on_trigger_exit(self, other):
        """Internal: Trigger on_trigger_exit lifecycle."""
        self._dispatch_runtime_event("on_trigger_exit", _wrap_native_builtin_component(other))

    def _call_on_draw_gizmos(self):
        """Internal: Trigger on_draw_gizmos lifecycle (editor only)."""
        self._dispatch_runtime_event("on_draw_gizmos")

    def _call_on_draw_gizmos_selected(self):
        """Internal: Trigger on_draw_gizmos_selected lifecycle (editor only)."""
        self._dispatch_runtime_event("on_draw_gizmos_selected")

