"""ComponentCoroutineMixin — extracted from InxComponent."""
from __future__ import annotations

"""
InxComponent - Base class for all Python-defined components.

Provides Unity-style lifecycle methods and property injection.
Users inherit from this class to create custom game logic.

Example:
    from Infernux.components import InxComponent, serialized_field
    
    class PlayerController(InxComponent):
        speed: float = serialized_field(default=5.0)
        
        def start(self):
            print("Player started!")
        
        def update(self, delta_time: float):
            pos = self.transform.position
            self.transform.position = Vector3(pos.x + self.speed * delta_time, pos.y, pos.z)
"""

from typing import Optional, Dict, Any, Type, TYPE_CHECKING, List
import copy
import threading
import weakref

from Infernux.lib import GameObject


class ComponentCoroutineMixin:
    """ComponentCoroutineMixin method group for InxComponent."""

    def start_coroutine(self, generator) -> 'Coroutine':
        """Start a coroutine on this component.

        Args:
            generator: A generator object (call your generator function first).

        Returns:
            A :class:`~Infernux.coroutine.Coroutine` handle that can be passed
            to :meth:`stop_coroutine` or ``yield``-ed from another coroutine to
            wait for completion.

        Example::

            from Infernux.coroutine import WaitForSeconds

            class Enemy(InxComponent):
                def start(self):
                    self.start_coroutine(self.patrol())

                def patrol(self):
                    while True:
                        debug.log("Moving left")
                        yield WaitForSeconds(2)
                        debug.log("Moving right")
                        yield WaitForSeconds(2)
        """
        from Infernux.coroutine import CoroutineScheduler
        if self._coroutine_scheduler is None:
            self._coroutine_scheduler = CoroutineScheduler()
        return self._coroutine_scheduler.start(generator, owner=self)

    def stop_coroutine(self, coroutine) -> None:
        """Stop a specific coroutine previously started with :meth:`start_coroutine`.

        Args:
            coroutine: The :class:`~Infernux.coroutine.Coroutine` handle.
        """
        if self._coroutine_scheduler is not None:
            self._coroutine_scheduler.stop(coroutine)

    def stop_all_coroutines(self) -> None:
        """Stop **all** coroutines running on this component."""
        if self._coroutine_scheduler is not None:
            self._coroutine_scheduler.stop_all()

    def _tick_coroutines_update(self, delta_time: float):
        """Advance coroutine work scheduled for the Update phase."""
        if self._coroutine_scheduler is not None:
            self._coroutine_scheduler.tick_update(delta_time)

    def _tick_coroutines_fixed_update(self, fixed_delta_time: float):
        """Advance coroutine work scheduled for the fixed-update phase."""
        if self._coroutine_scheduler is not None:
            self._coroutine_scheduler.tick_fixed_update(fixed_delta_time)

    def _tick_coroutines_late_update(self, delta_time: float):
        """Advance coroutine work scheduled for the late-update phase."""
        if self._coroutine_scheduler is not None:
            self._coroutine_scheduler.tick_late_update(delta_time)

    def _stop_coroutines_for_game_object_deactivate(self):
        """Unity stops all coroutines when the owning GameObject is deactivated."""
        if self._coroutine_scheduler is not None:
            self._coroutine_scheduler.stop_all()
            self._coroutine_scheduler = None

    @classmethod
    def _clear_all_instances(cls) -> None:
        """Clear the active-instances registry (call on scene unload/reload)."""
        cls._active_instances.clear()

