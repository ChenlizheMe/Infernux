"""
Infernux Gizmos System — Unity-style immediate-mode gizmo drawing.

Usage::

    from Infernux.gizmos import Gizmos

    class MyComponent(InxComponent):
        def on_draw_gizmos(self):
            Gizmos.color = (0, 1, 0)
            Gizmos.draw_wire_sphere(self.transform.position, 2.0)

        def on_draw_gizmos_selected(self):
            Gizmos.color = (1, 1, 0)
            Gizmos.draw_wire_cube(self.transform.position, (1, 1, 1))
"""

from .gizmos import Gizmos


def __getattr__(name):
    # Author scripts may import the drawing API in Players. Scene collection
    # belongs to the Editor and is loaded only when explicitly requested.
    if name == "GizmosCollector":
        from .collector import GizmosCollector
        return GizmosCollector
    raise AttributeError(name)

__all__ = [
    "Gizmos",
    "GizmosCollector",
]
