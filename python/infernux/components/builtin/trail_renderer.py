"""Native timed trails using the engine's shared ribbon rendering path."""
from infernux.components.builtin_component import CppProperty
from infernux.components.fields import FieldType
from infernux.lib import Vector3
from infernux.graph.ramp import AnimationCurve

from .line_renderer import LineRenderer, _get_positions, _get_width_curve, _set_width_curve, _position_values


def _position(value):
    return value if isinstance(value, Vector3) else Vector3(*value)


class TrailRenderer(LineRenderer):
    _cpp_type_name = "TrailRenderer"
    _component_menu_path_ = "Rendering/Trail Renderer"
    _display_name_key = "component.trail_renderer"

    time = CppProperty("time", FieldType.FLOAT, default=5.0, range=(0.0, 1000.0))
    min_vertex_distance = CppProperty("min_vertex_distance", FieldType.FLOAT, default=0.1, range=(0.0, 1000.0))
    emitting = CppProperty("emitting", FieldType.BOOL, default=True)
    width_curve = CppProperty("width_curve", FieldType.ANIMATION_CURVE,
                              default=AnimationCurve.constant(0, 1, 1),
                              native_getter=_get_width_curve, native_setter=_set_width_curve,
                              curve_non_negative=True)
    position_count = CppProperty("position_count", FieldType.INT, default=0, readonly=True)
    positions = CppProperty("positions", FieldType.LIST, default=[], element_type=FieldType.VEC3,
                            readonly=True, native_getter=_get_positions)
    loop = CppProperty("loop", FieldType.BOOL, default=False, readonly=True)
    use_world_space = CppProperty("use_world_space", FieldType.BOOL, default=True, readonly=True)

    def add_position(self, position) -> None:
        self._require_cpp_component().add_position(_position(position))

    def add_positions(self, positions) -> None:
        self._require_cpp_component().add_positions(_position_values(positions))

    def set_positions(self, positions) -> None:
        self._require_cpp_component().set_positions(_position_values(positions))

    def clear(self) -> None:
        self._require_cpp_component().clear()
