"""Independent minimum/preferred/flexible input for parent UI layout."""

from infernux.components import add_component_menu, serialized_field

from .inx_ui_component import InxUIComponent
from .inx_ui_screen_component import _invalidate_rect_cache, _rect_cache
from .ui_render_revision import is_unchanged_ui_value, mark_runtime_ui_dirty


@add_component_menu("UI/Layout Element")
class UILayoutElement(InxUIComponent):
    """Override each layout input independently; negative values are unset.

    Multiple elements on one object resolve each property by highest priority,
    then highest value at equal priority. The parent assigns the rectangles;
    this component does not move the Transform itself.
    """

    ignore_layout: bool = serialized_field(default=False, display_name_key="ui_comp.ignore_layout", group="Layout")
    min_width: float = serialized_field(default=-1.0, display_name_key="ui_comp.layout_min_width", group="Layout")
    min_height: float = serialized_field(default=-1.0, display_name_key="ui_comp.layout_min_height", group="Layout")
    preferred_width: float = serialized_field(default=-1.0, display_name_key="ui_comp.preferred_width", group="Layout")
    preferred_height: float = serialized_field(default=-1.0, display_name_key="ui_comp.preferred_height", group="Layout")
    flexible_width: float = serialized_field(default=-1.0, display_name_key="ui_comp.flexible_width", group="Layout")
    flexible_height: float = serialized_field(default=-1.0, display_name_key="ui_comp.flexible_height", group="Layout")
    layout_priority: int = serialized_field(default=1, display_name_key="ui_comp.layout_priority", group="Layout")
    _LAYOUT_FIELDS = frozenset({
        "ignore_layout", "min_width", "min_height", "preferred_width", "preferred_height",
        "flexible_width", "flexible_height", "layout_priority", "enabled",
    })

    def __setattr__(self, name, value):
        changed = name in self._LAYOUT_FIELDS and not is_unchanged_ui_value(self, name, value)
        super().__setattr__(name, value)
        if changed:
            self._set_layout_dirty()

    @staticmethod
    def _set_layout_dirty():
        _invalidate_rect_cache()
        mark_runtime_ui_dirty()

    def on_enable(self):
        self._set_layout_dirty()

    def on_disable(self):
        self._set_layout_dirty()

    def on_destroy(self):
        self._set_layout_dirty()

    def calculate_layout_input_horizontal(self):
        """Inputs are authored properties; no intrinsic measurement is needed."""

    def calculate_layout_input_vertical(self):
        """Inputs are authored properties; no intrinsic measurement is needed."""


def _layout_overrides(widget):
    """One cached, per-property priority resolution for a visual component."""
    owner = widget._try_get_game_object()
    if owner is None:
        return None
    key = (owner.id, "layout-element-inputs")
    if key in _rect_cache:
        return _rect_cache[key]
    elements = [component for component in owner.get_py_components()
                if isinstance(component, UILayoutElement) and component.enabled]
    if not elements:
        result = None
    else:
        values = []
        for field in ("min_width", "min_height", "preferred_width", "preferred_height",
                      "flexible_width", "flexible_height"):
            candidates = [(int(element.layout_priority), float(getattr(element, field)))
                          for element in elements if getattr(element, field) >= 0]
            values.append(max(candidates)[1] if candidates else -1.0)
        # One enabled, non-ignoring ILayoutIgnorer provider opts back in.
        result = (*values, all(element.ignore_layout for element in elements))
    _rect_cache[key] = result
    return result
