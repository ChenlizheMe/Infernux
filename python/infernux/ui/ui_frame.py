"""Figma-style deterministic layout frame."""

from __future__ import annotations

from infernux.components import add_component_menu, serialized_field

from .enums import (
    UILayoutAlign,
    UILayoutDirection,
    UILayoutJustify,
    UILayoutPosition,
    UILayoutSizing,
)
from .inx_ui_screen_component import InxUIScreenComponent, _rect_cache
from .ui_layout_element import _layout_overrides


@add_component_menu("UI/Frame")
class UIFrame(InxUIScreenComponent):
    """A visual-neutral container that can lay out its direct UI children."""

    layout_direction: UILayoutDirection = serialized_field(
        default=UILayoutDirection.None_, group="Auto Layout"
    )
    gap: float = serialized_field(default=0.0, range=(0.0, 100000.0), group="Auto Layout")
    padding_left: float = serialized_field(default=0.0, range=(0.0, 100000.0), group="Auto Layout")
    padding_right: float = serialized_field(default=0.0, range=(0.0, 100000.0), group="Auto Layout")
    padding_top: float = serialized_field(default=0.0, range=(0.0, 100000.0), group="Auto Layout")
    padding_bottom: float = serialized_field(default=0.0, range=(0.0, 100000.0), group="Auto Layout")
    align_items: UILayoutAlign = serialized_field(default=UILayoutAlign.Start, group="Auto Layout")
    justify_content: UILayoutJustify = serialized_field(
        default=UILayoutJustify.Start, group="Auto Layout"
    )
    clip_content: bool = serialized_field(default=False, group="Appearance")
    _GEOMETRY_FIELDS = InxUIScreenComponent._GEOMETRY_FIELDS | frozenset({
        "layout_direction", "gap", "padding_left", "padding_right",
        "padding_top", "padding_bottom", "align_items", "justify_content", "clip_content",
    })

    def rebuild_layout(self, canvas_width=None, canvas_height=None):
        """Publish current layout to native transforms; return whether it changed."""
        from ._layout_driver import rebuild_canvas, rebuild_world
        canvas = self.get_canvas()
        return (rebuild_canvas(canvas, canvas_width, canvas_height, force=True)
                if canvas is not None else rebuild_world(self, force=True))

    def on_disable(self):
        from ._layout_driver import release_frame
        release_frame(self)
        from .inx_ui_screen_component import _invalidate_rect_cache
        _invalidate_rect_cache()

    def on_destroy(self):
        from ._layout_driver import release_frame
        release_frame(self)

    def _direct_ui_children(self):
        go = self._try_get_game_object()
        if go is None:
            return []
        children = []
        for child_object in go.get_children():
            for component in child_object.get_py_components():
                if isinstance(component, InxUIScreenComponent):
                    children.append(component)
                    break
        return children

    def _flow_children(self):
        children = []
        for child in self._direct_ui_children():
            if not child.game_object.active_in_hierarchy:
                continue
            if child.layout_position != UILayoutPosition.Flow:
                continue
            overrides = _layout_overrides(child)
            if overrides is None or not overrides[6]:
                children.append(child)
        return children

    @staticmethod
    def _depends_on_parent_size(child, axis):
        mode = child.width_sizing if axis == 0 else child.height_sizing
        if mode != UILayoutSizing.Fill:
            return False
        overrides = _layout_overrides(child)
        return overrides is None or (overrides[axis + 2] < 0 and overrides[axis + 4] != 0)

    @staticmethod
    def _child_preferred_size(child):
        desired = child._layout_desired_size()
        overrides = _layout_overrides(child)
        if overrides is None:
            return desired
        result = []
        for axis, value in enumerate(desired):
            minimum = max(float(child.min_width if axis == 0 else child.min_height), overrides[axis])
            preferred = overrides[axis + 2]
            maximum = child.max_width if axis == 0 else child.max_height
            result.append(child._clamp_layout_extent(preferred if preferred >= 0 else value, minimum, maximum))
        return tuple(result)

    @staticmethod
    def _axis_input(child, axis, preferred, *, main_axis=False):
        """Resolve one axis without replacing existing Fixed/Hug/Fill policy."""
        minimum = float(child.min_width if axis == 0 else child.min_height)
        maximum = float(child.max_width if axis == 0 else child.max_height)
        mode = child.width_sizing if axis == 0 else child.height_sizing
        flexible = max(.001, float(child.layout_weight)) if mode == UILayoutSizing.Fill else 0.0
        overrides = _layout_overrides(child)
        if overrides is not None and any(overrides[axis + offset] >= 0 for offset in (0, 2, 4)):
            minimum = max(minimum, overrides[axis])
            if overrides[axis + 4] >= 0:
                flexible = overrides[axis + 4]
        elif flexible > 0:
            preferred = 0.0  # Existing Fill divides all remaining space.
        elif main_axis:
            minimum = preferred  # Existing fixed children do not shrink.
        return minimum, preferred, maximum, flexible

    def _main_axis_sizes(self, children, desired_sizes, available, horizontal):
        axis = 0 if horizontal else 1
        inputs = [self._axis_input(child, axis, desired[axis], main_axis=True)
                  for child, desired in zip(children, desired_sizes)]
        available -= self.gap * max(0, len(children) - 1)
        preferred = sum(item[1] for item in inputs)
        minimum = sum(item[0] for item in inputs)
        if preferred > available:
            factor = max(0.0, min(1.0, (available - minimum) / (preferred - minimum))) if preferred > minimum else 0.0
            return [low + (value - low) * factor for low, value, _, _ in inputs]

        sizes = [item[1] for item in inputs]
        remaining = max(0.0, available - preferred)
        unresolved = [index for index, item in enumerate(inputs) if item[3] > 0]
        while unresolved:
            total_weight = sum(inputs[index][3] for index in unresolved)
            constrained = []
            consumed = 0.0
            for index in unresolved:
                low, base, high, weight = inputs[index]
                share = base + remaining * weight / total_weight
                resolved = children[index]._clamp_layout_extent(share, low, high)
                if abs(resolved - share) > 1e-6:
                    sizes[index] = resolved
                    consumed += resolved - base
                    constrained.append(index)
            if not constrained:
                for index in unresolved:
                    sizes[index] = inputs[index][1] + remaining * inputs[index][3] / total_weight
                break
            unresolved = [index for index in unresolved if index not in constrained]
            remaining = max(0.0, remaining - consumed)
        return sizes

    def _layout_desired_size(self) -> tuple[float, float]:
        width, height = super()._layout_desired_size()
        direction = self.layout_direction
        if not self.enabled or direction == UILayoutDirection.None_ or (
            self.width_sizing != UILayoutSizing.Hug and self.height_sizing != UILayoutSizing.Hug
        ):
            return width, height

        # Intrinsic sizes and arrangement share the existing layout lifetime.
        # A nested Hug frame is measured once, not once per sibling query.
        cache_key = (self, "desired-size")
        cached = _rect_cache.get(cache_key)
        if cached is not None:
            return cached
        children = self._flow_children()
        if not children:
            content_width = content_height = 0.0
        else:
            sizes = [self._child_preferred_size(child) for child in children]
            if direction == UILayoutDirection.Horizontal:
                if any(self._depends_on_parent_size(child, 0) for child in children):
                    content_width = 0.0
                else:
                    content_width = sum(size[0] for size in sizes) + self.gap * (len(sizes) - 1)
                content_height = max(size[1] for size in sizes)
            else:
                content_width = max(size[0] for size in sizes)
                if any(self._depends_on_parent_size(child, 1) for child in children):
                    content_height = 0.0
                else:
                    content_height = sum(size[1] for size in sizes) + self.gap * (len(sizes) - 1)

        if self.width_sizing == UILayoutSizing.Hug:
            if any(
                self._depends_on_parent_size(child, 0) for child in children
            ):
                raise ValueError("A width-hugging UIFrame cannot contain a fill-width flow child")
            width = self._clamp_layout_extent(
                self.padding_left + content_width + self.padding_right,
                self.min_width, self.max_width,
            )
        if self.height_sizing == UILayoutSizing.Hug:
            if any(
                self._depends_on_parent_size(child, 1) for child in children
            ):
                raise ValueError("A height-hugging UIFrame cannot contain a fill-height flow child")
            height = self._clamp_layout_extent(
                self.padding_top + content_height + self.padding_bottom,
                self.min_height, self.max_height,
            )
        result = width, height
        _rect_cache[cache_key] = result
        return result

    def _layout_rect_for_child(self, target, canvas_width: float, canvas_height: float):
        frame_rect = self.get_rect(canvas_width, canvas_height)
        if not self.enabled or self.layout_direction == UILayoutDirection.None_ or target.layout_position == UILayoutPosition.Absolute:
            return target._absolute_rect(canvas_width, canvas_height, reference_rect=frame_rect)

        cache_key = (self, "flow-rects", canvas_width, canvas_height, frame_rect)
        rects = _rect_cache.get(cache_key)
        if rects is None:
            rects = self._arrange_flow_children(frame_rect)
            _rect_cache[cache_key] = rects
        if target in rects:
            return rects[target]
        return target._absolute_rect(canvas_width, canvas_height, reference_rect=frame_rect)

    def _arrange_flow_children(self, frame_rect):
        """Resolve the entire sibling group once at the layout boundary."""
        children = self._flow_children()
        if not children:
            return {}
        desired_sizes = [self._child_preferred_size(child) for child in children]

        fx, fy, fw, fh = frame_rect
        inner_x = fx + self.padding_left
        inner_y = fy + self.padding_top
        inner_w = max(0.0, fw - self.padding_left - self.padding_right)
        inner_h = max(0.0, fh - self.padding_top - self.padding_bottom)
        horizontal = self.layout_direction == UILayoutDirection.Horizontal
        main_available = inner_w if horizontal else inner_h
        main_sizes = self._main_axis_sizes(children, desired_sizes, main_available, horizontal)

        occupied = sum(float(value) for value in main_sizes) + self.gap * max(0, len(children) - 1)
        free_space = max(0.0, main_available - occupied)
        effective_gap = self.gap
        if self.justify_content == UILayoutJustify.Center:
            main_offset = free_space * 0.5
        elif self.justify_content == UILayoutJustify.End:
            main_offset = free_space
        elif self.justify_content == UILayoutJustify.SpaceBetween and len(children) > 1:
            main_offset = 0.0
            effective_gap += free_space / float(len(children) - 1)
        else:
            main_offset = 0.0
        cursor = (inner_x if horizontal else inner_y) + main_offset
        rects = {}
        for index, child in enumerate(children):
            desired_w, desired_h = desired_sizes[index]
            main_size = float(main_sizes[index])
            cross_available = inner_h if horizontal else inner_w
            desired_cross = desired_h if horizontal else desired_w
            minimum, _, maximum, flexible = self._axis_input(child, 1 if horizontal else 0, desired_cross)
            cross_size = cross_available if flexible > 0 else min(desired_cross, cross_available)
            cross_size = child._clamp_layout_extent(cross_size, minimum, maximum)

            if self.align_items == UILayoutAlign.Center:
                cross_offset = (cross_available - cross_size) * 0.5
            elif self.align_items == UILayoutAlign.End:
                cross_offset = cross_available - cross_size
            elif self.align_items == UILayoutAlign.Stretch:
                cross_offset = 0.0
                cross_size = child._clamp_layout_extent(cross_available, minimum, maximum)
            else:
                cross_offset = 0.0

            if horizontal:
                rect = (cursor, inner_y + cross_offset, main_size, cross_size)
            else:
                rect = (inner_x + cross_offset, cursor, cross_size, main_size)
            rects[child] = rect
            cursor += main_size + effective_gap

        return rects
