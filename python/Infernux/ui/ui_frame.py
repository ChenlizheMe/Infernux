"""Figma-style deterministic layout frame."""

from __future__ import annotations

from Infernux.components import add_component_menu, serialized_field

from .enums import (
    UILayoutAlign,
    UILayoutDirection,
    UILayoutJustify,
    UILayoutPosition,
    UILayoutSizing,
)
from .inx_ui_screen_component import InxUIScreenComponent, _rect_cache


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

    def _layout_desired_size(self) -> tuple[float, float]:
        width, height = super()._layout_desired_size()
        direction = self.layout_direction
        if direction == UILayoutDirection.None_ or (
            self.width_sizing != UILayoutSizing.Hug and self.height_sizing != UILayoutSizing.Hug
        ):
            return width, height

        # Intrinsic sizes and arrangement share the existing layout lifetime.
        # A nested Hug frame is measured once, not once per sibling query.
        cache_key = (self, "desired-size")
        cached = _rect_cache.get(cache_key)
        if cached is not None:
            return cached
        children = [
            child for child in self._direct_ui_children()
            if child.layout_position == UILayoutPosition.Flow
        ]
        if not children:
            content_width = content_height = 0.0
        else:
            sizes = [child._layout_desired_size() for child in children]
            if direction == UILayoutDirection.Horizontal:
                if any(child.width_sizing == UILayoutSizing.Fill for child in children):
                    content_width = 0.0
                else:
                    content_width = sum(size[0] for size in sizes) + self.gap * (len(sizes) - 1)
                content_height = max(size[1] for size in sizes)
            else:
                content_width = max(size[0] for size in sizes)
                if any(child.height_sizing == UILayoutSizing.Fill for child in children):
                    content_height = 0.0
                else:
                    content_height = sum(size[1] for size in sizes) + self.gap * (len(sizes) - 1)

        if self.width_sizing == UILayoutSizing.Hug:
            if any(
                child.width_sizing == UILayoutSizing.Fill for child in children
            ):
                raise ValueError("A width-hugging UIFrame cannot contain a fill-width flow child")
            width = self._clamp_layout_extent(
                self.padding_left + content_width + self.padding_right,
                self.min_width, self.max_width,
            )
        if self.height_sizing == UILayoutSizing.Hug:
            if any(
                child.height_sizing == UILayoutSizing.Fill for child in children
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
        if self.layout_direction == UILayoutDirection.None_ or target.layout_position == UILayoutPosition.Absolute:
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
        children = [
            child for child in self._direct_ui_children()
            if child.layout_position == UILayoutPosition.Flow
        ]
        if not children:
            return {}
        desired_sizes = [child._layout_desired_size() for child in children]

        fx, fy, fw, fh = frame_rect
        inner_x = fx + self.padding_left
        inner_y = fy + self.padding_top
        inner_w = max(0.0, fw - self.padding_left - self.padding_right)
        inner_h = max(0.0, fh - self.padding_top - self.padding_bottom)
        horizontal = self.layout_direction == UILayoutDirection.Horizontal
        main_available = inner_w if horizontal else inner_h
        main_sizes = []
        fill_indices = []
        fixed_total = self.gap * max(0, len(children) - 1)

        for child, (desired_w, desired_h) in zip(children, desired_sizes):
            fill = child.width_sizing == UILayoutSizing.Fill if horizontal else child.height_sizing == UILayoutSizing.Fill
            if fill:
                main_sizes.append(None)
                fill_indices.append(len(main_sizes) - 1)
            else:
                value = desired_w if horizontal else desired_h
                main_sizes.append(value)
                fixed_total += value

        remaining = max(0.0, main_available - fixed_total)
        unresolved = list(fill_indices)
        while unresolved:
            total_weight = sum(
                max(0.001, float(children[index].layout_weight))
                for index in unresolved
            )
            round_remaining = remaining
            constrained = []
            constrained_total = 0.0
            for index in unresolved:
                child = children[index]
                share = round_remaining * max(0.001, float(child.layout_weight)) / total_weight
                minimum = child.min_width if horizontal else child.min_height
                maximum = child.max_width if horizontal else child.max_height
                resolved = child._clamp_layout_extent(share, minimum, maximum)
                if abs(resolved - share) > 1e-6:
                    main_sizes[index] = resolved
                    constrained_total += resolved
                    constrained.append(index)
            if not constrained:
                for index in unresolved:
                    child = children[index]
                    main_sizes[index] = max(0.0, remaining) * max(
                        0.001, float(child.layout_weight)
                    ) / total_weight
                break
            unresolved = [index for index in unresolved if index not in constrained]
            remaining = max(0.0, remaining - constrained_total)

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
            cross_mode = child.height_sizing if horizontal else child.width_sizing
            cross_available = inner_h if horizontal else inner_w
            desired_cross = desired_h if horizontal else desired_w
            cross_size = cross_available if cross_mode == UILayoutSizing.Fill else min(desired_cross, cross_available)
            if horizontal:
                cross_size = child._clamp_layout_extent(cross_size, child.min_height, child.max_height)
            else:
                cross_size = child._clamp_layout_extent(cross_size, child.min_width, child.max_width)

            if self.align_items == UILayoutAlign.Center:
                cross_offset = (cross_available - cross_size) * 0.5
            elif self.align_items == UILayoutAlign.End:
                cross_offset = cross_available - cross_size
            elif self.align_items == UILayoutAlign.Stretch:
                cross_offset = 0.0
                if horizontal:
                    cross_size = child._clamp_layout_extent(
                        cross_available, child.min_height, child.max_height
                    )
                else:
                    cross_size = child._clamp_layout_extent(
                        cross_available, child.min_width, child.max_width
                    )
            else:
                cross_offset = 0.0

            if horizontal:
                rect = (cursor, inner_y + cross_offset, main_size, cross_size)
            else:
                rect = (inner_x + cross_offset, cursor, cross_size, main_size)
            rects[child] = rect
            cursor += main_size + effective_gap

        return rects
