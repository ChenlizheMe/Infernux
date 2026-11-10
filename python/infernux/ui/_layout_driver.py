"""Publish resolved UI rectangles into the scene's derived layout channels."""
from infernux.lib import Vector2
from infernux.lib._Infernux import _rect_layout_input_revision

from ._rect_transform import bind_rect, _bind_canvas
from .enums import UILayoutDirection
from .inx_ui_screen_component import _get_layout_revision, _invalidate_rect_cache


def _layout_key(elements, width, height):
    return elements, _get_layout_revision(), _rect_layout_input_revision(), float(width), float(height)


def release_frame(frame):
    """Release only this container's children; sibling drivers stay intact."""
    driver = frame.__dict__.get('_native_layout_owner')
    targets = frame.__dict__.pop('_native_driven_children', {})
    changed = False
    if driver is not None and bool(driver):
        for transform, layout in targets.values():
            if bool(transform):
                changed |= layout._clear_driver(driver)
    if changed:
        _invalidate_rect_cache()
    return changed


def release_layout(owner):
    """Restore the authored channels owned by a canvas or a world UI root."""
    targets = owner.__dict__.pop('_native_layout_targets', {})
    changed = False
    for driver, transform, layout in targets.values():
        if bool(transform) and bool(driver):
            changed |= layout._clear_driver(driver)
    root = owner.__dict__.pop('_native_viewport_driver', None)
    if root is not None and bool(root):
        changed |= root._rect_layout._clear_driver(root)
    owner.__dict__.pop('_native_layout_key', None)
    if changed:
        _invalidate_rect_cache()
    return changed


def _publish_groups(owner, frames, width, height):
    from .ui_frame import UIFrame
    from .ui_text import UIText
    active = [frame for frame in frames if isinstance(frame, UIFrame) and frame.enabled
              and frame.game_object.active_in_hierarchy and frame.layout_direction != UILayoutDirection.None_]
    wanted = {}
    for frame in active:
        driver = frame.transform
        for child in frame._flow_children():
            wanted[child.game_object.id] = driver, child.transform, bind_rect(child)
    intrinsic = []
    for element in frames:
        if (isinstance(element, UIText) and element.enabled and element.game_object.active_in_hierarchy
                and element.game_object.id not in wanted
                and element.transform is not owner.__dict__.get('_native_viewport_driver')
                and element.__dict__.get('_text_intrinsic_size') is not None):
            mask = 1 if element.is_auto_width() else 2 if element.is_auto_height() else 0
            if mask:
                transform, layout = element.transform, bind_rect(element)
                wanted[element.game_object.id] = transform, transform, layout
                intrinsic.append((element, layout, mask))
    previous = owner.__dict__.get('_native_layout_targets', {})
    changed = False
    for identifier, (driver, transform, layout) in previous.items():
        current = wanted.get(identifier)
        if current is None or current[0] is not driver:
            if bool(transform) and bool(driver):
                changed |= layout._clear_driver(driver)
    for element, layout, mask in intrinsic:
        changed |= layout._drive_size(element.transform, Vector2(*element.get_resolved_size()), mask)
    for frame in active:
        rect = frame.get_rect(width, height) if width is not None else frame.get_rect()
        fx, fy, fw, fh = rect
        pivot = frame.pivot
        children = {}
        for child, (x, y, w, h) in frame._arrange_flow_children(rect).items():
            child_pivot = child.pivot
            position = Vector2(x - fx - fw * pivot.x + w * child_pivot.x,
                               fy + fh * (1 - pivot.y) - y - h * (1 - child_pivot.y))
            layout = bind_rect(child)
            changed |= layout._drive_rect(frame.transform, Vector2(w, h), position)
            children[child.game_object.id] = child.transform, layout
        frame.__dict__['_native_layout_owner'] = frame.transform
        frame.__dict__['_native_driven_children'] = children
    owner.__dict__['_native_layout_targets'] = wanted
    return changed


def _publish_layout(owner, elements, width, height, measure_text, text_scale):
    changed = _publish_groups(owner, elements, width, height)
    if measure_text is not None:
        from .ui_text import UIText
        # First assign widths, then measure wrapped text against those widths.
        # The final pass publishes vertical sizes and dependent sibling poses.
        reflow = False
        for element in elements:
            if isinstance(element, UIText) and element.enabled and element.game_object.active_in_hierarchy:
                reflow |= element.resolve_text_layout(measure_text, text_scale)
        if reflow:
            _invalidate_rect_cache(layout=False)
            changed |= _publish_groups(owner, elements, width, height)
    return changed


def rebuild_canvas(canvas, width=None, height=None, *, force=False, measure_text=None, text_scale=1.0):
    width = float(canvas.reference_width if width is None else width)
    height = float(canvas.reference_height if height is None else height)
    elements = canvas._get_elements()
    key = canvas._cached_elements_version, _layout_key(elements, width, height), float(text_scale), measure_text is not None
    if not force and canvas.__dict__.get('_native_layout_key') == key:
        return False
    _invalidate_rect_cache(layout=False)
    root = _bind_canvas(canvas)
    changed = root._drive_size(canvas.transform, Vector2(width, height))
    canvas.__dict__['_native_viewport_driver'] = canvas.transform
    changed |= _publish_layout(canvas, elements, width, height, measure_text, text_scale)
    canvas.__dict__['_native_layout_key'] = (canvas._cached_elements_version,
                                             _layout_key(elements, width, height), float(text_scale), measure_text is not None)
    return changed


def rebuild_world(frame, *, force=False, measure_text=None):
    from .inx_ui_screen_component import InxUIScreenComponent
    from .ui_canvas import UICanvas
    root = frame.game_object
    if frame.__dict__.get('_layout_elements_owner') is not root:
        from infernux.lib._Infernux import _UIHierarchyDependencies
        frame.__dict__['_layout_hierarchy'] = _UIHierarchyDependencies(root)
        frame.__dict__['_layout_elements_owner'] = root
        frame.__dict__.pop('_layout_elements_topology', None)
    hierarchy = frame.__dict__['_layout_hierarchy']
    policy, topology = hierarchy.poll(), hierarchy.topology_revision
    if frame.__dict__.get('_layout_elements_topology') != topology:
        elements = []
        pending = [root]
        while pending:
            owner = pending.pop()
            components = owner.get_py_components()
            if owner is not root and any(isinstance(c, UICanvas) for c in components):
                continue
            elements.extend(c for c in components if isinstance(c, InxUIScreenComponent))
            pending.extend(reversed(owner.get_children()))
        frame.__dict__['_layout_elements'] = tuple(elements)
        frame.__dict__['_layout_elements_topology'] = topology
    elements = frame.__dict__['_layout_elements']
    key = policy, _layout_key(elements, 0, 0), measure_text is not None
    if not force and frame.__dict__.get('_native_layout_key') == key:
        return False
    _invalidate_rect_cache(layout=False)
    for element in elements:
        bind_rect(element)
    changed = _publish_layout(frame, elements, None, None, measure_text, 1.0)
    frame.__dict__['_native_layout_key'] = policy, _layout_key(elements, 0, 0), measure_text is not None
    return changed
