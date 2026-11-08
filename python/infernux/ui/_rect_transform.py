"""Connect UI declarations and batch transfers to the scene's rectangle owner."""

from infernux.batch import ComponentFieldBatch
from infernux.components.fields import SerializedFieldDescriptor
from infernux.lib import Vector2


class RectField(SerializedFieldDescriptor):
    """Serialized UI size/pivot backed by the Transform after layout binding."""

    _native_field_storage = True

    def get_raw(self, instance):
        layout = instance.__dict__.get('_native_rect_layout')
        if layout is None:
            return super().get_raw(instance)
        name = self.metadata.name
        return layout._authoring_rect[2 if name == 'width' else 3] if name != 'pivot' else layout.pivot

    def __get__(self, instance, owner):
        if instance is None:
            return self
        layout = instance.__dict__.get('_native_rect_layout')
        if layout is not None and self.metadata.name != 'pivot':
            return layout.rect[2 if self.metadata.name == 'width' else 3]
        return self.get_raw(instance)

    def __set__(self, instance, value):
        layout = instance.__dict__.get('_native_rect_layout')
        if layout is None:
            return super().__set__(instance, value)
        name = self.metadata.name
        if name == 'pivot':
            from infernux.components.fields import normalize_runtime_field_value
            layout.pivot = normalize_runtime_field_value(value, self.metadata)
        else:
            layout._set_authoring_size(0 if name == 'width' else 1, float(value))
        self._notify_runtime_value_changed(instance)

    def create_field_batch(self, targets):
        return RectFieldBatch(targets, self.metadata.name)


class RectProperty(property):
    """A derived rectangle property; canonical fields serialize with Transform."""

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, instance, owner):
        if instance is None:
            return self
        # A detached authoring declaration has a size but no parent/scene pose.
        if self.name == 'size_delta' and instance._try_get_game_object() is None:
            return Vector2(instance.width, instance.height)
        return getattr(bind_rect(instance), self.name)

    def __set__(self, instance, value):
        if self.name == 'size_delta' and instance._try_get_game_object() is None:
            instance.width, instance.height = value
            return
        setattr(bind_rect(instance), self.name, value)

    def create_field_batch(self, targets):
        return RectFieldBatch(targets, self.name)


def rect_field(declaration):
    return RectField(declaration.metadata)


class RectFieldBatch(ComponentFieldBatch):
    """Resolve owners once; numeric row reads/writes execute in native code."""

    __slots__ = ('_native', '_shape')

    def __init__(self, targets, name):
        from infernux.lib._Infernux import _RectLayoutBatch
        from infernux.batch import _get_np
        self.targets = tuple(targets)
        self.prop_name = name
        self._native = _RectLayoutBatch([bind_rect(target) for target in targets], name)
        self._shape = (len(targets),) if self._native.channels == 1 else (len(targets), self._native.channels)
        self.dtype = _get_np().dtype('float64' if self._native.channels == 1 else 'float32')

    def read(self):
        from infernux.batch import _get_np
        output = _get_np().empty(self._shape, dtype=self.dtype)
        self._native.read_into(output)
        return output

    def write(self, data):
        from infernux.batch import _get_np
        from infernux.components._component_lifecycle import notify_runtime_component_values
        from .inx_ui_screen_component import _invalidate_rect_cache
        from .ui_render_revision import mark_runtime_ui_dirty
        np = _get_np()
        values = np.asarray(data)
        if values.ndim != len(self._shape) or values.shape[1:] != self._shape[1:] or values.shape[0] < len(self.targets):
            raise ValueError(f'batch data must have shape {self._shape}, with at least {len(self.targets)} rows')
        if not np.can_cast(values.dtype, self.dtype, casting='safe'):
            raise ValueError(f'batch data cannot be safely converted from {values.dtype} to {self.dtype}')
        self._native.write(np.ascontiguousarray(values[:len(self.targets)], dtype=self.dtype))
        notify_runtime_component_values(self.targets, self.prop_name)
        _invalidate_rect_cache()
        for target in self.targets:
            mark_runtime_ui_dirty(target, subtree=True)


def bind_rect(component):
    """Materialize authored UI geometry once; subsequent access shares its owner."""
    cached = component.__dict__.get('_native_rect_layout')
    if cached is not None:
        return cached
    owner = component._try_get_game_object()
    if owner is None:
        raise RuntimeError('Rectangle layout requires an attached GameObject')
    from .inx_ui_screen_component import InxUIScreenComponent, _invalidate_rect_cache
    from .ui_canvas import UICanvas
    transform = owner.transform
    canvas = component.get_canvas()
    if canvas is not None:
        canvas.__dict__['_layout_rect_member'] = component
    if not transform._has_rect_layout:
        # Activate the path to its Canvas/world boundary before deriving anchors.
        parent = owner.get_parent()
        if parent is not None:
            parent_components = parent.get_py_components()
            parent_ui = next((c for c in parent_components if isinstance(c, InxUIScreenComponent)), None)
            parent_canvas = next((c for c in parent_components if isinstance(c, UICanvas)), None)
            if parent_canvas is not None:
                _bind_canvas(parent_canvas)
            elif parent_ui is not None:
                bind_rect(parent_ui)
            else:
                _bind_bridge(parent, canvas)
        position = transform.local_position
        width, height, pivot = float(component.width), float(component.height), component.pivot
        layout = transform._rect_layout
        layout._unit_mode = 1
        layout.pivot = pivot
        from .enums import ScreenAlignH, ScreenAlignV
        horizontal = {ScreenAlignH.Left: 0., ScreenAlignH.Center: .5, ScreenAlignH.Right: 1.}[component.align_h]
        vertical = {ScreenAlignV.Top: 1., ScreenAlignV.Center: .5, ScreenAlignV.Bottom: 0.}[component.align_v]
        layout.anchor_min = layout.anchor_max = Vector2(horizontal, vertical)
        layout.size_delta = Vector2(width, height)
        layout._set_authoring_local_position(position)
    else:
        layout = transform._rect_layout
        layout._unit_mode = 1
    # Several visuals on one object share one rectangle, not private mirrors.
    for sibling in owner.get_py_components():
        if isinstance(sibling, InxUIScreenComponent):
            sibling.__dict__['_native_rect_layout'] = layout
            values = sibling.__dict__.get('_field_values', {})
            for name in ('width', 'height', 'pivot'):
                values.pop(name, None)
    component.__dict__['_native_rect_layout'] = layout
    _invalidate_rect_cache()
    return layout


def _bind_canvas(canvas):
    transform = canvas.game_object.transform
    layout = transform._rect_layout
    layout._unit_mode = 2
    layout._set_authoring_size(0, float(canvas.reference_width))
    layout._set_authoring_size(1, float(canvas.reference_height))
    return layout


def _bind_bridge(owner, canvas):
    """Non-visual hierarchy nodes transmit their parent's rectangle and pose."""
    from .inx_ui_screen_component import InxUIScreenComponent
    from .ui_canvas import UICanvas
    transform = owner.transform
    if transform._has_rect_layout:
        return transform._rect_layout
    parent = owner.get_parent()
    if parent is not None:
        components = parent.get_py_components()
        parent_canvas = next((c for c in components if isinstance(c, UICanvas)), None)
        parent_ui = next((c for c in components if isinstance(c, InxUIScreenComponent)), None)
        if parent_canvas is not None:
            _bind_canvas(parent_canvas)
        elif parent_ui is not None:
            bind_rect(parent_ui)
        else:
            _bind_bridge(parent, canvas)
    position = transform.local_position
    layout = transform._rect_layout
    layout._unit_mode = 1
    layout.anchor_min, layout.anchor_max = Vector2(0, 0), Vector2(1, 1)
    layout.size_delta = Vector2(0, 0)
    layout._set_authoring_local_position(position)
    return layout


def screen_rect(component, width, height, reference_rect=None, size=None):
    """Evaluate a viewport without changing the authored native scene pose."""
    layout = bind_rect(component)
    canvas = component.get_canvas()
    boundary = canvas.game_object if canvas is not None else None
    frame, parent_pivot = _screen_parent(component.game_object.get_parent(), boundary, width, height)
    if reference_rect is not None:
        frame = reference_rect
    px, py, pw, ph = frame
    rect, position = layout._resolve_parent(Vector2(pw, ph), parent_pivot)
    rw, rh = size if size is not None else (
        component._clamp_layout_extent(rect.z, component.min_width, component.max_width),
        component._clamp_layout_extent(rect.w, component.min_height, component.max_height))
    from .enums import UILayoutSizing
    if size is None and (component.width_sizing == UILayoutSizing.Hug or component.height_sizing == UILayoutSizing.Hug):
        desired = component._layout_desired_size()
        rw = desired[0] if component.width_sizing == UILayoutSizing.Hug else rw
        rh = desired[1] if component.height_sizing == UILayoutSizing.Hug else rh
    pivot = layout.pivot
    # Logical rectangle coordinates have Y up; screen coordinates have Y down.
    return (px + pw * parent_pivot.x + position.x - rw * pivot.x,
            py + ph * (1 - parent_pivot.y) - position.y - rh * (1 - pivot.y), rw, rh)


def _screen_parent(owner, boundary, width, height):
    if owner is None or (boundary is not None and owner.id == boundary.id):
        return (0., 0., float(width), float(height)), Vector2(.5, .5)
    from .inx_ui_screen_component import InxUIScreenComponent
    parent_ui = next((c for c in owner.get_py_components() if isinstance(c, InxUIScreenComponent)), None)
    if parent_ui is not None:
        return parent_ui.get_rect(width, height), parent_ui.pivot
    frame, pivot = _screen_parent(owner.get_parent(), boundary, width, height)
    px, py, pw, ph = frame
    layout = owner.transform._rect_layout
    rect, position = layout._resolve_parent(Vector2(pw, ph), pivot)
    return (px + pw * pivot.x + position.x + rect.x,
            py + ph * (1 - pivot.y) - position.y - rect.y - rect.w, rect.z, rect.w), layout.pivot
