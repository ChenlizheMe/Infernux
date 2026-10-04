"""Retained native UI geometry, owned by one submission target.

Element edits dirty their subscribed groups. Unchanged geometry is published
as an ordered batch, without per-element measurement/key polling each frame.
"""

from weakref import WeakSet

from .ui_render_dispatch import _can_retain_runtime_commands, resolve_text_layout
from .inx_ui_screen_component import _get_layout_revision


class _CommandGroup:
    def __init__(self, elements):
        self.elements = tuple(elements)
        self.indices = {element: index for index, element in enumerate(self.elements)}
        self.geometry = [None] * len(self.elements)
        self.custom = {index for index, element in enumerate(self.elements)
                       if not _can_retain_runtime_commands(element)}
        self.dirty = set(range(len(self.elements)))
        self.layout_revision = None
        for element in self.elements:
            state = element.__dict__
            if '_ui_command_groups' not in state:
                state['_ui_command_groups'] = WeakSet()
            state['_ui_command_groups'].add(self)

    def invalidate(self, element):
        self.dirty.add(self.indices[element])

    def capture(self, index, renderer, draw, args):
        # Consume this edit before drawing: edits made by the draw itself must
        # still reach the next publication, not be erased on its return.
        self.dirty.discard(index)
        renderer.begin_command_packet()
        try:
            draw(self.elements[index], renderer, *args)
            self.geometry[index] = renderer.end_command_packet()
        except BaseException:
            renderer.abort_command_packet()
            raise


class UICommandPackets:
    def __init__(self):
        self.epoch = None
        self.groups = {}
        self.pending = []
        self.font_epoch = None

    def prepare(self, epoch, *, font_epoch=None):
        self.pending.clear()
        if font_epoch is not None:
            self.font_epoch = font_epoch
        if epoch != self.epoch:
            self.groups.clear()
            self.epoch = epoch

    def measure(self, element, renderer, scale):
        """Measure changed text before any sibling layout is consumed."""
        state = element.__dict__
        if state.get('_text_layout_font_epoch') != self.font_epoch:
            state.pop('_text_layout_key', None)
            state['_text_layout_font_epoch'] = self.font_epoch
        resolve_text_layout(element, renderer.measure_text, scale)

    def submit_elements(self, elements, renderer, draw, *args, scope=None, scale=1.0):
        # Membership, canvas settings and target dimensions belong to epoch.
        group = self.groups.get(scope)
        if group is None:
            group = self.groups[scope] = _CommandGroup(elements)
        try:
            for index in sorted(group.dirty | group.custom):
                self.measure(group.elements[index], renderer, scale)
            layout_revision = _get_layout_revision()
            if group.layout_revision != layout_revision:
                group.dirty.update(range(len(group.elements)))
                group.layout_revision = layout_revision

            if not group.custom:
                for index in sorted(group.dirty):
                    group.capture(index, renderer, draw, args)
                self.pending.extend(group.geometry)
                return

            # Custom draws keep their authored position and real renderer.
            # They may mutate a later sibling; observe that edit before it is
            # submitted, just as the ordinary sequential draw path does.
            for index, element in enumerate(group.elements):
                if index in group.custom:
                    self.flush(renderer)
                    group.dirty.discard(index)
                    draw(element, renderer, *args)
                    layout_revision = _get_layout_revision()
                    if layout_revision != group.layout_revision:
                        group.dirty.update(range(len(group.elements)))
                        group.layout_revision = layout_revision
                else:
                    if index in group.dirty:
                        group.capture(index, renderer, draw, args)
                    self.pending.append(group.geometry[index])
        except BaseException:
            # Discard an incomplete publication; do not keep stale packets as
            # a fallback or retry the failed draw here.
            self.groups.clear()
            self.pending.clear()
            raise

    def flush(self, renderer):
        if self.pending:
            renderer.append_command_packets(self.pending)
            self.pending.clear()
