"""Retained native UI geometry, owned by one submission target.

Element edits dirty their subscribed groups. Unchanged geometry is published
as an ordered batch, without per-element measurement/key polling each frame.
"""

from weakref import WeakSet

from .ui_render_dispatch import _can_retain_runtime_commands, resolve_text_layout
from .inx_ui_screen_component import _get_layout_revision


class _CommandGroup:
    def __init__(self, elements, texture_cache):
        self.elements = tuple(elements)
        self.indices = {element: index for index, element in enumerate(self.elements)}
        self.geometry = [None] * len(self.elements)
        self.custom = {index for index, element in enumerate(self.elements)
                       if not _can_retain_runtime_commands(element)}
        self.dirty = set(range(len(self.elements)))
        self.opacities = [1.0] * len(self.elements)
        self.opacity_dirty = self.dirty - self.custom
        self.layout_revision = None
        self.texture_cache = texture_cache
        self.textures = {}
        for element in self.elements:
            state = element.__dict__
            if '_ui_command_groups' not in state:
                state['_ui_command_groups'] = WeakSet()
            state['_ui_command_groups'].add(self)

    def invalidate(self, element):
        index = self.indices[element]
        self.dirty.add(index)
        if index not in self.custom:
            self.opacity_dirty.add(index)

    def invalidate_alpha(self, element):
        index = self.indices[element]
        if index in self.custom:
            self.dirty.add(index)
        else:
            self.opacity_dirty.add(index)

    def update_opacities(self):
        for index in self.opacity_dirty:
            self.opacities[index] = self.elements[index].get_effective_group_state()[0]
        self.opacity_dirty.clear()

    def capture(self, index, renderer, draw, args):
        # Consume this edit before drawing: edits made by the draw itself must
        # still reach the next publication, not be erased on its return.
        self.dirty.discard(index)
        renderer.begin_command_packet()
        self.texture_cache.begin_command_packet()
        try:
            if index in self.custom:
                draw(self.elements[index], renderer, *args)
            else:
                draw(self.elements[index], renderer, *args, defer_group_alpha=True)
            self.geometry[index] = renderer.end_command_packet()
        except BaseException:
            renderer.abort_command_packet()
            raise
        finally:
            bindings = self.texture_cache.end_command_packet()
            if bindings:
                self.textures[index] = bindings
            else:
                self.textures.pop(index, None)


class UICommandPackets:
    def __init__(self, texture_cache):
        self.epoch = None
        self.groups = {}
        self.pending = []
        self.pending_opacities = []
        self.font_epoch = None
        self.texture_cache = texture_cache
        self.texture_generation = None
        self.has_pending_textures = False

    def prepare(self, epoch, *, font_epoch=None):
        self.pending.clear()
        self.pending_opacities.clear()
        if font_epoch is not None:
            self.font_epoch = font_epoch
        if epoch != self.epoch:
            self.groups.clear()
            self.epoch = epoch
        cache = self.texture_cache
        if cache.has_pending or cache.generation != self.texture_generation:
            self.has_pending_textures = False
            for group in self.groups.values():
                for index, bindings in group.textures.items():
                    self.has_pending_textures |= any(state[1] for state in bindings.values())
                    if any(state[1] or cache.binding_state(guid) != state
                           for guid, state in bindings.items()):
                        group.dirty.add(index)
            self.texture_generation = cache.generation

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
            group = self.groups[scope] = _CommandGroup(elements, self.texture_cache)
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
                group.update_opacities()
                self.pending.extend(group.geometry)
                self.pending_opacities.extend(group.opacities)
                return

            # Custom draws keep their authored position and real renderer.
            # They may mutate a later sibling; observe that edit before it is
            # submitted, just as the ordinary sequential draw path does.
            for index, element in enumerate(group.elements):
                if index in group.custom:
                    self.flush(renderer)
                    # Native object/transform bindings require a command
                    # packet even for authored draws. Record and enqueue it
                    # here every submission; do not retain it as a clean draw.
                    group.capture(index, renderer, draw, args)
                    self.pending.append(group.geometry[index])
                    self.pending_opacities.append(1.0)
                    # A custom draw may change any text's intrinsic size.
                    # Publish those sizes before the next sibling consumes
                    # layout, including text already drawn earlier this frame.
                    # Clean groups do no extra work; the text layout key keeps
                    # unchanged glyph measurements cached.
                    for dirty_index in sorted(group.dirty):
                        self.measure(group.elements[dirty_index], renderer, scale)
                    layout_revision = _get_layout_revision()
                    if layout_revision != group.layout_revision:
                        group.dirty.update(range(len(group.elements)))
                        group.layout_revision = layout_revision
                else:
                    if index in group.dirty:
                        group.capture(index, renderer, draw, args)
                    group.update_opacities()
                    self.pending.append(group.geometry[index])
                    self.pending_opacities.append(group.opacities[index])
        except BaseException:
            # Discard an incomplete publication; do not keep stale packets as
            # a fallback or retry the failed draw here.
            self.groups.clear()
            self.pending.clear()
            self.pending_opacities.clear()
            raise

    def flush(self, renderer):
        if self.pending:
            renderer.append_command_packets(self.pending, self.pending_opacities)
            self.pending.clear()
            self.pending_opacities.clear()
