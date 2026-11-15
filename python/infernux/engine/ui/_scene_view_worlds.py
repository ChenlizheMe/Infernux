"""Scene-owned World selection shared by editor views."""
from time import perf_counter

from infernux.engine.i18n import t
from infernux.engine.play_mode import PlayModeState
from .dpi import editor_dpi_scale
from .theme import Theme, ImGuiCol, ImGuiStyleVar


class WorldViewState:
    def __init__(self):
        self.rows = ()
        self.ordinals = {}
        self.world_id = 0
        self.mode = None
        self.layout = 0
        self.input_mode = 0
        self.fps = self.frame_ms = 0.0
        self._sample_time = self._sample_frame = None
        self._next_sample = 0.0

    def prepare(self, manager, engine):
        self.mode = manager.state
        if manager.is_edit_mode:
            from infernux.lib import SceneManager
            self.rows = ((int(SceneManager.instance()._authoring_runtime_world_id), 'ready', '', True),)
        elif manager.state in (PlayModeState.PLAYING, PlayModeState.PAUSED, PlayModeState.ENTERING):
            from infernux import WorldManager
            self.rows = tuple((world.id, world.state, world.error, world.display_in_game)
                              for world in WorldManager.get_world_statuses() if not world.destroy_requested)
        else:
            self.rows = ()
        self.ordinals = {world: index + 1 for index, world in enumerate(
            world for world, state, _error, _display in self.rows if state in ('ready', 'preparing'))}
        visible = tuple(world for world, state, _error, display in self.rows
                        if display and state in ('ready', 'preparing'))
        if self.world_id not in visible:
            self.world_id = visible[0] if visible else 0
        now = perf_counter()
        if now >= self._next_sample:
            snapshot = engine.get_native_engine().renderer_frame_snapshot
            frame = int(snapshot['frame'])
            if self._sample_frame is not None and frame >= self._sample_frame:
                elapsed = now - self._sample_time
                frames = frame - self._sample_frame
                self.fps = frames / elapsed
                self.frame_ms = elapsed * 1000.0 / frames if frames else 0.0
            self._sample_time, self._sample_frame = now, frame
            self._next_sample = now + 1.0

    @property
    def fps_text(self):
        return f'FPS: {self.fps:.0f} ({self.frame_ms:.1f} ms)' if self.fps else 'FPS: --'

    def select(self, world_id):
        world_id = int(world_id)
        if not any(world == world_id and display and state in ('ready', 'preparing')
                   for world, state, _error, display in self.rows):
            raise ValueError(f'World {world_id} is not available for viewing')
        self.world_id = world_id


class SceneViewWorldsMixin:
    @property
    def world_view(self):
        return self._world_view

    @property
    def viewed_world_id(self):
        """Lifetime World ID viewed by Scene and Game, independent of the primary."""
        return self._world_view.world_id

    @viewed_world_id.setter
    def viewed_world_id(self, world_id):
        self._world_view.select(world_id)

    def _draw_world_controls(self, ctx, vp, cursor_x, cursor_y, scene_height):
        dpi = editor_dpi_scale(ctx)
        old_x, old_y = ctx.get_cursor_pos_x(), ctx.get_cursor_pos_y()
        ctx.set_cursor_pos_x(cursor_x + 8.0 * dpi)
        ctx.set_cursor_pos_y(cursor_y + scene_height - 34.0 * dpi)
        view = self._world_view
        ordinal = view.ordinals.get(view.world_id)
        label = f'{t("scene_view.world")} {ordinal} / {len(view.ordinals)}' if ordinal is not None else t('game_view.no_worlds')
        ctx.push_style_color(ImGuiCol.Button, *Theme.SCENE_OVERLAY_COMBO_BG)
        ctx.push_style_var_float(ImGuiStyleVar.FrameRounding, Theme.SCENE_OVERLAY_ROUNDING * dpi)
        if ctx.button(label + '##scene_world_controls'):
            ctx.open_popup('SceneWorldSettings')
        ctx.record_semantic_item('button', label, True, 'scene_view.world_controls')
        hovered = ctx.is_item_hovered()
        ctx.pop_style_var(1)
        ctx.pop_style_color(1)
        ctx.same_line()
        ctx.label(view.fps_text)
        if ctx.begin_popup('SceneWorldSettings'):
            self._draw_world_settings(ctx, dpi)
            ctx.end_popup()
            hovered = True
        ctx.set_cursor_pos_x(old_x)
        ctx.set_cursor_pos_y(old_y)
        return hovered

    def _draw_world_settings(self, ctx, dpi):
        from infernux import WorldManager
        view, manager = self._world_view, self._play_mode_manager
        active = manager.state in (PlayModeState.PLAYING, PlayModeState.PAUSED)
        if not self._world_count_editing:
            self._world_count_draft = len(view.ordinals) if active else manager.initial_world_count
        ctx.set_next_item_width(170.0 * dpi)
        self._world_count_draft = int(ctx.input_int(t('game_view.worlds') + '##scene_world_count', self._world_count_draft))
        committed = ctx.is_item_deactivated_after_edit()
        self._world_count_editing = ctx.is_item_active()
        if committed and manager.state != PlayModeState.ENTERING:
            target = max(0 if active else 1, self._world_count_draft)
            if active:
                WorldManager.set_world_count(target)
            elif manager.is_edit_mode:
                manager.initial_world_count = target
        selectable = tuple(world for world, state, _error, display in view.rows
                           if display and state in ('ready', 'preparing'))
        if selectable:
            selected = selectable.index(view.world_id)
            ctx.set_next_item_width(170.0 * dpi)
            selected = ctx.combo('##scene_world_selected', selected,
                                 [f'{t("scene_view.world")} {view.ordinals[world]}' for world in selectable], -1)
            self.viewed_world_id = selectable[selected]
        ctx.set_next_item_width(170.0 * dpi)
        view.layout = ctx.combo('##scene_world_layout', view.layout,
                               [t('game_view.selected'), t('game_view.grid')], -1)
        ctx.set_next_item_width(170.0 * dpi)
        view.input_mode = ctx.combo('##scene_world_input', view.input_mode,
                                   [t('game_view.broadcast'), t('game_view.selected_input')], -1)
        ready = sum(state == 'ready' for _, state, _, _ in view.rows)
        preparing = sum(state == 'preparing' for _, state, _, _ in view.rows)
        ctx.label(f'{ready} / {preparing} ' + t('game_view.ready_preparing'))
        for _world, _state, error, _display in view.rows:
            if error:
                ctx.label(error)
