"""
Animation State Machine Editor — visual node-graph editor for .animfsm files.

Displays states as nodes with connections representing transitions.
Drag from an output pin to an input pin to create a transition.
Click a node to edit its properties in the right-side inspector.
Opened from the Animation menu or by double-clicking a .animfsm file
in the Project panel.
"""

from __future__ import annotations

import os
import threading
from typing import Dict, List, Optional

from Infernux.core.anim_state_machine import AnimStateMachine, AnimState, AnimTransition
from Infernux.core.node_graph import (
    GraphLink,
    GraphNode,
    NodeGraph,
    NodeTypeDef,
    PinDef,
    PinKind,
)
from Infernux.debug import Debug
from Infernux.engine.i18n import t
from Infernux.lib import InxGUIContext

from .editor_panel import EditorPanel
from .igui import IGUI
from .node_graph_view import NodeGraphView
from .panel_registry import editor_panel
from .theme import Theme, ImGuiCol, ImGuiStyleVar


# ═══════════════════════════════════════════════════════════════════════════
# Node type definition for animation states
# ═══════════════════════════════════════════════════════════════════════════

_STATE_TYPE = NodeTypeDef(
    type_id="anim_state",
    label="State",
    header_color=(0.18, 0.40, 0.25, 1.0),
    pins=[
        PinDef(id="in", label="In", kind=PinKind.INPUT, color=(0.5, 0.8, 0.5, 1.0)),
        PinDef(id="out", label="Out", kind=PinKind.OUTPUT, color=(0.8, 0.8, 0.5, 1.0)),
    ],
    min_width=160.0,
)

_ENTRY_TYPE = NodeTypeDef(
    type_id="anim_entry",
    label="Entry",
    header_color=(0.45, 0.25, 0.15, 1.0),
    pins=[
        PinDef(id="out", label="", kind=PinKind.OUTPUT, color=(0.9, 0.6, 0.3, 1.0)),
    ],
    min_width=80.0,
    deletable=False,
)

_DETAIL_PANEL_W = 260.0


# ═══════════════════════════════════════════════════════════════════════════
# Panel
# ═══════════════════════════════════════════════════════════════════════════

@editor_panel(
    "Animation State Machine Editor",
    type_id="animfsm_editor",
    title_key="panel.animfsm_editor",
    menu_path="Animation",
)
class AnimFSMEditorPanel(EditorPanel):
    """Node-graph editor for animation state machines."""

    window_id = "animfsm_editor"

    def __init__(self):
        super().__init__(title="Animation State Machine Editor", window_id="animfsm_editor")
        self._fsm: Optional[AnimStateMachine] = None
        self._file_path: str = ""
        self._dirty: bool = False

        # Node graph
        self._graph = NodeGraph()
        self._graph.register_type(_STATE_TYPE)
        self._graph.register_type(_ENTRY_TYPE)

        self._view = NodeGraphView()
        self._view.graph = self._graph
        self._view.on_link_created = self._on_link_created
        self._view.on_link_deleted = self._on_link_deleted
        self._view.on_nodes_deleted = self._on_nodes_deleted
        self._view.on_node_add_request = self._on_node_add_request
        self._view.on_node_selected = self._on_node_selected
        self._view.on_canvas_drop = self._on_canvas_drop

        # Currently selected node uid
        self._selected_uid: str = ""

        # Maps: state name ↔ node uid
        self._name_to_uid: Dict[str, str] = {}
        self._uid_to_name: Dict[str, str] = {}

        # Entry node uid
        self._entry_uid: str = ""

        # Guard to avoid re-entrant selection clearing
        self._clearing_selection: bool = False

        # Start with a blank FSM
        self._new_fsm()

    # ── Public API ────────────────────────────────────────────────────

    def _open_animfsm(self, file_path: str):
        """Load an .animfsm file into the editor."""
        fsm = AnimStateMachine.load(file_path)
        if fsm is None:
            Debug.log_warning(f"Failed to load animfsm: {file_path}")
            return
        self._fsm = fsm
        self._file_path = file_path
        self._selected_uid = ""
        self._dirty = False
        for state in fsm.states:
            if not state.clip_guid and state.clip_path:
                state.clip_guid = self._resolve_guid(state.clip_path)
            if state.clip_guid:
                state.clip_path = ""
        self._sync_graph_from_fsm()

    def _new_fsm(self):
        """Create a blank FSM for editing."""
        self._fsm = AnimStateMachine(name="New State Machine")
        self._file_path = ""
        self._selected_uid = ""
        self._dirty = False
        self._sync_graph_from_fsm()

    # ── Lifecycle ──────────────────────────────────────────────────────

    def on_enable(self) -> None:
        from Infernux.engine.ui.selection_manager import SelectionManager
        from .event_bus import EditorEventBus, EditorEvent
        SelectionManager.instance().add_listener(self._on_global_selection_changed)
        bus = EditorEventBus.instance()
        bus.subscribe(EditorEvent.FILE_SELECTED, self._on_file_selected)
        bus.subscribe(EditorEvent.ASSET_CHANGED, self._on_asset_changed)
        bus.subscribe("clip_name_changed", self._on_clip_name_changed)
        try:
            from Infernux.engine.play_mode import PlayModeManager
            pmm = PlayModeManager.instance()
            if pmm:
                pmm.add_state_change_listener(self._on_play_mode_changed)
        except Exception:
            pass

    def on_disable(self) -> None:
        from Infernux.engine.ui.selection_manager import SelectionManager
        from .event_bus import EditorEventBus, EditorEvent
        SelectionManager.instance().remove_listener(self._on_global_selection_changed)
        bus = EditorEventBus.instance()
        bus.unsubscribe(EditorEvent.FILE_SELECTED, self._on_file_selected)
        bus.unsubscribe(EditorEvent.ASSET_CHANGED, self._on_asset_changed)
        bus.unsubscribe("clip_name_changed", self._on_clip_name_changed)
        try:
            from Infernux.engine.play_mode import PlayModeManager
            pmm = PlayModeManager.instance()
            if pmm:
                pmm.remove_state_change_listener(self._on_play_mode_changed)
        except Exception:
            pass

    def _window_title_suffix(self) -> str:
        return " *" if self._dirty else ""

    # ── State persistence ──────────────────────────────────────────────

    def save_state(self) -> dict:
        data: dict = {}
        if self._file_path:
            data["file_path"] = self._file_path
        if self._view:
            data["pan_x"] = self._view.pan_x
            data["pan_y"] = self._view.pan_y
            data["zoom"] = self._view.zoom
        return data

    def load_state(self, data: dict) -> None:
        file_path = data.get("file_path", "")
        if file_path and os.path.isfile(file_path):
            self._open_animfsm(file_path)
        if self._view:
            self._view.pan_x = float(data.get("pan_x", self._view.pan_x))
            self._view.pan_y = float(data.get("pan_y", self._view.pan_y))
            self._view.zoom = float(data.get("zoom", self._view.zoom))

    def _initial_size(self):
        return (900, 600)

    def _empty_state_hint(self) -> str:
        return t("animfsm_editor.open_hint")

    def _empty_state_drop_types(self):
        return ["ANIMFSM_FILE"]

    def _on_empty_state_drop(self, payload_type, payload):
        if payload_type == "ANIMFSM_FILE" and payload:
            self._open_animfsm(payload)

    # ═══════════════════════════════════════════════════════════════════
    # Rendering
    # ═══════════════════════════════════════════════════════════════════

    # ImGuiKey / ImGuiMod constants
    _IMGUI_MOD_CTRL = 1 << 12  # 4096
    _IMGUI_KEY_S = 564

    def on_render_content(self, ctx: InxGUIContext):
        # Ctrl+S save shortcut
        if (ctx.is_key_down(self._IMGUI_MOD_CTRL)
                and ctx.is_key_pressed(self._IMGUI_KEY_S)
                and self._fsm is not None):
            self._do_save()

        self._render_toolbar(ctx)
        ctx.separator()

        avail_w = ctx.get_content_region_avail_width()
        avail_h = ctx.get_content_region_avail_height()

        # Left: node graph canvas | Right: detail panel
        detail_w = min(_DETAIL_PANEL_W, avail_w * 0.35)
        graph_w = avail_w - detail_w - 4

        # Graph canvas
        if ctx.begin_child("##fsm_graph_region", graph_w, avail_h, False):
            self._view.render(ctx)
        ctx.end_child()

        ctx.same_line()

        # Detail panel
        if ctx.begin_child("##fsm_detail_region", detail_w, avail_h, True):
            self._render_detail_panel(ctx)
        ctx.end_child()

        # Accept .animfsm file drops
        payload = ctx.accept_drag_drop_payload("ANIMFSM_FILE")
        if payload:
            self._open_animfsm(payload)

    # ── Toolbar ───────────────────────────────────────────────────────

    def _render_toolbar(self, ctx: InxGUIContext):
        fsm = self._fsm
        if fsm is None:
            return

        if ctx.button(t("animfsm_editor.new")):
            self._new_fsm()
            return

        ctx.same_line()
        ctx.label(f"{t('animfsm_editor.name')}:")
        ctx.same_line()
        ctx.set_next_item_width(140)
        new_name = ctx.text_input("##fsm_name", fsm.name, 128)
        if new_name != fsm.name:
            fsm.name = new_name
            self._dirty = True

        ctx.same_line()
        # 2D / 3D mode selector
        _MODES = ["2d", "3d"]
        mode_idx = _MODES.index(fsm.mode) if fsm.mode in _MODES else 0
        ctx.label(f"{t('animfsm_editor.mode')}:")
        ctx.same_line()
        ctx.set_next_item_width(60)
        new_mode_idx = ctx.combo("##fsm_mode", mode_idx, ["2D", "3D"], 2)
        if new_mode_idx != mode_idx:
            fsm.mode = _MODES[new_mode_idx]
            self._dirty = True

        ctx.same_line()
        save_label = t("animfsm_editor.save") if self._file_path else t("animfsm_editor.save_as")
        if ctx.button(save_label):
            self._do_save()

        if self._file_path:
            ctx.same_line()
            ctx.label(self._file_path)

    # ── Detail panel (right side) ─────────────────────────────────────

    def _render_detail_panel(self, ctx: InxGUIContext):
        fsm = self._fsm
        if fsm is None:
            return

        node = self._graph.find_node(self._selected_uid)
        if node is None or node.type_id != "anim_state":
            ctx.label(t("animfsm_editor.no_state_selected"))
            return

        state_name = self._uid_to_name.get(node.uid, "")
        state = fsm.get_state(state_name)
        if state is None:
            return

        ctx.label(f"── {t('animfsm_editor.state_detail')} ──")

        # State name (read-only)
        ctx.label(t("animfsm_editor.state_name"))
        ctx.label(state.name)

        # Default state toggle
        is_default = (state.name == fsm.default_state)
        if not is_default:
            if ctx.button(t("animfsm_editor.set_default")):
                fsm.default_state = state.name
                self._update_entry_link()
                self._dirty = True
        else:
            ctx.label(f"[Default] {t('animfsm_editor.default_state')}")

        ctx.separator()

        # Clip reference (drop-only, no text input)
        ctx.label(t("animfsm_editor.clip_ref"))

        # Accept .animclip2d drops on the clip field
        if ctx.begin_drag_drop_target():
            payload = ctx.accept_drag_drop_payload("ANIMCLIP_FILE")
            if payload:
                self._assign_clip_to_state(state, payload, node)
            ctx.end_drag_drop_target()

        # Speed
        ctx.label(t("animfsm_editor.speed"))
        new_speed = ctx.drag_float("##speed", state.speed, 0.01, 0.0, 10.0)
        if new_speed != state.speed:
            state.speed = new_speed
            self._dirty = True

        # Loop
        new_loop = ctx.checkbox(f"{t('animfsm_editor.loop')}##state_loop", state.loop)
        if new_loop != state.loop:
            state.loop = new_loop
            self._dirty = True

        # Transitions (outgoing)
        ctx.separator()
        ctx.label(t("animfsm_editor.transitions"))

        outgoing_links = [
            lk for lk in self._graph.links
            if lk.source_node == node.uid and lk.source_pin == "out"
               and lk.target_pin == "in"
        ]
        remove_link_uid = ""
        for i, lk in enumerate(outgoing_links):
            target_name = self._uid_to_name.get(lk.target_node, "?")
            ctx.push_id(i)
            ctx.label(f"→ {target_name}")
            ctx.same_line()

            cond = lk.data.get("condition", "")
            new_cond = ctx.text_input("##cond", cond, 128)
            if new_cond != cond:
                lk.data["condition"] = new_cond
                self._sync_transition_condition(lk)
                self._dirty = True

            ctx.same_line()
            if ctx.button("X##del", width=24, height=20):
                remove_link_uid = lk.uid
            ctx.pop_id()

        if remove_link_uid:
            if self._remove_link_and_transition(remove_link_uid):
                self._dirty = True

    # ═══════════════════════════════════════════════════════════════════
    # FSM ↔ Graph synchronization
    # ═══════════════════════════════════════════════════════════════════

    def _sync_graph_from_fsm(self):
        """Rebuild the NodeGraph from the current AnimStateMachine."""
        self._graph.clear()
        self._name_to_uid.clear()
        self._uid_to_name.clear()
        self._entry_uid = ""

        fsm = self._fsm
        if fsm is None:
            return

        # Create entry node
        entry = self._graph.add_node("anim_entry", x=-100, y=50)
        entry.data["label"] = "Entry"
        self._entry_uid = entry.uid

        # Create state nodes
        y_offset = 0.0
        for state in fsm.states:
            px, py = state.position[0], state.position[1]
            if px == 0.0 and py == 0.0:
                px = 100.0
                py = y_offset
                y_offset += 80.0
            node = self._graph.add_node("anim_state", x=px, y=py)
            node.data["label"] = state.name
            node.data["subtitle"] = ""
            self._name_to_uid[state.name] = node.uid
            self._uid_to_name[node.uid] = state.name

        # Entry → default state link
        self._update_entry_link()

        # Create transition links
        for state in fsm.states:
            src_uid = self._name_to_uid.get(state.name, "")
            if not src_uid:
                continue
            for tr in state.transitions:
                dst_uid = self._name_to_uid.get(tr.target_state, "")
                if not dst_uid:
                    continue
                lk = self._graph.add_link(src_uid, "out", dst_uid, "in")
                if lk:
                    lk.data["condition"] = tr.condition

    def _sync_fsm_positions(self):
        """Write node positions back to FSM state.position fields."""
        fsm = self._fsm
        if fsm is None:
            return
        for state in fsm.states:
            uid = self._name_to_uid.get(state.name, "")
            node = self._graph.find_node(uid) if uid else None
            if node:
                state.position = [node.pos_x, node.pos_y]

    def _update_entry_link(self):
        """Ensure the entry node points to the current default state."""
        # Remove old entry links
        self._graph.links = [
            lk for lk in self._graph.links
            if lk.source_node != self._entry_uid
        ]
        fsm = self._fsm
        if fsm and fsm.default_state:
            dst_uid = self._name_to_uid.get(fsm.default_state, "")
            if dst_uid:
                self._graph.add_link(self._entry_uid, "out", dst_uid, "in")

    # ── Callbacks from NodeGraphView ──────────────────────────────────

    def _on_link_created(self, src_node: str, src_pin: str, dst_node: str, dst_pin: str):
        """User created a connection by dragging between pins."""
        # Entry node connections change the default state
        if src_node == self._entry_uid:
            target_name = self._uid_to_name.get(dst_node, "")
            if target_name and self._fsm:
                self._fsm.default_state = target_name
                self._update_entry_link()
                self._dirty = True
            return

        src_name = self._uid_to_name.get(src_node, "")
        dst_name = self._uid_to_name.get(dst_node, "")
        if not src_name or not dst_name or not self._fsm:
            return

        state = self._fsm.get_state(src_name)
        if state is None:
            return

        # Check for duplicate transition
        for tr in state.transitions:
            if tr.target_state == dst_name:
                return

        state.transitions.append(AnimTransition(target_state=dst_name))
        self._graph.add_link(src_node, src_pin, dst_node, dst_pin)
        self._dirty = True

    def _on_link_deleted(self, link_uid: str):
        if self._remove_link_and_transition(link_uid):
            self._dirty = True

    def _on_nodes_deleted(self, uids: List[str]):
        fsm = self._fsm
        if fsm is None:
            return
        for uid in uids:
            # Don't delete entry node
            if uid == self._entry_uid:
                continue
            name = self._uid_to_name.get(uid, "")
            if name:
                fsm.remove_state(name)
                del self._name_to_uid[name]
                del self._uid_to_name[uid]
            self._graph.remove_node(uid)
        if self._selected_uid in uids:
            self._selected_uid = ""
        self._update_entry_link()
        self._dirty = True

    def _on_node_add_request(self, type_id: str, x: float, y: float):
        if type_id != "anim_state":
            return
        fsm = self._fsm
        if fsm is None:
            return
        state = fsm.add_state()
        state.position = [x, y]
        node = self._graph.add_node("anim_state", x=x, y=y)
        node.data["label"] = state.name
        self._name_to_uid[state.name] = node.uid
        self._uid_to_name[node.uid] = state.name
        self._view.selected_nodes = [node.uid]
        self._selected_uid = node.uid
        self._update_entry_link()
        self._dirty = True

    def _on_node_selected(self, uid: str):
        self._selected_uid = uid
        # Participate in global single-selection: clear hierarchy/project
        self._clear_external_selection()

    def _clear_external_selection(self):
        """Clear hierarchy, scene, and project panel selection."""
        if self._clearing_selection:
            return
        self._clearing_selection = True
        try:
            from Infernux.engine.ui.selection_manager import SelectionManager
            from Infernux.engine.bootstrap import EditorBootstrap

            SelectionManager.instance().clear()
            bootstrap = EditorBootstrap.instance()
            pp = bootstrap.project_panel if bootstrap else None
            if pp:
                pp.clear_selection()
        finally:
            self._clearing_selection = False

    def _on_global_selection_changed(self):
        """SelectionManager listener — hierarchy/scene selected something."""
        if self._clearing_selection:
            return
        from Infernux.engine.ui.selection_manager import SelectionManager
        if SelectionManager.instance().get_ids():
            self._view.selected_nodes.clear()
            self._view.selected_link = ""
            self._selected_uid = ""

    def _on_file_selected(self, path):
        """EditorEvent.FILE_SELECTED — project panel selected a file."""
        if path:
            self._view.selected_nodes.clear()
            self._view.selected_link = ""
            self._selected_uid = ""

    def _on_play_mode_changed(self, event):
        """PlayModeEvent — auto-save dirty FSM before play."""
        from Infernux.engine.play_mode import PlayModeState
        if event.new_state == PlayModeState.PLAYING and self._dirty:
            self._do_save()

    def _on_clip_name_changed(self, clip_path: str, new_name: str):
        """A clip's display name was changed in the AnimClip2D editor."""
        if not clip_path or not new_name or not self._fsm:
            return
        guid = self._resolve_guid(clip_path)
        if not guid:
            return
        self._rename_states_for_clip(guid, new_name)

    def _on_asset_changed(self, file_path: str, event_type: str):
        """An asset was modified/renamed/deleted — update state names."""
        if not file_path or not self._fsm:
            return
        if file_path.endswith(".animclip2d") and event_type == "moved":
            new_stem = os.path.splitext(os.path.basename(file_path))[0]
            guid = self._resolve_guid(file_path)
            if guid:
                self._rename_states_for_clip(guid, new_stem)

    def _rename_states_for_clip(self, clip_guid: str, new_name: str):
        """Rename all states referencing *clip_guid* to *new_name*."""
        fsm = self._fsm
        if not fsm:
            return
        for state in fsm.states:
            if state.clip_guid != clip_guid or state.name == new_name:
                continue
            old_name = state.name
            # Avoid collisions
            if fsm.get_state(new_name) is not None:
                continue
            state.name = new_name
            uid = self._name_to_uid.pop(old_name, "")
            if uid:
                self._name_to_uid[new_name] = uid
                self._uid_to_name[uid] = new_name
                node = self._graph.find_node(uid)
                if node:
                    node.data["label"] = new_name
            if fsm.default_state == old_name:
                fsm.default_state = new_name
            for s in fsm.states:
                for tr in s.transitions:
                    if tr.target_state == old_name:
                        tr.target_state = new_name
            self._dirty = True

    def _on_canvas_drop(self, payload_type: str, payload: str, gx: float, gy: float):
        """Handle items dropped onto the node graph canvas."""
        if payload_type == "ANIMCLIP_FILE" and payload:
            # Check if dropped on an existing state node
            for uid, name in self._uid_to_name.items():
                node = self._graph.find_node(uid)
                if node and abs(node.pos_x - gx) < 80 and abs(node.pos_y - gy) < 40:
                    state = self._fsm.get_state(name) if self._fsm else None
                    if state:
                        self._assign_clip_to_state(state, payload, node)
                    return
            # Otherwise create a new state with this clip
            if self._fsm:
                state = self._fsm.add_state()
                state.position = [gx, gy]
                clip_name = os.path.splitext(os.path.basename(payload))[0]
                state.name = clip_name
                node = self._graph.add_node("anim_state", x=gx, y=gy)
                node.data["label"] = clip_name
                self._name_to_uid[state.name] = node.uid
                self._uid_to_name[node.uid] = state.name
                self._assign_clip_to_state(state, payload, node)
                self._view.selected_nodes = [node.uid]
                self._selected_uid = node.uid
                self._update_entry_link()

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _resolve_guid(path: str) -> str:
        """Resolve a file path to an asset GUID."""
        try:
            from Infernux.core.assets import AssetManager
            guid = AssetManager._get_guid_from_path(path)
            return guid or ""
        except Exception:
            return ""

    @staticmethod
    def _resolve_path(guid: str) -> str:
        """Resolve an asset GUID to a file path."""
        if not guid:
            return ""
        try:
            from Infernux.core.assets import AssetManager
            path = AssetManager._get_path_from_guid(guid)
            return path or ""
        except Exception:
            return ""

    def _assign_clip_to_state(self, state: AnimState, clip_path: str, node=None):
        """Assign a clip path/guid to a state and update the node subtitle."""
        state.clip_guid = self._resolve_guid(clip_path) if clip_path else ""
        state.clip_path = clip_path if clip_path and not state.clip_guid else ""
        self._dirty = True

    def _remove_link_and_transition(self, link_uid: str) -> bool:
        """Remove a graph link and the corresponding FSM transition."""
        lk = self._graph.find_link(link_uid)
        if lk is None:
            return False
        if lk.source_node == self._entry_uid:
            if self._fsm and self._uid_to_name.get(lk.target_node, "") == self._fsm.default_state:
                self._fsm.default_state = ""
            return self._graph.remove_link(link_uid)
        src_name = self._uid_to_name.get(lk.source_node, "")
        dst_name = self._uid_to_name.get(lk.target_node, "")
        if src_name and dst_name and self._fsm:
            state = self._fsm.get_state(src_name)
            if state:
                state.transitions = [
                    tr for tr in state.transitions if tr.target_state != dst_name
                ]
        return self._graph.remove_link(link_uid)

    def _sync_transition_condition(self, lk: GraphLink):
        """Sync condition from link.data back to FSM transition."""
        src_name = self._uid_to_name.get(lk.source_node, "")
        dst_name = self._uid_to_name.get(lk.target_node, "")
        if src_name and dst_name and self._fsm:
            state = self._fsm.get_state(src_name)
            if state:
                for tr in state.transitions:
                    if tr.target_state == dst_name:
                        tr.condition = lk.data.get("condition", "")
                        break

    # ── Save ──────────────────────────────────────────────────────────

    def _do_save(self):
        fsm = self._fsm
        if fsm is None:
            return
        # Sync node positions back before saving
        self._sync_fsm_positions()
        target = self._file_path or fsm.file_path
        if target:
            self._execute_save(target)
        else:
            self._show_save_as_dialog()

    def _show_save_as_dialog(self):
        from Infernux.engine.project_context import get_project_root
        root = get_project_root()
        initial_dir = os.path.join(root, "Assets") if root else "."
        safe_name = (self._fsm.name or "NewStateMachine").replace(" ", "_")
        default_filename = f"{safe_name}.animfsm"

        def _run():
            result = None
            try:
                from ._dialogs import save_file_dialog
                result = save_file_dialog(
                    title="Save Animation State Machine",
                    win32_filter="AnimFSM files (*.animfsm)\0*.animfsm\0All files (*.*)\0*.*\0\0",
                    initial_dir=initial_dir,
                    default_filename=default_filename,
                    default_ext="animfsm",
                    tk_filetypes=[("AnimFSM", "*.animfsm"), ("All Files", "*.*")],
                )
            except Exception as exc:
                Debug.log_warning(f"[AnimFSM] Save dialog error: {exc}")
            if result:
                self._execute_save(result)

        t = threading.Thread(target=_run, daemon=True)
        t.start()

    def _execute_save(self, target: str):
        fsm = self._fsm
        if fsm is None:
            return
        if fsm.save(target):
            self._file_path = target
            self._dirty = False
            Debug.log(f"Saved animfsm: {target}")
            self._hot_reload_animators(target)
        else:
            Debug.log_error(f"Failed to save animfsm: {target}")

    def _hot_reload_animators(self, fsm_path: str):
        """Reload running SpiritAnimators that reference this FSM."""
        try:
            from Infernux.engine.play_mode import PlayModeManager, PlayModeState
            pmm = PlayModeManager.instance()
            if not pmm or pmm.state != PlayModeState.PLAYING:
                return
            from Infernux.lib import SceneManager
            scene = SceneManager.instance().get_active_scene()
            if not scene:
                return
            from Infernux.components.animator2d import SpiritAnimator
            norm = os.path.normpath(fsm_path)
            for go in scene.get_all_objects():
                animator = go.get_component(SpiritAnimator)
                if animator and animator._fsm and os.path.normpath(
                        animator._fsm.file_path or "") == norm:
                    animator.reload_controller()
        except Exception:
            pass
