"""
Animation State Machine Editor — visual node-graph editor for .animfsm files.

Displays states as nodes with connections representing transitions.
Drag from an output pin to an input pin to create a transition.
Click a node to edit its properties in the right-side inspector.
Opened from the Animation menu or by double-clicking a .animfsm file
in the Project panel.
"""

from __future__ import annotations

import ast
import copy
import os
import re
import threading
from typing import Dict, List, Optional, Tuple

from Infernux.core.anim_state_machine import (
    AnimStateMachine,
    AnimState,
    AnimTransition,
    AnimParameter,
)
from Infernux.core.asset_ref import AnimationClipRef, get_asset_type_config
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
from .imgui_keys import KEY_S, MOD_CTRL
from .node_graph_view import NodeGraphView
from ._inspector_references import render_object_field, _picker_assets
from .inspector_utils import field_label, max_label_w
from .panel_registry import editor_panel
from .theme import ImGuiCol, Theme


# Legacy single-compare fallback when ast parse fails
_COND_NUM_RE = re.compile(
    r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(==|!=|<=|>=|<|>)\s*(-?[0-9]+(?:\.[0-9]*)?)\s*$"
)
_COND_NOT_RE = re.compile(r"^\s*not\s+([A-Za-z_][A-Za-z0-9_]*)\s*$")
_COND_BOOL_EQ_RE = re.compile(
    r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*==\s*(True|False)\s*$"
)

_OPS = ["<", ">", "<=", ">=", "==", "!="]


def _fmt_rhs_float(v: float) -> str:
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return f"{v:.8g}"


def _cmpop_to_str(op: ast.cmpop) -> Optional[str]:
    if isinstance(op, ast.Eq):
        return "=="
    if isinstance(op, ast.NotEq):
        return "!="
    if isinstance(op, ast.Lt):
        return "<"
    if isinstance(op, ast.LtE):
        return "<="
    if isinstance(op, ast.Gt):
        return ">"
    if isinstance(op, ast.GtE):
        return ">="
    return None


def _ast_to_float(node: ast.expr) -> Optional[float]:
    if isinstance(node, ast.Constant):
        v = node.value
        if isinstance(v, bool):
            return 1.0 if v else 0.0
        if isinstance(v, (int, float)):
            return float(v)
    if hasattr(ast, "Num") and isinstance(node, ast.Num):
        return float(node.n)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _ast_to_float(node.operand)
        if inner is not None:
            return -inner
    return None


def _compare_to_term(n: ast.Compare) -> Optional[dict]:
    if len(n.ops) != 1 or len(n.comparators) != 1:
        return None
    if not isinstance(n.left, ast.Name):
        return None
    op_s = _cmpop_to_str(n.ops[0])
    if not op_s:
        return None
    fv = _ast_to_float(n.comparators[0])
    if fv is None:
        return None
    return {"name": n.left.id, "op": op_s, "value": float(fv)}


def _flatten_and_only(node: ast.expr) -> List[ast.Compare]:
    """Only AND chains (Unity-style: multiple conditions are all required)."""
    if isinstance(node, ast.Compare):
        return [node]
    if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And):
        out: List[ast.Compare] = []
        for v in node.values:
            sub = _flatten_and_only(v)
            if not sub:
                return []
            out.extend(sub)
        return out
    return []


def _encode_condition_model(terms: List[dict]) -> str:
    """Encode as left-associative ``and`` chain (implicit between rows)."""
    if not terms:
        return ""
    if len(terms) == 1:
        t = terms[0]
        return f"({t['name']} {t['op']} {_fmt_rhs_float(float(t['value']))})"
    expr = f"({terms[0]['name']} {terms[0]['op']} {_fmt_rhs_float(float(terms[0]['value']))})"
    for i in range(1, len(terms)):
        tn = terms[i]
        part = f"({tn['name']} {tn['op']} {_fmt_rhs_float(float(tn['value']))})"
        expr = f"({expr} and {part})"
    return expr


def _legacy_condition_to_terms(cond: str) -> List[dict]:
    c = (cond or "").strip()
    if not c:
        return []
    m = _COND_NUM_RE.match(c)
    if m:
        name, op, num_s = m.group(1), m.group(2), m.group(3)
        try:
            v = float(num_s)
        except ValueError:
            return []
        return [{"name": name, "op": op, "value": v}]
    m2 = _COND_BOOL_EQ_RE.match(c)
    if m2:
        name, tf = m2.group(1), m2.group(2)
        return [{"name": name, "op": "==", "value": 1.0 if tf == "True" else 0.0}]
    m3 = _COND_NOT_RE.match(c)
    if m3:
        return [{"name": m3.group(1), "op": "==", "value": 0.0}]
    if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", c):
        return [{"name": c, "op": "==", "value": 1.0}]
    return []


def parse_condition_string_to_model(cond: str) -> List[dict]:
    c = (cond or "").strip()
    if not c:
        return []
    try:
        tree = ast.parse(c, mode="eval")
        body = tree.body
        terms_ast = _flatten_and_only(body)
        if not terms_ast:
            return _legacy_condition_to_terms(c)
        out: List[dict] = []
        for t in terms_ast:
            d = _compare_to_term(t)
            if not d:
                return _legacy_condition_to_terms(c)
            out.append(d)
        return out
    except (SyntaxError, ValueError, TypeError):
        pass
    return _legacy_condition_to_terms(c)


def _replace_identifier_in_expr(expr: str, old: str, new: str) -> str:
    if not old or old == new:
        return expr
    return re.sub(r"\b" + re.escape(old) + r"\b", new, expr)


# ═══════════════════════════════════════════════════════════════════════════
# Node type definition for animation states
# ═══════════════════════════════════════════════════════════════════════════

_STATE_TYPE = NodeTypeDef(
    type_id="anim_state",
    label="State",
    header_color=(0.20, 0.20, 0.22, 1.0),
    pins=[
        PinDef(
            id="in",
            label="In",
            kind=PinKind.INPUT,
            color=(0.50, 0.52, 0.55, 1.0),
            max_connections=1,
        ),
        PinDef(id="out", label="Out", kind=PinKind.OUTPUT, color=(0.52, 0.54, 0.56, 1.0)),
    ],
    min_width=148.0,
    body_bottom_pad=0.0,
    header_color_swatch=True,
    pin_label_color=(0.93, 0.94, 0.96, 1.0),
)

_ENTRY_TYPE = NodeTypeDef(
    type_id="anim_entry",
    label="Entry",
    header_color=(0.22, 0.21, 0.23, 1.0),
    pins=[
        PinDef(id="out", label="Start", kind=PinKind.OUTPUT, color=Theme.APPLY_BUTTON),
    ],
    min_width=88.0,
    deletable=False,
)

_DETAIL_PANEL_W = 232.0
_VARS_PANEL_W = 216.0


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
        self._view.on_node_drag_start = self._on_node_drag_start
        self._view.on_node_drag_end = self._on_node_drag_end
        self._view.on_before_selection_change = self._on_before_graph_selection_change

        # Currently selected node uid
        self._selected_uid: str = ""

        # Maps: state name ↔ node uid
        self._name_to_uid: Dict[str, str] = {}
        self._uid_to_name: Dict[str, str] = {}

        # Entry node uid
        self._entry_uid: str = ""

        # Guard to avoid re-entrant selection clearing
        self._clearing_selection: bool = False

        # Panel persistence: ``load_state`` may run from bootstrap or first render
        self._panel_state_restored_once: bool = False
        self._panel_restore_data: Optional[dict] = None
        self._undo_drag_snapshot: Optional[dict] = None
        self._pending_selection_undo_before: Optional[dict] = None
        self._clipboard_fsm_nodes: Optional[dict] = None
        self._hdr_color_popup_undo_before: Optional[dict] = None
        self._view.on_node_header_color_begin = self._on_node_header_color_popup_begin
        self._view.on_node_header_color_end = self._on_node_header_color_popup_end
        self._view.on_node_header_color_changed = self._on_node_header_color_changed

        self._view.on_copy = self._on_graph_copy
        self._view.on_paste = self._on_graph_paste

        # After toolbar "New", do not resurrect ``file_path`` from panel_state.json.
        self._explicit_new_without_disk: bool = False

        # Start with a blank FSM
        self._new_fsm()

    # ── Public API ────────────────────────────────────────────────────

    def _open_animfsm(self, file_path: str):
        """Load an .animfsm file into the editor."""
        fp = os.path.normpath((file_path or "").strip())
        if not fp:
            return
        if not os.path.isabs(fp):
            try:
                from Infernux.engine.project_context import get_project_root

                root = get_project_root()
            except Exception:
                root = None
            if root:
                fp = os.path.normpath(os.path.join(root, fp.replace("/", os.sep)))
            else:
                fp = os.path.normpath(os.path.abspath(fp))
        fsm = AnimStateMachine.load(fp)
        if fsm is None:
            Debug.log_warning(f"Failed to load animfsm: {fp}")
            return
        self._fsm = fsm
        self._file_path = fp
        self._explicit_new_without_disk = False
        self._selected_uid = ""
        self._dirty = False
        for state in fsm.states:
            if not state.clip_guid and state.clip_path:
                state.clip_guid = self._resolve_guid(state.clip_path)
            if state.clip_guid:
                state.clip_path = ""
        self._sync_graph_from_fsm()

    def _new_fsm(self, *, user_initiated: bool = False):
        """Create a blank FSM for editing."""
        self._fsm = AnimStateMachine(name="New State Machine")
        self._file_path = ""
        if user_initiated:
            self._explicit_new_without_disk = True
        self._selected_uid = ""
        self._dirty = False
        self._view.reset_camera_defaults()
        self._sync_graph_from_fsm()

    # ── Lifecycle ──────────────────────────────────────────────────────

    def on_enable(self) -> None:
        from Infernux.engine.ui.selection_manager import SelectionManager
        from .event_bus import EditorEventBus, EditorEvent
        SelectionManager.instance().add_listener(self._on_global_selection_changed)
        bus = EditorEventBus.instance()
        bus.subscribe(EditorEvent.FILE_SELECTED, self._on_file_selected)
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
        """Persist open file path even if ``_open_animfsm`` is deferred to first frame."""
        from Infernux.engine.ui import panel_state as _ps

        data: dict = {}
        fp = (self._file_path or "").strip()
        if not fp and self._fsm is not None:
            fp = (getattr(self._fsm, "file_path", None) or "").strip()
        rel_fallback = ""
        if not fp and self._panel_restore_data:
            fp = (self._panel_restore_data.get("file_path") or "").strip()
            rel_fallback = (self._panel_restore_data.get("file_path_rel") or "").strip()
        # Project / toolbar persist fires often while ``_file_path`` can be unset for a frame;
        # keep the last on-disk path from panel_state unless the user hit "New".
        if (
            not fp
            and not self._explicit_new_without_disk
        ):
            prev = _ps.get(f"panel:{self.window_id}") or {}
            pfp = (prev.get("file_path") or "").strip()
            if pfp:
                np = os.path.normpath(pfp)
                if os.path.isfile(np):
                    fp = np
            if not fp:
                rel_p = (prev.get("file_path_rel") or "").strip()
                if rel_p:
                    try:
                        from Infernux.engine.project_context import get_project_root

                        root = get_project_root()
                    except Exception:
                        root = None
                    if root:
                        cand = os.path.normpath(os.path.join(root, rel_p.replace("/", os.sep)))
                        if os.path.isfile(cand):
                            fp = cand
        if fp:
            data["file_path"] = fp
            try:
                from Infernux.engine.project_context import get_project_root

                root = get_project_root()
                if root:
                    abs_p = os.path.abspath(fp)
                    abs_r = os.path.abspath(root)
                    rel = os.path.relpath(abs_p, abs_r)
                    if not rel.startswith(".."):
                        data["file_path_rel"] = rel
            except (ValueError, OSError):
                pass
        elif rel_fallback:
            data["file_path_rel"] = rel_fallback
        if self._view:
            data["pan_x"] = self._view.pan_x
            data["pan_y"] = self._view.pan_y
            data["zoom"] = self._view.zoom
            cam_prev = _ps.get(f"panel:{self.window_id}") or {}
            # Queued restore not applied yet — keep prior graph-space snapshot.
            if getattr(self._view, "_pending_camera", None) is not None:
                for k in ("graph_canvas_w", "graph_canvas_h", "view_center_gx", "view_center_gy"):
                    if k in cam_prev:
                        data[k] = cam_prev[k]
            else:
                cw, ch = self._view._canvas_w, self._view._canvas_h
                if cw < 1.0 or ch < 1.0:
                    cw = float(getattr(self._view, "_last_graph_canvas_w", 0.0) or 0.0)
                    ch = float(getattr(self._view, "_last_graph_canvas_h", 0.0) or 0.0)
                if cw >= 1.0 and ch >= 1.0:
                    data["graph_canvas_w"] = cw
                    data["graph_canvas_h"] = ch
                if getattr(self._view, "_cam_center_initialized", False):
                    data["view_center_gx"] = float(self._view.cam_center_gx)
                    data["view_center_gy"] = float(self._view.cam_center_gy)
                else:
                    for k in ("view_center_gx", "view_center_gy"):
                        if k in cam_prev:
                            data[k] = cam_prev[k]
                    if cw < 1.0 or ch < 1.0:
                        for k in ("graph_canvas_w", "graph_canvas_h"):
                            if k in cam_prev:
                                data[k] = cam_prev[k]
        return data

    def _resolve_saved_fsm_path(self, data: dict) -> str:
        """Resolve persisted path using absolute path, then project-relative."""
        fp = (data.get("file_path") or "").strip()
        rel = (data.get("file_path_rel") or "").strip()
        if fp:
            nfp = os.path.normpath(fp)
            if os.path.isfile(nfp):
                return nfp
        if rel:
            try:
                from Infernux.engine.project_context import get_project_root

                root = get_project_root()
                if root:
                    cand = os.path.normpath(os.path.join(root, rel.replace("/", os.sep)))
                    if os.path.isfile(cand):
                        return cand
            except (OSError, ValueError):
                pass
        return ""

    def _apply_saved_camera_from_dict(self, data: dict) -> None:
        """Restore graph camera (prefers graph-space centre — stable across dock / window size)."""
        if not self._view:
            return
        px = float(data.get("pan_x", self._view.pan_x))
        py = float(data.get("pan_y", self._view.pan_y))
        z = float(data.get("zoom", self._view.zoom))
        z = max(Theme.NODE_GRAPH_ZOOM_MIN, min(Theme.NODE_GRAPH_ZOOM_MAX, z))
        if "view_center_gx" in data and "view_center_gy" in data:
            try:
                gxc = float(data["view_center_gx"])
                gyc = float(data["view_center_gy"])
            except (TypeError, ValueError):
                gxc = gyc = 0.0
            self._view.queue_camera_restore_graph_center(center_gx=gxc, center_gy=gyc, zoom=z)
            return
        rw = float(data.get("graph_canvas_w", 0) or 0)
        rh = float(data.get("graph_canvas_h", 0) or 0)
        if rw >= 1.0 and rh >= 1.0:
            self._view.queue_camera_restore(pan_x=px, pan_y=py, zoom=z, ref_w=rw, ref_h=rh)
        else:
            self._view.set_legacy_pan_zoom(px, py, z)

    def load_state(self, data: dict) -> None:
        if not data:
            self._panel_restore_data = None
            self._panel_state_restored_once = True
            return
        self._panel_restore_data = dict(data)
        self._apply_saved_camera_from_dict(data)
        self._panel_state_restored_once = False

    def _apply_pending_panel_restore(self) -> None:
        """Open saved .animfsm once project root can resolve relative paths."""
        if self._panel_state_restored_once:
            return
        data = self._panel_restore_data
        if not data:
            self._panel_state_restored_once = True
            return
        to_open = self._resolve_saved_fsm_path(data)
        if to_open:
            self._open_animfsm(to_open)
            self._panel_state_restored_once = True
            return
        fp = (data.get("file_path") or "").strip()
        rel = (data.get("file_path_rel") or "").strip()
        if not fp and not rel:
            self._panel_state_restored_once = True
            return
        try:
            from Infernux.engine.project_context import get_project_root

            root = get_project_root()
        except Exception:
            root = None
        if root is None:
            # Project root not ready yet — retry on later frames; do not mark done.
            return
        # Root is available but the file is still missing: log once and stop retrying.
        if not getattr(self, "_logged_fsm_restore_miss", False):
            self._logged_fsm_restore_miss = True
            Debug.log_warning(
                "AnimFSM editor: could not restore last .animfsm "
                f"(file missing or path invalid). file_path={fp!r} file_path_rel={rel!r}"
            )
        self._panel_state_restored_once = True

    # ── Undo (AnimStateMachine snapshots; edit mode only) ─────────────

    def _link_key_for_undo(self) -> Optional[Tuple[str, str]]:
        lid = (self._view.selected_link or "").strip()
        if not lid or not self._graph:
            return None
        lk = self._graph.find_link(lid)
        if lk is None:
            return None
        sn = self._uid_to_name.get(lk.source_node, "")
        dn = self._uid_to_name.get(lk.target_node, "")
        return (sn, dn) if sn and dn else None

    def _undo_snapshot(self) -> dict:
        """FSM data + graph selection (by state name so it survives graph rebuild)."""
        fsm_data: dict = {}
        if self._fsm is not None:
            fsm_data = copy.deepcopy(self._fsm.to_dict())
        return {
            "fsm": fsm_data,
            "ui": {
                "selected_state": self._uid_to_name.get(self._selected_uid, "") if self._selected_uid else "",
                "link_key": self._link_key_for_undo(),
            },
        }

    def _apply_undo_snapshot(self, snap: dict) -> None:
        if not isinstance(snap, dict):
            return
        if "fsm" in snap:
            fsm_data = snap.get("fsm") or {}
            ui = snap.get("ui") or {}
        else:
            fsm_data = snap
            ui = {}
        if not fsm_data:
            self._new_fsm()
        else:
            self._fsm = AnimStateMachine.from_dict(fsm_data)
            self._sync_graph_from_fsm()
        self._dirty = True
        sel = (ui.get("selected_state") or "").strip() if isinstance(ui, dict) else ""
        if sel and sel in self._name_to_uid:
            uid = self._name_to_uid[sel]
            self._selected_uid = uid
            self._view.selected_nodes = [uid]
        else:
            self._selected_uid = ""
            self._view.selected_nodes = []
        self._view.selected_link = ""
        lk_key = ui.get("link_key") if isinstance(ui, dict) else None
        if isinstance(lk_key, (list, tuple)) and len(lk_key) == 2:
            sn, dn = str(lk_key[0]), str(lk_key[1])
            if sn in self._name_to_uid and dn in self._name_to_uid:
                su = self._name_to_uid[sn]
                du = self._name_to_uid[dn]
                for lk in self._graph.links:
                    if lk.source_node == su and lk.target_node == du:
                        self._view.selected_link = lk.uid
                        break

    def _animfsm_undo_enabled(self) -> bool:
        from Infernux.engine.play_mode import PlayModeManager, PlayModeState
        from Infernux.engine.undo import UndoManager

        mgr = UndoManager.instance()
        if not mgr or not mgr.enabled:
            return False
        pmm = PlayModeManager.instance()
        if pmm and pmm.state != PlayModeState.EDIT:
            return False
        return True

    def _try_record_undo(self, description: str, before: dict, after: dict) -> None:
        if before == after:
            return
        if not self._animfsm_undo_enabled():
            return
        from Infernux.engine.undo import UndoManager, LambdaCommand

        mgr = UndoManager.instance()
        if not mgr:
            return

        def _apply(d: dict) -> None:
            self._apply_undo_snapshot(d)

        mgr.record(
            LambdaCommand(
                description,
                undo_fn=lambda: _apply(before),
                redo_fn=lambda: _apply(after),
                marks_dirty=False,
            )
        )

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

    def on_render_content(self, ctx: InxGUIContext):
        if not self._panel_state_restored_once:
            fp_live = (self._file_path or "").strip()
            if fp_live and os.path.isfile(fp_live):
                # Project / asset pipeline opened a file before the first paint; do not
                # replace it with a possibly stale ``panel_state`` snapshot from disk.
                from Infernux.engine.ui import panel_state as _ps

                pdata = _ps.get(f"panel:{self.window_id}") or {}
                if pdata:
                    self._apply_saved_camera_from_dict(pdata)
                self._panel_restore_data = None
                self._panel_state_restored_once = True
            elif self._panel_restore_data is None:
                from Infernux.engine.ui import panel_state as _ps

                data = _ps.get(f"panel:{self.window_id}")
                if data:
                    self.load_state(data)
                else:
                    self._panel_state_restored_once = True
            self._apply_pending_panel_restore()

        # Ctrl+S save shortcut
        if (ctx.is_key_down(MOD_CTRL)
                and ctx.is_key_pressed(KEY_S)
                and self._fsm is not None):
            self._do_save()

        self._render_toolbar(ctx)
        ctx.separator()

        avail_w = ctx.get_content_region_avail_width()
        avail_h = ctx.get_content_region_avail_height()

        vars_w = min(_VARS_PANEL_W, max(128.0, avail_w * 0.18))
        detail_w = min(_DETAIL_PANEL_W, max(168.0, avail_w * 0.22))
        graph_w = max(120.0, avail_w - vars_w - detail_w - 8.0)

        # Left: declared parameters (for transition conditions)
        if ctx.begin_child("##fsm_vars_region", vars_w, avail_h, True):
            self._render_variables_panel(ctx)
        ctx.end_child()

        ctx.same_line()

        # Center: graph canvas
        if ctx.begin_child("##fsm_graph_region", graph_w, avail_h, False):
            self._view.render(ctx)
        ctx.end_child()

        ctx.same_line()

        # Right: state detail
        if ctx.begin_child("##fsm_detail_region", detail_w, avail_h, True):
            self._render_detail_panel(ctx)
        ctx.end_child()

        # Accept .animfsm file drops
        payload = ctx.accept_drag_drop_payload("ANIMFSM_FILE")
        if payload:
            self._open_animfsm(payload)

    # ── Toolbar ───────────────────────────────────────────────────────

    def _clip_ext_flags(self):
        """Return (has_2d_clip, has_3d_clip) from resolved state clip paths."""
        has_2d = False
        has_3d = False
        if not self._fsm:
            return has_2d, has_3d
        for state in self._fsm.states:
            path = self._resolved_clip_path_for_state(state)
            if not path:
                continue
            ext = os.path.splitext(path)[1].lower()
            if ext == ".animclip2d":
                has_2d = True
            elif ext == ".animclip3d":
                has_3d = True
        return has_2d, has_3d

    @staticmethod
    def _resolved_clip_path_for_state(state: AnimState) -> str:
        path = (state.clip_path or "").strip()
        if not path and state.clip_guid:
            try:
                from Infernux.core.assets import AssetManager

                adb = getattr(AssetManager, "_asset_database", None)
                if adb:
                    path = adb.get_path_from_guid(state.clip_guid) or ""
            except Exception:
                pass
        return path.replace("\\", "/") if path else ""

    def _render_toolbar(self, ctx: InxGUIContext):
        fsm = self._fsm
        if fsm is None:
            return

        if ctx.button(t("animfsm_editor.new")):
            before = self._undo_snapshot()
            self._new_fsm(user_initiated=True)
            self._try_record_undo("New state machine", before, self._undo_snapshot())
            return

        ctx.same_line(0, 8)
        save_label = t("animfsm_editor.save") if self._file_path else t("animfsm_editor.save_as")
        if ctx.button(save_label):
            self._do_save()

        ctx.same_line(0, 16)
        ctx.label(f"{t('animfsm_editor.name')}:")
        ctx.same_line(0, 8)
        ctx.set_next_item_width(160)
        new_name = ctx.text_input("##fsm_name", fsm.name, 128)
        if new_name != fsm.name:
            before = self._undo_snapshot()
            fsm.name = new_name
            self._dirty = True
            self._try_record_undo("Rename FSM", before, self._undo_snapshot())

        ctx.same_line(0, 16)
        has_2d, has_3d = self._clip_ext_flags()
        ctx.label(f"{t('animfsm_editor.mode')}:")
        ctx.same_line(0, 8)
        ctx.set_next_item_width(72)
        _MODES = ["2d", "3d"]
        mode_idx = _MODES.index(fsm.mode) if fsm.mode in _MODES else 0
        lock_both = has_2d and has_3d
        ctx.begin_disabled(lock_both)
        new_mode_idx = ctx.combo("##fsm_mode", mode_idx, ["2D", "3D"], 2)
        ctx.end_disabled()
        if new_mode_idx != mode_idx and not lock_both:
            want = _MODES[new_mode_idx]
            if want == "3d" and has_2d:
                pass
            elif want == "2d" and has_3d:
                pass
            else:
                before = self._undo_snapshot()
                fsm.mode = want
                self._dirty = True
                self._try_record_undo("Change FSM mode", before, self._undo_snapshot())

        if self._file_path:
            ctx.same_line(0, 12)
            ctx.label(self._file_path)

    @staticmethod
    def _sanitize_param_identifier(raw: str) -> str:
        """Keep ``[A-Za-z_][A-Za-z0-9_]*`` for condition variable names."""
        s = (raw or "").strip()
        out: List[str] = []
        for i, ch in enumerate(s):
            if i == 0:
                if ch.isalpha() or ch == "_":
                    out.append(ch)
            else:
                if ch.isalnum() or ch == "_":
                    out.append(ch)
        return "".join(out)

    def _default_compare_term(self, fsm: AnimStateMachine) -> dict:
        p0 = fsm.parameters[0]
        return {"name": p0.name, "op": ">", "value": 0.0}

    def _apply_condition_model(self, lk: GraphLink, terms: List[dict]) -> None:
        cond = _encode_condition_model(terms)
        lk.data["cond_terms"] = [dict(x) for x in terms]
        lk.data.pop("cond_joins", None)
        old = str(lk.data.get("condition", "") or "")
        if cond == old:
            return
        before = self._undo_snapshot()
        lk.data["condition"] = cond
        self._sync_transition_condition(lk)
        self._dirty = True
        self._try_record_undo("Edit transition condition", before, self._undo_snapshot())

    def _render_transition_condition_block(self, ctx: InxGUIContext, lk: GraphLink) -> None:
        fsm = self._fsm
        if fsm is None:
            return
        cond = str(lk.data.get("condition", "") or "")
        if "cond_terms" not in lk.data:
            terms = parse_condition_string_to_model(cond)
            lk.data["cond_terms"] = terms
        else:
            terms = lk.data.get("cond_terms") or []
            if not isinstance(terms, list):
                terms = []
        lk.data.pop("cond_joins", None)

        has_p = len(fsm.parameters) > 0
        names = [p.name for p in fsm.parameters]
        mode_clip = t("animfsm_editor.cond_mode_clip_end")
        mode_param = t("animfsm_editor.cond_mode_parameter")
        clip_mode = (not cond.strip()) and len(terms) == 0

        if not has_p:
            ctx.push_style_color(ImGuiCol.Text, 0.55, 0.56, 0.58, 1.0)
            ctx.label(t("animfsm_editor.cond_no_parameters_hint"))
            ctx.pop_style_color(1)
            ctx.dummy(0, 2)
            ctx.set_next_item_width(-1)
            ctx.combo("##tmode", 0, [mode_clip, mode_param], 2)
            return

        ctx.set_next_item_width(-1)
        mid_idx = 0 if clip_mode else 1
        new_mid = ctx.combo("##tmode", mid_idx, [mode_clip, mode_param], 2)
        if new_mid != mid_idx:
            if new_mid == 0:
                self._apply_condition_model(lk, [])
            else:
                self._apply_condition_model(lk, [self._default_compare_term(fsm)])
            return

        if clip_mode:
            return

        if not terms:
            self._apply_condition_model(lk, [self._default_compare_term(fsm)])
            return

        # One row: [param] [op] [float] — multiple rows are implicitly AND (Unity-style).
        for i in range(len(terms)):
            ctx.push_id(i)
            tm = terms[i]
            pname = str(tm.get("name", names[0]))
            if pname not in names:
                pname = names[0]
            pi = names.index(pname)
            ctx.set_next_item_width(88)
            new_pi = ctx.combo("##pn", pi, names, len(names))
            ctx.same_line(0, 4)
            op = str(tm.get("op", ">"))
            if op not in _OPS:
                op = ">"
            oi = _OPS.index(op)
            ctx.set_next_item_width(48)
            new_oi = ctx.combo("##op", oi, _OPS, len(_OPS))
            ctx.same_line(0, 4)
            fv = float(tm.get("value", 0.0))
            ctx.set_next_item_width(-1)
            new_fv = ctx.drag_float("##fv", fv, 0.05, -1e9, 1e9)
            ctx.pop_id()

            if new_pi != pi:
                terms[i]["name"] = names[new_pi]
                self._apply_condition_model(lk, terms)
                return
            if new_oi != oi or new_fv != fv:
                terms[i]["op"] = _OPS[new_oi]
                terms[i]["value"] = float(new_fv)
                self._apply_condition_model(lk, terms)
                return

        ctx.dummy(0, 4)
        ctx.begin_group()
        if ctx.button("+##addrow", None, 28, 20):
            nt = list(terms)
            nt.append(dict(self._default_compare_term(fsm)))
            self._apply_condition_model(lk, nt)
        ctx.same_line(0, 6)
        if len(terms) > 1 and ctx.button("−##rmrow", None, 28, 20):
            nt = list(terms)
            nt.pop()
            self._apply_condition_model(lk, nt)
        ctx.end_group()

    def _rename_parameter_in_fsm(self, old_name: str, new_name: str) -> None:
        """Rename a parameter in all transition expressions and graph link data."""
        if not old_name or old_name == new_name or self._fsm is None:
            return
        for lk in self._graph.links:
            ct = lk.data.get("cond_terms")
            if isinstance(ct, list) and ct:
                changed = False
                for t in ct:
                    if isinstance(t, dict) and t.get("name") == old_name:
                        t["name"] = new_name
                        changed = True
                if changed:
                    lk.data["cond_terms"] = ct
                    lk.data["condition"] = _encode_condition_model(ct)
                    self._sync_transition_condition(lk)
            elif lk.data.get("condition"):
                lk.data["condition"] = _replace_identifier_in_expr(
                    str(lk.data["condition"]), old_name, new_name
                )
                self._sync_transition_condition(lk)

    def _render_variables_panel(self, ctx: InxGUIContext):
        """Left rail: parameters usable in transition condition expressions."""
        fsm = self._fsm
        if fsm is None:
            return

        ctx.push_style_color(ImGuiCol.Text, 0.55, 0.56, 0.58, 1.0)
        ctx.label(t("animfsm_editor.section_parameters"))
        ctx.pop_style_color(1)
        ctx.separator()
        ctx.dummy(0, 4)

        if ctx.button(t("animfsm_editor.add_parameter")):
            before = self._undo_snapshot()
            fsm.parameters.append(AnimParameter(name=f"var_{len(fsm.parameters)}", kind="float"))
            self._dirty = True
            self._try_record_undo("Add parameter", before, self._undo_snapshot())

        ctx.dummy(0, 4)
        remove_idx = -1
        kinds = ["bool", "float", "int"]
        for i, p in enumerate(fsm.parameters):
            if p.kind not in kinds:
                before = self._undo_snapshot()
                p.kind = "float"
                self._dirty = True
                self._try_record_undo("Fix parameter type", before, self._undo_snapshot())
            ctx.push_id(i)
            row_w = ctx.get_content_region_avail_width()
            COMBO_W = 72.0
            GAP = 8.0
            DEL_W = 22.0
            name_w = max(48.0, row_w - COMBO_W - GAP - DEL_W - GAP)

            ctx.set_next_item_width(COMBO_W)
            ki = kinds.index(p.kind) if p.kind in kinds else 1
            new_ki = ctx.combo("##pk", ki, [k.capitalize() for k in kinds], len(kinds))
            if new_ki != ki:
                before = self._undo_snapshot()
                p.kind = kinds[new_ki]
                self._dirty = True
                self._try_record_undo("Change parameter type", before, self._undo_snapshot())

            ctx.same_line(0, GAP)
            ctx.set_next_item_width(name_w)
            raw_name = ctx.text_input("##pname", p.name, 64)
            san = self._sanitize_param_identifier(raw_name)
            if san and san != p.name:
                before = self._undo_snapshot()
                self._rename_parameter_in_fsm(p.name, san)
                p.name = san
                self._dirty = True
                self._try_record_undo("Rename parameter", before, self._undo_snapshot())

            ctx.same_line(0, GAP)
            if ctx.button("−##prm_del", width=DEL_W, height=20):
                remove_idx = i

            ctx.set_next_item_width(-1)
            if p.kind == "bool":
                nb = ctx.checkbox("##pdef_bool", p.default_bool)
                if nb != p.default_bool:
                    before = self._undo_snapshot()
                    p.default_bool = nb
                    self._dirty = True
                    self._try_record_undo("Parameter default", before, self._undo_snapshot())
            elif p.kind == "float":
                nf = ctx.drag_float("##pdef_float", p.default_float, 0.01, -1.0e9, 1.0e9)
                if nf != p.default_float:
                    before = self._undo_snapshot()
                    p.default_float = nf
                    self._dirty = True
                    self._try_record_undo("Parameter default", before, self._undo_snapshot())
            else:
                ni = ctx.input_int("##pdef_int", p.default_int)
                if ni != p.default_int:
                    before = self._undo_snapshot()
                    p.default_int = ni
                    self._dirty = True
                    self._try_record_undo("Parameter default", before, self._undo_snapshot())

            ctx.dummy(0, 6)
            ctx.pop_id()

        if 0 <= remove_idx < len(fsm.parameters):
            before = self._undo_snapshot()
            fsm.parameters.pop(remove_idx)
            self._dirty = True
            self._try_record_undo("Remove parameter", before, self._undo_snapshot())

    # ── Detail panel (right side) ─────────────────────────────────────

    @staticmethod
    def _clip_ref_for_state(state: AnimState) -> AnimationClipRef:
        """Build an ``AnimationClipRef`` with path hint resolved for Inspector-style labels."""
        path = (state.clip_path or "").strip()
        if not path and state.clip_guid:
            try:
                from Infernux.core.assets import AssetManager

                adb = getattr(AssetManager, "_asset_database", None)
                if adb:
                    path = adb.get_path_from_guid(state.clip_guid) or ""
            except Exception:
                pass
        return AnimationClipRef(guid=state.clip_guid or "", path_hint=path)

    def _detail_checkbox_row_right(self, ctx: InxGUIContext, lw: float, label_key: str, wid: str, value: bool) -> bool:
        """Label left, checkbox square aligned to the right (inspector-style row)."""
        field_label(ctx, t(label_key), lw)
        ctx.same_line(0, 8)
        dx = ctx.get_content_region_avail_width() - Theme.INSPECTOR_CHECKBOX_SLOT_W
        if dx > 0:
            ctx.set_cursor_pos_x(ctx.get_cursor_pos_x() + dx)
        return ctx.checkbox(wid, value)

    def _render_clip_reference_row(
        self, ctx: InxGUIContext, state: AnimState, node, lw: float,
    ) -> None:
        """Same object-field UX as the main Inspector (basename, picker, drag-drop, clear)."""
        cfg = get_asset_type_config("AnimationClip") or {}
        type_hint = str(cfg.get("display", "AnimClip2D"))
        drag_type = str(cfg.get("drag_type", "ANIMCLIP_FILE"))
        extensions = cfg.get("extensions", ("*.animclip2d",))
        prefix = str(cfg.get("prefix", "aclip"))

        ref = self._clip_ref_for_state(state)
        display = ref.display_name

        def _picker(filt: str):
            result = []
            for g in extensions:
                result += _picker_assets(filt, g, assets_only=False)
            return result

        def _on_pick(path: str, _st=state, _nd=node):
            self._assign_clip_to_state(_st, path, _nd)

        def _on_clear(_st=state, _nd=node):
            self._clear_clip_from_state(_st, _nd)

        field_label(ctx, t("animfsm_editor.clip_ref"), lw)
        render_object_field(
            ctx,
            f"{prefix}_fsm_clip_{node.uid}",
            display,
            type_hint,
            accept_drag_type=drag_type,
            on_drop_callback=lambda p, _st=state, _nd=node: self._assign_clip_to_state(
                _st, str(p), _nd,
            ),
            picker_asset_items=_picker,
            on_pick=_on_pick,
            on_clear=_on_clear,
        )

    def _render_detail_panel(self, ctx: InxGUIContext):
        fsm = self._fsm
        if fsm is None:
            return

        node = self._graph.find_node(self._selected_uid)
        if node is None or node.type_id != "anim_state":
            ctx.push_style_color(ImGuiCol.Text, 0.50, 0.51, 0.53, 1.0)
            ctx.label(t("animfsm_editor.no_state_selected"))
            ctx.pop_style_color(1)
            return

        state_name = self._uid_to_name.get(node.uid, "")
        state = fsm.get_state(state_name)
        if state is None:
            return

        labels = [
            t("animfsm_editor.state_name"),
            t("animfsm_editor.clip_ref"),
            t("animfsm_editor.speed"),
            t("animfsm_editor.exit_time"),
            t("animfsm_editor.loop"),
            t("animfsm_editor.restart_same_clip"),
        ]
        lw = max_label_w(ctx, labels)

        is_default = (state.name == fsm.default_state)
        row_w = ctx.get_content_region_avail_width()
        base_x = ctx.get_cursor_pos_x()
        ctx.push_style_color(ImGuiCol.Text, 0.55, 0.56, 0.58, 1.0)
        ctx.label(t("animfsm_editor.section_state"))
        ctx.pop_style_color(1)
        ctx.same_line(0, 0)
        if is_default:
            badge = t("animfsm_editor.default_badge")
            tw = ctx.calc_text_width(badge)
            ctx.set_cursor_pos_x(base_x + row_w - tw)
            ctx.push_style_color(ImGuiCol.Text, 0.48, 0.65, 0.45, 1.0)
            ctx.label(badge)
            ctx.pop_style_color(1)
        else:
            set_lbl = t("animfsm_editor.set_default")
            btn_w = ctx.calc_text_width(set_lbl) + 24.0
            ctx.set_cursor_pos_x(base_x + row_w - btn_w)
            if ctx.button(set_lbl):
                before = self._undo_snapshot()
                fsm.default_state = state.name
                self._update_entry_link()
                self._dirty = True
                self._try_record_undo("Set default state", before, self._undo_snapshot())
        ctx.separator()
        ctx.dummy(0, 4)

        field_label(ctx, t("animfsm_editor.state_name"), lw)
        ctx.same_line(0, 8)
        ctx.set_next_item_width(-1)
        new_name = ctx.text_input("##state_name_edit", state.name, 128)
        if new_name != state.name:
            before = self._undo_snapshot()
            if self._try_rename_state(state, new_name.strip()):
                self._dirty = True
                self._try_record_undo("Rename state", before, self._undo_snapshot())

        ctx.dummy(0, Theme.INSPECTOR_SECTION_GAP)
        ctx.push_style_color(ImGuiCol.Text, 0.55, 0.56, 0.58, 1.0)
        ctx.label(t("animfsm_editor.section_reference"))
        ctx.pop_style_color(1)
        ctx.separator()
        ctx.dummy(0, 4)

        self._render_clip_reference_row(ctx, state, node, lw)

        ctx.dummy(0, Theme.INSPECTOR_SECTION_GAP)
        ctx.push_style_color(ImGuiCol.Text, 0.55, 0.56, 0.58, 1.0)
        ctx.label(t("animfsm_editor.section_playback"))
        ctx.pop_style_color(1)
        ctx.separator()
        ctx.dummy(0, 4)

        field_label(ctx, t("animfsm_editor.speed"), lw)
        ctx.same_line(0, 8)
        ctx.set_next_item_width(-1)
        new_speed = ctx.drag_float("##speed", state.speed, 0.01, 0.0, 10.0)
        if new_speed != state.speed:
            before = self._undo_snapshot()
            state.speed = new_speed
            self._dirty = True
            self._try_record_undo("Change playback speed", before, self._undo_snapshot())

        exit_pct = state.exit_time_normalized * 100.0
        field_label(ctx, t("animfsm_editor.exit_time"), lw)
        ctx.same_line(0, 8)
        ctx.set_next_item_width(-1)
        new_exit_pct = ctx.drag_float("##exit_time", exit_pct, 0.5, 0.0, 100.0)
        if new_exit_pct != exit_pct:
            before = self._undo_snapshot()
            state.exit_time_normalized = max(0.0, min(1.0, new_exit_pct / 100.0))
            self._dirty = True
            self._try_record_undo("Change exit time", before, self._undo_snapshot())

        new_loop = self._detail_checkbox_row_right(
            ctx, lw, "animfsm_editor.loop", "##state_loop", state.loop,
        )
        if new_loop != state.loop:
            before = self._undo_snapshot()
            state.loop = new_loop
            node.data["loop"] = state.loop
            self._dirty = True
            self._try_record_undo("Toggle loop", before, self._undo_snapshot())

        new_rs = self._detail_checkbox_row_right(
            ctx, lw, "animfsm_editor.restart_same_clip", "##state_restart_same", state.restart_same_clip,
        )
        if new_rs != state.restart_same_clip:
            before = self._undo_snapshot()
            state.restart_same_clip = new_rs
            node.data["restart_same_clip"] = state.restart_same_clip
            self._dirty = True
            self._try_record_undo("Toggle restart same clip", before, self._undo_snapshot())

        ctx.dummy(0, Theme.INSPECTOR_SECTION_GAP)
        ctx.push_style_color(ImGuiCol.Text, 0.55, 0.56, 0.58, 1.0)
        ctx.label(t("animfsm_editor.section_transitions"))
        ctx.pop_style_color(1)
        ctx.separator()
        ctx.dummy(0, 4)

        outgoing_links = [
            lk for lk in self._graph.links
            if lk.source_node == node.uid and lk.source_pin == "out"
               and lk.target_pin == "in"
        ]
        remove_link_uid = ""
        for i, lk in enumerate(outgoing_links):
            target_name = self._uid_to_name.get(lk.target_node, "?")
            ctx.push_id(i)
            ctx.begin_group()
            ctx.label(f"→ {target_name}")
            ctx.same_line()
            if ctx.button("×##del", None, 24, 20):
                remove_link_uid = lk.uid

            self._render_transition_condition_block(ctx, lk)
            ctx.end_group()
            ctx.dummy(0, 6)
            ctx.pop_id()

        if remove_link_uid:
            before = self._undo_snapshot()
            if self._remove_link_and_transition(remove_link_uid):
                self._dirty = True
                self._try_record_undo("Remove transition", before, self._undo_snapshot())

    # ═══════════════════════════════════════════════════════════════════
    # FSM ↔ Graph synchronization
    # ═══════════════════════════════════════════════════════════════════

    def _sync_state_header_from_node(self, uid: str) -> None:
        """Copy ``node.data[''header_color'']`` into :class:`AnimState`."""
        fsm = self._fsm
        if fsm is None:
            return
        name = self._uid_to_name.get(uid, "")
        if not name:
            return
        state = fsm.get_state(name)
        node = self._graph.find_node(uid)
        if state is None or node is None:
            return
        raw = node.data.get("header_color")
        hb = _STATE_TYPE.header_color
        br = float(hb[0])
        bg = float(hb[1])
        bb = float(hb[2])
        ba = float(hb[3]) if len(hb) > 3 else 1.0
        if isinstance(raw, (list, tuple)) and len(raw) >= 3:
            nr = max(0.0, min(1.0, float(raw[0])))
            ng = max(0.0, min(1.0, float(raw[1])))
            nb = max(0.0, min(1.0, float(raw[2])))
            na = max(0.0, min(1.0, float(raw[3]))) if len(raw) > 3 else 1.0
            if (abs(nr - br) < 1e-3 and abs(ng - bg) < 1e-3 and abs(nb - bb) < 1e-3
                    and abs(na - ba) < 1e-3):
                state.header_color = None
            else:
                state.header_color = (nr, ng, nb, na)
        else:
            state.header_color = None

    def _on_node_header_color_popup_begin(self, uid: str) -> None:
        self._hdr_color_popup_undo_before = self._undo_snapshot()

    def _on_node_header_color_popup_end(self, uid: str) -> None:
        self._sync_state_header_from_node(uid)
        if self._hdr_color_popup_undo_before is not None:
            self._try_record_undo(
                "Change state header color",
                self._hdr_color_popup_undo_before,
                self._undo_snapshot(),
            )
        self._hdr_color_popup_undo_before = None

    def _on_node_header_color_changed(self, uid: str) -> None:
        self._sync_state_header_from_node(uid)
        self._dirty = True

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
            node.data["loop"] = state.loop
            node.data["restart_same_clip"] = state.restart_same_clip
            if state.header_color is not None:
                node.data["header_color"] = [float(x) for x in state.header_color]
            else:
                node.data.pop("header_color", None)
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
                    lk.data["cond_terms"] = parse_condition_string_to_model(tr.condition)
                    lk.data.pop("cond_joins", None)

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

    def _unique_state_name(self, want: str) -> str:
        fsm = self._fsm
        if fsm is None:
            return want or "State"
        base = (want or "State").strip() or "State"
        if fsm.get_state(base) is None:
            return base
        n = fsm.state_count
        while True:
            cand = f"{base} {n}"
            if fsm.get_state(cand) is None:
                return cand
            n += 1

    def _on_graph_copy(self) -> None:
        fsm = self._fsm
        if fsm is None:
            return
        uids = [u for u in self._view.selected_nodes if u and u != self._entry_uid]
        names = [self._uid_to_name.get(u) for u in uids]
        names = [n for n in names if n]
        if not names:
            return
        name_set = set(names)
        states_copy: List[dict] = []
        for n in names:
            st = fsm.get_state(n)
            if not st:
                continue
            d = st.to_dict()
            d["transitions"] = []
            states_copy.append(copy.deepcopy(d))
        transitions: List[Tuple[str, dict]] = []
        for n in names:
            st = fsm.get_state(n)
            if not st:
                continue
            for tr in st.transitions:
                if tr.target_state in name_set:
                    transitions.append((n, tr.to_dict()))
        self._clipboard_fsm_nodes = {"states": states_copy, "transitions": transitions}

    def _on_graph_paste(self) -> None:
        blob = self._clipboard_fsm_nodes
        fsm = self._fsm
        if not blob or not fsm or not blob.get("states"):
            return
        before = self._undo_snapshot()
        old_names = [s["name"] for s in blob["states"]]
        name_map: Dict[str, str] = {}
        for old in old_names:
            name_map[old] = self._unique_state_name(old)
        for sd in blob["states"]:
            new_n = name_map[sd["name"]]
            state = AnimState.from_dict(sd)
            state.name = new_n
            state.transitions = []
            pos = list(sd.get("position", [0.0, 0.0]))
            if len(pos) < 2:
                pos = [0.0, 0.0]
            state.position = [float(pos[0]) + 48.0, float(pos[1]) + 48.0]
            fsm.states.append(state)
        for src_old, tr_d in blob.get("transitions", []):
            src_new = name_map.get(src_old)
            if not src_new:
                continue
            tr = AnimTransition.from_dict(tr_d)
            tgt_old = tr.target_state
            tr.target_state = name_map.get(tgt_old, tgt_old)
            st = fsm.get_state(src_new)
            if st:
                st.transitions.append(tr)
        self._sync_graph_from_fsm()
        new_uids: List[str] = []
        for o in old_names:
            nn = name_map.get(o)
            if nn:
                uid = self._name_to_uid.get(nn)
                if uid:
                    new_uids.append(uid)
        if new_uids:
            self._view.selected_nodes = new_uids
            self._selected_uid = new_uids[0]
        self._dirty = True
        self._try_record_undo("Paste states", before, self._undo_snapshot())

    # ── Callbacks from NodeGraphView ──────────────────────────────────

    def _on_node_drag_start(self, uid: str) -> None:
        if uid == self._entry_uid:
            self._undo_drag_snapshot = None
            return
        self._undo_drag_snapshot = self._undo_snapshot()

    def _on_node_drag_end(self, uid: str) -> None:
        if uid == self._entry_uid:
            self._undo_drag_snapshot = None
            return
        self._sync_fsm_positions()
        before = self._undo_drag_snapshot
        self._undo_drag_snapshot = None
        if before is None:
            return
        after = self._undo_snapshot()
        self._try_record_undo("Move state node", before, after)

    def _on_link_created(self, src_node: str, src_pin: str, dst_node: str, dst_pin: str):
        """User created a connection by dragging between pins."""
        # Entry node connections change the default state
        if src_node == self._entry_uid:
            target_name = self._uid_to_name.get(dst_node, "")
            if target_name and self._fsm:
                before = self._undo_snapshot()
                self._fsm.default_state = target_name
                self._update_entry_link()
                self._dirty = True
                self._try_record_undo("Set default state", before, self._undo_snapshot())
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

        before = self._undo_snapshot()
        state.transitions.append(AnimTransition(target_state=dst_name))
        lk = self._graph.add_link(src_node, src_pin, dst_node, dst_pin)
        if lk:
            lk.data["cond_terms"] = []
            lk.data.pop("cond_joins", None)
        self._dirty = True
        self._try_record_undo("Add transition", before, self._undo_snapshot())

    def _on_link_deleted(self, link_uid: str):
        before = self._undo_snapshot()
        if self._remove_link_and_transition(link_uid):
            self._dirty = True
            self._try_record_undo("Remove transition", before, self._undo_snapshot())

    def _on_nodes_deleted(self, uids: List[str]):
        fsm = self._fsm
        if fsm is None:
            return
        before = self._undo_snapshot()
        changed = False
        for uid in uids:
            # Don't delete entry node
            if uid == self._entry_uid:
                continue
            changed = True
            name = self._uid_to_name.get(uid, "")
            if name:
                fsm.remove_state(name)
                del self._name_to_uid[name]
                del self._uid_to_name[uid]
            self._graph.remove_node(uid)
        if self._selected_uid in uids:
            self._selected_uid = ""
        if changed:
            self._update_entry_link()
            self._dirty = True
            self._try_record_undo("Delete state", before, self._undo_snapshot())

    def _on_node_add_request(self, type_id: str, x: float, y: float):
        if type_id != "anim_state":
            return
        fsm = self._fsm
        if fsm is None:
            return
        before = self._undo_snapshot()
        state = fsm.add_state()
        state.position = [x, y]
        node = self._graph.add_node("anim_state", x=x, y=y)
        node.data["label"] = state.name
        node.data["loop"] = state.loop
        node.data["restart_same_clip"] = state.restart_same_clip
        self._name_to_uid[state.name] = node.uid
        self._uid_to_name[node.uid] = state.name
        self._view.selected_nodes = [node.uid]
        self._selected_uid = node.uid
        self._update_entry_link()
        self._dirty = True
        self._try_record_undo("Add state", before, self._undo_snapshot())

    def _on_before_graph_selection_change(self) -> None:
        if not self._animfsm_undo_enabled():
            return
        self._pending_selection_undo_before = self._undo_snapshot()

    def _on_node_selected(self, uid: str):
        self._selected_uid = uid
        self._clear_external_selection()
        if self._pending_selection_undo_before is not None:
            after = self._undo_snapshot()
            self._try_record_undo(
                "Graph selection",
                self._pending_selection_undo_before,
                after,
            )
            self._pending_selection_undo_before = None

    def _clear_external_selection(self):
        """Clear hierarchy / scene selection only; keep Project panel file selection."""
        if self._clearing_selection:
            return
        self._clearing_selection = True
        try:
            from Infernux.engine.ui.selection_manager import SelectionManager

            SelectionManager.instance().clear()
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

    def _on_canvas_drop(self, payload_type: str, payload: str, gx: float, gy: float):
        """Handle items dropped onto the node graph canvas."""
        if payload_type == "ANIMCLIP_FILE" and payload:
            # Check if dropped on an existing state node
            for uid, name in self._uid_to_name.items():
                node = self._graph.find_node(uid)
                if node and abs(node.pos_x - gx) < 80 and abs(node.pos_y - gy) < 40:
                    state = self._fsm.get_state(name) if self._fsm else None
                    if state:
                        self._assign_clip_to_state(state, payload, node, record_undo=True)
                    return
            # Otherwise create a new state with this clip (state name stays independent)
            if self._fsm:
                before = self._undo_snapshot()
                state = self._fsm.add_state()
                state.position = [gx, gy]
                node = self._graph.add_node("anim_state", x=gx, y=gy)
                node.data["label"] = state.name
                self._name_to_uid[state.name] = node.uid
                self._uid_to_name[node.uid] = state.name
                self._assign_clip_to_state(state, payload, node, record_undo=False)
                node.data["loop"] = state.loop
                node.data["restart_same_clip"] = state.restart_same_clip
                self._view.selected_nodes = [node.uid]
                self._selected_uid = node.uid
                self._update_entry_link()
                self._dirty = True
                self._try_record_undo("Drop clip to canvas", before, self._undo_snapshot())

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

    def _clear_clip_from_state(self, state: AnimState, node=None, *, record_undo: bool = True):
        before = self._undo_snapshot() if record_undo else None
        state.clip_guid = ""
        state.clip_path = ""
        self._dirty = True
        if record_undo and before is not None:
            self._try_record_undo("Clear clip", before, self._undo_snapshot())

    def _try_rename_state(self, state: AnimState, new_name: str) -> bool:
        """Rename an FSM state and keep graph / transitions consistent."""
        if not new_name or state.name == new_name:
            return False
        fsm = self._fsm
        if fsm is None or fsm.get_state(new_name) is not None:
            return False

        old_name = state.name
        uid = self._name_to_uid.get(old_name, "")
        if not uid:
            return False

        state.name = new_name
        del self._name_to_uid[old_name]
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

        return True

    def _assign_clip_to_state(self, state: AnimState, clip_path: str, node=None, *, record_undo: bool = True):
        """Assign a clip path/guid to a state."""
        before = self._undo_snapshot() if record_undo else None
        state.clip_guid = self._resolve_guid(clip_path) if clip_path else ""
        state.clip_path = "" if state.clip_guid else (clip_path or "")
        self._dirty = True
        if record_undo and before is not None:
            self._try_record_undo("Assign clip", before, self._undo_snapshot())

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
            target = os.path.normpath(target)
            self._file_path = target
            fsm.file_path = target
            self._explicit_new_without_disk = False
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
