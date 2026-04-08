"""Main wiring function for the C++ HierarchyPanel."""
from __future__ import annotations

from typing import TYPE_CHECKING

from Infernux.debug import Debug
from Infernux.engine.bootstrap_hierarchy._helpers import (
    _get_children_safe,
    _get_py_components_safe,
)

if TYPE_CHECKING:
    from Infernux.engine.bootstrap import EditorBootstrap


class _Ctx:
    """Thin namespace shared across hierarchy wiring helpers."""


# ═══════ Canvas / UI-mode queries ══════════════════════════════

def _wire_canvas_queries(ctx):
    """Wire canvas and UI-component detection callbacks."""
    hp = ctx.hp

    def _go_has_canvas(oid):
        from Infernux.lib import SceneManager as _SM
        from Infernux.ui import UICanvas
        scene = _SM.instance().get_active_scene()
        if not scene:
            return False
        go = scene.find_by_id(oid)
        if not go:
            return False
        for comp in _get_py_components_safe(go):
            if isinstance(comp, UICanvas):
                return True
        return False

    def _go_has_ui_screen_component(oid):
        from Infernux.lib import SceneManager as _SM
        from Infernux.ui.inx_ui_screen_component import InxUIScreenComponent
        scene = _SM.instance().get_active_scene()
        if not scene:
            return False
        go = scene.find_by_id(oid)
        if not go:
            return False
        for comp in _get_py_components_safe(go):
            if isinstance(comp, InxUIScreenComponent):
                return True
        return False

    def _parent_has_canvas_ancestor(oid):
        from Infernux.lib import SceneManager as _SM
        from Infernux.ui import UICanvas
        scene = _SM.instance().get_active_scene()
        if not scene:
            return False
        go = scene.find_by_id(oid)
        if not go:
            return False
        cur = go
        while cur is not None:
            for comp in _get_py_components_safe(cur):
                if isinstance(comp, UICanvas):
                    return True
            cur = cur.get_parent()
        return False

    def _has_canvas_descendant(oid):
        from Infernux.lib import SceneManager as _SM
        from Infernux.ui import UICanvas
        scene = _SM.instance().get_active_scene()
        if not scene:
            return False
        go = scene.find_by_id(oid)
        if not go:
            return False
        stack = [go]
        while stack:
            cur = stack.pop()
            for comp in _get_py_components_safe(cur):
                if isinstance(comp, UICanvas):
                    return True
            stack.extend(_get_children_safe(cur))
        return False

    hp.go_has_canvas = _go_has_canvas
    hp.go_has_ui_screen_component = _go_has_ui_screen_component
    hp.parent_has_canvas_ancestor = _parent_has_canvas_ancestor
    hp.has_canvas_descendant = _has_canvas_descendant


# ═══════ External drop & delete ════════════════════════════════

def _wire_drop_and_delete(ctx):
    """Wire prefab/model drop and delete callbacks."""
    hp = ctx.hp
    bs = ctx.bs
    sel = ctx.sel
    undo = ctx.undo

    def _instantiate_prefab(ref, parent_id, is_guid):
        from Infernux.lib import SceneManager, AssetRegistry
        from Infernux.engine.prefab_manager import instantiate_prefab, read_prefab_source_canvas
        from Infernux.ui import UICanvas as _UICanvasCls
        from Infernux.ui.ui_canvas_utils import invalidate_canvas_cache
        scene = SceneManager.instance().get_active_scene()
        if not scene:
            return
        adb = None
        registry = AssetRegistry.instance()
        if registry:
            adb = registry.get_asset_database()
        parent = scene.find_by_id(parent_id) if parent_id else None

        if parent is None:
            canvas_name = read_prefab_source_canvas(
                file_path=ref if not is_guid else None,
                guid=ref if is_guid else None,
                asset_database=adb,
            )
            if canvas_name:
                for root_obj in scene.get_root_objects():
                    if root_obj.name != canvas_name:
                        continue
                    for comp in _get_py_components_safe(root_obj):
                        if isinstance(comp, _UICanvasCls):
                            parent = root_obj
                            break
                    if parent is not None:
                        break
                if parent is None:
                    canvas_go = scene.create_game_object(canvas_name)
                    if canvas_go:
                        canvas_go.add_py_component(_UICanvasCls())
                        invalidate_canvas_cache()
                        undo.record_create(canvas_go.id, "Create Canvas")
                        parent = canvas_go

        try:
            if is_guid:
                new_obj = instantiate_prefab(guid=ref, scene=scene, parent=parent, asset_database=adb)
            else:
                new_obj = instantiate_prefab(file_path=ref, scene=scene, parent=parent, asset_database=adb)
        except Exception as exc:
            Debug.log_error(f"Prefab instantiation failed: {exc}")
            return
        if new_obj:
            sel.select(new_obj.id)
            undo.record_create(new_obj.id, "Instantiate Prefab")
            if hp.on_selection_changed:
                hp.on_selection_changed(new_obj.id)

    def _create_model_object(ref, parent_id, is_guid):
        from Infernux.lib import SceneManager, AssetRegistry
        scene = SceneManager.instance().get_active_scene()
        if not scene:
            return
        guid = ref if is_guid else ""
        if not guid:
            registry = AssetRegistry.instance()
            adb = registry.get_asset_database() if registry else None
            if not adb:
                return
            guid = adb.get_guid_from_path(ref)
        if not guid:
            return
        new_obj = scene.create_from_model(guid)
        if new_obj:
            from Infernux.lib import SceneManager as _SM2
            _finalize_drop(new_obj, parent_id, "Create Model",
                           sel, undo, hp, _SM2)

    hp.instantiate_prefab = _instantiate_prefab
    hp.create_model_object = _create_model_object

    def _delete_selected_objects():
        from Infernux.lib import SceneManager
        from Infernux.engine.undo import CompoundCommand, DeleteGameObjectCommand, UndoManager
        scene = SceneManager.instance().get_active_scene()
        if not scene:
            return
        ids = list(sel.get_ids())
        if not ids:
            return
        commands = [DeleteGameObjectCommand(oid, "Delete GameObject") for oid in ids]
        mgr = UndoManager.instance()
        if mgr:
            cmd = commands[0] if len(commands) == 1 else CompoundCommand(commands, "Delete GameObjects")
            mgr.execute(cmd)
        else:
            for oid in ids:
                obj = scene.find_by_id(oid)
                if obj:
                    scene.destroy_game_object(obj)
            sfm2 = bs.scene_file_manager
            if sfm2:
                sfm2.mark_dirty()
        sel.clear()
        if hp.on_selection_changed:
            hp.on_selection_changed(0)

    hp.delete_selected_objects = _delete_selected_objects


def _finalize_drop(new_obj, parent_id, description, sel, undo, hp, SceneManager):
    """Parent, select, and record undo for a newly created object."""
    if parent_id and parent_id != 0:
        scene = SceneManager.instance().get_active_scene()
        if scene:
            parent = scene.find_by_id(parent_id)
            if parent:
                new_obj.set_parent(parent)
    sel.select(new_obj.id)
    undo.record_create(new_obj.id, description)
    if hp.on_selection_changed:
        hp.on_selection_changed(new_obj.id)


# ═══════ Main entry point ══════════════════════════════════════

def wire_hierarchy_callbacks(bs: EditorBootstrap) -> None:
    """Wire C++ HierarchyPanel callbacks to Python managers."""
    hp = bs.hierarchy
    from Infernux.engine.ui.selection_manager import SelectionManager
    from Infernux.engine.i18n import t as _t
    from Infernux.engine.play_mode import PlayModeManager
    from Infernux.engine.ui import EditorEventBus

    sel = SelectionManager.instance()

    ctx = _Ctx()
    ctx.hp = hp
    ctx.bs = bs
    ctx.sel = sel
    ctx._t = _t
    ctx.EditorEventBus = EditorEventBus

    # -- Selection integration --
    hp.is_selected = lambda oid: sel.is_selected(oid)
    hp.select_id = lambda oid: sel.select(oid)
    hp.toggle_id = lambda oid: sel.toggle(oid)
    hp.range_select_id = lambda oid: sel.range_select(oid)
    hp.clear_selection = lambda: sel.clear()
    hp.get_primary = lambda: sel.get_primary()
    hp.get_selected_ids = lambda: sel.get_ids()
    hp.selection_count = lambda: sel.count()
    hp.is_selection_empty = lambda: sel.is_empty()
    hp.set_ordered_ids = lambda ids: sel.set_ordered_ids(ids)

    # -- Translation & warning --
    hp.translate = _t
    hp.show_warning = lambda msg: Debug.log_warning(msg)

    # -- Undo --
    from Infernux.engine.undo import HierarchyUndoTracker
    undo = HierarchyUndoTracker()
    ctx.undo = undo
    hp.undo_record_create = lambda oid, desc: undo.record_create(oid, desc)
    hp.undo_record_delete = lambda oid, desc: undo.record_delete(oid, desc)
    hp.undo_record_move = lambda oid, opid, npid, oidx, nidx: undo.record_move(oid, opid, npid, oidx, nidx)

    # -- Scene info --
    def _get_scene_display_name():
        sfm = bs.scene_file_manager
        return sfm.get_display_name() if sfm else ""

    def _is_prefab_mode():
        sfm = bs.scene_file_manager
        return bool(sfm and sfm.is_prefab_mode)

    def _get_prefab_display_name():
        sfm = bs.scene_file_manager
        if sfm:
            name = sfm.get_display_name()
            return _t("hierarchy.prefab_mode_header").format(name=name)
        return "Prefab"

    hp.get_scene_display_name = _get_scene_display_name
    hp.is_prefab_mode = _is_prefab_mode
    hp.get_prefab_display_name = _get_prefab_display_name

    # -- Runtime hidden IDs --
    def _get_runtime_hidden_ids():
        try:
            mgr = PlayModeManager.instance()
            if mgr is not None:
                return mgr.get_runtime_hidden_object_ids()
        except Exception as _exc:
            Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
        return set()

    hp.get_runtime_hidden_ids = _get_runtime_hidden_ids

    # -- Delegate to sub-wirers --
    _wire_canvas_queries(ctx)

    from Infernux.engine.bootstrap_hierarchy._creation import wire_creation_callbacks
    wire_creation_callbacks(ctx)

    from Infernux.engine.bootstrap_hierarchy._prefab_clipboard import (
        wire_prefab_actions, wire_clipboard,
    )
    wire_prefab_actions(ctx)
    wire_clipboard(ctx)

    _wire_drop_and_delete(ctx)
