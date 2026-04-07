"""
Hierarchy callback wiring — extracted from EditorBootstrap.

Provides :func:`wire_hierarchy_callbacks` which attaches all Python-side
callbacks to a C++ ``HierarchyPanel`` instance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Infernux.engine.bootstrap import EditorBootstrap


def wire_hierarchy_callbacks(bs: EditorBootstrap) -> None:
    """Wire C++ HierarchyPanel callbacks to Python managers."""
    hp = bs.hierarchy
    from Infernux.engine.ui.selection_manager import SelectionManager
    from Infernux.engine.i18n import t as _t
    from Infernux.debug import Debug
    from Infernux.engine.play_mode import PlayModeManager
    from Infernux.engine.ui import EditorEventBus

    sel = SelectionManager.instance()

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

    # -- Translation --
    hp.translate = _t

    # -- Warning --
    hp.show_warning = lambda msg: Debug.log_warning(msg)

    # -- Undo --
    from Infernux.engine.undo import HierarchyUndoTracker
    undo = HierarchyUndoTracker()
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
            pass
        return set()

    hp.get_runtime_hidden_ids = _get_runtime_hidden_ids

    # -- Canvas / UI-mode queries --
    def _go_has_canvas(oid):
        from Infernux.lib import SceneManager as _SM
        from Infernux.ui import UICanvas
        scene = _SM.instance().get_active_scene()
        if not scene:
            return False
        go = scene.find_by_id(oid)
        if not go:
            return False
        for comp in go.get_py_components():
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
        for comp in go.get_py_components():
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
            for comp in cur.get_py_components():
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
            for comp in cur.get_py_components():
                if isinstance(comp, UICanvas):
                    return True
            stack.extend(cur.get_children())
        return False

    hp.go_has_canvas = _go_has_canvas
    hp.go_has_ui_screen_component = _go_has_ui_screen_component
    hp.parent_has_canvas_ancestor = _parent_has_canvas_ancestor
    hp.has_canvas_descendant = _has_canvas_descendant

    # -- Context-menu creation callbacks --
    def _create_primitive(type_idx, parent_id):
        from Infernux.lib import SceneManager, PrimitiveType
        scene = SceneManager.instance().get_active_scene()
        if not scene:
            return
        types = [PrimitiveType.Cube, PrimitiveType.Sphere, PrimitiveType.Capsule,
                 PrimitiveType.Cylinder, PrimitiveType.Plane]
        if type_idx < 0 or type_idx >= len(types):
            return
        new_obj = scene.create_primitive(types[type_idx])
        if new_obj:
            _finalize(new_obj, parent_id, "Create Primitive")

    def _create_light(type_idx, parent_id):
        from Infernux.lib import SceneManager, LightType, LightShadows, Vector3
        scene = SceneManager.instance().get_active_scene()
        if not scene:
            return
        names = ["Directional Light", "Point Light", "Spot Light"]
        light_types = [LightType.Directional, LightType.Point, LightType.Spot]
        if type_idx < 0 or type_idx >= len(light_types):
            return
        new_obj = scene.create_game_object(names[type_idx])
        if not new_obj:
            return
        light_comp = new_obj.add_component("Light")
        if light_comp:
            light_comp.light_type = light_types[type_idx]
            light_comp.shadows = LightShadows.Hard
            light_comp.shadow_bias = 0.0
            if light_types[type_idx] == LightType.Directional:
                trans = new_obj.transform
                if trans:
                    trans.euler_angles = Vector3(50.0, -30.0, 0.0)
            elif light_types[type_idx] == LightType.Point:
                light_comp.range = 10.0
            elif light_types[type_idx] == LightType.Spot:
                light_comp.range = 10.0
                light_comp.outer_spot_angle = 45.0
                light_comp.spot_angle = 30.0
        _finalize(new_obj, parent_id, "Create Light")

    def _create_camera(parent_id):
        from Infernux.lib import SceneManager
        scene = SceneManager.instance().get_active_scene()
        if not scene:
            return
        new_obj = scene.create_game_object("Camera")
        if new_obj:
            cam = new_obj.add_component("Camera")
            _finalize(new_obj, parent_id, "Create Camera")

    def _create_render_stack(parent_id):
        from Infernux.lib import SceneManager
        from Infernux.renderstack import RenderStack as RenderStackCls
        scene = SceneManager.instance().get_active_scene()
        if not scene:
            return
        new_obj = scene.create_game_object("RenderStack")
        if not new_obj:
            return
        stack = new_obj.add_py_component(RenderStackCls())
        if stack is None:
            scene.destroy_game_object(new_obj)
            return
        _finalize(new_obj, parent_id, "Create RenderStack")

    def _create_empty(parent_id):
        from Infernux.lib import SceneManager
        scene = SceneManager.instance().get_active_scene()
        if scene:
            new_obj = scene.create_game_object("GameObject")
            if new_obj:
                _finalize(new_obj, parent_id, "Create Empty")

    def _create_ui_canvas(parent_id):
        from Infernux.lib import SceneManager
        from Infernux.ui import UICanvas as UICanvasCls
        from Infernux.ui.ui_canvas_utils import invalidate_canvas_cache
        scene = SceneManager.instance().get_active_scene()
        if not scene:
            return
        go = scene.create_game_object("Canvas")
        if go:
            go.add_py_component(UICanvasCls())
            invalidate_canvas_cache()
            _finalize(go, parent_id, "Create Canvas")

    def _create_ui_text(parent_id):
        from Infernux.lib import SceneManager
        from Infernux.ui import UIText as UITextCls
        from Infernux.ui.ui_canvas_utils import invalidate_canvas_cache
        scene = SceneManager.instance().get_active_scene()
        if not scene:
            return
        canvas_pid = _find_canvas_parent_id(scene, parent_id)
        go = scene.create_game_object("Text")
        if go:
            go.add_py_component(UITextCls())
            _finalize(go, canvas_pid, "Create Text")
            invalidate_canvas_cache()

    def _create_ui_button(parent_id):
        from Infernux.lib import SceneManager
        from Infernux.ui import UIButton as UIButtonCls
        from Infernux.ui.ui_canvas_utils import invalidate_canvas_cache
        scene = SceneManager.instance().get_active_scene()
        if not scene:
            return
        canvas_pid = _find_canvas_parent_id(scene, parent_id)
        go = scene.create_game_object("Button")
        if go:
            btn = UIButtonCls()
            btn.width = 160.0
            btn.height = 40.0
            go.add_py_component(btn)
            _finalize(go, canvas_pid, "Create Button")
            invalidate_canvas_cache()

    def _find_canvas_parent_id(scene, parent_id):
        if parent_id == 0:
            return 0
        from Infernux.ui import UICanvas
        obj = scene.find_by_id(parent_id)
        if not obj:
            return parent_id
        cur = obj
        while cur is not None:
            for c in cur.get_py_components():
                if isinstance(c, UICanvas):
                    return cur.id
            cur = cur.get_parent()
        return parent_id

    def _finalize(new_obj, parent_id, description):
        if parent_id and parent_id != 0:
            from Infernux.lib import SceneManager
            scene = SceneManager.instance().get_active_scene()
            if scene:
                parent = scene.find_by_id(parent_id)
                if parent:
                    new_obj.set_parent(parent)
        sel.select(new_obj.id)
        undo.record_create(new_obj.id, description)
        if hp.on_selection_changed:
            hp.on_selection_changed(new_obj.id)

    hp.create_primitive = _create_primitive
    hp.create_light = _create_light
    hp.create_camera = _create_camera
    hp.create_render_stack = _create_render_stack
    hp.create_empty = _create_empty
    hp.create_ui_canvas = _create_ui_canvas
    hp.create_ui_text = _create_ui_text
    hp.create_ui_button = _create_ui_button

    # -- Prefab actions --
    def _save_as_prefab(oid):
        from Infernux.lib import SceneManager, AssetRegistry
        from Infernux.engine.project_context import get_project_root
        from Infernux.engine.prefab_manager import save_prefab, PREFAB_EXTENSION
        import os
        scene = SceneManager.instance().get_active_scene()
        if not scene:
            return
        go = scene.find_by_id(oid)
        if not go:
            return
        root = get_project_root()
        if not root:
            return
        assets_dir = os.path.join(root, "Assets")
        os.makedirs(assets_dir, exist_ok=True)
        adb = None
        registry = AssetRegistry.instance()
        if registry:
            adb = registry.get_asset_database()
        from Infernux.engine.ui.project_file_ops import get_unique_name
        prefab_name = get_unique_name(assets_dir, go.name, PREFAB_EXTENSION)
        file_path = os.path.join(assets_dir, prefab_name + PREFAB_EXTENSION)
        if save_prefab(go, file_path, asset_database=adb):
            Debug.log_internal(f"Prefab saved: {file_path}")

    def _prefab_select_asset(oid):
        from Infernux.lib import SceneManager, AssetRegistry
        scene = SceneManager.instance().get_active_scene()
        go = scene.find_by_id(oid) if scene else None
        if not go:
            return
        guid = getattr(go, 'prefab_guid', '')
        path = _resolve_prefab(guid)
        if path:
            EditorEventBus.instance().emit("select_asset", path)

    def _prefab_open_asset(oid):
        from Infernux.lib import SceneManager
        scene = SceneManager.instance().get_active_scene()
        go = scene.find_by_id(oid) if scene else None
        if not go:
            return
        guid = getattr(go, 'prefab_guid', '')
        path = _resolve_prefab(guid)
        if path:
            EditorEventBus.instance().emit("open_asset", path)

    def _prefab_apply_overrides(oid):
        from Infernux.lib import SceneManager
        scene = SceneManager.instance().get_active_scene()
        go = scene.find_by_id(oid) if scene else None
        if not go:
            return
        guid = getattr(go, 'prefab_guid', '')
        path = _resolve_prefab(guid)
        if path:
            from Infernux.engine.prefab_overrides import apply_overrides_to_prefab
            apply_overrides_to_prefab(go, path)

    def _prefab_revert_overrides(oid):
        from Infernux.lib import SceneManager
        scene = SceneManager.instance().get_active_scene()
        go = scene.find_by_id(oid) if scene else None
        if not go:
            return
        guid = getattr(go, 'prefab_guid', '')
        path = _resolve_prefab(guid)
        if path:
            from Infernux.engine.prefab_overrides import revert_overrides
            revert_overrides(go, path)

    def _prefab_unpack(oid):
        from Infernux.lib import SceneManager
        scene = SceneManager.instance().get_active_scene()
        go = scene.find_by_id(oid) if scene else None
        if not go:
            return
        _unpack_recursive(go)
        Debug.log_internal(f"Unpacked prefab instance: {go.name}")

    def _unpack_recursive(obj):
        try:
            obj.prefab_guid = ""
            obj.prefab_root = False
        except Exception as _exc:
            Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
            pass
        try:
            for child in obj.get_children():
                _unpack_recursive(child)
        except Exception as _exc:
            Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
            pass

    def _resolve_prefab(guid):
        if not guid:
            return None
        try:
            from Infernux.lib import AssetRegistry
            registry = AssetRegistry.instance()
            if registry:
                adb = registry.get_asset_database()
                if adb:
                    return adb.get_path_from_guid(guid)
        except Exception as _exc:
            Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
            pass
        return None

    hp.save_as_prefab = _save_as_prefab
    hp.prefab_select_asset = _prefab_select_asset
    hp.prefab_open_asset = _prefab_open_asset
    hp.prefab_apply_overrides = _prefab_apply_overrides
    hp.prefab_revert_overrides = _prefab_revert_overrides
    hp.prefab_unpack = _prefab_unpack

    # -- Clipboard --
    _clipboard = {"entries": [], "cut": False}

    def _copy_selected(cut):
        from Infernux.lib import SceneManager
        scene = SceneManager.instance().get_active_scene()
        if not scene:
            return False
        ids = sel.get_ids()
        if not ids:
            return False
        selected_set = set(ids)
        roots = []
        for oid in ids:
            obj = scene.find_by_id(oid)
            if obj is None:
                continue
            parent = obj.get_parent()
            skip = False
            while parent is not None:
                if parent.id in selected_set:
                    skip = True
                    break
                parent = parent.get_parent()
            if not skip:
                roots.append(obj)
        if not roots:
            return False
        entries = []
        for obj in roots:
            parent = obj.get_parent()
            transform = getattr(obj, "transform", None)
            entries.append({
                "json": obj.serialize(),
                "source_parent_id": parent.id if parent else None,
                "source_sibling_index": transform.get_sibling_index() if transform else 0,
            })
        _clipboard["entries"] = entries
        _clipboard["cut"] = bool(cut)
        if cut:
            from Infernux.engine.undo import CompoundCommand, DeleteGameObjectCommand, UndoManager
            commands = [DeleteGameObjectCommand(obj.id, "Cut GameObject") for obj in roots]
            mgr = UndoManager.instance()
            if mgr:
                cmd = commands[0] if len(commands) == 1 else CompoundCommand(commands, "Cut GameObjects")
                mgr.execute(cmd)
            else:
                sfm2 = bs.scene_file_manager
                for obj in roots:
                    live = scene.find_by_id(obj.id)
                    if live:
                        scene.destroy_game_object(live)
                if sfm2:
                    sfm2.mark_dirty()
            sel.clear()
            if hp.on_selection_changed:
                hp.on_selection_changed(0)
        return True

    def _paste_clipboard():
        if not _clipboard["entries"]:
            return False
        from Infernux.lib import SceneManager
        from Infernux.engine.undo import CompoundCommand, CreateGameObjectCommand, UndoManager
        from Infernux.engine.prefab_manager import _restore_pending_py_components, _strip_prefab_runtime_fields
        import json
        scene = SceneManager.instance().get_active_scene()
        if not scene:
            return False
        explicit_parent = None
        if sel.count() == 1:
            explicit_parent = scene.find_by_id(sel.get_primary())
        created = []
        for entry in _clipboard["entries"]:
            parent = explicit_parent
            if parent is None:
                src_pid = entry.get("source_parent_id")
                if src_pid is not None:
                    parent = scene.find_by_id(src_pid)
            try:
                obj_data = json.loads(entry["json"])
            except Exception as _exc:
                Debug.log(f"[Suppressed] {type(_exc).__name__}: {_exc}")
                continue
            _strip_prefab_runtime_fields(obj_data)
            new_obj = scene.instantiate_from_json(json.dumps(obj_data), parent)
            if new_obj:
                created.append(new_obj)
        if created and scene.has_pending_py_components():
            sfm2 = bs.scene_file_manager
            adb = getattr(sfm2, "_asset_database", None) if sfm2 else None
            _restore_pending_py_components(scene, asset_database=adb)
        if not created:
            return False
        cids = [o.id for o in created]
        cmds = [CreateGameObjectCommand(cid, "Paste GameObject") for cid in cids]
        mgr = UndoManager.instance()
        if mgr:
            cmd = cmds[0] if len(cmds) == 1 else CompoundCommand(cmds, "Paste GameObjects")
            mgr.record(cmd)
        else:
            sfm2 = bs.scene_file_manager
            if sfm2:
                sfm2.mark_dirty()
        sel.set_ids(cids)
        if hp.on_selection_changed:
            hp.on_selection_changed(cids[-1] if cids else 0)
        if _clipboard["cut"]:
            _clipboard["entries"] = []
            _clipboard["cut"] = False
        return True

    def _has_clipboard_data():
        return bool(_clipboard["entries"])

    hp.copy_selected = _copy_selected
    hp.paste_clipboard = _paste_clipboard
    hp.has_clipboard_data = _has_clipboard_data

    # -- External drop (from Project panel) --
    def _instantiate_prefab(ref, parent_id, is_guid):
        from Infernux.lib import SceneManager, AssetRegistry
        from Infernux.engine.prefab_manager import instantiate_prefab
        scene = SceneManager.instance().get_active_scene()
        if not scene:
            return
        adb = None
        registry = AssetRegistry.instance()
        if registry:
            adb = registry.get_asset_database()
        parent = scene.find_by_id(parent_id) if parent_id else None
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
            _finalize(new_obj, parent_id, "Create Model")

    hp.instantiate_prefab = _instantiate_prefab
    hp.create_model_object = _create_model_object

    # -- Delete selected --
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
