"""Context-menu creation callbacks for the Hierarchy panel."""
from __future__ import annotations

from Infernux.debug import Debug
from Infernux.engine.bootstrap_hierarchy._helpers import _get_py_components_safe


def wire_creation_callbacks(ctx):
    """Wire all object-creation callbacks onto the hierarchy panel."""
    hp = ctx.hp
    sel = ctx.sel
    undo = ctx.undo

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

    def _create_primitive(type_idx, parent_id):
        from Infernux.lib import SceneManager, PrimitiveType
        scene = SceneManager.instance().get_active_scene()
        if not scene:
            return
        types = [PrimitiveType.Cube, PrimitiveType.Sphere, PrimitiveType.Capsule,
                 PrimitiveType.Cylinder, PrimitiveType.Plane, PrimitiveType.Quad]
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
            new_obj.add_component("Camera")
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

    def _find_canvas_parent_id(scene, parent_id):
        if parent_id == 0:
            return 0
        from Infernux.ui import UICanvas
        obj = scene.find_by_id(parent_id)
        if not obj:
            return parent_id
        cur = obj
        while cur is not None:
            for c in _get_py_components_safe(cur):
                if isinstance(c, UICanvas):
                    return cur.id
            cur = cur.get_parent()
        return parent_id

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

    hp.create_primitive = _create_primitive
    hp.create_light = _create_light
    hp.create_camera = _create_camera
    hp.create_render_stack = _create_render_stack
    hp.create_empty = _create_empty
    hp.create_ui_canvas = _create_ui_canvas
    hp.create_ui_text = _create_ui_text
    hp.create_ui_button = _create_ui_button
