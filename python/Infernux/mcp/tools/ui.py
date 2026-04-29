"""Semantic UI creation MCP tools."""

from __future__ import annotations

from typing import Any

from Infernux.mcp.tools.common import main_thread, register_tool_metadata, serialize_value


def register_ui_tools(mcp) -> None:
    _register_metadata()

    @mcp.tool(name="ui.create_canvas")
    def ui_create_canvas(name: str = "Canvas", reference_width: int = 1920, reference_height: int = 1080, select: bool = False) -> dict:
        """Create a UI Canvas."""

        def _create():
            obj, comp = _create_ui_object("canvas", name, 0)
            comp.reference_width = int(reference_width)
            comp.reference_height = int(reference_height)
            _select_if(obj, select)
            return _ui_snapshot(obj, comp)

        return main_thread("ui.create_canvas", _create)

    @mcp.tool(name="ui.create_text")
    def ui_create_text(name: str = "Text", parent_id: int = 0, text: str = "New Text", rect: dict[str, Any] | None = None) -> dict:
        """Create a UIText element."""

        def _create():
            obj, comp = _create_ui_object("text", name, int(parent_id or 0))
            comp.text = str(text)
            _apply_rect(comp, rect or {})
            return _ui_snapshot(obj, comp)

        return main_thread("ui.create_text", _create)

    @mcp.tool(name="ui.create_button")
    def ui_create_button(name: str = "Button", parent_id: int = 0, label: str = "Button", rect: dict[str, Any] | None = None) -> dict:
        """Create a UIButton element."""

        def _create():
            obj, comp = _create_ui_object("button", name, int(parent_id or 0))
            comp.label = str(label)
            _apply_rect(comp, rect or {})
            return _ui_snapshot(obj, comp)

        return main_thread("ui.create_button", _create)

    @mcp.tool(name="ui.create_image")
    def ui_create_image(name: str = "Image", parent_id: int = 0, texture_path: str = "", rect: dict[str, Any] | None = None) -> dict:
        """Create a UIImage element."""

        def _create():
            obj, comp = _create_ui_object("image", name, int(parent_id or 0))
            comp.texture_path = str(texture_path or "")
            _apply_rect(comp, rect or {})
            return _ui_snapshot(obj, comp)

        return main_thread("ui.create_image", _create)

    @mcp.tool(name="ui.create_panel")
    def ui_create_panel(name: str = "Panel", parent_id: int = 0, color: list | None = None, rect: dict[str, Any] | None = None) -> dict:
        """Create a solid-color panel using UIImage."""

        def _create():
            obj, comp = _create_ui_object("image", name, int(parent_id or 0))
            comp.texture_path = ""
            comp.color = color or [0.1, 0.1, 0.1, 0.85]
            _apply_rect(comp, rect or {})
            return _ui_snapshot(obj, comp)

        return main_thread("ui.create_panel", _create)

    @mcp.tool(name="ui.set_rect")
    def ui_set_rect(object_id: int, rect: dict[str, Any]) -> dict:
        """Set x/y/width/height/rotation on a UI screen component."""

        def _set_rect():
            obj = _find_game_object(object_id)
            comp = _find_ui_screen_component(obj)
            if comp is None:
                raise FileNotFoundError(f"GameObject {object_id} has no screen UI component.")
            _apply_rect(comp, rect or {})
            _mark_ui_dirty()
            return _ui_snapshot(obj, comp)

        return main_thread("ui.set_rect", _set_rect)

    @mcp.tool(name="ui.set_text")
    def ui_set_text(object_id: int, text: str) -> dict:
        """Set text/label on UIText or UIButton."""

        def _set_text():
            obj = _find_game_object(object_id)
            comp = _find_named_component(obj, {"UIText", "UIButton"})
            if comp is None:
                raise FileNotFoundError(f"GameObject {object_id} has no UIText or UIButton.")
            if type(comp).__name__ == "UIButton":
                comp.label = str(text)
            else:
                comp.text = str(text)
            _mark_ui_dirty()
            return _ui_snapshot(obj, comp)

        return main_thread("ui.set_text", _set_text)

    @mcp.tool(name="ui.inspect")
    def ui_inspect() -> dict:
        """Return a compact snapshot of UI canvases and elements."""

        def _inspect():
            from Infernux.lib import SceneManager
            scene = SceneManager.instance().get_active_scene()
            if not scene:
                raise RuntimeError("No active scene.")
            elements = []
            for obj in scene.get_all_objects() or []:
                comp = _find_ui_component(obj)
                if comp is not None:
                    elements.append(_ui_snapshot(obj, comp))
            return {"elements": elements}

        return main_thread("ui.inspect", _inspect)

    @mcp.tool(name="ui.find_by_text")
    def ui_find_by_text(text: str) -> dict:
        """Find UIText/UIButton elements by visible text."""

        def _find():
            from Infernux.lib import SceneManager
            scene = SceneManager.instance().get_active_scene()
            if not scene:
                raise RuntimeError("No active scene.")
            needle = str(text).lower()
            matches = []
            for obj in scene.get_all_objects() or []:
                comp = _find_named_component(obj, {"UIText", "UIButton"})
                if comp is None:
                    continue
                visible = str(getattr(comp, "label", getattr(comp, "text", "")))
                if needle in visible.lower():
                    matches.append(_ui_snapshot(obj, comp))
            return {"matches": matches}

        return main_thread("ui.find_by_text", _find)


def _create_ui_object(kind: str, name: str, parent_id: int):
    from Infernux.lib import SceneManager
    from Infernux.ui import UICanvas, UIText, UIButton
    from Infernux.ui.ui_image import UIImage
    scene = SceneManager.instance().get_active_scene()
    if not scene:
        raise RuntimeError("No active scene.")
    if kind != "canvas" and not parent_id:
        canvas = _find_first_canvas(scene)
        if canvas is None:
            canvas_obj, _canvas_comp = _create_ui_object("canvas", "Canvas", 0)
            parent_id = int(canvas_obj.id)
        else:
            parent_id = int(canvas.id)
    obj = scene.create_game_object(str(name or kind.title()))
    if parent_id:
        parent = scene.find_by_id(int(parent_id))
        if parent is None:
            raise FileNotFoundError(f"Parent GameObject {parent_id} was not found.")
        obj.set_parent(parent)
    cls = {"canvas": UICanvas, "text": UIText, "button": UIButton, "image": UIImage}[kind]
    comp = cls()
    obj.add_py_component(comp)
    _mark_ui_dirty()
    return obj, comp


def _find_first_canvas(scene):
    try:
        for obj in scene.get_all_objects() or []:
            if _find_named_component(obj, {"UICanvas"}) is not None:
                return obj
    except Exception:
        pass
    return None


def _apply_rect(comp, rect: dict[str, Any]) -> None:
    for key in ("x", "y", "width", "height", "rotation", "opacity", "corner_radius"):
        if key in rect:
            setattr(comp, key, float(rect[key]))


def _ui_snapshot(obj, comp) -> dict[str, Any]:
    data = {
        "object_id": int(obj.id),
        "name": str(obj.name),
        "type": type(comp).__name__,
        "parent_id": int(getattr(obj.get_parent(), "id", 0) or 0),
        "fields": {},
    }
    for key in ("text", "label", "x", "y", "width", "height", "rotation", "opacity", "corner_radius", "reference_width", "reference_height", "texture_path", "color"):
        if hasattr(comp, key):
            data["fields"][key] = serialize_value(getattr(comp, key))
    return data


def _find_game_object(object_id: int):
    from Infernux.mcp.tools.common import find_game_object
    return find_game_object(object_id)


def _find_named_component(obj, names: set[str]):
    try:
        for comp in obj.get_py_components() or []:
            if type(comp).__name__ in names:
                return comp
    except Exception:
        pass
    return None


def _find_ui_component(obj):
    return _find_named_component(obj, {"UICanvas", "UIText", "UIButton", "UIImage"})


def _find_ui_screen_component(obj):
    try:
        from Infernux.ui.inx_ui_screen_component import InxUIScreenComponent
        for comp in obj.get_py_components() or []:
            if isinstance(comp, InxUIScreenComponent):
                return comp
    except Exception:
        pass
    return None


def _mark_ui_dirty() -> None:
    try:
        from Infernux.ui.ui_canvas_utils import invalidate_canvas_cache
        invalidate_canvas_cache()
    except Exception:
        pass
    try:
        from Infernux.engine.scene_manager import SceneFileManager
        sfm = SceneFileManager.instance()
        if sfm:
            sfm.mark_dirty()
    except Exception:
        pass


def _select_if(obj, select: bool) -> None:
    if not select:
        return
    try:
        from Infernux.engine.ui.selection_manager import SelectionManager
        SelectionManager.instance().select(int(obj.id))
    except Exception:
        pass


def _register_metadata() -> None:
    for name, summary in {
        "ui.create_canvas": "Create a UICanvas root.",
        "ui.create_text": "Create a UIText element.",
        "ui.create_button": "Create a UIButton element.",
        "ui.create_panel": "Create a solid-color panel.",
        "ui.create_image": "Create a UIImage element.",
        "ui.set_rect": "Set UI element rectangle.",
        "ui.set_text": "Set text/label on a UI element.",
        "ui.inspect": "Inspect UI elements.",
        "ui.find_by_text": "Find UI elements by visible text.",
    }.items():
        register_tool_metadata(name, summary=summary)
