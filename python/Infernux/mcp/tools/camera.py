"""Camera, lighting, and render-view semantic MCP tools."""

from __future__ import annotations

from Infernux.mcp.tools.common import coerce_vector3, main_thread, register_tool_metadata


def register_camera_tools(mcp) -> None:
    _register_metadata()

    @mcp.tool(name="camera.find_main")
    def camera_find_main() -> dict:
        """Find likely main cameras in the active scene."""

        def _find():
            cameras = _find_cameras()
            preferred = _pick_main_camera(cameras)
            return {"main": preferred, "cameras": cameras}

        return main_thread("camera.find_main", _find)

    @mcp.tool(name="camera.ensure_main")
    def camera_ensure_main(name: str = "Main Camera", create_if_missing: bool = True) -> dict:
        """Return an existing main camera or create one if needed."""

        def _ensure():
            cameras = _find_cameras()
            chosen = _pick_main_camera(cameras)
            if chosen is None and create_if_missing:
                from Infernux.engine.hierarchy_creation_service import HierarchyCreationService
                created = HierarchyCreationService.instance().create("rendering.camera", name=name, select=False)
                chosen = {
                    "id": int(created["id"]),
                    "name": str(created["name"]),
                    "component_id": 0,
                    "reason": "created",
                }
            if chosen is None:
                raise FileNotFoundError("No camera found and create_if_missing is false.")
            _try_set_scene_main_camera(int(chosen["id"]))
            return {"camera": chosen, "created": chosen.get("reason") == "created"}

        return main_thread("camera.ensure_main", _ensure)

    @mcp.tool(name="camera.set_main")
    def camera_set_main(object_id: int) -> dict:
        """Set Scene.main_camera when supported by the binding."""

        def _set():
            obj = _find_game_object(object_id)
            if _find_component(obj, "Camera") is None:
                raise ValueError(f"GameObject {object_id} does not have a Camera component.")
            applied = _try_set_scene_main_camera(int(object_id))
            return {"object_id": int(object_id), "applied": applied}

        return main_thread("camera.set_main", _set)

    @mcp.tool(name="camera.attach_to_target")
    def camera_attach_to_target(
        camera_id: int,
        target_id: int,
        local_position: dict | list | tuple = None,
        local_euler_angles: dict | list | tuple = None,
        world_position_stays: bool = False,
    ) -> dict:
        """Parent a camera to a target with optional local offset."""

        def _attach():
            cam = _find_game_object(camera_id)
            target = _find_game_object(target_id)
            if _find_component(cam, "Camera") is None:
                raise ValueError(f"GameObject {camera_id} does not have a Camera component.")
            cam.set_parent(target, bool(world_position_stays))
            if local_position is not None:
                cam.transform.local_position = coerce_vector3(local_position)
            if local_euler_angles is not None:
                cam.transform.local_euler_angles = coerce_vector3(local_euler_angles)
            _mark_scene_dirty()
            return {
                "camera_id": int(cam.id),
                "target_id": int(target.id),
                "local_position": _vec(cam.transform.local_position),
                "local_euler_angles": _vec(cam.transform.local_euler_angles),
            }

        return main_thread("camera.attach_to_target", _attach)

    @mcp.tool(name="camera.setup_third_person")
    def camera_setup_third_person(
        target_id: int,
        camera_id: int = 0,
        local_position: dict | list | tuple = None,
        local_euler_angles: dict | list | tuple = None,
        field_of_view: float = 65.0,
    ) -> dict:
        """Configure a main camera as a third-person child of target_id."""

        def _setup():
            if camera_id:
                cam = _find_game_object(camera_id)
            else:
                ensured = _ensure_camera_object()
                cam = _find_game_object(int(ensured["id"]))
            target = _find_game_object(target_id)
            cam.set_parent(target, False)
            cam.transform.local_position = coerce_vector3(local_position or {"x": 0.0, "y": 3.0, "z": -7.0})
            cam.transform.local_euler_angles = coerce_vector3(local_euler_angles or {"x": 18.0, "y": 0.0, "z": 0.0})
            comp = _find_component(cam, "Camera")
            if comp is not None:
                try:
                    comp.field_of_view = float(field_of_view)
                except Exception:
                    pass
            _try_set_scene_main_camera(int(cam.id))
            _mark_scene_dirty()
            return {
                "camera_id": int(cam.id),
                "target_id": int(target.id),
                "local_position": _vec(cam.transform.local_position),
                "local_euler_angles": _vec(cam.transform.local_euler_angles),
                "field_of_view": float(field_of_view),
            }

        return main_thread("camera.setup_third_person", _setup)

    @mcp.tool(name="camera.setup_2d_card_game")
    def camera_setup_2d_card_game(
        camera_id: int = 0,
        position: dict | list | tuple = None,
        euler_angles: dict | list | tuple = None,
        orthographic_size: float = 8.0,
    ) -> dict:
        """Configure a stable orthographic camera for card/UI-heavy games."""

        def _setup():
            if camera_id:
                cam = _find_game_object(camera_id)
            else:
                ensured = _ensure_camera_object()
                cam = _find_game_object(int(ensured["id"]))
            cam.transform.position = coerce_vector3(position or {"x": 0.0, "y": 0.0, "z": 10.0})
            cam.transform.euler_angles = coerce_vector3(euler_angles or {"x": 0.0, "y": 180.0, "z": 0.0})
            comp = _find_component(cam, "Camera")
            if comp is not None:
                for field, value in (("projection_mode", 1), ("orthographic_size", float(orthographic_size)), ("near_clip", 0.01), ("far_clip", 1000.0)):
                    try:
                        setattr(comp, field, value)
                    except Exception:
                        pass
            _try_set_scene_main_camera(int(cam.id))
            _mark_scene_dirty()
            return {
                "camera_id": int(cam.id),
                "position": _vec(cam.transform.position),
                "euler_angles": _vec(cam.transform.euler_angles),
                "orthographic_size": float(orthographic_size),
            }

        return main_thread("camera.setup_2d_card_game", _setup)

    @mcp.tool(name="lighting.ensure_default")
    def lighting_ensure_default() -> dict:
        """Ensure the scene has a usable directional light."""

        def _ensure():
            from Infernux.engine.hierarchy_creation_service import HierarchyCreationService
            from Infernux.lib import SceneManager, Vector3
            scene = SceneManager.instance().get_active_scene()
            if not scene:
                raise RuntimeError("No active scene.")
            for obj in scene.get_all_objects() or []:
                if _find_component(obj, "Light") is not None:
                    return {"light_id": int(obj.id), "created": False}
            created = HierarchyCreationService.instance().create("light.directional", name="Directional Light", select=False)
            obj = scene.find_by_id(int(created["id"]))
            if obj and obj.transform:
                obj.transform.euler_angles = Vector3(50.0, -30.0, 0.0)
            return {"light_id": int(created["id"]), "created": True}

        return main_thread("lighting.ensure_default", _ensure)


def _find_cameras() -> list[dict]:
    from Infernux.lib import SceneManager
    scene = SceneManager.instance().get_active_scene()
    if not scene:
        raise RuntimeError("No active scene.")
    cameras = []
    main_camera = getattr(scene, "main_camera", None)
    main_owner_id = int(getattr(getattr(main_camera, "game_object", None), "id", 0) or 0)
    for obj in list(scene.get_all_objects() or []):
        comp = _find_component(obj, "Camera")
        if comp is None:
            continue
        cameras.append({
            "id": int(obj.id),
            "name": str(obj.name),
            "component_id": int(getattr(comp, "component_id", 0) or 0),
            "is_scene_main": main_owner_id == int(obj.id),
        })
    return cameras


def _pick_main_camera(cameras: list[dict]) -> dict | None:
    if not cameras:
        return None
    for cam in cameras:
        if cam.get("is_scene_main"):
            return {**cam, "reason": "scene.main_camera"}
    for cam in cameras:
        if str(cam.get("name", "")).lower() == "main camera":
            return {**cam, "reason": "name"}
    return {**cameras[0], "reason": "first_camera"}


def _ensure_camera_object() -> dict:
    chosen = _pick_main_camera(_find_cameras())
    if chosen is not None:
        return chosen
    from Infernux.engine.hierarchy_creation_service import HierarchyCreationService
    return HierarchyCreationService.instance().create("rendering.camera", name="Main Camera", select=False)


def _try_set_scene_main_camera(object_id: int) -> bool:
    from Infernux.lib import SceneManager
    scene = SceneManager.instance().get_active_scene()
    if not scene:
        return False
    obj = scene.find_by_id(int(object_id))
    if obj is None:
        return False
    comp = _find_component(obj, "Camera")
    if comp is None:
        return False
    try:
        scene.main_camera = comp
        return True
    except Exception:
        return False


def _find_game_object(object_id: int):
    from Infernux.mcp.tools.common import find_game_object
    return find_game_object(object_id)


def _find_component(obj, component_type: str):
    try:
        comp = obj.get_component(component_type)
        if comp is not None:
            return comp
    except Exception:
        pass
    try:
        for comp in obj.get_components() or []:
            if getattr(comp, "type_name", type(comp).__name__) == component_type:
                return comp
    except Exception:
        pass
    return None


def _mark_scene_dirty() -> None:
    try:
        from Infernux.engine.scene_manager import SceneFileManager
        sfm = SceneFileManager.instance()
        if sfm:
            sfm.mark_dirty()
    except Exception:
        pass


def _vec(value) -> list[float]:
    return [float(value.x), float(value.y), float(value.z)]


def _register_metadata() -> None:
    for name, summary in {
        "camera.find_main": "Find cameras and pick the best main camera candidate.",
        "camera.ensure_main": "Reuse an existing main camera or create one if missing.",
        "camera.set_main": "Set Scene.main_camera when the engine binding supports it.",
        "camera.attach_to_target": "Parent a camera to a target with local offset.",
        "camera.setup_third_person": "Configure a third-person camera rig.",
        "camera.setup_2d_card_game": "Configure an orthographic camera for card/UI games.",
        "lighting.ensure_default": "Ensure a directional light exists.",
    }.items():
        register_tool_metadata(name, summary=summary, next_suggested_tools=["scene.inspect", "runtime.read_errors"])
