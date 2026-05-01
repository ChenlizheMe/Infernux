"""Camera, lighting, and render-view semantic MCP tools."""

from __future__ import annotations

import math
from typing import Any

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

    @mcp.tool(name="camera.describe_view")
    def camera_describe_view(camera_id: int = 0) -> dict:
        """Describe a camera's transform, projection, and viewport."""

        def _describe():
            cam = _resolve_camera_object(camera_id, create_if_missing=False)
            comp = _find_component(cam, "Camera")
            if comp is None:
                raise ValueError(f"GameObject {int(cam.id)} does not have a Camera component.")
            return {"camera": _camera_snapshot(cam, comp)}

        return main_thread("camera.describe_view", _describe, arguments={"camera_id": camera_id})

    @mcp.tool(name="camera.visibility_report")
    def camera_visibility_report(
        camera_id: int = 0,
        target_ids: list[int] | None = None,
        target_query: dict[str, Any] | None = None,
        padding: float = 0.08,
    ) -> dict:
        """Report whether targets are inside the camera viewport."""

        def _report():
            cam = _resolve_camera_object(camera_id, create_if_missing=False)
            comp = _find_component(cam, "Camera")
            if comp is None:
                raise ValueError(f"GameObject {int(cam.id)} does not have a Camera component.")
            targets = _resolve_targets(target_ids or [], target_query or {})
            return _visibility_report(cam, comp, targets, float(padding))

        return main_thread(
            "camera.visibility_report",
            _report,
            arguments={"camera_id": camera_id, "target_ids": target_ids or [], "target_query": target_query or {}},
        )

    @mcp.tool(name="camera.frame_targets")
    def camera_frame_targets(
        camera_id: int = 0,
        target_ids: list[int] | None = None,
        target_query: dict[str, Any] | None = None,
        padding: float = 0.18,
        mode: str = "move_or_zoom",
    ) -> dict:
        """Adjust camera position or orthographic size to frame target objects."""

        def _frame():
            cam = _resolve_camera_object(camera_id, create_if_missing=True)
            comp = _find_component(cam, "Camera")
            if comp is None:
                raise ValueError(f"GameObject {int(cam.id)} does not have a Camera component.")
            targets = _resolve_targets(target_ids or [], target_query or {})
            if not targets:
                raise FileNotFoundError("No target objects matched target_ids or target_query.")
            bounds = _combined_bounds(targets)
            applied = _apply_frame(cam, comp, bounds, float(padding), str(mode or "move_or_zoom"))
            _try_set_scene_main_camera(int(cam.id))
            _mark_scene_dirty()
            report = _visibility_report(cam, comp, targets, float(padding))
            return {"camera": _camera_snapshot(cam, comp), "targets": [_target_entry(obj) for obj in targets], "bounds": bounds, "applied": applied, "visibility": report}

        return main_thread(
            "camera.frame_targets",
            _frame,
            arguments={"camera_id": camera_id, "target_ids": target_ids or [], "target_query": target_query or {}, "padding": padding, "mode": mode},
        )

    @mcp.tool(name="camera.look_at")
    def camera_look_at(
        camera_id: int = 0,
        target_id: int = 0,
        position: dict | list | tuple = None,
        distance: float = 0.0,
        height: float = 0.0,
    ) -> dict:
        """Point a camera at a target object or world position."""

        def _look_at():
            cam = _resolve_camera_object(camera_id, create_if_missing=True)
            if _find_component(cam, "Camera") is None:
                raise ValueError(f"GameObject {int(cam.id)} does not have a Camera component.")
            if target_id:
                target = _find_game_object(int(target_id))
                target_pos = _target_center(target)
            elif position is not None:
                value = coerce_vector3(position)
                target_pos = [float(value.x), float(value.y), float(value.z)]
            else:
                raise ValueError("camera.look_at requires target_id or position.")
            _look_at_position(cam, target_pos, float(distance or 0.0), float(height or 0.0))
            _try_set_scene_main_camera(int(cam.id))
            _mark_scene_dirty()
            comp = _find_component(cam, "Camera")
            return {"camera": _camera_snapshot(cam, comp), "target_position": target_pos}

        return main_thread("camera.look_at", _look_at, arguments={"camera_id": camera_id, "target_id": target_id, "position": position})

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


def _resolve_camera_object(camera_id: int = 0, *, create_if_missing: bool):
    if int(camera_id or 0):
        return _find_game_object(int(camera_id))
    cameras = _find_cameras()
    chosen = _pick_main_camera(cameras)
    if chosen is None:
        if not create_if_missing:
            raise FileNotFoundError("No camera found.")
        chosen = _ensure_camera_object()
    return _find_game_object(int(chosen["id"]))


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


def _resolve_targets(target_ids: list[int], target_query: dict[str, Any]) -> list[Any]:
    targets = []
    seen: set[int] = set()
    for object_id in target_ids or []:
        obj = _find_game_object(int(object_id))
        seen.add(int(obj.id))
        targets.append(obj)
    if target_query:
        from Infernux.lib import SceneManager
        scene = SceneManager.instance().get_active_scene()
        if not scene:
            raise RuntimeError("No active scene.")
        for obj in list(scene.get_all_objects() or []):
            if int(obj.id) in seen:
                continue
            if _matches_target_query(obj, target_query):
                seen.add(int(obj.id))
                targets.append(obj)
    if not targets and not target_query:
        targets = _default_subjects()
    return targets


def _default_subjects(limit: int = 8) -> list[Any]:
    from Infernux.lib import SceneManager
    scene = SceneManager.instance().get_active_scene()
    if not scene:
        raise RuntimeError("No active scene.")
    scored = []
    for obj in list(scene.get_all_objects() or []):
        score = _subject_score(obj)
        if score > 0:
            scored.append((score, obj))
    scored.sort(key=lambda item: (-item[0], str(getattr(item[1], "name", ""))))
    return [obj for _score, obj in scored[: max(int(limit), 1)]]


def _matches_target_query(obj, query: dict[str, Any]) -> bool:
    query = query or {}
    name = str(getattr(obj, "name", "")).lower()
    path = _object_path(obj).lower()
    if query.get("name_contains") and str(query["name_contains"]).lower() not in name:
        return False
    if query.get("path_contains") and str(query["path_contains"]).lower() not in path:
        return False
    if query.get("component_type") and _find_component(obj, str(query["component_type"])) is None:
        return False
    component_any = [str(item) for item in query.get("component_any", []) or []]
    if component_any and not any(_find_component(obj, item) is not None for item in component_any):
        return False
    if query.get("tag") and str(getattr(obj, "tag", "")) != str(query["tag"]):
        return False
    return True


def _subject_score(obj) -> int:
    names = set(_component_names(obj))
    lower = str(getattr(obj, "name", "")).lower()
    score = 0
    if names.intersection({"MeshRenderer", "SpriteRenderer", "SkinnedMeshRenderer"}):
        score += 6
    if names and not names.issubset({"Transform", "Camera", "Light", "UICanvas", "UIText", "UIButton", "UIImage", "RenderStack"}):
        score += 2
    for token, weight in {"player": 5, "main": 4, "target": 4, "subject": 4, "board": 3, "piece": 2, "ball": 2}.items():
        if token in lower:
            score += weight
    if names.intersection({"Camera", "Light"}) and not names.intersection({"MeshRenderer", "SpriteRenderer", "SkinnedMeshRenderer"}):
        score -= 8
    return score


def _visibility_report(cam, comp, targets: list[Any], padding: float) -> dict[str, Any]:
    viewport_w = max(int(getattr(comp, "pixel_width", 0) or 0), 1)
    viewport_h = max(int(getattr(comp, "pixel_height", 0) or 0), 1)
    if viewport_w == 1:
        viewport_w = 1920
    if viewport_h == 1:
        viewport_h = 1080
    margin_x = viewport_w * max(float(padding), 0.0)
    margin_y = viewport_h * max(float(padding), 0.0)
    target_reports = []
    all_points = []
    for target in targets:
        world_points = _bounds_points(_object_bounds(target))
        screen_points = [_world_to_screen(comp, point) for point in world_points]
        screen_points = [point for point in screen_points if point is not None]
        visible = bool(screen_points) and all(
            margin_x <= point[0] <= viewport_w - margin_x and margin_y <= point[1] <= viewport_h - margin_y
            for point in screen_points
        )
        all_points.extend(screen_points)
        target_reports.append({
            **_target_entry(target),
            "visible_with_padding": visible,
            "screen_bounds": _screen_bounds(screen_points),
            "screen_points": screen_points,
        })
    overall = bool(all_points) and all(
        margin_x <= point[0] <= viewport_w - margin_x and margin_y <= point[1] <= viewport_h - margin_y
        for point in all_points
    )
    return {
        "camera": _camera_snapshot(cam, comp),
        "viewport": {"width": viewport_w, "height": viewport_h, "padding": float(padding)},
        "all_visible_with_padding": overall,
        "targets": target_reports,
        "screen_bounds": _screen_bounds(all_points),
        "suggestion": _visibility_suggestion(_screen_bounds(all_points), viewport_w, viewport_h, margin_x, margin_y),
    }


def _apply_frame(cam, comp, bounds: dict[str, Any], padding: float, mode: str) -> dict[str, Any]:
    center = bounds["center"]
    size = bounds["size"]
    trans = cam.transform
    projection_mode = int(getattr(comp, "projection_mode", 0) or 0)
    try:
        from Infernux.lib import Vector3
    except Exception:
        Vector3 = None
    if projection_mode == 1:
        half_height = max(size[1] * 0.5, size[0] * 0.5 / max(float(getattr(comp, "aspect_ratio", 1.778) or 1.778), 0.1), 0.5)
        new_size = half_height * (1.0 + max(float(padding), 0.0) * 2.0)
        if mode in {"move", "move_or_zoom", "zoom"}:
            try:
                comp.orthographic_size = float(new_size)
            except Exception:
                pass
        old_pos = _vec(trans.position)
        if mode in {"move", "move_or_zoom"} and Vector3 is not None:
            trans.position = Vector3(float(center[0]), float(center[1]), float(old_pos[2]))
        return {"mode": mode, "projection": "orthographic", "orthographic_size": float(getattr(comp, "orthographic_size", new_size)), "center": center}

    radius = max(max(size) * 0.5, 0.5)
    fov = max(float(getattr(comp, "field_of_view", 60.0) or 60.0), 1.0)
    distance = radius / max(math.tan(math.radians(fov) * 0.5), 0.01)
    distance *= 1.0 + max(float(padding), 0.0) * 2.5
    height = max(size[1] * 0.18, 0.0)
    _look_at_position(cam, center, distance, height)
    return {"mode": mode, "projection": "perspective", "distance": distance, "center": center}


def _look_at_position(cam, target_pos: list[float], distance: float = 0.0, height: float = 0.0) -> None:
    trans = cam.transform
    pos = _vec(trans.position)
    if distance > 0.0:
        direction = _normalize([pos[0] - target_pos[0], pos[1] - target_pos[1], pos[2] - target_pos[2]])
        if _length(direction) < 0.001:
            direction = [0.0, 0.25, -1.0]
        pos = [
            target_pos[0] + direction[0] * distance,
            target_pos[1] + direction[1] * distance + float(height),
            target_pos[2] + direction[2] * distance,
        ]
        from Infernux.lib import Vector3
        trans.position = Vector3(float(pos[0]), float(pos[1]), float(pos[2]))
    dx = target_pos[0] - pos[0]
    dy = target_pos[1] - pos[1]
    dz = target_pos[2] - pos[2]
    yaw = math.degrees(math.atan2(dx, dz))
    flat = math.sqrt(dx * dx + dz * dz)
    pitch = -math.degrees(math.atan2(dy, max(flat, 0.0001)))
    from Infernux.lib import Vector3
    trans.euler_angles = Vector3(float(pitch), float(yaw), 0.0)


def _combined_bounds(objects: list[Any]) -> dict[str, Any]:
    boxes = [_object_bounds(obj) for obj in objects]
    min_v = [min(box["min"][i] for box in boxes) for i in range(3)]
    max_v = [max(box["max"][i] for box in boxes) for i in range(3)]
    center = [(min_v[i] + max_v[i]) * 0.5 for i in range(3)]
    size = [max_v[i] - min_v[i] for i in range(3)]
    return {"min": min_v, "max": max_v, "center": center, "size": size}


def _object_bounds(obj) -> dict[str, Any]:
    points = []
    for item in _object_and_descendants(obj):
        points.extend(_approx_object_points(item))
    if not points:
        points = [_target_center(obj)]
    min_v = [min(point[i] for point in points) for i in range(3)]
    max_v = [max(point[i] for point in points) for i in range(3)]
    center = [(min_v[i] + max_v[i]) * 0.5 for i in range(3)]
    size = [max_v[i] - min_v[i] for i in range(3)]
    return {"min": min_v, "max": max_v, "center": center, "size": size}


def _object_and_descendants(obj) -> list[Any]:
    result = [obj]
    try:
        children = list(obj.get_children() or [])
    except Exception:
        children = []
    for child in children:
        result.extend(_object_and_descendants(child))
    return result


def _approx_object_points(obj) -> list[list[float]]:
    pos = _target_center(obj)
    scale = _vector_list(getattr(getattr(obj, "transform", None), "local_scale", None), default=[1.0, 1.0, 1.0])
    half = [max(abs(scale[i]) * 0.5, 0.05) for i in range(3)]
    if not set(_component_names(obj)).intersection({"MeshRenderer", "SpriteRenderer", "SkinnedMeshRenderer"}):
        half = [0.05, 0.05, 0.05]
    return [
        [pos[0] - half[0], pos[1] - half[1], pos[2] - half[2]],
        [pos[0] + half[0], pos[1] + half[1], pos[2] + half[2]],
    ]


def _bounds_points(bounds: dict[str, Any]) -> list[list[float]]:
    mn = bounds["min"]
    mx = bounds["max"]
    return [
        [x, y, z]
        for x in (mn[0], mx[0])
        for y in (mn[1], mx[1])
        for z in (mn[2], mx[2])
    ]


def _world_to_screen(comp, point: list[float]) -> list[float] | None:
    try:
        value = comp.world_to_screen_point(float(point[0]), float(point[1]), float(point[2]))
        if value is None:
            return None
        return [float(value[0]), float(value[1])]
    except Exception:
        return None


def _screen_bounds(points: list[list[float]]) -> dict[str, Any]:
    if not points:
        return {"available": False}
    min_v = [min(point[i] for point in points) for i in range(2)]
    max_v = [max(point[i] for point in points) for i in range(2)]
    return {"available": True, "min": min_v, "max": max_v, "center": [(min_v[0] + max_v[0]) * 0.5, (min_v[1] + max_v[1]) * 0.5], "size": [max_v[0] - min_v[0], max_v[1] - min_v[1]]}


def _visibility_suggestion(bounds: dict[str, Any], width: int, height: int, margin_x: float, margin_y: float) -> dict[str, Any]:
    if not bounds.get("available"):
        return {"status": "unknown", "message": "No screen-space points were available. Try camera.frame_targets."}
    dx = 0.0
    dy = 0.0
    if bounds["min"][0] < margin_x:
        dx = bounds["min"][0] - margin_x
    elif bounds["max"][0] > width - margin_x:
        dx = bounds["max"][0] - (width - margin_x)
    if bounds["min"][1] < margin_y:
        dy = bounds["min"][1] - margin_y
    elif bounds["max"][1] > height - margin_y:
        dy = bounds["max"][1] - (height - margin_y)
    if abs(dx) <= 1e-4 and abs(dy) <= 1e-4:
        return {"status": "ok", "message": "Targets fit inside the camera view with padding."}
    return {"status": "needs_adjustment", "screen_offset": [dx, dy], "message": "Targets exceed the padded viewport; use camera.frame_targets."}


def _camera_snapshot(cam, comp) -> dict[str, Any]:
    return {
        "id": int(cam.id),
        "name": str(cam.name),
        "position": _vec(cam.transform.position),
        "euler_angles": _vec(cam.transform.euler_angles),
        "projection_mode": int(getattr(comp, "projection_mode", 0) or 0),
        "field_of_view": float(getattr(comp, "field_of_view", 60.0) or 60.0),
        "orthographic_size": float(getattr(comp, "orthographic_size", 0.0) or 0.0),
        "near_clip": float(getattr(comp, "near_clip", 0.0) or 0.0),
        "far_clip": float(getattr(comp, "far_clip", 0.0) or 0.0),
        "aspect_ratio": float(getattr(comp, "aspect_ratio", 1.778) or 1.778),
        "pixel_width": int(getattr(comp, "pixel_width", 0) or 0),
        "pixel_height": int(getattr(comp, "pixel_height", 0) or 0),
    }


def _target_entry(obj) -> dict[str, Any]:
    return {"id": int(obj.id), "name": str(obj.name), "path": _object_path(obj), "components": _component_names(obj)}


def _target_center(obj) -> list[float]:
    return _vector_list(getattr(getattr(obj, "transform", None), "position", None))


def _component_names(obj) -> list[str]:
    names: list[str] = []
    try:
        for comp in obj.get_components() or []:
            names.append(str(getattr(comp, "type_name", type(comp).__name__)))
    except Exception:
        pass
    try:
        for comp in obj.get_py_components() or []:
            type_name = str(getattr(comp, "type_name", type(comp).__name__))
            if type_name not in names:
                names.append(type_name)
    except Exception:
        pass
    return names


def _object_path(obj) -> str:
    parts = []
    current = obj
    while current is not None:
        parts.append(str(current.name))
        try:
            current = current.get_parent()
        except Exception:
            current = None
    return "/".join(reversed(parts))


def _vector_list(value, default: list[float] | None = None) -> list[float]:
    if value is None:
        return list(default or [0.0, 0.0, 0.0])
    return [float(getattr(value, axis, 0.0)) for axis in ("x", "y", "z")]


def _length(value: list[float]) -> float:
    return math.sqrt(sum(part * part for part in value))


def _normalize(value: list[float]) -> list[float]:
    length = _length(value)
    if length <= 0.0001:
        return [0.0, 0.0, 0.0]
    return [part / length for part in value]


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
        "camera.describe_view": "Describe camera transform, projection, and viewport.",
        "camera.visibility_report": "Report whether target objects fit inside a camera view.",
        "camera.frame_targets": "Move or zoom a camera to frame target objects.",
        "camera.look_at": "Point a camera at a target object or world position.",
        "camera.attach_to_target": "Parent a camera to a target with local offset.",
        "camera.setup_third_person": "Configure a third-person camera rig.",
        "camera.setup_2d_card_game": "Configure an orthographic camera for card/UI games.",
        "lighting.ensure_default": "Ensure a directional light exists.",
    }.items():
        category = "scene/lighting" if name.startswith("lighting.") else "camera/framing"
        register_tool_metadata(
            name,
            summary=summary,
            category=category,
            tags=["camera", "view", "framing", "visibility"] if name.startswith("camera.") else ["light", "scene"],
            aliases=["frame subject", "fit target", "main camera", "相机", "主体照全"] if name.startswith("camera.") else ["default light"],
            next_suggested_tools=["scene.query.summary", "scene.query.subjects", "camera.visibility_report", "runtime.read_errors"],
        )
