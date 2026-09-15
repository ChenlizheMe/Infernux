"""Runtime-owned Screen UI command submission.

Screen-space UI is render data, not presentation-panel data.  This service
builds the native ScreenUI command lists at the camera-submission boundary so
Game captures and hidden Game tabs consume the same current-frame HUD.
"""

from __future__ import annotations

import weakref
import math
from Infernux.engine.ui.runtime_canvas_snapshot import (
    collect_sorted_runtime_canvas_snapshot,
    runtime_canvas_snapshot_token,
)
from Infernux.ui.inx_ui_screen_component import (
    WORLD_UI_PIXELS_PER_UNIT,
    clear_rect_cache,
    is_ui_screen_component,
    _get_layout_revision,
)
from Infernux.ui.ui_render_dispatch import (
    dispatch as _ui_dispatch,
    resolve_text_layout as _resolve_text_layout,
    runtime_ui_revision as _runtime_ui_revision,
    canvas_elements,
    _runtime_command_epoch,
)
from Infernux.ui.ui_command_packets import UICommandPackets
from Infernux.ui.ui_texture_cache import get_shared_cache as _get_tex_cache


_world_input_targets = weakref.WeakKeyDictionary()
_world_elements_key = None
_world_elements = ()
_input_world_elements = None
_input_canvases = None
_input_canvas_token = None
_input_surfaces = ()
_world_projection_targets = ()
_world_projection_geometry = None


def _canvas_metrics(canvas, viewport_width: float, viewport_height: float):
    """Return scale and logical size for screen UI canvases.

    Player scene publication can briefly retain a Canvas instance created by a
    previous Python module identity.  Such an object still carries the
    authored scalar fields but may not expose the newer helper methods.  Keep
    the metric calculation at the submission boundary so that this
    compatibility case does not interrupt the render/input frame.
    """
    compute_scale = getattr(canvas, "compute_scale", None)
    compute_logical_size = getattr(canvas, "compute_logical_size", None)
    if callable(compute_scale) and callable(compute_logical_size):
        scale_x, scale_y, text_scale = compute_scale(
            float(viewport_width), float(viewport_height)
        )
        logical_width, logical_height = compute_logical_size(
            float(viewport_width), float(viewport_height)
        )
        return (
            float(scale_x), float(scale_y), float(text_scale),
            float(logical_width), float(logical_height),
        )

    ref_w = max(1.0, float(getattr(canvas, "reference_width", 1920.0)))
    ref_h = max(1.0, float(getattr(canvas, "reference_height", 1080.0)))
    screen_w = max(1.0, float(viewport_width))
    screen_h = max(1.0, float(viewport_height))
    mode = int(getattr(canvas, "ui_scale_mode", 1))
    if mode in (0, 2):
        scale = 1.0
    else:
        log_w = math.log2(screen_w / ref_w)
        log_h = math.log2(screen_h / ref_h)
        match_mode = int(getattr(canvas, "screen_match_mode", 0))
        if match_mode == 0:
            match = max(0.0, min(1.0, float(getattr(canvas, "match_width_or_height", 0.5))))
            scale = 2.0 ** (log_w * (1.0 - match) + log_h * match)
        elif match_mode == 1:
            scale = min(screen_w / ref_w, screen_h / ref_h)
        else:
            scale = max(screen_w / ref_w, screen_h / ref_h)
    if bool(getattr(canvas, "pixel_perfect", False)):
        scale = max(1.0, round(scale))
    scale_x = scale_y = max(scale, 1.0e-6)
    return scale_x, scale_y, min(scale_x, scale_y), screen_w / scale_x, screen_h / scale_y


class WorldUIElementTarget:
    """Pointer target for one Canvas-free UI element.

    Every world UI element owns an ordinary scene Transform. Width and height
    describe only the element's local geometry; they never establish a shared
    canvas boundary for descendants.
    """

    def __init__(self, element) -> None:
        self._element_ref = weakref.ref(element)
        self._size_revision = None
        self._logical_size = (1.0, 1.0)

    @property
    def element(self):
        element = self._element_ref()
        if element is None:
            raise RuntimeError("World UI input target outlived its scene component")
        return element

    @property
    def game_object(self):
        return self.element.game_object

    @property
    def enabled(self) -> bool:
        return bool(self.element.enabled)

    @property
    def input_logical_size(self) -> tuple[float, float]:
        revision = _get_layout_revision()
        if revision != self._size_revision:
            width, height = self.element.get_resolved_size()
            self._logical_size = max(1.0, width), max(1.0, height)
            self._size_revision = revision
        return self._logical_size

    def element_rect(self, element) -> tuple[float, float, float, float]:
        if element is not self.element:
            raise ValueError("A world UI input target only owns its element")
        width, height = self.input_logical_size
        return 0.0, 0.0, width, height

    @staticmethod
    def input_priority(position, _order: int) -> tuple[int, int, float]:
        distance = float(position[2]) if len(position) > 2 else float("inf")
        return 0, 0, -distance

    def raycast(self, canvas_x: float, canvas_y: float):
        if not (canvas_x == canvas_x and canvas_y == canvas_y):
            return None
        width, height = self.input_logical_size
        if not (0.0 <= canvas_x <= width and 0.0 <= canvas_y <= height):
            return None
        element = self.element
        element_object = element.game_object
        if element_object is None or not element_object.active_in_hierarchy:
            return None
        blocks = getattr(element, "effectively_blocks_raycast", None)
        if not (blocks() if callable(blocks) else element.raycast_target):
            return None
        if not element.enabled:
            return None
        return element


def _project_world_ui_targets(targets, ray_origin, ray_direction, layer_mask=0xffffffff):
    """Cross the native boundary once per ray, not once per element/property."""
    global _world_projection_targets, _world_projection_geometry
    from Infernux.lib import Vector3
    from Infernux.ui.ui_transform_dependencies import create_ui_transform_dependencies

    if targets != _world_projection_targets:
        _world_projection_geometry = create_ui_transform_dependencies([], [t.game_object for t in targets])
        _world_projection_targets = targets
    if not targets:
        return ()
    if not hasattr(ray_origin, "x"):
        ray_origin = Vector3(*map(float, ray_origin))
    if not hasattr(ray_direction, "x"):
        ray_direction = Vector3(*map(float, ray_direction))
    local = _world_projection_geometry.project_world_ray(
        ray_origin, ray_direction, int(layer_mask),
    )
    positions = []
    for target, (x, y, distance) in zip(targets, local):
        width, height = target.input_logical_size
        positions.append((x * WORLD_UI_PIXELS_PER_UNIT + width * 0.5,
                          -y * WORLD_UI_PIXELS_PER_UNIT + height * 0.5, distance))
    return tuple(positions)


def map_world_ui_ray(target, ray_origin, ray_direction):
    """Map one world ray using the same projection as batched runtime input."""
    position = _project_world_ui_targets((target,), ray_origin, ray_direction)[0]
    return position if position[0] == position[0] else None


def _world_targets(elements):
    targets = []
    for element in elements:
        target = _world_input_targets.get(element)
        if target is None:
            target = WorldUIElementTarget(element)
            _world_input_targets[element] = target
        targets.append(target)
    return tuple(targets)


def pick_world_ui_object_ids(scene, ray_origin, ray_direction, persistent_scene=None):
    """Return precise Canvas-free UI editor hits, nearest first.

    Scene selection follows visible authored geometry. Runtime pointer policy
    such as ``raycast_target`` and ``blocks_raycast`` must not make Text,
    Image, or decorative controls impossible to select in the editor.
    """
    from Infernux.ui import UIFrame

    hits = []
    targets = _world_targets(_collect_world_ui_elements(scene, persistent_scene))
    positions = _project_world_ui_targets(targets, ray_origin, ray_direction)
    for target, position in zip(targets, positions):
        element = target.element
        # UIFrame is a visual-neutral layout/grouping component.  Its authored
        # rectangle is useful to layout children, but there is no visible quad
        # for a Scene click to select; selecting it remains available through
        # the hierarchy.
        if isinstance(element, UIFrame):
            continue
        width, height = target.input_logical_size
        if not (0.0 <= position[0] <= width and 0.0 <= position[1] <= height):
            continue
        game_object = getattr(element, "game_object", None)
        if game_object is None or not game_object.active_in_hierarchy or not element.enabled:
            continue
        object_id = int(getattr(game_object, "id", 0) or 0)
        if object_id > 0:
            hits.append((float(position[2]), object_id))
    hits.sort(key=lambda item: item[0])
    return tuple(object_id for _distance, object_id in hits)


def collect_runtime_ui_input_surfaces(scene, persistent_scene=None):
    """Return world element targets followed by sorted screen/camera canvases."""
    global _input_world_elements, _input_canvases, _input_canvas_token, _input_surfaces
    elements = _collect_world_ui_elements(scene, persistent_scene)
    canvas_token = runtime_canvas_snapshot_token(scene, persistent_scene)
    if canvas_token != _input_canvas_token:
        canvases = tuple(collect_sorted_runtime_canvas_snapshot(scene, persistent_scene))
    else:
        canvases = _input_canvases or ()
    if (
        elements is not _input_world_elements
        or canvas_token != _input_canvas_token
        or canvases != _input_canvases
    ):
        _input_surfaces = _world_targets(elements) + canvases
        _input_world_elements, _input_canvases = elements, canvases
        _input_canvas_token = canvas_token
    return _input_surfaces


def map_runtime_ui_pointer(
    surfaces,
    camera,
    screen_x: float,
    screen_y: float,
    viewport_width: float,
    viewport_height: float,
    *,
    include_scene_hit: bool = False,
):
    """Map one viewport point into every runtime UI input surface.

    ``include_scene_hit`` lets the Game View share one native query between
    world-UI occlusion and ordinary GameObject mouse callbacks.  The default
    return remains the historical positions tuple for UI-only callers.
    """
    positions = []
    world_intersections = []
    ray_origin = ray_direction = None
    world_targets = tuple(s for s in surfaces if isinstance(s, WorldUIElementTarget))
    projected = ()
    if world_targets:
        if camera is not None:
            ray_origin, ray_direction = camera.screen_point_to_ray(
                float(screen_x), float(screen_y),
                float(viewport_width), float(viewport_height),
            )
            projected = _project_world_ui_targets(world_targets, ray_origin, ray_direction, camera.culling_mask)
    world_positions = iter(projected)

    for surface in surfaces:
        if isinstance(surface, WorldUIElementTarget):
            position = (float("nan"), float("nan"), float("inf"))
            if ray_origin is not None:
                position = next(world_positions)
                if position[0] == position[0]:
                    world_intersections.append((len(positions), position[2]))
            positions.append(position)
            continue

        scale_x, scale_y, _, logical_width, logical_height = _canvas_metrics(
            surface, viewport_width, viewport_height
        )
        set_input_logical_size = getattr(surface, "set_input_logical_size", None)
        if callable(set_input_logical_size):
            set_input_logical_size(logical_width, logical_height)
        positions.append(
            (
                float(screen_x) / max(scale_x, 1e-6),
                float(screen_y) / max(scale_y, 1e-6),
            )
        )

    scene_hit = None
    if world_intersections:
        from Infernux.physics import Physics

        furthest = max(distance for _, distance in world_intersections)
        if include_scene_hit:
            # One query serves both consumers: the closest trigger-inclusive
            # hit is the GameObject mouse target; the closest non-trigger hit
            # is the world-UI occluder. Keep the old 1000-unit mouse range
            # while allowing UI geometry farther away to retain occlusion.
            hits = list(Physics.raycast_all(
                ray_origin,
                ray_direction,
                max_distance=max(1000.0, furthest),
                layer_mask=int(camera.culling_mask),
                query_triggers=True,
            ) or ())
            hits.sort(key=lambda value: float(getattr(value, "distance", float("inf"))))
            scene_hit = next(
                (value for value in hits
                 if float(getattr(value, "distance", float("inf"))) <= 1000.0),
                None,
            )
            occluder = next(
                (value for value in hits
                 if not bool(getattr(getattr(value, "collider", None), "is_trigger", False))
                 and float(getattr(value, "distance", float("inf"))) < furthest),
                None,
            )
        else:
            occluder = Physics.raycast(
                ray_origin,
                ray_direction,
                max_distance=furthest,
                layer_mask=int(camera.culling_mask),
                query_triggers=False,
            )
        if occluder is not None:
            occluder_distance = float(occluder.distance)
            for index, distance in world_intersections:
                if occluder_distance + 1e-4 < distance:
                    positions[index] = (float("nan"), float("nan"), distance)

    result = tuple(positions)
    return (result, scene_hit) if include_scene_hit else result


def _collect_world_ui_elements(*scenes):
    """Share one hierarchy snapshot between UI rendering, input and picking."""
    global _world_elements_key, _world_elements
    from Infernux.ui import UICanvas
    from Infernux.ui.inx_ui_screen_component import InxUIScreenComponent

    key = tuple(
        (scene, int(getattr(scene, "world_id", 0)),
         int(getattr(scene, "structure_version", 0)),
         int(getattr(scene, "temporal_discontinuity_revision", 0)))
        for scene in scenes if scene is not None
    )
    if key == _world_elements_key:
        return _world_elements

    result = []

    def walk(game_object, canvas_ancestor: bool) -> None:
        components = tuple(game_object.get_py_components())
        canvas_here = canvas_ancestor or any(
            isinstance(component, UICanvas) for component in components
        )
        ui_component = next(
            (component for component in components if is_ui_screen_component(component)),
            None,
        )
        if ui_component is not None and not canvas_here:
            result.append(ui_component)
        for child in game_object.get_children():
            walk(child, canvas_here)

    seen = set()
    for scene in scenes:
        if scene is None or id(scene) in seen:
            continue
        seen.add(id(scene))
        for root_object in scene.get_root_objects():
            walk(root_object, False)
    _world_elements_key = key
    _world_elements = tuple(result)
    return _world_elements


class RuntimeScreenUISubmission:
    """Own the GPU Screen UI command snapshot for the active game target."""

    DEFAULT_CAPTURE_WIDTH = 1920
    DEFAULT_CAPTURE_HEIGHT = 1080

    def __init__(self, engine) -> None:
        self._engine_ref = weakref.ref(engine)
        self._target_width = self.DEFAULT_CAPTURE_WIDTH
        self._target_height = self.DEFAULT_CAPTURE_HEIGHT
        self._scene = None
        self._scene_structure_version = -1
        self._canvas_snapshot_token = None
        self._canvas_snapshot = ()
        self._last_submission_frame = -1
        self._command_packets = UICommandPackets()

    @property
    def target_size(self) -> tuple[int, int]:
        return self._target_width, self._target_height

    def set_target_size(self, width: int, height: int) -> None:
        width = int(width)
        height = int(height)
        if width < 1 or height < 1:
            return
        self._target_width = width
        self._target_height = height

    def submit(self) -> bool:
        """Publish current-frame UI commands before camera RenderGraph submission.

        Returns ``True`` when commands were rebuilt and ``False`` when the
        native renderer retained an identical cached command snapshot.
        """
        from Infernux.lib import SceneManager, ScreenUIList
        from Infernux.ui.enums import RenderMode

        engine = self._engine_ref()
        if engine is None:
            return False
        frame_token = int(getattr(engine, "_render_submission_frame", -1))
        if frame_token >= 0 and frame_token == self._last_submission_frame:
            return False
        renderer = engine.get_screen_ui_renderer()
        if renderer is None:
            return False

        width, height = self.target_size
        scene_manager = SceneManager.instance()
        scene = scene_manager.get_active_scene()
        persistent_scene = scene_manager.get_runtime_persistent_scene()
        if scene is None:
            self._scene = None
            self._scene_structure_version = -1
            self._canvas_snapshot_token = None
            self._canvas_snapshot = ()
            canvases = ()
        else:
            scene_identity = (scene, persistent_scene)
            structure_version = (
                int(getattr(scene, "structure_version", 0)),
                int(getattr(persistent_scene, "structure_version", 0)),
            )
            if scene_identity != self._scene or structure_version != self._scene_structure_version:
                clear_rect_cache((id(scene), id(persistent_scene), structure_version))
                self._scene = scene_identity
                self._scene_structure_version = structure_version
            canvas_token = runtime_canvas_snapshot_token(scene, persistent_scene)
            if canvas_token != self._canvas_snapshot_token:
                self._canvas_snapshot = tuple(
                    collect_sorted_runtime_canvas_snapshot(scene, persistent_scene)
                )
                self._canvas_snapshot_token = canvas_token
            canvases = self._canvas_snapshot
        world_elements = _collect_world_ui_elements(scene, persistent_scene)

        texture_cache = _get_tex_cache()
        revision = _runtime_ui_revision(
            scene, canvases, width, height, texture_cache.generation,
            world_elements, persistent_scene,
        )
        packets = self._command_packets
        if texture_cache.has_pending:
            packets.prepare(None)
        font_epoch = renderer.command_packet_epoch()
        packets.prepare((renderer, width, height, texture_cache.generation,
                         font_epoch, _runtime_command_epoch()), font_epoch=font_epoch)
        if texture_cache.has_pending:
            renderer.begin_frame(width, height)
        else:
            if renderer.begin_frame_cached(width, height, revision):
                self._last_submission_frame = frame_token
                return False

        if not canvases and not world_elements:
            self._last_submission_frame = frame_token
            return True

        get_texture_id = texture_cache.get_bound(engine)
        packets.submit_elements(
            world_elements, renderer, self._submit_world_element,
            get_texture_id, ScreenUIList,
        )
        for canvas in canvases:
            self._submit_canvas(
                canvas,
                renderer,
                get_texture_id,
                width,
                height,
                ScreenUIList,
                RenderMode,
                packets,
            )
        packets.flush(renderer)
        self._last_submission_frame = frame_token
        return True

    @staticmethod
    def _submit_world_element(element, renderer, get_texture_id, screen_ui_list) -> None:
        game_object = element.game_object
        if game_object is None or not game_object.active_in_hierarchy or not element.enabled:
            return
        begin_element = getattr(renderer, "begin_world_object", None)
        end_element = getattr(renderer, "end_world_element", None)
        world_list = getattr(screen_ui_list, "World", None)
        if not callable(begin_element) or not callable(end_element) or world_list is None:
            raise RuntimeError("This Player does not provide the world-space UI render capability")

        if callable(getattr(element, "resolve_text_layout", None)):
            _resolve_text_layout(element, renderer.measure_text, 1.0)
        logical_width, logical_height = (max(1.0, value) for value in element.get_resolved_size())
        begin_element(
            game_object,
            logical_width * 0.5,
            logical_height * 0.5,
        )
        try:
            _ui_dispatch(
                element,
                "runtime",
                renderer=renderer,
                ui_list=world_list,
                sx=0.0,
                sy=0.0,
                sw=logical_width,
                sh=logical_height,
                ref_w=logical_width,
                ref_h=logical_height,
                scale_x=1.0,
                scale_y=1.0,
                text_scale=1.0,
                get_tex_id=get_texture_id,
                world_transform_owned=True,
            )
        finally:
            end_element()

    @staticmethod
    def _submit_canvas(
        canvas,
        renderer,
        get_texture_id,
        game_width: int,
        game_height: int,
        screen_ui_list,
        render_mode,
        packets=None,
    ) -> None:
        canvas_object = getattr(canvas, "game_object", None)
        if canvas_object is not None and not canvas_object.active_in_hierarchy:
            return
        if not getattr(canvas, "enabled", True):
            return

        canvas_render_mode = getattr(canvas, "render_mode", render_mode.ScreenOverlay)
        if canvas_render_mode == render_mode.CameraOverlay:
            ui_list = screen_ui_list.Camera
        elif canvas_render_mode == render_mode.ScreenOverlay:
            ui_list = screen_ui_list.Overlay
        else:
            return

        if float(getattr(canvas, "reference_width", 1920)) < 1 or float(
            getattr(canvas, "reference_height", 1080)
        ) < 1:
            return

        scale_x, scale_y, text_scale, logical_width, logical_height = _canvas_metrics(
            canvas, game_width, game_height
        )

        elements = canvas_elements(canvas)
        args = (ui_list, logical_width, logical_height, scale_x, scale_y,
                text_scale, get_texture_id)
        if packets is not None:
            packets.submit_elements(
                elements, renderer, RuntimeScreenUISubmission._submit_screen_element,
                *args, scope=id(canvas), scale=text_scale,
            )
            return
        # Reduced Player profiles submit their platform's immediate UI API.
        for element in elements:
            if callable(getattr(element, "resolve_text_layout", None)):
                _resolve_text_layout(element, renderer.measure_text, text_scale)
        for element in elements:
            RuntimeScreenUISubmission._submit_screen_element(element, renderer, *args)

    @staticmethod
    def _submit_screen_element(element, renderer, ui_list, logical_width, logical_height,
                               scale_x, scale_y, text_scale, get_texture_id):
        element_object = getattr(element, "game_object", None)
        if element_object is not None and not element_object.active_in_hierarchy:
            return
        if not getattr(element, "enabled", True):
            return
        x, y, width, height = element.get_rect(logical_width, logical_height)
        begin_object = getattr(renderer, "begin_screen_object", None)
        if callable(begin_object):
            begin_object(element_object, ui_list, (x + width * 0.5) * scale_x,
                         (y + height * 0.5) * scale_y, scale_x, scale_y)
        clip = element.get_effective_clip_rect(logical_width, logical_height)
        if clip is not None:
            renderer.push_clip_rect(
                ui_list, clip[0] * scale_x, clip[1] * scale_y,
                clip[2] * scale_x, clip[3] * scale_y,
            )
        try:
            _ui_dispatch(
                element, "runtime", renderer=renderer, ui_list=ui_list,
                sx=x * scale_x, sy=y * scale_y, sw=width * scale_x, sh=height * scale_y,
                ref_w=logical_width, ref_h=logical_height, scale_x=scale_x,
                scale_y=scale_y, text_scale=text_scale, get_tex_id=get_texture_id,
            )
        finally:
            if clip is not None:
                renderer.pop_clip_rect(ui_list)
            if callable(begin_object):
                renderer.end_screen_object()


def __getattr__(name: str):
    """Keep the desktop pipeline wrapper lazy for reduced Player bindings.

    Cooked Web Players only need the platform-neutral command submission helpers.
    Importing the native render-pipeline base while loading those helpers would
    unnecessarily require the desktop callback binding in every Player profile.
    """
    if name == "RuntimeScreenUIRenderPipeline":
        from Infernux.engine.runtime_screen_ui_pipeline import (
            RuntimeScreenUIRenderPipeline,
        )

        return RuntimeScreenUIRenderPipeline
    raise AttributeError(name)
