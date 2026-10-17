from __future__ import annotations

from types import SimpleNamespace

import pytest
from infernux.ui import UICanvas


class _CanvasStub(SimpleNamespace):
    pixel_perfect = False
    pixel_adjust_rect = UICanvas.pixel_adjust_rect


def test_pending_image_waits_for_its_texture_without_a_colored_placeholder(scene):
    from infernux.core.asset_ref import TextureRef
    from infernux.ui import UIImage
    from infernux.ui.ui_render_dispatch import _runtime_render_image

    image = scene.create_game_object("HUD image").add_component(UIImage)
    image.texture = TextureRef(guid="a" * 32)
    draws = []
    renderer = SimpleNamespace(add_image=lambda *args: draws.append("image"),
                               add_filled_rect=lambda *args: draws.append("rect"))
    arguments = dict(renderer=renderer, ui_list=1, sx=0, sy=0, sw=20, sh=25, scale_x=1, scale_y=1)
    _runtime_render_image(image, get_tex_id=lambda _: 0, **arguments)
    assert not draws
    _runtime_render_image(image, get_tex_id=lambda _: 71, **arguments)
    assert draws == ["image"]
    image.texture = None
    _runtime_render_image(image, get_tex_id=lambda _: 0, **arguments)
    assert draws == ["image", "rect"]


class _Renderer:
    def __init__(self) -> None:
        self.cached_key = None
        self.begin_calls: list[tuple[int, int]] = []
        self.cached_calls: list[tuple[int, int, int]] = []
        self.clip_calls: list[tuple] = []
        self.world_calls: list[tuple] = []
        self.screen_calls: list[tuple] = []
        self.measure_calls: list[tuple] = []
        self.capturing = False

    def command_packet_epoch(self):
        return 1

    def begin_command_packet(self):
        self.capturing = True
        self.saved_calls = self.clip_calls, self.world_calls
        self.clip_calls, self.world_calls = [], []

    def end_command_packet(self):
        self.capturing = False
        result = self.clip_calls, self.world_calls
        self.clip_calls, self.world_calls = self.saved_calls
        return result

    def abort_command_packet(self):
        self.capturing = False
        self.clip_calls, self.world_calls = self.saved_calls

    def append_command_packets(self, packets):
        for clips, worlds in packets:
            self.clip_calls.extend(clips)
            self.world_calls.extend(self.resolve_world(call) for call in worlds)

    @staticmethod
    def resolve_world(call):
        if call[0] in ("object", "object_top"):
            _, obj, pivot_x, pivot_y = call
            element = next(c for c in obj.get_py_components() if hasattr(c, "world_ui_matrix"))
            result = ("begin", element.world_ui_matrix(), pivot_x, pivot_y, 1 << obj.layer)
            return result + (True,) if call[0] == "object_top" else result
        return call

    def begin_world_object(
        self, obj, pivot_x, pivot_y, always_on_top=False, billboard=False, constant_screen_size=False
    ):
        call = ("object_top" if always_on_top else "object", obj, pivot_x, pivot_y)
        self.world_calls.append(call if self.capturing else self.resolve_world(call))

    def begin_frame(self, width: int, height: int) -> None:
        self.begin_calls.append((width, height))
        self.cached_key = None

    def begin_frame_cached(self, width: int, height: int, revision: int) -> bool:
        key = (width, height, revision)
        self.cached_calls.append(key)
        if key == self.cached_key:
            return True
        self.cached_key = key
        return False

    def push_clip_rect(self, *values) -> None:
        self.clip_calls.append(("push", *values))

    def pop_clip_rect(self, *values) -> None:
        self.clip_calls.append(("pop", *values))

    def begin_world_element(self, *values) -> None:
        self.world_calls.append(("begin", *values))

    def begin_screen_object(self, *values) -> None:
        self.screen_calls.append(values)

    def end_screen_object(self) -> None:
        pass

    def end_world_element(self) -> None:
        self.world_calls.append(("end",))

    def measure_text(self, *values):
        self.measure_calls.append(values)
        return 246.0, 58.0


class _Engine:
    def __init__(self, renderer) -> None:
        self.renderer = renderer

    def get_screen_ui_renderer(self):
        return self.renderer


def _install_scene_manager(monkeypatch, scene, persistent_scene=None) -> None:
    import infernux.lib as lib

    monkeypatch.setattr(
        lib,
        "SceneManager",
        SimpleNamespace(
            instance=lambda: SimpleNamespace(
                get_active_scene=lambda: scene,
                get_runtime_persistent_scene=lambda: persistent_scene,
                scene_count=int(scene is not None),
                get_scene_at=lambda index: scene if index == 0 else None,
            )
        ),
    )


class _UIElementStub:
    """UI components have stable object identity, unlike SimpleNamespace."""

    def __init__(self, **fields):
        self.__dict__.update(fields)


def test_runtime_submission_publishes_latest_hud_without_a_game_panel(monkeypatch):
    import infernux.engine.runtime_screen_ui as module
    from infernux.engine.runtime_screen_ui import RuntimeScreenUISubmission
    from infernux.ui.enums import RenderMode

    renderer = _Renderer()
    engine = _Engine(renderer)
    engine._render_submission_frame = 1
    scene = SimpleNamespace(structure_version=4, get_root_objects=lambda: ())
    element = _UIElementStub(
        text="first",
        enabled=True,
        game_object=SimpleNamespace(active_in_hierarchy=True),
        get_rect=lambda *_args: (10.0, 5.0, 20.0, 8.0),
        get_effective_clip_rect=lambda *_args: (1.0, 2.0, 9.0, 10.0),
    )
    canvas = _CanvasStub(
        render_mode=RenderMode.ScreenOverlay,
        reference_width=100.0,
        reference_height=50.0,
        enabled=True,
        game_object=SimpleNamespace(active_in_hierarchy=True),
        compute_scale=lambda *_args: (2.0, 2.0, 2.0),
        compute_logical_size=lambda *_args: (100.0, 50.0),
        _get_elements=lambda: (element,),
    )
    texture_cache = SimpleNamespace(
        has_pending=False,
        generation=3,
        get_bound=lambda _engine: lambda _path: 71,
    )
    dispatches = []

    _install_scene_manager(monkeypatch, scene)
    monkeypatch.setattr(
        module,
        "collect_sorted_runtime_canvas_snapshot",
        lambda *_args, **_kwargs: [canvas],
    )
    monkeypatch.setattr(module, "clear_rect_cache", lambda *_args: None)
    monkeypatch.setattr(module, "_get_tex_cache", lambda: texture_cache)
    monkeypatch.setattr(
        module,
        "_runtime_ui_revision",
        lambda *_args: 100 if element.text == "first" else 101,
    )
    monkeypatch.setattr(
        module,
        "_ui_dispatch",
        lambda current, backend, **kwargs: dispatches.append((current.text, backend, kwargs)),
    )

    submission = RuntimeScreenUISubmission(engine)
    submission.set_target_size(200, 100)

    assert submission.submit() is True
    assert dispatches[0][0:2] == ("first", "runtime")
    assert dispatches[0][2]["sx"] == 20.0
    assert dispatches[0][2]["sy"] == 10.0
    assert renderer.clip_calls[0][0] == "push"
    assert renderer.clip_calls[0][2:] == (2.0, 4.0, 18.0, 20.0)
    assert renderer.clip_calls[1][0] == "pop"

    assert submission.submit() is False
    assert len(dispatches) == 1

    element.text = "latest"
    engine._render_submission_frame = 2
    assert submission.submit() is True
    assert [entry[0] for entry in dispatches] == ["first", "latest"]


def test_runtime_submission_anchors_against_live_logical_canvas(monkeypatch, scene):
    import infernux.engine.runtime_screen_ui as module
    from infernux.engine.runtime_screen_ui import RuntimeScreenUISubmission
    from infernux.ui.enums import RenderMode

    native_scene = scene
    renderer = _Renderer()
    engine = _Engine(renderer)
    engine._render_submission_frame = 1
    scene = SimpleNamespace(structure_version=1, get_root_objects=lambda: ())
    logical_sizes = []
    element = _UIElementStub(
        enabled=True,
        game_object=SimpleNamespace(active_in_hierarchy=True),
        get_rect=lambda width, height: logical_sizes.append((width, height)) or (32.0, 32.0, 520.0, 52.0),
        get_effective_clip_rect=lambda *_args: None,
    )
    canvas = _CanvasStub(
        render_mode=RenderMode.ScreenOverlay,
        reference_width=1920.0,
        reference_height=1080.0,
        enabled=True,
        game_object=SimpleNamespace(active_in_hierarchy=True),
        compute_scale=lambda *_args: (1.5, 1.5, 1.5),
        compute_logical_size=lambda *_args: (3200.0 / 1.5, 1440.0 / 1.5),
        _get_elements=lambda: (element,),
    )
    texture_cache = SimpleNamespace(
        has_pending=False,
        generation=1,
        get_bound=lambda _engine: lambda _path: 0,
    )
    dispatches = []
    _install_scene_manager(monkeypatch, scene)
    monkeypatch.setattr(
        module,
        "collect_sorted_runtime_canvas_snapshot",
        lambda *_args, **_kwargs: [canvas],
    )
    monkeypatch.setattr(module, "clear_rect_cache", lambda *_args: None)
    monkeypatch.setattr(module, "_get_tex_cache", lambda: texture_cache)
    monkeypatch.setattr(module, "_runtime_ui_revision", lambda *_args: 1)
    monkeypatch.setattr(
        module,
        "_ui_dispatch",
        lambda _element, _backend, **kwargs: dispatches.append(kwargs),
    )

    submission = RuntimeScreenUISubmission(engine)
    submission.set_target_size(3200, 1440)
    assert submission.submit() is True
    assert logical_sizes == [(3200.0 / 1.5, 1440.0 / 1.5)]
    assert dispatches[0]["sx"] == 48.0
    assert dispatches[0]["sy"] == 48.0
    assert dispatches[0]["ref_w"] == 3200.0 / 1.5
    assert dispatches[0]["ref_h"] == 1440.0 / 1.5

    # Pixel Perfect adjusts geometry, not the fractional CanvasScaler ratio.
    canvas = UICanvas()
    canvas.reference_width, canvas.reference_height = 1280, 720
    canvas.pixel_perfect = True
    native_scene.create_game_object("Fractional-scale Canvas").add_py_component(canvas)
    canvas._get_elements = lambda: (element,)
    rect = (32.2, 32.4, 520.3, 52.5)
    element.get_rect = lambda *_: rect
    element.get_effective_clip_rect = lambda *_: (0.2, 0.4, 1279.9, 719.6)
    for target, scale in (((1920, 1080), 1.5), ((960, 540), 0.75)):
        assert canvas.compute_scale(*target) == pytest.approx((scale, scale, scale))
        assert canvas.compute_logical_size(*target) == pytest.approx((1280, 720))
        submission = RuntimeScreenUISubmission(engine)
        submission.set_target_size(*target)
        assert submission.submit()
        draw = dispatches[-1]
        raster = canvas.pixel_adjust_rect(rect, scale, scale)
        assert (draw["sx"], draw["sy"], draw["sw"], draw["sh"]) == pytest.approx(raster)
        assert all(value == round(value) for value in raster)
        assert renderer.screen_calls[-1][2:4] == pytest.approx(
            (raster[0] + raster[2] * 0.5, raster[1] + raster[3] * 0.5),
        )
        assert renderer.clip_calls[-2][2:] == tuple(round(v * scale) for v in (0.2, 0.4, 1279.9, 719.6))
        assert element.get_rect(1280, 720) == rect
        assert abs(raster[0] / scale - rect[0]) <= 0.5 / scale
        assert abs(raster[1] / scale - rect[1]) <= 0.5 / scale


def test_runtime_submission_resolves_text_before_geometry_and_keeps_authored_size(monkeypatch, scene):
    import infernux.engine.runtime_screen_ui as module
    from infernux.engine.runtime_screen_ui import RuntimeScreenUISubmission
    from infernux.ui import UICanvas, UIText, TextResizeMode
    from infernux.ui.enums import RenderMode

    renderer = _Renderer()
    engine = _Engine(renderer)
    engine._render_submission_frame = 1
    runtime_scene = SimpleNamespace(structure_version=1, get_root_objects=lambda: ())
    element = UIText()
    owner = scene.create_game_object("Auto text Canvas")
    owner.add_py_component(UICanvas())
    text_owner = scene.create_game_object("Auto text")
    text_owner.set_parent(owner)
    text_owner.add_py_component(element)
    element.width = 17.0
    element.height = 29.0
    element.resize_mode = TextResizeMode.AutoWidth
    canvas = _CanvasStub(
        render_mode=RenderMode.ScreenOverlay,
        reference_width=100.0,
        reference_height=50.0,
        enabled=True,
        game_object=SimpleNamespace(active_in_hierarchy=True),
        compute_scale=lambda *_args: (2.0, 2.0, 2.0),
        compute_logical_size=lambda *_args: (100.0, 50.0),
        _get_elements=lambda: (element,),
    )
    texture_cache = SimpleNamespace(
        has_pending=False,
        generation=1,
        get_bound=lambda _engine: lambda _path: 0,
    )
    dispatches = []
    _install_scene_manager(monkeypatch, runtime_scene)
    monkeypatch.setattr(module, "collect_sorted_runtime_canvas_snapshot", lambda *_args: [canvas])
    monkeypatch.setattr(module, "_get_tex_cache", lambda: texture_cache)
    monkeypatch.setattr(module, "_runtime_ui_revision", lambda *_args: 1)
    monkeypatch.setattr(
        module,
        "_ui_dispatch",
        lambda _element, _backend, **kwargs: dispatches.append(kwargs),
    )

    submission = RuntimeScreenUISubmission(engine)
    submission.set_target_size(200, 100)
    assert submission.submit() is True

    assert renderer.measure_calls[0][1:3] == (36.0, 0.0)
    assert dispatches[0]["sw"] == pytest.approx(246.0)
    assert dispatches[0]["sh"] == pytest.approx(58.0)
    assert (element.width, element.height) == (17.0, 29.0)


def test_world_auto_text_uses_one_size_for_draw_input_and_rect_tool(monkeypatch, scene):
    import infernux.engine.runtime_screen_ui as module
    from infernux.engine.ui.ui_rect_manipulation import resolve_world_ui_frame
    from infernux.lib import ScreenUIList
    from infernux.ui import UIText, TextResizeMode

    text = UIText()
    text.width, text.height = 17.0, 29.0
    text.resize_mode = TextResizeMode.AutoWidth
    scene.create_game_object("World Auto Text").add_py_component(text)
    renderer = _Renderer()
    draws = []
    monkeypatch.setattr(module, "_ui_dispatch", lambda *args, **kwargs: draws.append(kwargs))
    module.RuntimeScreenUISubmission._submit_world_element(text, renderer, lambda _: 0, ScreenUIList)
    assert text.get_resolved_size() == pytest.approx((246.0, 29.0))
    assert (draws[0]["sw"], draws[0]["sh"]) == pytest.approx((246.0, 29.0))
    assert renderer.world_calls[0][2:4] == pytest.approx((123.0, 14.5))
    assert module.WorldUIElementTarget(text).input_logical_size == pytest.approx((246.0, 29.0))
    assert resolve_world_ui_frame(text)["half_size"] == pytest.approx((1.23, 0.145))
    assert (text.width, text.height) == (17.0, 29.0)


def test_canvas_free_ui_submits_each_element_as_transform_owned_world_geometry(monkeypatch, scene):
    import infernux.engine.runtime_screen_ui as module
    from infernux.engine.runtime_screen_ui import RuntimeScreenUISubmission
    from infernux.lib import ScreenUIList, Vector3
    from infernux.ui import UIFrame, UIImage

    root_object = scene.create_game_object("World panel")
    root_object.transform.position = Vector3(2.0, 3.0, 4.0)
    root = UIFrame()
    root.width = 200.0
    root.height = 100.0
    root.clip_content = True
    root_object.add_py_component(root)
    authored_transform = root_object.transform.serialize_document()
    authored_layout = (root.width, root.height)

    child_object = scene.create_game_object("World image")
    child_object.set_parent(root_object)
    child_object.layer = 30
    child = UIImage()
    child.width = 40.0
    child.height = 30.0
    child_object.add_py_component(child)
    child_object.transform.local_position = Vector3(-0.55, 0.25, 0.0)
    assert tuple(child_object.transform.local_position) == pytest.approx((-0.55, 0.25, 0.0))
    assert child.get_rect(200.0, 100.0) == pytest.approx((0.0, 0.0, 40.0, 30.0))

    child_object.transform.local_position = Vector3(-0.5, 0.2, 0.0)
    assert child.get_rect(200.0, 100.0) == pytest.approx((0.0, 0.0, 40.0, 30.0))
    child_object.transform.local_position = Vector3(-0.55, 0.25, 0.0)

    renderer = _Renderer()
    engine = _Engine(renderer)
    engine._render_submission_frame = 1
    texture_cache = SimpleNamespace(
        has_pending=False,
        generation=1,
        get_bound=lambda _engine: lambda _path: 0,
    )
    dispatches = []
    _install_scene_manager(monkeypatch, scene)
    monkeypatch.setattr(
        module,
        "collect_sorted_runtime_canvas_snapshot",
        lambda *_args, **_kwargs: (),
    )
    monkeypatch.setattr(module, "_get_tex_cache", lambda: texture_cache)
    monkeypatch.setattr(
        module,
        "_ui_dispatch",
        lambda element, backend, **kwargs: dispatches.append((element, backend, kwargs)),
    )

    submission = RuntimeScreenUISubmission(engine)
    assert submission.submit() is True

    assert renderer.cached_calls[0][:2] == (1920, 1080)
    assert [call[0] for call in renderer.world_calls] == ["begin", "end", "begin", "end"]
    assert len(renderer.world_calls[0][1]) == 16
    assert renderer.world_calls[0][2:] == (100.0, 50.0, 1)
    assert len(renderer.world_calls[2][1]) == 16
    assert renderer.world_calls[2][2:] == (20.0, 15.0, 1 << 30)
    assert [entry[0] for entry in dispatches] == [root, child]
    assert all(entry[1] == "runtime" for entry in dispatches)
    assert all(entry[2]["ui_list"] == ScreenUIList.World for entry in dispatches)
    assert all(entry[2]["world_transform_owned"] is True for entry in dispatches)
    assert dispatches[1][2]["sx"] == 0.0
    assert dispatches[1][2]["sy"] == 0.0
    assert renderer.clip_calls == []
    assert root_object.transform.serialize_document() == authored_transform
    assert (root.width, root.height) == authored_layout

    engine._render_submission_frame += 1
    assert submission.submit() is False  # Unchanged world UI reuses native commands.
    assert len(dispatches) == 2
    root_object.transform.position = Vector3(5.0, 3.0, 4.0)
    engine._render_submission_frame += 1
    assert submission.submit() is True  # Parent pose changes the child's world geometry too.
    assert renderer.world_calls[-2][1][12] == pytest.approx(4.45)
    child.enabled = False
    engine._render_submission_frame += 1
    assert submission.submit() is True
    assert dispatches[-1][0] is root


def test_world_ui_snapshot_tracks_membership_and_canvas_reparenting(scene):
    from infernux.engine.runtime_screen_ui import _collect_world_ui_elements
    from infernux.ui import UIText, UICanvas

    obj = scene.create_game_object("World text")
    text = UIText()
    obj.add_py_component(text)
    first = _collect_world_ui_elements(scene)
    assert first == (text,)
    assert _collect_world_ui_elements(scene) is first
    canvas_object = scene.create_game_object("Screen canvas")
    canvas_object.add_py_component(UICanvas())
    obj.set_parent(canvas_object)
    assert _collect_world_ui_elements(scene) == ()
    obj.set_parent(None)
    assert _collect_world_ui_elements(scene) == (text,)
    other = UIText()
    scene.create_game_object("Other text").add_py_component(other)
    assert _collect_world_ui_elements(scene) == (text, other)
    assert _collect_world_ui_elements(None) == ()


def test_world_ui_snapshot_uses_world_identity_when_native_scene_address_is_reused():
    from infernux.engine.runtime_screen_ui import _collect_world_ui_elements
    from infernux.ui.ui_render_dispatch import runtime_ui_revision
    from infernux.ui import UIText, UIImage

    old = UIText()
    new = UIImage()
    root = SimpleNamespace(get_py_components=lambda: (old,), get_children=lambda: ())
    # pybind may return the same wrapper when a native address is reused.
    # Equal structure counters cannot identify a different loaded World.
    proxy = SimpleNamespace(
        world_id=41, structure_version=3, temporal_discontinuity_revision=0, get_root_objects=lambda: (root,)
    )
    assert _collect_world_ui_elements(proxy) == (old,)
    old_revision = runtime_ui_revision(proxy, (), 1920, 1080)
    proxy.world_id = 42
    root.get_py_components = lambda: (new,)
    assert _collect_world_ui_elements(proxy) == (new,)
    assert runtime_ui_revision(proxy, (), 1920, 1080) != old_revision


def test_world_ui_revision_tracks_group_material_visibility_and_persistent_scene(scene, monkeypatch):
    from infernux.ui import UIText, UIGroup
    from infernux.ui.ui_render_dispatch import runtime_ui_revision
    import infernux.ui.ui_render_dispatch as dispatch
    from infernux.lib import Vector3

    parent = scene.create_game_object("Group")
    group = UIGroup()
    parent.add_py_component(group)
    obj = scene.create_game_object("Text")
    obj.set_parent(parent)
    text = UIText()
    obj.add_py_component(text)
    persistent = SimpleNamespace(structure_version=1)
    from infernux.core.material import Material
    from infernux.lib import InxMaterial

    text.material = Material(InxMaterial("Tracked UI material", "Unlit"))
    version = [1]
    monkeypatch.setattr(dispatch, "material_visual_revision", lambda *_args: ("material", version[0]))

    def revision():
        return runtime_ui_revision(scene, (), 1920, 1080, 1, (text,), persistent)

    last = revision()
    assert revision() == last
    for mutate in (
        lambda: setattr(group, "alpha", 0.25),
        lambda: setattr(parent.transform, "position", Vector3(1, 2, 3)),
        lambda: setattr(parent.transform, "euler_angles", Vector3(0, 20, 0)),
        lambda: setattr(obj, "layer", 3),
        lambda: setattr(text, "enabled", False),
        lambda: setattr(persistent, "structure_version", 2),
        lambda: version.__setitem__(0, 2),
    ):
        mutate()
        current = revision()
        assert current != last
        assert revision() == current
        last = current


def test_world_ui_pointer_uses_the_element_coordinate_contract(scene, monkeypatch):
    from infernux.engine.runtime_screen_ui import (
        WorldUIElementTarget,
        map_runtime_ui_pointer,
    )
    from infernux.physics import Physics

    from infernux.lib import Vector3 as vector
    from infernux.ui import UIText

    obj = scene.create_game_object("World text")
    obj.layer = 30
    root = UIText()
    obj.add_py_component(root)
    root.width = 200.0
    root.height = 100.0
    ray_arguments = []
    camera = SimpleNamespace(
        culling_mask=1 << 30,
        screen_point_to_ray=lambda *args: (
            ray_arguments.append(args)
            or (
                vector(0.25, -0.1, 2.0),
                vector(0.0, 0.0, -1.0),
            )
        ),
    )
    monkeypatch.setattr(Physics, "raycast", lambda *_args, **_kwargs: None)

    position = map_runtime_ui_pointer((WorldUIElementTarget(root),), camera, 320.0, 180.0, 640.0, 360.0)[0]

    assert position == pytest.approx((125.0, 60.0, 2.0))
    assert ray_arguments == [(320.0, 180.0, 640.0, 360.0)]

    camera.culling_mask = 1
    hidden = map_runtime_ui_pointer((WorldUIElementTarget(root),), camera, 320.0, 180.0, 640.0, 360.0)[0]
    assert hidden[0] != hidden[0] and hidden[1] != hidden[1]


def test_world_ui_scene_pick_uses_each_elements_own_transform_and_z(scene):
    from infernux.engine.runtime_screen_ui import (
        WorldUIElementTarget,
        map_world_ui_ray,
        pick_world_ui_object_ids,
    )
    from infernux.lib import Vector3
    from infernux.ui import UIFrame, UIText

    panel_object = scene.create_game_object("World panel")
    panel = UIFrame()
    # The child deliberately sits far outside this tiny parent's rectangle:
    # world hierarchy is not a clipping canvas.
    panel.width = 1.0
    panel.height = 1.0
    panel_object.add_py_component(panel)
    panel_object.transform.position = Vector3(0.0, 0.0, 0.0)

    text_object = scene.create_game_object("Free world text")
    text_object.set_parent(panel_object)
    text = UIText()
    text.width = 120.0
    text.height = 40.0
    text.raycast_target = False
    text_object.add_py_component(text)
    text_object.transform.position = Vector3(2.0, 0.0, 1.0)

    assert text.world_ui_matrix()[12:15] == pytest.approx((2.0, 0.0, 1.0))
    assert (
        map_world_ui_ray(
            WorldUIElementTarget(panel),
            Vector3(2.0, 0.0, 5.0),
            Vector3(0.0, 0.0, -1.0),
        )[0]
        > panel.width
    )

    hits = pick_world_ui_object_ids(
        scene,
        Vector3(2.0, 0.0, 5.0),
        Vector3(0.0, 0.0, -1.0),
    )

    assert hits == (text_object.id,)


def test_world_ui_never_inherits_parent_frame_clipping(scene):
    from infernux.ui import UIFrame, UIText

    panel_object = scene.create_game_object("World panel")
    panel = UIFrame()
    panel.clip_content = True
    panel_object.add_py_component(panel)

    text_object = scene.create_game_object("Overflowing text")
    text_object.set_parent(panel_object)
    text = UIText()
    text_object.add_py_component(text)

    assert text.is_world_space() is True
    assert text.get_effective_clip_rect(1920.0, 1080.0) is None


def test_world_ui_pointer_is_rejected_behind_a_physics_occluder(scene, monkeypatch):
    from infernux.engine.runtime_screen_ui import (
        WorldUIElementTarget,
        map_runtime_ui_pointer,
    )
    from infernux.physics import Physics

    from infernux.lib import Vector3 as vector
    from infernux.ui import UIText

    obj = scene.create_game_object("Occluded world text")
    root = UIText()
    obj.add_py_component(root)
    root.width = 100.0
    root.height = 100.0
    camera = SimpleNamespace(
        culling_mask=0xFFFFFFFF,
        screen_point_to_ray=lambda *_args: (
            vector(0.0, 0.0, 3.0),
            vector(0.0, 0.0, -1.0),
        ),
    )
    monkeypatch.setattr(Physics, "raycast", lambda *_args, **_kwargs: SimpleNamespace(distance=1.0))

    position = map_runtime_ui_pointer((WorldUIElementTarget(root),), camera, 0.0, 0.0, 100.0, 100.0)[0]

    assert position[0] != position[0]
    assert position[1] != position[1]
    assert position[2] == pytest.approx(3.0)


def test_world_ui_input_target_identity_survives_frame_collection(monkeypatch):
    import infernux.engine.runtime_screen_ui as module

    class Root:
        pass

    root = Root()
    scene = SimpleNamespace()
    monkeypatch.setattr(module, "_collect_world_ui_elements", lambda *_args: (root,))
    monkeypatch.setattr(
        module,
        "collect_sorted_runtime_canvas_snapshot",
        lambda *_args: (),
    )

    first = module.collect_runtime_ui_input_surfaces(scene)
    second = module.collect_runtime_ui_input_surfaces(scene)

    assert first[0] is second[0]


def test_runtime_submission_clears_stale_commands_without_an_active_scene(monkeypatch):
    from infernux.engine.runtime_screen_ui import RuntimeScreenUISubmission

    renderer = _Renderer()
    engine = _Engine(renderer)
    _install_scene_manager(monkeypatch, None)
    submission = RuntimeScreenUISubmission(engine)

    assert submission.submit() is True
    assert renderer.cached_calls[0][0:2] == (1920, 1080)
    assert submission.submit() is False


def test_script_free_physics_barriers_do_not_open_an_unowned_native_frame(monkeypatch):
    from infernux.components._component_lifecycle import RuntimeExecutionScheduler
    from infernux.engine.engine import Engine
    from infernux.engine.runtime_change_journal import RuntimeFrameBarrier
    import infernux.compute as compute

    anchors = []
    monkeypatch.setattr(compute, "_poll_transform_bindings", lambda: anchors.append("pose"))
    engine = Engine.__new__(Engine)
    scheduler = RuntimeExecutionScheduler(name="script-free-barrier")
    engine._runtime_scheduler = scheduler
    engine._runtime_scene_manager = SimpleNamespace(get_fixed_time_step=lambda: 0.02)
    try:
        for _ in range(3):
            for barrier in (
                RuntimeFrameBarrier.TRANSFORM_TO_PHYSICS,
                RuntimeFrameBarrier.PHYSICS_SIMULATION,
                RuntimeFrameBarrier.PHYSICS_TO_TRANSFORM,
                RuntimeFrameBarrier.PENDING_DESTROY,
            ):
                assert engine._consume_runtime_frame_barrier(barrier) is None
            assert scheduler._native_frame is None
            assert not tuple(getattr(scheduler, "_active_frames", ()))
        assert anchors == ["pose"] * 3
        # A real native begin still enables both physics callback phases.
        scheduler.begin_native_frame()
        engine._consume_runtime_frame_barrier(RuntimeFrameBarrier.TRANSFORM_TO_PHYSICS)
        engine._consume_runtime_frame_barrier(RuntimeFrameBarrier.PHYSICS_TO_TRANSFORM)
        scheduler.end_native_frame()
        assert scheduler.profiler_snapshot()["native_phase_dispatches"] == 2
        assert scheduler._native_frame is None
    finally:
        scheduler.clear()
