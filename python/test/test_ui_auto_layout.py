"""Deterministic Figma-style UIFrame layout integration tests."""

import pytest

from Infernux.ui import (
    UIFrame,
    UIGroup,
    UIImage,
    UICanvas,
    UILayoutAlign,
    UILayoutDirection,
    UILayoutJustify,
    UILayoutPosition,
    UILayoutSizing,
    UIText,
    TextResizeMode,
)
from Infernux.ui.ui_render_dispatch import extract_common


def _child(scene, parent, name, width, height):
    game_object = scene.create_game_object(name)
    game_object.set_parent(parent)
    component = UIImage()
    component.width = width
    component.height = height
    game_object.add_py_component(component)
    return component


def _screen_root(scene, name, size=(1920, 1080)):
    """Screen layout requires a Canvas; unattached UI is world geometry."""
    canvas_object = scene.create_game_object(name + " Canvas")
    canvas = UICanvas()
    canvas.reference_width, canvas.reference_height = size
    canvas_object.add_py_component(canvas)
    root = scene.create_game_object(name)
    root.set_parent(canvas_object)
    return root


def test_horizontal_frame_resolves_fixed_fill_padding_and_gap(scene):
    frame_object = _screen_root(scene, "Horizontal Frame")
    frame = UIFrame()
    frame.x = 100
    frame.y = 50
    frame.width = 300
    frame.height = 100
    frame.layout_direction = UILayoutDirection.Horizontal
    frame.padding_left = frame.padding_right = 10
    frame.padding_top = frame.padding_bottom = 10
    frame.gap = 5
    frame_object.add_py_component(frame)

    fixed = _child(scene, frame_object, "Fixed", 50, 20)
    fill = _child(scene, frame_object, "Fill", 10, 10)
    fill.width_sizing = UILayoutSizing.Fill
    fill.height_sizing = UILayoutSizing.Fill

    assert fixed.get_rect(1920, 1080) == pytest.approx((110, 60, 50, 20))
    assert fill.get_rect(1920, 1080) == pytest.approx((165, 60, 225, 80))


def test_vertical_frame_distributes_fill_by_weight_and_centers_cross_axis(scene):
    frame_object = _screen_root(scene, "Vertical Frame")
    frame = UIFrame()
    frame.width = 200
    frame.height = 300
    frame.layout_direction = UILayoutDirection.Vertical
    frame.align_items = UILayoutAlign.Center
    frame.padding_top = frame.padding_bottom = 10
    frame.gap = 10
    frame_object.add_py_component(frame)

    first = _child(scene, frame_object, "First", 40, 10)
    second = _child(scene, frame_object, "Second", 60, 10)
    first.height_sizing = second.height_sizing = UILayoutSizing.Fill
    first.layout_weight = 1
    second.layout_weight = 3

    assert first.get_rect(1920, 1080) == pytest.approx((80, 10, 40, 67.5))
    assert second.get_rect(1920, 1080) == pytest.approx((70, 87.5, 60, 202.5))


def test_main_axis_justify_and_space_between(scene):
    frame_object = _screen_root(scene, "Justified Frame", (800, 600))
    frame = UIFrame()
    frame.width = 300
    frame.height = 60
    frame.layout_direction = UILayoutDirection.Horizontal
    frame.justify_content = UILayoutJustify.Center
    frame.gap = 10
    frame_object.add_py_component(frame)

    first = _child(scene, frame_object, "First", 50, 20)
    second = _child(scene, frame_object, "Second", 50, 20)

    assert first.get_rect(800, 600) == pytest.approx((95, 0, 50, 20))
    assert second.get_rect(800, 600) == pytest.approx((155, 0, 50, 20))

    frame.justify_content = UILayoutJustify.End
    assert first.get_rect(800, 600) == pytest.approx((190, 0, 50, 20))
    assert second.get_rect(800, 600) == pytest.approx((250, 0, 50, 20))

    frame.justify_content = UILayoutJustify.SpaceBetween
    assert first.get_rect(800, 600) == pytest.approx((0, 0, 50, 20))
    assert second.get_rect(800, 600) == pytest.approx((250, 0, 50, 20))


def test_fill_constraints_redistribute_remaining_main_axis_space(scene):
    frame_object = _screen_root(scene, "Constrained Fill Frame", (800, 600))
    frame = UIFrame()
    frame.width = 200
    frame.height = 60
    frame.layout_direction = UILayoutDirection.Horizontal
    frame_object.add_py_component(frame)

    first = _child(scene, frame_object, "First", 10, 40)
    second = _child(scene, frame_object, "Second", 10, 40)
    third = _child(scene, frame_object, "Third", 10, 40)
    first.width_sizing = second.width_sizing = third.width_sizing = UILayoutSizing.Fill
    first.max_width = 30
    second.max_width = 40

    assert first.get_rect(800, 600) == pytest.approx((0, 0, 30, 40))
    assert second.get_rect(800, 600) == pytest.approx((30, 0, 40, 40))
    assert third.get_rect(800, 600) == pytest.approx((70, 0, 130, 40))


def test_hug_frame_and_absolute_child_use_one_parent_rect(scene):
    frame_object = _screen_root(scene, "Hug Frame")
    frame = UIFrame()
    frame.x = 20
    frame.y = 30
    frame.width_sizing = UILayoutSizing.Hug
    frame.height_sizing = UILayoutSizing.Hug
    frame.layout_direction = UILayoutDirection.Horizontal
    frame.padding_left = 4
    frame.padding_right = 6
    frame.padding_top = frame.padding_bottom = 5
    frame.gap = 10
    frame_object.add_py_component(frame)

    first = _child(scene, frame_object, "First", 40, 20)
    second = _child(scene, frame_object, "Second", 60, 30)
    absolute = _child(scene, frame_object, "Absolute", 10, 12)
    absolute.layout_position = UILayoutPosition.Absolute
    absolute.x = 7
    absolute.y = 8

    assert frame.get_rect(1920, 1080) == pytest.approx((20, 30, 120, 40))
    assert first.get_rect(1920, 1080) == pytest.approx((24, 35, 40, 20))
    assert second.get_rect(1920, 1080) == pytest.approx((74, 35, 60, 30))
    assert absolute.get_rect(1920, 1080) == pytest.approx((27, 38, 10, 12))


def test_hug_fill_cycle_is_rejected_without_iterative_guessing(scene):
    frame_object = _screen_root(scene, "Invalid Frame")
    frame = UIFrame()
    frame.width_sizing = UILayoutSizing.Hug
    frame.layout_direction = UILayoutDirection.Horizontal
    frame_object.add_py_component(frame)
    child = _child(scene, frame_object, "Fill", 10, 10)
    child.width_sizing = UILayoutSizing.Fill

    with pytest.raises(ValueError, match="width-hugging.*fill-width"):
        frame.get_rect(1920, 1080)


def test_ui_group_multiplies_alpha_and_independently_gates_raycast(scene):
    group_object = scene.create_game_object("Group")
    outer = UIGroup()
    outer.alpha = 0.5
    group_object.add_py_component(outer)

    nested_object = scene.create_game_object("Nested")
    nested_object.set_parent(group_object)
    inner = UIGroup()
    inner.alpha = 0.4
    inner.interactable = False
    nested_object.add_py_component(inner)

    image = _child(scene, nested_object, "Image", 50, 50)
    image.opacity = 0.5

    assert image.get_effective_group_state() == pytest.approx((0.2, False, True))
    assert extract_common(image)["opacity"] == pytest.approx(0.1)
    assert image.is_effectively_interactable() is False
    assert image.effectively_blocks_raycast() is True

    inner.blocks_raycast = False
    assert image.effectively_blocks_raycast() is False
    from Infernux.engine.runtime_screen_ui import WorldUIElementTarget
    target = WorldUIElementTarget(image)
    assert target.raycast(1.0, 1.0) is None

    inner.blocks_raycast = True
    assert target.raycast(1.0, 1.0) is image


def test_nested_frame_clip_rects_intersect_in_canvas_space(scene):
    outer_object = _screen_root(scene, "Outer Clip")
    outer = UIFrame()
    outer.x = 10
    outer.y = 20
    outer.width = 100
    outer.height = 80
    outer.clip_content = True
    outer_object.add_py_component(outer)

    inner_object = scene.create_game_object("Inner Clip")
    inner_object.set_parent(outer_object)
    inner = UIFrame()
    inner.x = 50
    inner.y = 30
    inner.width = 100
    inner.height = 100
    inner.clip_content = True
    inner_object.add_py_component(inner)

    image = _child(scene, inner_object, "Clipped Image", 200, 200)

    assert image.get_effective_clip_rect(1920, 1080) == pytest.approx((60, 50, 110, 100))


def test_canvas_raycast_uses_live_layout_extent_and_editor_can_override_it(scene):
    canvas = UICanvas()
    root = scene.create_game_object('Canvas')
    root.add_py_component(canvas)
    element = _child(scene, root, 'Corner image', 100, 100)
    element._set_layout_rect_origin(1820., 980., 1920., 1080.)
    canvas.set_input_logical_size(2560.0, 1440.0)

    assert canvas.raycast(2200.0, 1200.0) is element
    assert canvas.raycast(1850.0, 1040.0) is None

    assert canvas.raycast(1850.0, 1040.0, 0.0, 1920.0, 1080.0) is element
    assert canvas.raycast(2200.0, 1200.0, 0.0, 1920.0, 1080.0) is None

    with pytest.raises(ValueError, match="must be provided together"):
        canvas.raycast(0.0, 0.0, layout_width=1920.0)


@pytest.mark.parametrize('world', [False, True])
def test_text_intrinsic_layout_is_derived_and_shared_with_hit_testing(scene, world):
    text = UIText()
    root = scene.create_game_object("Auto text") if world else _screen_root(scene, "Auto text")
    text.width = 17.0
    text.height = 29.0
    root.add_py_component(text)
    text.resize_mode = TextResizeMode.AutoWidth
    measurements = []

    def measure(content, font_size, wrap_width, font_path, line_height, letter_spacing):
        measurements.append(
            (content, font_size, wrap_width, font_path, line_height, letter_spacing)
        )
        return 246.0, 58.0

    assert text.resolve_text_layout(measure, scale=2.0)
    assert text.width == 17.0
    assert text.height == 29.0
    assert text.get_resolved_size() == pytest.approx((123.0, 29.0))
    assert text.get_rect(1920.0, 1080.0)[2:] == pytest.approx((123.0, 29.0))
    assert text.contains_point(122.0, 20.0, 1920.0, 1080.0)
    assert not text.contains_point(124.0, 20.0, 1920.0, 1080.0)
    assert measurements == [("New Text", 36.0, 0.0, "", 1.2, 0.0)]

    # A different preview scale produces the same logical box and never writes
    # the authored size back into the scene document.
    assert not text.resolve_text_layout(
        lambda *_args: (123.0, 29.0), scale=1.0
    )
    assert (text.width, text.height) == (17.0, 29.0)

    if world:
        from Infernux.engine.runtime_screen_ui import WorldUIElementTarget
        target = WorldUIElementTarget(text)
        assert target.raycast(122.0, 20.0) is text
        assert target.raycast(124.0, 20.0) is None
    else:
        canvas = text.get_canvas()
        assert canvas.raycast(122.0, 20.0, layout_width=1920.0, layout_height=1080.0) is text
        assert canvas.raycast(124.0, 20.0, layout_width=1920.0, layout_height=1080.0) is None


def test_auto_height_measurement_uses_authored_wrap_width():
    text = UIText()
    text.width = 240.0
    text.height = 12.0
    text.resize_mode = TextResizeMode.AutoHeight
    received = []

    def measure(*args):
        received.append(args)
        return 360.0, 144.0

    text.resolve_text_layout(measure, scale=1.5)

    assert received[0][2] == pytest.approx(360.0)
    assert text.get_resolved_size() == pytest.approx((240.0, 96.0))
    assert (text.width, text.height) == (240.0, 12.0)


def test_text_intrinsic_layout_forwards_explicit_font_chain():
    from Infernux.ui.inx_ui_screen_component import _get_layout_revision
    text = UIText()
    text.fallback_font_paths = ["Assets/Fonts/CJK.ttf", "Assets/Fonts/Emoji.ttf"]
    received = []

    def measure(*arguments):
        received.append(arguments)
        return 120.0, 24.0

    before = _get_layout_revision()
    # A fixed text box measures glyphs without moving sibling layout bounds.
    assert text.resolve_text_layout(measure) is False
    assert _get_layout_revision() == before
    assert received[0][-1] == ["Assets/Fonts/CJK.ttf", "Assets/Fonts/Emoji.ttf"]


@pytest.mark.parametrize('direction', [UILayoutDirection.Horizontal, UILayoutDirection.Vertical])
def test_flow_group_measures_each_child_once_per_arrangement(scene, monkeypatch, direction):
    from Infernux.ui.inx_ui_screen_component import clear_rect_cache

    owner = _screen_root(scene, 'Batch')
    frame = UIFrame()
    owner.add_py_component(frame)
    frame.width, frame.height = 10000., 10000.
    frame.layout_direction = direction
    children = [_child(scene, owner, str(index), 20., 20.) for index in range(100)]
    measured = []
    original = UIImage._layout_desired_size

    def measure(self):
        measured.append(self)
        return original(self)

    monkeypatch.setattr(UIImage, '_layout_desired_size', measure)
    clear_rect_cache(object())
    # A late child can be the first query; it must publish every sibling rect.
    rects = {child: child.get_rect(1920., 1080.) for child in reversed(children)}
    assert measured == children
    axis = 0 if direction == UILayoutDirection.Horizontal else 1
    assert rects[children[-1]][axis] - rects[children[0]][axis] == 1980.
    assert [child.get_rect(1920., 1080.) for child in children] == [rects[c] for c in children]
    assert measured == children

    frame.gap = 3.
    measured.clear()
    updated = [child.get_rect(1920., 1080.) for child in children]
    assert measured == children
    assert updated[-1][axis] - updated[0][axis] == 2277.


def test_nested_hug_measures_subtrees_without_sibling_rewalk(scene, monkeypatch):
    from Infernux.ui.inx_ui_screen_component import clear_rect_cache

    owner = _screen_root(scene, 'Nested batch')
    outer = UIFrame()
    owner.add_py_component(outer)
    outer.layout_direction = UILayoutDirection.Vertical
    outer.width_sizing = outer.height_sizing = UILayoutSizing.Hug
    leaves = []
    for group in range(8):
        obj = scene.create_game_object(f'Row {group}')
        obj.set_parent(owner)
        row = UIFrame()
        obj.add_py_component(row)
        row.layout_direction = UILayoutDirection.Horizontal
        row.width_sizing = row.height_sizing = UILayoutSizing.Hug
        leaves.extend(_child(scene, obj, f'Leaf {index}', 20., 30.) for index in range(8))
    calls = 0
    original = UIImage._layout_desired_size

    def measure(self):
        nonlocal calls
        calls += 1
        return original(self)

    monkeypatch.setattr(UIImage, '_layout_desired_size', measure)
    clear_rect_cache(object())
    actual = [child.get_rect(1920., 1080.) for child in leaves]
    assert calls == len(leaves) * 2  # one intrinsic measure and one arrangement
    assert outer.get_rect(1920., 1080.)[2:] == pytest.approx((160., 240.))
    assert actual[7][0] - actual[0][0] == 140.
    assert actual[-1][1] - actual[0][1] == 210.
    leaves[0].width = 50.
    assert outer.get_rect(1920., 1080.)[2:] == pytest.approx((190., 240.))
    assert leaves[1].get_rect(1920., 1080.)[0] - leaves[0].get_rect(1920., 1080.)[0] == 50.


@pytest.mark.parametrize('mutation', ['padding', 'direction', 'fill', 'absolute', 'intrinsic', 'viewport'])
def test_batched_flow_matches_fresh_layout_after_mutation(scene, mutation):
    from Infernux.ui.inx_ui_screen_component import clear_rect_cache

    owner = _screen_root(scene, 'Live batch')
    frame = UIFrame()
    owner.add_py_component(frame)
    frame.width, frame.height = 400., 200.
    frame.layout_direction = UILayoutDirection.Horizontal
    first = _child(scene, owner, 'First', 30., 40.)
    text_owner = scene.create_game_object('Intrinsic')
    text_owner.set_parent(owner)
    text = UIText()
    text_owner.add_py_component(text)
    text.resize_mode = TextResizeMode.AutoWidth
    text.resolve_text_layout(lambda *_: (50., 30.))
    last = _child(scene, owner, 'Last', 60., 50.)
    children = [first, text, last]
    viewport = (800., 600.)
    before = [child.get_rect(*viewport) for child in children]
    if mutation == 'padding':
        frame.padding_left = 17.
    elif mutation == 'direction':
        frame.layout_direction = UILayoutDirection.Vertical
        frame.align_items = UILayoutAlign.End
    elif mutation == 'fill':
        first.width_sizing = last.width_sizing = UILayoutSizing.Fill
        first.max_width = 80.
        last.layout_weight = 3.
    elif mutation == 'absolute':
        first.layout_position = UILayoutPosition.Absolute
        first.x = 101.
    elif mutation == 'intrinsic':
        text.text = 'new content'
        text.resolve_text_layout(lambda *_: (150., 30.))
    else:
        viewport = (1600., 900.)
    actual = [child.get_rect(*viewport) for child in children]
    clear_rect_cache(object())
    expected = [child.get_rect(*viewport) for child in children]
    assert actual == expected
    assert actual != before


def test_world_frame_does_not_arrange_or_clip_world_children(scene, monkeypatch):
    from Infernux.lib import Vector3

    root = scene.create_game_object('World frame')
    frame = UIFrame()
    root.add_py_component(frame)
    frame.width, frame.height = 1., 1.
    frame.layout_direction = UILayoutDirection.Horizontal
    frame.clip_content = True
    child = _child(scene, root, 'World leaf', 200., 100.)
    child.game_object.transform.local_position = Vector3(4., 2., 3.)

    def forbidden(*_):
        raise AssertionError('World geometry must not enter screen arrangement')

    monkeypatch.setattr(UIFrame, '_arrange_flow_children', forbidden)
    assert child.get_rect(1920., 1080.) == (0., 0., 200., 100.)
    assert child.get_effective_clip_rect(1920., 1080.) is None
    assert child.world_ui_matrix()[12:15] == [4., 2., 3.]


@pytest.mark.parametrize('mutation', ['create', 'destroy', 'reparent', 'native_parent_move'])
def test_flow_layout_observes_native_scene_publication(scene, mutation):
    from Infernux.lib import Vector3
    from Infernux.ui.inx_ui_screen_component import clear_rect_cache
    from Infernux.ui.ui_render_dispatch import runtime_ui_revision

    owner = _screen_root(scene, 'Published flow')
    frame = UIFrame()
    owner.add_py_component(frame)
    frame.layout_direction = UILayoutDirection.Horizontal
    frame.width_sizing = UILayoutSizing.Hug
    first = _child(scene, owner, 'First', 40., 20.)
    second = _child(scene, owner, 'Second', 60., 20.)
    outside = scene.create_game_object('Outside')
    other = _child(scene, outside, 'Other', 30., 20.)
    canvas = frame.get_canvas()

    def publish_and_read():
        runtime_ui_revision(scene, [canvas], 1920, 1080)
        return frame.get_rect(1920., 1080.), second.get_rect(1920., 1080.)

    before = publish_and_read()
    if mutation == 'create':
        _child(scene, owner, 'Added', 90., 20.)
    elif mutation == 'destroy':
        scene.destroy_game_object(first.game_object)
        scene.process_pending_destroys()
    elif mutation == 'reparent':
        other.game_object.set_parent(owner)
    else:
        owner.transform.local_position += Vector3(30., 50., 0.)
    actual = publish_and_read()
    clear_rect_cache(object())
    assert actual == publish_and_read()
    assert actual != before


@pytest.mark.parametrize('world', [False, True])
def test_one_rect_query_resolves_owner_once_and_tracks_live_hierarchy(scene, monkeypatch, world):
    from Infernux.components._component_native import ComponentNativeMixin
    from Infernux.lib import Vector3
    from Infernux.ui.inx_ui_screen_component import clear_rect_cache

    screen_root = _screen_root(scene, 'Owner query')
    owner = scene.create_game_object('Text')
    if not world:
        owner.set_parent(screen_root)
    text = UIText()
    owner.add_py_component(text)
    text.width, text.height = 100., 40.
    owner.transform.local_position = Vector3(12., 34., 5.)
    clear_rect_cache(object())
    text.get_rect(1920., 1080.)
    calls = []
    original = ComponentNativeMixin._try_get_game_object

    def resolve(self):
        calls.append(self)
        return original(self)

    monkeypatch.setattr(ComponentNativeMixin, '_try_get_game_object', resolve)
    before = text.get_rect(1920., 1080.)
    assert calls == [text]
    assert text.is_world_space() is world
    if world:
        assert before == (0., 0., 100., 40.)
        owner.set_parent(screen_root)
    else:
        owner.set_parent(None)
    clear_rect_cache(object())
    assert text.is_world_space() is not world
    after = text.get_rect(1920., 1080.)
    assert before != after
    scene.destroy_game_object(owner)
    scene.process_pending_destroys()
    assert text._try_get_game_object() is None
