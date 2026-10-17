"""Retained UI extraction must match a fresh submission after every mutation."""

from types import SimpleNamespace

import pytest


class Renderer:
    def __init__(self):
        self.key = None
        self.commands = []
        self.measures = 0
        self.capture = None
        self.font_epoch = 1
        self.captures = 0
        self.appends = 0

    def command_packet_epoch(self):
        return self.font_epoch

    def begin_command_packet(self):
        assert self.capture is None
        self.capture = []
        self.captures += 1

    def end_command_packet(self):
        packet = tuple(self.capture)
        self.capture = None
        return packet

    def abort_command_packet(self):
        self.capture = None
        self.key = None

    def append_command_packets(self, packets):
        self.appends += 1
        for packet in packets:
            self.commands.extend(self.resolve(command) for command in packet)

    @staticmethod
    def resolve(command):
        name, args, kwargs = command
        if name == "world_object":
            obj, pivot_x, pivot_y = args
            element = next(c for c in obj.get_py_components() if hasattr(c, "world_ui_matrix"))
            return "begin_world_element", (element.world_ui_matrix(), pivot_x, pivot_y, 1 << obj.layer), kwargs
        return command

    def begin_world_object(
        self, obj, pivot_x, pivot_y, always_on_top=False, billboard=False, constant_screen_size=False
    ):
        policies = {}
        if always_on_top:
            policies["always_on_top"] = True
        if billboard:
            policies["billboard"] = True
        if constant_screen_size:
            policies["constant_screen_size"] = True
        command = ("world_object", (obj, pivot_x, pivot_y), policies)
        if self.capture is None:
            self.commands.append(self.resolve(command))
        else:
            self.capture.append(command)

    def begin_frame_cached(self, width, height, revision):
        key = width, height, revision, self.font_epoch
        cached = key == self.key
        self.key = key
        return cached

    def begin_frame(self, *_):
        self.key = None

    def measure_text(self, text, size, *_):
        self.measures += 1
        return len(text) * size * 0.5, size

    def __getattr__(self, name):
        if name not in {
            "begin_world_element",
            "end_world_element",
            "push_clip_rect",
            "pop_clip_rect",
            "add_text",
            "add_image",
            "add_filled_rect",
        }:
            raise AttributeError(name)

        def emit(*args, **kwargs):
            output = self.commands if self.capture is None else self.capture
            output.append((name, args, kwargs))

        setattr(self, name, emit)
        return emit


def test_ui_material_generation_rebuilds_screen_and_world_packets(ui):
    from infernux.core.material import Material

    element = ui.add()
    revision = [3]
    native = SimpleNamespace(
        guid="ui-authored-guid",
        name="Authored UI",
        file_path="",
        _texture_assets_pending=False,
        get_version=lambda: revision[0],
        _get_render_texture=lambda _name: None,
        has_property=lambda _name: False,
        shader_name="Authored UI",
        vert_shader_name="ui-vertex-guid",
        frag_shader_name="ui-fragment-guid",
        get_render_state=lambda: SimpleNamespace(alpha_clip_enabled=False, alpha_clip_threshold=0.0),
    )
    material = Material(native)
    element.material = material
    assert element.material is material

    def set_binding(*args):
        output = ui.renderer.commands if ui.renderer.capture is None else ui.renderer.capture
        output.append(("set_material_binding", args, {}))

    ui.renderer.set_material_binding = set_binding
    rebuilt, commands = ui.frame()
    assert rebuilt
    assert any(
        command[0] == "set_material_binding" and command[1][1:3] == ("ui-authored-guid", 3) for command in commands
    )
    revision[0] = 4
    rebuilt, commands = ui.frame()
    assert rebuilt
    assert any(
        command[0] == "set_material_binding" and command[1][1:3] == ("ui-authored-guid", 4) for command in commands
    )


@pytest.fixture(params=[False, True], ids=["screen", "world"])
def ui(scene, monkeypatch, request):
    from infernux.ui import UICanvas, UIText
    import infernux.lib as lib
    import infernux.engine.runtime_screen_ui as module

    root = scene.create_game_object("Root")
    canvas = None
    if not request.param:
        canvas = UICanvas()
        root.add_py_component(canvas)
    renderer = Renderer()

    class Engine:
        _render_submission_frame = 0

        def get_screen_ui_renderer(self):
            return renderer

    engine = Engine()
    manager = SimpleNamespace(
        get_active_scene=lambda: scene,
        get_runtime_persistent_scene=lambda: None,
        scene_count=1,
        get_scene_at=lambda index: scene if index == 0 else None,
    )
    monkeypatch.setattr(lib, "SceneManager", SimpleNamespace(instance=lambda: manager))
    texture_cache = SimpleNamespace(has_pending=False, generation=0, texture_id=71)
    texture_cache.get_bound = lambda _: lambda source: texture_cache.texture_id + getattr(source, "revision", 0)
    monkeypatch.setattr(module, "_get_tex_cache", lambda: texture_cache)
    submission = module.RuntimeScreenUISubmission(engine)
    extracted = []
    dispatch = module._ui_dispatch

    def inspect_dispatch(element, *args, **kwargs):
        extracted.append(element)
        return dispatch(element, *args, **kwargs)

    monkeypatch.setattr(module, "_ui_dispatch", inspect_dispatch)

    def add(cls=UIText, parent=root):
        obj = scene.create_game_object(cls.__name__)
        obj.set_parent(parent)
        element = cls()
        obj.add_py_component(element)
        return element

    def frame():
        engine._render_submission_frame += 1
        renderer.commands.clear()
        extracted.clear()
        rebuilt = submission.submit()
        return rebuilt, list(renderer.commands)

    def matches_fresh():
        from infernux.ui.ui_command_packets import UICommandPackets

        rebuilt, actual = frame()
        assert rebuilt
        submission._command_packets = UICommandPackets()
        renderer.key = None
        _, expected = frame()
        assert actual == expected
        return actual

    return SimpleNamespace(
        root=root,
        canvas=canvas,
        world=request.param,
        scene=scene,
        add=add,
        frame=frame,
        matches_fresh=matches_fresh,
        renderer=renderer,
        extracted=extracted,
        submission=submission,
        textures=texture_cache,
    )


def test_text_edit_and_font_epoch_preserve_packet_locality(ui, monkeypatch):
    from infernux.ui import ui_command_packets as packets

    labels = [ui.add() for _ in range(40)]
    ui.frame()
    assert ui.renderer.captures == 40
    assert ui.renderer.appends == 1
    measures = ui.renderer.measures
    visits, classifications = [], []
    measure = packets.UICommandPackets.measure
    classify = packets._can_retain_runtime_commands

    def counted_measure(owner, element, *args):
        visits.append(element)
        return measure(owner, element, *args)

    def counted_classify(element):
        classifications.append(element)
        return classify(element)

    monkeypatch.setattr(packets.UICommandPackets, "measure", counted_measure)
    monkeypatch.setattr(packets, "_can_retain_runtime_commands", counted_classify)
    labels[17].color = [0.2, 0.5, 1.0, 1.0]
    assert ui.frame()[0]
    assert ui.extracted == [labels[17]]
    assert ui.renderer.measures == measures
    assert ui.renderer.captures == 41
    assert classifications == []

    visits.clear()
    labels[17].text = "Changed once"
    rebuilt, actual = ui.frame()
    assert rebuilt
    assert visits == [labels[17]]
    assert ui.extracted == [labels[17]]
    assert ui.renderer.measures == measures + 1
    assert ui.renderer.captures == 42
    assert ui.renderer.appends == 3
    assert classifications == []
    assert sum(name == "add_text" for name, _, _ in actual) == 40
    _assert_current_commands_match_fresh(ui, actual)
    assert ui.frame()[0] is False

    captures, appends = ui.renderer.captures, ui.renderer.appends
    ui.renderer.font_epoch += 1
    ui.frame()
    assert ui.renderer.captures == captures + 40
    assert ui.renderer.appends == appends + 1
    assert set(ui.extracted) == set(labels)


def test_retained_submission_tracks_hierarchy_viewport_and_visibility_edits(ui):
    from infernux.components import serialized_field
    from infernux.lib import Vector3
    from infernux.ui import UIGroup, UIImage
    from infernux.ui.ui_render_revision import get_runtime_ui_revision, mark_runtime_ui_dirty

    group = UIGroup()
    ui.root.add_py_component(group)
    label, other = ui.add(), ui.add()
    ui.frame()

    label.game_object.transform.local_position = Vector3(10, 20, 3)
    label.game_object.transform.local_euler_angles = Vector3(10, 20, 30)
    ui.matches_fresh()
    other.game_object.transform.local_position = Vector3(1, 2, 3)
    ui.root.transform.local_position = Vector3(20, 30, 40)
    ui.root.transform.local_euler_angles = Vector3(20, 30, 40)
    ui.root.transform.local_scale = Vector3(2, 3, 4)
    ui.matches_fresh()

    label.game_object.set_parent(other.game_object)
    ui.matches_fresh()
    ui.submission.set_target_size(1024, 768)
    ui.matches_fresh()
    group.alpha = 0.25
    commands = ui.matches_fresh()
    assert all(args[9] == 0.25 for name, args, _ in commands if name == "add_text")

    label.mirror_x = True
    ui.matches_fresh()
    label.enabled = False
    commands = ui.matches_fresh()
    assert sum(name == "add_text" for name, _, _ in commands) == 1
    label.enabled = True
    ui.matches_fresh()
    label.game_object.active = False
    commands = ui.matches_fresh()
    assert sum(name == "add_text" for name, _, _ in commands) == 1
    label.game_object.active = True
    ui.matches_fresh()

    label.color[0] = 0.25
    mark_runtime_ui_dirty()
    ui.matches_fresh()
    assert ui.frame()[0] is False

    image = ui.add(UIImage)
    image.color = [1.0, 1.0, 1.0, 0.0]
    image.game_object.active = False
    ui.frame()
    revision = get_runtime_ui_revision()
    captures, appends = ui.renderer.captures, ui.renderer.appends
    image.color = [1, 1, 1, 0]
    assert get_runtime_ui_revision() == revision
    assert ui.frame() == (False, [])
    assert (ui.renderer.captures, ui.renderer.appends) == (captures, appends)

    image.game_object.active = True
    ui.frame()
    image.color = (0.25, 0.5, 1, 1)
    assert get_runtime_ui_revision() > revision
    assert ui.frame()[0] and ui.extracted == [image]
    color = image.color
    color[0] = 0.75
    image.color = color
    assert ui.frame()[0] and ui.extracted == [image]
    image.color[1] = 0.25
    mark_runtime_ui_dirty(image)
    assert ui.frame()[0] and ui.extracted == [image]

    class VectorImage(UIImage):
        offset: Vector3 = serialized_field(default=Vector3(1, 2, 3))

    vector_image = ui.add(VectorImage)
    ui.frame()
    revision = get_runtime_ui_revision()
    vector_image.offset = [1, 2, 3]
    assert get_runtime_ui_revision() == revision
    assert ui.frame() == (False, [])
    vector_image.offset = (4, 2, 3)
    assert ui.frame()[0] and ui.extracted == [vector_image]


@pytest.mark.parametrize("ui", [True], indirect=True, ids=["world"])
def test_two_submission_owners_observe_edits_independently(ui):
    from infernux.engine.runtime_screen_ui import RuntimeScreenUISubmission

    first, second = ui.add(), ui.add()
    ui.frame()
    renderer = Renderer()

    class OtherEngine:
        _render_submission_frame = 0

        def get_screen_ui_renderer(self):
            return renderer

    engine = OtherEngine()
    other = RuntimeScreenUISubmission(engine)
    other.submit()
    first.text = "Both owners need this edit"
    _, expected = ui.frame()
    # Another edit while this owner is not drawing must remain pending too.
    second.color = [0.2, 0.5, 0.8, 1.0]
    _, expected = ui.frame()
    engine._render_submission_frame += 1
    renderer.commands.clear()
    assert other.submit()
    assert renderer.commands == expected
    assert renderer.captures == 4  # Two initial packets, two changed packets.


@pytest.mark.parametrize("ui", [False], indirect=True, ids=["screen"])
def test_released_submission_groups_do_not_stay_on_components(ui):
    import gc
    import weakref

    label = ui.add()
    ui.frame()
    packets = ui.submission._command_packets
    groups = [weakref.ref(group) for group in packets.groups.values()]
    packets.prepare(("different target",))
    gc.collect()
    assert all(group() is None for group in groups)
    assert not label.__dict__["_ui_command_groups"]
    label.text = "No dead subscriber callbacks"


def test_parent_ui_pose_updates_descendants_not_unrelated_branches(ui):
    from infernux.lib import Vector3
    from infernux.ui import UIFrame

    parent = ui.add(UIFrame)
    child = ui.add(parent=parent.game_object)
    bridge = ui.scene.create_game_object("Non-UI hierarchy bridge")
    bridge.set_parent(parent.game_object)
    descendant = ui.add(parent=bridge)
    unrelated = ui.add()
    ui.frame()
    parent.game_object.transform.local_position = Vector3(30.0, 40.0, 2.0)
    rebuilt, actual = ui.frame()
    assert rebuilt
    assert set(ui.extracted) == (set() if ui.world else {parent, child, descendant})
    assert unrelated not in ui.extracted
    _assert_current_commands_match_fresh(ui, actual)
    if not ui.world:
        from infernux.ui import UICanvas
        from infernux.ui.inx_ui_screen_component import _get_layout_revision
        from infernux.ui.ui_render_dispatch import _runtime_command_epoch

        nested_canvas = ui.scene.create_game_object("Nested Canvas")
        nested_canvas.set_parent(parent.game_object)
        nested_canvas.add_component(UICanvas)
        nested = ui.add(parent=nested_canvas)
        ui.frame()
        revision, epoch = _get_layout_revision(), _runtime_command_epoch()
        for move in (
            lambda: parent.set_rect(200, 300, parent.width, parent.height, 1920, 1080),
            lambda: parent.set_visual_position(240, 320, 1920, 1080),
            lambda: parent.set_layout_rotation(30),
        ):
            move()
            _, actual = ui.frame()
            assert set(ui.extracted) == {parent, child, descendant}
            assert unrelated not in ui.extracted and nested not in ui.extracted
            assert _get_layout_revision() == revision
            assert _runtime_command_epoch() == epoch
            _assert_current_commands_match_fresh(ui, actual)
        captures = ui.renderer.captures
        child.set_rect(20, 30, child.width, child.height, 1920, 1080)
        ui.frame()
        assert ui.extracted == [child]
        assert ui.renderer.captures == captures + 1


@pytest.mark.parametrize("position", ["before", "after"])
def test_custom_draw_remeasures_successive_text_and_size_edits(ui, position):
    from infernux.ui import UIButton, UIText, UIFrame, UILayoutDirection, TextResizeMode
    from infernux.ui import ui_render_dispatch as dispatch

    parent = ui.add(UIFrame)
    parent.layout_direction = UILayoutDirection.Horizontal
    parent.width = 2000
    parent.clip_content = True
    if position == "before":
        target = ui.add(UIText, parent.game_object)
        custom = ui.add(UIButton, parent.game_object)
    else:
        custom = ui.add(UIButton, parent.game_object)
        target = ui.add(UIText, parent.game_object)
    tail = ui.add(UIText, parent.game_object)
    tail.text = "tail"
    target.text, target.font_size = "A", 18
    target.resize_mode = TextResizeMode.AutoWidth
    original = dispatch.get_ui_renderer("UIButton", "runtime")
    mutate = False
    change = "text"

    def draw(element, renderer, **kwargs):
        if mutate:
            if change == "position":
                parent.set_visual_position(200, 100, 1920, 1080)
            else:
                setattr(target, change, "TenLetters" if change == "text" else 36)

    def text_command(commands, content):
        return next(args for name, args, _ in commands if name == "add_text" and args[5] == content)

    try:
        dispatch.register_ui_renderer("UIButton", "runtime", draw)
        _, before = ui.frame()
        ui.frame()
        old_width = target.get_resolved_size()[0]
        tail_before = text_command(before, "tail")
        scale = (tail_before[3] - tail_before[1]) / tail.width
        mutate = True
        custom.label = "Publish mutation"
        _, after = ui.frame()
        expected_width = len(target.text) * target.font_size * 0.5
        assert target.get_resolved_size()[0] == pytest.approx(expected_width)
        tail_after = text_command(after, "tail")
        if not ui.world:
            assert tail_after[1] - tail_before[1] == pytest.approx((expected_width - old_width) * scale)
        if position == "after":
            command = text_command(after, target.text)
            assert command[3] - command[1] == pytest.approx(expected_width * scale)
        # An already-drawn element changes on the next publication; later
        # elements use the new layout immediately. Warm custom frames must
        # neither remeasure native glyphs nor rebuild unchanged packets.
        mutate = False
        _, settled = ui.frame()
        measures, captures = ui.renderer.measures, ui.renderer.captures
        rebuilt, warm = ui.frame()
        assert not rebuilt and warm == []
        assert ui.renderer.measures == measures
        assert ui.renderer.captures == captures
        command = text_command(settled, target.text)
        assert command[3] - command[1] == pytest.approx(expected_width * scale)

        old_width = target.get_resolved_size()[0]
        tail_before = text_command(settled, "tail")
        change = "font_size"
        mutate = True
        custom.label = "Publish font-size mutation"
        _, after = ui.frame()
        expected_width = len(target.text) * 36 * 0.5
        assert target.get_resolved_size()[0] == pytest.approx(expected_width)
        if not ui.world:
            assert text_command(after, "tail")[1] - tail_before[1] == pytest.approx(
                (expected_width - old_width) * scale
            )
        mutate = False
        _, settled = ui.frame()
        command = text_command(settled, target.text)
        assert command[3] - command[1] == pytest.approx(expected_width * scale)
        assert ui.frame() == (False, [])
        if not ui.world:
            tail_before = text_command(settled, "tail")
            shift = 200 - parent.get_visual_rect(1920, 1080)[0]
            change, mutate = "position", True
            custom.label = "Publish parent movement"
            _, after = ui.frame()
            assert text_command(after, "tail")[1] - tail_before[1] == pytest.approx(shift * scale)
            mutate = False
            _, settled = ui.frame()
            _assert_current_commands_match_fresh(ui, settled)
    finally:
        dispatch.register_ui_renderer("UIButton", "runtime", original)


def _assert_current_commands_match_fresh(ui, actual):
    from infernux.ui.ui_command_packets import UICommandPackets

    ui.submission._command_packets = UICommandPackets()
    ui.renderer.key = None
    assert ui.frame()[1] == actual


@pytest.mark.parametrize("ui", [False], indirect=True, ids=["screen"])
def test_material_owner_replacement_does_not_hide_other_reference_owners(ui, monkeypatch):
    from infernux.core.material import Material
    from infernux.lib import InxMaterial

    original = Material(InxMaterial("Previous owner", "Unlit"))
    replacement = Material(InxMaterial("Replacement owner", "Unlit"))
    replacement.set_color("baseColor", 0.1, 0.5, 0.9, 1.0)
    first, second = ui.add(), ui.add()
    first.material = original
    second.material = original
    ui.frame()
    reference = type(first).material.get_raw(first)
    reference.invalidate()
    # Resource publication can replace one resolved alias without a field edit.
    monkeypatch.setattr(reference, "_do_resolve", lambda: replacement)
    rebuilt, actual = ui.frame()
    assert rebuilt
    assert ui.extracted == [first]
    _assert_current_commands_match_fresh(ui, actual)


@pytest.mark.parametrize("ui", [True], indirect=True, ids=["world"])
def test_material_binding_document_restore_and_removal_match_fresh(ui, monkeypatch):
    from infernux.components.value_document import make_asset_ref
    from infernux.core.assets import AssetManager
    from infernux.core.material import Material
    from infernux.lib import InxMaterial

    material = Material(InxMaterial("Inspector material", "Unlit"))
    material.set_color("baseColor", 0.2, 0.4, 0.8, 1.0)
    monkeypatch.setattr(AssetManager, "load_by_guid", lambda *_args, **_kwargs: material)
    label = ui.add()
    ui.add()
    ui.frame()
    for reference in (make_asset_ref("Material", "1234567890abcdef1234567890abcdef"), None):
        label._deserialize_fields_document({"material": reference})
        _, actual = ui.frame()
        assert ui.extracted == [label]
        _assert_current_commands_match_fresh(ui, actual)


@pytest.mark.parametrize("ui", [False], indirect=True, ids=["screen"])
def test_renderer_replacement_releases_old_bound_calls(ui):
    label = ui.add()
    ui.frame()
    original = ui.submission._engine_ref().get_screen_ui_renderer
    replacement = Renderer()
    ui.submission._engine_ref().get_screen_ui_renderer = lambda: replacement
    label.text = "New renderer"
    ui.frame()
    assert ui.renderer.commands == []
    assert any(name == "add_text" for name, _, _ in replacement.commands)
    ui.submission._engine_ref().get_screen_ui_renderer = original


def test_destroyed_element_packets_do_not_survive_membership_change(ui):
    doomed = ui.add()
    ui.add()
    ui.frame()
    ui.scene.destroy_game_object(doomed.game_object)
    ui.scene.process_pending_destroys()
    commands = ui.matches_fresh()
    assert sum(name == "add_text" for name, _, _ in commands) == 1
    assert all(doomed not in group.indices for group in ui.submission._command_packets.groups.values())
    replacement = ui.add()
    replacement.text = "Replacement"
    commands = ui.matches_fresh()
    assert sum(name == "add_text" for name, _, _ in commands) == 2


@pytest.mark.parametrize("ui", [False], indirect=True, ids=["screen"])
def test_registered_renderer_replacement_invalidates_warm_packets(ui):
    from infernux.ui import ui_render_dispatch as dispatch

    label = ui.add()
    ui.frame()
    original = dispatch.get_ui_renderer("UIText", "runtime")
    calls = []

    def custom(element, renderer, **kwargs):
        calls.append(element)
        renderer.add_text(kwargs["ui_list"], 0, 0, 20, 20, "REPLACED")

    try:
        dispatch.register_ui_renderer("UIText", "runtime", custom)
        _, commands = ui.frame()
        assert calls == [label]
        assert [args[5] for name, args, _ in commands if name == "add_text"] == ["REPLACED"]
        label.color = [0.2, 0.5, 0.8, 1.0]
        ui.frame()
        assert calls == [label, label]
    finally:
        dispatch.register_ui_renderer("UIText", "runtime", original)
    _, commands = ui.frame()
    assert [args[5] for name, args, _ in commands if name == "add_text"] == [label.text]


def test_packet_capture_failure_does_not_publish_partial_geometry(ui, monkeypatch):
    import infernux.engine.runtime_screen_ui as runtime

    ui.add()
    original = runtime._ui_dispatch

    def fail(*_args, **_kwargs):
        raise RuntimeError("deliberate capture failure")

    monkeypatch.setattr(runtime, "_ui_dispatch", fail)
    with pytest.raises(RuntimeError, match="deliberate capture failure"):
        ui.frame()
    assert ui.renderer.capture is None
    assert not ui.submission._command_packets.groups
    monkeypatch.setattr(runtime, "_ui_dispatch", original)
    ui.matches_fresh()


@pytest.mark.parametrize("inactive,ui", [(False, True), (True, False)], indirect=["ui"])
def test_font_epoch_remeasures_intrinsic_size_without_field_edits(ui, monkeypatch, inactive):
    from infernux.ui import TextResizeMode

    label = ui.add()
    label.resize_mode = TextResizeMode.AutoWidth
    ui.frame()
    previous = label.get_resolved_size()
    if inactive:
        label.enabled = False
        ui.frame()
    monkeypatch.setattr(ui.renderer, "measure_text", lambda *_: (previous[0] + 40, previous[1] + 8))
    ui.renderer.font_epoch += 1
    ui.frame()
    if inactive:
        label.enabled = True
        ui.frame()
    assert label.get_resolved_size() == (previous[0] + 40, label.height)


def test_custom_renderer_preserves_order_between_retained_packets(ui, monkeypatch):
    from infernux.ui import UIButton
    from infernux.ui import ui_render_dispatch as dispatch

    first = ui.add(UIButton)
    first.label = "before"
    custom = ui.add()
    custom.text = "middle"
    last = ui.add(UIButton)
    last.label = "after"

    def draw(element, renderer, **kwargs):
        renderer.add_text(kwargs["ui_list"], 0, 0, 20, 20, element.text)

    monkeypatch.setitem(dispatch._RENDERERS, ("UIText", "runtime"), draw)
    dispatch._RESOLVED_RENDERERS.clear()
    _, commands = ui.frame()
    assert [args[5] for name, args, _ in commands if name == "add_text"] == ["before", "middle", "after"]
    assert ui.renderer.appends == 2
    first.color = [1, 0, 0, 1]
    commands = ui.matches_fresh()
    assert [args[5] for name, args, _ in commands if name == "add_text"] == ["before", "middle", "after"]
    dispatch._RESOLVED_RENDERERS.clear()
