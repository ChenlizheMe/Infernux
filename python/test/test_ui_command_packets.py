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
        if name == 'world_object':
            obj, pivot_x, pivot_y = args
            element = next(c for c in obj.get_py_components() if hasattr(c, 'world_ui_matrix'))
            return 'begin_world_element', (element.world_ui_matrix(), pivot_x, pivot_y, 1 << obj.layer), kwargs
        return command

    def begin_world_object(self, obj, pivot_x, pivot_y, always_on_top=False,
                           billboard=False, constant_screen_size=False):
        policies = {}
        if always_on_top:
            policies['always_on_top'] = True
        if billboard:
            policies['billboard'] = True
        if constant_screen_size:
            policies['constant_screen_size'] = True
        command = ('world_object', (obj, pivot_x, pivot_y), policies)
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
        return len(text) * size * .5, size

    def __getattr__(self, name):
        if name not in {'begin_world_element', 'end_world_element', 'push_clip_rect',
                        'pop_clip_rect', 'add_text', 'add_image', 'add_filled_rect'}:
            raise AttributeError(name)
        def emit(*args, **kwargs):
            output = self.commands if self.capture is None else self.capture
            output.append((name, args, kwargs))
        setattr(self, name, emit)
        return emit


def test_runtime_ui_material_contract_uses_guid_generation_and_pipeline_key():
    from Infernux.ui.ui_render_dispatch import _bind_runtime_material

    calls = []
    native = SimpleNamespace(
        guid="ui-material-guid",
        get_version=lambda: 9,
        shader_name="UiSurface",
        vert_shader_name="Ui.vert",
        frag_shader_name="Ui.frag",
        get_render_state=lambda: SimpleNamespace(
            blend_enable=True,
            src_color_blend_factor=6,
            dst_color_blend_factor=7,
            color_blend_op=0,
            depth_test_enable=False,
            depth_write_enable=False,
            depth_compare_op=7,
            alpha_clip_enabled=False,
        ),
    )
    renderer = SimpleNamespace(set_material_binding=lambda *args: calls.append(args))
    _bind_runtime_material(renderer, "Overlay", {"_native": native})
    assert calls == [
        (
            "Overlay",
            "ui-material-guid",
            9,
            "ui|shader=UiSurface:Ui.vert:Ui.frag|state=1,6,7,0,0,0,7,0",
            (1.0, 1.0, 1.0, 1.0),
            False,
            0.0,
        )
    ]
    _bind_runtime_material(renderer, "Overlay", {"_native": None})
    assert calls[-1] == ("Overlay", "", 0, "")


def test_ui_material_generation_rebuilds_screen_and_world_packets(ui):
    from Infernux.core.asset_ref import MaterialRef

    element = ui.add()
    revision = [3]
    native = SimpleNamespace(
        guid='ui-authored-guid', name='Authored UI', _texture_assets_pending=False,
        get_version=lambda: revision[0],
        _get_render_texture=lambda _name: None,
        has_property=lambda _name: False,
        shader_name='Authored UI', vert_shader_name='ui-vertex-guid',
        frag_shader_name='ui-fragment-guid',
        get_render_state=lambda: SimpleNamespace(alpha_clip_enabled=False, alpha_clip_threshold=0.0),
    )
    element.material = MaterialRef(guid='ui-authored-guid')
    type(element).material.get_raw(element)._cached = native
    def set_binding(*args):
        output = ui.renderer.commands if ui.renderer.capture is None else ui.renderer.capture
        output.append(('set_material_binding', args, {}))
    ui.renderer.set_material_binding = set_binding
    rebuilt, commands = ui.frame()
    assert rebuilt
    assert any(command[0] == 'set_material_binding' and command[1][1:3] == ('ui-authored-guid', 3)
               for command in commands)
    revision[0] = 4
    rebuilt, commands = ui.frame()
    assert rebuilt
    assert any(command[0] == 'set_material_binding' and command[1][1:3] == ('ui-authored-guid', 4)
               for command in commands)


@pytest.fixture(params=[False, True], ids=['screen', 'world'])
def ui(scene, monkeypatch, request):
    from Infernux.ui import UICanvas, UIText
    import Infernux.lib as lib
    import Infernux.engine.runtime_screen_ui as module

    root = scene.create_game_object('Root')
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
    manager = SimpleNamespace(get_active_scene=lambda: scene, get_runtime_persistent_scene=lambda: None)
    monkeypatch.setattr(lib, 'SceneManager', SimpleNamespace(instance=lambda: manager))
    texture_cache = SimpleNamespace(has_pending=False, generation=0, texture_id=71)
    texture_cache.get_bound = lambda _: lambda source: texture_cache.texture_id + getattr(source, 'revision', 0)
    monkeypatch.setattr(module, '_get_tex_cache', lambda: texture_cache)
    submission = module.RuntimeScreenUISubmission(engine)
    extracted = []
    dispatch = module._ui_dispatch
    def inspect_dispatch(element, *args, **kwargs):
        extracted.append(element)
        return dispatch(element, *args, **kwargs)
    monkeypatch.setattr(module, '_ui_dispatch', inspect_dispatch)

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
        from Infernux.ui.ui_command_packets import UICommandPackets
        rebuilt, actual = frame()
        assert rebuilt
        submission._command_packets = UICommandPackets()
        renderer.key = None
        _, expected = frame()
        assert actual == expected
        return actual

    return SimpleNamespace(root=root, canvas=canvas, world=request.param, scene=scene,
        add=add, frame=frame, matches_fresh=matches_fresh, renderer=renderer,
        extracted=extracted, submission=submission, textures=texture_cache)


@pytest.mark.parametrize('field', ['color', 'text'])
def test_only_changed_label_reextracts_and_remeasures(ui, field):
    labels = [ui.add() for _ in range(40)]
    ui.frame()
    for index in range(4):
        before = ui.renderer.measures
        setattr(labels[0], field, [index / 4, .5, 1, 1] if field == 'color' else f'Frame {index}')
        assert ui.frame()[0]
        assert ui.extracted == [labels[0]]
        assert ui.renderer.measures - before == (field == 'text')
        assert sum(name == 'add_text' for name, _, _ in ui.renderer.commands) == 40
    assert ui.frame()[0] is False


@pytest.mark.parametrize('change', ['transform', 'parent_transform', 'group', 'mirror',
                                   'enabled', 'active', 'reparent', 'viewport', 'global_dirty'])
def test_shared_dependencies_match_fresh_submission(ui, change):
    from Infernux.lib import Vector3
    from Infernux.ui import UIGroup
    from Infernux.ui.ui_render_revision import mark_runtime_ui_dirty
    group = UIGroup()
    ui.root.add_py_component(group)
    label, other = ui.add(), ui.add()
    ui.frame()
    if change == 'transform':
        label.game_object.transform.local_position = Vector3(10, 20, 3)
        label.game_object.transform.local_euler_angles = Vector3(10, 20, 30)
    elif change == 'parent_transform':
        # A parent's scale moves world UI but never scales its own geometry.
        other.game_object.transform.local_position = Vector3(1, 2, 3)
        ui.root.transform.local_position = Vector3(20, 30, 40)
        ui.root.transform.local_euler_angles = Vector3(20, 30, 40)
        ui.root.transform.local_scale = Vector3(2, 3, 4)
    elif change == 'group':
        group.alpha = .25
    elif change == 'mirror':
        label.mirror_x = True
    elif change == 'enabled':
        label.enabled = False
    elif change == 'active':
        label.game_object.active = False
    elif change == 'reparent':
        label.game_object.set_parent(other.game_object)
    elif change == 'viewport':
        ui.submission.set_target_size(1024, 768)
    else:
        # Tools can explicitly publish an in-place edit.
        label.color[0] = .25
        mark_runtime_ui_dirty()
    commands = ui.matches_fresh()
    if change == 'group':
        assert all(args[9] == .25 for name, args, _ in commands if name == 'add_text')


def test_button_private_state_rebuilds_only_the_button(ui):
    from Infernux.ui import UIButton
    button, label = ui.add(UIButton), ui.add()
    ui.frame()
    button.on_pointer_enter(None)
    ui.frame()
    assert ui.extracted == [button]
    button.on_pointer_down(None)
    ui.matches_fresh()
    button.on_pointer_exit(None)
    ui.matches_fresh()


def test_local_change_visits_only_the_changed_text_measurement(ui, monkeypatch):
    from Infernux.ui.ui_command_packets import UICommandPackets
    labels = [ui.add() for _ in range(100)]
    ui.frame()
    visits = []
    original = UICommandPackets.measure
    def measure(self, element, *args):
        visits.append(element)
        return original(self, element, *args)
    monkeypatch.setattr(UICommandPackets, 'measure', measure)
    labels[17].text = 'Changed once'
    _, actual = ui.frame()
    assert visits == [labels[17]]
    assert ui.extracted == [labels[17]]
    _assert_current_commands_match_fresh(ui, actual)


def test_two_submission_owners_observe_edits_independently(ui):
    from Infernux.engine.runtime_screen_ui import RuntimeScreenUISubmission
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
    first.text = 'Both owners need this edit'
    _, expected = ui.frame()
    # Another edit while this owner is not drawing must remain pending too.
    second.color = [.2, .5, .8, 1.]
    _, expected = ui.frame()
    engine._render_submission_frame += 1
    renderer.commands.clear()
    assert other.submit()
    assert renderer.commands == expected
    assert renderer.captures == 4  # Two initial packets, two changed packets.


def test_released_submission_groups_do_not_stay_on_components(ui):
    import gc
    import weakref
    label = ui.add()
    ui.frame()
    packets = ui.submission._command_packets
    groups = [weakref.ref(group) for group in packets.groups.values()]
    packets.prepare(('different target',))
    gc.collect()
    assert all(group() is None for group in groups)
    assert not label.__dict__['_ui_command_groups']
    label.text = 'No dead subscriber callbacks'


@pytest.mark.parametrize('pose', ['position', 'rotation'])
def test_transform_edit_rebuilds_only_the_affected_leaf(ui, pose):
    from Infernux.lib import Vector3
    labels = [ui.add() for _ in range(24)]
    ui.frame()
    target = labels[9]
    if pose == 'position':
        target.game_object.transform.local_position = Vector3(30., 40., 2.)
    else:
        target.game_object.transform.local_euler_angles = Vector3(0., 0., 37.)
    rebuilt, actual = ui.frame()
    assert rebuilt
    assert ui.extracted == ([] if ui.world else [target])
    _assert_current_commands_match_fresh(ui, actual)


def test_parent_ui_pose_updates_descendants_not_unrelated_branches(ui):
    from Infernux.lib import Vector3
    from Infernux.ui import UIFrame
    parent = ui.add(UIFrame)
    child = ui.add(parent=parent.game_object)
    bridge = ui.scene.create_game_object('Non-UI hierarchy bridge')
    bridge.set_parent(parent.game_object)
    descendant = ui.add(parent=bridge)
    unrelated = ui.add()
    ui.frame()
    parent.game_object.transform.local_position = Vector3(30., 40., 2.)
    rebuilt, actual = ui.frame()
    assert rebuilt
    assert set(ui.extracted) == (set() if ui.world else {parent, child, descendant})
    assert unrelated not in ui.extracted
    _assert_current_commands_match_fresh(ui, actual)


def test_world_layer_edit_does_not_rebuild_other_world_geometry(ui):
    first, second = ui.add(), ui.add()
    ui.frame()
    first.game_object.layer = 8
    rebuilt, actual = ui.frame()
    if ui.world:
        assert rebuilt
        assert ui.extracted == []
        _assert_current_commands_match_fresh(ui, actual)
    else:
        assert not rebuilt  # Screen UI ignores its GameObject layer.


def test_world_parent_animation_reuses_local_geometry(ui, monkeypatch):
    if not ui.world:
        pytest.skip('World geometry is independent of its scene pose')
    from Infernux.lib import Vector3
    from Infernux.ui.ui_command_packets import UICommandPackets
    labels = [ui.add() for _ in range(60)]
    for index, label in enumerate(labels):
        label.game_object.transform.local_position = Vector3(index * .1, 0, 0)
    ui.frame()
    captures = ui.renderer.captures
    def unexpected(*_):
        raise AssertionError('World pose-only edits must not measure text')
    with monkeypatch.context() as patch:
        patch.setattr(UICommandPackets, 'measure', unexpected)
        for index in range(4):
            ui.root.transform.local_position = Vector3(float(index + 1), 2., 3.)
            ui.root.transform.local_euler_angles = Vector3(15., float(index * 10), 20.)
            _, actual = ui.frame()
            assert ui.extracted == []
            assert ui.renderer.captures == captures
    _assert_current_commands_match_fresh(ui, actual)


def test_screen_and_world_pose_invalidations_stay_independent(ui):
    from Infernux.lib import Vector3
    from Infernux.ui import UICanvas
    first, untouched = ui.add(), ui.add()
    island = ui.scene.create_game_object('Other UI space')
    if ui.world:
        island.add_py_component(UICanvas())
    other = ui.add(parent=island)
    ui.frame()
    for index, target in enumerate((first, other)):
        target.game_object.transform.local_position = Vector3(20. + index, 30., 1.)
        _, actual = ui.frame()
        target_world = ui.world if target is first else not ui.world
        assert ui.extracted == ([] if target_world else [target])
        assert untouched not in ui.extracted
        _assert_current_commands_match_fresh(ui, actual)


def test_custom_draw_layout_edit_reaches_later_cached_sibling(ui):
    from Infernux.ui import UIButton, UIText, UIFrame, UILayoutDirection
    from Infernux.ui import ui_render_dispatch as dispatch
    parent = ui.add(UIFrame)
    parent.layout_direction = UILayoutDirection.Horizontal
    first = ui.add(UIButton, parent.game_object)
    custom = ui.add(UIText, parent.game_object)
    last = ui.add(UIButton, parent.game_object)
    first.label, last.label = 'first', 'last'
    original = dispatch.get_ui_renderer('UIText', 'runtime')
    new_width = [160.]
    def draw(element, renderer, **kwargs):
        first.width = new_width[0]
        renderer.add_text(kwargs['ui_list'], 0, 0, 20, 20, 'custom')
    try:
        dispatch.register_ui_renderer('UIText', 'runtime', draw)
        _, before = ui.frame()
        new_width[0] = 320.
        custom.text = 'Change layout during this custom draw'
        _, after = ui.frame()
        def last_x(commands):
            return next(args[1] for name, args, _ in commands
                        if name == 'add_text' and args[5] == 'last')
        assert last_x(after) == pytest.approx(last_x(before) + (0. if ui.world else 160.))
    finally:
        dispatch.register_ui_renderer('UIText', 'runtime', original)


def test_frame_clip_and_auto_sized_text_refresh_siblings(ui):
    from Infernux.ui import UIFrame, UIText, UILayoutDirection, TextResizeMode
    parent = ui.add(UIFrame)
    parent.layout_direction = UILayoutDirection.Horizontal
    first, second = ui.add(UIText, parent.game_object), ui.add(UIText, parent.game_object)
    first.resize_mode = TextResizeMode.AutoWidth
    ui.frame()
    first.text = 'Much wider text label changes the following sibling position'
    ui.matches_fresh()
    parent.clip_content = True
    commands = ui.matches_fresh()
    assert any(name == 'push_clip_rect' for name, _, _ in commands) is not ui.world
    parent.clip_content = False
    commands = ui.matches_fresh()
    assert not any(name == 'push_clip_rect' for name, _, _ in commands)


def test_material_version_and_image_resize_refresh_commands(ui):
    from Infernux.lib import InxMaterial
    from Infernux.core.material import Material
    from Infernux.core.render_texture import RenderTexture
    from Infernux.ui import UIImage
    image = ui.add(UIImage)
    material = Material(InxMaterial('UI packets', 'Unlit'))
    image.material = material
    target = RenderTexture.__new__(RenderTexture)
    target._native = SimpleNamespace(revision=1, asset_guid='')
    image.texture = target
    ui.frame()
    material.set_color('baseColor', .2, .4, .6, 1.)
    ui.matches_fresh()
    target._native.revision += 1
    commands = ui.matches_fresh()
    assert next(args[1] for name, args, _ in commands if name == 'add_image') == 73
    ui.textures.generation += 1
    ui.textures.texture_id = 90
    ui.matches_fresh()
    ui.textures.has_pending = True
    ui.textures.texture_id = 110
    ui.matches_fresh()
    ui.textures.texture_id = 120
    ui.matches_fresh()


def _assert_current_commands_match_fresh(ui, actual):
    from Infernux.ui.ui_command_packets import UICommandPackets
    ui.submission._command_packets = UICommandPackets()
    ui.renderer.key = None
    assert ui.frame()[1] == actual


@pytest.mark.parametrize('slot', ['material', 'text_material'])
def test_material_edit_reextracts_only_its_consumers(ui, slot):
    from Infernux.lib import InxMaterial
    from Infernux.core.material import Material
    from Infernux.ui import UIButton
    materials = [Material(InxMaterial(f'UI source {i}', 'Unlit')) for i in range(3)]
    buttons = [ui.add(UIButton) for _ in range(12)]
    for index, button in enumerate(buttons):
        setattr(button, slot, materials[index % 3])
    untouched = ui.add()
    ui.frame()
    materials[1].set_color('baseColor', .2, .4, .8, 1.)
    rebuilt, actual = ui.frame()
    assert rebuilt
    assert ui.extracted == buttons[1::3]
    assert untouched not in ui.extracted
    _assert_current_commands_match_fresh(ui, actual)

    # A binding edit must not rebuild other controls using the same sources.
    setattr(buttons[0], slot, materials[1])
    _, actual = ui.frame()
    assert ui.extracted == [buttons[0]]
    _assert_current_commands_match_fresh(ui, actual)
    setattr(buttons[0], slot, None)
    _, actual = ui.frame()
    assert ui.extracted == [buttons[0]]
    _assert_current_commands_match_fresh(ui, actual)


@pytest.mark.parametrize('change', ['text', 'color', 'width', 'hover'])
def test_visual_edits_keep_resource_binding_membership(ui, change):
    from Infernux.core.material import Material
    from Infernux.lib import InxMaterial
    from Infernux.ui import UIButton
    from Infernux.ui import ui_render_dispatch as dispatch
    material = Material(InxMaterial('Retained binding', 'Unlit'))
    labels = [ui.add() for _ in range(24)]
    for label in labels:
        label.material = material
    button = ui.add(UIButton)
    ui.frame()
    bindings = dispatch._runtime_dependencies.materials
    if change == 'hover':
        button.on_pointer_enter(None)
    else:
        setattr(labels[0], change, {'text': 'Changed', 'color': [.2, .5, .8, 1.], 'width': 290.}[change])
    ui.frame()
    assert dispatch._runtime_dependencies.materials is bindings


def test_material_owner_replacement_does_not_hide_other_reference_owners(ui, monkeypatch):
    from Infernux.core.material import Material
    from Infernux.lib import InxMaterial
    original = Material(InxMaterial('Previous owner', 'Unlit'))
    replacement = Material(InxMaterial('Replacement owner', 'Unlit'))
    replacement.set_color('baseColor', .1, .5, .9, 1.)
    first, second = ui.add(), ui.add()
    first.material = original
    second.material = original
    ui.frame()
    reference = type(first).material.get_raw(first)
    reference.invalidate()
    # Resource publication can replace one resolved alias without a field edit.
    monkeypatch.setattr(reference, '_do_resolve', lambda: replacement)
    rebuilt, actual = ui.frame()
    assert rebuilt
    assert ui.extracted == [first]
    _assert_current_commands_match_fresh(ui, actual)


def test_material_binding_document_restore_and_removal_match_fresh(ui, monkeypatch):
    from Infernux.components.value_document import make_asset_ref
    from Infernux.core.assets import AssetManager
    from Infernux.core.material import Material
    from Infernux.lib import InxMaterial
    material = Material(InxMaterial('Inspector material', 'Unlit'))
    material.set_color('baseColor', .2, .4, .8, 1.)
    monkeypatch.setattr(AssetManager, 'load_by_guid', lambda *_args, **_kwargs: material)
    label, other = ui.add(), ui.add()
    ui.frame()
    for reference in (make_asset_ref('Material', '1234567890abcdef1234567890abcdef'), None):
        label._deserialize_fields_document({'material': reference})
        _, actual = ui.frame()
        assert ui.extracted == [label]
        _assert_current_commands_match_fresh(ui, actual)


def test_render_texture_resize_reextracts_only_its_images(ui):
    from Infernux.core.render_texture import RenderTexture
    from Infernux.ui import UIImage
    targets = []
    for revision in (1, 10):
        target = RenderTexture.__new__(RenderTexture)
        target._native = SimpleNamespace(revision=revision, asset_guid='')
        targets.append(target)
    images = [ui.add(UIImage) for _ in range(12)]
    for index, image in enumerate(images):
        image.texture = targets[index % 2]
    ui.add()
    ui.frame()
    targets[0]._native.revision += 1
    _, actual = ui.frame()
    assert ui.extracted == images[::2]
    _assert_current_commands_match_fresh(ui, actual)


def test_button_render_texture_assignment_tracks_later_target_revision(ui):
    from Infernux.core.render_texture import RenderTexture
    from Infernux.ui import UIButton

    button = ui.add(UIButton)
    ui.frame()
    target = RenderTexture.__new__(RenderTexture)
    target._native = SimpleNamespace(revision=1, asset_guid='')
    button.background_texture = target
    ui.frame()
    target._native.revision += 1
    _, actual = ui.frame()
    assert ui.extracted == [button]
    _assert_current_commands_match_fresh(ui, actual)


def test_renderer_replacement_releases_old_bound_calls(ui):
    label = ui.add()
    ui.frame()
    original = ui.submission._engine_ref().get_screen_ui_renderer
    replacement = Renderer()
    ui.submission._engine_ref().get_screen_ui_renderer = lambda: replacement
    label.text = 'New renderer'
    ui.frame()
    assert ui.renderer.commands == []
    assert any(name == 'add_text' for name, _, _ in replacement.commands)
    ui.submission._engine_ref().get_screen_ui_renderer = original


def test_destroyed_element_packets_do_not_survive_membership_change(ui):
    doomed, survivor = ui.add(), ui.add()
    ui.frame()
    ui.scene.destroy_game_object(doomed.game_object)
    ui.scene.process_pending_destroys()
    commands = ui.matches_fresh()
    assert sum(name == 'add_text' for name, _, _ in commands) == 1
    assert all(doomed not in group.indices for group in ui.submission._command_packets.groups.values())
    replacement = ui.add()
    replacement.text = 'Replacement'
    commands = ui.matches_fresh()
    assert sum(name == 'add_text' for name, _, _ in commands) == 2


def test_explicit_font_chain_and_measured_size_publication(ui, monkeypatch):
    from Infernux.application import Application
    from Infernux.core.asset_ref import create_asset_ref
    from Infernux.engine.project_context import set_runtime_asset_resolver
    from Infernux.ui import TextResizeMode
    from Infernux.ui.ui_render_revision import mark_runtime_ui_dirty
    paths = {
        "font-cjk": "Assets/Fonts/CJK.ttf",
        "font-emoji": "Assets/Fonts/Emoji.ttf",
    }
    monkeypatch.setattr(Application, "is_player", staticmethod(lambda: True))
    set_runtime_asset_resolver(lambda guid: paths.get(guid))
    label = ui.add()
    label.resize_mode = TextResizeMode.AutoWidth
    ui.frame()
    try:
        label.fallback_fonts = [
            create_asset_ref("Font", guid="font-cjk", path_hint="stale.ttf")
        ]
        commands = ui.matches_fresh()
        assert next(args[-1] for name, args, _ in commands if name == 'add_text') == ['Assets/Fonts/CJK.ttf']
        label.fallback_fonts = [
            create_asset_ref("Font", guid="font-cjk", path_hint="stale.ttf"),
            create_asset_ref("Font", guid="font-emoji", path_hint="stale.ttf"),
        ]
        mark_runtime_ui_dirty()
        commands = ui.matches_fresh()
        assert len(next(args[-1] for name, args, _ in commands if name == 'add_text')) == 2
    finally:
        set_runtime_asset_resolver(None)


def test_custom_renderer_keeps_its_native_renderer_contract(ui, monkeypatch):
    from Infernux.ui import ui_render_dispatch as dispatch
    calls = []
    def custom(element, renderer, **_):
        assert renderer is ui.renderer
        calls.append(element.text)
    monkeypatch.setitem(dispatch._RENDERERS, ('UIText', 'runtime'), custom)
    dispatch._RESOLVED_RENDERERS.clear()
    label = ui.add()
    ui.frame()
    label.text = 'custom second frame'
    ui.frame()
    assert calls == ['New Text', 'custom second frame']
    dispatch._RESOLVED_RENDERERS.clear()


def test_geometry_packets_batch_and_font_epoch_invalidates(ui):
    labels = [ui.add() for _ in range(12)]
    ui.frame()
    assert ui.renderer.captures == 12
    assert ui.renderer.appends == 1
    labels[5].text = 'Only this packet changes'
    ui.frame()
    assert ui.renderer.captures == 13
    assert ui.renderer.appends == 2
    assert ui.extracted == [labels[5]]
    ui.renderer.font_epoch += 1
    ui.frame()
    assert ui.renderer.captures == 25
    assert ui.renderer.appends == 3
    assert set(ui.extracted) == set(labels)


def test_unchanged_packets_do_not_rediscover_renderer_capabilities(ui, monkeypatch):
    from Infernux.ui import ui_command_packets as packets
    labels = [ui.add() for _ in range(24)]
    ui.frame()
    calls = []
    original = packets._can_retain_runtime_commands
    def inspect(element):
        calls.append(element)
        return original(element)
    monkeypatch.setattr(packets, '_can_retain_runtime_commands', inspect)
    labels[4].text = 'Only this element changed'
    ui.frame()
    assert calls == []  # Membership already owns the renderer classification.


def test_registered_renderer_replacement_invalidates_warm_packets(ui):
    from Infernux.ui import ui_render_dispatch as dispatch
    label = ui.add()
    ui.frame()
    original = dispatch.get_ui_renderer('UIText', 'runtime')
    calls = []
    def custom(element, renderer, **kwargs):
        calls.append(element)
        renderer.add_text(kwargs['ui_list'], 0, 0, 20, 20, 'REPLACED')
    try:
        dispatch.register_ui_renderer('UIText', 'runtime', custom)
        _, commands = ui.frame()
        assert calls == [label]
        assert [args[5] for name, args, _ in commands if name == 'add_text'] == ['REPLACED']
        label.color = [.2, .5, .8, 1.]
        ui.frame()
        assert calls == [label, label]
    finally:
        dispatch.register_ui_renderer('UIText', 'runtime', original)
    _, commands = ui.frame()
    assert [args[5] for name, args, _ in commands if name == 'add_text'] == [label.text]


def test_packet_capture_failure_does_not_publish_partial_geometry(ui, monkeypatch):
    import Infernux.engine.runtime_screen_ui as runtime
    label = ui.add()
    original = runtime._ui_dispatch
    def fail(*_args, **_kwargs):
        raise RuntimeError('deliberate capture failure')
    monkeypatch.setattr(runtime, '_ui_dispatch', fail)
    with pytest.raises(RuntimeError, match='deliberate capture failure'):
        ui.frame()
    assert ui.renderer.capture is None
    assert not ui.submission._command_packets.groups
    monkeypatch.setattr(runtime, '_ui_dispatch', original)
    ui.matches_fresh()


@pytest.mark.parametrize('inactive', [False, True])
def test_font_epoch_remeasures_intrinsic_size_without_field_edits(ui, monkeypatch, inactive):
    from Infernux.ui import TextResizeMode
    label = ui.add()
    label.resize_mode = TextResizeMode.AutoWidth
    ui.frame()
    previous = label.get_resolved_size()
    if inactive:
        label.enabled = False
        ui.frame()
    monkeypatch.setattr(ui.renderer, 'measure_text', lambda *_: (previous[0] + 40, previous[1] + 8))
    ui.renderer.font_epoch += 1
    ui.frame()
    if inactive:
        label.enabled = True
        ui.frame()
    assert label.get_resolved_size() == (previous[0] + 40, label.height)


def test_custom_renderer_preserves_order_between_retained_packets(ui, monkeypatch):
    from Infernux.ui import UIButton
    from Infernux.ui import ui_render_dispatch as dispatch
    first = ui.add(UIButton)
    first.label = 'before'
    custom = ui.add()
    custom.text = 'middle'
    last = ui.add(UIButton)
    last.label = 'after'
    def draw(element, renderer, **kwargs):
        renderer.add_text(kwargs['ui_list'], 0, 0, 20, 20, element.text)
    monkeypatch.setitem(dispatch._RENDERERS, ('UIText', 'runtime'), draw)
    dispatch._RESOLVED_RENDERERS.clear()
    _, commands = ui.frame()
    assert [args[5] for name, args, _ in commands if name == 'add_text'] == ['before', 'middle', 'after']
    assert ui.renderer.appends == 2
    first.color = [1, 0, 0, 1]
    commands = ui.matches_fresh()
    assert [args[5] for name, args, _ in commands if name == 'add_text'] == ['before', 'middle', 'after']
    dispatch._RESOLVED_RENDERERS.clear()
