"""Public RenderTexture argument and ownership contract; real GPU coverage is native."""
from types import SimpleNamespace

import pytest
import Infernux as inx
from Infernux.application import Application
from Infernux.core import RenderTexture
from Infernux.lib import PixelFormat, SampleCount
from Infernux.rendergraph.graph import RenderGraph


@pytest.mark.parametrize('size', [(0, 1), (1, -1), (True, 1), (1.5, 2), ('64', 64)])
def test_size_requires_positive_integer_pixels(size):
    with pytest.raises(ValueError, match='positive integers'):
        RenderTexture(*size)


@pytest.mark.parametrize('samples', [0, 3, 16, True, 4.0])
def test_sample_count_is_explicit(samples):
    with pytest.raises(ValueError, match='samples'):
        RenderTexture(64, 32, samples=samples)


def test_graphical_engine_required(monkeypatch):
    monkeypatch.setattr(Application, '_current_engine', staticmethod(lambda: None))
    with pytest.raises(RuntimeError, match='graphical engine'):
        RenderTexture(64, 32)


def test_filter_is_not_silently_replaced():
    with pytest.raises(ValueError, match='filter'):
        RenderTexture(64, 32, filter='automatic')


@pytest.mark.parametrize('scale', [0.5, (0.5,), (0, 1), (-1, 1), (True, 1), ('0.5', 1),
                                    (float('inf'), 1), (1, float('nan'))])
def test_relative_scale_requires_two_positive_finite_numbers(scale):
    with pytest.raises(ValueError, match='scale'):
        RenderTexture(scale=scale)


def test_relative_and_absolute_size_cannot_be_combined():
    with pytest.raises(ValueError, match='not both'):
        RenderTexture(64, 32, scale=(0.5, 0.5))


def test_native_descriptor_and_resize_boundary(monkeypatch):
    class NativeTarget:
        revision = 1
        is_valid = True
        resident_bytes = 64 * 32 * (8 + 8 * 4 + 4 * 4)

        def __init__(self, description):
            self.description = description

        @property
        def width(self):
            return self.description.width

        @property
        def height(self):
            return self.description.height

        def resize(self, width, height):
            if (width, height) == (self.description.width, self.description.height):
                return False
            self.description.width, self.description.height = width, height
            self.revision += 1
            return True

    host = SimpleNamespace(_create_render_texture=NativeTarget)
    monkeypatch.setattr(Application, '_current_engine', staticmethod(
        lambda: SimpleNamespace(get_native_engine=lambda: host)))
    target = RenderTexture(64, 32, format=PixelFormat.RGBA16_SFLOAT,
                           depth_format=PixelFormat.D32_SFLOAT, samples=4, filter='nearest', storage=True,
                           sampled_depth=True)
    assert inx.RenderTexture is RenderTexture
    assert (target.width, target.height, target.samples) == (64, 32, 4)
    assert target.format == PixelFormat.RGBA16_SFLOAT
    assert target.depth_format == PixelFormat.D32_SFLOAT
    assert target.sampled_depth
    assert target._native.description.samples == SampleCount.COUNT_4
    assert target._native.description.storage and not target._native.description.linear_filter
    assert target.is_valid and target.resident_bytes > 0
    assert not target.resize(64, 32) and target.revision == 1
    assert target.resize(128, 96) and target.revision == 2
    assert (target.width, target.height) == (128, 96)
    with pytest.raises(ValueError):
        target.resize(0, 32)
    assert target.revision == 2
    with pytest.raises(AttributeError):
        target.width = 1


@pytest.fixture
def graphical_application(engine, monkeypatch):
    """Use the live Vulkan test engine for public resource allocation."""
    monkeypatch.setattr(Application, '_current_engine', staticmethod(
        lambda: SimpleNamespace(get_native_engine=lambda: engine)))


@pytest.mark.parametrize('resolution', [(641, 401), (1920, 1080)])
def test_bilingual_relative_texture_example(graphical_application, engine, scene, resolution):
    import re
    from pathlib import Path

    guide = Path(__file__).parents[2] / 'docs/learn/rendergraph-advanced.md'
    blocks = re.findall(r'```python\n(.*?)\n```', guide.read_text(encoding='utf-8'), re.DOTALL)
    examples = [block for block in blocks if block.startswith('PixelFormat = inx.rendergraph.Format')]
    assert len(examples) == 2 and examples[0] == examples[1]
    before = engine.renderer_frame_snapshot
    previous = before['game_target_width'], before['game_target_height']
    camera = scene.create_game_object('GuideCamera').add_component('Camera')
    scope = {'inx': inx, 'camera': camera, 'image': SimpleNamespace()}
    try:
        engine.resize_game_render_target(*resolution)
        exec(examples[0], scope)
        half = scope['half_size']
        assert (half.width, half.height) == tuple((value + 1) // 2 for value in resolution)
        assert half.scale == (.5, .5) and half.samples == 4
        assert scope['fixed'].scale is None
        assert camera.target_texture._native is half._native
        assert scope['image'].texture is half
        with pytest.raises(AttributeError):
            half.scale = (1, 1)
    finally:
        camera.target_texture = None
        if all(previous):
            engine.resize_game_render_target(*previous)


def test_relative_target_tracks_game_pixels_not_scene_size(graphical_application, engine):
    snapshot = engine.renderer_frame_snapshot
    old_game = snapshot['game_target_width'], snapshot['game_target_height']
    old_scene = snapshot['scene_target_width'], snapshot['scene_target_height']
    try:
        engine.resize_game_render_target(101, 99)
        target = RenderTexture(scale=(0.5, 0.25), depth_format=PixelFormat.D32_SFLOAT, samples=4)
        other = RenderTexture(scale=(0.25, 0.5))
        fixed = RenderTexture(19, 11)
        assert target.scale == (0.5, 0.25) and fixed.scale is None
        assert (target.width, target.height) == (51, 25)
        revision = target.revision
        engine.resize_scene_render_target(213, 115)
        assert target.revision == revision
        engine.resize_game_render_target(203, 107)
        assert (target.width, target.height) == (102, 27)
        assert (other.width, other.height) == (51, 54)
        assert (fixed.width, fixed.height, fixed.revision) == (19, 11, 1)
        assert target.revision == revision + 1
        engine.resize_game_render_target(203, 107)
        engine.resize_game_render_target(0, 0)
        assert target.revision == revision + 1
        with pytest.raises(ValueError, match='Game output'):
            target.resize(45, 23)
        del target, other
        engine.resize_game_render_target(111, 73)  # expired registrations do not retain resources
    finally:
        engine.resize_scene_render_target(*old_scene)
        engine.resize_game_render_target(*old_game)


def test_import_retains_native_owner_and_observes_resize(graphical_application):
    target = RenderTexture(37, 23, format=PixelFormat.RGBA16_SFLOAT)
    graph = RenderGraph('Persistent target')
    output = graph.import_texture('target', target)
    assert graph.import_texture('another_label', target) is output
    with graph.add_pass('Clear') as render_pass:
        render_pass.write_color(output).set_clear(color=(.2, .4, .8, 1.0))
    graph.set_output(output)
    target.resize(53, 29)
    description = graph.build()
    assert len(description.textures) == 1
    texture = description.textures[0]
    assert texture.render_texture is target._native
    assert (texture.width, texture.height) == (53, 29)
    assert texture.role.name == 'PERSISTENT'


def test_import_does_not_shadow_existing_resource(graphical_application):
    target = RenderTexture(16, 16)
    other = RenderTexture(16, 16)
    graph = RenderGraph()
    graph.create_texture('occupied')
    graph.create_buffer('buffer', 16)
    output = graph.import_texture('target', target)
    assert graph.import_texture('target', target) is output
    for name in ('occupied', 'buffer'):
        with pytest.raises(ValueError, match='already exists'):
            graph.import_texture(name, target)
    with pytest.raises(ValueError, match='already exists'):
        graph.import_texture('target', other)
    with pytest.raises(TypeError, match='RenderTexture'):
        graph.import_texture('not_a_target', 'asset-guid')


def test_import_scope_and_single_sample_attachment_identity(graphical_application):
    graph = RenderGraph()
    target = RenderTexture(16, 16)
    with graph.name_scope('offscreen'):
        output = graph.import_texture('color', target)
        assert graph.import_texture('resolve', target, attachment='resolve') is output
    assert output.name == 'offscreen/color'
    with pytest.raises(ValueError, match='no depth'):
        graph.import_texture('depth', target, attachment='depth')
    with pytest.raises(ValueError, match='attachment must be'):
        graph.import_texture('invalid', target, attachment='stencil')


@pytest.mark.parametrize('samples', [1, 2, 4])
def test_import_msaa_depth_and_resolve_descriptions(graphical_application, samples):
    target = RenderTexture(37, 23, samples=samples, depth_format=PixelFormat.D32_SFLOAT)
    graph = RenderGraph()
    color = graph.import_texture('color', target)
    depth = graph.import_texture('depth', target, attachment='depth')
    resolve = graph.import_texture('resolve', target, attachment='resolve')
    assert graph.import_texture('depth_alias', target, attachment='depth') is depth
    with pytest.raises(ValueError, match='already exists'):
        graph.import_texture('color', target, attachment='depth')
    with graph.add_pass('Clear') as p:
        p.write_color(color).write_depth(depth).set_clear(color=(.1, .2, .3, 1), depth=.25)
        if samples > 1:
            p.write_resolve(resolve)
    graph.set_output(resolve)
    description = graph.build()
    assert len(description.textures) == (2 if samples == 1 else 3)
    assert color.samples == depth.samples == samples
    assert depth.is_depth and resolve.samples == 1
    assert description.textures[0].attachment.name == 'COLOR'
    assert description.textures[1].attachment.name == 'DEPTH'
    assert description.passes[0].write_depth == depth.name
    if samples > 1:
        assert description.textures[2].attachment.name == 'RESOLVE'
        assert description.passes[0].resolve_color == resolve.name


def test_sampled_depth_requires_a_depth_attachment(graphical_application):
    with pytest.raises(ValueError, match='depth'):
        RenderTexture(16, 16, sampled_depth=True)


def test_camera_output_owns_target_without_fabricating_asset_identity(graphical_application, scene):
    camera = scene.create_game_object('OffscreenCamera').add_component('Camera')
    before = camera.serialize_document()
    assert camera.target_texture is None
    target = RenderTexture(37, 23, samples=4, depth_format=PixelFormat.D32_SFLOAT)
    try:
        camera.target_texture = target
        native = target._native
        assert camera.target_texture._native is native
        assert camera.serialize_document() == before
        target.resize(53, 29)
        assert (camera.target_texture.width, camera.target_texture.height) == (53, 29)
        del target
        assert camera.target_texture.is_valid
        with pytest.raises(TypeError, match='RenderTexture'):
            camera.target_texture = 'not-an-asset-guid'
        with pytest.raises(ValueError, match='depth'):
            camera.target_texture = RenderTexture(8, 8)
        assert camera.target_texture._native is native
        assert camera.serialize_document() == before
    finally:
        camera.target_texture = None
    assert camera.target_texture is None


def test_scene_and_game_target_replacement_uses_rhi_attachments(engine):
    previous_samples = engine.get_msaa_samples()
    before = engine.renderer_frame_snapshot
    previous_scene = (before['scene_target_width'], before['scene_target_height'])
    previous_game = (before['game_target_width'], before['game_target_height'])
    try:
        engine.resize_scene_render_target(79, 43)
        engine.resize_game_render_target(113, 67)
        generation = engine.get_game_render_target_generation()
        for samples in engine.msaa_state['supported_samples']:
            engine.set_msaa_samples(samples)
            state = engine.msaa_state
            assert state['active_samples'] == samples
            assert state['scene_target_aligned'] and state['game_target_aligned']
            assert state['material_pipelines_aligned']
            assert state['scene_msaa_color_bytes'] == (0 if samples == 1 else 79 * 43 * 8 * samples)
            assert state['game_msaa_color_bytes'] == (0 if samples == 1 else 113 * 67 * 8 * samples)
        engine.resize_game_render_target(157, 89)
        after = engine.renderer_frame_snapshot
        assert (after['game_target_width'], after['game_target_height']) == (157, 89)
        assert engine.get_game_render_target_generation() > generation
    finally:
        engine.set_msaa_samples(previous_samples)
        if all(previous_scene):
            engine.resize_scene_render_target(*previous_scene)
        if all(previous_game):
            engine.resize_game_render_target(*previous_game)


def test_camera_cache_observes_additive_unload_without_a_render_frame(engine, scene):
    from Infernux.lib import SceneManager

    manager = SceneManager.instance()
    baseline = set(engine.renderer_frame_snapshot['game_camera_ids'])
    # Reuse allocations repeatedly so stale borrowed camera pointers cannot
    # accidentally pass because their old memory still looks like a Camera.
    for index in range(12):
        extra = manager.create_scene(f'CameraLifetime{index}')
        camera = extra.create_game_object('TemporaryCamera').add_component('Camera')
        camera_id = camera.component_id
        assert camera_id in engine.renderer_frame_snapshot['game_camera_ids']
        manager.unload_scene(extra)
        del camera, extra
        assert set(engine.renderer_frame_snapshot['game_camera_ids']) == baseline


def test_camera_cache_observes_enabled_and_order_edits_in_same_frame(engine, scene):
    camera_a = scene.create_game_object('CameraA').add_component('Camera')
    camera_b = scene.create_game_object('CameraB').add_component('Camera')
    a, b = camera_a.component_id, camera_b.component_id
    before = engine.renderer_frame_snapshot['game_camera_ids']
    assert a in before and b in before
    camera_a.enabled = False
    assert a not in engine.renderer_frame_snapshot['game_camera_ids']
    camera_a.enabled = True
    camera_a.depth = 20
    ordered = engine.renderer_frame_snapshot['game_camera_ids']
    assert ordered.index(b) < ordered.index(a)
