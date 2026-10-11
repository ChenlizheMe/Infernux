"""Authored coverage through real linked Forward, Depth and Shadow programs."""
from pathlib import Path
import struct
import zlib

import numpy as np

from infernux.core.assets import AssetManager
from infernux.core.material import Material
from infernux.lib import LightShadows, SceneManager, Vector3
from infernux.rendergraph import Format
from infernux.renderstack import RenderPipeline, DefaultForwardPipeline, DefaultForwardPlusPipeline, DefaultDeferredPipeline


STEPS = ('baseline', 'solid', 'masked_off', 'masked_on', 'reference', 'negative_off', 'clipped', 'restored')
DEPTH_MONITOR = '''#version 450
ShaderInfo {
    Name "Surface Discard Depth Monitor"
    Hidden On
    Capabilities [Fullscreen]
    Resources { Texture2D coverageColor Texture2D coverageDepth }
    Inputs { Float2 inUV }
    Outputs { Float4 outColor }
}
void main() {
    float forwardHit = texture(coverageColor, inUV).r > 0.9 ? 1.0 : 0.0;
    float depthHit = texture(coverageDepth, inUV).r < 1.0 ? 1.0 : 0.0;
    outColor = vec4(forwardHit, depthHit, abs(forwardHit - depthHit), 1.0);
}
'''


def quad(left=-.5):
    u = left + .5
    return (f'v {left} -0.5 0\nv 0.5 -0.5 0\nv 0.5 0.5 0\nv {left} 0.5 0\n'
            f'vt {u} 0\nvt 1 0\nvt 1 1\nvt {u} 1\nvn 0 0 -1\n'
            'f 1/1/1 3/3/1 2/2/1\nf 1/1/1 4/4/1 3/3/1\n')


def mask_png():
    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))
    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', 128, 1, 8, 6, 0, 0, 0)) +
            chunk(b'IDAT', zlib.compress(b'\0' + b'\0\0\0\xff' * 64 + b'\xff\xff\xff\xff' * 64)) + chunk(b'IEND', b''))


class DepthCoveragePipeline(RenderPipeline):
    name = 'Surface Discard Coverage'

    def define_topology(self, graph):
        graph.set_msaa_samples(1)
        color = graph.create_texture('color', camera_target=True)
        forward = graph.create_texture('forward', format=Format.RGBA16_SFLOAT, samples=1)
        forward_depth = graph.create_texture('forward_depth', format=Format.D32_SFLOAT, samples=1)
        depth = graph.create_texture('depth', format=Format.D32_SFLOAT, samples=1)
        graph.add_pass('Forward coverage').write_color(forward).write_depth(forward_depth).set_clear(
            color=(0,0,0,0), depth=1.).draw_renderers(material_pass='forward')
        graph.add_pass('Depth coverage').write_depth(depth).set_clear(depth=1.).draw_renderers(material_pass='depth')
        graph.add_pass('Compare coverage').write_color(color).set_texture('coverageColor', forward).set_texture(
            'coverageDepth', depth).fullscreen_quad('Surface Discard Depth Monitor')
        graph.screen_ui_section(resources={'color'})
        graph.set_output(color)


class SurfaceDiscardCase:
    def __init__(self, native, scene, directory, kind='direct', pipeline='forward', samples=1):
        self.native, self.scene = native, scene
        self.database = native.get_asset_database()
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.assets, self.objects, self.images = [], [], {}
        self.kind, self.depth = kind, pipeline == 'depth'
        self.samples = samples
        manager = SceneManager.instance()
        self.previous_roots = [(obj, obj.active_self)
                               for index in range(manager.scene_count)
                               for obj in manager.get_scene_at(index).get_root_objects()]
        for obj, _ in self.previous_roots:
            obj.active = False

        def asset(name, source):
            path = self.directory / name
            if isinstance(source, bytes):
                path.write_bytes(source)
            else:
                path.write_text(source, encoding='utf-8')
            result = AssetManager.import_asset(str(path), database=self.database)
            assert result.succeeded, result.error
            self.assets.append(path)
            return result.guid

        self.full_mesh = asset('Full.obj', quad())
        self.half_mesh = asset('Half.obj', quad(0.))
        mask = asset('Mask.png', mask_png())
        operation = {'direct':'if (getUV().x < material.cutoff) discard;',
                     'helper':'alphaClip(getUV().x, material.cutoff);',
                     'texture':'alphaClip(sampleAlbedoAlpha(maskTexture).r, material.cutoff);'}[kind]
        asset('Caster.frag', '''#version 450
ShaderInfo {
    Name "Surface Discard Caster"
    ShadingModel Unlit
    Cull Off
    CastShadows On
    Properties { Float cutoff = 0.0 Float alphaValue = 1.0 Texture2D maskTexture = white }
}
void surface(out SurfaceData s) {
    s = InitSurfaceData();
    ''' + operation + '''
    s.albedo = vec3(1,0,1);
    s.alpha = material.alphaValue;
}
''')
        asset('Receiver.frag', '''#version 450
ShaderInfo { Name "Surface Discard Receiver" ShadingModel PBR Cull Off }
void surface(out SurfaceData s) { s = InitSurfaceData(); s.albedo = vec3(.5); s.smoothness = 0.; }
''')
        asset('Monitor.frag', DEPTH_MONITOR)

        def obj(name, position=(0,0,0), scale=(1,1,1)):
            value = scene.create_game_object(name)
            value.layer = 18
            value.transform.position = Vector3(*position)
            value.transform.local_scale = Vector3(*scale)
            self.objects.append(value)
            return value

        self.camera = obj('Surface Discard Camera', (0,0,-6)).add_component('Camera')
        self.camera.culling_mask = 1 << 18
        sun = obj('Surface Discard Sun')
        sun.transform.euler_angles = Vector3(0,45,0)
        light = sun.add_component('Light')
        light.shadows, light.intensity = LightShadows.Hard, 1.
        receiver = obj('Surface Discard Receiver', scale=(6,4,1)).add_component('MeshRenderer')
        receiver.set_mesh_asset_guid(self.full_mesh)
        self.receiver_material = Material.create_lit()
        self.receiver_material.frag_shader_name = 'Surface Discard Receiver'
        receiver.set_material(0, self.receiver_material.native)
        receiver.enabled = not self.depth
        self.caster = obj('Surface Discard Caster', (-.5,0,-1.5)).add_component('MeshRenderer')
        self.caster.set_mesh_asset_guid(self.full_mesh)
        self.material = Material.create_unlit()
        self.material.frag_shader_name = 'Surface Discard Caster'
        self.material.set_texture('maskTexture', mask)
        self.caster.set_material(0, self.material.native)
        self.pipeline = {'forward':DefaultForwardPipeline, 'forward_plus':DefaultForwardPlusPipeline,
                         'deferred':DefaultDeferredPipeline, 'depth':DepthCoveragePipeline}[pipeline]()
        if not self.depth:
            self.pipeline.msaa_samples = samples
            self.pipeline.shadow_resolution = 1024
        self.change('baseline')

    def change(self, step):
        assert step in STEPS
        self.caster.enabled = step != 'baseline'
        self.caster.set_mesh_asset_guid(self.half_mesh if step == 'reference' else self.full_mesh)
        self.material.set_float('cutoff', 0. if step in ('solid', 'reference') else .5)
        self.material.set_float('alphaValue', -.2 if step in ('negative_off', 'clipped') else 1.)
        self.material.alpha_clip_threshold = .5
        self.material.alpha_clip_enabled = step in ('masked_on', 'clipped')
        self.step = step

    def observe(self, pixels):
        pixels = pixels.copy().astype(np.float32)
        self.images[self.step] = pixels
        if self.depth:
            np.testing.assert_array_equal(pixels[...,0] > .5, pixels[...,1] > .5)
            assert not np.any(pixels[...,2] > .01), self.step
        if self.step != 'restored':
            return dict(step=self.step)
        images = self.images
        visible = images['masked_on'][...,0] > .95
        if not self.depth:
            visible &= (images['masked_on'][...,1] < .5) & (images['masked_on'][...,2] > .95)
        assert visible.sum() > 100, 'Masked caster not visible'
        solid_visible = images['solid'][...,0] > .95
        if not self.depth:
            solid_visible &= (images['solid'][...,1] < .5) & (images['solid'][...,2] > .95)
        assert solid_visible.sum() > visible.sum() * 1.7, 'Mask does not cut the caster'
        if not self.depth:
            shadow = (images['baseline'][...,0] - images['masked_on'][...,0] > .08) & ~visible
            assert shadow.sum() > 50, 'Separated receiver shadow is missing'
        else:
            shadow = visible
        for name in ('masked_off', 'negative_off', 'restored'):
            np.testing.assert_allclose(images[name][...,:3], images['masked_on'][...,:3], atol=.005, err_msg=name)
        reference_pixels = np.ones(visible.shape, dtype=bool)
        if self.samples > 1:
            # Geometry edges have per-sample MSAA coverage; fragment discard
            # applies to the pixel's covered samples. Compare their interiors
            # and the entire separated shadow, excluding just the caster's
            # one-pixel silhouette where these coverage rules differ.
            reference_visible = (images['reference'][...,0] > .95) & (images['reference'][...,1] < .5)
            coverage = visible | reference_visible
            neighbors = [np.roll(coverage, (y, x), axis=(0, 1))
                         for y in (-1, 0, 1) for x in (-1, 0, 1)]
            boundary = np.logical_or.reduce(neighbors) != np.logical_and.reduce(neighbors)
            reference_pixels &= ~boundary
        np.testing.assert_allclose(images['reference'][reference_pixels,:3], images['masked_on'][reference_pixels,:3],
                                   atol=.005, err_msg='CPU geometry reference')
        np.testing.assert_allclose(images['clipped'][...,:3], images['baseline'][...,:3], atol=.005, err_msg='Unified alpha clip')
        return dict(step=self.step, visible=int(visible.sum()), solid=int(solid_visible.sum()), shadow=int(shadow.sum()))

    def close(self):
        for obj in self.objects:
            self.scene.destroy_game_object(obj)
        self.pipeline.dispose()
        for path in self.assets:
            assert self.database.delete_asset(str(path))
        for obj, active in self.previous_roots:
            obj.active = active
