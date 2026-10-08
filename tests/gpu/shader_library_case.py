"""Authored shader-library calls, rendered on an imported orthographic quad."""
from pathlib import Path
import math

import numpy as np

from infernux.core.assets import AssetManager
from infernux.core.material import Material
from infernux.lib import CameraProjection, SceneManager, Vector3
from infernux.rendergraph import Format
from infernux.renderstack import RenderPipeline
from tests.gpu.surface_discard_case import quad


ROTATIONS = [
    ((1,0,0), (0,0,1), 0.),
    ((1,0,0), (0,0,1), math.pi/2),
    ((1,0,0), (0,0,1), -math.pi/2),
    ((0,1,0), (1,0,0), math.pi/2),
    ((0,0,1), (0,1,0), math.pi/2),
    ((2,-3,.5), (1,2,-3), .73),
    ((2,-3,.5), (10,20,-30), .73),
    ((2,-3,.5), (1,2,-3), -.73),
    ((1,0,0), (0,0,1), math.pi),
    ((1,0,0), (0,0,1), 2*math.pi),
]
CROSSES = [((.5,.5),(.25,.25),.05), ((.37,.61),(.31,.19),.037),
           ((.15,.8),(.27,.15),.063), ((.5,.5),(.13,.36),.025)]
SHADER = '''#version 450
ShaderInfo {
    Name "Shader Library Geometry"
    ShadingModel Unlit
    Cull Off
    Imports ["Lib Common", "Lib Vertex Utils", "Lib Shapes"]
    Properties {
        Float mode = 0.0
        Float3 value = [1,0,0]
        Float3 axis = [0,0,1]
        Float angle = 0.0
        Float2 center = [0.5,0.5]
        Float2 extent = [0.25,0.25]
        Float thickness = 0.05
    }
}
void surface(out SurfaceData s) {
    s = InitSurfaceData();
    if (material.mode < 1.5) {
        vec3 value = material.mode < 0.5
            ? rotateAboutAxis(material.value, material.axis, material.angle)
            : rotateAroundAxis(material.value, material.axis, material.angle);
        s.albedo = vec3(0.5) + value * 0.125;
    } else {
        vec2 uv = getUV();
        s.albedo = vec3(crossSDF(uv, material.center, material.extent, material.thickness), uv);
    }
}
'''


def quaternion_reference(value, axis, angle):
    """Independent double-precision Hamilton product q * v * conjugate(q)."""
    axis = np.asarray(axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    q = np.r_[math.cos(angle/2), axis * math.sin(angle/2)]
    def multiply(a, b):
        w, x, y, z = a
        s, i, j, k = b
        return np.array([w*s-x*i-y*j-z*k, w*i+x*s+y*k-z*j,
                         w*j-x*k+y*s+z*i, w*k+x*j-y*i+z*s])
    return multiply(multiply(q, np.r_[0., value]), q * [1,-1,-1,-1])[1:]


def linear_from_display(values):
    return np.where(values <= .04045, values/12.92, ((values+.055)/1.055)**2.4)


class LibraryPipeline(RenderPipeline):
    name = 'Shader Library Geometry'

    def define_topology(self, graph):
        graph.set_msaa_samples(1)
        color = graph.create_texture('color', camera_target=True)
        depth = graph.create_texture('depth', format=Format.D32_SFLOAT, samples=1)
        graph.add_pass('Library surface').write_color(color).write_depth(depth).set_clear(
            color=(0,0,0,0), depth=1.).draw_renderers(material_pass='forward')
        graph.screen_ui_section(resources={'color'})
        graph.set_output(color)


class ShaderLibraryCase:
    def __init__(self, native, scene, directory, kind):
        self.native, self.scene, self.kind = native, scene, kind
        self.database = native.get_asset_database()
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.assets, self.objects, self.results = [], [], []
        manager = SceneManager.instance()
        self.previous_roots = [(obj, obj.active_self) for index in range(manager.scene_count)
                               for obj in manager.get_scene_at(index).get_root_objects()]
        for obj, _ in self.previous_roots:
            obj.active = False
        imported = {}
        for name, text in [('Library.frag', SHADER), ('Quad.obj', quad())]:
            path = self.directory / name
            path.write_text(text, encoding='utf-8')
            result = AssetManager.import_asset(str(path), database=self.database)
            assert result.succeeded, result.error
            self.assets.append(path)
            imported[name] = result.guid
        owner = scene.create_game_object('Library Camera')
        self.objects.append(owner)
        owner.transform.position = Vector3(0,0,-3)
        self.camera = owner.add_component('Camera')
        self.camera.projection_mode = CameraProjection.Orthographic
        self.camera.orthographic_size = .5
        self.camera.culling_mask = 1 << 18
        owner = scene.create_game_object('Library Surface')
        self.objects.append(owner)
        owner.layer = 18
        renderer = owner.add_component('MeshRenderer')
        renderer.set_mesh_asset_guid(imported['Quad.obj'])
        self.material = Material.create_unlit()
        self.material.frag_shader_name = 'Shader Library Geometry'
        renderer.set_material(0, self.material.native)
        self.pipeline = LibraryPipeline()
        self.phases = len(ROTATIONS)*2 if kind == 'rotation' else len(CROSSES)
        self.change(0)

    def change(self, phase):
        assert 0 <= phase < self.phases
        self.phase = phase
        if self.kind == 'rotation':
            value, axis, angle = ROTATIONS[phase % len(ROTATIONS)]
            self.material.set_float('mode', phase // len(ROTATIONS))
            self.material.set_vector3('value', *value)
            self.material.set_vector3('axis', *axis)
            self.material.set_float('angle', angle)
        else:
            center, extent, thickness = CROSSES[phase]
            self.material.set_float('mode', 2.)
            self.material.set_vector2('center', *center)
            self.material.set_vector2('extent', *extent)
            self.material.set_float('thickness', thickness)

    def observe(self, pixels):
        pixels = pixels.copy().astype(np.float64)
        assert np.isfinite(pixels).all()
        np.testing.assert_allclose(pixels[...,3], 1., atol=.001)
        linear = linear_from_display(pixels[...,:3])
        if self.kind == 'rotation':
            expected = quaternion_reference(*ROTATIONS[self.phase % len(ROTATIONS)])
            actual = (linear - .5) * 8.
            np.testing.assert_allclose(actual, np.broadcast_to(expected, actual.shape), atol=.015)
            result = dict(phase=self.phase, actual=actual[actual.shape[0]//2,actual.shape[1]//2].tolist(), expected=expected.tolist())
        else:
            center, extent, thickness = CROSSES[self.phase]
            uv = linear[...,1:3]
            assert uv[...,0].min() < .02 and uv[...,0].max() > .98
            assert uv[...,1].min() < .02 and uv[...,1].max() > .98
            d = np.abs(uv - center)
            mask = linear[...,0]
            expected = ((d[...,0] < thickness) & (d[...,1] < extent[1])) | ((d[...,1] < thickness) & (d[...,0] < extent[0]))
            margin = 3. / min(pixels.shape[:2])
            interior = (np.abs(d[...,0]-thickness) > margin) & (np.abs(d[...,1]-thickness) > margin)
            interior &= (np.abs(d[...,0]-extent[0]) > margin) & (np.abs(d[...,1]-extent[1]) > margin)
            assert interior.sum() > pixels.shape[0]*pixels.shape[1]*.5
            # Readback is display-encoded RGBA16F. Undoing sRGB amplifies one
            # half-float step below white into ~0.00111 in linear space.
            # Check binary interior coverage in the sampled attachment space
            # with exactly that representable step, rather than a CPU/GPU
            # gamma roundtrip tolerance. Geometry and edge checks stay below.
            half_step = 1. - float(np.nextafter(np.float16(1.), np.float16(0.)))
            np.testing.assert_allclose(pixels[...,0][interior], expected[interior].astype(float),
                                       atol=half_step, rtol=0.)
            edge = (mask > .001) & (mask < .999)
            assert edge.sum() > 10, 'Antialiasing removed from strip edges'
            result = dict(phase=self.phase, checked=int(interior.sum()), inside=int(expected[interior].sum()), antialiased=int(edge.sum()))
        self.results.append(result)
        return result

    def close(self):
        for obj in self.objects:
            self.scene.destroy_game_object(obj)
        self.pipeline.dispose()
        for path in self.assets:
            path.unlink()
            assert self.database.delete_asset(str(path))
        for obj, active in self.previous_roots:
            obj.active = active
