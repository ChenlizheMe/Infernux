"""Imported UV0 normals: instance transforms versus independently baked meshes."""
from pathlib import Path

import numpy as np
from PIL import Image

from infernux.core.assets import AssetManager
from infernux.core.asset_types import TextureImportSettings, TextureType, TextureCompression
from infernux.core.material import Material
from infernux.lib import SceneManager, Vector3
from infernux.renderstack import DefaultForwardPipeline, DefaultForwardPlusPipeline, DefaultDeferredPipeline


SCALES = [(1.,1.,1.), (2.,2.,2.), (2.,.5,1.), (-2.,.5,1.), (2.,-.5,1.), (1.,1.,1.)]
STEPS = ('flat', 'instance', 'baked', 'restored')
TANGENT = np.array((.8,.48,.36), dtype=np.float64)
BITANGENT = np.array((-.6,.64,.48), dtype=np.float64)
NORMAL = np.array((0.,.6,-.8), dtype=np.float64)
POSITIONS = np.array([TANGENT*u + BITANGENT*v for u,v in ((-.5,-.5),(.5,-.5),(.5,.5),(-.5,.5))])
SHADER = '''#version 450
ShaderInfo {
    Name "Instance Normal Map"
    ShadingModel PBR
    Cull Off
    Properties { Texture2D normalMap = normal Float normalScale = 1.0 }
}
void surface(out SurfaceData s) {
    s = InitSurfaceData();
    s.albedo = vec3(0.65,0.4,0.2);
    s.smoothness = 0.3;
    s.normalWS = sampleNormal(normalMap, getUV(0), 0, material.normalScale);
}
'''


def source_obj(mirror_uv):
    uv = [(0,0),(1,0),(1,1),(0,1)]
    if mirror_uv:
        uv = [(u,1-v) for u,v in uv]
    faces = 'f 1/1/1 3/3/1 2/2/1\nf 1/1/1 4/4/1 3/3/1\n'
    return ('\n'.join('v '+' '.join(f'{x:.12f}' for x in point) for point in POSITIONS) + '\n' +
            '\n'.join(f'vt {u} {v}' for u,v in uv) + '\nvn '+' '.join(f'{x:.12f}' for x in NORMAL) + '\n' +
            faces)


def reference_basis(scale, mirror_uv):
    """Independent double-precision geometry oracle, including UV orientation."""
    scale = np.asarray(scale, dtype=np.float64)
    tangent = TANGENT * scale
    tangent /= np.linalg.norm(tangent)
    normal = NORMAL / scale
    normal /= np.linalg.norm(normal)
    uv_derivative = BITANGENT * scale * (1. if mirror_uv else -1.)
    handedness = np.sign(np.dot(np.cross(normal, tangent), uv_derivative))
    return normal, np.append(tangent, handedness)


class MeshNormalMapCase:
    def __init__(self, native, scene, directory, pipeline='forward', samples=1, mirror_uv=False):
        self.native, self.scene = native, scene
        self.database = native.get_asset_database()
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.assets, self.objects, self.images, self.results = [], [], {}, []
        manager = SceneManager.instance()
        self.previous_roots = [(obj, obj.active_self) for index in range(manager.scene_count)
                               for obj in manager.get_scene_at(index).get_root_objects()]
        for obj, _ in self.previous_roots:
            obj.active = False
        def asset(name, text=None):
            path = self.directory / name
            if text is not None:
                path.write_text(text, encoding='utf-8')
            result = AssetManager.import_asset(str(path), database=self.database)
            assert result.succeeded, result.error
            self.assets.append(path)
            return result.guid
        self.mesh = asset('Geometry.obj', source_obj(mirror_uv))
        self.mirror_uv = mirror_uv
        asset('NormalMap.frag', SHADER)
        y, x = np.indices((32,32))
        texture = np.empty((32,32,4), dtype=np.uint8)
        texture[...,0] = np.where(x < 16, 175, 90)
        texture[...,1] = np.where(y < 16, 190, 75)
        texture[...,2:] = 255
        normal_path = self.directory / 'Normal.png'
        Image.fromarray(texture).save(normal_path)
        normal_guid = asset('Normal.png')
        settings = TextureImportSettings(texture_type=TextureType.NORMAL_MAP,
            compression=TextureCompression.NONE, generate_mipmaps=False)
        assert AssetManager.apply_import_settings('texture', str(normal_path), settings)
        def owner(name, position=(0,0,0)):
            obj = scene.create_game_object(name)
            obj.layer = 18
            obj.transform.position = Vector3(*position)
            self.objects.append(obj)
            return obj
        self.camera = owner('Normal Map Camera',(0,0,-5)).add_component('Camera')
        self.camera.culling_mask = 1 << 18
        sun = owner('Normal Map Sun')
        sun.transform.euler_angles = Vector3(15,-25,0)
        sun.add_component('Light').intensity = 1.5
        self.owner = owner('Normal Mapped Mesh')
        self.renderer = self.owner.add_component('MeshRenderer')
        self.material = Material.create_lit()
        self.material.frag_shader_name = 'Instance Normal Map'
        self.material.set_texture('normalMap', normal_guid)
        self.renderer.set_material(0, self.material.native)
        self.pipeline = {'forward':DefaultForwardPipeline, 'forward_plus':DefaultForwardPlusPipeline,
                         'deferred':DefaultDeferredPipeline}[pipeline]()
        self.pipeline.msaa_samples = samples
        self.phases = len(SCALES)*len(STEPS)
        self.change(0)

    def change(self, phase):
        assert 0 <= phase < self.phases
        self.phase = phase
        scale, step = SCALES[phase//4], STEPS[phase%4]
        baked = step == 'baked'
        self.owner.transform.local_scale = Vector3(*(SCALES[0] if baked else scale))
        if baked:
            normal, tangent = reference_basis(scale, self.mirror_uv)
            uv = np.array([(0,1),(1,1),(1,0),(0,0)], dtype=np.float32)
            if self.mirror_uv:
                uv[:,1] = 1. - uv[:,1]
            # Explicit reference basis avoids regenerating MikkTSpace from a
            # reflected mesh whose unchanged winding opposes its supplied normal.
            # The tested object still uses an imported OBJ and generated UV0 basis.
            self.renderer.set_inline_mesh_data(
                np.asarray(POSITIONS * scale, dtype=np.float32),
                np.tile(normal, (4,1)).astype(np.float32), uv,
                np.array([0,2,1,0,3,2], dtype=np.uint32), 'CPU normal-map reference',
                tangents=np.tile(tangent, (4,1)).astype(np.float32))
        else:
            self.renderer.set_mesh_asset_guid(self.mesh)
        self.material.set_float('normalScale', 0. if step == 'flat' else 1.)

    def observe(self, pixels):
        pixels = pixels.copy().astype(np.float32)
        scale, step = SCALES[self.phase//4], STEPS[self.phase%4]
        self.images[step] = pixels
        result = dict(phase=self.phase, scale=scale, step=step)
        if step == 'restored':
            np.testing.assert_allclose(self.images['instance'], self.images['baked'], atol=.005,
                                       err_msg='CPU-baked geometry and explicit double-precision UV0 basis')
            np.testing.assert_array_equal(self.images['instance'], pixels)
            changed = np.abs(self.images['flat'][...,:3] - pixels[...,:3]).max(axis=-1) > .02
            assert changed.sum() > 100, 'Normal texture did not affect surface lighting'
            result['mapped_pixels'] = int(changed.sum())
            result['reference_max_error'] = float(np.abs(self.images['instance']-self.images['baked']).max())
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
