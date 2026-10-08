"""Shared desktop/pytest fixture for resident-scene shadow rendering."""
import numpy as np

from infernux.lib import CameraClearFlags, InxMaterial, LightShadows, SceneManager, Vector3, vec4f
from infernux.renderstack import DefaultDeferredPipeline, DefaultForwardPipeline, DefaultForwardPlusPipeline


class AdditiveShadowCase:
    phases = ('unshadowed', 'shadow_light_scene', 'shadow_receiver_scene',
              'shadows_off', 'restored', 'disabled', 'enabled', 'unloaded')

    def __init__(self, scene, path):
        self.scene = scene
        self.manager = SceneManager.instance()
        self.previous_active = self.manager.get_active_scene()
        self.previous_camera = scene.main_camera
        self.roots = [(obj, obj.active_self) for index in range(self.manager.scene_count)
                      for obj in self.manager.get_scene_at(index).get_root_objects()]
        for obj, _ in self.roots:
            obj.active = False
        self.objects = []
        self.images = {}
        self.pipeline = {'forward': DefaultForwardPipeline, 'forward_plus': DefaultForwardPlusPipeline,
                         'deferred': DefaultDeferredPipeline}[path]()
        self.pipeline.msaa_samples = 1
        self.pipeline.shadow_resolution = 1024
        self.path = path
        camera_object = self.create_object('Additive Shadow Camera')
        camera_object.transform.position = Vector3(0, 0, -6)
        camera = camera_object.add_component('Camera')
        camera.set_clip_planes(.1, 100.)
        camera.clear_flags = CameraClearFlags.SolidColor
        camera.background_color = vec4f(0, 0, 0, 1)
        scene.main_camera = camera
        receiver = self.create_quad('Receiver', (0, 0, 0), (6, 4, 1))
        material = InxMaterial.create_default_lit()
        material.set_color('baseColor', .5, .5, .5, 1.)
        material.set_float('smoothness', 0.)
        receiver.set_material(0, material)
        caster = self.create_quad('Caster', (-.5, 0, -1.5), (1, 1, 1))
        material = InxMaterial.create_default_unlit()
        material.set_color('baseColor', 1., 0., 1., 1.)
        caster.set_material(0, material)
        self.light_scene = self.manager.create_scene('Additive Shadow Light')
        light_object = self.light_scene.create_game_object('Sun')
        light_object.transform.euler_angles = Vector3(0, 45, 0)
        self.light = light_object.add_component('Light')
        self.light.intensity = 1.
        self.change(0)

    def create_object(self, name):
        obj = self.scene.create_game_object(name)
        self.objects.append(obj)
        return obj

    def create_quad(self, name, position, scale):
        obj = self.create_object(name)
        obj.transform.position = Vector3(*position)
        obj.transform.local_scale = Vector3(*scale)
        renderer = obj.add_component('MeshRenderer')
        renderer.set_inline_mesh_data(
            np.array([[-.5, -.5, 0], [.5, -.5, 0], [.5, .5, 0], [-.5, .5, 0]], dtype=np.float32),
            np.array([[0, 0, -1]] * 4, dtype=np.float32),
            np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32),
            np.array([0, 2, 1, 0, 3, 2], dtype=np.uint32), name)
        return renderer

    def change(self, phase):
        self.phase = phase
        if phase == 0:
            self.manager.set_active_scene(self.light_scene)
            self.light.shadows = LightShadows.NoShadows
        elif phase in (1, 4):
            self.light.shadows = LightShadows.Hard
        elif phase == 2:
            self.manager.set_active_scene(self.scene)
        elif phase == 3:
            self.light.shadows = LightShadows.NoShadows
        elif phase == 5:
            self.light.enabled = False
        elif phase == 6:
            self.light.enabled = True
        elif phase == 7:
            self.manager.unload_scene(self.light_scene)
            self.light_scene = None
            self.light = None

    def observe(self, pixels):
        name = self.phases[self.phase]
        pixels = np.asarray(pixels, dtype=np.float32).copy()
        assert np.isfinite(pixels).all()
        self.images[name] = pixels
        if self.phase == 0:
            visible = (pixels[..., 0] > .5) & (pixels[..., 1] < .15) & (pixels[..., 2] > .5)
            assert visible.sum() > 100, 'Fixture caster must be visible'
        elif self.phase == 1:
            baseline = self.images['unshadowed']
            self.shadow = baseline[..., 0] - pixels[..., 0] > .08
            assert self.shadow.sum() > 100, 'Fixture must render a separated shadow'
        elif self.phase in (2, 4, 6):
            np.testing.assert_allclose(pixels, self.images['shadow_light_scene'], atol=.005,
                                       err_msg=f'{self.path}: {name} lost or changed a resident shadow')
        elif self.phase == 3:
            np.testing.assert_allclose(pixels, self.images['unshadowed'], atol=.005)
        elif self.phase == 5:
            assert (self.images['unshadowed'][..., 0] - pixels[..., 0] > .08).sum() > self.shadow.sum()
        elif self.phase == 7:
            np.testing.assert_allclose(pixels, self.images['disabled'], atol=.005,
                                       err_msg='Unloaded light still affects the render world')
        return dict(pipeline=self.path, phase=name, size=list(pixels.shape),
                    red_mean=float(pixels[..., 0].mean()),
                    shadow_pixels=int(self.shadow.sum()) if self.phase else 0)

    def close(self):
        self.manager.set_active_scene(self.previous_active)
        if self.light_scene is not None:
            self.manager.unload_scene(self.light_scene)
            self.light_scene = None
        self.scene.main_camera = self.previous_camera
        for obj in self.objects:
            self.scene.destroy_game_object(obj)
        self.scene.process_pending_destroys()
        for obj, active in self.roots:
            obj.active = active
        self.pipeline.dispose()
