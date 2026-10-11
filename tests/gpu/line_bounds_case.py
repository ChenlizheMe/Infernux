"""A ribbon whose centerline is outside the frustum still contributes visible width."""
import numpy as np

from infernux.graph.ramp import AnimationCurve, Keyframe
from infernux.lib import CameraClearFlags, CameraProjection, LineAlignment, SceneManager, Vector3, vec4f
from infernux.renderstack import RenderStackPipeline
from tests.acceptance.line_bounds import CURVES


class LineBoundsCase:
    phases = ('constant', 'narrow', 'overshoot', 'zero')

    def __init__(self, native, scene):
        self.native, self.scene = native, scene
        self.previous_camera = scene.main_camera
        manager = SceneManager.instance()
        self.roots = [(obj, obj.active_self) for i in range(manager.scene_count)
                      for obj in manager.get_scene_at(i).get_root_objects()]
        for obj, _ in self.roots:
            obj.active = False
        self.camera_object = scene.create_game_object('LineBoundsCamera')
        self.camera_object.transform.position = Vector3(0, 0, -4)
        self.camera = self.camera_object.add_component('Camera')
        self.camera.projection_mode = CameraProjection.Orthographic
        self.camera.orthographic_size = 1.
        self.camera.set_clip_planes(.1, 10.)
        self.camera.clear_flags = CameraClearFlags.SolidColor
        self.camera.background_color = vec4f(0, 0, 0, 1)
        scene.main_camera = self.camera
        self.line_object = scene.create_game_object('CulledCenterVisibleRibbon')
        self.line = self.line_object.add_component('LineRenderer')
        self.line.alignment = LineAlignment.TransformZ
        self.line.set_positions([(-.5, 1.25, 0), (.005, 1.25, 0), (.5, 1.25, 0)])
        self.line.start_color = self.line.end_color = [1., 0., 0., 1.]
        self.pipeline = RenderStackPipeline()
        self.change(0)

    def change(self, phase):
        self.phase = phase
        self.line.width_curve = AnimationCurve(tuple(Keyframe(*key) for key in CURVES[self.phases[phase]]))

    def observe(self, pixels):
        pixels = np.asarray(pixels)
        height, width = pixels.shape[:2]
        clip = self.camera.projection_matrix @ self.camera.view_matrix @ np.array([0., 1.25, 0., 1.])
        assert abs(clip[1] / clip[3]) > 1., clip
        sample = pixels[height//2-1:height//2+2, width//2-1:width//2+2, :3].astype(float)
        if self.phases[self.phase] == 'zero':
            assert np.max(np.abs(sample)) < .01, sample
        else:
            assert ((sample[..., 0] > .25) & (sample[..., 0] > 3*sample[..., 1])).all(), sample
        return dict(phase=self.phases[self.phase], center_clip=clip.tolist(),
                    color=sample.mean(axis=(0, 1)).tolist(),
                    bounds=list(self.line._cpp_component.get_world_bounds()), size=[width, height])

    def close(self):
        self.scene.main_camera = self.previous_camera
        self.scene.destroy_game_object(self.line_object)
        self.scene.destroy_game_object(self.camera_object)
        self.scene.process_pending_destroys()
        for obj, active in self.roots:
            obj.active = active
        self.pipeline.dispose()
