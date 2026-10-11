"""Verify rendered billboard pixels and public runtime pointer dispatch agree."""
import numpy as np

from infernux.engine.runtime_event_queue import drain
from infernux.engine.runtime_screen_ui import collect_runtime_ui_input_surfaces, map_runtime_ui_pointer
from infernux.lib import SceneManager, Vector3
from infernux.renderstack import RenderStackPipeline
from infernux.ui import UIButton
from infernux.ui.ui_event_system import UIEventProcessor


class WorldUIAffineCase:
    phases = (('rigid', False), ('scaled', False), ('sheared', False),
              ('mirrored', False), ('scaled', True), ('sheared', True))

    def __init__(self, native, scene):
        self.native, self.scene = native, scene
        self.previous_camera = scene.main_camera
        manager = SceneManager.instance()
        self.roots = [(obj, obj.active_self) for i in range(manager.scene_count)
                      for obj in manager.get_scene_at(i).get_root_objects()]
        for obj, _ in self.roots:
            obj.active = False
        self.camera_object = scene.create_game_object('Affine223Camera')
        self.camera = self.camera_object.add_component('Camera')
        self.camera.set_clip_planes(.1, 100.)
        scene.main_camera = self.camera
        self.button_object = scene.create_game_object('Affine223Button')
        self.anchor = np.array([0., 0., 4.])
        self.button_object.transform.position = Vector3(*self.anchor)
        self.button = self.button_object.add_py_component(UIButton())
        self.button.label = ''
        self.button.background_color = [1., 0., 0., 1.]
        self.button.width, self.button.height = 160., 80.
        self.button.world_billboard = True
        self.button.world_always_on_top = True
        self.clicks = 0
        self.button.on_click.add_listener(self._clicked)
        self.processor = UIEventProcessor()
        self.pipeline = RenderStackPipeline()
        self.change(0)

    def _clicked(self):
        self.clicks += 1

    def change(self, phase):
        self.phase = phase
        kind, constant = self.phases[phase]
        matrix = np.eye(4)
        if kind in ('scaled', 'mirrored'):
            matrix[0, 0], matrix[1, 1] = (-2. if kind == 'mirrored' else 2.), .75
        elif kind == 'sheared':
            matrix[0, 1], matrix[1, 0] = .7, .2
            matrix[2, 0], matrix[2, 1] = .15, -.1
        self.camera.view_matrix = np.linalg.inv(matrix)
        self.button.world_constant_screen_size = constant

    def observe_and_click(self, pixels):
        pixels = np.asarray(pixels)
        h, w = pixels.shape[:2]
        view, projection = self.camera.view_matrix, self.camera.projection_matrix
        basis = self.camera.camera_to_world_matrix[:3, :2]
        local = np.array([.64, .25])
        clip_w = (projection @ view @ np.append(self.anchor, 1.))[3]
        scale = 200. * clip_w / (abs(projection[1, 1]) * h) if self.button.world_constant_screen_size else 1.
        point = self.anchor + basis @ local * scale
        clip = projection @ view @ np.append(point, 1.)
        screen = (clip[:2] / clip[3] + 1.) * np.array([w, h]) * .5
        x, y = np.floor(screen).astype(int)
        assert 1 <= x < w-1 and 1 <= y < h-1, screen
        sample = pixels[y-1:y+2, x-1:x+2, :3].astype(float)
        assert ((sample[..., 0] > .25) & (sample[..., 0] > sample[..., 1] * 3)).mean() == 1., (screen, sample.mean(axis=(0,1)))
        surfaces = collect_runtime_ui_input_surfaces(self.scene)
        mapped = map_runtime_ui_pointer(surfaces, self.camera, *screen, w, h)
        target_index = next(i for i, target in enumerate(surfaces) if target.element is self.button)
        np.testing.assert_allclose(mapped[target_index][:2], [144., 15.], atol=.01)
        before = self.clicks
        self.processor.process(surfaces, mapped, True, False, True, (0., 0.), .016)
        drain()
        self.processor.process(surfaces, mapped, False, True, False, (0., 0.), .016)
        drain()
        assert self.clicks == before + 1, self.processor._last_pointer_debug
        return dict(phase=self.phases[self.phase], screen=screen.tolist(), size=[w,h],
                    color=sample.mean(axis=(0,1)).tolist(), local=list(mapped[target_index]),clicks=self.clicks)

    def close(self):
        self.scene.main_camera = self.previous_camera
        self.scene.destroy_game_object(self.button_object)
        self.scene.destroy_game_object(self.camera_object)
        self.scene.process_pending_destroys()
        for obj, active in self.roots:
            obj.active = active
        self.native.get_screen_ui_renderer().begin_frame(640, 480)
        self.pipeline.dispose()
