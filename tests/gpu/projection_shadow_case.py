"""Actual shadow pixels must depend only on the camera's effective matrices."""
import numpy as np

from infernux.lib import CameraProjection, LightShadows
from tests.gpu.additive_shadow_case import AdditiveShadowCase


class ProjectionShadowCase:
    phases = ('unshadowed', 'shadow', 'dormant_parameters_changed', 'near_far_changed_again', 'restored')

    def __init__(self, scene, path, kind):
        self.fixture = AdditiveShadowCase(scene, path)
        self.pipeline = self.fixture.pipeline
        self.camera = camera = self.fixture.camera
        camera.field_of_view = 60.
        camera.orthographic_size = 3.
        camera.projection_mode = CameraProjection.Orthographic if kind == 'orthographic' else CameraProjection.Perspective
        projection = np.asarray(camera.projection_matrix).copy()
        # The public component derives aspect from its viewport. This explicit
        # matrix override uses the fixed 240 x 180 readback target from frame 0.
        projection[0, 0] = -projection[1, 1] * .75
        if kind == 'asymmetric':
            projection[0, 2] += .3
            projection[1, 2] -= .1
        elif kind == 'oblique':
            projection = camera.calculate_oblique_matrix((.3, .1, 1., -1.))
        elif kind == 'infinite':
            projection[2, 2] = 1.
            projection[2, 3] = -.1
        elif kind == 'affine':
            affine = np.eye(4, dtype=np.float32)
            affine[0, 0], affine[0, 1], affine[2, 2] = .9, .1, 1.5
            camera.view_matrix = affine @ np.asarray(camera.view_matrix)
        self.projection = np.asarray(projection).copy()
        camera.projection_matrix = self.projection
        self.kind, self.path = kind, path
        self.images = {}
        self.change(0)

    def change(self, phase):
        self.phase = phase
        if phase == 1:
            self.fixture.light.shadows = LightShadows.Hard
        elif phase == 2:
            self.camera.set_clip_planes(.01, 5000.)
            self.camera.field_of_view = 20.
            self.camera.orthographic_size = 20.
            self.camera.projection_mode = CameraProjection.Orthographic
        elif phase == 3:
            self.camera.set_clip_planes(10., 11.)
        elif phase == 4:
            self.camera.set_clip_planes(.1, 100.)
            self.camera.field_of_view = 60.
            self.camera.orthographic_size = 3.
            self.camera.projection_mode = CameraProjection.Orthographic if self.kind == 'orthographic' else CameraProjection.Perspective
        np.testing.assert_array_equal(self.camera.projection_matrix, self.projection)

    def observe(self, pixels):
        pixels = np.asarray(pixels, dtype=np.float32).copy()
        assert np.isfinite(pixels).all()
        name = self.phases[self.phase]
        self.images[name] = pixels
        if self.phase == 0:
            visible = (pixels[..., 0] > .5) & (pixels[..., 1] < .15) & (pixels[..., 2] > .5)
            assert visible.sum() > 100, 'Fixture caster must be visible'
        elif self.phase == 1:
            self.shadow = self.images['unshadowed'][..., 0] - pixels[..., 0] > .08
            assert self.shadow.sum() > 100, 'Fixture must render a separated shadow'
        else:
            np.testing.assert_allclose(pixels, self.images['shadow'], atol=.005,
                                       err_msg=f'{self.path}/{self.kind}: dormant camera parameters changed actual shadows')
        return dict(pipeline=self.path, projection=self.kind, phase=name, size=list(pixels.shape),
                    red_mean=float(pixels[..., 0].mean()),
                    shadow_pixels=int(self.shadow.sum()) if self.phase else 0)

    def close(self):
        self.fixture.close()
