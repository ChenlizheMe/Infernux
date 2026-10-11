"""Public authored-line/baked-geometry bounds contract, reusable by editor probes."""
import numpy as np

from infernux.graph.ramp import AnimationCurve, Keyframe
from infernux.lib import LineAlignment, LineTextureMode, Vector3


CURVES = {
    'constant': ((0, 4, 0, 0), (1, 4, 0, 0)),
    'broad': ((0, 0, 0, 0), (.5, 4, 0, 0), (1, 0, 0, 0)),
    'narrow': ((0, 0, 0, 0), (.5, 0, 0, 0), (.505, 4, 0, 0), (.51, 0, 0, 0), (1, 0, 0, 0)),
    'overshoot': ((0, 0, 0, 0), (.5, 0, 0, 2000), (.51, 0, -2000, 0), (1, 0, 0, 0)),
    'repeat': ((.5, 0, 0, 0), (.505, 4, 0, 0), (.51, 0, 0, 0)),
    'ping_pong': ((.5, 0, 0, 0), (.505, 4, 0, 0), (.51, 0, 0, 0)),
    'single': ((.505, 3, 0, 0),),
    'zero': ((0, 0, 0, -100), (1, 0, 100, 0)),
}


def verify_line_bounds(scene, style, space, shape):
    owner = scene.create_game_object('LineBoundsContract')
    target = scene.create_game_object('LineBoundsBake')
    try:
        line = owner.add_component('LineRenderer')
        mesh = target.add_component('MeshRenderer')
        line.alignment = LineAlignment.TransformZ
        line.use_world_space = space == 'world'
        line.texture_mode = LineTextureMode.Tile
        line.width_multiplier = 1.75
        if space == 'affine':
            owner.transform.position = Vector3(3, -2, 1)
            owner.transform.euler_angles = Vector3(23, 37, -12)
            owner.transform.local_scale = Vector3(-2, .75, 3)
        line.set_positions([(0, 0, 0), (.505, 0, 0), (1, 0, 0)])
        if shape == 'rounded':
            line.num_corner_vertices = 4
            line.num_cap_vertices = 6
        elif shape == 'loop':
            line.loop = True
        wrap = style if style in ('repeat', 'ping_pong') else 'clamp'
        line.width_curve = AnimationCurve(tuple(Keyframe(*key) for key in CURVES[style]), wrap, wrap)

        def observe():
            line.bake_mesh(mesh, use_transform=True)
            vertices = np.asarray(mesh._cpp_component.get_positions(), dtype=float)
            bounds = np.asarray(line._cpp_component.get_world_bounds(), dtype=float)
            assert vertices.shape[0] >= 6
            assert np.isfinite(vertices).all() and np.isfinite(bounds).all()
            assert np.all(vertices >= bounds[:3] - 1e-4), (style, space, shape, vertices.min(axis=0), bounds)
            assert np.all(vertices <= bounds[3:] + 1e-4), (style, space, shape, vertices.max(axis=0), bounds)
            return bounds, vertices

        bounds, vertices = observe()
        if style == 'narrow' and shape == 'plain' and space != 'affine':
            assert np.isclose(vertices[:, 1].max(), 3.5, atol=1e-4)
        # Rebuilding positions/multiplier must also publish a fresh width bound.
        line.width_multiplier = .5
        line.set_position(1, (.5025, 0, 0))
        reduced, _ = observe()
        line.width_multiplier = 0.
        cleared, _ = observe()
        return dict(style=style, space=space, shape=shape, bounds=bounds.tolist(),
                    reduced=reduced.tolist(), cleared=cleared.tolist(), vertices=len(vertices))
    finally:
        scene.destroy_game_object(owner)
        scene.destroy_game_object(target)
        scene.process_pending_destroys()
