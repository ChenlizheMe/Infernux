"""Compare built-in mesh tangent frames to their actual UV parameterization."""
import numpy as np

from infernux.lib import PrimitiveType


PRIMITIVES = ('Cube', 'Quad', 'Plane', 'Cylinder', 'Sphere', 'Capsule')


def verify_primitive_tangents(scene, name):
    owner = scene.create_primitive(getattr(PrimitiveType, name), 'TangentContract' + name)
    try:
        renderer = owner.get_component('MeshRenderer')._cpp_component
        positions = np.asarray(renderer.get_positions(), dtype=float)
        normals = np.asarray(renderer.get_normals(), dtype=float)
        tangents = np.asarray(renderer.get_tangents(), dtype=float)
        uv = np.asarray(renderer.get_uvs(), dtype=float)
        triangles = np.asarray(renderer.get_indices(), dtype=int).reshape(-1, 3)
        assert len(positions) and len(triangles)
        assert np.isfinite(tangents).all() and np.isfinite(normals).all()
        np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1., atol=1e-5)
        np.testing.assert_allclose(np.linalg.norm(tangents[:, :3], axis=1), 1., atol=1e-5)
        np.testing.assert_allclose(np.sum(normals*tangents[:, :3], axis=1), 0., atol=1e-5)
        np.testing.assert_array_equal(np.abs(tangents[:, 3]), 1.)
        # Longitude has no unique derivative at a pole; exclude the whole fan
        # triangle from this finite-difference oracle, but check its basis above.
        polar = np.zeros(len(triangles), dtype=bool)
        if name in ('Sphere', 'Capsule'):
            polar = (np.abs(normals[triangles, 1]) > .999999).any(axis=1)
        edges = positions[triangles[:, 1:]] - positions[triangles[:, :1]]
        delta = uv[triangles[:, 1:]] - uv[triangles[:, :1]]
        determinant = delta[:, 0, 0]*delta[:, 1, 1] - delta[:, 0, 1]*delta[:, 1, 0]
        valid = (~polar) & (np.abs(determinant) > 1e-8) & (np.linalg.norm(np.cross(edges[:, 0], edges[:, 1]), axis=1) > 1e-8)
        indices, edges, delta, determinant = triangles[valid], edges[valid], delta[valid], determinant[valid]
        assert len(indices)
        u = (edges[:, 0]*delta[:, 1, 1, None] - edges[:, 1]*delta[:, 0, 1, None]) / determinant[:, None]
        v = (edges[:, 1]*delta[:, 0, 0, None] - edges[:, 0]*delta[:, 1, 0, None]) / determinant[:, None]
        n, tangent = normals[indices], tangents[indices]
        def project(direction):
            projected = direction[:, None, :] - n*np.sum(n*direction[:, None, :], axis=-1, keepdims=True)
            return projected / np.linalg.norm(projected, axis=-1, keepdims=True)
        u_dot = np.sum(tangent[..., :3]*project(u), axis=-1)
        bitangent = np.cross(n, tangent[..., :3])*tangent[..., 3, None]
        v_dot = np.sum(bitangent*project(v), axis=-1)
        result = dict(name=name, vertices=len(positions), regular_triangles=len(indices),
                      polar_triangles=int(polar.sum()), min_u=float(u_dot.min()), min_v=float(v_dot.min()))
        assert np.isfinite(u_dot).all() and np.isfinite(v_dot).all(), result
        threshold = .99999 if name in ('Plane', 'Quad', 'Cube') else .5
        assert u_dot.min() >= threshold and v_dot.min() >= threshold, result
        return result
    finally:
        scene.destroy_game_object(owner)
        scene.process_pending_destroys()
