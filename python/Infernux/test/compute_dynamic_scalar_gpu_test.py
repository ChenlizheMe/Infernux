"""Real Vulkan regression for changing scalar arguments on a resident buffer."""
from pathlib import Path
import math
import tempfile

import numpy as np

import Infernux as inx
from Infernux import Engine


@inx.compute.kernel
def stamp(domain, vertices, start_x, start_z, end_x, end_z, radius, depth):
    i = inx.compute.index(domain)
    x = vertices[i, 0]
    z = vertices[i, 2]
    segment_x = end_x - start_x
    segment_z = end_z - start_z
    segment_length_sq = segment_x * segment_x + segment_z * segment_z
    t = 0.0
    if segment_length_sq > 1.0e-8:
        t = ((x - start_x) * segment_x + (z - start_z) * segment_z) / segment_length_sq
    t = max(0.0, min(1.0, t))
    nearest_x = start_x + segment_x * t
    nearest_z = start_z + segment_z * t
    offset_x = x - nearest_x
    offset_z = z - nearest_z
    distance_sq = offset_x * offset_x + offset_z * offset_z
    if distance_sq < radius * radius:
        falloff = 1.0 - math.sqrt(distance_sq) / radius
        target_height = -depth * falloff * falloff
        if vertices[i, 1] > target_height:
            vertices[i, 1] = target_height


def main():
    with tempfile.TemporaryDirectory(prefix="infernux-dynamic-scalars-") as root:
        project = Path(root)
        (project / "Assets").mkdir()
        (project / "ProjectSettings").mkdir()
        frontend = Engine()
        engine = frontend.get_native_engine()
        domain = None
        vertices = None
        try:
            frontend.init_renderer(64, 64, str(project))
            resolution = 128
            axis = np.linspace(-3.0, 3.0, resolution, dtype=np.float32)
            grid_x, grid_z = np.meshgrid(axis, axis)
            source = np.zeros((resolution * resolution, 23), dtype=np.float32)
            source[:, 0] = grid_x.reshape(-1)
            source[:, 2] = grid_z.reshape(-1)
            source[:, 4] = 1.0
            source[:, 6] = 1.0
            source[:, 9] = 1.0
            source[:, 10:13] = 1.0
            row = np.arange(resolution - 1, dtype=np.int32)[:, None] * resolution
            column = np.arange(resolution - 1, dtype=np.int32)[None, :]
            a = (row + column).reshape(-1)
            b = a + 1
            c = a + resolution
            d = c + 1
            indices = np.stack((a, c, b, b, c, d), axis=1).reshape(-1)
            domain = inx.buffer(shape=len(source), dtype=np.int32, device="gpu")
            vertices = inx.buffer(shape=source.shape, dtype=np.float32, device="gpu", data=source)
            inx.compute._enable_automatic_mesh_attributes(
                vertices, source[:, :3], indices, normals=True, tangents=True
            )
            previous_x = previous_z = 0.0
            for frame in range(256):
                time = frame / 120.0
                x = 2.1 * math.sin(time * 0.55)
                z = 1.7 * math.sin(time * 0.83)
                inx.compute.launch(
                    stamp,
                    params=(domain, vertices, previous_x, previous_z, x, z, 0.42, 0.24),
                )
                previous_x, previous_z = x, z
            result = vertices.get_data().numpy(copy=False)
            np.testing.assert_array_equal(result[:, 0], source[:, 0])
            np.testing.assert_array_equal(result[:, 2], source[:, 2])
            assert bool(np.isfinite(result).all())
            assert float(result[:, 1].max()) <= 1.0e-7
            assert float(result[:, 1].min()) >= -0.240001
        finally:
            if vertices is not None:
                vertices.close()
            if domain is not None:
                domain.close()
            inx.compute._release_engine_resources()
            engine.cleanup()
    print("INFERNUX_COMPUTE_DYNAMIC_SCALARS_OK")


if __name__ == "__main__":
    main()
