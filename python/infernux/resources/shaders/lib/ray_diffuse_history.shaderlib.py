import infernux.shader as shader
from infernux.shader.lib.camera import linear_eye_depth
from infernux.shader.lib.ray_signal import pack_ray_signal, ray_grid_pixel, unpack_ray_signal
from infernux.shader.lib.surface import decode_octahedral_normal


def diffuse_history(history: shader.UInt2D, surfaces: shader.UInt2D, primary: shader.uvec4,
                    history_uv: shader.vec2, previous_uv: shader.vec2, expected_eye: float,
                    extent: shader.ivec2, full: shader.ivec2) -> shader.uvec4:
    # Reproject onto the actual raster representatives of the sparse grid, including odd extents.
    location = history_uv * shader.vec2(full) - shader.vec2(0.5)
    center = shader.ivec2(history_uv * shader.vec2(extent))
    site = ray_grid_pixel(center, extent, full)
    base = center - shader.ivec2(location < shader.vec2(site))
    lo = ray_grid_pixel(shader.max(base, shader.ivec2(0)), extent, full)
    hi = ray_grid_pixel(shader.min(base + shader.ivec2(1), extent - shader.ivec2(1)), extent, full)
    fraction = shader.saturate((location - shader.vec2(lo)) / shader.vec2(shader.max(hi - lo, shader.ivec2(1))))
    fraction = shader.vec2(1.0 if base.x < 0 else fraction.x, 1.0 if base.y < 0 else fraction.y)
    normal = decode_octahedral_normal(shader.unpack_half_2x16(primary.z))
    camera = shader.camera()
    reference = shader.vec3(0.0, 0.0, 1.0) if shader.abs(normal.z) < 0.9 else shader.vec3(0.0, 1.0, 0.0)
    tangent = shader.cross(normal, reference)
    prev = camera.previous_view_projection_matrix
    t = prev @ shader.vec4(tangent, 0.0)
    b = prev @ shader.vec4(shader.cross(normal, tangent), 0.0)
    device = ((camera.projection_params.y - expected_eye) / (camera.projection_params.y - camera.projection_params.x)
              if shader.abs(camera.projection_matrix[3][3]) > 0.5 else
              (1.0 / expected_eye - camera.depth_params.w) / camera.depth_params.z)
    clip = shader.vec3(previous_uv * shader.vec2(2.0, -2.0) + shader.vec2(-1.0, 1.0), device)
    # Device Z is affine over a plane. Project its tangent directions once, then test each tap
    # at its own pixel instead of comparing the depth of two different points on a tilted face.
    plane = shader.cross(t.xyz - clip * t.w, b.xyz - clip * b.w)
    if shader.abs(plane.z) <= 1.0e-10:
        return shader.uvec4(0)
    slope = plane.xy / plane.z
    color = shader.vec3(0.0)
    age = 0.0
    second = 0.0
    total = 0.0
    luma = shader.vec3(0.2126, 0.7152, 0.0722)
    for tap in range(4):
        offset = shader.ivec2(tap & 1, tap >> 1)
        cell = base + offset
        blend = shader.vec2(1.0) - shader.abs(fraction - shader.vec2(offset))
        weight = blend.x * blend.y
        if weight <= 0.0 or shader.any(cell < shader.ivec2(0)) or shader.any(cell >= extent):
            continue
        source = ray_grid_pixel(cell, extent, full)
        witness = shader.load(surfaces, source, 0)
        if witness.x != primary.x or shader.dot(normal, decode_octahedral_normal(shader.unpack_half_2x16(witness.z))) < 0.94:
            continue
        sample_uv = (shader.vec2(source) + shader.vec2(0.5)) / shader.vec2(full)
        sample_device = device - shader.dot(slope, (sample_uv - previous_uv) * shader.vec2(2.0, -2.0))
        predicted = linear_eye_depth(sample_device, camera)
        observed = linear_eye_depth(shader.as_float(witness.y), camera)
        if predicted <= 0.0 or shader.abs(predicted - observed) > 0.003 + 0.008 * observed:
            continue
        packet = shader.load(history, cell, 0)
        value = unpack_ray_signal(packet)
        if value.a <= 0.0:
            continue
        luminance = shader.dot(value.rgb, luma)
        second += weight * (shader.as_float(packet.w) + luminance * luminance)
        color += weight * value.rgb
        age += weight * value.a
        total += weight
    if total <= 0.0:
        return shader.uvec4(0)
    color /= total
    luminance = shader.dot(color, luma)
    variance = shader.max(second / total - luminance * luminance, 0.0)
    return pack_ray_signal(shader.vec4(color, shader.max(1.0, shader.round(age))), shader.uvec2(0, shader.as_uint(variance)))
