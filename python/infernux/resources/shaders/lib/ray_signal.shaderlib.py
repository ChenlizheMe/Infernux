import infernux.shader as shader
from infernux.shader.lib.camera import linear_eye_depth
from infernux.shader.lib.surface import decode_octahedral_normal


def unpack_ray_signal(packet: shader.uvec4) -> shader.vec4:
    return shader.vec4(shader.unpack_half_2x16(packet.x), shader.unpack_half_2x16(packet.y))


def pack_ray_signal(signal: shader.vec4, witness: shader.uvec2) -> shader.uvec4:
    return shader.uvec4(shader.pack_half_2x16(signal.rg), shader.pack_half_2x16(signal.ba), witness)


def ray_grid_pixel(pixel: shader.ivec2, source: shader.ivec2, target: shader.ivec2) -> shader.ivec2:
    # Divide positive integer pixel centers. Correct the floating quotient
    # with exact products at the boundary of an odd-sized cell.
    numerator = (2 * pixel + shader.ivec2(1)) * target
    denominator = 2 * source
    quotient = shader.ivec2(numerator / denominator)
    too_high = shader.ivec2(quotient * denominator > numerator)
    too_low = shader.ivec2((quotient + shader.ivec2(1)) * denominator <= numerator)
    return quotient - too_high + too_low


def resolve_reduced_ray(signal: shader.UInt2D, primary: shader.UInt2D, frame: shader.Color2D,
                        pixel: shader.ivec2, full: shader.ivec2, extent: shader.ivec2,
                        surface_id: shader.uint, device_depth: float,
                        normal: shader.vec3, roughness: float) -> shader.vec4:
    if shader.load(primary, pixel, 0).x != surface_id:
        return shader.vec4(0.0)

    center = ray_grid_pixel(pixel, full, extent)
    representative = ray_grid_pixel(center, extent, full)
    if not shader.any(pixel != representative):
        return unpack_ray_signal(shader.load(signal, center, 0))

    first = center - shader.ivec2(pixel < representative)
    camera = shader.camera()
    depth = linear_eye_depth(device_depth, camera)
    radiance = shader.vec3(0.0)
    total = 0.0
    for tap in range(4):
        cell = first + shader.ivec2(tap & 1, tap >> 1)
        if shader.any(cell < shader.ivec2(0)) or shader.any(cell >= extent):
            continue
        source = ray_grid_pixel(cell, extent, full)
        witness = shader.load(primary, source, 0)
        if witness.x != surface_id:
            continue
        packet = shader.load(signal, cell, 0)
        estimate = unpack_ray_signal(packet)
        if estimate.a <= 0.0:
            continue

        sample_frame = shader.load(frame, source, 0)
        sample_normal = decode_octahedral_normal(sample_frame.xy)
        sample_depth = linear_eye_depth(shader.as_float(witness.y), camera)
        depth_band = 0.005 + 0.017 * shader.min(depth, sample_depth)
        depth_weight = shader.saturate(1.0 - shader.abs(depth - sample_depth) / depth_band)
        normal_weight = shader.saturate((shader.dot(normal, sample_normal) - 0.94) / 0.06)
        roughness_weight = shader.saturate(1.0 - shader.abs(roughness - sample_frame.z) / 0.15)
        offset = shader.vec2(pixel - source)
        weight = depth_weight * normal_weight * roughness_weight / (1.0 + shader.dot(offset, offset))
        if weight <= 0.0:
            continue

        # Each tap estimates the diffuse integral over its own hemisphere;
        # different directions can validly hit different surfaces or the sky.
        radiance += weight * estimate.rgb
        total += weight
    if total <= 0.0:
        return shader.vec4(0.0)
    return shader.vec4(radiance / total, 1.0)


def resolve_ray_spatial(signal: shader.UInt2D, primary: shader.UInt2D, frame: shader.Color2D,
                        pixel: shader.ivec2, surface_id: shader.uint, device_depth: float,
                        normal: shader.vec3, roughness: float, specular: bool) -> shader.vec4:
    """One cross of covered neighbors, evaluated only for display lighting."""
    packet = shader.load(signal, pixel, 0)
    center = unpack_ray_signal(packet)
    if center.a <= 0.0:
        return center

    extent = shader.ivec2(shader.dimensions(signal, 0))
    camera = shader.camera()
    depth = linear_eye_depth(device_depth, camera)
    distance = shader.as_float(packet.w)
    depth_scale = 0.008 if specular else 0.017
    normal_min = 0.985 if specular else 0.94
    roughness_band = 0.08 if specular else 0.15
    # Near-mirror signals carry details that a neighboring screen ray cannot represent.
    tap_scale = 0.25 * roughness * roughness if specular else 0.5
    radiance = center.rgb
    total = 1.0
    for tap in range(4):
        axis = shader.ivec2(1, 0) if tap < 2 else shader.ivec2(0, 1)
        sign = -1 if tap % 2 == 0 else 1
        neighbor = pixel + sign * axis
        if shader.any(neighbor < shader.ivec2(0)) or shader.any(neighbor >= extent):
            continue
        witness = shader.load(primary, neighbor, 0)
        if witness.x != surface_id:
            continue
        candidate = shader.load(signal, neighbor, 0)
        estimate = unpack_ray_signal(candidate)
        if estimate.a <= 0.0 or (specular and candidate.z != packet.z):
            continue

        sample_frame = shader.load(frame, neighbor, 0)
        sample_normal = decode_octahedral_normal(sample_frame.xy)
        sample_depth = linear_eye_depth(shader.as_float(witness.y), camera)
        depth_band = 0.005 + depth_scale * shader.min(depth, sample_depth)
        depth_weight = shader.saturate(1.0 - shader.abs(depth - sample_depth) / depth_band)
        normal_weight = shader.saturate((shader.dot(normal, sample_normal) - normal_min) / (1.0 - normal_min))
        roughness_weight = shader.saturate(1.0 - shader.abs(roughness - sample_frame.z) / roughness_band)
        weight = tap_scale * depth_weight * normal_weight * roughness_weight
        if specular:
            sample_distance = shader.as_float(candidate.w)
            distance_band = 0.02 + 0.035 * shader.min(distance, sample_distance)
            weight *= shader.saturate(1.0 - shader.abs(distance - sample_distance) / distance_band)
        radiance += weight * estimate.rgb
        total += weight
    return shader.vec4(radiance / total, center.a)
