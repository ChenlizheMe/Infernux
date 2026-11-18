import infernux.shader as shader
from infernux.shader.lib.camera import linear_eye_depth
from infernux.shader.lib.surface import decode_octahedral_normal


def unpack_ray_signal(packet: shader.uvec4) -> shader.vec4:
    # Three signed 19-bit floats (8 exponent bits, 10 fraction bits) and a 7-bit sample count.
    red = packet.x & shader.uint(0x7FFFF)
    green = (packet.x >> 19) | ((packet.y & shader.uint(63)) << 13)
    blue = (packet.y >> 6) & shader.uint(0x7FFFF)
    rgb = shader.as_float(shader.uvec3(red, green, blue) << 13)
    return shader.vec4(rgb, shader.float(packet.y >> 25))


def pack_ray_signal(signal: shader.vec4, witness: shader.uvec2) -> shader.uvec4:
    # Keep the FP32 exponent through temporal averaging. Round the fraction to nearest even.
    bits = shader.as_uint(signal.rgb)
    rgb = (bits + shader.uvec3(0xFFF) + ((bits >> 13) & shader.uint(1))) >> 13
    low = rgb.r | (rgb.g << 19)
    high = (rgb.g >> 13) | (rgb.b << 6) | (shader.uint(signal.a) << 25)
    return shader.uvec4(low, high, witness)


def specular_hit_distance(witness: shader.uint) -> float:
    return shader.as_float(witness & shader.uint(0xFFFF0000))


def specular_luminance_variance(witness: shader.uint) -> float:
    return shader.as_float(witness << 16)


def specular_witness(distance: float, variance: float) -> shader.uint:
    # Both nonnegative values retain the FP32 exponent. Distance precision is finer
    # than the reflected-footprint tolerance, including very near and distant hits.
    return (shader.as_uint(distance) & shader.uint(0xFFFF0000)) | (shader.as_uint(variance) >> 16)


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
        offset = shader.vec2(pixel - source)
        # Diffuse irradiance is incident light. A roughness texture on the receiver
        # must not discard irradiance samples from the same geometric surface.
        weight = depth_weight * normal_weight / (1.0 + shader.dot(offset, offset))
        if weight <= 0.0:
            continue

        # Each tap estimates the diffuse integral over its own hemisphere;
        # different directions can validly hit different surfaces or the sky.
        radiance += weight * estimate.rgb
        total += weight
    if total <= 0.0:
        return shader.vec4(0.0)
    return shader.vec4(radiance / total, 1.0)


def resolve_ray_specular(signal: shader.UInt2D, primary: shader.UInt2D, frame: shader.Color2D,
                        pixel: shader.ivec2, surface_id: shader.uint, device_depth: float,
                        normal: shader.vec3, roughness: float) -> shader.vec4:
    """Filter uncertain reflected light within the same first and second surfaces."""
    packet = shader.load(signal, pixel, 0)
    center = unpack_ray_signal(packet)
    if center.a <= 0.0:
        return center

    extent = shader.ivec2(shader.dimensions(signal, 0))
    camera = shader.camera()
    depth = linear_eye_depth(device_depth, camera)
    distance = specular_hit_distance(packet.w)
    variance = specular_luminance_variance(packet.w) / shader.max(center.a - 1.0, 1.0)
    footprint_weight = 0.25 * roughness * roughness
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
        if estimate.a <= 0.0 or candidate.z != packet.z:
            continue

        sample_frame = shader.load(frame, neighbor, 0)
        sample_normal = decode_octahedral_normal(sample_frame.xy)
        sample_depth = linear_eye_depth(shader.as_float(witness.y), camera)
        depth_band = 0.005 + 0.008 * shader.min(depth, sample_depth)
        depth_weight = shader.saturate(1.0 - shader.abs(depth - sample_depth) / depth_band)
        normal_weight = shader.saturate((shader.dot(normal, sample_normal) - 0.985) / 0.015)
        roughness_weight = shader.saturate(1.0 - shader.abs(roughness - sample_frame.z) / 0.08)
        sample_distance = specular_hit_distance(candidate.w)
        distance_band = 0.02 + 0.035 * shader.min(distance, sample_distance)
        distance_weight = shader.saturate(1.0 - shader.abs(distance - sample_distance) / distance_band)
        tap_weight = footprint_weight
        if center.a >= 4.0 and estimate.a >= 4.0 and variance > 0.0:
            sample_variance = specular_luminance_variance(candidate.w) / (estimate.a - 1.0)
            uncertainty = variance + sample_variance
            difference = center.rgb - estimate.rgb
            # Reject detailed texture on the same reflected surface, including isoluminant edges.
            difference2 = shader.dot(difference * difference, shader.vec3(0.2126, 0.7152, 0.0722))
            agreement = shader.exp2(-difference2 / shader.max(4.0 * uncertainty, 1.0e-12))
            tap_weight = shader.max(tap_weight, agreement * variance / uncertainty)
        weight = tap_weight * depth_weight * normal_weight * roughness_weight * distance_weight
        radiance += weight * estimate.rgb
        total += weight
    return shader.vec4(radiance / total, center.a)


def resolve_ray_diffuse(signal: shader.UInt2D, primary: shader.UInt2D, frame: shader.Color2D,
                        pixel: shader.ivec2, surface_id: shader.uint, depth: float,
                        normal: shader.vec3, roughness: float) -> shader.vec4:
    full = shader.ivec2(shader.dimensions(primary, 0))
    extent = shader.ivec2(shader.dimensions(signal, 0))
    if shader.any(full != extent):
        return resolve_reduced_ray(signal, primary, frame, pixel, full, extent, surface_id, depth, normal, roughness)
    return unpack_ray_signal(shader.load(signal, pixel, 0))
