import infernux.shader as shader


def unpack_ray_signal(packet: shader.uvec4) -> shader.vec4:
    return shader.vec4(shader.unpack_half_2x16(packet.x), shader.unpack_half_2x16(packet.y))


def pack_ray_signal(signal: shader.vec4, witness: shader.uvec2) -> shader.uvec4:
    return shader.uvec4(shader.pack_half_2x16(signal.rg), shader.pack_half_2x16(signal.ba), witness)
