import infernux.shader as shader


def decode_octahedral_normal(encoded: shader.vec2) -> shader.vec3:
    z = 1.0 - shader.abs(encoded.x) - shader.abs(encoded.y)
    adjustment = shader.saturate(-z)
    x = encoded.x + (-adjustment if encoded.x >= 0.0 else adjustment)
    y = encoded.y + (-adjustment if encoded.y >= 0.0 else adjustment)
    return shader.normalize(shader.vec3(x, y, z))


def decode_model_scalars(bits: shader.uint) -> shader.vec2:
    return shader.vec2(shader.f16_to_f32(bits & shader.uint(65535)), shader.f16_to_f32(bits >> 16))


def texture_uv(ctx: shader.SurfaceContext, uv_set: int, transform: shader.vec4) -> shader.vec2:
    uv = ctx.uv1 if uv_set == 1 else ctx.uv0
    return uv * transform.xy + transform.zw


def tangent_to_world(normal_ts: shader.vec3, ctx: shader.SurfaceContext) -> shader.vec3:
    normal = shader.normalize(ctx.normal_ws)
    tangent = shader.normalize(ctx.tangent_ws.xyz)
    bitangent = shader.cross(normal, tangent) * ctx.tangent_ws.w
    return shader.normalize(tangent * normal_ts.x + bitangent * normal_ts.y + normal * normal_ts.z)


def _decode_normal_rg(encoded: shader.vec2, scale: float) -> shader.vec3:
    encoded_xy = encoded * 2.0 - 1.0
    return shader.normalize(shader.vec3(encoded_xy * scale, shader.sqrt(shader.max(1.0 - shader.dot(encoded_xy, encoded_xy), 0.0))))


def _cotangent_to_world(normal_ts: shader.vec3, ctx: shader.SurfaceContext,
                        position_dx: shader.vec3, position_dy: shader.vec3,
                        uv_dx: shader.vec2, uv_dy: shader.vec2) -> shader.vec3:
    normal = shader.normalize(ctx.normal_ws)
    perpendicular_y = shader.cross(position_dy, normal)
    perpendicular_x = shader.cross(normal, position_dx)
    tangent = perpendicular_y * uv_dx.x + perpendicular_x * uv_dy.x
    bitangent = perpendicular_y * uv_dx.y + perpendicular_x * uv_dy.y
    basis_length_sq = shader.max(shader.dot(tangent, tangent), shader.dot(bitangent, bitangent))
    if basis_length_sq <= 1e-12:
        return tangent_to_world(normal_ts, ctx)
    inverse_length = shader.rsqrt(basis_length_sq)
    return shader.normalize((tangent * normal_ts.x + bitangent * normal_ts.y) * inverse_length + normal * normal_ts.z)


def sample_normal(image: shader.Color2D, uv: shader.vec2, uv_set: int, scale: float,
                  ctx: shader.SurfaceContext) -> shader.vec3:
    normal_ts = _decode_normal_rg(shader.sample(image, uv).rg, scale)
    if uv_set == 0:
        return tangent_to_world(normal_ts, ctx)
    return _cotangent_to_world(normal_ts, ctx, shader.ddx(ctx.position_ws), shader.ddy(ctx.position_ws),
                               shader.ddx(uv), shader.ddy(uv))
