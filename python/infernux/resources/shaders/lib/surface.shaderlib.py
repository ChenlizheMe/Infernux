import infernux.shader as shader


def texture_uv(ctx: shader.SurfaceContext, uv_set: int, transform: shader.vec4) -> shader.vec2:
    uv = ctx.uv1 if uv_set == 1 else ctx.uv0
    return uv * transform.xy + transform.zw


def tangent_to_world(normal_ts: shader.vec3, ctx: shader.SurfaceContext) -> shader.vec3:
    normal = shader.normalize(ctx.normal_ws)
    tangent = shader.normalize(ctx.tangent_ws.xyz)
    bitangent = shader.cross(normal, tangent) * ctx.tangent_ws.w
    return shader.normalize(tangent * normal_ts.x + bitangent * normal_ts.y + normal * normal_ts.z)


def sample_normal(image: shader.Color2D, uv: shader.vec2, uv_set: int, scale: float,
                  ctx: shader.SurfaceContext) -> shader.vec3:
    encoded_xy = shader.sample(image, uv).rg * 2.0 - 1.0
    normal_ts = shader.normalize(shader.vec3(encoded_xy * scale, shader.sqrt(shader.max(1.0 - shader.dot(encoded_xy, encoded_xy), 0.0))))
    if uv_set == 0:
        return tangent_to_world(normal_ts, ctx)
    normal = shader.normalize(ctx.normal_ws)
    position_dx = shader.ddx(ctx.position_ws)
    position_dy = shader.ddy(ctx.position_ws)
    uv_dx = shader.ddx(uv)
    uv_dy = shader.ddy(uv)
    perpendicular_y = shader.cross(position_dy, normal)
    perpendicular_x = shader.cross(normal, position_dx)
    tangent = perpendicular_y * uv_dx.x + perpendicular_x * uv_dy.x
    bitangent = perpendicular_y * uv_dx.y + perpendicular_x * uv_dy.y
    basis_length_sq = shader.max(shader.dot(tangent, tangent), shader.dot(bitangent, bitangent))
    if basis_length_sq <= 1e-12:
        return tangent_to_world(normal_ts, ctx)
    inverse_length = shader.rsqrt(basis_length_sq)
    return shader.normalize((tangent * normal_ts.x + bitangent * normal_ts.y) * inverse_length + normal * normal_ts.z)
