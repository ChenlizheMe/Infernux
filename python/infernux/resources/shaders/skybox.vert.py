import infernux.shader as shader


class Skybox(shader.vert):
    shader_id = "infernux/skybox/vertex"

    def deform(self, vertex: shader.VertexData, ctx: shader.VertexContext) -> shader.VertexData:
        return vertex

    def project(self, vertex: shader.mesh_input, ctx: shader.ProjectionContext) -> shader.vec4:
        direction = vertex.position_ws
        projection = ctx.projection_matrix
        if shader.abs(projection[3][3]) > 0.5:
            half_width = 1.0 / shader.abs(projection[0][0])
            half_height = 1.0 / shader.abs(projection[1][1])
            direction *= shader.max(half_width, half_height) * 1.8
        rotated = shader.mat3(ctx.view_matrix) @ direction
        clip = projection @ shader.vec4(rotated, 1.0)
        return shader.vec4(clip.xy, 0.0, clip.w)
