import infernux.shader as shader


class GridVertex(shader.vert):
    shader_id = "infernux/editor/grid_vertex"
    hidden = True

    def deform(self, vertex: shader.VertexData, ctx: shader.VertexContext) -> shader.VertexData:
        return vertex

    def project(self, vertex: shader.mesh_input, ctx: shader.ProjectionContext) -> shader.vec4:
        return shader.vec4(vertex.position_ws.xy, 0.0, 1.0)
